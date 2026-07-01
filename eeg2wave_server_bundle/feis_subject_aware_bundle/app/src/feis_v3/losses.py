from __future__ import annotations

from itertools import combinations
from typing import Any

import torch
import torch.nn.functional as F


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    mask_flat = mask.reshape(-1).to(loss.dtype)
    return (loss * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)


def _masked_acc(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, top_k: int = 1) -> torch.Tensor:
    pred = logits.topk(k=min(top_k, logits.shape[-1]), dim=-1).indices
    ok = (pred == target.unsqueeze(-1)).any(dim=-1).to(mask.dtype)
    return (ok * mask).sum() / mask.sum().clamp_min(1.0)


def _semantic_hist(tokens: torch.Tensor, mask: torch.Tensor, vocab: int) -> torch.Tensor:
    one_hot = F.one_hot(tokens.clamp(0, vocab - 1), num_classes=vocab).float()
    weighted = one_hot * mask.unsqueeze(-1).float()
    return weighted.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)


def _clip_nce(pred: torch.Tensor, target_hist: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target_hist, dim=-1)
    logits = pred @ target.t() / max(float(temperature), 1e-4)
    labels = torch.arange(pred.shape[0], device=pred.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc


def _ctc_loss(ctc_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1).long().clamp_min(1)
    if int(lengths.max().item()) > ctc_logits.shape[1]:
        return ctc_logits.new_tensor(0.0)
    log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)
    input_lengths = torch.full((ctc_logits.shape[0],), ctc_logits.shape[1], dtype=torch.long, device=ctc_logits.device)
    shifted_targets = target + 1
    flat = torch.cat([shifted_targets[i, : int(lengths[i].item())] for i in range(target.shape[0])], dim=0)
    if ctc_logits.device.type == "mps":
        # PyTorch does not currently implement aten::_ctc_loss on MPS.
        # Keep the rest of the model on MPS, but compute this alignment term
        # on CPU and move the scalar back so training can continue on Mac.
        return F.ctc_loss(
            log_probs.cpu(),
            flat.cpu(),
            input_lengths.cpu(),
            lengths.cpu(),
            blank=0,
            zero_infinity=True,
        ).to(ctc_logits.device)
    return F.ctc_loss(log_probs, flat, input_lengths, lengths, blank=0, zero_infinity=True)


def _soft_ot_proxy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    grid = torch.linspace(0.0, 1.0, vocab, device=logits.device)
    pred_curve = (logits.softmax(dim=-1) * grid).sum(dim=-1)
    target_curve = target.float() / max(vocab - 1, 1)
    return (((pred_curve - target_curve) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def _same_group_losses(out: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    subjects = list(batch.get("subject_id", []))
    labels = list(batch.get("label", []))
    if len(subjects) < 2:
        zero = out["content_embed"].new_tensor(0.0)
        return zero, zero, zero
    content = F.normalize(out["content_embed"], dim=-1)
    variant = F.normalize(out["variant_embed"], dim=-1)
    repeat_losses = []
    content_pull = []
    voice_push = []
    for i, j in combinations(range(len(subjects)), 2):
        same_label = labels[i] == labels[j]
        same_subject = subjects[i] == subjects[j]
        ccos = (content[i] * content[j]).sum()
        vcos = (variant[i] * variant[j]).sum()
        if same_subject and same_label:
            repeat_losses.append(1.0 - ccos)
        if same_label and not same_subject:
            content_pull.append(1.0 - ccos)
            voice_push.append(F.relu(vcos + 0.1))
    zero = out["content_embed"].new_tensor(0.0)
    repeat = torch.stack(repeat_losses).mean() if repeat_losses else zero
    pull = torch.stack(content_pull).mean() if content_pull else zero
    push = torch.stack(voice_push).mean() if voice_push else zero
    return repeat, pull, push


def _subject_confusion_loss(logits: torch.Tensor) -> torch.Tensor:
    logp = logits.log_softmax(dim=-1)
    uniform = torch.full_like(logp, 1.0 / max(logp.shape[-1], 1))
    return F.kl_div(logp, uniform, reduction="batchmean", log_target=False)


def compute_feis_v3_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, Any],
    aligner: str = "hybrid",
    train_phase: str = "joint",
    lambda_semantic_ce: float = 1.0,
    lambda_prompt_ce: float = 0.5,
    lambda_clip: float = 0.25,
    lambda_ctc: float = 0.2,
    lambda_ot: float = 0.1,
    lambda_perceiver: float = 0.2,
    lambda_repeat: float = 0.1,
    lambda_cross_subject: float = 0.05,
    lambda_voice_push: float = 0.05,
    lambda_codec_ce: float = 1.0,
    lambda_prosody: float = 0.2,
    lambda_variant: float = 0.2,
    lambda_subject_confusion: float = 0.02,
    lambda_moe: float = 0.02,
    contrast_temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    aligner = str(aligner).lower()
    train_phase = str(train_phase).lower()
    if train_phase not in {"alignment", "codec", "joint"}:
        raise ValueError(f"Unsupported FEIS v3 train_phase={train_phase!r}")
    semantic_target = batch["semantic_token_ids"].to(out["semantic_logits"].device)
    semantic_mask = batch["semantic_token_mask"].to(out["semantic_logits"].device).float()
    codec_target = batch["codec_token_ids"].to(out["codec_logits"].device)
    codec_mask = batch["codec_token_mask"].to(out["codec_logits"].device).float()
    label_idx = batch["label_idx"].to(out["prompt_logits"].device)
    variant_id = batch["audio_variant_cluster_id"].to(out["audio_variant_logits"].device)

    semantic_ce = _masked_ce(out["semantic_logits"], semantic_target, semantic_mask)
    prompt_ce = F.cross_entropy(out["prompt_logits"], label_idx)
    target_hist = _semantic_hist(semantic_target, semantic_mask, out["semantic_logits"].shape[-1])
    clip_loss, clip_acc = _clip_nce(out["content_clip"], target_hist, contrast_temperature)
    ctc = _ctc_loss(out["ctc_logits"], semantic_target, semantic_mask)
    ot = _soft_ot_proxy(out["semantic_logits"], semantic_target, semantic_mask)
    perceiver = _masked_ce(out["perceiver_logits"], semantic_target, semantic_mask)
    repeat, content_pull, voice_push = _same_group_losses(out, batch)
    codec_ce = _masked_ce(out["codec_logits"], codec_target, codec_mask)
    active = batch["prosody_active"].to(out["prosody_active_logits"].device).float()
    duration = batch["prosody_duration"].to(out["prosody_duration"].device).float()
    energy = batch["prosody_energy"].to(out["prosody_energy"].device).float()
    onset = batch["prosody_onset"].to(out["prosody_onset_logits"].device).float()
    prosody = (
        F.binary_cross_entropy_with_logits(out["prosody_active_logits"], active)
        + F.binary_cross_entropy_with_logits(out["prosody_onset_logits"], onset)
        + F.mse_loss(out["prosody_duration"], duration)
        + F.mse_loss(out["prosody_energy"], energy)
    )
    variant_ce = F.cross_entropy(out["audio_variant_logits"], variant_id.clamp(0, out["audio_variant_logits"].shape[-1] - 1))
    subject_confusion = _subject_confusion_loss(out["subject_adv_logits"])
    moe = out.get("moe_load_balance", codec_ce.new_tensor(0.0)) + out.get("moe_channel_sparsity", codec_ce.new_tensor(0.0)) * 0.1

    enabled = {
        "semantic_ce": aligner in {"mlp", "clip", "ctc", "ot", "perceiver", "hybrid"},
        "prompt_ce": aligner in {"mlp", "hybrid"},
        "clip": aligner in {"clip", "hybrid"},
        "ctc": aligner in {"ctc", "hybrid"},
        "ot": aligner in {"ot", "hybrid"},
        "perceiver": aligner in {"perceiver", "hybrid"},
        "repeat": aligner == "hybrid",
        "cross": aligner == "hybrid",
    }
    alignment_total = semantic_ce.new_tensor(0.0)
    if enabled["semantic_ce"]:
        alignment_total = alignment_total + semantic_ce * lambda_semantic_ce
    if enabled["prompt_ce"]:
        alignment_total = alignment_total + prompt_ce * lambda_prompt_ce
    if enabled["clip"]:
        alignment_total = alignment_total + clip_loss * lambda_clip
    if enabled["ctc"]:
        alignment_total = alignment_total + ctc * lambda_ctc
    if enabled["ot"]:
        alignment_total = alignment_total + ot * lambda_ot
    if enabled["perceiver"]:
        alignment_total = alignment_total + perceiver * lambda_perceiver
    if enabled["repeat"]:
        alignment_total = alignment_total + repeat * lambda_repeat
    if enabled["cross"]:
        alignment_total = alignment_total + content_pull * lambda_cross_subject + voice_push * lambda_voice_push

    codec_generation_total = codec_ce * lambda_codec_ce + prosody * lambda_prosody + variant_ce * lambda_variant
    regularizer_total = subject_confusion * lambda_subject_confusion + moe * lambda_moe
    if train_phase == "alignment":
        total = alignment_total + regularizer_total
    elif train_phase == "codec":
        total = codec_generation_total + regularizer_total
    else:
        total = alignment_total + codec_generation_total + regularizer_total

    return {
        "total": total,
        "alignment_total": alignment_total.detach(),
        "codec_generation_total": codec_generation_total.detach(),
        "regularizer_total": regularizer_total.detach(),
        "semantic_token_ce": semantic_ce.detach(),
        "semantic_token_ctc": ctc.detach(),
        "prompt_ce": prompt_ce.detach(),
        "eeg_audio_clip_nce": clip_loss.detach(),
        "clip_retrieval_acc": clip_acc.detach(),
        "monotonic_soft_ot": ot.detach(),
        "perceiver_token_ce": perceiver.detach(),
        "same_subject_label_repeat_consistency": repeat.detach(),
        "same_label_cross_subject_content_pull": content_pull.detach(),
        "same_label_different_subject_voice_push": voice_push.detach(),
        "codec_token_ce": codec_ce.detach(),
        "prosody_event_loss": prosody.detach(),
        "audio_variant_ce": variant_ce.detach(),
        "subject_adversarial_content": subject_confusion.detach(),
        "channel_moe_regularizers": moe.detach(),
        "semantic_top1": _masked_acc(out["semantic_logits"], semantic_target, semantic_mask, 1).detach(),
        "semantic_top3": _masked_acc(out["semantic_logits"], semantic_target, semantic_mask, 3).detach(),
        "codec_top1": _masked_acc(out["codec_logits"], codec_target, codec_mask, 1).detach(),
        "codec_top3": _masked_acc(out["codec_logits"], codec_target, codec_mask, 3).detach(),
        "prompt_acc": (out["prompt_logits"].argmax(dim=-1) == label_idx).float().mean().detach(),
    }
