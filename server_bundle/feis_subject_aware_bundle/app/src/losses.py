from __future__ import annotations

import torch
import torch.nn.functional as F


def _waveform_2d(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 3:
        waveform = waveform.squeeze(1)
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform [B, T] or [B, 1, T], got {tuple(waveform.shape)}")
    return waveform


def _stft_magnitude(waveform: torch.Tensor, fft_size: int, hop_size: int, win_size: int) -> torch.Tensor:
    waveform = _waveform_2d(waveform)
    window = torch.hann_window(win_size, device=waveform.device, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=True,
        return_complex=True,
    )
    return spec.abs()


def multi_resolution_stft_loss(
    pred_wav: torch.Tensor,
    target_wav: torch.Tensor,
    fft_sizes: list[int] | tuple[int, ...] = (512, 1024, 2048),
    hop_sizes: list[int] | tuple[int, ...] = (128, 256, 512),
    win_sizes: list[int] | tuple[int, ...] = (512, 1024, 2048),
) -> torch.Tensor:
    total = pred_wav.new_tensor(0.0)
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        pred_mag = _stft_magnitude(pred_wav, fft_size, hop_size, win_size)
        target_mag = _stft_magnitude(target_wav, fft_size, hop_size, win_size)
        mag_loss = F.l1_loss(pred_mag, target_mag)
        sc_loss = (
            torch.linalg.norm(target_mag - pred_mag, dim=(-2, -1))
            / (torch.linalg.norm(target_mag, dim=(-2, -1)) + 1e-8)
        ).mean()
        total = total + mag_loss + sc_loss
    return total / len(tuple(fft_sizes))


def multi_resolution_log_stft_loss(
    pred_wav: torch.Tensor,
    target_wav: torch.Tensor,
    fft_sizes: list[int] | tuple[int, ...] = (512, 1024, 2048),
    hop_sizes: list[int] | tuple[int, ...] = (128, 256, 512),
    win_sizes: list[int] | tuple[int, ...] = (512, 1024, 2048),
) -> torch.Tensor:
    total = pred_wav.new_tensor(0.0)
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        pred_mag = _stft_magnitude(pred_wav, fft_size, hop_size, win_size)
        target_mag = _stft_magnitude(target_wav, fft_size, hop_size, win_size)
        total = total + F.l1_loss(torch.log(pred_mag + 1e-5), torch.log(target_mag + 1e-5))
    return total / len(tuple(fft_sizes))


def rms_loss(pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
    pred_wav = _waveform_2d(pred_wav)
    target_wav = _waveform_2d(target_wav)
    pred_rms = torch.sqrt(torch.mean(pred_wav.pow(2), dim=-1) + 1e-8)
    target_rms = torch.sqrt(torch.mean(target_wav.pow(2), dim=-1) + 1e-8)
    return F.l1_loss(torch.log(pred_rms + 1e-5), torch.log(target_rms + 1e-5))


def envelope_loss(pred_wav: torch.Tensor, target_wav: torch.Tensor, frame_size: int = 400) -> torch.Tensor:
    pred_wav = _waveform_2d(pred_wav).unsqueeze(1)
    target_wav = _waveform_2d(target_wav).unsqueeze(1)
    pred_env = F.avg_pool1d(pred_wav.abs(), kernel_size=frame_size, stride=frame_size // 2, padding=frame_size // 4)
    target_env = F.avg_pool1d(target_wav.abs(), kernel_size=frame_size, stride=frame_size // 2, padding=frame_size // 4)
    return F.l1_loss(pred_env, target_env)


def compute_total_loss(
    pred_wav: torch.Tensor,
    target_wav: torch.Tensor,
    vq_loss: torch.Tensor,
    lambda_stft: float = 0.5,
    lambda_log_stft: float = 0.25,
    lambda_rms: float = 0.5,
    lambda_envelope: float = 1.0,
    lambda_vq: float = 0.1,
    cls_logits: torch.Tensor | None = None,
    label_ids: torch.Tensor | None = None,
    lambda_cls: float = 1.0,
) -> dict[str, torch.Tensor]:
    target_wav = _waveform_2d(target_wav)
    pred_wav = _waveform_2d(pred_wav)
    l1 = F.l1_loss(pred_wav, target_wav)
    stft = multi_resolution_stft_loss(pred_wav, target_wav)
    log_stft = multi_resolution_log_stft_loss(pred_wav, target_wav)
    rms = rms_loss(pred_wav, target_wav)
    envelope = envelope_loss(pred_wav, target_wav)
    total = (
        l1
        + float(lambda_stft) * stft
        + float(lambda_log_stft) * log_stft
        + float(lambda_rms) * rms
        + float(lambda_envelope) * envelope
        + float(lambda_vq) * vq_loss
    )
    cls_loss = pred_wav.new_tensor(0.0)
    cls_acc = pred_wav.new_tensor(0.0)
    if cls_logits is not None and label_ids is not None:
        cls_loss = F.cross_entropy(cls_logits, label_ids.long())
        cls_pred = torch.argmax(cls_logits, dim=-1)
        cls_acc = (cls_pred == label_ids.long()).float().mean()
        total = total + float(lambda_cls) * cls_loss
    return {
        "total": total,
        "l1": l1,
        "stft": stft,
        "log_stft": log_stft,
        "rms": rms,
        "envelope": envelope,
        "vq": vq_loss,
        "cls": cls_loss,
        "cls_acc": cls_acc,
    }


def stft_distance_single(
    pred_wav: torch.Tensor,
    target_wav: torch.Tensor,
    fft_sizes: list[int] | tuple[int, ...] = (512, 1024, 2048),
    hop_sizes: list[int] | tuple[int, ...] = (128, 256, 512),
    win_sizes: list[int] | tuple[int, ...] = (512, 1024, 2048),
) -> float:
    pred = pred_wav.reshape(1, -1)
    target = target_wav.reshape(1, -1)
    return float(multi_resolution_stft_loss(pred, target, fft_sizes, hop_sizes, win_sizes).item())
