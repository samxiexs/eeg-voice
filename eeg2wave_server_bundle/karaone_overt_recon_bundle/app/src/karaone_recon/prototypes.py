from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class KaraOneSemanticMelPrototypes:
    """Train-split-only semantic acoustic prototypes for v4.

    The prototype cache is an EEG-only inference prior: predicted semantic-token
    distributions map to a Mel template. Label prototypes are loaded only for
    oracle diagnostics and must not be used by the default generation path.
    """

    def __init__(self, cache_path: str | Path):
        payload = np.load(Path(cache_path), allow_pickle=True)
        self.path = Path(cache_path)
        self.global_mean_mel = payload["global_mean_mel"].astype(np.float32)
        self.semantic_token_mean_mel = payload["semantic_token_mean_mel"].astype(np.float32)
        self.token_lag_sec = payload["token_lag_sec"].astype(np.float32)
        self.label_mean_mel = payload["label_mean_mel"].astype(np.float32)
        self.label_lag_sec = payload["label_lag_sec"].astype(np.float32)
        self.label_vocab = payload["label_vocab"].astype(str).tolist()
        self.global_lag_sec = float(payload["global_lag_sec"]) if "global_lag_sec" in payload.files else 0.0
        self.vocab_size = int(self.semantic_token_mean_mel.shape[0])
        self.target_steps = int(self.semantic_token_mean_mel.shape[1])
        self.target_dim = int(self.semantic_token_mean_mel.shape[2])

    def to_tensors(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "TorchSemanticMelPrototypes":
        return TorchSemanticMelPrototypes(
            token_proto=torch.as_tensor(self.semantic_token_mean_mel, device=device, dtype=dtype),
            token_lag=torch.as_tensor(self.token_lag_sec, device=device, dtype=dtype),
            label_proto=torch.as_tensor(self.label_mean_mel, device=device, dtype=dtype),
            label_lag=torch.as_tensor(self.label_lag_sec, device=device, dtype=dtype),
            global_proto=torch.as_tensor(self.global_mean_mel, device=device, dtype=dtype),
            global_lag=torch.as_tensor(float(self.global_lag_sec), device=device, dtype=dtype),
        )


class TorchSemanticMelPrototypes:
    def __init__(
        self,
        token_proto: torch.Tensor,
        token_lag: torch.Tensor,
        label_proto: torch.Tensor,
        label_lag: torch.Tensor,
        global_proto: torch.Tensor,
        global_lag: torch.Tensor,
    ):
        self.token_proto = token_proto
        self.token_lag = token_lag
        self.label_proto = label_proto
        self.label_lag = label_lag
        self.global_proto = global_proto
        self.global_lag = global_lag

    @property
    def vocab_size(self) -> int:
        return int(self.token_proto.shape[0])

    def _token_weights_from_logits(self, logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        if mask is not None:
            weight = mask.to(device=logits.device, dtype=probs.dtype).unsqueeze(-1)
            probs = probs * weight
            denom = weight.sum(dim=1).clamp_min(1.0)
        else:
            denom = torch.full((logits.shape[0], 1), float(max(logits.shape[1], 1)), device=logits.device, dtype=probs.dtype)
        return probs.sum(dim=1) / denom

    def prototype_from_logits(self, logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        weights = self._token_weights_from_logits(logits, mask)
        return torch.einsum("bk,ktd->btd", weights, self.token_proto)

    def lag_from_logits(self, logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        weights = self._token_weights_from_logits(logits, mask)
        return weights @ self.token_lag

    def prototype_from_token_targets(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = tokens.to(device=self.token_proto.device, dtype=torch.long).clamp(min=0, max=max(self.vocab_size - 1, 0))
        one_hot = F.one_hot(tokens, num_classes=self.vocab_size).to(self.token_proto.dtype)
        if mask is not None:
            weight = mask.to(device=self.token_proto.device, dtype=self.token_proto.dtype).unsqueeze(-1)
            one_hot = one_hot * weight
            denom = weight.sum(dim=1).clamp_min(1.0)
        else:
            denom = torch.full((tokens.shape[0], 1), float(max(tokens.shape[1], 1)), device=tokens.device, dtype=self.token_proto.dtype)
        weights = one_hot.sum(dim=1) / denom
        return torch.einsum("bk,ktd->btd", weights, self.token_proto)

    def label_prototype(self, label_idx: torch.Tensor) -> torch.Tensor:
        idx = label_idx.to(device=self.label_proto.device, dtype=torch.long).clamp(min=0, max=max(self.label_proto.shape[0] - 1, 0))
        return self.label_proto[idx]

    def label_lag(self, label_idx: torch.Tensor) -> torch.Tensor:
        idx = label_idx.to(device=self.label_lag.device, dtype=torch.long).clamp(min=0, max=max(self.label_lag.shape[0] - 1, 0))
        return self.label_lag[idx]


def semantic_prototype_payload(
    target_seq: np.ndarray,
    token_seq: np.ndarray,
    token_mask: np.ndarray,
    label_idx: np.ndarray,
    label_vocab: list[str],
    vocab_size: int,
    lag_sec: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    target_seq = np.asarray(target_seq, dtype=np.float32)
    token_seq = np.asarray(token_seq, dtype=np.int64)
    token_mask = np.asarray(token_mask, dtype=np.float32)
    label_idx = np.asarray(label_idx, dtype=np.int64)
    n_labels = len(label_vocab)
    global_mean = target_seq.mean(axis=0).astype(np.float32)

    label_proto = np.zeros((n_labels, target_seq.shape[1], target_seq.shape[2]), dtype=np.float32)
    label_count = np.zeros((n_labels,), dtype=np.float32)
    token_proto = np.zeros((int(vocab_size), target_seq.shape[1], target_seq.shape[2]), dtype=np.float32)
    token_count = np.zeros((int(vocab_size),), dtype=np.float32)
    token_lag_sum = np.zeros((int(vocab_size),), dtype=np.float32)
    label_lag_sum = np.zeros((n_labels,), dtype=np.float32)
    label_lag_count = np.zeros((n_labels,), dtype=np.float32)
    lag = np.zeros((target_seq.shape[0],), dtype=np.float32) if lag_sec is None else np.asarray(lag_sec, dtype=np.float32)

    for i in range(target_seq.shape[0]):
        li = int(label_idx[i])
        if 0 <= li < n_labels:
            label_proto[li] += target_seq[i]
            label_count[li] += 1.0
            label_lag_sum[li] += lag[i]
            label_lag_count[li] += 1.0
        active = token_mask[i] > 0
        if not bool(active.any()):
            continue
        counts = np.bincount(token_seq[i][active].clip(0, int(vocab_size) - 1), minlength=int(vocab_size)).astype(np.float32)
        nz = np.nonzero(counts > 0)[0]
        for k in nz.tolist():
            weight = float(counts[k])
            token_proto[k] += target_seq[i] * weight
            token_count[k] += weight
            token_lag_sum[k] += lag[i] * weight

    for li in range(n_labels):
        if label_count[li] > 0:
            label_proto[li] /= label_count[li]
        else:
            label_proto[li] = global_mean
    for k in range(int(vocab_size)):
        if token_count[k] > 0:
            token_proto[k] /= token_count[k]
        else:
            token_proto[k] = global_mean
    global_lag = float(np.mean(lag)) if lag.size else 0.0
    token_lag = np.where(token_count > 0, token_lag_sum / np.maximum(token_count, 1.0), global_lag).astype(np.float32)
    label_lag = np.where(label_lag_count > 0, label_lag_sum / np.maximum(label_lag_count, 1.0), global_lag).astype(np.float32)
    return {
        "global_mean_mel": global_mean.astype(np.float32),
        "semantic_token_mean_mel": token_proto.astype(np.float32),
        "token_counts": token_count.astype(np.float32),
        "token_lag_sec": token_lag.astype(np.float32),
        "label_mean_mel": label_proto.astype(np.float32),
        "label_counts": label_count.astype(np.float32),
        "label_lag_sec": label_lag.astype(np.float32),
        "label_vocab": np.asarray(label_vocab, dtype=str),
        "global_lag_sec": np.asarray(global_lag, dtype=np.float32),
    }
