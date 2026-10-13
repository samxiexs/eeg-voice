"""Cross-fitted acoustic TRFs and input-differentiable neural consistency."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .perceived_data import recording_dir
from .utils import atomic_checkpoint, atomic_json, load_checkpoint, read_json


def lag_design(phi, lags=33):
    phi = np.asarray(phi)
    output = np.zeros((len(phi), lags * phi.shape[1] + 1), dtype=np.float64)
    for lag in range(lags):
        output[lag:, lag * phi.shape[1]:(lag + 1) * phi.shape[1]] = phi[:len(phi) - lag]
    output[:, -1] = 1
    return output


def correlations(pred, truth, mask=None):
    mask = np.ones_like(truth, dtype=bool) if mask is None else mask
    n = mask.sum(0).clip(1)
    x = np.where(mask, pred, 0); y = np.where(mask, truth, 0)
    cross = (x * y).sum(0) - x.sum(0) * y.sum(0) / n
    vx = (x * x).sum(0) - x.sum(0)**2 / n
    vy = (y * y).sum(0) - y.sum(0)**2 / n
    return np.divide(cross, np.sqrt(np.maximum(vx * vy, 0)), out=np.zeros_like(cross), where=(vx > 1e-12) & (vy > 1e-12))


def ridge_fit(x, y, mask, penalty):
    weights = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    groups = {}
    for c in range(y.shape[1]):
        key = np.packbits(mask[:, c]).tobytes()
        groups.setdefault(key, []).append(c)
    for columns in groups.values():
        valid = mask[:, columns[0]]
        if valid.sum() < x.shape[1]: continue
        xx = x[valid]; yy = y[valid][:, columns]
        gram = xx.T @ xx / len(xx)
        regularizer = np.eye(x.shape[1]) * penalty
        regularizer[-1, -1] = 0  # Intercept is not penalized.
        weights[:, columns] = np.linalg.solve(gram + regularizer + 1e-8 * np.eye(len(gram)), xx.T @ yy / len(xx))
    return weights


def circular_null(chunks, permutations=500, min_shift=128, seed=11):
    """Exact masked cyclic correlations via FFT; return [permutations,channels].

    Masking both edges removes the cyclic join from all shifted comparisons.
    Six sufficient statistics handle shifted invalid channels/frames correctly.
    """
    rng = np.random.default_rng(seed)
    total = None
    for pred, truth, mask in chunks:
        mask = mask.copy()
        mask[:33] = False; mask[-33:] = False
        n = len(pred)
        if n <= 2 * min_shift: continue
        shifts = rng.integers(min_shift, n - min_shift + 1, permutations)
        m = mask.astype(np.float64)
        x, y = np.where(mask, pred, 0), np.where(mask, truth, 0)
        fm, fx, fy, fx2, fy2 = [np.fft.rfft(a, axis=0) for a in (m, x, y, x*x, y*y)]
        def cross(a, b): return np.fft.irfft(a.conj() * b, n=n, axis=0)[shifts]
        statistics = np.stack([cross(fm, fm), cross(fx, fm), cross(fm, fy), cross(fx2, fm), cross(fm, fy2), cross(fx, fy)])
        total = statistics if total is None else total + statistics
    if total is None: raise ValueError("Too little audio for circular-shift controls")
    n, sx, sy, sx2, sy2, sxy = total
    n = n.clip(1)
    covariance = sxy - sx * sy / n
    variance = (sx2 - sx*sx/n).clip(0) * (sy2 - sy*sy/n).clip(0)
    return np.divide(covariance, np.sqrt(variance), out=np.zeros_like(covariance), where=variance > 1e-12)


def bh_mask(pvalues, q=.05):
    order = np.argsort(pvalues)
    accepted = pvalues[order] <= q * (np.arange(len(order)) + 1) / len(order)
    result = np.zeros(len(order), dtype=bool)
    if accepted.any(): result[order[:np.flatnonzero(accepted)[-1] + 1]] = True
    return result


def trf_clip(cfg, manifest, fold_stats, subject, audio):
    matching = [m for m in manifest["recordings"] if m["subject"] == subject and m["audio"] == audio]
    if len(matching) != 1: raise ValueError(f"Missing/ambiguous recording {subject}/{audio}")
    meta = matching[0]
    with np.load(cfg.dir("artifacts") / "audio_full" / f"{audio}.npz") as data: phi = data["phi"].astype(np.float64)
    stats = read_json(cfg.dir("artifacts") / "stats/audio.json")["phi"]
    phi = (phi - stats["mean"]) / stats["std"]
    design = lag_design(phi, cfg["trf"]["lags"])
    low = np.load(recording_dir(cfg, subject, audio) / "neural.npy", mmap_mode="r")
    offset = round(meta["offset"] * 64)
    truth = np.array(low[:, offset:offset + len(phi)].T, dtype=np.float64)
    truth = (truth - fold_stats["neural"]["mean"]) / fold_stats["neural"]["std"]
    mask = np.zeros_like(truth, dtype=bool)
    for row in manifest["windows"]:
        if row["subject"] == subject and row["audio"] == audio:
            start = round(row["start"] * 64)
            mask[start:start + 256] |= np.array(row["good_channels"])[None]
    guard = round(cfg["preprocessing"]["guard_seconds"] * 64)
    mask[:guard] = False; mask[-guard:] = False
    return design[guard:-guard], truth[guard:-guard], mask[guard:-guard]


def fit_trfs(cfg, fold):
    from .metrics import bootstrap_mean
    manifest = read_json(cfg.dir("artifacts") / "manifest.json")
    stats = read_json(cfg.dir("artifacts") / "stats" / f"{fold}.json")
    folder = cfg.dir("artifacts") / "trf" / fold
    folder.mkdir(parents=True, exist_ok=True)
    from .pipeline import provenance
    if (folder / "gate.json").exists():
        previous = read_json(folder / "gate.json")
        if previous.get("provenance") != provenance(cfg, fold):
            raise ValueError("TRF cache has stale code/data/statistics provenance")
        if all((folder / f"{s}.pt").exists() for s in cfg.fold_subjects(fold, "train")):
            return previous
    summaries, validation_deltas, validation_rs = {}, [], []
    t = cfg["trf"]
    for subject in cfg.fold_subjects(fold, "train"):
        chunks = [trf_clip(cfg, manifest, stats, subject, audio) for audio in cfg.splits["audio"]["train"]]
        best = None
        for alpha in t["ridge"]:
            oof, losses = [], []
            for k in range(3):
                train = [c for i, c in enumerate(chunks) if i % 3 != k]
                weights = ridge_fit(np.concatenate([c[0] for c in train]), np.concatenate([c[1] for c in train]), np.concatenate([c[2] for c in train]), alpha)
                for i, (x, y, mask) in enumerate(chunks):
                    if i % 3 != k: continue
                    prediction = x @ weights
                    losses.append(float(((prediction - y)**2)[mask].mean()))
                    oof.append((i, prediction, y, mask))
            candidate = (float(np.mean(losses)), alpha, sorted(oof, key=lambda c: c[0]))
            if best is None or candidate[0] < best[0]: best = candidate
        _, alpha, oof = best
        predictions = np.concatenate([c[1] for c in oof]); truth = np.concatenate([c[2] for c in oof]); mask = np.concatenate([c[3] for c in oof])
        r = correlations(predictions, truth, mask)
        null = circular_null([(c[1], c[2], c[3]) for c in oof], t["permutations"], round(t["min_shift_seconds"] * 64))
        p = (1 + (null >= r[None]).sum(0)) / (t["permutations"] + 1)
        reliable = bh_mask(p, t["fdr"]) & (r > 0)
        selected = np.flatnonzero(reliable)
        selected = selected[np.argsort(-r[selected])[:t["max_channels"]]]
        keep = np.zeros(128, dtype=bool); keep[selected] = True
        residual = np.where(mask, (truth - predictions)**2, 0).sum(0) / mask.sum(0).clip(1)
        residual = residual.clip(t["variance_floor"])
        all_x = np.concatenate([c[0] for c in chunks]); all_y = np.concatenate([c[1] for c in chunks]); all_mask = np.concatenate([c[2] for c in chunks])
        weights = ridge_fit(all_x, all_y, all_mask, alpha)
        phi_stats = read_json(cfg.dir("artifacts") / "stats/audio.json")["phi"]
        checkpoint_value = {"weights": torch.tensor(weights, dtype=torch.float32), "selected": torch.from_numpy(keep), "variance": torch.tensor(residual, dtype=torch.float32), "phi_mean": torch.tensor(phi_stats["mean"], dtype=torch.float32), "phi_std": torch.tensor(phi_stats["std"], dtype=torch.float32), "config_hash": cfg.hash, "subject": subject, "training_audio": cfg.splits["audio"]["train"], "lags": t["lags"]}
        atomic_checkpoint(folder / f"{subject}.pt", checkpoint_value)
        validation = [trf_clip(cfg, manifest, stats, subject, a) for a in cfg.splits["audio"]["validation"]]
        val_chunks = [(x @ weights, y, m) for x, y, m in validation]
        val_r = correlations(np.concatenate([c[0] for c in val_chunks]), np.concatenate([c[1] for c in val_chunks]), np.concatenate([c[2] for c in val_chunks]))
        val_null = circular_null(val_chunks, t["permutations"], round(t["min_shift_seconds"] * 64))
        qualified = len(selected) >= t["min_channels"]
        if qualified:
            validation_rs.append(float(val_r[keep].mean()))
            validation_deltas.append(float((val_r[keep] - val_null[:, keep].mean(0)).mean()))
        summaries[subject] = {"ridge": alpha, "selected_channels": selected.tolist(), "crossfit_r": r.tolist(), "pvalues": p.tolist(), "validation_r": val_r.tolist(), "qualified": qualified}
        print(f"TRF {fold}/{subject}: {len(selected)} reliable channels", flush=True)
    fraction = sum(s["qualified"] for s in summaries.values()) / len(summaries)
    ci = bootstrap_mean(validation_deltas) if validation_deltas else {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    median_r = float(np.median(validation_rs)) if validation_rs else 0.0
    gate = {"config_hash": cfg.hash, "provenance": provenance(cfg, fold), "passed": fraction >= t["min_subject_fraction"] and median_r >= t["min_validation_r"] and ci["lower"] > 0, "qualified_fraction": fraction, "median_validation_r": median_r, "correct_vs_shift": ci, "subjects": summaries}
    atomic_json(folder / "gate.json", gate)
    return gate


class FrozenTRF(nn.Module):
    def __init__(self, checkpoint_value):
        super().__init__()
        for name in ["weights", "selected", "variance", "phi_mean", "phi_std"]:
            self.register_buffer(name, checkpoint_value[name])
        self.lags = checkpoint_value["lags"]

    def forward(self, phi):
        phi = (phi - self.phi_mean) / self.phi_std
        columns = []
        for lag in range(self.lags):
            shifted = torch.nn.functional.pad(phi.transpose(1, 2), (lag, 0)).transpose(1, 2)[:, :phi.shape[1]]
            columns.append(shifted)
        design = torch.cat(columns + [torch.ones_like(phi[..., :1])], dim=-1)
        return design @ self.weights


def load_trfs(cfg, fold, device):
    gate = read_json(cfg.dir("artifacts") / "trf" / fold / "gate.json")
    if gate["config_hash"] != cfg.hash or not gate["passed"]:
        raise ValueError("Neural Consistency is blocked by the held-out-audio TRF gate")
    return {s: FrozenTRF(load_checkpoint(cfg.dir("artifacts") / "trf" / fold / f"{s}.pt")).to(device) for s, summary in gate["subjects"].items() if summary["qualified"]}


def neural_consistency(wave, target_eeg, channel_mask, model: FrozenTRF):
    from .perceived_audio import acoustic_features
    prediction = model(acoustic_features(wave))
    if prediction.shape != target_eeg.shape: raise ValueError("Neural time grids disagree")
    valid = channel_mask & model.selected[None]
    mask = valid[:, None].expand_as(prediction).clone()
    mask[:, :39] = False; mask[:, 250:] = False
    if not mask.any(): return prediction.sum() * 0
    squared = (prediction - target_eeg).square() / model.variance[None, None]
    return squared[mask].mean()
