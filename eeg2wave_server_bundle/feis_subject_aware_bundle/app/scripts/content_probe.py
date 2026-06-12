"""Stage-1 content decodability probe (V2_PLAN): is there ANY content signal in
the FEIS EEG, independent of the generative model?

A simple, transparent decoder (closed-form ridge classifier) on within-subject
EEG -> 16-way content, with k-fold CV and a LABEL-PERMUTATION test (real p-value
and chance band). Done per stage (stimuli / thinking) and for coarse phonological
categories vs their correct majority baseline.

Rationale: a generative model cannot extract more content than a clean decoder
can. If this probe is at chance, FEIS content is not decodable -> stop stacking
generative machinery (V2_PLAN Stage-1 gate).

  python scripts/content_probe.py --config configs/factored.yaml \
      --stages stimuli,thinking --folds 5 --permutations 200
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import load_simple_yaml, resolve_bundle_path, resolve_feis_root, write_json
from src.feis_factored.targets import MANNER, VOICING, VOWEL_CONSONANT, FactoredTargets


def parse_args():
    p = argparse.ArgumentParser(description="Within-subject EEG content decodability + permutation test.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "factored.yaml"))
    p.add_argument("--target", default="content", choices=["content", "subject", "stage"],
                   help="content=within-subject 16-way (the science); "
                        "subject/stage=POSITIVE CONTROL (same features+probe should decode these EASILY)")
    p.add_argument("--stages", default="stimuli,thinking")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--bands", type=int, default=5)
    p.add_argument("--include-anomalous", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def _norm_subject(s: str) -> str:
    s = str(s)
    return s.zfill(2) if s.isdigit() else s


def featurize(x: np.ndarray, n_bands: int) -> np.ndarray:
    """EEG [C, L] -> fs-agnostic features: log-var, log mean|diff|, log band power."""
    eps = 1e-8
    var = np.log(np.var(x, axis=1) + eps)                                  # [C]
    diff = np.log(np.mean(np.abs(np.diff(x, axis=1)), axis=1) + eps)       # [C]
    mag = np.abs(np.fft.rfft(x, axis=1))                                   # [C, F]
    F = mag.shape[1]
    edges = np.linspace(1, F, n_bands + 1).astype(int)                    # skip DC bin 0
    bands = []
    for b in range(n_bands):
        lo, hi = edges[b], max(edges[b + 1], edges[b] + 1)
        bands.append(np.log(np.sum(mag[:, lo:hi] ** 2, axis=1) + eps))     # [C]
    return np.concatenate([var, diff] + bands, axis=0).astype(np.float64)  # [C*(2+n_bands)]


def load_subject_trials(feis_root: Path, targets: FactoredTargets, subjects, stages, n_bands):
    """-> dict[(subject,stage)] = (X [N,F], y_label_id [N])."""
    # label per (subject, trial_index, stage)
    lab_map: dict[tuple, str] = {}
    with (feis_root / "segments.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            stage = r.get("segment_stage")
            if stage not in stages:
                continue
            sub = _norm_subject(r["subject_id"])
            lab_map[(sub, int(r["trial_index"]), stage)] = str(r["label"])

    out: dict[tuple, tuple] = {}
    for sub in subjects:
        npz = feis_root / "subjects" / f"{sub}.npz"
        if not npz.exists():
            continue
        b = np.load(npz, allow_pickle=True)
        tix = b["trial_indices"].astype(int)
        for stage in stages:
            key = f"stage__{stage}"
            if key not in b.files:
                continue
            arr = b[key].astype(np.float32)                              # [n, C, L]
            X, y = [], []
            for pos, ti in enumerate(tix.tolist()):
                lab = lab_map.get((sub, ti, stage))
                if lab is None or not targets.has_cell(sub, lab):
                    continue
                X.append(featurize(arr[pos], n_bands))
                y.append(targets.label_to_id[lab])
            if X:
                out[(sub, stage)] = (np.stack(X, 0), np.asarray(y, dtype=int))
    return out


def ridge_cv_predict(X, y, n_labels, folds, lam, rng):
    """Closed-form ridge one-vs-all, k-fold CV. Returns predicted label per sample."""
    n = X.shape[0]
    order = rng.permutation(n)
    pred = np.full(n, -1, dtype=int)
    fold_id = np.zeros(n, dtype=int)
    fold_id[order] = np.arange(n) % folds
    Yoh = np.eye(n_labels)[y]
    for f in range(folds):
        te = fold_id == f
        tr = ~te
        if tr.sum() < n_labels or te.sum() == 0:
            pred[te] = rng.randint(0, n_labels, size=int(te.sum()))
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        Xtr = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], 1)
        Xte = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], 1)
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        W = np.linalg.solve(A, Xtr.T @ Yoh[tr])
        pred[te] = np.argmax(Xte @ W, axis=1)
    return pred


def assemble_groups(data, target, n_labels, subject_to_id, stage_to_id):
    """Return (groups[list of (X,y)], n_classes, mode).

    content : within-subject 16-way  -> one group per subject (y=content id)   [the science]
    subject : cross-subject id        -> ONE pooled group       (y=subject id)  [positive control]
    stage   : stimuli vs thinking     -> ONE pooled group       (y=stage id)    [positive control]
    """
    if target == "content":
        return [(X, y) for (sub, stg), (X, y) in data.items()], n_labels, "within"
    if target == "subject":
        Xs = np.concatenate([X for (X, _) in data.values()], 0)
        ys = np.concatenate([np.full(len(y), subject_to_id[sub])
                             for (sub, stg), (X, y) in data.items()], 0)
        return [(Xs, ys)], len(subject_to_id), "global"
    # stage
    Xs = np.concatenate([X for (X, _) in data.values()], 0)
    ys = np.concatenate([np.full(len(y), stage_to_id[stg])
                         for (sub, stg), (X, y) in data.items()], 0)
    return [(Xs, ys)], len(stage_to_id), "global"


def decode_accuracy(groups, n_classes, folds, lam, rng, perm: bool = False):
    """Pooled CV accuracy over groups (works for within-subject and global modes)."""
    correct = total = 0
    for (X, y) in groups:
        yy = y.copy()
        if perm:
            rng.shuffle(yy)
        pred = ridge_cv_predict(X, yy, n_classes, folds, lam, rng)
        correct += int(np.sum(pred == yy)); total += len(yy)
    return correct / max(total, 1)


def coarse_from_pred(data, targets, n_labels, folds, lam, rng):
    """Observed coarse accuracy (true labels) + majority baselines."""
    inv = {v: k for k, v in targets.label_to_id.items()}
    maps = {"manner": MANNER, "voicing": VOICING, "vc": VOWEL_CONSONANT}
    hit = {k: 0 for k in maps}; total = 0
    cat_counts = {k: defaultdict(int) for k in maps}
    for (sub, stage), (X, y) in data.items():
        pred = ridge_cv_predict(X, y, n_labels, folds, lam, rng)
        for pi, yi in zip(pred, y):
            pl, tl = inv[int(pi)], inv[int(yi)]
            for k, mp in maps.items():
                hit[k] += int(mp[pl] == mp[tl])
                cat_counts[k][mp[tl]] += 1
            total += 1
    obs = {k: hit[k] / max(total, 1) for k in maps}
    majority = {k: max(cat_counts[k].values()) / max(total, 1) for k in maps}
    return obs, majority


def main():
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    stages = tuple(args.stages.split(","))
    rng = np.random.RandomState(args.seed)

    targets = FactoredTargets(resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR))
    feis_root = resolve_feis_root(resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR))
    subjects = [s for s in targets.subject_vocab if (args.include_anomalous or s != "05")]
    subject_to_id = {s: i for i, s in enumerate(subjects)}
    stage_to_id = {s: i for i, s in enumerate(stages)}
    n_labels = len(targets.label_vocab)

    all_data = load_subject_trials(feis_root, targets, subjects, stages, args.bands)
    report = {"target": args.target, "n_labels": n_labels, "folds": args.folds,
              "permutations": args.permutations, "blocks": {}}

    def run_block(name, data):
        groups, n_cls, _ = assemble_groups(data, args.target, n_labels, subject_to_id, stage_to_id)
        chance = 1.0 / n_cls
        n_trials = sum(len(y) for _, y in groups)
        obs = decode_accuracy(groups, n_cls, args.folds, args.ridge, np.random.RandomState(args.seed))
        null = np.asarray([decode_accuracy(groups, n_cls, args.folds, args.ridge,
                                           np.random.RandomState(args.seed + 1 + p), perm=True)
                           for p in range(args.permutations)])
        p_value = float((np.sum(null >= obs) + 1) / (len(null) + 1))
        sig = bool(p_value < 0.05 and obs > chance)
        block = {"n_classes": n_cls, "n_trials": n_trials, "top1": obs, "chance": chance,
                 "fold_factor": obs / chance, "null_mean": float(null.mean()),
                 "null_p95": float(np.percentile(null, 95)), "p_value": p_value, "significant": sig}
        if args.target == "content":
            cobs, cmaj = coarse_from_pred(data, targets, n_labels, args.folds, args.ridge,
                                          np.random.RandomState(args.seed))
            block["coarse_obs"] = cobs; block["coarse_majority"] = cmaj
            block["coarse_gain_over_majority"] = {k: cobs[k] - cmaj[k] for k in cobs}
        report["blocks"][name] = block
        print(f"[{args.target}:{name}] n={n_trials} classes={n_cls} top1={obs:.4f} "
              f"chance={chance:.4f} ({obs/chance:.1f}x) null95={block['null_p95']:.4f} "
              f"p={p_value:.3f} sig={sig}")
        if args.target == "content":
            print("   coarse obs/majority: " + ", ".join(
                f"{k}={block['coarse_obs'][k]:.3f}/{block['coarse_majority'][k]:.3f}"
                f"(Δ{block['coarse_gain_over_majority'][k]:+.3f})" for k in block["coarse_obs"]))

    if args.target == "stage":
        # stage decoding is inherently cross-stage -> one block over all data
        run_block("all", all_data)
    else:
        for stage in stages:
            data = {k: v for k, v in all_data.items() if k[1] == stage}
            if data:
                run_block(stage, data)

    any_sig = any(b["significant"] for b in report["blocks"].values())
    if args.target == "content":
        report["verdict"] = ("DECODABLE — at least one stage significantly above chance; "
                             "Stage-2 (generative) is justified."
                             if any_sig else
                             "NOT DECODABLE — at chance (permutation p>=0.05). Per V2_PLAN Stage-1 "
                             "gate: stop FEIS-only; pivot to auditory dataset / pretraining.")
    else:
        report["verdict"] = (
            f"POSITIVE CONTROL PASSED — same features+probe decode '{args.target}' well above chance "
            "(p<0.05). The pipeline detects real signal, so the content NULL is a true negative, not a bug."
            if any_sig else
            f"WARNING — positive control FAILED on '{args.target}'. The probe pipeline may be broken; "
            "fix it before trusting the content null.")
    out_dir = Path(args.out_dir) if args.out_dir else (
        resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR) / "content_probe")
    write_json(out_dir / f"probe_{args.target}_{'_'.join(stages)}.json", report)
    print(f"[verdict] {report['verdict']}")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
