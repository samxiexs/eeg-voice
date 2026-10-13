"""Independent 60-second supervised calibration; only a residual adapter changes."""
from __future__ import annotations
import torch
from torch import nn
from torch.utils.data import default_collate
import numpy as np
from ..utils import atomic_checkpoint, seed_all
from .protocol import provenance, subject_identity


class SubjectAdapter(nn.Module):
    def __init__(self, width=256, bottleneck=16):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def residual(self, hidden): return self.up(torch.nn.functional.gelu(self.down(hidden)))
    def forward(self, hidden): return hidden + self.residual(hidden)


def validate_calibration(dataset, seconds=60):
    if not len(dataset): raise ValueError('No independently processed calibration windows')
    subjects = {subject_identity(row) for row in dataset.rows}
    if len(subjects) != 1: raise ValueError('Each adapter is calibrated on one unseen person')
    intervals = set()
    for row in dataset.rows:
        receipt = row.get('calibration', {})
        if not isinstance(receipt, dict): receipt = {}
        if not receipt.get('independently_filtered', row.get('independently_filtered', False)):
            raise ValueError('Calibration must use independently filtered shards, including all context in its budget')
        start = float(receipt.get('start', row.get('calibration_start', 0)))
        stop = float(receipt.get('stop', row.get('calibration_stop', seconds)))
        if not 0 < stop - start <= seconds + 1e-6: raise ValueError('Calibration shard exceeds 60-second context budget')
        intervals.add((str(row.get('recording', row.get('audio'))), start, stop))
    # Multiple shards cannot silently turn a nominal 60-second calibration into minutes.
    if sum(b-a for _, a, b in intervals) > seconds + 1e-6:
        raise ValueError('Total independent calibration support exceeds 60 seconds')
    return next(iter(subjects))


def calibrate_adapter(cfg, encoder, dataset, protocol='P1', fold='F0', seed=11,
                      device='cpu', output_path=None):
    from .training import alignment_loss, move_batch
    person = validate_calibration(dataset)
    seed_all(seed)
    encoder.to(device).eval().requires_grad_(False)
    adapter = SubjectAdapter().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    # One microbatch per variable montage window; fixed effective batch 16.
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        for index in rng.integers(len(dataset), size=16):
            batch = move_batch(default_collate([dataset[int(index)]]), device)
            with torch.no_grad(): base = encoder.hidden(batch)
            if isinstance(base, dict): base = base['hidden']
            adapted = adapter(base.detach())
            out = encoder.heads(adapted)
            loss, _ = alignment_loss(out, batch, cfg, contrast=False, uncertainty=False)
            (loss / 16 + .001 * adapter.residual(base.detach()).square().mean() / 16).backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
    adapter.eval().requires_grad_(False)
    if output_path is not None:
        atomic_checkpoint(output_path, {'adapter': adapter.state_dict(), 'subject': person,
                          'provenance': provenance(cfg, protocol, fold), 'calibration_seconds': 60,
                          'updates': 100, 'batch': 16, 'checkpoint_selection': 'last-fixed-update',
                          'reliability': 'source-calibrator-carried-forward-unverified'})
    return adapter
