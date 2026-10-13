#!/usr/bin/env python3
"""Reproduce the source-unit audit; no model outcomes or test-stimulus labels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from eegspeech.config import Config
from eegspeech.utils import atomic_json,digest


def main():
    cfg=Config()
    source=cfg.dir("raw")/"source_calibration/dataSub7.mat"
    original=loadmat(source,simplify_cells=True)["eeg"]
    reference=np.asarray(original["data"][0][:5000],dtype=np.float64)
    # Subject numbering differs between CND and OpenNeuro; identity was matched
    # by the channel DC profile on F0 training-side recordings, never WER.
    path=cfg.dir("raw")/"sub-008/eeg/sub-008_task-listening_run-01_eeg.eeg"
    binary=np.fromfile(path,dtype="<f4").reshape(-1,128).astype(np.float64)
    nominal=binary*1e-7  # Pinned .vhdr: resolution .1 µV / binary_float32.
    restored=resample_poly(nominal,1,4,axis=0)
    n=min(len(restored),len(reference));a=restored[128:n-128];b=reference[128:n-128]
    a=a-a.mean(0);b=b-b.mean(0)
    corr=(a*b).sum(0)/np.sqrt((a*a).sum(0)*(b*b).sum(0))
    dc_error=float(np.linalg.norm(nominal[:5120].mean(0)-reference[:1280].mean(0))/np.linalg.norm(reference[:1280].mean(0)))
    scale=float(np.sum(a*b)/np.sum(a*a))
    matches=float(np.median(corr))>.99 and dc_error<1e-4 and abs(scale-1)<.01
    if not matches:raise ValueError("Author source does not confirm the chosen numeric scale")
    report={"dataset_revision":cfg["dataset"]["revision"],"source_matches":matches,"header_unit_multiplier":1e-6,"source_url":"https://www.data.cnspworkshop.net/data/datasetCND_LalorNatSpeech.zip","archive_member":"LalorNatSpeech/dataCND/dataSub7.mat","source_sha256":digest(source),"source_fs":int(original["fs"]),"openneuro_subject":"sub-008","openneuro_run":1,"source_subject":7,"eeg_sha256":digest(path),"channel_dc_relative_error":dc_error,"dynamic_correlation_min":float(corr.min()),"dynamic_correlation_median":float(np.median(corr)),"source_over_nominal_mne_numeric_ac_scale":scale,"interpretation":"BrainVision's nominal volt values numerically match the original CND values; multiplying them by 1e6 again creates an export-unit discrepancy. Restore native CND numbers and interpret them as microvolts using BioSemi gain/range and source amplitude evidence.","absolute_unit_caveat":"CND struct has no explicit unit field. Numeric restoration is directly verified; the microvolt interpretation is inferred from acquisition scale, not an author-issued unit correction.","manufacturer_reference":"https://www.biosemi.com/faq/adjust_gain.htm","pybv_reference":"https://pybv.readthedocs.io/en/stable/generated/pybv.write_brainvision.html","calibration_scope":"fixed corpus-export correction; no per-record scaling fit; no decoder outcomes used"}
    atomic_json(cfg.root/cfg["dataset"]["unit_calibration_report"],report)
    print(f"Source scale verified: median dynamic r={np.median(corr):.8f}, DC error={dc_error:.3g}")


if __name__=="__main__":main()
