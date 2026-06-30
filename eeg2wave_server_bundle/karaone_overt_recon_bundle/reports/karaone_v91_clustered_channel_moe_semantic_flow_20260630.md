# KaraOne v9.1 Clustered Channel-MoE Semantic Flow

Date: 2026-06-30

## What v9.1 Adds

v9.1 is a new pipeline beside v9, not a patch over v8.  It keeps the v9
semantic-first gate and adds three canonical components:

1. Train-only cluster bank
   - EEG descriptors: per-trial normalized log variance, bandpower, covariance sketch, low-frequency envelope.
   - Speech descriptors: HuBERT summary, semantic token histogram, active/duration/energy/onset prosody.
   - Cross-modal clusters: k-means over concatenated train-normalized EEG and speech descriptors.
   - Centroids are fitted only from `subject_train`; `P02` and `MM21` are assigned to existing centroids for evaluation only.

2. Channel-MoE EEG encoder
   - Replaces v9 simple channel reliability with sparse top-k channel gating.
   - Uses learned channel embeddings plus per-channel descriptors.
   - Assigns channels to functional experts and projects expert temporal streams into EEG patch tokens.
   - Exposes `channel_gate`, `channel_assign`, load balance, entropy and sparsity diagnostics.

3. Speech-specific codec-space flow
   - Uses factorized semantic/prosody condition instead of waveform regression.
   - Implements a NeuroSonic-style time-conditioned gated Transformer with AdaLN, RMSNorm and deterministic Heun sampling.
   - Adds codec consistency and chunk-boundary continuity losses.
   - Waveform/transport remains diagnostic unless semantic/prosody gates pass.

## Key Files

- `app/src/karaone_v91/clusters.py`: descriptor extraction, train-only k-means, audit payloads.
- `app/src/karaone_v91/data.py`: cluster bank wrapper, clustered dataset, cluster-balanced sampler, channel names.
- `app/src/karaone_v91/model.py`: Channel-MoE EEG encoder and v9.1 semantic/prosody/flow model.
- `app/src/karaone_v91/losses.py`: cluster-aware InfoNCE, hard negatives, MoE regularizers.
- `app/src/karaone_v91/transport.py`: codec-space conditional flow and Heun sampler.
- `app/src/karaone_v91/eval.py`: v9 metrics plus cluster metrics and channel report writers.
- `app/scripts/build_karaone_v91_clusters.py`: Stage 0 cluster/audit cache builder.
- `app/scripts/audit_karaone_v91_protocol.py`: train-only cluster and gate audit.
- `app/scripts/train_karaone_v91.py`: pretrain/align/transport runner.
- `run_karaone_v91.sh`: one-command shell wrapper.
- `app/tests/test_karaone_v91_smoke.py`: synthetic forward/backward, sampler, flow and channel reports.

## Run Commands

From bundle root:

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/karaone_overt_recon_bundle

# Stage 0: train-only overt cluster bank + protocol audit
./run_karaone_v91.sh audit

# Synthetic smoke + one-step real-data align smoke
MAX_STEPS=2 DEVICE=cpu ./run_karaone_v91.sh smoke

# Overt route: clusters -> audit -> pretrain -> align
DEVICE=cpu ./run_karaone_v91.sh overt 20 v91_overt_full_20260630

# Thinking route: train thinking semantic/prosody alignment only
DEVICE=cpu ./run_karaone_v91.sh thinking 20 v91_thinking_align_20260630

# Stage-specific runs
DEVICE=cpu ./run_karaone_v91.sh pretrain 20 v91_pretrain
DEVICE=cpu ./run_karaone_v91.sh align 40 v91_align
CKPT=artifacts/outputs_karaone/<run>/checkpoints/best.pt DEVICE=cpu ./run_karaone_v91.sh flow 10 v91_flow
```

## Research Gate

v9.1 keeps the same semantic-first interpretation: transport/audio generation is not evidence of EEG-to-speech success unless subject-holdout semantic/prosody decoding passes.

Subject validation gate:

```text
semantic_over_zero_gain > 0.01
semantic_over_mean_gain > 0
semantic_top3_gain_over_mean > 0.02
same_label_cross_subject_gain >= 0
prompt_acc >= 0.13
pred_std_ratio_median in [0.7, 1.5]
pred_pairwise_corr_median < 0.75
channel gate entropy not collapsed
```

Subject test should have the same signs and is reported separately.

## Verification Performed

```text
python3 -m py_compile app/src/karaone_v91/*.py app/scripts/build_karaone_v91_clusters.py app/scripts/audit_karaone_v91_protocol.py app/scripts/train_karaone_v91.py app/tests/test_karaone_v91_smoke.py
/opt/anaconda3/bin/python app/tests/test_karaone_v91_smoke.py
/opt/anaconda3/bin/python app/scripts/build_karaone_v91_clusters.py --config app/configs/karaone_v91.yaml --stages overt_like --out /tmp/karaone_v91_clusters_smoke.npz --audit-out /tmp/karaone_v91_cluster_audit_smoke.json --max-rows 48
/opt/anaconda3/bin/python app/scripts/audit_karaone_v91_protocol.py --config app/configs/karaone_v91.yaml --cluster-audit /tmp/karaone_v91_cluster_audit_smoke.json --out /tmp/karaone_v91_protocol_audit_smoke.json
/opt/anaconda3/bin/python app/scripts/train_karaone_v91.py --config app/configs/karaone_v91.yaml --phase align --stages overt_like --epochs 1 --batch-size 4 --device cpu --cluster-bank /tmp/karaone_v91_clusters_smoke.npz --max-steps 1 --run-suffix smoke_tmp
```

The 1-step smoke is expected not to pass the research gate.  It verified code
closure and produced channel diagnostics.  The smoke metrics showed non-collapsed
Channel-MoE gates, but semantic predictions were still collapsed because the
model was effectively untrained.

## Remaining Work

- Run full Stage 0 cluster bank without `--max-rows` for overt and thinking.
- Train overt pretrain and alignment long enough to evaluate semantic gate.
- Only train/report flow as diagnostic until semantic/prosody gate passes.
- Add permutation or leave-channel-out importance runs for stable channel claims.
- Add rendered/oracle audio path for multi-resolution STFT loss once codec render is wired.
