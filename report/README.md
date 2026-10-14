# Perceived EEG-to-Speech reports

The current research plan is NeuroSpeech v2.1. Its data stage is complete (see the progress report); the model code has only run on synthetic fixtures. The v1 interface checks below still describe the single-dataset pipeline.

| File | Purpose |
|---|---|
| [NeuroSpeech research plan v2.1](neurospeech_research_plan_2026-10-13.md) | Current plan (English): four English perceived-speech sources (Kymata English only) and KaraOne for transfer; REVE adaptation, physiological alignment, CFM, Simulation and cross-subject training; batch-by-batch processing that keeps only processed data |
| [Plan, Chinese original](neurospeech_research_plan_2026-10-13.zh.md) | Unchanged source of the translation |
| [Progress report: data stage](neurospeech_progress_2026-10-13.md) · [中文版](neurospeech_progress_2026-10-13.zh.md) | What was done after the plan: datasets, synchronization findings, annotations, data decisions, registry and splits, differences from the plan, and what is not done |
| [Dataset size and English-directory audit](dataset_storage_audit_2026-10-13.json) | Kymata English whitelist (117.767 GB of selected files, with per-file SHA256 and URLs), KaraOne archives (24.824 GB) and source evidence; processed sizes are budget estimates |
| [NeuroSpeech model interface check](neurospeech_model_interface_check_2026-10-13.json) | Pinned REVE / WavLM / Vocos / Whisper revisions and a small synthetic-input interface check |
| Historical execution plan v1 (`06_execution_plan_2026-10-13.md`) | Protocol of the single-dataset TCN implementation; superseded by v2.1. The file was removed from the repository and is still available with `git show 3b8ca941:report/06_execution_plan_2026-10-13.md` (Chinese) |
| [Project run instructions](../README.md) | v2.1 data commands and status, v1 entry points, run order and the EEG-only input contract |
| [Cleanup record](cleanup_2026-10-13.json) | Old pipelines, data caches and failed output directories deleted with the user's authorization |
| [Real-file and model checks](implementation_checks.json) | v1: real EEG / WavLM / Vocos / ASR interfaces and the full gradient chain; scientific performance is still unverified |
| [Dataset unit audit](dataset_unit_audit.json) | ds004408: check of the original authors' numeric scale, the scaling evidence and the limits on interpreting absolute units |

Per-dataset ingest evidence (source receipts, synchronization measurements, cleanup logs) lives next to the data in `data/processed/neurospeech/{dataset}/manifest.json`, `metadata/` and `cleanup.jsonl`.

After formal training and evaluation the `report` command generates the result tables; a computation-graph check is never presented as a speech-reconstruction result.
