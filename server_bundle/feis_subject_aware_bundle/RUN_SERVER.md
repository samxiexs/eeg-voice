# FEIS Server Bundle

This folder is intended to be copied to a server as a single unit.

## Layout

```text
feis_subject_aware_bundle/
  app/
  data/feis/
  models/
  artifacts/
  reports/
```

## Local SSL Model Path

The alignment SSL config points to:

`app/configs/alignment_ssl_local.yaml -> targets.ssl_model_name_or_path = ../models/hubert-base-ls960`

The SSL model directory has already been copied into the bundle.

Use it like this:

```bash
cd app
python scripts/extract_audio_targets.py --config configs/alignment_ssl_local.yaml
python scripts/train_alignment.py --config configs/alignment_ssl_local.yaml --protocol G
```
