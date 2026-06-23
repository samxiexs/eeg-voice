# Run FEIS EEG-Only

All commands run from `app/`.

```bash
bash run_direct_eeg_only.sh
```

Useful overrides:

```bash
STAGES=stimuli,thinking EPOCHS=120 bash run_direct_eeg_only.sh
DEVICE=cuda SPLIT=test_seen bash run_direct_eeg_only.sh
```

The runnable path accepts only EEG and stage indices. Any legacy route using an
external identity code has been removed from the app code surface.
