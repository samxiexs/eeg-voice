"""FEIS factored EEG->speech (grid / content x speaker generation).

Realises the "grid" view of FEIS:

    content(16 labels) x speaker(21 subjects) x stage(hear/imagine) x repetition(~10)

Each (subject,label) cell shares ONE target wav (the subject's own recording).
The model FACTORS the target into:
  - content  : decoded FROM the EEG (the hard, scientific part; speaker-independent)
  - speaker  : taken from the KNOWN subject id (an embedding, NOT decoded from EEG)
and a generator combines them into an EnCodec-latent sequence -> frozen vocoder.

Why this beats plain classification:
  - target-aware supervised contrastive (positives = same label) uses the
    "10 reps share one target" structure correctly (no false negatives);
  - hold-out-cell split tests generating UNSEEN (subject x label) combinations,
    which a 336-way classifier/lookup cannot do.
"""

from .model import FactoredEEG2Speech, FactoredConfig

__all__ = ["FactoredEEG2Speech", "FactoredConfig"]
