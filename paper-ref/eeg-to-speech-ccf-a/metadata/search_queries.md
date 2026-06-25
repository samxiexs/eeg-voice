# Search Queries and Selection Notes

## User-requested search directions

- EEG speech reconstruction
- EEG-to-speech reconstruction diffusion
- neural audio codec speech generation
- speech token language model audio generation
- factorized codec speech synthesis

## Inclusion rule

- Include CCF-A venue papers that define audio representation, codec/tokenization, or speech generation interfaces relevant to EEG-to-wav reconstruction.
- Include direct EEG-to-speech / neural decoding papers even when arXiv-only because they are closest to the FEIS/KaraOne task.
- Keep the set compact: about 25 papers, grouped by implementation utility rather than exhaustive citation coverage.

## PDF source preference

1. Reuse existing local PDFs under `paper-ref/unclassified-root-papers/` when present.
2. Prefer official proceedings PDFs for CCF-A papers.
3. Fall back to arXiv PDFs when proceedings PDFs are unavailable or unstable.
4. Record the stable landing page in `papers.csv:url` and the actual source candidates in `download_status`.
