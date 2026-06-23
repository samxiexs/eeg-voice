# KaraOne data analysis

- Data root: `/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/karaone_overt_recon_bundle/app/../data/karaone`
- Subjects: 14 (`MM05 MM08 MM09 MM10 MM11 MM12 MM14 MM15 MM16 MM18 MM19 MM20 MM21 P02`)
- Trials: 1913
- Segments: 7652
- Labels: 11 (`/diy/`, `/iy/`, `/m/`, `/n/`, `/piy/`, `/tiy/`, `/uw/`, `gnaw`, `knew`, `pat`, `pot`)
- Audio duration median: 1.440s
- Subject bundles: 14, same structure: True

## Trial counts by subject

- MM05: 165
- MM08: 131
- MM09: 132
- MM10: 132
- MM11: 132
- MM12: 132
- MM14: 132
- MM15: 132
- MM16: 132
- MM18: 132
- MM19: 132
- MM20: 132
- MM21: 132
- P02: 165

## Label counts

- /diy/: 174
- /iy/: 173
- /m/: 174
- /n/: 174
- /piy/: 174
- /tiy/: 174
- /uw/: 174
- gnaw: 174
- knew: 174
- pat: 174
- pot: 174

## Stage length summary

- clearing: n=1913, valid min/median/p95/max=858/1279/1280/1286
- overt_like: n=1913, valid min/median/p95/max=371/575/722/1317
- stimulus_like: n=1913, valid min/median/p95/max=491/514/703/1316
- thinking: n=1913, valid min/median/p95/max=1239/1280/1282/1289

## Recommended experiment use

- Primary positive-control task: `overt_like` EEG -> same-trial wav.
- Main imagined-speech task: initialize from overt reconstruction, then fine-tune/evaluate `thinking` EEG -> same-trial overt wav.
- Always report zero-EEG, mean-latent, and oracle-codec controls.
