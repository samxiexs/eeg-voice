# EEG-audio dataset probe detailed report

This report is generated from real remote metadata probes. Full EEG archives are not downloaded by default; partial byte-range artifacts are explicitly marked.

## Dataset Summary

| Dataset | Source | Priority | Fit | URL |
| --- | --- | --- | --- | --- |
| ds004408 | OpenNeuro | core | Natural speech pretraining; word/phoneme TextGrid alignment | https://openneuro.org/datasets/ds004408 |
| ds004940 | OpenNeuro | pretraining-response | Heard English sentence/word EEG with audio stimuli and response task; auxiliary only for imagined-speech work | https://openneuro.org/datasets/ds004940 |
| ds007591 | OpenNeuro | secondary | Speech production/covert speech sanity check; small subject count | https://openneuro.org/datasets/ds007591 |
| ds005170 | OpenNeuro | p2-imagined | Chinese imagined speech probe; raw EDF plus FIF/PKL derivatives and text stimuli | https://openneuro.org/datasets/ds005170 |
| ds003626 | OpenNeuro | p2-inner-speech | Spanish inner/pronounced/visualized speech commands; 10 subjects and 5640 trials | https://openneuro.org/datasets/ds003626 |
| ds004306 | OpenNeuro | p2-semantic-proxy | Auditory/visual/orthographic perception and semantic imagination proxy | https://openneuro.org/datasets/ds004306 |
| kara_one | Web | p2-imagined-overt | Imagined and vocalized phonemic/single-word prompts with EEG, face tracking, and audio | https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html |
| feis_3554128 | Zenodo | p2-low-density | Heard/imagined/spoken English phonemes plus Chinese syllables with recorded audio | https://zenodo.org/records/3554128 |
| ugr_mindvoice | Web | p2-overt-covert | Iberian Spanish overt/covert EEG-audio dataset; use OSF listing and GitHub code first | https://osf.io/6sh5d |

## Target-Level Evidence

### ds004408 - EEG responses to continuous naturalistic speech

OpenNeuro S3 first page: `1000` keys; truncated=`True`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 338 | False | outputs/imagined_speech_probe_artifacts/ds004408/01_dataset_description.json | Name=EEG responses to continuous naturalistic speech; DatasetDOI=doi:10.18112/openneuro.ds004408.v1.0.8; License=CC0 |
| participants | tsv | 577 | False | outputs/imagined_speech_probe_artifacts/ds004408/02_participants.tsv | columns=['\ufeffparticipant_id', 'age', 'sex', 'hand', 'weight', 'height']; preview_rows=[['sub-001', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a'], ['sub-002', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a']] |
| audio01 TextGrid | textgrid | 180689 | False | outputs/imagined_speech_probe_artifacts/ds004408/03_audio01_TextGrid.TextGrid | nonempty_labels=2465; duration_sec_hint=177.54; first_nonempty=['sil', 'HH', 'IY1', 'W', 'AH0', 'Z', 'AH0', 'N', 'OW1', 'L'] |
| audio01 wav header | wav | 4096 | True | outputs/imagined_speech_probe_artifacts/ds004408/04_audio01_wav_header.wav.header.bin | sample_rate=44100; channels=2; bits=16; duration_sec_est=177.563 |
| run01 EEG sidecar | json | 487 | False | outputs/imagined_speech_probe_artifacts/ds004408/05_run01_EEG_sidecar.json | SamplingFrequency=512.0; EEGReference=n/a; Manufacturer=Brain Products |
| run01 channels | tsv | 5920 | False | outputs/imagined_speech_probe_artifacts/ds004408/06_run01_channels.tsv | columns=['name', 'type', 'units', 'description', 'sampling_frequency', 'status', 'status_description']; preview_rows=[['A1', 'EEG', 'V', 'ElectroEncephaloGram', '512.0', 'good', 'n/a'], ['A2', 'EEG', 'V', 'ElectroEncephaloGram', '512.0', 'good', 'n/a']] |
| run01 BrainVision header | text | 2984 | False | outputs/imagined_speech_probe_artifacts/ds004408/07_run01_BrainVision_header.txt | bytes=2984; magic=None; range=None |
| run01 raw EEG bytes | binary | 512 | True | outputs/imagined_speech_probe_artifacts/ds004408/08_run01_raw_EEG_bytes.bin | bytes=512; magic=78870f5132ad1c51e66b6dd05bead0cf; range=bytes 0-511/51380224 |

### ds004940 - Auditory N400 active/passive sentence EEG

OpenNeuro S3 first page: `1000` keys; truncated=`True`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 20752 | False | outputs/imagined_speech_probe_artifacts/ds004940/01_dataset_description.json | Name=Neurophysiological measures of covert semantic processing in neurotypical adolescents actively ignoring spoken sentence inputs: A high-density event-related potential (ERP) study.; DatasetDOI=doi:10.18112/openneuro.ds004940.v1.0.1; License=CC0 |
| README | text | 34317 | False | outputs/imagined_speech_probe_artifacts/ds004940/02_README.txt | bytes=34317; magic=None; range=None |
| stimulus parameters | tsv | 67451 | False | outputs/imagined_speech_probe_artifacts/ds004940/03_stimulus_parameters.tsv | columns=['stim_key', 'stim_file', '1', '2', '3', '4', '5', '6', '7', '8', 'known_word', 'stim_dur(s)', 'target_onset(s)', 'target_end(s)', 'target_dur(s)', 'cloze-probability%_div', 'linguistic-group_div', 'linguistic-group_reasoning']; preview_rows=[['sound1', 'NPC_bake.wav', 'The', 'cake', 'is', 'in', 'the', 'oven', 'to', 'bake.', 'bake', '3.578784', '3.117626', '3.548376', '0.43075', '0.92248062', '1', 'Expectation not disrupted/Congruent and incongruent type match'], ['sound2', 'NPC_cat.wav', 'Meow', 'goes', 'the', 'cat.', 'n/a', 'n/a', 'n/a', 'n/a', 'cat', '1.856939', '1.408013', '1.820172', '0.412159', '0.968992248', '1', 'Expectation not disrupted/Congruent and incongruent type match']] |
| sub001 active events | tsv | 88603 | False | outputs/imagined_speech_probe_artifacts/ds004940/04_sub001_active_events.tsv | columns=['onset', 'duration', 'stim_onset_s_', 'stim_dur_s_', 'type', 'trial_type', 'stim_file']; preview_rows=[['n/a', 'n/a', '0.993', '1.16585', 'Picture-Sound', 'intro', '(Intro_01)Thanksforparticipating.wav'], ['n/a', 'n/a', '2.156', '5.523152', 'Picture-Sound', 'intro', '(Intro_02)TaskDescription.wav']] |
| sub001 active channels | tsv | 3342 | False | outputs/imagined_speech_probe_artifacts/ds004940/05_sub001_active_channels.tsv | columns=['name', 'type', 'units', 'status', 'status_description']; preview_rows=[['A1_Cz', 'EEG', 'microV', 'good', 'n/a'], ['A2', 'EEG', 'microV', 'good', 'n/a']] |
| example stimulus wav header | wav | 4096 | True | outputs/imagined_speech_probe_artifacts/ds004940/06_example_stimulus_wav_header.wav.header.bin | sample_rate=44100; channels=1; bits=16; duration_sec_est=2.613 |
| sub001 active BDF bytes | binary | 512 | True | outputs/imagined_speech_probe_artifacts/ds004940/07_sub001_active_BDF_bytes.bin | bytes=512; magic=ff42494f53454d494163743031202020; range=bytes 0-511/675522048 |

### ds007591 - Delineating neural contributions to EEG-based speech decoding

OpenNeuro S3 first page: `126` keys; truncated=`False`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 1122 | False | outputs/imagined_speech_probe_artifacts/ds007591/01_dataset_description.json | Name=Delineating neural contributions to EEG-based speech decoding; DatasetDOI=doi:10.18112/openneuro.ds007591.v1.0.1; License=CC0 |
| participants | tsv | 65 | False | outputs/imagined_speech_probe_artifacts/ds007591/02_participants.tsv | columns=['participant_id', 'age', 'sex']; preview_rows=[['sub-1', 'n/a', 'n/a'], ['sub-2', 'n/a', 'n/a']] |
| events | tsv | 5456 | False | outputs/imagined_speech_probe_artifacts/ds007591/03_events.tsv | columns=['onset', 'duration', 'trial_type', 'value', 'session_type', 'task_condition']; preview_rows=[['58.96484375', '6.25', 'yellow', '4', 'calibration', 'minimally overt'], ['68.859375', '6.25', 'green', '0', 'calibration', 'minimally overt']] |
| channels | tsv | 3135 | False | outputs/imagined_speech_probe_artifacts/ds007591/04_channels.tsv | columns=['name', 'type', 'units', 'sampling_frequency', 'status']; preview_rows=[['EEG001', 'EEG', 'V', '256', 'good'], ['EEG002', 'EEG', 'V', '256', 'good']] |
| EEG sidecar | json | 823 | False | outputs/imagined_speech_probe_artifacts/ds007591/05_EEG_sidecar.json | SamplingFrequency=256; EEGReference=n/a (raw, pre-reference); Manufacturer=g.tec medical engineering GmbH |
| raw EDF bytes | binary | 512 | True | outputs/imagined_speech_probe_artifacts/ds007591/06_raw_EDF_bytes.bin | bytes=512; magic=30202020202020205820582058205820; range=bytes 0-511/82497276 |

### ds005170 - Chisco Chinese imagined speech

OpenNeuro S3 first page: `1000` keys; truncated=`True`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 1440 | False | outputs/imagined_speech_probe_artifacts/ds005170/01_dataset_description.json | Name=Chisco; DatasetDOI=doi:10.18112/openneuro.ds005170.v1.1.2; License=CC0 |
| README | text | 2778 | False | outputs/imagined_speech_probe_artifacts/ds005170/02_README.txt | bytes=2778; magic=None; range=None |
| text split xlsx | xlsx | 11390 | False | outputs/imagined_speech_probe_artifacts/ds005170/03_text_split_xlsx.xlsx | sheets=[]; shape=(None, None); preview_rows=[] |
| sub01 ses01 raw EDF bytes | binary | 512 | True | outputs/imagined_speech_probe_artifacts/ds005170/04_sub01_ses01_raw_EDF_bytes.bin | bytes=512; magic=30202020202020205820582058205820; range=bytes 0-511/595530560 |
| sub01 preprocessed FIF bytes | binary | 512 | True | outputs/imagined_speech_probe_artifacts/ds005170/05_sub01_preprocessed_FIF_bytes.bin | bytes=512; magic=000000640000001f0000001400000000; range=bytes 0-511/152755040 |

### ds003626 - Inner Speech

OpenNeuro S3 first page: `187` keys; truncated=`False`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 647 | False | outputs/imagined_speech_probe_artifacts/ds003626/01_dataset_description.json | Name=Inner Speech; DatasetDOI=doi:10.18112/openneuro.ds003626.v2.1.2; License=CC0 |
| README | text | 1675 | False | outputs/imagined_speech_probe_artifacts/ds003626/02_README.txt | bytes=1675; magic=None; range=None |
| sub01 ses01 events dat | binary | 2048 | True | outputs/imagined_speech_probe_artifacts/ds003626/03_sub01_ses01_events_dat.bin | bytes=2048; magic=8002636e756d70792e636f72652e6d75; range=bytes 0-2047/6813 |
| sub01 ses01 EEG epochs bytes | binary | 512 | True | outputs/imagined_speech_probe_artifacts/ds003626/04_sub01_ses01_EEG_epochs_bytes.bin | bytes=512; magic=000000640000001f0000001400000000; range=bytes 0-511/236165355 |

### ds004306 - EEG Semantic Imagination and Perception Dataset

OpenNeuro S3 first page: `564` keys; truncated=`False`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 543 | False | outputs/imagined_speech_probe_artifacts/ds004306/01_dataset_description.json | Name=EEG Semantic Imagination and Perception Dataset; DatasetDOI=doi:10.18112/openneuro.ds004306.v1.0.2; License=CC0 |
| participants | tsv | 324 | False | outputs/imagined_speech_probe_artifacts/ds004306/02_participants.tsv | columns=['participant_id', 'vviq', 'bais', 'cap', ' vision_impaired', 'hearing_impaired']; preview_rows=[['sub-03', '4.25', '4.8', 'L', 'n', 'n'], ['sub-08', '3.1', '4.2', 'M', 'n', 'n']] |
| README | text | 801 | False | outputs/imagined_speech_probe_artifacts/ds004306/03_README.txt | bytes=801; magic=None; range=None |
| flower audio bytes | binary | 512 | True | outputs/imagined_speech_probe_artifacts/ds004306/04_flower_audio_bytes.bin | bytes=512; magic=4f6767530002000000000000000031ca; range=bytes 0-511/35314 |
| sub10 preprocessed FIF bytes | binary | 512 | True | outputs/imagined_speech_probe_artifacts/ds004306/05_sub10_preprocessed_FIF_bytes.bin | bytes=512; magic=000000640000001f0000001400000000; range=bytes 0-511/313763012 |

### kara_one - Kara One imagined and articulated speech


| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset page | text | 13409 | False | outputs/imagined_speech_probe_artifacts/kara_one/01_dataset_page.txt | bytes=13409; magic=None; range=None |

### feis_3554128 - Fourteen-channel EEG with Imagined Speech (FEIS) dataset

DOI: `10.5281/zenodo.3554128`
Zenodo file count: `1`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| GitHub README | text | 6299 | False | outputs/imagined_speech_probe_artifacts/feis_3554128/01_GitHub_README.txt | bytes=6299; magic=None; range=None |
| FEIS subject 01 wav listing | json |  |  |  | HTTPError: HTTP Error 403: rate limit exceeded |
| FEIS subject 01 thinking/stimuli zip listing | json |  |  |  | HTTPError: HTTP Error 403: rate limit exceeded |
| FEIS subject 01 prompt wav header | wav | 4096 | True | outputs/imagined_speech_probe_artifacts/feis_3554128/04_FEIS_subject_01_prompt_wav_header.wav.header.bin | sample_rate=44100; channels=1; bits=16; duration_sec_est=1.0 |

### ugr_mindvoice - UGR-MINDVOICE


| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| OSF root listing | json | 5269 | False | outputs/imagined_speech_probe_artifacts/ugr_mindvoice/01_OSF_root_listing.json | json parsed |
| dataset_description | json | 214 | False | outputs/imagined_speech_probe_artifacts/ugr_mindvoice/02_dataset_description.json | Name=[BCI Dataset for Spanish Speech Synthesis] |
| participants | tsv | 450 | False | outputs/imagined_speech_probe_artifacts/ugr_mindvoice/03_participants.tsv | columns=['\ufeffparticipant_id', 'age', 'sex', 'hand', 'weight', 'height']; preview_rows=[['sub-01', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a'], ['sub-02', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a']] |
| sub01 eeg sidecar | json | 463 | False | outputs/imagined_speech_probe_artifacts/ugr_mindvoice/04_sub01_eeg_sidecar.json | SamplingFrequency=1000.0; EEGReference=n/a; Manufacturer=n/a |
| sub01 eeg events | tsv | 653645 | False | outputs/imagined_speech_probe_artifacts/ugr_mindvoice/05_sub01_eeg_events.tsv | columns=['\ufeffonset', 'duration', 'trial_type', 'value', 'sample']; preview_rows=[['16.301', '0.0', 'SilentSyllablesPracticeStartFixation:Text_FAS', '4373', '16301'], ['17.511', '0.0', 'SilentSyllablesPracticeEndFixation:Text_FAS', '5323', '17511']] |
| sub01 channels | tsv | 3720 | False | outputs/imagined_speech_probe_artifacts/ugr_mindvoice/06_sub01_channels.tsv | columns=['\ufeffname', 'type', 'units', 'low_cutoff', 'high_cutoff', 'description', 'sampling_frequency', 'status', 'status_description']; preview_rows=[['Fp1', 'EEG', 'V', '0.0', '500.0', 'ElectroEncephaloGram', '1000.0', 'good', 'n/a'], ['Fz', 'EEG', 'V', '0.0', '500.0', 'ElectroEncephaloGram', '1000.0', 'good', 'n/a']] |
| sub01 audio events | tsv | 634605 | False | outputs/imagined_speech_probe_artifacts/ugr_mindvoice/07_sub01_audio_events.tsv | columns=['onset', 'duration', 'description']; preview_rows=[['16.30090754201592', '0.0', 'SilentSyllablesPracticeStartFixation:Text_FAS'], ['17.51148288070908', '0.0', 'SilentSyllablesPracticeEndFixation:Text_FAS']] |
| GitHub README | text | 16300 | False | outputs/imagined_speech_probe_artifacts/ugr_mindvoice/08_GitHub_README.txt | bytes=16300; magic=None; range=None |
| GitHub config | text | 2288 | False | outputs/imagined_speech_probe_artifacts/ugr_mindvoice/09_GitHub_config.txt | bytes=2288; magic=None; range=None |
