# Voice EEG Dataset Sample Downloads

## 1. 样例根目录

样例数据统一放在：

```text
data/voice_eeg_dataset_samples/
```

该目录已经加入 `.gitignore`，不会进入 Git commit。目录内有：

- `README.md`: 样例目录总览。
- `manifest.json`: 每个数据集的下载/复制状态。
- `<category>/<dataset_slug>/README.md`: 单个数据集的样例说明。
- `<category>/<dataset_slug>/status.json`: 单个数据集的机器可读状态。
- `<category>/<dataset_slug>/local/`: 从本地已有数据复制的完整样例。
- `<category>/<dataset_slug>/remote/`: 自动下载的公开 metadata 或小文件。
- `<category>/<dataset_slug>/probe_artifacts/`: 之前探测阶段生成的小型可审计文件。

重新生成命令：

```bash
python3 scripts/download_voice_eeg_dataset_samples.py
```

允许下载较大远程文件时使用：

```bash
python3 scripts/download_voice_eeg_dataset_samples.py --allow-large
```

只更新某个数据集：

```bash
python3 scripts/download_voice_eeg_dataset_samples.py --dataset ds005345
python3 scripts/download_voice_eeg_dataset_samples.py --dataset ds006104
```

## 2. 当前运行结果

本次运行结果：

| 项目 | 数量 |
| --- | ---: |
| 数据集目录 | 21 |
| 已有本地样例或公开小文件 | 21 |
| 需要人工授权/选文件/大文件下载 | 0 |
| 样例目录大小 | 约 4.3G |

已经具备较完整本地样例的数据集：

| 数据集 | 当前样例内容 |
| --- | --- |
| `ds004408` | `sub-001` run-01 BrainVision EEG `.eeg/.vhdr/.vmrk`，`audio01.wav`，`audio01.TextGrid` |
| `ds005345` | `sub-01` run-1 到 run-4 EEG `.npz`，`single_female.wav`，`single_male.wav`，`mix.wav`，acoustic/word CSV |
| `ds004718` | 粤语句子 wav，`sub-HK001_task-lppHK_eeg_preprocessed.set`，timing / acoustic probe files |
| `ds006104` | `sub-P01` 和 `sub-S01` 派生 EEG `.npz`，events/channels，8 条 happy/angry 短语音 wav |

已经有公开 metadata 或 probe snippets 的数据集：

| 数据集 | 当前状态 |
| --- | --- |
| `ds006434` | OpenNeuro metadata/events/channels/stimulus wav + probe snippets |
| Etard continuous speech EEG `7086209` | Zenodo metadata |
| `ds007591` | OpenNeuro metadata + probe snippets |
| ESAA `7078451` | Zenodo metadata + previous probe snippets |
| NJU AAD `7253438` | Zenodo metadata |
| AASD `17413336` | Zenodo metadata |
| MS-AASD `17149387` | Zenodo metadata |
| `ds006465` / 3M-CPSEED | OpenNeuro dataset metadata |
| Cantonese tone/syllable `7750292` | Zenodo metadata |
| KUL AAD `4004271` | Zenodo metadata + previous probe snippets |
| DTU AAD `1199011` | Zenodo metadata + previous probe snippets |
| 255ch EEG-AAD `4518754` | Zenodo metadata + previous probe snippets |
| OpenMIIR | previous probe snippets |
| MUSIN-G `ds003774` | OpenNeuro metadata + previous probe snippets |
| MAD-EEG `4537751` | Zenodo metadata + previous probe snippets |
| SparrKULee / EEGDash | EEGDash record HTML |
| Weissbart natural speech EEG | Zenodo metadata |

## 3. 已移除的手动项

之前那 13 个 `manual_required` 数据集没有你本地已经下载好的样例文件，也不适合继续占当前样例位，所以已经从当前 active sample pool 移除。替换进来的 3 个公开数据集是：

| 新增数据集 | 当前状态 |
| --- | --- |
| Etard continuous speech EEG `7086209` | Zenodo metadata 已落盘 |
| AASD `17413336` | Zenodo metadata 已落盘 |
| MS-AASD `17149387` | Zenodo metadata 已落盘 |

## 4. 当前可直接用于代码 dry-run 的路径

英文自然语音：

```text
data/voice_eeg_dataset_samples/english/ds004408/local/eeg/
data/voice_eeg_dataset_samples/english/ds004408/local/stimuli/
```

普通话多说话人：

```text
data/voice_eeg_dataset_samples/mandarin_synthetic/ds005345/local/eeg/
data/voice_eeg_dataset_samples/mandarin_synthetic/ds005345/local/stimuli/
data/voice_eeg_dataset_samples/mandarin_synthetic/ds005345/local/annotation/
```

粤语自然语音：

```text
data/voice_eeg_dataset_samples/cantonese/ds004718/local/
```

受控短语音：

```text
data/voice_eeg_dataset_samples/controlled_speech/ds006104/local/
```

## 5. 当前结论

当前样例目录已经能支撑第一轮工程验证：

```text
ds004408: EEG + English audiobook audio/TextGrid
ds005345: EEG + Mandarin synthetic female/male/mix audio + annotation
ds004718: EEG + Cantonese sentence audio + timing/acoustic probes
ds006104: EEG + controlled phoneme/CV/VC/style stimuli
```

后续优先从这些已经落到样例根目录的数据集中继续扩展真正的 subject/trial：

```text
ESAA
NJU AAD
Etard continuous speech EEG
AASD
MS-AASD
KUL AAD
DTU AAD
255ch EEG-AAD
```
