# OpenVoice-EEG 0722 V1：开放词汇、label-free EEG→语音

## 研究目标与声明边界

正式推理只有四个输入：

```text
generate(eeg, channel_xyz, channel_mask, time_mask)
```

主生成路径不接收 label、label probability、文本、subject ID、dataset ID 或真实音频。
Label 只进入独立 semantic auxiliary head；`text_only_ablation` 是评估对照，不是主模型。

当前证据边界固定为：KaraOne 可做同 trial 声学监督；FEIS 只做 subject-label 弱语义监督；
ds004306 只做 EEG 自监督、通道鲁棒性与 prototype-level 演示。不能把后两者描述成逐 trial
精确语音重建，更不能把 prompted imagined speech 直接描述成临床幻听重建。

## 已实现的工程结构

- 配置：`app/configs/open_vocab_0722_v1.yaml`
- label folds：`app/configs/split/open_vocab_0722_label_folds.yaml`
- 模型/数据/loss/metrics/lineage：`app/src/open_vocab_0722/`
- 训练、合成、审计、gate：`app/scripts/*open_vocab_0722*.py`
- runner：`app/run_open_vocab_0722_v1.sh`
- 输出：`artifacts/open_vocab_0722_v1/`

旧 0715/0721 checkpoint 不兼容。只有其不依赖 label 的共享权重可以另写显式迁移器使用；
0722 默认重新训练 label-free 音频先验。

## 数据轨道

### Track A

沿用已验证的 14×1280、256 Hz NPZ。`prepare` 生成 montage registry 并审计：

- authoritative subject split 完整且互斥；
- pairing confidence 只允许三种已声明等级；
- KaraOne/FEIS held-out reference audio 不得标为 audio-fit；
- ds004306 三个 category sound 明确不具备生成监督资格。

### Track B

`track-b` 从原始 stage/EEGLAB 文件生成独立 variable-montage 输出：FEIS 14、KaraOne 约62、
ds004306 排除 M1/M2/EOG 后的有效 EEG 通道。它不插值不存在的电极，保留真实坐标、通道和
时间 mask，且不覆盖 Track A。脚本会输出绝对路径 runtime config。

在 Track B 生成、registry/QC、通道置换不变性和 full/common view 验证完成前，不能删除
48 GB ds004306 原始 `.set/.fdt`。

## 音频先验

冻结 EnCodec 和声学 teacher；正式中英文公开语料模式使用 XLS-R 300M，尚未加入公开语料的
project-only 探索模式使用英语 HuBERT base。两种 teacher 的 cache、lineage、checkpoint 和输出目录
完全隔离，HuBERT checkpoint 不得用于正式 XLS-R 实验。声学 token 被分片保存，避免一次加载约100小时 teacher token。
Audio Condition Encoder 将声学 teacher token 投影到50×192 condition；MaskGIT 只接收 condition，
没有 label embedding 或 dataset slice。公开语音 manifest 对 LibriTTS clean/AISHELL-1 做
固定 seed、speaker-disjoint split、2秒切片和 held-out SHA 排除；不读取文本训练生成器。

## EEG encoder 与 Adapter-MoE

每通道64 sample patch、hop32，加入三维坐标/时间/局部质量；50个 learned queries 汇聚
任意数量的 channel-time tokens。Router 只看到有效局部 token，函数签名中没有 label、
dataset 或 subject。前5 epoch 为 sigmoid soft routing，之后 universal FFN 始终开启并选
Top-2 specialist；4个48维低秩 adapter 带10% expert dropout。

参数量匹配 dense control 使用相同 adapter 参数，但固定均匀权重、关闭 adaptive routing。
`make_open_vocab_0722_ablation.py` 按强制消融顺序生成独立 lineage/output 配置。

## Loss 路由

总目标实现为 code CE、global exact/weak CLIP、monotonic local alignment、结构 loss、
same-label semantic multi-positive、text auxiliary、channel consistency、domain adversary、
masked EEG reconstruction 和 MoE regularization。

- KaraOne：exact code/global/local/structure + semantic/self-supervision；
- FEIS：0.25 global weak pair + same-label/text semantic + self-supervision，不用 code/local/structure；
- ds004306：只用 self-supervision、channel/domain/MoE。

同 label 不同 trial 在 semantic space 是0.15弱正，在 acoustic space 保持 hard negative。

## 泛化实验

- G1：subject-disjoint seen-label；
- G2：KaraOne `gnaw/knew/pat/pot` leave-one-word-out，训练受试者内验证；
- G3：leave-one-word-out + unseen subject；
- FEIS 同名归一化 label 与 KaraOne holdout 联动排除。

G2/G3 每个 fold 必须跑 seeds 15/31/47；至少2/3单独通过且 pooled 通过，才能声称开放词汇。
`seed-config` 会为三个 seed 生成隔离的 EEG/checkpoint/output 配置，同时只读复用同一个
label-free audio prior；`aggregate-seeds --reports ... --output ...` 生成 gate 所需 seed summary。

## 正式评估

每个样本生成 reference、codec oracle、audio-condition oracle、EEG-conditioned、same-label
shuffled、any shuffled、channel shuffled、zero EEG、text-only、dataset-prior 十种目录。
主指标是±250 ms envelope correlation、soft-DTW envelope、modulation correlation、80-bin
log-mel MAE、multi-resolution STFT、retrieval R@1/R@5/MRR（当前 manifest 写R@1/R@5）和可选
XLS-R content cosine。raw waveform correlation 只作补充。

正式 gate 只读取 KaraOne validation 的 same-trial 样本，要求 EEG-conditioned 同时优于
same-label shuffled 与 zero；还绑定 model audit、lineage、audio/eeg checkpoint SHA 和
validation report SHA。缺少三 seed matched-dense 对照时，MoE 正式 gate 必须保持失败，
但 validation WAV 仍可作为 exploratory 结果检查。locked test 没有 bypass。

## 推荐执行顺序

```bash
bash app/run_open_vocab_0722_v1.sh prepare

LIBRITTS_ROOT=/absolute/path/to/LibriTTS \
AISHELL_ROOT=/absolute/path/to/AISHELL-1 \
bash app/run_open_vocab_0722_v1.sh public-manifest

bash app/run_open_vocab_0722_v1.sh public-cache
bash app/run_open_vocab_0722_v1.sh teachers
bash app/run_open_vocab_0722_v1.sh train-audio
bash app/run_open_vocab_0722_v1.sh pretrain-eeg
bash app/run_open_vocab_0722_v1.sh train-eeg
bash app/run_open_vocab_0722_v1.sh select-eeg
bash app/run_open_vocab_0722_v1.sh synthesize karaone validation
bash app/run_open_vocab_0722_v1.sh synthesize feis validation
bash app/run_open_vocab_0722_v1.sh synthesize ds004306 validation
bash app/run_open_vocab_0722_v1.sh audit-model
bash app/run_open_vocab_0722_v1.sh gate
```

一键 Track A G1 proof-of-concept：

```bash
LIBRITTS_ROOT=/absolute/path/to/LibriTTS \
AISHELL_ROOT=/absolute/path/to/AISHELL-1 \
DEVICE=mps bash app/run_open_vocab_0722_v1.sh all
```

所有长阶段都有 tqdm，runner 另有总阶段进度条。`PROJECT_ONLY=1` 只允许 smoke/debug，不能
作为公开语音开放词汇结果。locked test 只有 gate 真正通过后才可运行：

```bash
bash app/run_open_vocab_0722_v1.sh test
bash app/run_open_vocab_0722_v1.sh synthesize karaone test
```

## 强制消融

依次生成配置并各自训练三 seed：

```bash
bash app/run_open_vocab_0722_v1.sh ablation-config label_free_dense
bash app/run_open_vocab_0722_v1.sh ablation-config global_clip
bash app/run_open_vocab_0722_v1.sh ablation-config local_alignment
bash app/run_open_vocab_0722_v1.sh ablation-config semantic_text
bash app/run_open_vocab_0722_v1.sh ablation-config adapter_moe
```

若 Adapter-MoE 三 seed 平均 composite 未比 matched dense 高0.02，或新受试者下降、25%通道
缺失下降超过15%、condition cosine低于0.80、dataset probe超过0.383、expert dying/collapse，
正式模型回退 dense control。
