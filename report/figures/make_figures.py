"""Framework figures for report/ (pure drawing, no data):

    .venv-aligned-local/bin/python report/figures/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
for f in ('/System/Library/Fonts/Hiragino Sans GB.ttc', '/System/Library/Fonts/STHeiti Medium.ttc'):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams['font.family'] = [font_manager.FontProperties(fname=f).get_name(), 'DejaVu Sans']
        break
plt.rcParams['axes.unicode_minus'] = False

SPEECH, EEG, GEN, EVAL, VOICE = '#dbe9f6', '#fde2c8', '#d9f0d3', '#ece2f4', '#f6f1c9'
EDGE = {SPEECH: '#3b78b0', EEG: '#d2731e', GEN: '#3d9a3d', EVAL: '#7b5ba6', VOICE: '#a8901c'}


def box(ax, x, y, w, h, title, body='', color=SPEECH, size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8',
                                fc=color, ec=EDGE[color], lw=1.4))
    if body:
        ax.text(x + w / 2, y + h - 1.0, title, ha='center', va='top', fontsize=size, weight='bold')
        ax.text(x + w / 2, y + h - 3.6, body, ha='center', va='top', fontsize=size - 0.9, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=size, weight='bold')
    return dict(l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y + h), b=(x + w / 2, y),
                tl=(x + w * .3, y + h), bl=(x + w * .3, y), tr=(x + w * .7, y + h), br=(x + w * .7, y))


def arrow(ax, a, b, label='', color='#444', style='-|>', ls='-', rad=0., lx=0., ly=0., size=7.6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, color=color, lw=1.3,
                                 linestyle=ls, connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))
    if label:
        ax.text((a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly, label, ha='center', va='center',
                fontsize=size, color=color, style='italic',
                bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.9))


def lane(ax, y, h, text, color):
    ax.add_patch(FancyBboxPatch((0.5, y), 139, h, boxstyle='round,pad=0,rounding_size=1.2',
                                fc=color, ec='none', alpha=0.28))
    ax.text(1.4, y + h - 0.9, text, ha='left', va='top', fontsize=10, weight='bold', color=EDGE[color])


def framework():
    fig, ax = plt.subplots(figsize=(19, 10.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 78)
    ax.axis('off')

    lane(ax, 52, 25, '① Speech side：只用合成语音（TTS）构建，训练时不接触任何 EEG', SPEECH)
    lane(ax, 27, 24, '② EEG side：每个人单独训练（within-person），held-out block 不参与训练与 alignment', EEG)
    lane(ax, 1, 25, '③ Generation & evaluation：生成路径中没有任何 argmax / 候选选择', GEN)

    # ① speech side
    text = box(ax, 2, 59, 16, 12, 'Text', 'Items: 4 西语词 (TOL)\n5 英语短语 (BCI2020)\nSentences: ~6.5k\n中文句 (Chisco)', size=9)
    tts = box(ax, 21, 59, 17, 12, 'TTS (macOS say)', 'Items: 8-12 voices\n× 3 rates\nSentences: 6 voices\n(3F / 3M)', size=9)
    hub = box(ax, 41.5, 63.5, 18, 9, 'HuBERT', 'EN HuBERT-base /\nchinese-hubert-base\nlayer 由跨 voice 识别率选', size=9)
    pca = box(ax, 62.5, 63.5, 17, 9, 'Centre · PCA · whiten', 'Items: K-1 dims\nSentences: 16 dims\n+ log duration', size=9)
    anc = box(ax, 82.5, 63.5, 16, 9, 'CLIP anchors  aₖ', 'speech embedding space\n(frozen speech tower)', size=9)
    mel = box(ax, 41.5, 54, 18, 7, 'SpeechT5 log-mel', '80-bin, decoder 的训练目标', size=9)
    vtr = box(ax, 62.5, 54, 17, 7, 'Voice traits', 'WavLM x-vector\n+ median log F0', color=VOICE, size=9)
    trn = box(ax, 102, 55, 36, 18, 'Train mel diffusion (speech only)',
              'condition = content ⊕ voice traits\n'
              'Items: content = Σ pₖaₖ, p ~ Dirichlet (30% one-hot)\n'
              'Sentences: content = (aₛ + noise) / ‖·‖\n'
              '每个 fold 只用训练句 (open-vocabulary renderer)\n'
              '15% null content → classifier-free guidance\n'
              'v-prediction · cosine schedule · EMA', size=10)
    arrow(ax, text['r'], tts['l'])
    arrow(ax, tts['r'], hub['l'])
    arrow(ax, tts['r'], mel['l'])
    arrow(ax, hub['r'], pca['l'])
    arrow(ax, pca['r'], anc['l'])
    arrow(ax, anc['r'], (102, 68))
    arrow(ax, mel['r'], vtr['l'], style='-', ls=':')
    arrow(ax, vtr['r'], (102, 57.5), 'mel target + traits', lx=5, ly=1.6)

    # ② EEG side
    raw = box(ax, 2, 32, 15, 13, 'Imagined-speech EEG', 'one held-out trial\nTOL 128-ch\nBCI2020 64-ch\nChisco 122-ch (3.3 s)', color=EEG, size=9)
    ali = box(ax, 20, 32, 17, 13, 'Euclidean alignment', '协方差白化，只用该人\n非 held-out 的数据\n+ causal drift removal\n(减去前 20-30 个 trial 均值)', color=EEG, size=9)
    fea = box(ax, 40, 32, 19, 13, 'Filter-bank log-power', '8 bands 1-120 Hz (含 >55 Hz)\nwhole trial + time course\n(3 或 6 段, 相对整段的偏差)\nPCA → 512 (Chisco)', color=EEG, size=9)
    enc = box(ax, 62, 32, 20, 13, 'Personal CLIP encoder', 'Linear: InfoNCE (L-BFGS) / ridge\ncoordinate search by inner CV\ntop-3 calibrated ensemble\nChisco 另有 deep encoder (图 2)', color=EEG, size=9)
    brg = box(ax, 85, 30, 21, 17, 'Continuous condition c', 'Items: calibrated posterior\nc = E[a | EEG] = Σₖ p(k|EEG) aₖ\n\nSentences: c = z / ‖z‖\n\n连续点，不选 item / 句子', color=EEG, size=10)
    pv = box(ax, 117, 32, 21, 13, "Person's voice", 'TOL / BCI2020: 每人指定\n一个合成 voice\nChisco: 按 participants.tsv\n的性别指定合成 voice', color=VOICE, size=9)
    arrow(ax, raw['r'], ali['l'])
    arrow(ax, ali['r'], fea['l'])
    arrow(ax, fea['r'], enc['l'])
    arrow(ax, enc['r'], brg['l'], 'z', ly=1.4)
    arrow(ax, anc['b'], enc['t'], 'InfoNCE targets\n(EEG ↔ speech)', color=EDGE[SPEECH], lx=-7, ly=-4)

    # ③ generation & evaluation (right to left)
    dec = box(ax, 90, 4, 30, 16, 'Mel diffusion decoder', 'adaLN-zero blocks (DiT-style)\ndilated depthwise conv + self-attention\nDDIM 25-50 steps\nCFG 只作用于 content (guidance 2-3)', color=GEN, size=10)
    voc = box(ax, 69, 6, 17, 12, 'SpeechT5 HiFi-GAN', 'log-mel → 16 kHz\nwaveform', color=GEN, size=9)
    out = box(ax, 54, 6, 11, 12, '重建语音', 'wav', color=GEN, size=9)
    evl = box(ax, 24, 3, 26, 18, 'Evaluation (只在这里出现候选)',
              'Items: Whisper-small forced choice\n/ DTW-MCD / HuBERT centroid\n'
              'Sentences: 生成语音的 HuBERT 在 run 内\n~130 句中的 rank percentile, top-k,\n'
              'duration r, Whisper CER\nVoice: speaker ID, sex, pitch error\n'
              'Controls: wrong trial / prior / oracle', color=EVAL, size=9.5)
    ax.text(2.5, 18, 'TOL = Thinking Out Loud\n  (Nieto et al. 2022)\nBCI2020 = BCI Competition\n  2020 Track 3\nChisco = Zhang et al. 2024\n\n生成路径（橙→绿）只传递\n连续向量；候选集合只用于\n右→左最后一步的评估',
            ha='left', va='top', fontsize=8.2, color='#444', linespacing=1.4)
    arrow(ax, brg['b'], (98, 20), 'content c', color=EDGE[EEG], lx=-4)
    arrow(ax, (111.5, 55), (111.5, 20), 'trained weights\n(frozen at test)', color=EDGE[SPEECH], ls='--', ly=6)
    arrow(ax, pv['b'], (117, 20), 'voice traits', color=EDGE[VOICE], lx=3)
    arrow(ax, dec['l'], voc['r'])
    arrow(ax, voc['l'], out['r'])
    arrow(ax, out['l'], evl['r'])
    fig.suptitle('Figure 1.  EEG → speech reconstruction framework（Items 与 Sentences 两条 pipeline 共享同一结构）',
                 fontsize=14, weight='bold', y=0.985)
    fig.savefig(HERE / 'framework.png', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def deep_encoder():
    fig, ax = plt.subplots(figsize=(19, 3.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 26)
    ax.axis('off')
    w, y, h = 17.5, 13, 11.5
    xs = [1 + i * 20 for i in range(7)]
    spec = [
        ('EEG epoch', 'C × T, 128 Hz\nrandom 3 s crop (train)\ncentre crop (test)', EEG),
        ('Euclidean alignment', '每人每 fold 一次\n不含 held-out runs\n(batch 内施加)', EEG),
        ('Temporal filter bank', '32 learned band-passes\nlog-spaced sinc 初始化\n0.25 s kernel\n所有人共享', EEG),
        ('Spatial filters', 'depthwise, 2 / band\nper person\n(EEGNet; subject layer\nof Défossez 2023)', EEG),
        ('Log-power time course', 'square → avg pool 0.25 s\nstride 0.125 s → log\n→ normalise\n(ShallowConvNet)', EEG),
        ('Readout', 'per-person linear\n→ z ∈ CLIP space\n(16 dims + duration)', EEG),
        ('CLIP loss', 'logits = s · cos(z, aₛ)\nCE over all training\nsentence anchors\n(frozen speech tower)', SPEECH),
    ]
    boxes = [box(ax, x, y, w, h, t, b, c, size=10) for x, (t, b, c) in zip(xs, spec)]
    for a, b in zip(boxes, boxes[1:]):
        arrow(ax, a['r'], b['l'])
    ax.text(70, 8, 'Training：dropout 0.5 · AdamW lr 1e-3, wd 0.01 · batch 128 · ≤ 40 epochs · '
            'early stopping (patience 6) on validation runs，指标与测试相同（run 内 rank percentile）',
            ha='center', fontsize=9.5, color='#333')
    ax.text(70, 3.5, 'Inference：z / ‖z‖ 直接作为 mel diffusion renderer 的 content condition（与 linear encoder 相同的下游）',
            ha='center', fontsize=9.5, color='#333')
    fig.suptitle('Figure 2.  Chisco deep encoder（ShallowConvNet-style end-to-end CLIP encoder，eegspeech/deep.py）',
                 fontsize=14, weight='bold', y=0.98)
    fig.savefig(HERE / 'deep_encoder.png', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    framework()
    deep_encoder()
    print('written', HERE / 'framework.png', HERE / 'deep_encoder.png')
