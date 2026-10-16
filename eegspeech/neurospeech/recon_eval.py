"""Evaluation of EEG -> speech reconstruction with measures comparable to the literature.

The 3 s fine-grained measures of `reconstruct` ask for spectro-temporal detail at 43 ms that EEG hardly carries.
Studies of speech tracking in EEG instead correlate band-limited (1-8 Hz) envelopes over long continuous stretches,
where linear backward models reach r of about 0.1-0.2. This module adds those measures:

* the whole usable span of every test record (one person, one content block) is reconstructed by sliding the 3 s
  window in 1 s steps and averaging the overlapping codes, then decoded once;
* broadband and 8-band log-energy envelopes from the Mel, band-passed to 1-8 Hz, correlated over the whole span;
* the same for the audio-side reference (the decoder on the true code: the ceiling of this decoder) and for the
  output without EEG; the null is the predicted envelope circularly shifted against the true one (>= 5 s);
* the modulation spectrum (0.5-16 Hz) of the broadband envelope, as a measure of speech-like rhythm that does not
  need time alignment (compared with no EEG, since any speech-like output scores well on it).

The fine-grained frame measures, word identification and whole-sentence listening of `reconstruct` are kept and
reported next to them.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import math
import numpy as np
import torch
from scipy import signal
from ..utils import atomic_json
from .softalign import DATASETS, SEG, FRAME, GUARD, PRE_S, POST_S, MEL_RATE, covers, fixed_set, status, two_way_bootstrap
from .reconstruct import (build_generator, natural, mel_frames, word_spans, word_identification, reconstruction_metrics, listen, _eeg_batch)

BAND = signal.butter(2, [1, 8], btype='band', fs=MEL_RATE, output='sos')
EDGE = int(2*MEL_RATE)


def band_envelopes(mel, stats, bands=8):
    """[1+bands, T]: broadband and per-band log energy of a standardized Mel [100, T]."""
    x = natural(mel[None], stats)[0]
    return torch.cat([torch.logsumexp(x, 0)[None], torch.stack([torch.logsumexp(g, 0) for g in torch.chunk(x, bands, 0)])]).numpy().astype(np.float64)


def filtered(env):
    y = signal.sosfiltfilt(BAND, env, axis=-1)[..., EDGE:-EDGE]
    return (y-y.mean(-1, keepdims=True))/(y.std(-1, keepdims=True)+1e-9)


def modulation(env):
    """Log power of the broadband envelope between 0.5 and 16 Hz (Welch, 4 s segments)."""
    f, p = signal.welch(env-env.mean(), fs=MEL_RATE, nperseg=int(4*MEL_RATE)); keep = (f >= .5) & (f <= 16)
    return np.log(p[keep]+1e-12)


@torch.no_grad()
def stretch(gen, cache, dataset, i, device, null=False, hop=1.):
    """Code of the whole usable span of record i from overlapping 3 s windows; returns (first anchor, code [T, D])."""
    r = cache.records[dataset][i]; lo = math.ceil((r['start']+GUARD+PRE_S)/FRAME)*FRAME; hi = r['start']+r['duration']-GUARD-POST_S
    anchors = [round((lo+k*hop)/FRAME)*FRAME for k in range(int((hi-lo)/hop)+1)]
    anchors = [a for a in anchors if covers(r, a)]
    if len(anchors) < 8: return None, None
    n = int(round((anchors[-1]-anchors[0])/FRAME))+SEG; dim = gen.audio_decoder.dim
    total = torch.zeros(n, dim); weight = torch.zeros(n, 1); taper = torch.sin(math.pi*(torch.arange(SEG)+.5)/SEG)[:, None]
    for b in range(0, len(anchors), 64):
        part = [(i, a) for a in anchors[b:b+64]]; z = gen.code(*_eeg_batch(cache, dataset, part, device, null)).cpu()
        for (_, a), y in zip(part, z):
            k = int(round((a-anchors[0])/FRAME)); total[k:k+SEG] += y*taper; weight[k:k+SEG] += taper
    code = total/weight.clamp(min=1e-6)
    if gen.audio_decoder.route == 'embedding': code = torch.nn.functional.normalize(code, dim=-1)
    return anchors[0], code


def correlations(pred, true, rng, shifts=100):
    """Pearson r of the filtered envelopes per band (row 0 broadband), and the circular-shift null of the broadband r."""
    p, t = filtered(pred), filtered(true); r = (p*t).mean(-1); n = p.shape[-1]
    lags = rng.integers(int(5*MEL_RATE), max(int(5*MEL_RATE)+1, n-int(5*MEL_RATE)), shifts)
    null = np.array([(np.roll(p[0], k)*t[0]).mean() for k in lags])
    return r, null


@torch.no_grad()
def long_stretch_metrics(gen, cache, people, device, max_records=None, seed=11, decode_device='cpu'):
    """Literature-style measures over the whole test span of every record of `people`, with ceiling and nulls."""
    gen.eval(); rng = np.random.default_rng(seed); per = defaultdict(lambda: defaultdict(list)); keys = defaultdict(list)
    for dataset in DATASETS:
        rows = cache.records.get(dataset, [])
        records = [i for i, r in enumerate(rows) if r['key'] in people and r['split'] == 'test' and r.get('condition', 'clean') == 'clean'][:max_records]
        for i in records:
            a0, code = stretch(gen, cache, dataset, i, device)
            if code is None: continue
            _, empty = stretch(gen, cache, dataset, i, device, null=True); n = code.shape[0]; content = rows[i]['audio']
            true_mel = cache.mel_batch([(dataset, content, a0)], mel_frames(n))[0]; wv, en = cache.audio_batch([(dataset, content, a0)], n)
            dec = gen.audio_decoder.to(decode_device); aligner = gen.aligner.to(decode_device); at = lambda x: x.to(decode_device)
            outputs = {'eeg': dec(at(code[None]), true_mel.shape[-1])[0].cpu(), 'no_eeg': dec(at(empty[None]), true_mel.shape[-1])[0].cpu(),
                       'audio_reference': dec(dec.code(aligner, at(wv), at(en)), true_mel.shape[-1])[0].cpu()}
            gen.to(device)
            truth = band_envelopes(true_mel, cache.stats); spectrum = modulation(truth[0])
            for label, mel in outputs.items():
                env = band_envelopes(mel, cache.stats); r, null = correlations(env, truth, rng)
                per[dataset][f'{label}_broadband_r'].append(r[0]); per[dataset][f'{label}_bands_r'].append(r[1:].mean())
                per[dataset][f'{label}_null_r'].append(null.mean()); per[dataset][f'{label}_p'].append((1+np.sum(null >= r[0]))/(1+len(null)))
                per[dataset][f'{label}_modulation_r'].append(np.corrcoef(modulation(env[0]), spectrum)[0, 1])
            keys[dataset].append((rows[i]['key'], f'{content}@{rows[i]["start"]:.1f}', (len(truth[0])-2*EDGE)/MEL_RATE))
    result = {}
    for dataset, values in per.items():
        v = {k: float(np.mean(x)) for k, x in values.items()}; people_, blocks, seconds = zip(*keys[dataset])
        v.update(records=len(keys[dataset]), people=len(set(people_)), minutes=float(sum(seconds)/60),
                 eeg_above_null=int(sum(p < .05 for p in values['eeg_p'])), ceiling_share=float(v['eeg_broadband_r']/max(v['audio_reference_broadband_r'], 1e-6)))
        boot = two_way_bootstrap(np.array(values['eeg_broadband_r']), list(people_), list(blocks), rng, 2000)
        v['eeg_broadband_r_ci95'] = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]; result[dataset] = v
    found = [d for d in DATASETS if d in result]
    result['pooled'] = {k: float(np.mean([result[d][k] for d in found])) for k in result[found[0]] if isinstance(result[found[0]][k], float) and not k.endswith('ci95')} if found else {}
    return result


@torch.no_grad()
def evaluate(cfg, cache, folder, out, run, device='cpu', checkpoint='best.pt', max_records=None, sentences=5, bootstrap=2000, permutations=1000):
    """All measures for one EEG-side run: long stretches (literature-style), fine frames, words, sentences."""
    folder, out = Path(folder), Path(out); out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(folder/checkpoint, map_location=device, weights_only=False)
    gen = build_generator(cache, saved['align'], saved['audio'], device); gen.load_state_dict(saved['generator']); gen.eval(); spans = word_spans(cfg, cache)
    limit = 30 if max_records else None  # smoke checks only
    summary = {'run': run, 'checkpoint': checkpoint, 'step': saved['step'], 'minutes': saved['minutes'], 'route': saved['route'], 'smoke': cache.smoke}
    groups = (('seen_people_unseen_speech', set(cache.subjects)), ('unseen_people_unseen_speech', set(cfg.fold_subjects('P1', 'F0', 'test'))))
    for name, people in groups:
        status(folder, stage=f'evaluation v2: {name}'); items = fixed_set(cache, people, 'test')
        summary[name] = {'long': long_stretch_metrics(gen, cache, people, device, max_records),
                         'frames': reconstruction_metrics(gen, cache, items, device, limit=limit, permutations=permutations, bootstrap=bootstrap)['pooled'],
                         'words': word_identification(gen, cache, spans, items, device, limit=limit, bootstrap=bootstrap)['pooled']}
    gen = gen.to('cpu'); summary['sentences'] = {}
    for name, people in groups:
        rows, heard = listen(cfg, cache, gen, gen.audio_decoder, gen.aligner, people, out/name.split('_')[0], sentences if name.startswith('seen') else 2)
        summary['sentences'][name] = {'summary': heard, 'clips': rows}
    atomic_json(out/'evaluation.json', summary); write(summary, out); status(folder, stage='evaluation v2 done')
    return summary


def write(summary, out):
    nan = float('nan'); g = lambda d, k: d.get(k, nan)
    lines = [f"# EEG → 语音重建评估（调整后）：{summary['run']}（{'WavLM 代码' if summary['route'] == 'wavlm' else '128 维共享嵌入'}，{summary['checkpoint']}，训练 {summary['minutes']} 分钟）", '',
             '冒烟测试：只检查代码能否运行，不是实验结果。' if summary.get('smoke') else '全部是没参与训练的测试内容。',
             '**主指标（与文献可比）**：每条测试记录（一个人 × 一个内容块）的整段可用时长用 1 秒步长的 3 秒窗口重建后整体解码；从 Mel 得到宽带与 8 个频带的对数能量包络，'
             '带通 1–8 Hz 后在整段上算 Pearson r（文献中 EEG 线性逆向模型约 0.1–0.2）。空假设是预测包络相对真实包络循环错开 ≥ 5 秒；'
             '“音频侧参照”是同一解码器吃真实代码的结果，即这条链路的上限；“上限占比” = EEG 的宽带 r ÷ 音频侧参照的宽带 r。'
             '调制谱相关（0.5–16 Hz）只衡量节奏像不像语音，不要求时间对齐，所以要和无 EEG 比。', '']
    for name, title in (('seen_people_unseen_speech', '训练过的人，没听过的语音'), ('unseen_people_unseen_speech', '没见过的人，没听过的语音')):
        L = summary[name]['long']; p = L.get('pooled', {}); fr = summary[name]['frames']; w = summary[name]['words']
        lines += [f'## {title}', '', '### 整段包络（主指标）', '',
                  f"合并：宽带 r {g(p, 'eeg_broadband_r'):.3f}（循环错位空假设 {g(p, 'eeg_null_r'):.3f}，无 EEG {g(p, 'no_eeg_broadband_r'):.3f}，音频侧参照 {g(p, 'audio_reference_broadband_r'):.3f}，"
                  f"上限占比 {g(p, 'ceiling_share'):.2f}）；8 频带平均 r {g(p, 'eeg_bands_r'):.3f}（音频侧参照 {g(p, 'audio_reference_bands_r'):.3f}）", '',
                  '| 数据集 | 记录 | 人 | 分钟 | 宽带 r（95% 区间） | 显著的记录（p<0.05） | 错位空假设 | 无 EEG | 音频侧参照 | 上限占比 | 8 频带 r（参照） | 调制谱 r（无 EEG / 参照） |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for d in DATASETS:
            if d not in L: continue
            v = L[d]; ci = v['eeg_broadband_r_ci95']
            lines.append(f"| {d} | {v['records']} | {v['people']} | {v['minutes']:.0f} | {v['eeg_broadband_r']:.3f}（{ci[0]:.3f}–{ci[1]:.3f}） | {v['eeg_above_null']} / {v['records']} | "
                         f"{v['eeg_null_r']:.3f} | {v['no_eeg_broadband_r']:.3f} | {v['audio_reference_broadband_r']:.3f} | {v['ceiling_share']:.2f} | "
                         f"{v['eeg_bands_r']:.3f}（{v['audio_reference_bands_r']:.3f}） | {v['eeg_modulation_r']:.2f}（{v['no_eeg_modulation_r']:.2f} / {v['audio_reference_modulation_r']:.2f}） |")
        lines += ['', '### 细粒度与词（辅助）', '',
                  f"- 3 秒窗口动态 Mel r（约 43 ms 分辨率）：{g(fr, 'dynamic_mel_r'):.3f}（错时 {g(fr, 'dynamic_mel_r_null'):.3f}，无 EEG {g(fr, 'no_eeg_dynamic_mel_r'):.3f}）；"
                  f"3 秒窗口包络 r（未滤波）：{g(fr, 'envelope_r'):.3f}；输出离散度 {g(fr, 'spread_ratio'):.2f}",
                  f"- 认词排名百分位：{g(w, 'rank_percentile'):.3f}（区间 {w.get('rank_percentile_ci95', [nan, nan])[0]:.3f}–{w.get('rank_percentile_ci95', [nan, nan])[1]:.3f}；无 EEG {g(w, 'no_eeg_rank_percentile'):.3f}；0.5 = 偶然）", '']
        heard = summary['sentences'][name]['summary']
        if heard:
            lines += ['### 整句（Whisper）', '', '| 输出 | 词错误率 | 参考词出现在转写里的比例 |', '|---|---|---|']
            lines += [f"| {label} | {v['wer']:.2f} | {v['word_hits']:.2f} |" for label, v in heard.items()]
            lines += ['', f"音频在 `{name.split('_')[0]}/audio/`，频谱图在 `{name.split('_')[0]}/figures/`。", '']
    (Path(out)/'evaluation.md').write_text('\n'.join(lines)+'\n')
