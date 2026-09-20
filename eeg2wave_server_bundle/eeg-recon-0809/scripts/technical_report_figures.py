#!/usr/bin/env python3
"""Figures for reports/updated-results_2026-09-19.md (data example, dataset statistics, pipeline,
training curves, result bars).  Reads only cached artifacts and existing outputs; writes PNGs to
reports/figures/technical_report/.  Run from the project root with the aligned venv."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src')); sys.path.insert(0, str(ROOT / 'scripts'))
import audio_comparison as ac
import generative_recovery as gr

OUT = ROOT / 'reports/figures/technical_report'
CACHE = ROOT / 'outputs/generative_recovery/cache'
MANIFEST = ROOT / 'artifacts/aligned_speech_local_v1/manifest.csv'
CHANNELS = ['A1', 'A19', 'A23', 'B22', 'C21', 'D19', 'C10', 'D8']      # frontal, central, parietal, occipital, temporal samples


def channel_order():
    import yaml
    cfg = yaml.safe_load((ROOT / 'configs/training_data_v2.yaml').read_text())
    return cfg['sources']['ds004940']['channel_order']


def data_example():
    audio = gr.Audio(CACHE); val = gr.Role(CACHE, 'validation')
    i = [k for k, r in enumerate(val.rows) if r['trial_id'] == 'ds004940-0072a31be8bb923199e1'][0]
    row = val.rows[i]; frames = int(row['duration_frames']); duration = frames * 256 / 16000
    wave = np.asarray(audio.wave(CACHE)[row['audio_index']], dtype=np.float32)
    mel = audio.mel[row['audio_index']].numpy()
    order = channel_order(); picks = [order.index(c) for c in CHANNELS]
    eeg = np.asarray(val.eeg[i], dtype=np.float32)
    members = val.groups[row['content']]
    pooled = np.mean([np.asarray(val.eeg[j], dtype=np.float32) for j in members], 0)
    t_eeg = -0.25 + np.arange(eeg.shape[1]) / 256
    times, f0 = gr.yin_f0(wave)
    env = ac.envelope(wave); t_env = np.arange(len(env)) * 160 / 16000
    fig, axes = plt.subplots(6, 1, figsize=(13, 14.6), gridspec_kw=dict(height_ratios=[1, 1.6, 1, 2.4, 2.4, 1.4]))
    t_wave = np.arange(len(wave)) / 16000
    axes[0].plot(t_wave, wave, color='k', linewidth=.4); axes[0].set_ylabel('amplitude'); axes[0].set_xlim(-.25, 4.35)
    axes[0].set_title(f'presented sentence "I play with my best friend."  ({duration:.2f} s of speech in a fixed 4 s window, 16 kHz)', loc='left', fontsize=10)
    axes[1].imshow(mel, origin='lower', aspect='auto', cmap='magma', extent=[0, 251 * 256 / 16000, 0, 80], vmin=-7, vmax=1)
    axes[1].set_ylabel('SpeechT5 mel bin'); axes[1].set_xlim(-.25, 4.35)
    axes[1].set_title('native SpeechT5 log-mel target (80 bins x 251 frames, hop 16 ms; padding = -10)', loc='left', fontsize=10)
    ax = axes[2]; ax.plot(t_env, env / env.max(), color='k', label='RMS envelope (normalised)'); ax.set_ylabel('envelope'); ax.set_xlim(-.25, 4.35)
    twin = ax.twinx(); twin.plot(times, f0, '.', color='tab:red', markersize=3, label='F0 (YIN)'); twin.set_ylabel('F0 (Hz)', color='tab:red'); twin.set_ylim(100, 300)
    ax.set_title('energy envelope (black) and pitch contour (red)', loc='left', fontsize=10)
    for k, (ax, data, name) in enumerate(((axes[3], eeg, f'EEG of this trial ({row["subject"]}), 8 of 128 channels, normalised (median/MAD) units'),
                                         (axes[4], pooled, f'same channels averaged over all {len(members)} presentations of the sentence ({len(members)} participants): only sentence-locked activity survives averaging'))):
        scale = 4. if k == 0 else 1.5
        for c, ch in enumerate(picks):
            ax.plot(t_eeg, data[ch] - c * scale, linewidth=.6, color='tab:blue' if k == 0 else 'tab:green')
        ax.set_yticks([-c * scale for c in range(len(picks))]); ax.set_yticklabels(CHANNELS)
        ax.axvline(0, color='k', linestyle=':', linewidth=.8); ax.axvline(duration, color='k', linestyle=':', linewidth=.8)
        ax.set_xlim(-.25, 4.35); ax.set_title(name, loc='left', fontsize=10)
    # MFCC panel: 40-point DCT of the 80-bin log-mel with c0 dropped, shown at absolute time so it aligns
    # with the panels above.  The cached "relative MFCC" target is this feature with the voiced region
    # resampled to a fixed 161 frames (time-normalised), so it does not carry a fixed frame rate.
    from scipy.fft import dct
    mfcc = dct(ac.log_mel(wave, bands=80), type=2, axis=0, norm='ortho')[1:14]
    mfcc = (mfcc - mfcc.mean(1, keepdims=True)) / mfcc.std(1, keepdims=True).clip(1e-5)
    span = float(np.abs(mfcc).max())
    ax = axes[5]
    ax.imshow(mfcc, origin='lower', aspect='auto', cmap='coolwarm', vmin=-span, vmax=span,
              extent=[0, mfcc.shape[1] * 160 / 16000, 1, 13])
    ax.axvline(0, color='k', linestyle=':', linewidth=.8); ax.axvline(duration, color='k', linestyle=':', linewidth=.8)
    ax.set_xlim(-.25, 4.35); ax.set_ylabel('MFCC c1-c13')
    ax.set_title('MFCC c1-c13 (40-point DCT of the log-mel, c0 dropped, per-coefficient CMVN). '
                 'Legacy target, display only; cached form resamples the voiced region to 161 frames.', loc='left', fontsize=10)
    axes[4].set_xlabel('')
    ax.set_xlabel('seconds relative to sentence onset (dotted: onset and offset of the presented speech)')
    fig.tight_layout(); fig.savefig(OUT / 'fig1_data_example.png', dpi=110); plt.close(fig)


def dataset_statistics():
    m = pd.read_csv(MANIFEST, keep_default_na=False)
    sentences = m.drop_duplicates('content_group')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colours = {'train': 'tab:blue', 'validation': 'tab:orange', 'test': 'tab:green'}
    bins = np.linspace(1.5, 4., 26)
    axes[0].hist([sentences[sentences.role == r].stimulus_duration_seconds for r in ('train', 'validation', 'test')], bins=bins,
                 stacked=True, color=[colours[r] for r in ('train', 'validation', 'test')], label=[f'{r} ({int((sentences.role == r).sum())} sentences)' for r in ('train', 'validation', 'test')])
    axes[0].set_xlabel('sentence duration (s)'); axes[0].set_ylabel('sentences'); axes[0].legend(fontsize=8); axes[0].set_title('402 unique sentences by split', fontsize=10)
    per = m.groupby(['subject', 'role']).size().unstack(fill_value=0).loc[sorted(m.subject.unique())]
    bottom = np.zeros(len(per))
    for r in ('train', 'validation', 'test'):
        axes[1].bar(range(len(per)), per[r], bottom=bottom, color=colours[r], label=r); bottom += per[r].to_numpy()
    axes[1].set_xticks(range(len(per))); axes[1].set_xticklabels([s.replace('sub-', '') for s in per.index], fontsize=7); axes[1].set_xlabel('participant'); axes[1].set_ylabel('trials')
    axes[1].set_title('6,641 trials: every participant appears in every split', fontsize=10); axes[1].legend(fontsize=8)
    counts = m.groupby('content_group').size()
    axes[2].hist(counts, bins=np.arange(13.5, 18.5, 1), color='tab:gray'); axes[2].set_xlabel('presentations per sentence (participants who heard it)'); axes[2].set_ylabel('sentences')
    axes[2].set_title('each sentence was heard by 14-17 participants', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'fig2_dataset_statistics.png', dpi=110); plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(15, 6.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 6.2); ax.axis('off')

    def box(x, y, w, h, text, colour, fontsize=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04', facecolor=colour, edgecolor='k', linewidth=.8))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize)

    def arrow(x0, y0, x1, y1, text='', frozen=False):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', linewidth=1.2, linestyle='--' if frozen else '-'))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + .12, text, ha='center', fontsize=7.5, color='dimgray')
    # audio side (top)
    box(.2, 4.6, 1.9, 1.1, 'presented WAV\n16 kHz, 4 s window\n(64000,)', '#f2f2f2')
    box(2.9, 4.6, 2.3, 1.1, 'HuBERT-base, layer 9\n(adapted on train audio)\n(199, 768) @ 50 Hz', '#dde8f7')
    box(6.0, 4.6, 2.3, 1.1, 'AcousticDecoder\n768 -> 256 x6 -> 80\n(80, 251) @ 62.5 Hz', '#dde8f7')
    box(9.1, 4.6, 2.2, 1.1, 'SpeechT5 HiFi-GAN\n(adapted on train audio)\n(64000,)', '#dde8f7')
    box(12.2, 4.6, 2.6, 1.1, 'audio-only ceilings:\nmel -> vocoder  STOI 0.97\naudio -> teacher -> diffusion 0.89', '#f2f2f2', 7.5)
    arrow(2.1, 5.15, 2.9, 5.15); arrow(5.2, 5.15, 6.0, 5.15, 'teacher target'); arrow(8.3, 5.15, 9.1, 5.15); arrow(11.3, 5.15, 12.2, 5.15)
    # EEG side (middle)
    box(.2, 2.5, 1.9, 1.3, 'EEG trial\n128 ch x 1178 samples\n256 Hz, -0.25..4.35 s', '#fbe5d6')
    box(2.9, 2.5, 2.3, 1.3, 'recovery-v3 encoder (frozen)\nsubject mix -> spatial -> temporal\n-> 6 dilated blocks -> head\n1.08 M params', '#fbe5d6', 7.5)
    box(6.0, 2.5, 2.3, 1.3, 'conditioning (18 x 251)\n16 PCs of head output\n+ predicted duration\n+ envelope decoder (0.98 M)', '#fbe5d6', 7.5)
    box(9.1, 2.5, 2.2, 1.3, 'conditional diffusion\nv-prediction, 6 blocks\n7.25 M params, CFG w=2\n(80, 251) sample', '#e2f0d9', 7.5)
    arrow(2.1, 3.15, 2.9, 3.15); arrow(5.2, 3.15, 6.0, 3.15); arrow(8.3, 3.15, 9.1, 3.15); arrow(10.2, 3.8, 10.2, 4.6, 'mel -> vocoder')
    # training signals (bottom)
    box(2.9, .4, 2.3, 1.2, 'encoder training (v3)\nbatch CLIP on HuBERT frames\n+ mel L1 (frozen decoder)\n+ duration L1', '#fff2cc', 7.5)
    box(6.0, .4, 2.3, 1.2, 'cross-fitting\n2 encoders + 2 envelope decoders\non disjoint sentence halves\n(features never see their sentence)', '#fff2cc', 7.5)
    box(9.1, .4, 2.2, 1.2, 'diffusion training\nEEG cond 55% / teacher 30%\n/ null 15%; pooled-EEG aug.\n8000 updates, batch 32', '#fff2cc', 7.5)
    arrow(4.05, 1.6, 4.05, 2.5, frozen=True); arrow(7.15, 1.6, 7.15, 2.5, frozen=True); arrow(10.2, 1.6, 10.2, 2.5, frozen=True)
    box(12.2, 2.5, 2.6, 1.3, 'controls, same noise seed:\nzero EEG / wrong-trial EEG /\ntime-block shuffle /\npooled (17 presentations) / pooled-wrong', '#f2f2f2', 7.5)
    arrow(11.3, 3.15, 12.2, 3.15)
    ax.text(.2, 6.0, 'DS004940 EEG -> speech: audio side (top, trained on train-fold audio only), EEG side (middle), training signals (bottom)', fontsize=10)
    fig.savefig(OUT / 'fig3_pipeline.png', dpi=120, bbox_inches='tight'); plt.close(fig)


def training_curves():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    v3 = json.load(open(ROOT / 'outputs/aligned_recovery_v3/full_seed322_positional/metrics.json'))['history']
    u = [h['update'] for h in v3]
    axes[0].plot(u, [h['validation']['native_mel_mae'] for h in v3], 'o-', label='validation mel MAE', markersize=3)
    axes[0].plot(u, [h['validation']['template_mae'] for h in v3], '--', color='gray', label='median template')
    axes[0].plot(u, [h['train_probe']['native_mel_mae'] for h in v3], 'o-', color='tab:orange', markersize=3, label='train-probe mel MAE')
    axes[0].axvline(1800, color='k', linestyle=':', linewidth=.8); axes[0].set_title('v3 encoder (seed 322): mel MAE, best_passed at 1800', fontsize=9); axes[0].set_xlabel('update'); axes[0].legend(fontsize=7)
    axes[1].plot(u, [h['validation']['envelope_corr'] for h in v3], 'o-', markersize=3, label='validation envelope r')
    axes[1].plot(u, [h['validation']['envelope_template_corr'] for h in v3], '--', color='gray', label='template')
    axes[1].plot(u, [h['train_probe']['envelope_corr'] for h in v3], 'o-', color='tab:orange', markersize=3, label='train-probe envelope r')
    axes[1].plot(u, [h['validation']['zero_gain'] * 10 for h in v3], 's-', color='tab:red', markersize=3, label='zero-EEG gain x10 (mel MAE)')
    axes[1].axvline(1800, color='k', linestyle=':', linewidth=.8); axes[1].set_title('v3 encoder: envelope r and EEG gain (train >> validation = over-informative train features)', fontsize=8); axes[1].set_xlabel('update'); axes[1].legend(fontsize=7)
    for run, colour in (('compact', 'tab:orange'), ('crossfit', 'tab:green')):
        h = json.load(open(ROOT / f'outputs/generative_recovery/{run}/metrics.json'))['history']
        x = [r['update'] for r in h]
        axes[2].plot(x, [r['val_loss_eeg'] for r in h], '-', color=colour, label=f'{run}: EEG-conditioned')
        axes[2].plot(x, [r['val_loss_null'] for r in h], '--', color=colour, label=f'{run}: unconditional')
        axes[3].plot(x, [r['sampled']['correct']['envelope_corr'] for r in h], '-', color=colour, label=f'{run}: real EEG')
        axes[3].plot(x, [r['sampled']['zero']['envelope_corr'] for r in h], ':', color=colour, label=f'{run}: zero EEG')
        if 'wrong_trial' in h[-1]['sampled']:
            axes[3].plot(x, [r['sampled']['wrong_trial']['envelope_corr'] for r in h], '--', color=colour, label=f'{run}: wrong-trial EEG')
    axes[2].set_yscale('log'); axes[2].set_title('diffusion: validation denoising loss (fixed t, fixed noise)', fontsize=9); axes[2].set_xlabel('update'); axes[2].legend(fontsize=7)
    axes[3].set_title('diffusion: envelope r of sampled mel vs target (64 val trials, 25 steps, w=2)', fontsize=9); axes[3].set_xlabel('update'); axes[3].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'fig4_training_curves.png', dpi=110); plt.close(fig)


def result_bars():
    names = ['regression', 'correct', 'zero', 'wrong_trial', 'time_block_shuffle', 'pooled', 'teacher_oracle']
    labels = ['v3 regression\n(real EEG)', 'diffusion\nreal EEG', 'zero EEG', 'wrong-trial\nEEG', 'time-block\nshuffle', 'pooled EEG\n(17 pres.)', 'audio→teacher\n(ceiling)']
    runs = {'crossfit': 'cross-fitted features (headline)', 'compact': 'seed-322 features'}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    width = .38
    for k, (run, title) in enumerate(runs.items()):
        c = json.load(open(ROOT / f'outputs/generative_recovery/{run}/export_validation/comparison.json'))
        x = np.arange(len(names)) + (k - .5) * width
        stoi = [c['per_condition'][n]['stoi'] for n in names]; stoi_ci = np.array([c['per_condition'][n]['stoi_ci95'] for n in names])
        env = [c['per_condition'][n]['envelope_corr'] for n in names]; env_ci = np.array([c['per_condition'][n]['envelope_ci95'] for n in names])
        afc = [c['two_alternative_forced_choice'][n]['envelope_corr']['accuracy'] for n in names]; afc_ci = np.array([c['two_alternative_forced_choice'][n]['envelope_corr']['ci95'] for n in names])
        for ax, val, ci in ((axes[0], stoi, stoi_ci), (axes[1], env, env_ci), (axes[2], afc, afc_ci)):
            err = np.abs(ci.T - np.array(val)[None])
            ax.bar(x, val, width, yerr=err, capsize=2, label=title, color='tab:green' if run == 'crossfit' else 'tab:orange', alpha=.85)
    for ax, t in zip(axes, ('STOI vs presented sentence (240 validation trials)', 'envelope correlation vs presented sentence', '2AFC by envelope: own sentence vs duration-matched foil (chance 0.5)')):
        ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(labels, fontsize=7.5); ax.set_title(t, fontsize=9); ax.legend(fontsize=7)
    axes[2].axhline(.5, color='k', linestyle=':', linewidth=.8)
    fig.tight_layout(); fig.savefig(OUT / 'fig5_results_bars.png', dpi=110); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (data_example, dataset_statistics, pipeline, training_curves, result_bars):
        fn(); print('wrote', fn.__name__, flush=True)
    import shutil
    copies = {'fig6_mels_compact.png': ROOT / 'outputs/generative_recovery/compact/export_validation/comparison.png',
              'fig7_mels_crossfit.png': ROOT / 'outputs/generative_recovery/crossfit/export_validation/comparison.png',
              'fig8_features_stacked_example.png': ROOT / 'outputs/generative_recovery/listening/A_seed322_features/01_I_play_with_my_best_friend/features_stacked.png',
              'fig9_features_overlay_example.png': ROOT / 'outputs/generative_recovery/listening/A_seed322_features/01_I_play_with_my_best_friend/features_overlay.png',
              'fig10_v3_audio_comparison.png': ROOT / 'outputs/aligned_recovery_v3/audio_comparison/audio_comparison.png',
              'fig11_mels_pooled_control.png': ROOT / 'outputs/generative_recovery/compact/export_validation_pooledcontrol/comparison.png'}
    for name, src in copies.items():
        if src.exists():
            shutil.copyfile(src, OUT / name); print('copied', name)


if __name__ == '__main__':
    main()
