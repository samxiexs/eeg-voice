"""Generate all figures for the 0611 progress report.

Run from the bundle root:  python report_0611/make_figures.py
Outputs PNGs into report_0611/assets/.
"""
from __future__ import annotations
import os, csv, glob, json
from collections import Counter, defaultdict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
FEIS = os.path.join(BUNDLE, "data", "feis")
KARA = os.path.join(BUNDLE, "data", "karaone")
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREY, RED = "#2f6db5", "#e08a1e", "#888888", "#c0392b"

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", name)

def logmel(wav, sr):
    m = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
    return librosa.power_to_db(m + 1e-6)

# ---------- load per-subject trial counts / labels ----------
def subject_stats(root):
    subs = sorted(glob.glob(os.path.join(root, "subjects", "*.npz")))
    counts, labels_all = {}, []
    for s in subs:
        d = np.load(s, allow_pickle=True)
        name = os.path.basename(s)[:-4]
        counts[name] = len(d["labels"]); labels_all += d["labels"].tolist()
    return counts, labels_all

feis_counts, feis_labels = subject_stats(FEIS)
kara_counts, kara_labels = subject_stats(KARA)

# ===== FEIS F1: trials per subject =====
fig, ax = plt.subplots(figsize=(8, 3))
ks = list(feis_counts); ax.bar(ks, [feis_counts[k] for k in ks], color=BLUE)
ax.axhline(160, ls="--", c=GREY, lw=1); ax.set_title("FEIS: trials per subject (21 subjects)")
ax.set_ylabel("trials"); ax.tick_params(axis="x", rotation=90, labelsize=7)
save(fig, "feis_trials_per_subject.png")

# ===== FEIS F2: label distribution (subject 01) =====
d01 = np.load(os.path.join(FEIS, "subjects", "01.npz"), allow_pickle=True)
c = Counter(d01["labels"].tolist()); ks = sorted(c)
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(ks, [c[k] for k in ks], color=BLUE); ax.set_title("FEIS subject 01: 16 prompts x 10 trials")
ax.set_ylabel("count"); ax.tick_params(axis="x", rotation=45, labelsize=8)
save(fig, "feis_label_counts.png")

# ===== FEIS F3: thinking EEG example (14 ch) =====
eeg = d01["stage__thinking"][0]  # [14,1280]
ch = d01["channel_names"].tolist()
t = np.arange(eeg.shape[1]) / 256.0
fig, ax = plt.subplots(figsize=(8, 4.5))
off = 0; step = np.nanmax(np.abs(eeg)) * 1.1
for i in range(eeg.shape[0]):
    ax.plot(t, eeg[i] + off, lw=0.6, color=BLUE); off += step
ax.set_yticks([step*i for i in range(14)]); ax.set_yticklabels(ch, fontsize=7)
ax.set_xlabel("time (s)"); ax.set_title("FEIS thinking EEG — 14 ch, 5 s @256Hz (subj01 trial0)")
save(fig, "feis_thinking_eeg_example.png")

# ===== FEIS F4: canonical audio durations =====
durs = []
for w in glob.glob(os.path.join(FEIS, "audio", "*", "*.wav")):
    x, sr = sf.read(w); durs.append(len(x)/sr)
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(durs, bins=20, color=BLUE); ax.set_title(f"FEIS audio durations (n={len(durs)} canonical wavs)")
ax.set_xlabel("seconds"); ax.set_ylabel("count")
save(fig, "feis_audio_durations.png")

# ===== FEIS F5: subject-specific targets — same prompt 'f', different subjects =====
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, sid in zip(axes, ["01", "02", "03"]):
    p = os.path.join(FEIS, "audio", sid, "f.wav")
    x, sr = sf.read(p); ax.imshow(logmel(x.astype("float32"), sr), origin="lower", aspect="auto", cmap="magma")
    ax.set_title(f"subject {sid} — /f/"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("FEIS targets are SUBJECT-SPECIFIC: same prompt, different recordings", y=1.04)
save(fig, "feis_subject_specific_mel.png")

# ===== KaraOne K1: trials per subject =====
ks = list(kara_counts)
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(ks, [kara_counts[k] for k in ks], color=ORANGE)
ax.set_title("KaraOne: trials per subject (14 subjects)"); ax.set_ylabel("trials")
ax.tick_params(axis="x", rotation=45, labelsize=7)
save(fig, "karaone_trials_per_subject.png")

# ===== KaraOne K2: label distribution =====
c = Counter(kara_labels); ks = sorted(c)
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(ks, [c[k] for k in ks], color=ORANGE); ax.set_title("KaraOne: 11 prompts (pooled over 14 subjects)")
ax.set_ylabel("count"); ax.tick_params(axis="x", rotation=45, labelsize=8)
save(fig, "karaone_label_counts.png")

# ===== KaraOne K3: thinking EEG example (62 ch heatmap) =====
dk = np.load(os.path.join(KARA, "subjects", "MM05.npz"), allow_pickle=True)
eeg = dk["stage__thinking"][0]  # [62, ~1272]
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(eeg, aspect="auto", cmap="RdBu_r", vmin=-np.nanstd(eeg)*3, vmax=np.nanstd(eeg)*3)
ax.set_title("KaraOne thinking EEG — 62 ch heatmap (MM05 trial0)")
ax.set_xlabel("time samples (256Hz)"); ax.set_ylabel("channel")
fig.colorbar(im, ax=ax, fraction=0.025)
save(fig, "karaone_thinking_eeg_example.png")

# ===== KaraOne K4: audio durations (variable, trial-sync) =====
durs = []
for w in glob.glob(os.path.join(KARA, "audio", "*", "*.wav")):
    x, sr = sf.read(w); durs.append(len(x)/sr)
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(durs, bins=30, color=ORANGE); ax.set_title(f"KaraOne audio durations (n={len(durs)} trial-sync wavs)")
ax.set_xlabel("seconds"); ax.set_ylabel("count")
save(fig, "karaone_audio_durations.png")

# ===== KaraOne K5: same subject+label different trials -> different audio =====
rows = [r for r in csv.DictReader(open(os.path.join(KARA, "segments.csv")))
        if r["segment_stage"] == "thinking" and r["subject_id"] == "MM05"]
by = defaultdict(list)
for r in rows: by[r["label"]].append(r["audio_path"])
lbl = max(by, key=lambda k: len(by[k])); paths = by[lbl][:3]
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, rel in zip(axes, paths):
    x, sr = sf.read(os.path.join(KARA, rel))
    ax.imshow(logmel(x.astype("float32"), sr), origin="lower", aspect="auto", cmap="magma")
    ax.set_title(os.path.basename(rel)); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"KaraOne is TRIAL-SPECIFIC: MM05 '{lbl}', 3 different trials = 3 different recordings", y=1.04)
save(fig, "karaone_trial_specific_mel.png")

# ===== Comparison =====
metrics = ["channels", "subjects", "distinct target wavs (/100)", "audio len var"]
feis_v = [14, 21, 336/100, 0.0]
kara_v = [62, 14, 1913/100, 1.0]
x = np.arange(len(metrics)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(x-w/2, feis_v, w, label="FEIS", color=BLUE)
ax.bar(x+w/2, kara_v, w, label="KaraOne", color=ORANGE)
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=8); ax.legend()
ax.set_title("FEIS vs KaraOne (audio len var: 0=fixed canonical, 1=variable trial-sync)")
save(fig, "feis_vs_karaone_compare.png")

# ===== R1: old waveform baseline mode collapse =====
nt = {"m":112,"fleece":93,"n":55,"v":50,"zh":8,"(other)":2}
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.bar(list(nt), list(nt.values()), color=RED)
ax.set_title("Old waveform-regression: predictions collapse (G, 320 test trials)")
ax.set_ylabel("# trials predicted as")
save(fig, "collapse_pred_histogram.png")

# ===== R2: speaking teacher metrics with chance & subject-ID theoretical =====
names = ["template_top1", "template_top5", "label_top1", "label_top5"]
vals = [0.0875, 0.331, 0.078, 0.3125]
subj_theory = [0.0625, 0.3125, None, None]  # "got subject, random prompt"
chance = [1/320, 5/320, 1/16, 5/16]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(x, vals, 0.5, color=BLUE, label="observed")
for i, c in enumerate(chance):
    ax.hlines(c, i-0.25, i+0.25, color=GREY, lw=2, label="chance" if i == 0 else None)
for i, s in enumerate(subj_theory):
    if s is not None:
        ax.hlines(s, i-0.25, i+0.25, color=RED, lw=2, ls="--",
                  label="subject-ID only (random prompt)" if i == 0 else None)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_ylabel("accuracy")
ax.set_title("Speaking teacher: scores ~ subject identity, NOT prompt content"); ax.legend(fontsize=7)
save(fig, "speaking_metrics_bars.png")

# ===== R3: training curves (from logs) =====
sp_clsacc = [.058,.053,.057,.054,.051,.051,.052,.051,.052,.043,.051,.052,.051,.056,.057,.059,.068,.066,.078,.079,.084,.088,.091,.097,.100,.106,.098,.108,.111,.108]
sp_valcos = [.415,.546,.581,.587,.587,.599,.599,.607,.604,.605,.611,.614,.611,.617,.617,.612,.617,.618,.615,.612,.616,.619,.617,.617,.614,.617,.616,.616,.616,.616]
th_clsacc = [.044,.052,.048,.048,.046,.053,.051,.051,.054,.051,.056,.058,.056,.063,.065,.072,.082,.082,.085,.106,.112,.119,.121,.145,.151,.174,.188,.205,.208,.235,.255,.283]
th_valcos = [.616,.612,.618,.616,.616,.613,.617,.613,.618,.615,.614,.614,.614,.605,.605,.603,.601,.598,.601,.600,.602,.595,.593,.597,.590,.594,.598,.596,.616,.602,.597,.602]
th_valcls = [.059,.066,.056,.059,.062,.056,.069,.066,.066,.062,.072,.069,.062,.072,.069,.053,.069,.062,.084,.059,.069,.056,.059,.056,.053,.081,.072,.066,.078,.084,.097,.062]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
ax.plot(sp_valcos, color=BLUE, label="val latent cos"); ax.plot(sp_clsacc, color=ORANGE, label="train cls_acc")
ax.set_title("Speaking teacher (30 ep): healthy"); ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.set_ylim(0, 0.7)
ax = axes[1]
ax.plot(th_clsacc, color=ORANGE, label="train cls_acc (↑ overfit)")
ax.plot(th_valcls, color=RED, label="val cls_acc (flat)")
ax.plot(th_valcos, color=BLUE, label="val latent cos (flat)")
ax.set_title("Thinking main (32 ep): overfitting"); ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.set_ylim(0, 0.7)
save(fig, "training_curves.png")

# ===== R4: subject vs content decomposition =====
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.bar(["template_top1"], [1/320], color=GREY, label="pure chance (1/320)")
ax.bar(["template_top1"], [0.0625-1/320], bottom=[1/320], color=RED, label="subject identity")
ax.bar(["template_top1"], [0.0875-0.0625], bottom=[0.0625], color=BLUE, label="actual prompt content")
ax.axhline(0.0875, ls="--", c="k", lw=1)
ax.set_ylabel("template_top1 = 0.0875"); ax.set_title("Where the score comes from")
ax.legend(fontsize=8)
save(fig, "subject_vs_content_decomposition.png")

print("DONE — all figures in", OUT)
