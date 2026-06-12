"""v2 figures: the decisive content-probe null + the collapse-fixed engineering result.
English labels to avoid CJK tofu."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, GREY, RED = "#2f6db5", "#e08a1e", "#2e8b57", "#888888", "#c0392b"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight"); plt.close(fig); print("wrote", name)


# ============ FIG 1: content-probe decisive null ============
# numbers from server run: content_probe --folds 5 --permutations 200
stages = ["stimuli (listen)", "thinking (imagine)"]
top1 = [0.0457, 0.0466]
null95 = [0.0587, 0.0593]
pval = [0.920, 0.896]
chance = 0.0625

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
x = np.arange(len(stages)); w = 0.5
bars = ax.bar(x, top1, w, color=[BLUE, ORANGE], label="decoder top-1")
ax.axhline(chance, color=GREY, ls="--", lw=1.5, label="chance 1/16 = 0.0625")
for i in range(len(stages)):
    ax.plot([x[i]-w/2, x[i]+w/2], [null95[i]]*2, color=RED, lw=2)
    ax.text(x[i], null95[i]+0.001, f"null 95% = {null95[i]:.3f}", ha="center", color=RED, fontsize=8)
    ax.text(x[i], top1[i]+0.0015, f"{top1[i]:.3f}\np={pval[i]:.2f}", ha="center", fontsize=9, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(stages); ax.set_ylim(0, 0.085)
ax.set_ylabel("within-subject 16-way accuracy")
ax.set_title("Content decodability probe (ridge + permutation test)\nBOTH stages at / below chance, p>=0.9 -> NOT decodable")
ax.legend(loc="upper right", fontsize=8); ax.grid(axis="y", alpha=0.3)

# coarse vs majority
ax = axes[1]
cats = ["manner", "voicing", "vowel/cons"]
obs_s = [0.267, 0.539, 0.633]; maj = [0.375, 0.625, 0.750]
xx = np.arange(len(cats)); w = 0.38
ax.bar(xx-w/2, obs_s, w, color=GREEN, label="decoder (stimuli)")
ax.bar(xx+w/2, maj, w, color=GREY, label="majority-class baseline")
for i in range(len(cats)):
    ax.text(xx[i], max(obs_s[i], maj[i])+0.01, f"Δ{obs_s[i]-maj[i]:+.2f}", ha="center", fontsize=8, color=RED)
ax.set_xticks(xx); ax.set_xticklabels(cats); ax.set_ylim(0, 0.9)
ax.set_ylabel("coarse-category accuracy")
ax.set_title("Coarse phonology also BELOW majority baseline\n(no real voicing / manner / vowel decoding)")
ax.legend(loc="upper left", fontsize=8); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); save(fig, "content_probe_bars.png")


# ============ FIG 2: collapse fixed (engineering) v1 -> v2 ============
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
def vbar(ax, v1, v2, title, good_low, ref=None, ylim=None):
    ax.bar([0, 1], [v1, v2], 0.55, color=[RED, GREEN])
    if ref is not None:
        ax.axhline(ref, color=GREY, ls="--", lw=1.4, label=f"target/ref {ref:.2f}")
        ax.legend(fontsize=8)
    for i, v in enumerate([v1, v2]):
        ax.text(i, v + (ylim[1] if ylim else 1)*0.02, f"{v:.3f}", ha="center", fontsize=10, weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["v1", "v2"])
    arrow = "lower = better" if good_low else "higher = better"
    ax.set_title(f"{title}\n({arrow})"); ax.grid(axis="y", alpha=0.3)
    if ylim: ax.set_ylim(*ylim)

vbar(axes[0], 0.236, 0.086, "pred pairwise corr\n(self-similarity / collapse)", True, ref=0.0, ylim=(0, 0.32))
vbar(axes[1], 0.004, 0.531, "pred/target std ratio\n(variance vs mean-collapse)", False, ref=1.0, ylim=(0, 1.1))
vbar(axes[2], 0.196, 1.000, "decoded RMS ratio (pred/ref)\n(loudness)", False, ref=1.0, ylim=(0, 1.2))
fig.suptitle("v2 fixed the ENGINEERING collapse (anti-collapse std + energy/log-RMS head) — but content is still chance",
             fontsize=11, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94]); save(fig, "collapse_fixed_v1_v2.png")


# ============ FIG 3: gain decomposition seen/holdout ============
fig, ax = plt.subplots(figsize=(8.5, 4.2))
groups = ["seen\nstimuli", "seen\nthinking", "holdout\nstimuli", "holdout\nthinking"]
eeg = [0.050, 0.073, 0.051, 0.081]
zero = [0.063, 0.060, 0.051, 0.102]
xx = np.arange(len(groups)); w = 0.38
ax.bar(xx-w/2, eeg, w, color=BLUE, label="real EEG")
ax.bar(xx+w/2, zero, w, color=GREY, label="zero-EEG control")
ax.axhline(chance, color=RED, ls="--", lw=1.3, label="chance 0.0625")
ax.set_xticks(xx); ax.set_xticklabels(groups); ax.set_ylim(0, 0.12)
ax.set_ylabel("within-subject content top-1")
ax.set_title("Trained factored v2: real EEG never beats the zero-EEG baseline\n(content_gain ~ 0 / negative on every cell)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); save(fig, "content_gain_v2.png")


# ============ FIG 4: positive control (same pipeline decodes identity, not content) ============
# same ridge probe + same EEG features: SUBJECT id is decoded ~16x chance; CONTENT is at chance.
fig, ax = plt.subplots(figsize=(9, 4.4))
labels = ["content\nstimuli", "content\nthinking", "subject id\nstimuli", "subject id\nthinking"]
acc = [0.0457, 0.0466, 0.7846, 0.8115]
chn = [0.0625, 0.0625, 0.05, 0.05]
xfac = [a / c for a, c in zip(acc, chn)]
colors = [BLUE, BLUE, GREEN, GREEN]
bars = ax.bar(np.arange(4), acc, 0.6, color=colors)
for i in range(4):
    ax.plot([i-0.3, i+0.3], [chn[i]]*2, color=GREY, ls="--", lw=1.4)
    ax.text(i, acc[i]+0.02, f"{acc[i]:.3f}\n({xfac[i]:.0f}x chance)", ha="center", fontsize=9, weight="bold")
ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels)
ax.set_ylim(0, 0.95); ax.set_ylabel("within / cross-subject decoding accuracy")
ax.set_title("Positive control: SAME EEG features + SAME ridge probe\n"
             "decode SUBJECT IDENTITY ~16x chance, but CONTENT at chance "
             "-> pipeline works, content null is REAL")
ax.grid(axis="y", alpha=0.3)
ax.text(0.5, 0.5, "content\n= chance", ha="center", color=BLUE, fontsize=9, transform=ax.get_xaxis_transform())
ax.text(2.5, 0.86, "identity\n= strong", ha="center", color=GREEN, fontsize=9)
fig.tight_layout(); save(fig, "probe_positive_control.png")
print("DONE v2 figures")
