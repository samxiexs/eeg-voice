"""Factored-model figures for the 0611 report (English labels to avoid CJK tofu)."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#2f6db5", "#e08a1e", "#2e8b57", "#888888", "#c0392b", "#7d5ba6"

def save(fig, name): fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight"); plt.close(fig); print("wrote", name)
def box(ax, x, y, w, h, t, fc, tc="white", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="none"))
    ax.text(x+w/2, y+h/2, t, ha="center", va="center", color=tc, fontsize=fs, weight="bold")
def arr(ax, x1, y1, x2, y2, c="#444", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.6, color=c, linestyle=ls))

# ===== Factored architecture =====
fig, ax = plt.subplots(figsize=(11.5, 4.6)); ax.set_xlim(0, 11.5); ax.set_ylim(0, 4.6); ax.axis("off")
box(ax, 0.2, 1.8, 1.4, 1.0, "EEG\n14x1280", BLUE)
box(ax, 1.9, 1.8, 1.5, 1.0, "Encoder\n+FiLM(stage)", PURPLE)
box(ax, 3.8, 2.55, 1.9, 0.7, "content head\n(WHAT, from EEG)", GREEN, fs=8)
box(ax, 3.8, 0.5, 1.9, 0.7, "subject id ->\nspeaker emb (WHO)", ORANGE, fs=8)
box(ax, 3.8, 1.55, 1.9, 0.7, "adversary (GRL)\nremove identity", RED, fs=7.5)
box(ax, 6.1, 1.7, 1.7, 1.0, "GENERATOR\ncontent x speaker", GREY, fs=8)
box(ax, 8.1, 1.7, 1.5, 1.0, "FROZEN\nEnCodec\ndecoder", RED, fs=8)
box(ax, 9.8, 1.95, 0.9, 0.5, "wav", "#333", fs=8)
arr(ax, 1.6, 2.3, 1.9, 2.3); arr(ax, 3.4, 2.4, 3.8, 2.9); arr(ax, 3.4, 2.2, 3.8, 1.9)
arr(ax, 5.7, 2.9, 6.1, 2.4); arr(ax, 5.7, 0.85, 6.1, 2.0, ORANGE)
arr(ax, 7.8, 2.2, 8.1, 2.2); arr(ax, 9.6, 2.2, 9.8, 2.2)
ax.text(5.7, 1.9, "GRL", fontsize=7, color=RED)
ax.text(5.75, 4.35, "Factored: content (decoded from EEG) x speaker (given id) -> generate EnCodec latent -> frozen vocoder",
        ha="center", fontsize=11, weight="bold")
ax.text(5.75, 0.12, "supervised contrastive (same-label +) | content CE | content->content-prototype | speaker->voice-prototype | adversary(content !-> subject)",
        ha="center", fontsize=7.5, color="#333")
save(fig, "arch_factored.png")

# ===== Grid + hold-out-cell =====
fig, ax = plt.subplots(figsize=(9.5, 5.2))
nS, nL = 20, 16
grid = np.zeros((nS, nL))
for i in range(nS):
    grid[i, (i) % nL] = 1     # Latin-square held-out cells
ax.imshow(grid, cmap="Oranges", vmin=0, vmax=1.6, aspect="auto")
ax.set_xticks(range(nL)); ax.set_xticklabels(
    ["f","fleece","goose","k","m","n","ng","p","s","sh","t","thought","trap","v","z","zh"], rotation=90, fontsize=7)
ax.set_yticks(range(nS)); ax.set_yticklabels([f"sub{ i+1:02d}" for i in range(nS)], fontsize=6)
ax.set_xlabel("content (16 prompts)"); ax.set_ylabel("speaker (20 subjects)")
ax.set_title("FEIS grid = content x speaker (each cell = 1 target wav, ~10 EEG reps x 2 stages)\n"
             "orange = HELD-OUT cells (unseen subject x content) -> test 'beyond classification'")
save(fig, "grid_holdout.png")
print("DONE factored figures")
