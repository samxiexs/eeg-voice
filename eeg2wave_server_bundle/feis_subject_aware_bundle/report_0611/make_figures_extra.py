"""Extra figures for the 0611 report: architecture, multi-dataset, results, roadmap."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#2f6db5", "#e08a1e", "#2e8b57", "#888888", "#c0392b", "#7d5ba6"

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight"); plt.close(fig); print("wrote", name)

def box(ax, x, y, w, h, text, fc, tc="white", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec="none"))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", color=tc, fontsize=fs, weight="bold", wrap=True)

def arrow(ax, x1, y1, x2, y2, c="#444444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.6, color=c))

# ===== ARCH 1: v3 decoupled pipeline =====
fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")
box(ax, 0.2, 1.6, 1.5, 1.0, "EEG\n14×1280", BLUE)
box(ax, 2.0, 1.6, 1.5, 1.0, "Spatial\nAdapter\n(1×1 conv)", BLUE)
box(ax, 3.8, 1.4, 1.8, 1.4, "Temporal Trunk\n(dilated conv)\n+ FiLM(subject)", PURPLE)
box(ax, 6.0, 2.7, 2.0, 0.7, "content head\n→ EnCodec latent 75×128", GREEN, fs=8)
box(ax, 6.0, 1.75, 2.0, 0.7, "contrastive head\n(InfoNCE)", ORANGE, fs=8)
box(ax, 6.0, 0.8, 2.0, 0.7, "class head\n→ 16 prompts", GREY, fs=8)
box(ax, 8.4, 2.55, 1.5, 1.0, "FROZEN\nEnCodec\ndecoder", RED, fs=9)
box(ax, 10.0, 2.7, 0.9, 0.7, "wav", "#333333", fs=9)
for (x1,y1,x2,y2) in [(1.7,2.1,2.0,2.1),(3.5,2.1,3.8,2.1),(5.6,2.3,6.0,3.05),(5.6,2.1,6.0,2.1),(5.6,1.9,6.0,1.15),(8.0,3.05,8.4,3.05),(9.9,3.05,10.0,3.05)]:
    arrow(ax, x1, y1, x2, y2)
ax.text(5.5, 0.2, "Objective = InfoNCE + latent cosine/MSE + class CE  (all anti-collapse; NO raw-waveform regression)",
        ha="center", fontsize=9, color="#333333")
ax.text(5.5, 4.0, "v3 decoupled: recognition (EEG -> subject-specific speech latent) + frozen vocoder synthesis",
        ha="center", fontsize=11, weight="bold")
save(fig, "arch_v3_pipeline.png")

# ===== RESULTS consolidation =====
fig, ax = plt.subplots(figsize=(9.5, 4.0))
groups = ["A. waveform\nbaseline (G)", "C. codec align\nretrieval (G)", "v3 speaking\nlabel_top1", "v3 speaking\ntemplate_top1", "v3 speaking\ntemplate_top5"]
vals = [0.003, 0.0031, 0.078, 0.0875, 0.331]
chance = [0.0625, 0.0625, 0.0625, 1/320, 5/320]
x = np.arange(len(groups))
bars = ax.bar(x, vals, 0.55, color=[RED, RED, GREY, BLUE, BLUE])
for i, c in enumerate(chance):
    ax.hlines(c, i-0.28, i+0.28, color="k", lw=1.5, ls=":", label="chance" if i == 0 else None)
for i, v in enumerate(vals):
    ax.text(i, v+0.008, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8); ax.set_ylabel("accuracy")
ax.set_title("Results: collapse -> alignment -> v3 (dotted = per-metric chance)"); ax.legend(fontsize=8)
ax.text(0.5, 0.33, "A/C ~ chance\n(collapse / weak)", fontsize=8, color=RED)
ax.text(3.0, 0.30, "template inflated\nby subject ID", fontsize=8, color=BLUE)
save(fig, "results_consolidation.png")

# ===== ROADMAP timeline =====
fig, ax = plt.subplots(figsize=(11, 3.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.2); ax.axis("off")
phases = [
    ("P1 demo: A/B/C routes", "DONE", GREEN),
    ("P2 v3 diagnosis: subject-ID confound", "DONE", GREEN),
    ("P3 factored: content x speaker + disentangle", "BUILT", GREEN),
    ("P4 train factored -> decisive numbers", "NEXT", ORANGE),
    ("P5 auditory percept/imagery dataset", "PLANNED", GREY),
    ("P6 paper eval+ablation", "PLANNED", GREY),
]
n = len(phases); ax.plot([0.5, 10.5], [1.8, 1.8], color="#bbb", lw=2, zorder=0)
for i, (name, status, c) in enumerate(phases):
    x = 0.8 + i*1.9
    ax.scatter([x], [1.8], s=240, color=c, zorder=2, edgecolors="white")
    ax.text(x, 2.35, name, ha="center", fontsize=8, weight="bold")
    ax.text(x, 1.35, status, ha="center", fontsize=8, color=c)
ax.text(5.5, 2.95, "Project Roadmap", ha="center", fontsize=12, weight="bold")
save(fig, "roadmap_timeline.png")

print("DONE extra figures")
