#!/usr/bin/env python3
"""Draws fig_measurements.pdf (the results figure of the paper) from the two result files.

Usage: python figures/make_figures.py   (run from the repository root)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
first = [json.loads(l) for l in (ROOT / "results/results.jsonl").read_text().splitlines() if l.strip()]
follow = [json.loads(l) for l in (ROOT / "results/results_followup.jsonl").read_text().splitlines() if l.strip()]

MODELS = [("llama-7b", "LLaMA 7B", "#b5532f", "o"), ("mistral-7b", "Mistral 7B", "#2f6f9f", "s"),
          ("qwen2.5-7b", "Qwen2.5-7B", "#3f8f5f", "^")]
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.titlesize": 7.5,
                     "legend.frameon": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.0), gridspec_kw=dict(width_ratios=[1, 1, 1.75]))
ticks = [1024, 4096, 16384, 65536]
ticklabels = ["1K", "4K", "16K", "64K"]

# (a) KV-cache size, first run
for key, label, color, marker in MODELS:
    rows = sorted((r for r in first if r["exp"] == "e2" and r["model"] == key and r["status"] == "ok"),
                  key=lambda r: r["ctx"])
    ax[0].plot([r["ctx"] for r in rows], [r["kv_cache_mib"] / 1024 for r in rows], color=color, marker=marker,
               ms=3, lw=1, label=label)
ax[0].plot([32768], [16], marker="x", color="#b5532f", ms=5, mew=1.2, ls="none")
ax[0].annotate("OOM", (32768, 16), textcoords="offset points", xytext=(-22, -3), color="#b5532f", fontsize=6.5)
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log", base=2)
ax[0].set_xticks(ticks, ticklabels); ax[0].set_yticks([1 / 16, 1 / 4, 1, 4, 16], ["1/16", "1/4", "1", "4", "16"])
ax[0].set_xlabel("Context length (tokens)"); ax[0].set_ylabel("KV-cache size (GiB)"); ax[0].set_title("(a)")
ax[0].legend(loc="lower right", fontsize=6, handlelength=1.4, borderaxespad=0.2)

# (b) decode latency (solid) and cache-concatenation cost alone (dashed), follow-up run
for key, label, color, marker in MODELS:
    rows = sorted((r for r in follow if r["exp"] == "latency" and r["model"] == key and r["status"] == "ok"),
                  key=lambda r: r["ctx"])
    x = [r["ctx"] for r in rows]
    ax[1].plot(x, [r["decode_ms_median"] for r in rows], color=color, marker=marker, ms=3, lw=1)
    ax[1].plot(x, [r["concat_ms"] for r in rows], color=color, lw=0.9, ls="--")
ax[1].plot([], [], color="0.3", lw=1, label="decode step"); ax[1].plot([], [], color="0.3", lw=0.9, ls="--",
                                                                      label="cache copy alone")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(ticks, ticklabels); ax[1].set_ylim(0, 200)
ax[1].set_xlabel("Context length (tokens)"); ax[1].set_ylabel("Time per token (ms)"); ax[1].set_title("(b)")
ax[1].legend(loc="upper left", fontsize=6, handlelength=1.6, borderaxespad=0.2)

# (c) Mistral distance sweep, window on (8K and 16K) and window off (8K)
rng = np.random.default_rng(0)
lanes = [("default", 8192, "window on, 8K", 2), ("default", 16384, "window on, 16K", 1),
         ("nowindow", 8192, "window off, 8K", 0)]
for variant, ctx, label, y in lanes:
    rows = [r for r in follow if r["exp"] == "sweep" and r["variant"] == variant and r["ctx"] == ctx]
    for ok, marker, color, size in ((True, "o", "#2f6f9f", 9), (False, "x", "#c0392b", 11)):
        sel = [r for r in rows if bool(r["correct"]) == ok]
        ax[2].scatter([r["dist_second"] for r in sel], y + rng.uniform(-0.22, 0.22, len(sel)), marker=marker,
                      s=size, color=color, linewidths=0.8, facecolors="none" if ok else color)
ax[2].axvline(4096, color="0.35", lw=0.7, ls=":")
ax[2].annotate("W = 4,096", (4096, 2.66), textcoords="offset points", xytext=(4, 0), fontsize=6, color="0.25")
ax[2].scatter([], [], marker="o", s=9, color="#2f6f9f", facecolors="none", linewidths=0.8, label="retrieved")
ax[2].scatter([], [], marker="x", s=11, color="#c0392b", linewidths=0.8, label="missed")
ax[2].legend(loc="lower left", fontsize=6, ncol=2, borderaxespad=0.1, handletextpad=0.2, columnspacing=0.8, bbox_to_anchor=(0.0, -0.01))
ax[2].set_yticks([y for *_, y in lanes], [l for _, _, l, _ in lanes]); ax[2].set_ylim(-0.85, 2.95)
ax[2].set_xlim(3560, 4640); ax[2].set_xticks([3600, 3800, 4000, 4096, 4200, 4400, 4600], ["3,600", "3,800", "4,000", "", "4,200", "4,400", "4,600"])
ax[2].set_xlabel("Key-to-end distance (tokens)"); ax[2].set_title("(c)")

fig.tight_layout(pad=0.4, w_pad=0.9)
out = ROOT / "paper/fig_measurements.pdf"
fig.savefig(out)
print("wrote", out)
