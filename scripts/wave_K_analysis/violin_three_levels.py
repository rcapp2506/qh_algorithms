#!/usr/bin/env python3
"""violin_three_levels.py -- Generator for Fig. violin_three_models.png (Ch. 3).

Reproduces the three-level violin display of per-seed final-epoch validation
accuracy (Level 1 = CCNN-big, Level 2 = CCNN-small matched, Level 3 = hybrid
QCNN noiseless) with:
  * black points: individual per-seed accuracies (R = 10)
  * red squares with vertical red bars: across-seed mean with bootstrap 95% CI
    on the mean (10,000 resamples, fixed RNG seed for reproducibility)
  * trainable-parameter counts annotated below each level
  * paired Wilcoxon exact two-sided p-values annotated on brackets, taken from
    the same paired_wilcoxon used by aggregate_cap3_stats.py (single source of
    truth for the tie handling: differences snapped to the 1/N_val grid).

Output: violin_three_models.png next to this script (copy into
PhDThesis/chapters/qa_figures/ to update the manuscript figure).
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate_cap3_stats import load, arr, paired_wilcoxon, CSV

HERE = Path(__file__).resolve().parent
OUT = HERE / "violin_three_models.png"

RNG = np.random.default_rng(42)
N_BOOT = 10_000

LEVELS = [
    ("ccnn_big", "Level 1\nCCNN-big", "85,617,041 params"),
    ("ccnn_small", "Level 2\nCCNN-small (matched)", "463,898 params"),
    ("qcnn_hybrid", "Level 3\nhybrid QCNN (sim)", "463,574 params"),
]


def boot_ci_mean(x, n_boot=N_BOOT, alpha=0.05):
    means = np.array([RNG.choice(x, size=x.size, replace=True).mean()
                      for _ in range(n_boot)])
    return np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)


def main():
    data = load(CSV)
    series, seeds = {}, None
    for model, _, _ in LEVELS:
        series[model], s = arr(data, "main_9q", model)
        assert seeds is None or s == seeds, "seed schedules differ"
        seeds = s

    p = {}
    for a, b in [("qcnn_hybrid", "ccnn_small"),
                 ("ccnn_big", "ccnn_small"),
                 ("qcnn_hybrid", "ccnn_big")]:
        p[(a, b)] = paired_wilcoxon(series[a], series[b])["p_exact"]

    fig, ax = plt.subplots(figsize=(11.13, 6.54), dpi=150)
    xs = np.arange(1, len(LEVELS) + 1)
    vals = [series[m] * 100 for m, _, _ in LEVELS]

    parts = ax.violinplot(vals, positions=xs, widths=0.7,
                          showmeans=False, showextrema=False)
    for body, color in zip(parts["bodies"],
                           ["tab:blue", "tab:orange", "tab:green"]):
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor("none")

    for x, v in zip(xs, vals):
        jitter = RNG.uniform(-0.06, 0.06, size=v.size)
        ax.plot(x + jitter, v, "o", color="black", ms=4.5, zorder=3)
        lo, hi = boot_ci_mean(v)
        ax.errorbar(x, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]],
                    fmt="s", color="red", ms=8, capsize=5,
                    elinewidth=2, zorder=4)

    def bracket(x1, x2, y, text):
        ax.plot([x1, x1, x2, x2], [y - 0.15, y, y, y - 0.15],
                lw=1.1, color="0.25")
        ax.text((x1 + x2) / 2, y + 0.08, text, ha="center",
                va="bottom", fontsize=10.5, color="0.15")

    top = max(v.max() for v in vals)
    bracket(2, 3, top + 1.0, f"$p = {p[('qcnn_hybrid','ccnn_small')]:.3f}$")
    bracket(1, 2, top + 2.4, f"$p = {p[('ccnn_big','ccnn_small')]:.3f}$")
    bracket(1, 3, top + 3.8, f"$p = {p[('qcnn_hybrid','ccnn_big')]:.3f}$")

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl, _ in LEVELS], fontsize=11)
    for x, (_, _, params) in zip(xs, LEVELS):
        ax.annotate(params, xy=(x, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -46), textcoords="offset points",
                    ha="center", fontsize=9.5, color="0.35")
    ax.set_ylabel("Final-epoch validation accuracy [%]", fontsize=12)
    ax.set_ylim(min(v.min() for v in vals) - 1.5, top + 5.4)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    print("annotated p-values:",
          {f"{a} vs {b}": round(v, 4) for (a, b), v in p.items()})


if __name__ == "__main__":
    main()
