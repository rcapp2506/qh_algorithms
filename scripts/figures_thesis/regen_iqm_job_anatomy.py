"""Ricostruzione di chapters/qa_figures/iqm_job_anatomy.png con il fix
proposto da Roberto: le annotation 'Credit cost = ...' devono essere
INTERNE alla barra IQM billing window, non flottanti sulla destra (dove
sforavano il grafico e si sovrapponevano al titolo del subplot
successivo).

I numeri sono presi dalla caption della Fig. 3.17 del manoscritto
(sec:cost_model, Phase E):
- Job 28 circuits: wall 12.10s (compile 1.24s + queue 4.39s + exec 6.16s
  + post-proc ~0.31s), billing window 7.41s (= execution 6.16s +
  init/teardown 1.25s), credit cost = ceil(7.41) * 0.75 = 8 * 0.75 = 6.00
- Job 273 circuits: wall 70.04s (compile 6.64s + queue 4.83s + exec
  56.73s + altro ~1.84s), billing 63.38s, credit cost = ceil(63.38) *
  0.75 = 64 * 0.75 = 48.00
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIG_DIR = Path(__file__).parent
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 110,
    "axes.titlesize": 12,
})

# Colori coerenti con la caption
COL = {
    "validation":   "#cfe2f3",
    "compile":      "#3a78bf",
    "queue":        "#f7c895",
    "execution":    "#e8741f",
    "postproc":     "#a7e0a7",
    "other":        "#cccccc",
    "billing":      "#f4c542",   # giallo per la barra hatched
    "ceil_extra":   "#9b7d4e",
}

# Pannelli (a) 28 circuits  e (b) 273 circuits
PANELS = [
    {
        "title": "(a) Job with 28 circuits  —  median of 180 forward / weight-shift jobs of Phase E",
        # Wall-time stack (sec)
        "wall": [
            ("validation",  0.14),
            ("compile",     1.24),
            ("queue",       4.39),
            ("execution",   6.16),
            ("postproc",    0.17),
        ],
        # IQM billing window: starts at start of execution, length = runtime_seconds
        "billing_start":   1.24 + 0.14 + 4.39,   # = 5.77 (where execution starts)
        "runtime_s":       7.41,
        "credit_cost":     6.00,
        "ceil_ratio":      8 / 7.41,   # 7.41 -> ceil = 8s
        "xlim":            (0, 17),    # esteso per dar spazio al label
    },
    {
        "title": "(b) Job with 273 circuits  —  median of 3 validation jobs of Phase E (20 images $\\times$ 150 patches / K=11)",
        "wall": [
            ("validation",  0.14),
            ("compile",     6.64),
            ("queue",       4.83),
            ("execution",  56.73),
            ("postproc",    1.70),
        ],
        "billing_start":   0.14 + 6.64 + 4.83,   # = 11.61
        "runtime_s":      63.38,
        "credit_cost":    48.00,
        "ceil_ratio":     64 / 63.38,
        "xlim":           (0, 90),
    },
]

fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.0))

for ax, p in zip(axes, PANELS):
    ax.set_title(p["title"], loc="left", fontweight="bold")

    # Layout vertical: 2 barre, "IQM billing" sopra e "Wall time" sotto
    y_billing = 1
    y_wall    = 0
    bar_h     = 0.55

    # --- Wall time bar (stack) ---
    x_cursor = 0.0
    for label, dur in p["wall"]:
        ax.barh(y_wall, dur, left=x_cursor, height=bar_h,
                color=COL[label], edgecolor="white", linewidth=0.3)
        # Etichetta al centro del segmento solo se abbastanza grande
        if dur > p["xlim"][1] * 0.025:
            ax.text(x_cursor + dur/2, y_wall,
                    f"{dur:.2f}s",
                    ha="center", va="center",
                    color=("white" if label in {"compile","execution"} else "black"),
                    fontsize=9, fontweight="bold")
        x_cursor += dur
    wall_total = x_cursor

    # --- IQM billing bar (hatched) ---
    bill_x0 = p["billing_start"]
    bill_w  = p["runtime_s"]
    # Parte hatched (runtime billed) — colore yellow
    rect = mpatches.Rectangle((bill_x0, y_billing - bar_h/2),
                               bill_w, bar_h,
                               facecolor=COL["billing"], alpha=0.55,
                               edgecolor="goldenrod", linewidth=1.2,
                               hatch="//")
    ax.add_patch(rect)
    # Ceil-rounding extra
    ceil_total = np.ceil(p["runtime_s"])
    ceil_extra_w = ceil_total - p["runtime_s"]
    if ceil_extra_w > 0:
        rect_ce = mpatches.Rectangle((bill_x0 + bill_w, y_billing - bar_h/2),
                                      ceil_extra_w, bar_h,
                                      facecolor=COL["ceil_extra"], alpha=0.7,
                                      edgecolor="saddlebrown", linewidth=1.0,
                                      hatch="xx")
        ax.add_patch(rect_ce)
    # Etichetta runtime_seconds DENTRO la barra hatched, centrata
    ax.text(bill_x0 + bill_w / 2, y_billing,
            f"runtime_seconds = {p['runtime_s']:.2f}s",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="saddlebrown")
    # Etichetta Credit cost DENTRO la barra hatched, in alto a destra
    # (NUOVA POSIZIONE: prima era fuori dal grafico, ora resta interna)
    credit_text = (
        f"Credit cost = $\\lceil${p['runtime_s']:.2f}$\\rceil\\times 0.75$\n"
        f"= {int(ceil_total)} $\\times$ 0.75 = {p['credit_cost']:.2f} cred"
    )
    # Posiziona dentro la barra, leggermente sopra il centro per non sovrapporsi a runtime_seconds
    ax.text(bill_x0 + bill_w * 0.97, y_billing + 0.42,
            credit_text,
            ha="right", va="top",
            fontsize=8.5, color="darkred",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="darkred",
                      alpha=0.92, linewidth=0.8))

    # Y axis labels
    ax.set_yticks([y_wall, y_billing])
    ax.set_yticklabels(["Wall time\n(client side)", "IQM billing\n(QPU runtime)"],
                       fontsize=10)
    ax.set_ylim(-0.55, 1.75)
    ax.set_xlim(p["xlim"])
    ax.set_xlabel("seconds")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Legend in basso
legend_patches = [
    mpatches.Patch(color=COL["validation"],  label="validation"),
    mpatches.Patch(color=COL["compile"],     label="compile"),
    mpatches.Patch(color=COL["queue"],       label="queue (QPU scheduling)"),
    mpatches.Patch(color=COL["execution"],   label="execution (on QPU)"),
    mpatches.Patch(color=COL["postproc"],    label="post-processing"),
    mpatches.Patch(color=COL["other"],       label="other (negligible)"),
    mpatches.Patch(facecolor=COL["billing"], alpha=0.55, hatch="//",
                   edgecolor="goldenrod", label="IQM billing window"),
    mpatches.Patch(facecolor=COL["ceil_extra"], alpha=0.7, hatch="xx",
                   edgecolor="saddlebrown", label="ceil-rounding extra"),
]
fig.legend(handles=legend_patches, loc="lower center",
           ncol=4, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle(
    "Anatomy of an IQM Resonance job: pipeline phases vs IQM billing\n"
    "(median values across 184 jobs, Emerald, 2 May 2026)",
    fontsize=12, fontweight="bold", y=1.01)

plt.tight_layout(rect=[0, 0.04, 1, 0.98])
out = FIG_DIR / "iqm_job_anatomy.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"=> {out}")
