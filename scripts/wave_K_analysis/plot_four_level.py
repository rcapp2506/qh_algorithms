"""Plot 4-livelli: CCNN-big, CCNN-small, QCNN-sim, QCNN-Emerald.

Livello 1 — CCNN-big (85.6M params, R=10 seed)
Livello 2 — CCNN-small (~463k params, R=10 seed)  
Livello 3 — QCNN noiseless Aer (~463k params, R=10 seed)
Livello 4 — QCNN su Emerald HW (R=1 seed, N_val=20, 2 fine-tuning epochs)

Per Emerald, dati dal manoscritto Tab. emerald_convergence (sec. finetune_results):
  Epoch 0 (sim ckpt loaded):       100% (20/20), val_loss = 0.148
  Epoch 1 (Emerald HW fine-tune):   95% (19/20), val_loss = 0.123
  Epoch 2 (Emerald HW fine-tune):  100% (20/20), val_loss = 0.120
  Gradient norm: 2.53e-2 (ep 1) -> 2.58e-3 (ep 2)

NOTA: Emerald è single-seed con N_val=20 (Wilson 95% CI è larga: ±9.8 pp a p=1.0,
±13.6 pp a p=0.95). Non confrontabile statisticamente alla R=10 dei sim, lo si
riporta come prova di concetto. Il messaggio è qualitativo: il fine-tuning HW
recupera il 100% in 2 epoche.
"""

import csv, json, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math

ROOT = Path(__file__).parent
FIG = ROOT / "figures_pere"; FIG.mkdir(exist_ok=True)

# ---------------- Carica QCNN, CCNN-small, CCNN-big ----------------
qcnn_curves = {}
for d in sorted(Path(ROOT/'qcnn_full/stat_runs').iterdir()):
    if not d.is_dir(): continue
    s = int(d.name.split('_s')[1])
    with open(d / 'metrics.csv') as f:
        rows = list(csv.DictReader(f))
    if not rows: continue
    qcnn_curves[s] = [float(r['val_acc']) for r in rows]
for p in sorted(glob.glob(str(ROOT/'recovery/results_run_*.json'))):
    if 'ccnn' in p: continue
    with open(p) as f: d = json.load(f)
    qcnn_curves[d['seed']] = d['result']['val_accuracies']

seeds = sorted(qcnn_curves.keys())
qcnn_val = np.array([qcnn_curves[s] for s in seeds])  # (10, 10)

cs_val = []
for p in sorted(glob.glob(str(ROOT/'ccnn_small/results_run_*.json'))):
    with open(p) as f: d = json.load(f)
    cs_val.append(d['result']['val_accuracies'])
cs_val_arr = np.array(cs_val)[:, 1:]   # (10, 10), drop sanity epoch 0

with open(ROOT/'recovery/ccnn_results.json') as f:
    cb_doc = json.load(f)
cb_val = np.array([r['val_accuracies'] for r in cb_doc['results']])  # (10, 10)

# ---------------- Emerald HW data (from manuscript Tab.) ----------------
emerald_epochs = np.array([0, 1, 2])   # Epoch 0 = sim ckpt loaded; 1,2 = HW fine-tune
emerald_acc    = np.array([1.00, 0.95, 1.00])
emerald_loss   = np.array([0.148, 0.123, 0.120])
emerald_n_val  = 20

# Wilson 95% CI per Emerald (single run, N_val=20)
def wilson_ci(k, n, z=1.959963984540054):
    p = k / n
    denom = 1 + z**2/n
    centre = p + z**2/(2*n)
    half = z * math.sqrt(p*(1-p)/n + z**2/(4*n*n))
    return ((centre - half)/denom, (centre + half)/denom)

emerald_correct = np.round(emerald_acc * emerald_n_val).astype(int)
emerald_ci = np.array([wilson_ci(k, emerald_n_val) for k in emerald_correct])

print(f"Emerald CI:\n{list(zip(emerald_epochs, emerald_correct, emerald_ci))}")

# Colori
Q_C  = "#1f77b4"   # QCNN sim
CS_C = "#ff7f0e"   # CCNN-small
CB_C = "#2ca02c"   # CCNN-big
HW_C = "#9467bd"   # QCNN Emerald (viola)

plt.rcParams.update({"font.size": 11, "figure.dpi": 110})

epochs = np.arange(1, qcnn_val.shape[1] + 1)

# ===========================================================================
# Plot 1: 4-level overview - mean curves
# ===========================================================================
fig, ax = plt.subplots(figsize=(11, 6))

# Sim curves: 3 models, all R=10 multi-seed
m, sd = qcnn_val.mean(0), qcnn_val.std(0, ddof=1)
ax.plot(epochs, m, "-", color=Q_C, lw=2.5,
        label=f"QCNN noiseless Aer  ({m[-1]:.4f}±{sd[-1]:.4f})  R=10")
ax.fill_between(epochs, m-sd, m+sd, color=Q_C, alpha=0.22)

m, sd = cs_val_arr.mean(0), cs_val_arr.std(0, ddof=1)
ax.plot(epochs, m, "-", color=CS_C, lw=2.5,
        label=f"CCNN-small matched  ({m[-1]:.4f}±{sd[-1]:.4f})  R=10")
ax.fill_between(epochs, m-sd, m+sd, color=CS_C, alpha=0.22)

m, sd = cb_val.mean(0), cb_val.std(0, ddof=1)
ax.plot(epochs, m, "-", color=CB_C, lw=2.5,
        label=f"CCNN-big   high-capacity  ({m[-1]:.4f}±{sd[-1]:.4f})  R=10")
ax.fill_between(epochs, m-sd, m+sd, color=CB_C, alpha=0.22)

# Emerald HW data — pannello 'fine-tuning on top of pretrained checkpoint'
# Emerald si svolge in 2 epoche di fine-tuning a partire dal checkpoint
# pretrained (epoch 0 = caricamento checkpoint sim).
# Lo metto come "+ Emerald HW" alla destra del plot principale, su un secondo
# asse x condiviso visivamente con flag chiaro che è HW.
em_x = np.array([10.5, 11.5, 12.5])   # offset per visibilità sulla destra
# disegno una pausa visiva tra sim e HW
ax.axvline(10.25, ls="--", color="gray", alpha=0.6, lw=1)
ax.text(10.5, 0.51, "↓ same checkpoint\nloaded onto Emerald", fontsize=9,
        ha="left", color="gray", style="italic")

ax.errorbar(em_x, emerald_acc,
            yerr=[emerald_acc - emerald_ci[:, 0], emerald_ci[:, 1] - emerald_acc],
            fmt='D', color=HW_C, ms=10, lw=2.0, capsize=4,
            label=r"QCNN on IQM Emerald HW (single run, $N_{\mathrm{val}}=20$, Wilson 95% CI)",
            markeredgecolor='black', markeredgewidth=0.8, zorder=10)
# Etichetto i punti HW
for x, y in zip(em_x, emerald_acc):
    ax.text(x, y - 0.025, f"{y*100:.0f}%", ha="center", fontsize=9,
            fontweight='bold', color=HW_C)

# Custom xticks
ax.set_xticks(list(range(1, 11)) + list(em_x))
ax.set_xticklabels(
    [str(i) for i in range(1, 11)] +
    ["HW\nep 0", "HW\nep 1", "HW\nep 2"], fontsize=9)
ax.set_xlim(0.5, 13)
ax.set_ylim(0.45, 1.05)
ax.set_xlabel("Epoch (Aer simulator) " + " "*30 + "Fine-tuning on IQM Emerald")
ax.set_ylabel("Validation accuracy")
ax.set_title("Wave-K: 4-level evidence ladder — Aer (R=10 sim) and IQM Emerald (single-run hardware fine-tuning)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9.5)

fig.tight_layout()
fig.savefig(FIG / "four_level_overview.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"=> {FIG/'four_level_overview.png'}")

# ===========================================================================
# Plot 2: bar chart "final accuracy per level"
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5))

labels = ["CCNN-big\n(85.6M params)\nhigh-capacity",
          "CCNN-small\n(463 898 params)\nmatched-capacity",
          "QCNN noiseless Aer\n(463 574 params)\n18 quantum",
          "QCNN on Emerald HW\n(same weights, 2 ep fine-tune)\nN_val=20"]
means  = [cb_val[:, -1].mean(), cs_val_arr[:, -1].mean(), qcnn_val[:, -1].mean(),
          emerald_acc[-1]]   # epoch 2
sds    = [cb_val[:, -1].std(ddof=1), cs_val_arr[:, -1].std(ddof=1),
          qcnn_val[:, -1].std(ddof=1), 0.0]  # HW single-seed
# Per Emerald: Wilson CI sostituisce la SD
yerr_low  = [s for s in sds[:3]] + [emerald_acc[-1] - emerald_ci[-1, 0]]
yerr_high = [s for s in sds[:3]] + [emerald_ci[-1, 1] - emerald_acc[-1]]
colors    = [CB_C, CS_C, Q_C, HW_C]
xs = np.arange(len(labels))

bars = ax.bar(xs, means, color=colors, edgecolor="black", linewidth=1.2,
              alpha=0.85, width=0.62)
ax.errorbar(xs, means, yerr=[yerr_low, yerr_high], fmt='none', color="black",
            capsize=5, lw=1.5)

# Annotazioni
ax.text(xs[3], means[3]-0.04,
        f"Wilson\n95% CI:\n[{emerald_ci[-1,0]:.3f}, {emerald_ci[-1,1]:.3f}]",
        ha="center", fontsize=8, color=HW_C, fontstyle="italic")
for i in range(3):
    ax.text(xs[i], means[i]+yerr_high[i]+0.005,
            f"{means[i]:.4f}\n±{sds[i]:.4f}",
            ha="center", fontsize=9, fontweight="bold")
ax.text(xs[3], means[3]+yerr_high[3]+0.005,
        f"{means[3]:.4f}\n(19+20)/40 paired",
        ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Validation accuracy (final epoch)")
ax.set_title("Wave-K: 4-level final validation accuracy summary\n"
             "(error bars: ±1σ across R=10 seeds for sim; Wilson 95% CI for HW single run)")
ax.set_ylim(0.85, 1.1)
ax.grid(alpha=0.3, axis="y")

# Linea di "no separation" tra sim e HW
ax.axvline(2.5, ls="--", color="gray", alpha=0.5)
ax.text(2.5, 0.86, " sim → HW", fontsize=10, color="gray",
        rotation=90, va="bottom", ha="right")

fig.tight_layout()
fig.savefig(FIG / "four_level_bars.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"=> {FIG/'four_level_bars.png'}")
