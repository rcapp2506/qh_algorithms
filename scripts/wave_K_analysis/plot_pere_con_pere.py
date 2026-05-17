"""Plot pere-con-pere: 3 modelli su 10 seed, con curve QCNN COMPLETE n=10.

Plot 1: paired slope + per-seed Δ bars (QCNN, CCNN-small, CCNN-big)
Plot 2: violin + jitter per i 3 modelli
Plot 3: curves overlay QCNN vs CCNN-small (ENTRAMBI n=10)
"""
import csv, json, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent
FIG = ROOT / "figures_pere"; FIG.mkdir(exist_ok=True)

# Carico summary
with open(ROOT/'output_pere/summary_pere_con_pere.json') as f:
    summ = json.load(f)

seeds = summ['config']['seeds']
qcnn = np.array([v for _, v in summ['raw']['qcnn']])
cs   = np.array([v for _, v in summ['raw']['ccnn_small']])
cb   = np.array([v for _, v in summ['raw']['ccnn_big']])

R = len(seeds)

# ============================================================
# Carico le 10 curve QCNN COMPLETE: 7 da metrics.csv + 3 da JSON recovery
# ============================================================
qcnn_curves = {}
for d in sorted(Path(ROOT/'qcnn_full/stat_runs').iterdir()):
    if not d.is_dir(): continue
    s = int(d.name.split('_s')[1])
    with open(d / 'metrics.csv') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    qcnn_curves[s] = dict(
        val_accs=[float(r['val_acc']) for r in rows],
        train_accs=[float(r['train_acc']) for r in rows],
        val_losses=[float(r['val_loss']) for r in rows],
        train_losses=[float(r['train_loss']) for r in rows],
    )
for p in sorted(glob.glob(str(ROOT/'recovery/results_run_*.json'))):
    if 'ccnn' in p: continue
    with open(p) as f: d = json.load(f)
    s = d['seed']
    r = d['result']
    qcnn_curves[s] = dict(
        val_accs=r['val_accuracies'],
        train_accs=r['train_accuracies'],
        val_losses=r['val_losses'],
        train_losses=r['train_losses'],
    )
assert sorted(qcnn_curves.keys()) == seeds, "QCNN curves missing for some seeds"

qcnn_val_arr   = np.array([qcnn_curves[s]['val_accs']   for s in seeds])   # (10, 10)
qcnn_train_arr = np.array([qcnn_curves[s]['train_accs'] for s in seeds])
qcnn_val_loss  = np.array([qcnn_curves[s]['val_losses']  for s in seeds])
qcnn_train_loss= np.array([qcnn_curves[s]['train_losses'] for s in seeds])

# CCNN-small (10 curve)
cs_val_curves = []
cs_train_curves = []
cs_val_loss   = []
cs_train_loss = []
for p in sorted(glob.glob(str(ROOT/'ccnn_small/results_run_*.json'))):
    with open(p) as f: d = json.load(f)
    r = d['result']
    cs_val_curves.append(r['val_accuracies'])
    cs_train_curves.append(r['train_accuracies'])
    cs_val_loss.append(r['val_losses'])
    cs_train_loss.append(r['train_losses'])
# Lightning include sanity check (epoca 0); val_accuracies len 11 — drop epoca 0
cs_val_arr     = np.array(cs_val_curves)[:, 1:]      # (10, 10)
cs_train_arr   = np.array(cs_train_curves)            # (10, 10)
cs_val_loss_arr= np.array(cs_val_loss)[:, 1:]
cs_train_loss_arr = np.array(cs_train_loss)

print(f"QCNN val curves shape: {qcnn_val_arr.shape}")
print(f"CCNN-small val curves shape: {cs_val_arr.shape}")

# Colori
Q_C  = "#1f77b4"   # QCNN blue
CS_C = "#ff7f0e"   # CCNN-small orange
CB_C = "#2ca02c"   # CCNN-big green

plt.rcParams.update({"font.size": 11, "figure.dpi": 110})

epochs = np.arange(1, qcnn_val_arr.shape[1] + 1)

# --- Plot 1: triple comparison (slope + diff bars) ----------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

# (a) slope plot — 3 colonne
ax = axes[0]
positions = [0, 1, 2]
for i in range(R):
    ax.plot(positions, [qcnn[i], cs[i], cb[i]], "-",
            color="gray", alpha=0.5, lw=1.2, marker='o', ms=5)
ax.plot(positions, [qcnn.mean(), cs.mean(), cb.mean()], "s-",
        color="#d62728", ms=12, lw=3, label="across-seed mean",
        markeredgecolor='black', markeredgewidth=1.5, zorder=10)
for x, m, sd, n_p in zip(positions, [qcnn.mean(), cs.mean(), cb.mean()],
                          [qcnn.std(ddof=1), cs.std(ddof=1), cb.std(ddof=1)],
                          [463574, 463898, 85617041]):
    ax.text(x, m + 0.005, f"{m:.4f}±{sd:.4f}", ha="center", fontsize=10,
            fontweight='bold', color="#d62728")
    ax.text(x, 0.895, f"{n_p:,}\nparams", ha="center", fontsize=8.5,
            color='darkblue', fontfamily='DejaVu Sans Mono')
ax.set_xticks(positions)
ax.set_xticklabels(["QCNN\n(noiseless Aer)", "CCNN-small\n(matched ≈463k)",
                     "CCNN-big\n(high-cap 85.6M)"], fontsize=10)
ax.set_ylabel("Final validation accuracy")
ax.set_title(f"Per-seed paired comparison across 3 models (R={R})")
ax.set_xlim(-0.4, 2.4); ax.set_ylim(0.88, 1.0)
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="upper left")

# (b) diff bars: Δ(QCNN − small) e Δ(big − small)
ax = axes[1]
diff_qs = qcnn - cs
diff_qb_offset = (cb - cs)  # capacity effect
x = np.arange(R)
w = 0.35
b1 = ax.bar(x - w/2, diff_qs, w,
            color=[Q_C if d <= 0 else "#999" for d in diff_qs],
            edgecolor="black", linewidth=0.7,
            label="Δ = QCNN − CCNN-small")
b2 = ax.bar(x + w/2, cb - cs, w,
            color=[CB_C if d >= 0 else "#999" for d in cb - cs],
            edgecolor="black", linewidth=0.7,
            label="Δ = CCNN-big − CCNN-small (capacity effect)")
ax.axhline(0, color="black", lw=0.8)
ax.axhline(diff_qs.mean(), ls="--", color=Q_C, lw=1.5,
           label=f"mean Δ(QCNN−small) = {diff_qs.mean():+.4f}")
ax.axhline((cb - cs).mean(), ls="--", color=CB_C, lw=1.5,
           label=f"mean Δ(big−small) = {(cb-cs).mean():+.4f}")
ax.set_xticks(x)
ax.set_xticklabels([f"s={s}" for s in seeds], rotation=45)
ax.set_ylabel("Accuracy difference")
ax.set_title("Per-seed differences: matched-capacity test vs capacity effect")
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "triple_paired_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"=> {FIG / 'triple_paired_comparison.png'}")

# --- Plot 2: violin con jitter (3 distrib.) ------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.0))
data = [qcnn, cs, cb]
positions = [0, 1, 2]
parts = ax.violinplot(data, positions=positions, widths=0.65,
                      showmeans=False, showmedians=False, showextrema=False)
for pc, color in zip(parts['bodies'], [Q_C, CS_C, CB_C]):
    pc.set_facecolor(color); pc.set_alpha(0.4); pc.set_edgecolor("black")

rng = np.random.default_rng(7)
for pos, arr, ci_key in zip(positions, data, ['qcnn', 'ccnn_small', 'ccnn_big']):
    jit = rng.uniform(-0.07, 0.07, len(arr))
    ax.scatter(np.full_like(arr, pos) + jit, arr, color="black", s=30,
               alpha=0.75, zorder=3)
    ci = summ['bootstrap_ci'][ci_key]
    ax.plot([pos, pos], ci, color="red", lw=3.5, alpha=0.65, zorder=2)
    ax.plot(pos, arr.mean(), "rs", ms=10, zorder=4)

# Brackets per i p-values
def bracket(ax, x1, x2, y, txt):
    ax.plot([x1, x1, x2, x2], [y, y+0.003, y+0.003, y], color="black", lw=1.2)
    ax.text((x1+x2)/2, y+0.005, txt, ha="center", fontsize=9)

p_qs = summ['paired_qcnn_vs_ccnn_small']['wilcoxon_exact_two_sided']['p']
p_qb = summ['paired_qcnn_vs_ccnn_big']['wilcoxon_exact_two_sided']['p']
p_bs = summ['paired_ccnn_big_vs_ccnn_small']['wilcoxon_exact_two_sided']['p']

bracket(ax, 0, 1, 1.005, f"Wilcoxon p={p_qs:.3f} *")
bracket(ax, 1, 2, 1.018, f"Wilcoxon p={p_bs:.3f} *")
bracket(ax, 0, 2, 1.031, f"Wilcoxon p={p_qb:.3f} **")

ax.set_xticks(positions)
ax.set_xticklabels([
    f"QCNN\n0.9445±0.0243\n(463 574 params,\n18 quantum)",
    f"CCNN-small\n0.9700±0.0133\n(463 898 params)",
    f"CCNN-big\n0.9835±0.0047\n(85.6M params)",
], fontsize=10)
ax.set_ylabel("Final validation accuracy")
ax.set_title(f"Wave-K R=10 — pere con pere: 3 models, paired comparison\n"
             "(red bar: bootstrap 95% CI on mean; * p<0.05, ** p<0.01)")
ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0.88, 1.05)
fig.tight_layout()
fig.savefig(FIG / "violin_three_models.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"=> {FIG / 'violin_three_models.png'}")

# --- Plot 3: curves overlay QCNN vs CCNN-small (entrambi n=10) ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))

# Pannello loss
ax = axes[0]
m, sd = qcnn_train_loss.mean(0), qcnn_train_loss.std(0, ddof=1)
ax.plot(epochs, m, "-", color=Q_C, lw=1.5, alpha=0.6, label=r"QCNN train (mean, n=10)")
ax.fill_between(epochs, m-sd, m+sd, color=Q_C, alpha=0.10)
m, sd = qcnn_val_loss.mean(0), qcnn_val_loss.std(0, ddof=1)
ax.plot(epochs, m, "-", color=Q_C, lw=2.5, label=r"QCNN val (mean, n=10)")
ax.fill_between(epochs, m-sd, m+sd, color=Q_C, alpha=0.22)

m, sd = cs_train_loss_arr.mean(0), cs_train_loss_arr.std(0, ddof=1)
ax.plot(epochs, m, "--", color=CS_C, lw=1.5, alpha=0.6, label=r"CCNN-small train (mean, n=10)")
ax.fill_between(epochs, m-sd, m+sd, color=CS_C, alpha=0.10)
m, sd = cs_val_loss_arr.mean(0), cs_val_loss_arr.std(0, ddof=1)
ax.plot(epochs, m, "--", color=CS_C, lw=2.5, label=r"CCNN-small val (mean, n=10)")
ax.fill_between(epochs, m-sd, m+sd, color=CS_C, alpha=0.22)

ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
ax.set_title("Loss vs epoch — pere con pere")
ax.set_xlim(1, epochs.max())
ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)

# Pannello accuracy
ax = axes[1]
m, sd = qcnn_train_arr.mean(0), qcnn_train_arr.std(0, ddof=1)
ax.plot(epochs, m, "-", color=Q_C, lw=1.5, alpha=0.6, label=r"QCNN train (mean, n=10)")
ax.fill_between(epochs, m-sd, m+sd, color=Q_C, alpha=0.10)
m, sd = qcnn_val_arr.mean(0), qcnn_val_arr.std(0, ddof=1)
ax.plot(epochs, m, "-", color=Q_C, lw=2.5, label=r"QCNN val (mean, n=10)")
ax.fill_between(epochs, m-sd, m+sd, color=Q_C, alpha=0.22)

m, sd = cs_train_arr.mean(0), cs_train_arr.std(0, ddof=1)
ax.plot(epochs, m, "--", color=CS_C, lw=1.5, alpha=0.6, label=r"CCNN-small train (mean, n=10)")
ax.fill_between(epochs, m-sd, m+sd, color=CS_C, alpha=0.10)
m, sd = cs_val_arr.mean(0), cs_val_arr.std(0, ddof=1)
ax.plot(epochs, m, "--", color=CS_C, lw=2.5, label=r"CCNN-small val (mean, n=10)")
ax.fill_between(epochs, m-sd, m+sd, color=CS_C, alpha=0.22)

# CCNN-big reference
ax.axhline(0.9835, ls=":", color=CB_C, lw=1.8, label="CCNN-big final = 0.9835")
ax.axhspan(0.9835 - 0.0047, 0.9835 + 0.0047, color=CB_C, alpha=0.08)

ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy")
ax.set_title("Accuracy vs epoch — pere con pere")
ax.set_xlim(1, epochs.max()); ax.set_ylim(0.45, 1.02)
ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)

fig.suptitle(f"Wave-K: hybrid QCNN vs matched-capacity CCNN-small, R=10 multi-seed campaign\n"
             "(curves mean ±1σ over 10 seeds for both models; CCNN-big shown as horizontal reference)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIG / "curves_QCNN_vs_CCNN_small.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"=> {FIG / 'curves_QCNN_vs_CCNN_small.png'}")
