"""
Wave-K final aggregation: R=10 QCNN noiseless multi-seed analysis.

Inputs:
  - 7 seeds from the handoff memory (run 00,01,02,03,04,05,09)
  - 3 file JSON di recovery (run 06,07,08) in ./recovery/

Output:
  - Stat tables for final_val_acc and best_val_acc
  - Wilson 95% CI on the empirical mean (on pooled k/N and on mean p)
  - One-sample comparison vs CCNN reference 0.9835 ± 0.0047 (R=10)
  - Three figures: mean_bands.png, QCNN_distributions.png, QCNN_wilson_intervals.png
  - Summary JSON saved in output/

Methodological double check (cf. project memory):
  - mean/std: numpy ddof=1 vs scipy.stats.describe
  - Wilson CI: formula chiusa vs statsmodels.proportion_confint
  - one-sample vs CCNN: scipy ttest_1samp + Wilcoxon signed-rank one-sample
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sstats

# ---------------------------------------------------------------------------
# 1.  DATA: 7 seeds from memory + 3 from recovery JSON
# ---------------------------------------------------------------------------
# From the handoff memory (verified in the metrics.csv files of stat_runs.zip)
memory_seven = [
    # (run_idx, seed, final_val_acc, best_val_acc)
    (0,   42, 0.9550, 0.9850),
    (1,  153, 0.9650, 0.9650),
    (2,  264, 0.9150, 0.9500),
    (3,  375, 0.9350, 0.9750),
    (4,  486, 0.9800, 0.9800),
    (5,  597, 0.9700, 0.9800),
    (9, 1041, 0.9100, 0.9350),
]

# 3 recovery dai JSON
recovery_paths = sorted(Path(__file__).parent.glob("recovery/results_run_*.json"))
recovery_three = []
for p in recovery_paths:
    with open(p) as f:
        d = json.load(f)
    r = d["result"]
    recovery_three.append((d["run_idx"], d["seed"], r["final_val_acc"], r["best_val_acc"]))

# Sanity: ci aspettiamo run 06,07,08
assert sorted(x[0] for x in recovery_three) == [6, 7, 8], f"recovery runs mismatch: {recovery_three}"

# Union + sort by run_idx -> R=10 table
all_runs = sorted(memory_seven + recovery_three, key=lambda x: x[0])
assert [x[0] for x in all_runs] == list(range(10))

run_idx   = np.array([x[0] for x in all_runs])
seeds     = np.array([x[1] for x in all_runs])
final_acc = np.array([x[2] for x in all_runs])
best_acc  = np.array([x[3] for x in all_runs])

N_VAL = 200  # validation set size (max_samples_per_class=100, 2 classes)
R     = 10

print("=" * 72)
print(" Wave-K R=10 QCNN NOISELESS — aggregated table")
print("=" * 72)
print(f"{'run':>4} {'seed':>5} {'final_val_acc':>14} {'best_val_acc':>13}")
for i, s, f_, b_ in all_runs:
    print(f"{i:>4} {s:>5} {f_:>14.4f} {b_:>13.4f}")
print()

# ---------------------------------------------------------------------------
# 2.  STATISTICS - double-check with independent methods
# ---------------------------------------------------------------------------
def descrip_pair(label, x):
    """Doppio metodo: NumPy ddof=1 vs scipy.stats.describe."""
    m_np  = float(np.mean(x))
    s_np  = float(np.std(x, ddof=1))   # sample std
    desc  = sstats.describe(x)
    m_sp  = float(desc.mean)
    s_sp  = float(math.sqrt(desc.variance))   # variance is ddof=1
    se    = s_np / math.sqrt(len(x))
    assert abs(m_np - m_sp) < 1e-12, f"mean mismatch {label}"
    assert abs(s_np - s_sp) < 1e-10, f"std mismatch {label}"
    # 95% CI of the mean (Student-t, df=N-1)
    t_crit = sstats.t.ppf(0.975, df=len(x) - 1)
    ci_lo = m_np - t_crit * se
    ci_hi = m_np + t_crit * se
    return dict(mean=m_np, std=s_np, se=se,
                ci95_lo=ci_lo, ci95_hi=ci_hi,
                min=float(np.min(x)), max=float(np.max(x)),
                median=float(np.median(x)))

stat_final = descrip_pair("final_val_acc", final_acc)
stat_best  = descrip_pair("best_val_acc",  best_acc)

print("--- final_val_acc (R=10) ---")
for k, v in stat_final.items():
    print(f"  {k:>10}: {v:.5f}")
print("--- best_val_acc  (R=10) ---")
for k, v in stat_best.items():
    print(f"  {k:>10}: {v:.5f}")
print()

# ---------------------------------------------------------------------------
# 3.  WILSON 95% CI - on pooled k (sum of all correct) and on mean accuracy
# ---------------------------------------------------------------------------
# Wilson CI formula chiusa (one-proportion, two-sided 95%)
# NOTA: usiamo z esatto (norm.ppf(0.975)) per allineare con statsmodels.
Z_975 = sstats.norm.ppf(0.975)   # ≈ 1.959963984540054
def wilson_ci_closed(k, n, z=Z_975):
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half   = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)

# For each run, k = round(final_val_acc * N_val)
k_per_run = np.round(final_acc * N_VAL).astype(int)
k_total   = int(np.sum(k_per_run))            # pooled correct
n_total   = N_VAL * R                          # pooled trials
p_pool    = k_total / n_total

ci_pool_closed = wilson_ci_closed(k_total, n_total)

# Cross-check con statsmodels
try:
    from statsmodels.stats.proportion import proportion_confint
    ci_pool_sm = proportion_confint(k_total, n_total, alpha=0.05, method="wilson")
    assert abs(ci_pool_closed[0] - ci_pool_sm[0]) < 1e-10, "Wilson CI mismatch lo"
    assert abs(ci_pool_closed[1] - ci_pool_sm[1]) < 1e-10, "Wilson CI mismatch hi"
    ci_method_check = "✓ closed vs statsmodels OK"
except ImportError:
    ci_method_check = "(statsmodels not available — closed-form only)"

print("--- Wilson 95% CI ---")
print(f"  pooled k/n   = {k_total}/{n_total} = {p_pool:.5f}")
print(f"  CI 95% pool  = [{ci_pool_closed[0]:.5f}, {ci_pool_closed[1]:.5f}]")
print(f"  cross-check  : {ci_method_check}")

# Per-seed Wilson intervals (per il plot)
wilson_per_seed = np.array([wilson_ci_closed(int(k), N_VAL) for k in k_per_run])
print()

# ---------------------------------------------------------------------------
# 4.  CONFRONTO vs CCNN baseline (one-sample)
# ---------------------------------------------------------------------------
# Handoff memory: CCNN baseline R=10  -> 0.9835 +/- 0.0047
CCNN_MEAN = 0.9835
CCNN_STD  = 0.0047   # SD across the 10 CCNN seed (handoff)

# (a) one-sample t-test: QCNN final_val_acc vs μ0 = CCNN_MEAN
t_stat, t_pval = sstats.ttest_1samp(final_acc, popmean=CCNN_MEAN, alternative="two-sided")

# (b) one-sample Wilcoxon signed-rank: H0: median = CCNN_MEAN
diffs = final_acc - CCNN_MEAN
w_stat, w_pval = sstats.wilcoxon(diffs, alternative="two-sided",
                                  zero_method="wilcox", correction=False)

# (c) Welch-like approximate two-sample t (treating CCNN as known mean,std)
# QCNN sample (N=10) vs CCNN sample (N=10 implied by handoff std)
# Pooled-variance Welch t:
N_CCNN = 10
welch_t = (np.mean(final_acc) - CCNN_MEAN) / math.sqrt(np.var(final_acc, ddof=1)/R + CCNN_STD**2/N_CCNN)
welch_df_num = (np.var(final_acc, ddof=1)/R + CCNN_STD**2/N_CCNN) ** 2
welch_df_den = ((np.var(final_acc, ddof=1)/R) ** 2) / (R - 1) + ((CCNN_STD**2/N_CCNN) ** 2) / (N_CCNN - 1)
welch_df = welch_df_num / welch_df_den
welch_p  = 2.0 * (1.0 - sstats.t.cdf(abs(welch_t), df=welch_df))

# Effect size (Cohen's d, pooled SD)
pooled_sd = math.sqrt(((R-1)*np.var(final_acc, ddof=1) + (N_CCNN-1)*CCNN_STD**2) / (R + N_CCNN - 2))
cohen_d = (np.mean(final_acc) - CCNN_MEAN) / pooled_sd

print("--- vs CCNN baseline (mean=0.9835, sd=0.0047 da handoff) ---")
print(f"  QCNN final_val_acc mean = {np.mean(final_acc):.4f}")
print(f"  CCNN reference     mean = {CCNN_MEAN:.4f}")
print(f"  Δ (QCNN - CCNN)         = {np.mean(final_acc) - CCNN_MEAN:+.4f}")
print(f"  one-sample t-test       : t = {t_stat:.3f}, p = {t_pval:.2e}")
print(f"  signed-rank (one-sample): W = {w_stat:.2f}, p = {w_pval:.2e}")
print(f"  Welch two-sample approx : t = {welch_t:.3f}, df ≈ {welch_df:.1f}, p = {welch_p:.2e}")
print(f"  Cohen's d (pooled SD)   : d = {cohen_d:.3f}")
print()

# ---------------------------------------------------------------------------
# 5.  CONVERGENCE: extract learning curves from the 3 recovery files
#     (for the 7 missing ones use only the final values; for the bands use R=3 of the recovery
#      + faithful reconstruction from metrics if available.  For now build the bands
#      with the available data and flag it.)
# ---------------------------------------------------------------------------
val_curves = []
train_curves = []
for p in recovery_paths:
    with open(p) as f:
        d = json.load(f)
    val_curves.append(d["result"]["val_accuracies"])
    train_curves.append(d["result"]["train_accuracies"])
val_curves_arr = np.array(val_curves)        # shape (3, 10) — recovery only
train_curves_arr = np.array(train_curves)    # shape (3, 10)

n_epochs = val_curves_arr.shape[1]
epochs_x = np.arange(1, n_epochs + 1)

val_mean  = val_curves_arr.mean(axis=0)
val_std   = val_curves_arr.std(axis=0, ddof=1)
train_mean = train_curves_arr.mean(axis=0)
train_std  = train_curves_arr.std(axis=0, ddof=1)

print(f"--- learning-curve bands (recovery only, n=3): WARNING ---")
print(f"  NOTE: bands built on the 3 recovery seeds; for publication")
print(f"        the curves of the 7 memory seeds (metrics.csv) would also need to be included.")
print()

# ---------------------------------------------------------------------------
# 6.  PLOTS
# ---------------------------------------------------------------------------
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 110,
})

# --- Plot 1: mean_bands.png (val accuracy vs epoch, mean ± std) -------------
fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.plot(epochs_x, val_mean, "-", color="#1f77b4", lw=2.0, label="QCNN val (mean)")
ax.fill_between(epochs_x, val_mean - val_std, val_mean + val_std,
                color="#1f77b4", alpha=0.22, label=r"$\pm\,1\sigma$ (recovery, n=3)")
ax.plot(epochs_x, train_mean, "--", color="#d62728", lw=1.7,
        label="QCNN train (mean)")
ax.fill_between(epochs_x, train_mean - train_std, train_mean + train_std,
                color="#d62728", alpha=0.18)
ax.axhline(CCNN_MEAN, ls=":", color="#2ca02c", lw=1.5,
           label=f"CCNN reference ({CCNN_MEAN:.4f})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("QCNN noiseless — mean ± σ training curves (R=10 final, n=3 curves)")
ax.set_xlim(1, n_epochs)
ax.set_ylim(0.45, 1.02)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "mean_bands.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# --- Plot 2: QCNN_distributions.png (hist + KDE-like + per-seed strip) ------
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

# (a) histogram of final_val_acc + best_val_acc
bins = np.arange(0.88, 1.001, 0.01)
axes[0].hist(final_acc, bins=bins, alpha=0.55, color="#1f77b4",
             edgecolor="black", label="final_val_acc")
axes[0].hist(best_acc,  bins=bins, alpha=0.45, color="#ff7f0e",
             edgecolor="black", label="best_val_acc")
axes[0].axvline(np.mean(final_acc), color="#1f77b4", ls="--", lw=1.5,
                label=f"mean final = {np.mean(final_acc):.4f}")
axes[0].axvline(np.mean(best_acc),  color="#ff7f0e", ls="--", lw=1.5,
                label=f"mean best  = {np.mean(best_acc):.4f}")
axes[0].axvline(CCNN_MEAN, color="#2ca02c", ls=":",  lw=1.8,
                label=f"CCNN ({CCNN_MEAN:.4f})")
axes[0].set_xlabel("Validation accuracy")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution across R=10 seeds")
axes[0].grid(alpha=0.3)
axes[0].legend(loc="upper left", fontsize=8)

# (b) per-seed strip plot
xs = np.arange(R)
axes[1].vlines(xs, final_acc, best_acc, color="#888", lw=1.2, alpha=0.6)
axes[1].scatter(xs, final_acc, color="#1f77b4", s=40, zorder=3, label="final_val_acc")
axes[1].scatter(xs, best_acc,  color="#ff7f0e", s=40, marker="^", zorder=3, label="best_val_acc")
axes[1].axhline(np.mean(final_acc), color="#1f77b4", ls="--", lw=1.0, alpha=0.7)
axes[1].axhline(CCNN_MEAN, color="#2ca02c", ls=":",  lw=1.5, label=f"CCNN {CCNN_MEAN:.4f}")
axes[1].set_xticks(xs)
axes[1].set_xticklabels([f"r{r}\ns{s}" for r, s in zip(run_idx, seeds)], fontsize=7)
axes[1].set_ylabel("Validation accuracy")
axes[1].set_title("Per-seed final vs best accuracy")
axes[1].grid(alpha=0.3)
axes[1].legend(loc="lower right", fontsize=8)

fig.tight_layout()
fig.savefig(FIG_DIR / "QCNN_distributions.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# --- Plot 3: QCNN_wilson_intervals.png (forest plot Wilson per seed + pooled) -
fig, ax = plt.subplots(figsize=(7.5, 5.0))
y_pos = np.arange(R)
for i in range(R):
    lo, hi = wilson_per_seed[i]
    p_i = final_acc[i]
    ax.plot([lo, hi], [i, i], color="#1f77b4", lw=2.0)
    ax.plot(p_i, i, "o", color="#1f77b4", markersize=6)

# Pooled CI as a band
ax.axvspan(ci_pool_closed[0], ci_pool_closed[1], color="#1f77b4", alpha=0.10,
           label=f"pooled Wilson 95% CI [{ci_pool_closed[0]:.4f}, {ci_pool_closed[1]:.4f}]")
ax.axvline(p_pool, color="#1f77b4", ls="--", lw=1.5,
           label=f"pooled p̂ = {p_pool:.4f}")
ax.axvline(CCNN_MEAN, color="#2ca02c", ls=":", lw=1.8,
           label=f"CCNN reference ({CCNN_MEAN:.4f})")

ax.set_yticks(y_pos)
ax.set_yticklabels([f"run {r:02d} (seed {s})" for r, s in zip(run_idx, seeds)])
ax.set_xlabel("Validation accuracy (final epoch)")
ax.set_title("Wilson 95% CI per seed + pooled (QCNN noiseless, R=10)")
ax.grid(alpha=0.3, axis="x")
ax.legend(loc="lower left", fontsize=9)
ax.set_xlim(0.85, 1.005)
fig.tight_layout()
fig.savefig(FIG_DIR / "QCNN_wilson_intervals.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"--- Plots scritti in {FIG_DIR}/ ---")
for p in sorted(FIG_DIR.glob("*.png")):
    print(f"   {p.name}  ({p.stat().st_size//1024} KB)")
print()

# ---------------------------------------------------------------------------
# 7.  SUMMARY JSON
# ---------------------------------------------------------------------------
summary = dict(
    architecture="hybrid_qcnn_v1_single  (num_qubits=9, kernel=3, C_channels=6)",
    config=dict(R=R, N_val=N_VAL, max_epochs=10, batch_size=16, lr=0.001,
                max_samples_per_class=100),
    runs=[dict(run_idx=int(i), seed=int(s),
               final_val_acc=float(f_), best_val_acc=float(b_))
          for i, s, f_, b_ in all_runs],
    stats=dict(
        final_val_acc=stat_final,
        best_val_acc=stat_best,
    ),
    pooled_wilson=dict(
        k_total=int(k_total), n_total=int(n_total),
        p_hat=float(p_pool),
        ci95_lo=float(ci_pool_closed[0]),
        ci95_hi=float(ci_pool_closed[1]),
    ),
    ccnn_reference=dict(mean=CCNN_MEAN, std=CCNN_STD, R=N_CCNN,
                        source="handoff project memory"),
    comparison_vs_ccnn=dict(
        delta_mean=float(np.mean(final_acc) - CCNN_MEAN),
        one_sample_t=dict(t=float(t_stat), p=float(t_pval)),
        one_sample_wilcoxon=dict(W=float(w_stat), p=float(w_pval)),
        welch_two_sample=dict(t=float(welch_t), df=float(welch_df), p=float(welch_p)),
        cohen_d_pooled=float(cohen_d),
    ),
)

OUT = Path(__file__).parent / "output" / "summary_R10_noiseless.json"
OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary scritto in {OUT}")

# ---------------------------------------------------------------------------
# 8.  Snippet pronti per il TeX (i 6 numeri chiave da sostituire)
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(" SNIPPET TeX (6 numeri chiave)")
print("=" * 72)
print(f"  1. QCNN  final_val_acc  : {stat_final['mean']:.4f} \\pm {stat_final['std']:.4f}")
print(f"  2. QCNN  best_val_acc   : {stat_best['mean']:.4f} \\pm {stat_best['std']:.4f}")
print(f"  3. QCNN  range          : [{stat_final['min']:.4f}, {stat_final['max']:.4f}]")
print(f"  4. Wilson 95% pooled CI : [{ci_pool_closed[0]:.4f}, {ci_pool_closed[1]:.4f}]")
print(f"  5. Δ vs CCNN (mean)     : {np.mean(final_acc)-CCNN_MEAN:+.4f}")
print(f"     CCNN ref            : {CCNN_MEAN:.4f} \\pm {CCNN_STD:.4f}")
print(f"  6. p-value (Welch)      : {welch_p:.2e}    (Cohen's d = {cohen_d:.2f})")
