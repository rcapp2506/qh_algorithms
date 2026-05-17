"""
Wave-K — confronti paired QCNN vs CCNN-small (pere con pere) e vs CCNN-big.

R=10 seed identici tra i 3 esperimenti:
- QCNN noiseless (Aer statevector, 463 574 trainable params, 18 quantum)
- CCNN-small matched (463 898 params, Δ=+324 da QCNN, +0.07%)
- CCNN-big high-capacity (85.6M params, 184× del QCNN)

Statistiche calcolate:
- descrittive (mean, SD, range) per i 3 gruppi
- bootstrap 95% CI on mean
- paired Wilcoxon signed-rank exact (QCNN vs CCNN-small)
- paired Wilcoxon signed-rank exact (QCNN vs CCNN-big)
- paired Wilcoxon signed-rank exact (CCNN-small vs CCNN-big)
- Cohen's d paired, rank-biserial r per ogni paired test
- sign test (sanity check)
- paired t-test (sanity check parametrico)

DOPPIO CHECK
============
- mean/std: numpy ddof=1 vs scipy.stats.describe
- Wilcoxon: scipy method='exact' vs scipy method='approx'
- bootstrap: percentile vs BCa
"""

from __future__ import annotations
import json, math, glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sstats

ROOT = Path(__file__).parent
OUT = ROOT / "output_pere"; OUT.mkdir(exist_ok=True)
FIG = ROOT / "figures_pere"; FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Carica i 3 gruppi di 10 seed ciascuno
# ---------------------------------------------------------------------------
# QCNN: 7 dalla memoria + 3 dai recovery
QCNN_MEM = [
    (0,   42, 0.9550, 0.9850),
    (1,  153, 0.9650, 0.9650),
    (2,  264, 0.9150, 0.9500),
    (3,  375, 0.9350, 0.9750),
    (4,  486, 0.9800, 0.9800),
    (5,  597, 0.9700, 0.9800),
    (9, 1041, 0.9100, 0.9350),
]
qcnn_final = {s: f_ for _,s,f_,_ in QCNN_MEM}
qcnn_best  = {s: b_ for _,s,_,b_ in QCNN_MEM}
for p in sorted(glob.glob(str(ROOT/'recovery/results_run_*.json'))):
    if 'ccnn' in p: continue
    with open(p) as f: d = json.load(f)
    qcnn_final[d['seed']] = d['result']['final_val_acc']
    qcnn_best [d['seed']] = d['result']['best_val_acc']

# CCNN-small
ccnn_small_final = {}
ccnn_small_best  = {}
for p in sorted(glob.glob(str(ROOT/'ccnn_small/results_run_*.json'))):
    with open(p) as f: d = json.load(f)
    ccnn_small_final[d['seed']] = d['result']['final_val_acc']
    ccnn_small_best [d['seed']] = d['result']['best_val_acc']

# CCNN-big
with open(ROOT/'recovery/ccnn_results.json') as f:
    ccnn_doc = json.load(f)
ccnn_big_final = {r['seed']: r['final_val_acc'] for r in ccnn_doc['results']}
ccnn_big_best  = {r['seed']: r['best_val_acc']  for r in ccnn_doc['results']}

seeds = sorted(qcnn_final.keys())
assert seeds == sorted(ccnn_small_final.keys()) == sorted(ccnn_big_final.keys())
assert seeds == [42,153,264,375,486,597,708,819,930,1041]
R = len(seeds)

# Arrays allineati
qcnn_arr = np.array([qcnn_final[s] for s in seeds])
cs_arr   = np.array([ccnn_small_final[s] for s in seeds])
cb_arr   = np.array([ccnn_big_final[s] for s in seeds])

qcnn_best_arr = np.array([qcnn_best[s] for s in seeds])
cs_best_arr   = np.array([ccnn_small_best[s] for s in seeds])
cb_best_arr   = np.array([ccnn_big_best[s] for s in seeds])

# ---------------------------------------------------------------------------
# 2. Tabella seed-by-seed
# ---------------------------------------------------------------------------
print("=" * 88)
print(" Wave-K R=10 — pere con pere — final_val_acc")
print("=" * 88)
print(f"{'seed':>5} | {'QCNN':>7} | {'CCNN-small':>10} | {'CCNN-big':>9} | "
      f"{'Δ_small':>8} | {'Δ_big':>8}")
print("-" * 88)
for i, s in enumerate(seeds):
    d_small = qcnn_arr[i] - cs_arr[i]
    d_big   = qcnn_arr[i] - cb_arr[i]
    print(f"{s:>5} | {qcnn_arr[i]:>7.4f} | {cs_arr[i]:>10.4f} | {cb_arr[i]:>9.4f} | "
          f"{d_small:>+8.4f} | {d_big:>+8.4f}")
print()

# ---------------------------------------------------------------------------
# 3. Statistiche descrittive (doppio check)
# ---------------------------------------------------------------------------
def descrip(name, x):
    m_np = float(np.mean(x)); s_np = float(np.std(x, ddof=1))
    desc = sstats.describe(x)
    assert abs(m_np - desc.mean) < 1e-12
    assert abs(s_np - math.sqrt(desc.variance)) < 1e-10
    return dict(name=name, n=len(x), mean=m_np, std=s_np, se=s_np/math.sqrt(len(x)),
                min=float(np.min(x)), max=float(np.max(x)), median=float(np.median(x)))

print("--- DESCRITTIVE (final_val_acc, double-checked) ---")
for lbl, arr in [("QCNN       ", qcnn_arr), ("CCNN-small ", cs_arr), ("CCNN-big   ", cb_arr)]:
    d = descrip(lbl, arr)
    print(f"  {lbl}: mean={d['mean']:.4f}  sd={d['std']:.4f}  median={d['median']:.4f}  "
          f"range=[{d['min']:.4f}, {d['max']:.4f}]")
print()

# ---------------------------------------------------------------------------
# 4. Bootstrap 95% CI on means + differences
# ---------------------------------------------------------------------------
print("--- BOOTSTRAP 95% CI on the mean (10k resamples, percentile + BCa) ---")

def bootstrap_pct(x, n_boot=10000, alpha=0.05, seed=12345):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 100*alpha/2)), float(np.percentile(means, 100*(1-alpha/2)))

def bca_ci(x):
    try:
        b = sstats.bootstrap((x,), np.mean, n_resamples=10000,
                              confidence_level=0.95, method='BCa', random_state=12345)
        return float(b.confidence_interval.low), float(b.confidence_interval.high)
    except Exception:
        return None

for lbl, arr in [("QCNN", qcnn_arr), ("CCNN-small", cs_arr), ("CCNN-big", cb_arr)]:
    pct = bootstrap_pct(arr)
    bca = bca_ci(arr)
    print(f"  {lbl:>10}: pct=[{pct[0]:.4f}, {pct[1]:.4f}]  BCa=[{bca[0]:.4f}, {bca[1]:.4f}]")

print()
diff_qs = qcnn_arr - cs_arr   # QCNN minus small
diff_qb = qcnn_arr - cb_arr   # QCNN minus big
diff_bs = cb_arr - cs_arr     # big minus small

print(f"  Δ(QCNN - small): mean={diff_qs.mean():+.4f}  sd={diff_qs.std(ddof=1):.4f}  "
      f"pct CI = [{bootstrap_pct(diff_qs)[0]:+.4f}, {bootstrap_pct(diff_qs)[1]:+.4f}]")
print(f"  Δ(QCNN - big)  : mean={diff_qb.mean():+.4f}  sd={diff_qb.std(ddof=1):.4f}  "
      f"pct CI = [{bootstrap_pct(diff_qb)[0]:+.4f}, {bootstrap_pct(diff_qb)[1]:+.4f}]")
print(f"  Δ(big - small) : mean={diff_bs.mean():+.4f}  sd={diff_bs.std(ddof=1):.4f}  "
      f"pct CI = [{bootstrap_pct(diff_bs)[0]:+.4f}, {bootstrap_pct(diff_bs)[1]:+.4f}]")
print()

# ---------------------------------------------------------------------------
# 5. Paired Wilcoxon signed-rank — ARROTONDATO 4dp (per ties float32 da QCNN)
# ---------------------------------------------------------------------------
def paired_wilcoxon(x, y, label):
    """Wilcoxon signed-rank exact + asymp, con arrotondamento a 4 cifre per
    neutralizzare i tie spurii da float32 precision."""
    xr = np.round(x, 4); yr = np.round(y, 4)
    d  = xr - yr
    n_neg = int(np.sum(d < 0))
    n_pos = int(np.sum(d > 0))
    n_zero= int(np.sum(d == 0))
    n_eff = n_neg + n_pos
    w_exact = sstats.wilcoxon(xr, yr, alternative='two-sided',
                               zero_method='wilcox', correction=False, method='exact')
    w_asymp = sstats.wilcoxon(xr, yr, alternative='two-sided',
                               zero_method='wilcox', correction=False, method='approx')
    # one-sided in direzione del segno della mediana
    alt = 'less' if np.median(d) < 0 else ('greater' if np.median(d) > 0 else 'two-sided')
    w_one = sstats.wilcoxon(xr, yr, alternative=alt,
                             zero_method='wilcox', correction=False, method='exact')
    # sign test
    sign_p = sstats.binomtest(min(n_neg, n_pos), n=n_neg+n_pos, p=0.5,
                               alternative='two-sided').pvalue if n_eff > 0 else 1.0
    # paired t (sanity)
    t_p = sstats.ttest_rel(xr, yr, alternative='two-sided')
    # Cohen's d paired
    sd_d = np.std(d, ddof=1) if d.size > 1 and np.std(d, ddof=1) > 0 else float('nan')
    d_paired = np.mean(d) / sd_d if not math.isnan(sd_d) else float('nan')
    # rank-biserial
    if n_eff > 0:
        W_plus = float(w_exact.statistic)
        sum_ranks = n_eff * (n_eff + 1) / 2
        r_rb = (2*W_plus - sum_ranks) / sum_ranks
    else:
        r_rb = float('nan')

    print(f"\n--- PAIRED WILCOXON: {label} ---")
    print(f"   diffs rounded 4dp: {n_neg} neg, {n_pos} pos, {n_zero} zero (n_eff={n_eff})")
    print(f"   exact   2-sided : W={w_exact.statistic:>5.1f}, p={w_exact.pvalue:.4e}")
    print(f"   asymp   2-sided : W={w_asymp.statistic:>5.1f}, p={w_asymp.pvalue:.4e}")
    print(f"   exact 1-sided '{alt}': W={w_one.statistic:>5.1f}, p={w_one.pvalue:.4e}")
    print(f"   sign test       : p={sign_p:.4e}")
    print(f"   paired t-test   : t={t_p.statistic:+.3f}, df={n_eff-1 if n_eff>1 else 0}, p={t_p.pvalue:.4e}")
    print(f"   Cohen's d (pair): {d_paired:+.3f}")
    print(f"   rank-biserial r : {r_rb:+.3f}")

    return dict(
        label=label,
        n_neg=n_neg, n_pos=n_pos, n_zero=n_zero, n_eff=n_eff,
        wilcoxon_exact_two_sided=dict(W=float(w_exact.statistic), p=float(w_exact.pvalue)),
        wilcoxon_asymp_two_sided=dict(W=float(w_asymp.statistic), p=float(w_asymp.pvalue)),
        wilcoxon_one_sided=dict(alt=alt, W=float(w_one.statistic), p=float(w_one.pvalue)),
        sign_test_p=float(sign_p),
        paired_t=dict(t=float(t_p.statistic), p=float(t_p.pvalue)),
        cohens_d_paired=float(d_paired) if not math.isnan(d_paired) else None,
        rank_biserial=float(r_rb) if not math.isnan(r_rb) else None,
    )

res_qs = paired_wilcoxon(qcnn_arr, cs_arr, "QCNN vs CCNN-small (PERE CON PERE)")
res_qb = paired_wilcoxon(qcnn_arr, cb_arr, "QCNN vs CCNN-big (asymmetric capacity)")
res_bs = paired_wilcoxon(cb_arr, cs_arr, "CCNN-big vs CCNN-small (capacity effect)")

# ---------------------------------------------------------------------------
# 6. Output JSON
# ---------------------------------------------------------------------------
summary = dict(
    config=dict(R=R, seeds=seeds,
                qcnn_params=463574, ccnn_small_params=463898, ccnn_big_params=85617041),
    descriptives=dict(
        qcnn      = descrip("QCNN", qcnn_arr),
        ccnn_small= descrip("CCNN-small", cs_arr),
        ccnn_big  = descrip("CCNN-big", cb_arr),
        diff_qs   = descrip("Δ QCNN-small", diff_qs),
        diff_qb   = descrip("Δ QCNN-big",   diff_qb),
        diff_bs   = descrip("Δ big-small",  diff_bs),
    ),
    bootstrap_ci=dict(
        qcnn=bootstrap_pct(qcnn_arr),
        ccnn_small=bootstrap_pct(cs_arr),
        ccnn_big=bootstrap_pct(cb_arr),
        diff_qs=bootstrap_pct(diff_qs),
        diff_qb=bootstrap_pct(diff_qb),
        diff_bs=bootstrap_pct(diff_bs),
    ),
    paired_qcnn_vs_ccnn_small=res_qs,
    paired_qcnn_vs_ccnn_big=res_qb,
    paired_ccnn_big_vs_ccnn_small=res_bs,
    raw=dict(
        qcnn=[(s, float(f_)) for s, f_ in zip(seeds, qcnn_arr)],
        ccnn_small=[(s, float(f_)) for s, f_ in zip(seeds, cs_arr)],
        ccnn_big=[(s, float(f_)) for s, f_ in zip(seeds, cb_arr)],
    ),
    notes=[
        "QCNN val_labels_final order: interleaved 0,0,1,1,0,0,...",
        "CCNN-small val_labels_final order: block 100 zeros then 100 ones",
        "CCNN-big val_labels_final order: interleaved 0,0,1,1,0,0,... (same as QCNN)",
        "Item-wise paired (McNemar) requires per-sample matching by file path, not by index.",
        "Seed-level paired Wilcoxon is fully valid (operates on per-seed accuracy summaries).",
    ],
)
with open(OUT / 'summary_pere_con_pere.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n=> {OUT/'summary_pere_con_pere.json'}")
