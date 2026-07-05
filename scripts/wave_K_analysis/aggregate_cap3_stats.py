#!/usr/bin/env python3
"""
aggregate_cap3_stats.py -- Single source of truth for the Chapter 3 statistics.

Reads the committed per-seed results (cap3_results_per_seed.csv) and recomputes,
from the raw numbers alone, every statistic quoted in the manuscript:

  * per-model mean and sample std (ddof=1) of final validation accuracy
  * paired Wilcoxon signed-rank tests between architectures (main 9q campaign)
    and between noise conditions (4q noise-resilience study)
  * paired effect size d_z = mean(delta) / std(delta, ddof=1)

Tie handling.  The validation set has n_val = 200, so every accuracy is an
integer multiple of 1/200 = 0.005 and ties in |delta| are GENUINE, not float
artefacts.  Paired differences are therefore snapped to the exact quantisation
grid (integer multiples of 1/n_val) before ranking, so tied |delta| are
bit-identical and receive proper midranks; the exact signed-rank test is then
used.  The pinned p is the brute-force enumeration of all 2**n_eff sign
patterns under midranks (zero_method='wilcox'), which is library-independent;
scipy's exact value is reported alongside and asserted equal whenever the
non-zero |delta| are tie-free (the unambiguous case).  The normal-approximation
p with tie correction is reported as a further robustness check.

Note: rounding the accuracies alone does NOT remove the float artefact (the
*differences* of rounded floats are still numerically unequal); an earlier
revision did exactly that and pinned p = 0.0625 for the noise comparison,
which is the tie-free-enumeration value.  Under proper midranks the six tied
non-zero differences make the signed-rank test degenerate to the sign test:
p = 14/64 = 0.21875.

Output: prints a table and writes cap3_stats_summary.json next to this script.
Exit status is non-zero if any recomputed value disagrees with the value quoted
in the manuscript (REFERENCE below), so the script doubles as a regression guard.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sstats

HERE = Path(__file__).resolve().parent
CSV = HERE / "cap3_results_per_seed.csv"

# Values quoted in the manuscript (Chapter 3). Recomputation must match these.
REFERENCE = {
    "mean_std": {
        ("main_9q", "qcnn_hybrid"): (0.9445, 0.0243),
        ("main_9q", "ccnn_small"): (0.9700, 0.0133),
        ("main_9q", "ccnn_big"):   (0.9835, 0.0047),
        ("noise_4q", "noiseless"): (0.9640, 0.0147),
        ("noise_4q", "noisy"):     (0.9620, 0.0136),
    },
    "wilcoxon": {  # (a, b): (p, dz or None)
        ("qcnn_hybrid", "ccnn_small"): (0.027, -0.87),
        ("qcnn_hybrid", "ccnn_big"):   (0.004, None),
        ("ccnn_big",   "ccnn_small"):  (0.023, +0.97),
        ("noisy",      "noiseless"):   (0.219, None),
    },
}


def load(csv_path):
    data = defaultdict(dict)  # (experiment, model) -> {seed: final_val_acc}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            data[(r["experiment"], r["model"])][int(r["seed"])] = float(r["final_val_acc"])
    return data


def arr(data, experiment, model):
    d = data[(experiment, model)]
    return np.array([d[s] for s in sorted(d)]), sorted(d)


N_VAL = 200  # validation-set size: accuracies are exact multiples of 1/N_VAL


def _exact_midrank_p(d):
    """AUTHORITATIVE two-sided exact signed-rank p: enumerate all 2**n_eff sign
    patterns on the midranks of |d| (zeros discarded, Wilcoxon zero policy) and
    count patterns with min(W+, W-) <= observed.  Since the sign-flip null makes
    the W+ distribution symmetric about S/2 (S = sum of ranks), this equals the
    classical 2*min-tail definition.  Library-independent; feasible n_eff <= 20."""
    nz = d[d != 0]
    r = sstats.rankdata(np.abs(nz))
    w_obs = min(r[nz > 0].sum(), r[nz < 0].sum())
    n = nz.size
    count = 0
    for pattern in range(2 ** n):
        mask = np.array([(pattern >> i) & 1 for i in range(n)], dtype=bool)
        if min(r[mask].sum(), r[~mask].sum()) <= w_obs + 1e-9:
            count += 1
    return count / 2 ** n


def paired_wilcoxon(x, y):
    """Snap differences to the 1/N_VAL grid (genuine ties become bit-identical);
    p from the authoritative midrank enumeration; scipy exact + tie-corrected
    approx reported as cross-checks.  When the non-zero |d| are tie-free the
    convention is unambiguous and scipy is asserted equal to the enumeration."""
    d = np.round((x - y) * N_VAL) / N_VAL
    n_neg, n_pos, n_zero = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
    w_scipy = sstats.wilcoxon(d, alternative="two-sided",
                              zero_method="wilcox", correction=False, method="exact")
    w_approx = sstats.wilcoxon(d, alternative="two-sided",
                               zero_method="wilcox", correction=False, method="approx")
    p_enum = _exact_midrank_p(d)
    nz = d[d != 0]
    ties_present = np.unique(np.abs(nz)).size < nz.size
    if not ties_present:
        assert abs(p_enum - float(w_scipy.pvalue)) < 1e-12, (
            f"tie-free case: scipy ({w_scipy.pvalue}) != enumeration ({p_enum})")
    r = sstats.rankdata(np.abs(nz))
    w_stat = float(min(r[nz > 0].sum(), r[nz < 0].sum()))
    dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
    return {
        "mean_delta": float(d.mean()), "n_neg": n_neg, "n_pos": n_pos, "n_zero": n_zero,
        "W_midrank": w_stat, "p_exact": p_enum,
        "p_scipy_exact": float(w_scipy.pvalue), "ties_present": bool(ties_present),
        "p_approx_tiecorr": float(w_approx.pvalue), "dz": dz,
    }


def main():
    data = load(CSV)
    summary = {"per_model": {}, "paired_tests": {}}
    ok = True

    print("=" * 72)
    print("Chapter 3 statistics -- recomputed from cap3_results_per_seed.csv")
    print("=" * 72)
    print("\nPer-model final validation accuracy (mean +/- sample std, ddof=1):")
    for (exp, model), (ref_m, ref_s) in REFERENCE["mean_std"].items():
        a, _ = arr(data, exp, model)
        m, s = float(a.mean()), float(a.std(ddof=1))
        match = abs(m - ref_m) < 5e-4 and abs(s - ref_s) < 5e-4
        ok &= match
        summary["per_model"][f"{exp}/{model}"] = {"mean": m, "std": s, "n": int(a.size)}
        print(f"  {exp:8s} {model:12s} {m:.4f} +/- {s:.4f}   "
              f"[manuscript {ref_m:.4f} +/- {ref_s:.4f}]  {'OK' if match else 'MISMATCH'}")

    print("\nPaired signed-rank tests (exact, grid-snapped tie handling):")
    # resolve which experiment each model belongs to
    model_exp = {m: e for (e, m) in data}
    for (a_name, b_name), (ref_p, ref_dz) in REFERENCE["wilcoxon"].items():
        exp = model_exp[a_name]
        xa, _ = arr(data, exp, a_name)
        xb, _ = arr(data, exp, b_name)
        res = paired_wilcoxon(xa, xb)
        match = abs(res["p_exact"] - ref_p) < 1e-3
        if ref_dz is not None:
            match &= abs(res["dz"] - ref_dz) < 1e-2
        ok &= match
        summary["paired_tests"][f"{a_name}_vs_{b_name}"] = res
        dz_txt = f", dz={res['dz']:+.2f}" if ref_dz is not None else ""
        print(f"  {a_name:12s} vs {b_name:12s}: "
              f"W={res['W_midrank']:.1f} p_exact={res['p_exact']:.4f} "
              f"(scipy={res['p_scipy_exact']:.4f}, "
              f"approx_tiecorr={res['p_approx_tiecorr']:.4f}){dz_txt}  "
              f"[manuscript p={ref_p}]  {'OK' if match else 'MISMATCH'}")

    (HERE / "cap3_stats_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {HERE / 'cap3_stats_summary.json'}")
    print("\nRESULT:", "ALL VALUES REPRODUCE THE MANUSCRIPT" if ok else "DISCREPANCY DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
