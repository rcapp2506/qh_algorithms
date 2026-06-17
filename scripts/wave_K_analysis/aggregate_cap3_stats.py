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
artefacts.  Accuracies are therefore rounded to 4 decimals before ranking so
that scipy assigns tied ranks correctly; the exact signed-rank test is then
used (zero_method='wilcox', correction=False).  The normal-approximation p with
tie correction is reported alongside as a robustness check.

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
        ("ccnn_big",   "ccnn_small"):  (0.027, None),
        ("noisy",      "noiseless"):   (0.062, None),
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


def paired_wilcoxon(x, y):
    """Round to 4 decimals (genuine ties), exact signed-rank + approx cross-check."""
    x = np.round(x, 4); y = np.round(y, 4); d = x - y
    n_neg, n_pos, n_zero = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
    w_exact = sstats.wilcoxon(x, y, alternative="two-sided",
                              zero_method="wilcox", correction=False, method="exact")
    w_approx = sstats.wilcoxon(x, y, alternative="two-sided",
                               zero_method="wilcox", correction=False, method="approx")
    nz = d[d != 0]
    dz = float(nz.mean() / nz.std(ddof=1)) if nz.size > 1 else float("nan")
    return {
        "mean_delta": float(d.mean()), "n_neg": n_neg, "n_pos": n_pos, "n_zero": n_zero,
        "W_exact": float(w_exact.statistic), "p_exact": float(w_exact.pvalue),
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

    print("\nPaired signed-rank tests (exact, 4-decimal tie handling):")
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
              f"W={res['W_exact']:.1f} p_exact={res['p_exact']:.4f} "
              f"(approx_tiecorr={res['p_approx_tiecorr']:.4f}){dz_txt}  "
              f"[manuscript p={ref_p}]  {'OK' if match else 'MISMATCH'}")

    (HERE / "cap3_stats_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {HERE / 'cap3_stats_summary.json'}")
    print("\nRESULT:", "ALL VALUES REPRODUCE THE MANUSCRIPT" if ok else "DISCREPANCY DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
