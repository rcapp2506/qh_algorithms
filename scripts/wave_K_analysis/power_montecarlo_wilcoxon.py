#!/usr/bin/env python3
"""
power_montecarlo_wilcoxon.py — Monte Carlo power of the exact two-sided
Wilcoxon signed-rank test used throughout Ch. 3 (defense appendix frame B4).

Methodology, exactly as declared on B4:
  - paired sample size n = 10 (the R = 10 seed campaign);
  - two-sided exact test, alpha = 0.05;
  - alternatives: normal shifts, i.e. paired differences X_i ~ N(d_z, 1),
    so the standardised effect size is d_z = mu/sigma;
  - d_z grid: 0.5, 0.86 (the observed |d_z| vs the matched control),
    1.0, 1.3;
  - 4000 Monte Carlo replications per d_z, fixed seed.

Published B4 values (suite v5.9+): power = 0.29, 0.65, 0.78, 0.94.
Monte Carlo standard error at 4000 reps: sqrt(p(1-p)/4000) <= 0.008,
so reproduction is asserted within +/- 0.02 of the published table.

Output: power_montecarlo_results.json (config + per-d_z power + MC SE).
"""
import json
import numpy as np
from scipy.stats import wilcoxon

ALPHA = 0.05
N = 10
REPS = 4000
SEED = 42
DZ_GRID = [0.5, 0.86, 1.0, 1.3]
PUBLISHED = {0.5: 0.29, 0.86: 0.65, 1.0: 0.78, 1.3: 0.94}

rng = np.random.default_rng(SEED)
results = {}
for dz in DZ_GRID:
    rejections = 0
    for _ in range(REPS):
        x = rng.normal(loc=dz, scale=1.0, size=N)
        p = wilcoxon(x, alternative="two-sided", method="exact").pvalue
        rejections += (p <= ALPHA)
    power = rejections / REPS
    se = float(np.sqrt(power * (1 - power) / REPS))
    results[dz] = {"power": power, "mc_se": se, "published": PUBLISHED[dz]}
    print(f"d_z = {dz:4.2f}:  power = {power:.3f}  (MC SE {se:.3f})"
          f"   published {PUBLISHED[dz]:.2f}   |diff| = {abs(power-PUBLISHED[dz]):.3f}")

for dz, r in results.items():
    assert abs(r["power"] - r["published"]) <= 0.02, \
        f"d_z={dz}: {r['power']:.3f} vs published {r['published']:.2f}"
print("\nAll published B4 values reproduced within ±0.02 (≈2.5 MC SE).")

json.dump({"config": {"n": N, "alpha": ALPHA, "test": "wilcoxon signed-rank, "
                      "two-sided, exact", "alternative": "normal shift N(dz,1)",
                      "reps": REPS, "seed": SEED},
           "results": {str(k): v for k, v in results.items()}},
          open("power_montecarlo_results.json", "w"), indent=2)
print("saved power_montecarlo_results.json")
