# Chapter 3 statistics — single source of truth

`cap3_results_per_seed.csv` is the authoritative per-seed record behind every
statistic quoted in Chapter 3. It is a tidy long-format table:

| column | meaning |
|---|---|
| `experiment` | `main_9q` (architecture comparison) or `noise_4q` (noise-resilience study) |
| `model` | `qcnn_hybrid`, `ccnn_small`, `ccnn_big` (main_9q); `noiseless`, `noisy` (noise_4q) |
| `seed` | RNG seed of the run |
| `final_val_acc` | final-epoch validation accuracy |
| `best_val_acc` | best-epoch validation accuracy |
| `n_val` | validation-set size (200 → accuracies are multiples of 1/200 = 0.005) |

Values were extracted from the raw run outputs
(`results_9q_noiseless_multiseed`, `results_small_test`, `Output_CCNN_v1_multiseed`,
`results_noiseless_seeds`, `results_noisy_seeds`) and rounded to 4 decimals.

`aggregate_cap3_stats.py` recomputes, from this CSV alone, every number used in
the manuscript: per-model mean ± sample std (ddof=1), the paired Wilcoxon
signed-rank tests between architectures and between noise conditions, and the
paired effect size d_z. It self-checks each value against the manuscript and
exits non-zero on any disagreement, so it doubles as a regression guard.

Tie handling: accuracies are integer multiples of 1/200, so |Δ| ties are
genuine. Accuracies are rounded to 4 decimals before ranking (removing float32
noise that would otherwise break real ties) and the **exact** signed-rank test
is used. At R = 10 with ties the normal approximation is unreliable and is
reported only as a cross-check.

```
python aggregate_cap3_stats.py
```

## Statistical-algorithms audit (Chapter 3)

`power_montecarlo_wilcoxon.py` + `power_montecarlo_results.json` are an **audit
performed to validate the statistical algorithms used in Chapter 3**: a Monte
Carlo estimate of the power of the exact two-sided Wilcoxon signed-rank test at
the campaign's sample size (n = 10 paired seeds, α = 0.05), under normal shift
alternatives N(d_z, 1), 4000 replications, fixed seed 42.

Audited claims (defense technical appendix, frame B4):

| d_z | 0.50 | 0.86 | 1.00 | 1.30 |
|---|---|---|---|---|
| power (published) | 0.29 | 0.65 | 0.78 | 0.94 |
| power (this audit, seed 42) | 0.294 | 0.654 | 0.777 | 0.943 |
| power (independent seed 999) | 0.275 | 0.659 | 0.785 | 0.942 |

All published values reproduced within ≤ 2 Monte Carlo standard errors
(SE ≤ 0.008 at 4000 reps); the script asserts agreement within ±0.02.
Reading: the R = 10 design is powered for large effects (d_z ≳ 1); smaller
effects are reported as "not detectable at R = 10" with the Hodges–Lehmann
bound doing the quantitative work.

```
python power_montecarlo_wilcoxon.py
```
