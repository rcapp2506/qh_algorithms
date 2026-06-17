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
