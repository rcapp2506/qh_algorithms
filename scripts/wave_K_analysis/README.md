# Wave-K analysis scripts

Statistical-analysis utilities for the Wave-K multi-seed R=10 campaign
of Cap. 3 of the PhD thesis "New Perspectives on Quantum Technologies"
(Roberto Cappuccio, Università di Siena, Ciclo XXXVII).

These scripts assume the following directory layout at the project root
(or pass paths explicitly):

```
.
├── recovery/
│   ├── results_run_06_seed_00708.json    # QCNN recovery (3 seed full)
│   ├── results_run_07_seed_00819.json
│   ├── results_run_08_seed_00930.json
│   └── ccnn_results.json                 # CCNN-big multi-seed (10 seed)
├── ccnn_small/
│   └── results_run_NN_seed_SSSSS.json    # CCNN-small (10 seed),
│                                         # output of run_ccnn_small_matched_multiseed.py
├── qcnn_full/stat_runs/
│   └── run_NN_sSSS/metrics.csv           # QCNN per-epoch curves (7 of 10 seed)
```

## Scripts

### `aggregate_R10_noiseless.py`
Aggregate QCNN R=10 noiseless results: descriptive stats with double-check
(numpy ddof=1 vs scipy.stats.describe), Wilson 95% CI on the pooled
binomial proportion (closed-form vs statsmodels), and one-sample tests
against the CCNN-big baseline. Produces `summary_R10_noiseless.json` and
three PNG plots.

### `audit_params.py`
Reproducible parameter count of the three architectures (QCNN hybrid,
CCNN-big high-capacity, CCNN-small matched-capacity), with block-level
decomposition. Confirms that CCNN-small differs from QCNN by 0.07% of
total trainable parameters.

### `paired_pere_con_pere.py`
Three paired Wilcoxon signed-rank tests across the three simulator
levels (Level 1 = CCNN-big, Level 2 = CCNN-small, Level 3 = QCNN). Each
test is performed in both exact and asymptotic form, with rounding to
4 decimal places to neutralise float32 precision artefacts. Includes
sign tests, paired t-tests (parametric sanity), and effect sizes
(paired Cohen's d, rank-biserial correlation). Produces
`summary_pere_con_pere.json`.

### `plot_pere_con_pere.py`
Three figures: per-seed slope plot across the three levels with per-seed
Δ bars; violin plot of the three levels with annotated paired p-values;
mean ± σ curves of QCNN versus CCNN-small over all 10 seeds (QCNN
curves rebuilt from 7 metrics.csv in stat_runs.zip + 3 recovery JSON).

### `plot_four_level.py`
Two figures: 4-level evidence ladder overview (sim curves + Emerald HW
points with Wilson 95% CI); compact bar summary at the final epoch.
The Emerald data are hard-coded from the manuscript's
`Tab. emerald_convergence`.

## Reproducibility

All scripts perform a double-check on every quantitative claim with two
independent methods, in line with the project memory rule
*"never declare a result ready without two methods"*. Bootstrap intervals
are computed with both the percentile method and the BCa method;
proportional CIs with both a closed-form formula and `statsmodels`;
McNemar tests with both `scipy.stats.binomtest` and
`statsmodels.stats.contingency_tables.mcnemar`.

## Numerical results summary (Wave-K closure, 2026-05-17)

| Level | Architecture            | Params       | Final acc          | Paired vs Level 2     | Cohen's d |
|------:|:------------------------|-------------:|-------------------:|:----------------------|----------:|
| 1     | CCNN-big high-capacity  | 85,617,041   | 0.9835 ± 0.0047    | W=3.5,  p=2.7×10⁻²    | +0.97     |
| 2     | CCNN-small matched      |    463,898   | 0.9700 ± 0.0133    | — (reference)         |    —      |
| 3     | QCNN noiseless (hybrid) |    463,574   | 0.9445 ± 0.0243    | W=6.0,  p=2.7×10⁻²    | −0.87     |
| 4     | QCNN on IQM Emerald HW  | (same as L3) | 1.00 at epoch 2    | (single run, N_val=20)|    —      |

Item-level McNemar (Level 3 vs Level 1, 600 paired items pooled over 3
QCNN seeds with on-disk per-item correctness): b=30, c=3, exact pooled
p = 1.4×10⁻⁶.
