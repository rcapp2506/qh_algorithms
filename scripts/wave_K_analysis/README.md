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

### `aggregate_cap3_stats.py`
Single source of truth for the Cap. 3 paired statistics. Runs the three
paired Wilcoxon signed-rank tests across the simulator levels (QCNN vs
CCNN-small, QCNN vs CCNN-big, CCNN-big vs CCNN-small), each in exact and
asymptotic form with 4-decimal rounding to neutralise float32 artefacts,
plus sign tests, paired t-tests, and paired effect sizes (Cohen's d,
rank-biserial). Reads `cap3_results_per_seed.csv` and asserts every value
against the manuscript. Supersedes the earlier per-level analysis scripts.

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
