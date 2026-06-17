# Cost-model single source of truth (Chapter 3, Phase E)

All cost-model numbers reported in Chapter 3 (Sec. `cost_model`) are derived
from the **real IQM Resonance job records** of the Phase E fine-tuning campaign
(IQM Emerald, 2026-05-02). This directory and `scripts/cost_model/` make that
derivation reproducible end-to-end; nothing is hand-entered or synthetic.

## Files

- `data/iqm_cost_model/iqm_jobs_raw_dump.json`
  Real job metadata dumped from the IQM Resonance jobs API
  (`GET /webapp/api/jobs/paginated`, Bearer `IQM_TOKEN`, zero credits — metadata
  only). Each record carries the server-side timestamps, the billed
  `runtime_seconds`, and the dashboard `credit_cost`. 200 records total
  (full account history); the cost-model dataset is the 195 completed Emerald
  jobs created on 2026-05-02.
- `data/iqm_cost_model/iqm_cost_model_per_job.csv`
  Tidy per-job table produced by the analysis script (phase durations, billing
  window, credit cost).
- `scripts/cost_model/analyze_iqm_cost.py`
  Self-verifying analysis: loads the dump, reproduces the cost-model numbers
  (asserting them against the manuscript), exports the CSV, and regenerates
  both cost-model figures from the real data.

## Cost-model dataset (195 jobs)

| group | jobs | N | role |
|---|---|---|---|
| production | 180 | 28 | forward / weight-shift batches (m=2, K=11) |
| production | 3 | 273 | end-of-epoch validation (20 imgs x 150 patches / 11) |
| exploration | 12 | 1,5,20,50 | small-N timing-estimation probes |

## Reproduced quantities (all asserted by the script)

- **Per-job runtime fit** `runtime_seconds(N) = A + B*N`:
  `A = 1.49 +/- 0.07 s`, `B = 0.225 +/- 0.001 s/circ`, `R^2 = 0.992` (n=195).
  This is the single cost-model equation used in the chapter.
- **Billing rule** `credit = ceil(runtime_seconds) * 0.75`: holds for 195/195 jobs.
- **Billing window vs execution** (the point of the job-anatomy figure):
  the billed `runtime_seconds` is the QPU **billing window**, equal to the
  execution window plus a per-job init/teardown overhead.
  - N=28: `7.41 s = 6.16 (exec) + 1.24 (init/teardown)`, `credit = ceil(7.41)*0.75 = 6.00`
  - N=273: `63.38 s = 56.73 (exec) + 6.64 (init/teardown)`, `credit = ceil(63.38)*0.75 = 48.00`
- **Production totals**: 1282.5 credits over 1580.7 s of billed QPU runtime.
- **Exploratory probes**: 12 jobs, 51.8 credits, of which the immediate
  pre-fine-tuning timing-estimation batch at 14:58 (N=1,20,50) is 18.0 credits.

## Regenerate

```bash
python3 scripts/cost_model/analyze_iqm_cost.py
```

Writes `iqm_cost_model_per_job.csv` and, under `scripts/cost_model/figures/`,
`cost_model_validation.png` and `iqm_job_anatomy.png` (then copied into the
thesis `chapters/qa_figures/`).
