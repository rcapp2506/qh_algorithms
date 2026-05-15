# Multi-seed statistical campaign

This directory hosts the multi-seed training infrastructure for the
Chapter 3 of the thesis (Quantum Algorithms). The goal is to back the
across-run variability statements in the manuscript with $R=10$
independent replications per architecture, plus the appropriate
inferential layer for the cross-architecture comparison.

## Files

- `multirun.py` — shared module. Provides:
  - `SEED_LIST = [42, 43, ..., 51]`
  - `set_global_seeds(seed)` — fixes PyTorch, NumPy, Python `random`,
    DataLoader workers
  - `run_multiseed(...)` — driver, loops over the seed list and saves
    one curves-CSV and one predictions-CSV per (architecture, seed)
  - `load_aggregated(csv_dir, architecture)` — aggregator
  - `wilson_ci`, `wilcoxon_paired`, `bootstrap_ci_mean` — statistics
  - `plot_with_band`, `plot_single_architecture_with_std`,
    `plot_three_architectures` — plotting with paler-tint variance band

- `HQCNN.ipynb` — Hybrid Q-CNN (architecture label `qcnn`). The
  multi-seed cells are appended at the end of the notebook; the
  pre-existing single-seed demo cells are kept as a smoke test.

- `qiskit_one_q.ipynb` — Classical CNN ablation (label `ccnn`).
  Multi-seed cells appended at the end. **Note**: the appended cells
  assume the model class is called `HybridNet` and the data module is
  `EuroSATDataModule`. Edit `make_model()` / `make_data()` in the
  injected cells if your local notebook differs.

- `esa_modello.ipynb` — Pure quantum reference (label `pure_q`). Same
  notes as above.

- `multirun_aggregate.ipynb` — cross-architecture aggregator. Run last,
  after the three single-architecture notebooks have produced their
  CSVs. Produces the summary table, the Wilcoxon paired tests, the
  cross-architecture plot, and a LaTeX-ready snippet.

## Order of execution

```
HQCNN.ipynb            -> multirun_csv/qcnn/{curves,predictions}_seed*.csv
                       -> ../PhDThesis/chapters/qa_figures/QCNN_with_std.png

qiskit_one_q.ipynb     -> multirun_csv/ccnn/{...}.csv
                       -> ../PhDThesis/chapters/qa_figures/CCNN_with_std.png

esa_modello.ipynb      -> multirun_csv/pure_q/{...}.csv
                       -> ../PhDThesis/chapters/qa_figures/Quantum-only_with_std.png

multirun_aggregate.ipynb
                       -> across-architecture summary
                       -> Wilcoxon paired tests
                       -> ../PhDThesis/chapters/qa_figures/cross_architecture_with_std.png
                       -> LaTeX snippet for Cap.3 sec:results-stats
```

The wall-clock cost of a single $R=10$ run is approximately
$10 \times (\text{wall time of the single-seed demo})$. On CPU this is
typically a few hours per architecture; on a single GPU it should fit
in an hour or two per architecture.

## Determinism

The three notebooks share `pl.seed_everything(seed, workers=True)`, so
that the only thing that varies across seeds is, as documented, the
weight initialisation and the mini-batch ordering. The train/validation
split itself is a function of the seed (because
`EuroSATDataset.__init__` calls `random.shuffle(self.data)` and we seed
the global `random` module via `seed_everything`). The seed is
therefore the unit of replication: for a given seed, all three
architectures see the same split, so the Wilcoxon paired comparison in
`multirun_aggregate.ipynb` pairs runs that used the same split.

## Statistics: what we report and what we do not

We report:
- across-run mean ± 1 standard deviation per epoch (the variance band
  in the figures)
- bootstrap 95% percentile CI on the across-run mean (descriptive)
- Wilson 95% CI on the single-run validation accuracy (descriptive
  single-run uncertainty; with $N_\text{val}\sim 20$ the half-width is
  large by construction, so the interval should not be used as a
  cross-architecture discriminator)
- Wilcoxon signed-rank paired test on the $R=10$ across-seed accuracy
  differences (the appropriate inferential test in this setup)

We do **not** report:
- McNemar's exact test on per-item discordant counts: with
  $N_\text{val}\sim 20$ and accuracy $\sim 0.9$ the discordant counts
  are 2–3 in total and McNemar has essentially zero power.

This is consistent with the protocol documented in
`chapters/quantum_algorithms.tex`, Sec. *Statistical uncertainty across
runs* of Chapter 3.
