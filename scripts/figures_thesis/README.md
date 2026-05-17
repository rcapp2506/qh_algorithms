# Figure-regeneration scripts for Chapter 3 of the thesis

This folder contains standalone Python scripts that regenerate the
hardware-side figures of Cap. 3 of the PhD thesis "New Perspectives on
Quantum Technologies" (Roberto Cappuccio, Università di Siena, Ciclo
XXXVII), when their original generator notebook is not at hand.

Each script reproduces a single figure from the explicit numerical
values reported in the corresponding caption + table of the
manuscript, without depending on the Phase D / Phase E raw dataset
that produced it. Numerical fidelity is by design: the layout was
tweaked across multiple review rounds (Roberto's annotations), the
underlying numbers remained unchanged.

## Files

### `regen_iqm_job_anatomy.py`
Regenerates `chapters/qa_figures/iqm_job_anatomy.png` (Fig. 3.16 in the
final PDF), which is the *Anatomy of an IQM Resonance job* figure:
two-bar phase-decomposition timeline for the 28-circuit (forward /
weight-shift) job and the 273-circuit (validation) job, with the IQM
billing window superimposed and the credit-cost computation annotated.

Numbers from caption: wall stack 1.24/4.39/6.16 s (validate/compile/
queue/execute) and runtime_seconds=7.41 s for N=28; analogous
6.64/4.83/56.73 s and runtime_seconds=63.38 s for N=273.

Layout fix vs. the original notebook plot: the "Credit cost = ..." box
sits inside the IQM billing hatched bar (top-right of the bar) instead
of floating outside the plot area (where it visually overlapped the
title of the subplot below).

### `regen_cost_model_validation.py`
Regenerates `chapters/qa_figures/cost_model_validation.png` (Fig. 3.17
in the final PDF), which is the *Cost-model validation* figure:
(a) scatter + linear fit of the per-job QPU runtime vs N (n=195 jobs);
(b) stacked-bar phase decomposition at N=28 (180 jobs) and N=273
(3 jobs), with the wall:/billed: annotations inside the plot area.

Numbers from caption: A=1.49 s, B=0.225 s/circuit, R²=0.992; per-bar
totals N=28 (wall 12.6 s, billed 6.2 s) and N=273 (wall 69.8 s, billed
56.7 s).

Layout fix vs. the original notebook plot: ylim raised from 85 s to
95 s, wall/billed bboxes positioned at +3 s above each bar top (inside
the plot area) instead of overlapping the subplot title, and the
in-slice annotation threshold raised so that values <2 s no longer
clip below the x-axis at the bottom of the N=28 bar.

## Usage

Both scripts are standalone (only require matplotlib + numpy). Run from
their directory:

```bash
cd scripts/figures_thesis
python regen_iqm_job_anatomy.py        # writes alongside the script
python regen_cost_model_validation.py  # writes alongside the script
```

To deploy in the manuscript repo, copy the output PNG over the
corresponding file in `PhDThesis/chapters/qa_figures/`.
