"""
multirun.py — Multi-seed training runs and statistical analysis for
the QCNN / classical-CNN / pure-quantum benchmarks of Cap.3 of the thesis.

This module is shared by the three notebooks
    HQCNN.ipynb           (architecture = 'qcnn')
    qiskit_one_q.ipynb    (architecture = 'ccnn')
    esa_modello.ipynb     (architecture = 'pure_q')
and by the cross-architecture aggregator
    multirun_aggregate.ipynb

The contract:
  * run_multiseed() trains R=10 independent runs under a fixed seed list,
    each run differing only in the random seed used at weight initialisation,
    at mini-batch shuffling, and at the train/val split inside
    EuroSATDataset.__init__ (which calls random.shuffle on the file list).
  * For each run we save train/val accuracy and loss per epoch, plus the
    final per-item correctness vector on the validation set.
  * The aggregator then reports the across-run mean +- standard deviation
    per epoch (the band displayed in the *_with_std.png plots), plus
    Wilson 95% confidence intervals on single-run accuracies and Wilcoxon
    signed-rank paired tests across architectures.

Notes on the statistical layer.
  We deliberately keep the statistical procedures appropriate to the
  scale of the experiment (R=10 seeds, N_val=20 items per run on the
  binary EuroSAT subset). In this regime:
    - Wilson 95% CI on a single-run accuracy is reported as a descriptive
      uncertainty figure, but its half-width is large (~+/-10-15 pp at
      p_hat ~ 0.9) and it should not be over-interpreted.
    - Wilcoxon signed-rank PAIRED test on the R=10 across-seed differences
      between two architectures is the scientifically appropriate test in
      this setup: it does not assume normality, it pairs on the seed (so
      it cancels seed-induced variation), and at R=10 it has usable
      statistical power.
    - We do NOT compute McNemar's test on per-item discordant counts:
      with N_val=20 and accuracy ~ 0.9 the discordant counts are 2-3 in
      total and McNemar has essentially zero power.
    - Bootstrap percentile CI on the across-run mean accuracy is reported
      as a complement, again as a descriptive figure.

Reference: discussion at the end of Wave K Round 2, Cap.3 revision.
"""

from __future__ import annotations

import os
import csv
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Seed handling
# ---------------------------------------------------------------------------

SEED_LIST: list[int] = list(range(42, 52))   # 10 seeds: 42, 43, ..., 51


def set_global_seeds(seed: int) -> None:
    """Fix every RNG that touches a training run: PyTorch (incl. CUDA),
    NumPy, Python `random`, and the DataLoader workers.

    Relies on pytorch_lightning.seed_everything, which seeds:
      - Python `random` (used by EuroSATDataset.__init__ -> random.shuffle)
      - NumPy
      - Torch CPU and CUDA
    and, with workers=True, also seeds DataLoader workers via the
    `PL_GLOBAL_SEED` env var that PL propagates to torch.utils.data.
    """
    import pytorch_lightning as pl
    pl.seed_everything(seed, workers=True)


# ---------------------------------------------------------------------------
# Lightning callback: per-epoch metrics + per-item final predictions
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """All quantities saved from one (architecture, seed) run."""
    architecture: str
    seed: int
    n_val: int = 0
    train_loss_per_epoch: list[float] = field(default_factory=list)
    train_acc_per_epoch:  list[float] = field(default_factory=list)
    val_loss_per_epoch:   list[float] = field(default_factory=list)
    val_acc_per_epoch:    list[float] = field(default_factory=list)
    val_correct_final:    list[int]   = field(default_factory=list)   # per-item 0/1
    val_labels_final:     list[int]   = field(default_factory=list)


def _make_metrics_callback():
    """Build a Lightning callback that collects per-epoch train/val metrics
    and, at the final validation pass, the per-item correctness vector on
    the validation set.

    Imported lazily so that this module remains importable in environments
    where pytorch_lightning is not installed (e.g. during static analysis).
    """
    import pytorch_lightning as pl
    import torch

    class MetricsCollector(pl.Callback):

        def __init__(self, result: RunResult):
            super().__init__()
            self.result = result
            self._val_batch_correct: list[int] = []
            self._val_batch_labels:  list[int] = []

        # ----- per-epoch scalars (train/val loss/acc) -----
        def on_train_epoch_end(self, trainer, pl_module):
            cm = trainer.callback_metrics
            if 'train_loss_epoch' in cm:
                self.result.train_loss_per_epoch.append(float(cm['train_loss_epoch'].item()))
            elif 'train_loss' in cm:
                self.result.train_loss_per_epoch.append(float(cm['train_loss'].item()))
            if 'train_accuracy_epoch' in cm:
                self.result.train_acc_per_epoch.append(float(cm['train_accuracy_epoch'].item()))
            elif 'train_accuracy' in cm:
                self.result.train_acc_per_epoch.append(float(cm['train_accuracy'].item()))

        # ----- per-item validation collection -----
        def on_validation_epoch_start(self, trainer, pl_module):
            self._val_batch_correct = []
            self._val_batch_labels  = []

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            inputs, labels = batch
            with torch.no_grad():
                logits = pl_module(inputs)
                _, preds = torch.max(logits.detach(), dim=1)
            corr = (preds == labels).to(torch.int64).cpu().tolist()
            labs = labels.cpu().tolist()
            self._val_batch_correct.extend(int(c) for c in corr)
            self._val_batch_labels.extend(int(l) for l in labs)

        def on_validation_epoch_end(self, trainer, pl_module):
            cm = trainer.callback_metrics
            if 'val_loss' in cm:
                self.result.val_loss_per_epoch.append(float(cm['val_loss'].item()))
            if 'val_accuracy' in cm:
                self.result.val_acc_per_epoch.append(float(cm['val_accuracy'].item()))
            # Per-item buffer: overwrite at every epoch; at end of training
            # the latest one is the one referred to the last-epoch model.
            self.result.val_correct_final = list(self._val_batch_correct)
            self.result.val_labels_final  = list(self._val_batch_labels)
            self.result.n_val = len(self._val_batch_correct)

    return MetricsCollector


# ---------------------------------------------------------------------------
# Per-run training driver
# ---------------------------------------------------------------------------

def run_single_seed(
    model_factory: Callable[[], 'pl.LightningModule'],
    data_factory:  Callable[[], 'pl.LightningDataModule'],
    architecture: str,
    seed: int,
    max_epochs: int = 40,
    log_dir: Optional[str] = None,
    accelerator: str = 'cpu',
) -> RunResult:
    """Run one training under a fixed seed and collect the metrics.

    `model_factory` and `data_factory` are zero-argument callables that
    instantiate a fresh LightningModule and DataModule respectively. Both
    must be re-instantiated inside the function: re-using existing
    instances across seeds would silently break the seed-determinism
    contract (Lightning would carry initialised weights and shuffled
    datasets from the previous run).
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    set_global_seeds(seed)
    result = RunResult(architecture=architecture, seed=seed)

    model = model_factory()
    data_module = data_factory()

    MetricsCollector = _make_metrics_callback()
    metrics_cb = MetricsCollector(result)

    if log_dir is None:
        log_dir = os.path.join('multirun_logs', architecture, f'seed_{seed}')
    os.makedirs(log_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'ckpt'),
        filename=f'{architecture}_seed{seed}_best',
        monitor='val_loss',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[ckpt_cb, metrics_cb],
        logger=False,                # disabled to keep multi-run light
        accelerator=accelerator,
        enable_progress_bar=False,   # quieter loop over R=10 seeds
        deterministic=True,
    )
    trainer.fit(model, data_module)
    return result


def save_run_csvs(result: RunResult, output_dir: str) -> None:
    """Persist a RunResult to two CSVs:
       - {arch}/curves_seed{seed}.csv  (epoch, train_loss, train_acc, val_loss, val_acc)
       - {arch}/predictions_seed{seed}.csv  (item_idx, label, correct)
    """
    arch_dir = os.path.join(output_dir, result.architecture)
    os.makedirs(arch_dir, exist_ok=True)

    curves_path = os.path.join(arch_dir, f'curves_seed{result.seed}.csv')
    with open(curves_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        n = max(len(result.train_loss_per_epoch),
                len(result.val_loss_per_epoch))
        for e in range(n):
            row = [
                e + 1,
                _safe_get(result.train_loss_per_epoch, e),
                _safe_get(result.train_acc_per_epoch,  e),
                _safe_get(result.val_loss_per_epoch,   e),
                _safe_get(result.val_acc_per_epoch,    e),
            ]
            w.writerow(row)

    preds_path = os.path.join(arch_dir, f'predictions_seed{result.seed}.csv')
    with open(preds_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['item_idx', 'label', 'correct'])
        for i, (lab, corr) in enumerate(zip(result.val_labels_final,
                                            result.val_correct_final)):
            w.writerow([i, lab, corr])


def _safe_get(lst, i):
    try:
        return lst[i]
    except IndexError:
        return ''


# ---------------------------------------------------------------------------
# Multi-seed driver
# ---------------------------------------------------------------------------

def run_multiseed(
    model_factory: Callable[[], 'pl.LightningModule'],
    data_factory:  Callable[[], 'pl.LightningDataModule'],
    architecture: str,
    output_dir: str = 'multirun_csv',
    n_runs: int = 10,
    max_epochs: int = 40,
    accelerator: str = 'cpu',
    seed_list: Optional[list[int]] = None,
) -> list[RunResult]:
    """Loop over the seed list and save one curve-CSV + one predictions-CSV
    per (architecture, seed). Returns the list of RunResult objects so the
    caller can also keep them in memory for further analysis."""
    if seed_list is None:
        seed_list = SEED_LIST[:n_runs]
    results: list[RunResult] = []
    for k, seed in enumerate(seed_list):
        print(f'[multirun] {architecture}: seed {seed} ({k+1}/{len(seed_list)}) ...', flush=True)
        res = run_single_seed(
            model_factory=model_factory,
            data_factory=data_factory,
            architecture=architecture,
            seed=seed,
            max_epochs=max_epochs,
            accelerator=accelerator,
        )
        save_run_csvs(res, output_dir)
        results.append(res)
        final_acc = res.val_acc_per_epoch[-1] if res.val_acc_per_epoch else float('nan')
        print(f'[multirun] {architecture} seed {seed}: final val_acc = {final_acc:.4f}',
              flush=True)
    return results


# ---------------------------------------------------------------------------
# Aggregation from CSV
# ---------------------------------------------------------------------------

@dataclass
class AggregatedRuns:
    architecture: str
    n_runs: int
    n_epochs: int
    train_loss: np.ndarray   # shape (n_runs, n_epochs)
    train_acc:  np.ndarray
    val_loss:   np.ndarray
    val_acc:    np.ndarray
    final_val_acc: np.ndarray   # shape (n_runs,)
    n_val: int
    val_correct_per_run: np.ndarray  # shape (n_runs, n_val), per-item 0/1


def load_aggregated(csv_dir: str, architecture: str) -> AggregatedRuns:
    """Load all CSVs produced by run_multiseed for a single architecture
    and return the aggregated arrays in seed-sorted order."""
    arch_dir = os.path.join(csv_dir, architecture)
    if not os.path.isdir(arch_dir):
        raise FileNotFoundError(f'no run directory for architecture {architecture!r}: {arch_dir}')

    curve_files = sorted(f for f in os.listdir(arch_dir) if f.startswith('curves_seed'))
    if not curve_files:
        raise FileNotFoundError(f'no curve CSVs in {arch_dir}')

    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    for fn in curve_files:
        rows = list(_read_csv(os.path.join(arch_dir, fn)))
        tl, ta, vl, va = [], [], [], []
        for r in rows:
            tl.append(_to_float(r['train_loss']))
            ta.append(_to_float(r['train_accuracy']))
            vl.append(_to_float(r['val_loss']))
            va.append(_to_float(r['val_accuracy']))
        train_loss.append(tl); train_acc.append(ta)
        val_loss.append(vl);   val_acc.append(va)

    # Truncate to common epoch count (in case of an interrupted seed)
    n_epochs = min(len(x) for x in val_acc)
    train_loss = np.array([x[:n_epochs] for x in train_loss], dtype=float)
    train_acc  = np.array([x[:n_epochs] for x in train_acc],  dtype=float)
    val_loss   = np.array([x[:n_epochs] for x in val_loss],   dtype=float)
    val_acc    = np.array([x[:n_epochs] for x in val_acc],    dtype=float)

    final_val_acc = val_acc[:, -1]

    # Per-item predictions
    pred_files = sorted(f for f in os.listdir(arch_dir) if f.startswith('predictions_seed'))
    val_correct_per_run = []
    for fn in pred_files:
        rows = list(_read_csv(os.path.join(arch_dir, fn)))
        val_correct_per_run.append([int(r['correct']) for r in rows])
    n_val = min(len(v) for v in val_correct_per_run) if val_correct_per_run else 0
    val_correct_per_run = np.array([v[:n_val] for v in val_correct_per_run], dtype=int) \
                          if val_correct_per_run else np.zeros((train_loss.shape[0], 0), dtype=int)

    return AggregatedRuns(
        architecture=architecture,
        n_runs=train_loss.shape[0],
        n_epochs=n_epochs,
        train_loss=train_loss, train_acc=train_acc,
        val_loss=val_loss,     val_acc=val_acc,
        final_val_acc=final_val_acc,
        n_val=n_val,
        val_correct_per_run=val_correct_per_run,
    )


def _read_csv(path):
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            yield row


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Statistics: Wilson, Wilcoxon, bootstrap
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k / n at confidence
    level 1 - alpha. Returned as a (lower, upper) tuple, clipped to [0, 1].
    Reported here as a descriptive single-run uncertainty; with N_val ~ 20
    the half-width is large and the interval should not be used to
    discriminate between architectures.
    """
    if n <= 0:
        return (0.0, 0.0)
    # 1 - alpha/2 quantile of N(0, 1); use scipy if available, else a
    # closed-form approximation good to ~5e-4 in the range alpha in [0.001, 0.5].
    try:
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - alpha / 2.0))
    except ImportError:
        z = _inv_normal_approx(1.0 - alpha / 2.0)
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _inv_normal_approx(p: float) -> float:
    """Inverse standard-normal CDF (Beasley-Springer-Moro coarse approx)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    q = p - 0.5
    if abs(q) <= 0.425:
        r = q * q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        den = ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0
        return num / den
    # tails: cruder
    r = math.sqrt(-math.log(min(p, 1.0 - p)))
    return (math.copysign(1.0, q)) * r


def wilcoxon_paired(acc_A: np.ndarray, acc_B: np.ndarray,
                    alternative: str = 'two-sided') -> dict:
    """Wilcoxon signed-rank paired test on the across-seed accuracy
    differences A_i - B_i, i = 1, ..., R. Paired on the seed, so any
    seed-induced variation cancels.

    Returns a dict with statistic, p-value, the differences vector, the
    number of non-zero differences (which scipy uses for the exact distr.)
    and a human-readable interpretation.
    """
    from scipy.stats import wilcoxon
    acc_A = np.asarray(acc_A, dtype=float)
    acc_B = np.asarray(acc_B, dtype=float)
    if acc_A.shape != acc_B.shape:
        raise ValueError(f'shape mismatch: {acc_A.shape} vs {acc_B.shape}')
    diffs = acc_A - acc_B
    n_nonzero = int(np.sum(diffs != 0))
    # With R=10, scipy uses the exact distribution by default when no
    # method is forced; we make it explicit here so it does not switch
    # to the approximation under future scipy versions.
    try:
        res = wilcoxon(acc_A, acc_B, alternative=alternative,
                       zero_method='wilcox', method='exact')
    except TypeError:
        # older scipy may not accept method='exact'
        res = wilcoxon(acc_A, acc_B, alternative=alternative,
                       zero_method='wilcox')
    return {
        'statistic': float(res.statistic),
        'p_value':   float(res.pvalue),
        'differences': diffs.tolist(),
        'n_pairs': int(acc_A.size),
        'n_nonzero': n_nonzero,
        'alternative': alternative,
    }


def bootstrap_ci_mean(values: np.ndarray,
                      n_resamples: int = 10000,
                      alpha: float = 0.05,
                      seed: int = 0) -> dict:
    """Percentile bootstrap CI on the mean of `values`. With R=10 seeds
    this is a coarse but useful descriptive figure for the mean
    across-run accuracy. Returned as a dict with 'mean', 'ci_low', 'ci_high'.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = values.size
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = sample.mean()
    lo, hi = np.percentile(boots, [100.0 * alpha / 2.0,
                                   100.0 * (1.0 - alpha / 2.0)])
    return {
        'mean':    float(values.mean()),
        'std':     float(values.std(ddof=1)) if n > 1 else float('nan'),
        'ci_low':  float(lo),
        'ci_high': float(hi),
        'alpha':   alpha,
        'n':       int(n),
    }


# ---------------------------------------------------------------------------
# Plotting with variance band
# ---------------------------------------------------------------------------

def plot_with_band(
    ax,
    x: np.ndarray,
    series: np.ndarray,         # shape (n_runs, n_x)
    color: str = 'C0',
    label: Optional[str] = None,
    band_alpha: float = 0.22,
    line_kwargs: Optional[dict] = None,
    band_label: Optional[str] = None,
):
    """Plot the across-run mean of `series` as a solid line in `color`,
    with the +/- 1 std band drawn as a fill_between in the same color
    but with reduced alpha (which matplotlib composites as a paler tint).

    `series` is expected to be shaped (n_runs, n_x). The function does
    not call ax.legend(); the caller is responsible for that, so the same
    helper can be reused across panels and metrics.
    """
    series = np.asarray(series, dtype=float)
    mean = np.nanmean(series, axis=0)
    std  = np.nanstd(series,  axis=0, ddof=1) if series.shape[0] > 1 else np.zeros_like(mean)
    lkw = {'linewidth': 1.8}
    if line_kwargs:
        lkw.update(line_kwargs)
    line, = ax.plot(x, mean, color=color, label=label, **lkw)
    ax.fill_between(x, mean - std, mean + std,
                    color=color, alpha=band_alpha, linewidth=0,
                    label=band_label)
    return line, mean, std


def plot_single_architecture_with_std(
    aggregated: AggregatedRuns,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Reproduce the *_with_std.png plot used in Cap.3 (training-and-validation
    accuracy vs epoch), now backed by the actual R=10 runs and with the
    variance band drawn in a paler tint of the line color.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, aggregated.n_epochs + 1)

    plot_with_band(ax, epochs, aggregated.train_acc,
                   color='#e08214', label='Train Accuracy',
                   band_label=r'$\pm 1\sigma$ (train)')
    plot_with_band(ax, epochs, aggregated.val_acc,
                   color='#2166ac', label='Validation Accuracy',
                   band_label=r'$\pm 1\sigma$ (val)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(title or f'Accuracy vs Epoch — {aggregated.architecture.upper()}')
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.legend(loc='lower right', fontsize=9, ncol=2)

    fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=160)
    return fig, ax


def plot_three_architectures(
    aggregated_by_arch: dict[str, AggregatedRuns],
    output_path: Optional[str] = None,
    title: str = 'Validation accuracy vs epoch — three architectures',
):
    """One-panel comparison of the three architectures, mean + variance band
    on validation accuracy. The colours are colour-blind safe; the band
    alpha matches the single-architecture plot."""
    import matplotlib.pyplot as plt
    palette = {
        'qcnn':   '#2166ac',
        'ccnn':   '#b2182b',
        'pure_q': '#1a9850',
    }
    label_of = {
        'qcnn':   'Hybrid Q-CNN',
        'ccnn':   'Classical CNN (ablation)',
        'pure_q': 'Pure quantum (Sebastianelli)',
    }
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for arch, agg in aggregated_by_arch.items():
        epochs = np.arange(1, agg.n_epochs + 1)
        plot_with_band(ax, epochs, agg.val_acc,
                       color=palette.get(arch, 'C0'),
                       label=label_of.get(arch, arch))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation accuracy')
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=160)
    return fig, ax


# ---------------------------------------------------------------------------
# Convenience: one-shot summary table for the manuscript
# ---------------------------------------------------------------------------

def summary_table(aggregated_by_arch: dict[str, AggregatedRuns]) -> str:
    """Return a plain-text table summarising, for each architecture:
       mean +/- std of the final-epoch validation accuracy across R runs,
       bootstrap 95% CI on that mean, and the Wilson 95% CI on the
       seed-by-seed accuracy (median over seeds, as a descriptive single-
       run uncertainty figure).
    """
    lines = []
    header = f'{"architecture":>14s}  {"R":>3s}  {"N_val":>5s}  ' \
             f'{"mean":>8s}  {"std":>8s}  {"boot 95% CI":>22s}  {"Wilson 95% (median)":>22s}'
    lines.append(header)
    lines.append('-' * len(header))
    for arch, agg in aggregated_by_arch.items():
        final = agg.final_val_acc
        boot = bootstrap_ci_mean(final)
        # median Wilson CI across seeds
        wilsons_lo, wilsons_hi = [], []
        for k in (final * agg.n_val).round().astype(int):
            lo, hi = wilson_ci(int(k), agg.n_val)
            wilsons_lo.append(lo); wilsons_hi.append(hi)
        med_lo, med_hi = float(np.median(wilsons_lo)), float(np.median(wilsons_hi))
        lines.append(
            f'{arch:>14s}  {agg.n_runs:>3d}  {agg.n_val:>5d}  '
            f'{boot["mean"]:>8.4f}  {boot["std"]:>8.4f}  '
            f'[{boot["ci_low"]:.4f}, {boot["ci_high"]:.4f}]  '
            f'[{med_lo:.4f}, {med_hi:.4f}]'
        )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Sanity self-test (does not require pytorch_lightning)
# ---------------------------------------------------------------------------

def _self_test():
    # Wilson
    lo, hi = wilson_ci(18, 20)
    assert 0.65 < lo < 0.75 and 0.97 < hi < 0.99, (lo, hi)
    # Bootstrap
    rng = np.random.default_rng(0)
    vals = rng.normal(loc=0.9, scale=0.02, size=10)
    res = bootstrap_ci_mean(vals, seed=1)
    assert res['ci_low'] < res['mean'] < res['ci_high']
    # Wilcoxon (if scipy is available)
    try:
        from scipy.stats import wilcoxon as _w  # noqa: F401
        out = wilcoxon_paired(np.array([0.90, 0.92, 0.91, 0.93, 0.89,
                                        0.92, 0.91, 0.90, 0.93, 0.92]),
                              np.array([0.88, 0.89, 0.90, 0.90, 0.87,
                                        0.89, 0.88, 0.87, 0.91, 0.89]))
        assert out['p_value'] < 0.05, out
    except ImportError:
        pass
    print('multirun.py self-test OK')


if __name__ == '__main__':
    _self_test()
