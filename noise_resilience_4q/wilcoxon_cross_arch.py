#!/usr/bin/env python3
"""wilcoxon_cross_arch.py — Cross-architecture statistical test for Wave-K.

Compares R identical seeds across up to 4 architectures:
  - 4q noiseless (Aer statevector, local Mac run)
  - 4q noisy (Aer density_matrix + ibm_fez restricted, Levante run)
  - 9q noiseless (Aer statevector, existing multiseed run)
  - 9q noisy projected (output of project_4q_to_9q.py, optional)

For each pair of architectures (i, j) it computes:
  - Wilcoxon signed-rank paired test (scipy.stats.wilcoxon)
  - Mean difference (mean Δ) and Cohen's d_z paired effect size
  - Significance at level α=0.05

For each architecture it computes a Wilson 95% CI on the mean accuracy via
Normal-approximation sampling (R≥10 seeds → Wilson is fine).

Output
------
- Formatted table printed to stdout
- (optional) JSON with all results
- (optional) violin plot + mean±CI per architecture

Usage
-----
  python wilcoxon_cross_arch.py \\
      --runs 4q_noiseless:./results_4q_noiseless \\
             4q_noisy:./results_4q_noisy \\
             9q_noiseless:./results_9q_noiseless \\
      [--projection summary_projection.json] \\
      [--out results_wilcoxon.json] \\
      [--plot results_wilcoxon.png]
"""

from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.stats import wilcoxon, norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available — some tests will be skipped", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Result loading
# ─────────────────────────────────────────────────────────────────────────────

def load_arch_results(directory: str,
                      metric: str = 'final_val_acc') -> dict:
    """Loads results for an architecture from a directory of JSON files.

    Returns {seed: metric}. Expects files `results_seed_NNN.json`, each
    with keys `seed` and the desired `metric`.
    """
    out = {}
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"Directory non trovata: {directory}")
    for p in sorted(d.glob('results_seed_*.json')):
        with open(p) as f:
            r = json.load(f)
        if 'seed' not in r or metric not in r:
            print(f"  ⚠️  {p}: missing 'seed' or '{metric}', skip")
            continue
        out[int(r['seed'])] = float(r[metric])
    if not out:
        raise RuntimeError(f"No valid results found in {directory}")
    return out


def load_projection(path: str) -> dict:
    """Loads the summary from project_4q_to_9q.py to add the projected points."""
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Wilson 95% CI for mean accuracy (Normal approximation; R seeds → ok)
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(accs: np.ndarray, conf: float = 0.95) -> tuple:
    """Wilson confidence interval on the mean (over the seed sample).

    For R seeds with final_val_acc, the CI is on the mean. We use the
    Normal approximation (Student-t for small R); for the classical
    Wilson score on a binomial proportion one would use `proportion_confint`,
    but here the accuracies are already means over the val set, so what is more appropriate
    is a CI on the mean via a t-test.
    """
    R = len(accs)
    mean = float(accs.mean())
    se = float(accs.std(ddof=1) / np.sqrt(R)) if R > 1 else 0.0
    if HAS_SCIPY and R > 1:
        from scipy.stats import t
        tcrit = t.ppf(1 - (1-conf)/2, df=R-1)
        return mean - tcrit*se, mean, mean + tcrit*se
    # Fallback Normal
    zcrit = 1.96 if conf == 0.95 else norm.ppf(1 - (1-conf)/2)
    return mean - zcrit*se, mean, mean + zcrit*se


# ─────────────────────────────────────────────────────────────────────────────
# Paired Wilcoxon e effect size
# ─────────────────────────────────────────────────────────────────────────────

def compare_two_archs(a_dict: dict, b_dict: dict,
                      min_paired: int = 5) -> dict:
    """Statistical comparison between 2 architectures, automatically paired or unpaired.

    Strategy:
      - If # common seeds >= min_paired -> paired Wilcoxon (more powerful,
        controls the initialisation-driven variance)
      - Otherwise -> unpaired Mann-Whitney U on all seeds of each architecture

    a_dict, b_dict: {seed: accuracy} for each of the 2 architectures
    """
    common = sorted(set(a_dict.keys()) & set(b_dict.keys()))
    if len(common) >= min_paired:
        # PAIRED Wilcoxon on the common seeds
        a = np.array([a_dict[s] for s in common])
        b = np.array([b_dict[s] for s in common])
        diff = a - b
        out = {
            'test_type': 'wilcoxon_paired',
            'n_pairs':   len(common),
            'common_seeds': common,
            'mean_a': float(a.mean()), 'std_a': float(a.std(ddof=1)),
            'mean_b': float(b.mean()), 'std_b': float(b.std(ddof=1)),
            'mean_diff': float(diff.mean()),
            'std_diff':  float(diff.std(ddof=1)),
        }
        out['effect_size_dz'] = (out['mean_diff'] / out['std_diff']
                                  if out['std_diff'] > 0 else np.nan)
        if HAS_SCIPY:
            try:
                w, p = wilcoxon(a, b, zero_method='wilcox',
                                alternative='two-sided')
                out['statistic'] = float(w)
                out['p_value'] = float(p)
                out['significant_alpha_005'] = bool(p < 0.05)
            except ValueError as e:
                out['statistic'] = None
                out['p_value'] = None
                out['significant_alpha_005'] = False
                out['note'] = str(e)
    else:
        # UNPAIRED Mann-Whitney U (Wilcoxon rank-sum) on all seeds
        a = np.array(list(a_dict.values()))
        b = np.array(list(b_dict.values()))
        out = {
            'test_type': 'mann_whitney_u_unpaired',
            'n_a': len(a), 'n_b': len(b),
            'common_seeds': common,
            'mean_a': float(a.mean()), 'std_a': float(a.std(ddof=1)),
            'mean_b': float(b.mean()), 'std_b': float(b.std(ddof=1)),
            'mean_diff': float(a.mean() - b.mean()),
            'std_diff':  float(np.sqrt(a.var(ddof=1)/len(a)
                                       + b.var(ddof=1)/len(b))),
        }
        # Cohen's d for independent groups (pooled std)
        pooled_var = ((len(a)-1)*a.var(ddof=1)
                      + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2)
        out['effect_size_d'] = (out['mean_diff'] / np.sqrt(pooled_var)
                                 if pooled_var > 0 else np.nan)
        if HAS_SCIPY:
            try:
                from scipy.stats import mannwhitneyu
                u, p = mannwhitneyu(a, b, alternative='two-sided')
                out['statistic'] = float(u)
                out['p_value'] = float(p)
                out['significant_alpha_005'] = bool(p < 0.05)
            except ValueError as e:
                out['statistic'] = None
                out['p_value'] = None
                out['significant_alpha_005'] = False
                out['note'] = str(e)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_arch_arg(s: str) -> tuple:
    """Parser per --runs <label>:<directory>."""
    if ':' not in s:
        raise argparse.ArgumentTypeError(
            f"Bad format: '{s}'. Use <label>:<directory>")
    label, directory = s.split(':', 1)
    return label.strip(), directory.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', type=parse_arch_arg, required=True,
                    metavar='LABEL:DIR',
                    help='List of architectures, e.g. 4q_noisy:./results_4q_noisy')
    ap.add_argument('--metric', default='final_val_acc',
                    help='Metric to extract from the JSON files (default: final_val_acc)')
    ap.add_argument('--projection', default=None,
                    help='(Optional) summary_projection.json to add 9q_noisy_projected')
    ap.add_argument('--out', default=None,
                    help='(Optional) save the results to JSON')
    ap.add_argument('--plot', default=None,
                    help='(Optional) save a violin + mean±CI plot to PNG')
    args = ap.parse_args()

    # 1. Load per architecture
    arch_data = {}
    for label, directory in args.runs:
        accs_dict = load_arch_results(directory, metric=args.metric)
        arch_data[label] = accs_dict
        print(f"  [{label}] R={len(accs_dict)} seeds from {directory}")

    # 2. Add projected points if requested (NOT paired, plottable only
    #    as mean +/- CI)
    projection = None
    if args.projection:
        projection = load_projection(args.projection)
        print(f"\n  Proiezione 9q-noisy caricata da {args.projection}:")
        print(f"    Accuracy 9q-noisy projected: "
              f"{projection.get('acc_9q_noisy_projected_mean', 'N/A')}")

    # 3. Identify ALL seeds present in any architecture
    all_seeds_per_arch = {label: set(d.keys()) for label, d in arch_data.items()}
    n_common_all = len(set.intersection(*all_seeds_per_arch.values())) \
                   if arch_data else 0
    print(f"\n  Seeds in common across ALL architectures: {n_common_all}")
    if n_common_all < 2:
        print(f"  WARNING: only {n_common_all} seeds in common across all architectures: "
              f"some pairs will use unpaired (Mann-Whitney) tests instead of "
              f"paired (Wilcoxon).")

    # 4. Wilson CI per architecture (over all seeds, not only the common ones)
    print(f"\n{'='*70}")
    print(f"  Wilson 95% CI per architecture (metric: {args.metric})")
    print(f"{'='*70}")
    print(f"  {'arch':<28} {'R':>3} {'mean':>8} {'std':>8} "
          f"{'95% CI lo':>10} {'95% CI hi':>10}")
    print("  " + "-"*70)
    ci_table = {}
    for label, accs in arch_data.items():
        arr = np.array([accs[s] for s in sorted(accs.keys())])
        lo, mean, hi = wilson_ci(arr)
        ci_table[label] = {'R': len(arr), 'mean': mean, 'std': float(arr.std(ddof=1)),
                           'ci_lo': lo, 'ci_hi': hi,
                           'per_seed': {int(s): accs[s] for s in sorted(accs.keys())}}
        print(f"  {label:<28} {len(arr):>3d} {mean:>8.4f} {arr.std(ddof=1):>8.4f} "
              f"{lo:>10.4f} {hi:>10.4f}")
    if projection:
        print(f"  {'9q_noisy_projected':<28} {'--':>3} "
              f"{projection['acc_9q_noisy_projected_mean']:>8.4f} {'--':>8} "
              f"{projection['acc_9q_noisy_projected_ci_lo']:>10.4f} "
              f"{projection['acc_9q_noisy_projected_ci_hi']:>10.4f}")

    # 5. Confronto a coppie (paired o unpaired automaticamente)
    print(f"\n{'='*78}")
    print(f"  Pairwise comparisons across architectures")
    print(f"{'='*78}")
    pair_results = {}
    labels = list(arch_data.keys())
    print(f"  {'pair':<48} {'test':<10} {'Δmean':>9} {'eff':>7} {'p-val':>9} {'sig':>4}")
    print("  " + "-"*78)
    for la, lb in itertools.combinations(labels, 2):
        res = compare_two_archs(arch_data[la], arch_data[lb])
        pair_label = f"{la}  vs  {lb}"
        pair_results[pair_label] = res
        sig = '✓' if res.get('significant_alpha_005') else '·'
        pv = res.get('p_value')
        pv_s = f"{pv:.4f}" if pv is not None else "—"
        eff_key = 'effect_size_dz' if res['test_type'] == 'wilcoxon_paired' \
                  else 'effect_size_d'
        eff = res.get(eff_key)
        eff_s = (f"{eff:+.2f}" if eff is not None and not np.isnan(eff)
                 else "—")
        test_short = ('pair' if res['test_type'] == 'wilcoxon_paired'
                       else 'unpair')
        n_info = (f"R={res['n_pairs']}" if res['test_type'] == 'wilcoxon_paired'
                   else f"{res['n_a']}vs{res['n_b']}")
        print(f"  {pair_label:<48} {test_short:<5}{n_info:<5} "
              f"{res['mean_diff']:>+9.4f} {eff_s:>7} {pv_s:>9} {sig:>4}")

    # 6. Optional plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            data_to_plot = []
            xlabels = []
            for label in labels:
                arr = np.array([arch_data[label][s] for s in sorted(arch_data[label])])
                data_to_plot.append(arr)
                xlabels.append(f"{label}\n(R={len(arr)})")
            parts = ax.violinplot(data_to_plot, showmedians=True, showmeans=False)
            ax.set_xticks(range(1, len(labels)+1))
            ax.set_xticklabels(xlabels, fontsize=9)
            ax.set_ylabel(args.metric)
            ax.set_title(f"Cross-architecture comparison ({args.metric})")
            # Add mean + CI ticks
            for i, (label, arr) in enumerate(zip(labels, data_to_plot), start=1):
                lo, mean, hi = wilson_ci(arr)
                ax.errorbar(i, mean, yerr=[[mean-lo], [hi-mean]],
                            fmt='o', color='red', capsize=4,
                            label='mean ± 95% CI' if i == 1 else None)
            # Proiezione 9q noisy se presente (solo punto + CI, no violin)
            if projection:
                x = len(labels) + 1
                m = projection['acc_9q_noisy_projected_mean']
                lo = projection['acc_9q_noisy_projected_ci_lo']
                hi = projection['acc_9q_noisy_projected_ci_hi']
                ax.errorbar(x, m, yerr=[[m-lo], [hi-m]], fmt='s',
                            color='purple', capsize=4,
                            label='9q noisy projected')
                ax.set_xticks(list(range(1, len(labels)+1)) + [x])
                ax.set_xticklabels(xlabels + ['9q noisy\n(projected)'], fontsize=9)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.plot, dpi=120)
            print(f"\n  Plot saved: {args.plot}")
        except ImportError:
            print(f"\n  ⚠️  matplotlib not available, plot skipped")

    # 7. Save
    if args.out:
        summary = {
            'metric': args.metric,
            'architectures': ci_table,
            'pair_tests': pair_results,
            'n_seeds_common_to_all': n_common_all,
        }
        if projection:
            summary['projection_9q_noisy'] = {
                'mean': projection['acc_9q_noisy_projected_mean'],
                'ci_lo': projection['acc_9q_noisy_projected_ci_lo'],
                'ci_hi': projection['acc_9q_noisy_projected_ci_hi'],
            }
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  ✓ Saved: {args.out}")


if __name__ == '__main__':
    main()
