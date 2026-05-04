#!/usr/bin/env python3
"""
Visualise Pareto sweep results from results.json produced by train.py.

Writes publication-ready PDF + PNG figures to <results_dir>/figures/:

  pareto_frontier_{best,final}.pdf  - coherence vs nDCG scatter, λ-coloured,
                                       error bars on both axes (main result)
  metric_sweep_{best,final}.pdf     - 3-panel: nDCG@k / coherence / grad norm
                                       vs λ, with ±1 std dev bands
  training_curves.pdf               - val nDCG and coherence over epochs for
                                       a representative subset of λ values

All figures target a two-column IEEE/ACM paper (double column = 7.16 in).

Usage:
    python scripts/plot_sweep.py --results_dir saved_models/final_sweep
    python scripts/plot_sweep.py --results_dir saved_models/final_sweep \\
        --ckpt best --select_lambdas 0 1 5 15 50
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


SINGLE_COL = 3.5
DOUBLE_COL = 7.16

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.framealpha': 0.92,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.35,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Colorblind-friendly palette for k-value lines (red, blue, green, purple)
K_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']


def load_results(results_dir: Path):
    path = results_dir / 'results.json'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found')
    with open(path) as f:
        d = json.load(f)
    return d['config'], d['runs']


def _save(fig, base: Path) -> None:
    for suffix in ('.pdf', '.png'):
        fig.savefig(base.with_suffix(suffix))
    plt.close(fig)
    print(f'  saved {base}.pdf / .png')


def _extract(runs: list, k_values: list[int], ckpt_key: str) -> dict:
    """Return per-lambda mean/std arrays for test metrics and grad norm."""
    lambdas = np.array([r['coherence_weight'] for r in runs])

    ndcg_m: dict[int, list] = {k: [] for k in k_values}
    ndcg_s: dict[int, list] = {k: [] for k in k_values}
    coh_m, coh_s = [], []
    ce_m, ce_s = [], []
    gn_m, gn_s = [], []

    for r in runs:
        mean = r['mean'][ckpt_key]
        std = r['std'][ckpt_key]

        for k in k_values:
            key = f'test_ndcg_{k}'
            ndcg_m[k].append(mean.get(key, np.nan))
            ndcg_s[k].append(std.get(key, np.nan))
        coh_m.append(mean.get('test_eval_coherence', np.nan))
        coh_s.append(std.get('test_eval_coherence', np.nan))
        ce_m.append(mean.get('test_ce_loss', np.nan))
        ce_s.append(std.get('test_ce_loss', np.nan))

        per_seed = [
            float(np.median([e['train_grad_norm'] for e in sr['epochs']]))
            for sr in r['seed_runs']
        ]
        gn_m.append(np.mean(per_seed))
        gn_s.append(np.std(per_seed))

    return {
        'lambdas': lambdas,
        'ndcg_m': {k: np.array(v) for k, v in ndcg_m.items()},
        'ndcg_s': {k: np.array(v) for k, v in ndcg_s.items()},
        'coh_m': np.array(coh_m), 'coh_s': np.array(coh_s),
        'ce_m': np.array(ce_m), 'ce_s': np.array(ce_s),
        'gn_m': np.array(gn_m), 'gn_s': np.array(gn_s),
    }


def _lambda_xaxis(ax, lambdas: np.ndarray) -> None:
    """Symlog x-axis so λ=0 sits at the origin and the rest are log-spaced."""
    nonzero = lambdas[lambdas > 0]
    linthresh = float(nonzero.min()) / 2.0 if len(nonzero) else 0.05
    ax.set_xscale('symlog', linthresh=linthresh)
    ax.set_xlim(left=-linthresh * 0.5, right=float(lambdas.max()) * 1.15)
    ax.set_xlabel(r'Coherence weight $\lambda$')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))


def fig_pareto(runs, k_values, ckpt_key, n_seeds, fig_dir, tag):
    primary_k = max(k_values)
    d = _extract(runs, k_values, ckpt_key)
    lambdas = d['lambdas']
    x, y = d['coh_m'], d['ndcg_m'][primary_k]
    xe, ye = d['coh_s'], d['ndcg_s'][primary_k]

    baseline = lambdas == 0
    nonzero = ~baseline

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.35))

    ax.plot(x, y, color='#bbbbbb', linewidth=0.9, zorder=1)

    ax.errorbar(x, y, xerr=xe, yerr=ye,
                fmt='none', ecolor='#cccccc', elinewidth=0.8, capsize=2.5, zorder=2)

    if baseline.any():
        ax.scatter(x[baseline], y[baseline],
                   s=90, marker='*', color='#1a1a2e', zorder=5,
                   edgecolors='white', linewidths=0.5,
                   label=r'$\lambda = 0$ (CE-only)')

    if nonzero.any():
        ax.scatter(x[nonzero], y[nonzero],
                   color='#4c72b0', s=48, zorder=4,
                   edgecolors='white', linewidths=0.5)

        for xi, yi, lam in zip(x[nonzero], y[nonzero], lambdas[nonzero]):
            ax.annotate(rf'$\lambda={lam:g}$', (xi, yi),
                        xytext=(6, 4), textcoords='offset points', fontsize=7)

    title_str = 'best val ckpt' if ckpt_key == 'test_best' else 'final epoch'
    ax.set_xlabel('Eval coherence')
    ax.set_ylabel(f'nDCG@{primary_k}')
    ax.set_title(
        f'Pareto frontier ({title_str})\nmean ± std over {n_seeds} seeds',
        fontsize=8,
    )
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, which='both')
    fig.tight_layout()
    _save(fig, fig_dir / f'pareto_frontier_{tag}')


def fig_metric_sweep(runs, k_values, ckpt_key, n_seeds, fig_dir, tag):
    d = _extract(runs, k_values, ckpt_key)
    lambdas = d['lambdas']

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.75))

    ax = axes[0]
    for i, k in enumerate(sorted(k_values)):
        c = K_COLORS[i % len(K_COLORS)]
        mu, sig = d['ndcg_m'][k], d['ndcg_s'][k]
        ax.plot(lambdas, mu, color=c, label=f'@{k}')
        ax.fill_between(lambdas, mu - sig, mu + sig, alpha=0.18, color=c)
    _lambda_xaxis(ax, lambdas)
    ax.set_ylabel('nDCG@k')
    ax.set_title(r'Accuracy vs. $\lambda$')
    ax.legend(title='k', ncol=2, fontsize=7)
    ax.grid(True, which='both')

    ax = axes[1]
    c = '#2ca02c'
    mu, sig = d['coh_m'], d['coh_s']
    ax.plot(lambdas, mu, color=c)
    ax.fill_between(lambdas, mu - sig, mu + sig, alpha=0.20, color=c)
    _lambda_xaxis(ax, lambdas)
    ax.set_ylabel('Eval coherence')
    ax.set_title(r'Coherence vs. $\lambda$')
    ax.grid(True, which='both')

    ax = axes[2]
    c = '#9467bd'
    mu, sig = d['gn_m'], d['gn_s']
    ax.plot(lambdas, mu, color=c)
    ax.errorbar(lambdas, mu, yerr=sig,
                fmt='none', ecolor=c, alpha=0.65, elinewidth=0.8, capsize=2.5)
    _lambda_xaxis(ax, lambdas)
    ax.set_ylabel('Median grad norm (pre-clip)')
    ax.set_title(r'Gradient norm vs. $\lambda$')
    ax.grid(True, which='both')

    title_str = 'best val ckpt' if ckpt_key == 'test_best' else 'final epoch'
    fig.suptitle(
        f'Sweep metrics - {title_str} (mean ± std, {n_seeds} seeds)',
        fontsize=9, y=1.03,
    )
    fig.tight_layout()
    _save(fig, fig_dir / f'metric_sweep_{tag}')


def fig_training_curves(runs, k_values, n_seeds, fig_dir,
                        select_lambdas: list[float] | None = None):
    primary_k = max(k_values)
    lambdas = [r['coherence_weight'] for r in runs]
    n = len(runs)

    if select_lambdas is not None:
        indices = [i for i, lam in enumerate(lambdas) if lam in select_lambdas]
    else:
        step = max((n - 1) / 5.0, 1.0)
        indices = sorted({0, n - 1} | {round(step * i) for i in range(1, 5)})

    try:
        cmap = matplotlib.colormaps['tab10']
    except AttributeError:
        cmap = plt.get_cmap('tab10')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9))

    for plot_i, run_i in enumerate(indices):
        run = runs[run_i]
        lam = lambdas[run_i]
        color = cmap(plot_i / max(len(indices) - 1, 1))
        label = rf'$\lambda={lam:g}$'

        n_epochs = len(run['seed_runs'][0]['epochs'])
        epochs = np.arange(1, n_epochs + 1)

        ndcg_arr = np.array([
            [e[f'val_ndcg_{primary_k}'] for e in sr['epochs']]
            for sr in run['seed_runs']
        ])
        coh_arr = np.array([
            [e['val_eval_coherence'] for e in sr['epochs']]
            for sr in run['seed_runs']
        ])

        for arr, ax in ((ndcg_arr, ax1), (coh_arr, ax2)):
            mu = arr.mean(axis=0)
            sig = arr.std(axis=0)
            ax.plot(epochs, mu, color=color, label=label)
            ax.fill_between(epochs, mu - sig, mu + sig, alpha=0.15, color=color)

    for ax, ylabel, title in (
        (ax1, f'Val nDCG@{primary_k}', f'Val nDCG@{primary_k}'),
        (ax2, 'Val eval coherence', 'Val coherence'),
    ):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

    ax1.legend(fontsize=7, ncol=2, loc='lower right')
    fig.suptitle(
        f'Training curves (mean ± std, {n_seeds} seeds)',
        fontsize=9, y=1.03,
    )
    fig.tight_layout()
    _save(fig, fig_dir / 'training_curves')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot Pareto sweep results')
    p.add_argument('--results_dir', type=Path, required=True,
                   help='directory containing results.json')
    p.add_argument('--ckpt', choices=['best', 'final', 'both'], default='both',
                   help='checkpoint type to plot (default: both)')
    p.add_argument('--select_lambdas', type=float, nargs='+', default=None,
                   help='λ values for training-curve plot (default: auto ~6)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config, runs = load_results(args.results_dir)

    k_values = config.get('k_values', [1, 5, 10, 20])
    n_seeds = len(config.get('seeds', [0]))

    fig_dir = args.results_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    print(f'\nwriting figures to {fig_dir}/')

    ckpts: list[tuple[str, str]] = []
    if args.ckpt in ('best',  'both'):
        ckpts.append(('test_best',  'best'))
    if args.ckpt in ('final', 'both'):
        ckpts.append(('test_final', 'final'))

    for ckpt_key, tag in ckpts:
        print(f'\n[{tag} checkpoint]')
        fig_pareto(runs, k_values, ckpt_key, n_seeds, fig_dir, tag)
        fig_metric_sweep(runs, k_values, ckpt_key, n_seeds, fig_dir, tag)

    print('\n[training curves]')
    fig_training_curves(runs, k_values, n_seeds, fig_dir, args.select_lambdas)

    n_files = len(list(fig_dir.iterdir()))
    print(f'\ndone - {n_files} files in {fig_dir}/')


if __name__ == '__main__':
    main()
