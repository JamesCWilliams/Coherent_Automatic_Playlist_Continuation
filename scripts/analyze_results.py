#!/usr/bin/env python3
"""
Post-training analysis: nDCG@k breakdown and prediction diversity.

nDCG@k ratios read from results.json only -- no GPU needed.
Diversity analysis loads each checkpoint and runs greedy inference on the
test set to measure unique-track coverage, entropy, and popularity bias.

Usage:
    python scripts/analyze_results.py \\
        --results_dir saved_models/small_sweep_8k_big_model

nDCG ratios only (no checkpoint loading):
    python scripts/analyze_results.py \\
        --results_dir saved_models/small_sweep_8k_big_model \\
        --skip_diversity

Specific lambdas for diversity:
    python scripts/analyze_results.py \\
        --results_dir saved_models/small_sweep_8k_big_model \\
        --lambdas 0.0 5.0 15.0
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.data_loading.mpd.reader import MPDConfig
from modules.data_loading.mpd.make_datasets import make_mpd_loaders
from modules.models.decode_only_transformer import ModelConfig, DecodeOnlyTransformer


def analyze_ndcg_ratios(runs: list[dict], k_values: list[int]) -> None:
    baseline = next((r for r in runs if r['coherence_weight'] == 0.0), None)
    if baseline is None:
        print('WARNING: no lambda=0 baseline; skipping delta computation.')
        base_scores = {k: 0.0 for k in k_values}
    else:
        base_scores = {k: baseline['mean']['test_final'].get(f'test_ndcg_{k}', float('nan'))
                       for k in k_values}

    k_hdr = '  '.join(f'nDCG@{k:2d}' for k in k_values)
    d_hdr = '  '.join(f' D@{k:2d} ' for k in k_values)
    width = 10 + len(k_values) * 10 + 6 + len(k_values) * 8
    print(f'\n{"lambda":>8}  {k_hdr}  ||  {d_hdr}')
    print('-' * width)

    for run in runs:
        lam = run['coherence_weight']
        vals = {k: run['mean']['test_final'].get(f'test_ndcg_{k}', float('nan')) for k in k_values}
        deltas = {k: vals[k] - base_scores[k] for k in k_values}
        s_str = '  '.join(f'{vals[k]:>8.4f}' for k in k_values)
        d_str = '  '.join(f'{deltas[k]:>+6.4f}' for k in k_values)
        print(f'{lam:>8.2f}  {s_str}  ||  {d_str}')

    print()
    print('rank-k sensitivity: does coherence penalise early ranks more than late?')
    for run in runs:
        if run['coherence_weight'] == 0.0:
            continue
        lam = run['coherence_weight']
        deltas = [run['mean']['test_final'].get(f'test_ndcg_{k}', float('nan')) - base_scores[k]
                  for k in k_values]
        slope = (deltas[-1] - deltas[0]) / max(k_values[-1] - k_values[0], 1)
        k0, k1 = k_values[0], k_values[-1]
        print(f'  lambda={lam:5.1f}: D@{k0}={deltas[0]:+.4f} -> D@{k1}={deltas[-1]:+.4f}'
              f'  (slope {slope:+.6f}/rank)')


@torch.no_grad()
def collect_predictions(
    model: DecodeOnlyTransformer,
    loader,
    device: torch.device,
) -> tuple[Counter, int]:
    model.eval()
    pred_counter: Counter = Counter()
    n_total = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1)

        valid_mask = labels != -100
        valid_preds = preds[valid_mask].cpu().tolist()
        pred_counter.update(valid_preds)
        n_total += len(valid_preds)

    return pred_counter, n_total


def diversity_metrics(counter: Counter, n_positions: int, vocab_size: int) -> dict:
    unique_tracks = len(counter)
    coverage_pct = unique_tracks / vocab_size * 100

    entropy = 0.0
    for cnt in counter.values():
        p = cnt / n_positions
        entropy -= p * math.log(p)
    max_entropy = math.log(vocab_size)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    sorted_counts = sorted(counter.values(), reverse=True)
    top10_pct = sum(sorted_counts[:10])  / n_positions * 100
    top100_pct = sum(sorted_counts[:100]) / n_positions * 100

    n = len(sorted_counts)
    cumsum, gini_sum = 0, 0
    for c in sorted(sorted_counts):
        cumsum += c
        gini_sum += cumsum
    gini = 1.0 - 2.0 * gini_sum / max(n * n_positions, 1)

    return {
        'unique_tracks': unique_tracks,
        'coverage_pct': coverage_pct,
        'entropy_nats': entropy,
        'norm_entropy': norm_entropy,
        'top10_pct': top10_pct,
        'top100_pct': top100_pct,
        'gini': gini,
        'n_positions': n_positions,
    }


def analyze_diversity(
    runs: list[dict],
    lambdas: list[float] | None,
    config: dict,
    project_root: Path,
    device: torch.device,
) -> dict:
    mpd_config = MPDConfig(
        data_dir=project_root / 'datasets' / 'MPD' / 'data',
        min_track_freq=config['min_track_freq'],
        max_seq_len=config['max_seq_len'],
        max_train_playlists=config['max_train_playlists'],
    )
    print('\nloading dataset for diversity analysis...')
    _, _, test_loader, vocab, train_sequences = make_mpd_loaders(
        config=mpd_config, batch_size=64, num_workers=0
    )
    vocab_size = len(vocab)

    train_freq: Counter = Counter()
    for seq in train_sequences:
        train_freq.update(seq)

    model_cfg = ModelConfig(
        num_tracks=vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        pad_idx=vocab.pad_idx,
        tie_weights=True,
    )

    target_lambdas = set(lambdas) if lambdas else None
    selected_runs = [r for r in runs
                      if target_lambdas is None or r['coherence_weight'] in target_lambdas]

    counters: dict[float, Counter] = {}
    metrics: dict[float, dict]   = {}

    for run in selected_runs:
        lam = run['coherence_weight']
        seed_runs = run.get('seed_runs', [])
        ckpt_rel = seed_runs[0].get('checkpoint', '') if seed_runs else ''
        ckpt = (project_root / ckpt_rel
                    if not Path(ckpt_rel).is_absolute() else Path(ckpt_rel))

        if not ckpt.exists():
            print(f'  lambda={lam}: checkpoint not found at {ckpt}, skipping.')
            continue

        print(f'  collecting predictions for lambda={lam}...')
        model = DecodeOnlyTransformer(model_cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        counter, n_pos = collect_predictions(model, test_loader, device)
        counters[lam] = counter
        metrics[lam] = diversity_metrics(counter, n_pos, vocab_size)
        del model

    if not metrics:
        print('no checkpoints found.')
        return {}

    print('\nprediction diversity (test set, greedy top-1):')
    hdr = (f'{"lambda":>8}  {"unique":>8}  {"cover%":>8}  '
           f'{"H_norm":>8}  {"top10%":>8}  {"top100%":>9}  {"gini":>8}')
    print(hdr)
    print('-' * len(hdr))
    for lam in sorted(metrics):
        m = metrics[lam]
        print(f'{lam:>8.2f}  {m["unique_tracks"]:>8,}  {m["coverage_pct"]:>8.2f}'
              f'  {m["norm_entropy"]:>8.4f}  {m["top10_pct"]:>8.2f}'
              f'  {m["top100_pct"]:>9.2f}  {m["gini"]:>8.4f}')

    print('\npopularity bias: spearman rho (pred_freq vs. train_freq)')
    print('  higher rho -> model is popularity-biased')
    print('  lower rho after coherence -> regularisation breaks popularity collapse')
    try:
        from scipy.stats import spearmanr
        all_tracks = sorted({t for c in counters.values() for t in c})
        tf_vec = [train_freq.get(t, 0) for t in all_tracks]
        for lam in sorted(counters):
            pf_vec = [counters[lam].get(t, 0) for t in all_tracks]
            rho, pval = spearmanr(tf_vec, pf_vec)
            n_unique = metrics[lam]['unique_tracks']
            print(f'  lambda={lam:5.1f}:  rho={rho:.4f}  (p={pval:.2e})'
                  f'  [{n_unique:,} unique predicted tracks]')
    except ImportError:
        print('  scipy not installed -- run: pip install scipy')

    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', type=Path,
                   default=Path('saved_models/small_sweep_8k_big_model'))
    p.add_argument('--lambdas', type=float, nargs='+', default=None)
    p.add_argument('--skip_diversity', action='store_true')
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'device: {device}')

    project_root = Path(__file__).resolve().parent.parent
    results_path = args.results_dir / 'results.json'

    if not results_path.exists():
        print(f'ERROR: {results_path} not found', file=sys.stderr)
        sys.exit(1)

    with open(results_path, encoding='utf-8') as f:
        data = json.load(f)

    config = data['config']
    runs = data['runs']
    k_values = config.get('k_values', [1, 5, 10, 20])

    print(f'\nresults: {results_path}')
    print(f'lambdas: {[r["coherence_weight"] for r in runs]}')

    print('\nnDCG@k breakdown (test_final -- at convergence):')
    analyze_ndcg_ratios(runs, k_values)

    if not args.skip_diversity:
        metrics = analyze_diversity(
            runs=runs,
            lambdas=args.lambdas,
            config=config,
            project_root=project_root,
            device=device,
        )
        if metrics:
            out_path = args.results_dir / 'diversity_metrics.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in metrics.items()}, f, indent=2)
            print(f'\ndiversity metrics saved -> {out_path}')


if __name__ == '__main__':
    main()
