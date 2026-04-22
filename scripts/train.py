#!/usr/bin/env python3
"""
Pareto-frontier sweep: nDCG@k vs. sequential coherence.

Trains one model per coherence weight (lambda) and records accuracy and
coherence quality on the held-out test set. Supports three coherence scoring
modes: sequential, prefix_mean, and combined.

Usage:
    python scripts/train.py --max_train_playlists 128 --num_epochs 5

Coherence ablation:
    python scripts/train.py \\
        --max_train_playlists 8000 --num_epochs 20 \\
        --d_model 256 --n_layers 4 --d_ff 512 \\
        --coherence_mode prefix_mean \\
        --coherence_weights 0.0 5.0 10.0
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.data_loading.mpd.reader import MPDConfig
from modules.data_loading.mpd.make_datasets import make_mpd_loaders
from modules.models.decode_only_transformer import ModelConfig, DecodeOnlyTransformer
from modules.coherence.cooccurence import (
    build_cooccurrence_store,
    build_dense_similarity_matrix,
    sequential_coherence_scores_fast,
)
from modules.coherence.losses import combined_ce_coherence_loss
from modules.utilities.logging import log as write_log


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prefix_mean_coherence_scores_fast(
    input_ids: torch.Tensor,
    similarity_matrix: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # At position t, returns the mean similarity row over all real tracks
    # seen in positions 0..t. Special-token positions have all-zero rows
    # in the similarity matrix so they don't affect the average.
    all_rows = similarity_matrix[input_ids]                       # (B, T, V)
    is_real = (all_rows.sum(dim=-1, keepdim=True) > 0).float()   # (B, T, 1)
    if attention_mask is not None:
        is_real = is_real * attention_mask.unsqueeze(-1).float()
    cum_sum   = (all_rows * is_real).cumsum(dim=1)                # (B, T, V)
    cum_count = is_real.cumsum(dim=1).clamp(min=1)                # (B, T, 1)
    return cum_sum / cum_count


def _get_coherence_scores(
    input_ids: torch.Tensor,
    similarity_matrix: torch.Tensor,
    attention_mask: torch.Tensor | None,
    mode: str,
    alpha: float,
) -> torch.Tensor:
    if mode == 'sequential':
        return sequential_coherence_scores_fast(input_ids, similarity_matrix, attention_mask)
    if mode == 'prefix_mean':
        return _prefix_mean_coherence_scores_fast(input_ids, similarity_matrix, attention_mask)
    seq   = sequential_coherence_scores_fast(input_ids, similarity_matrix, attention_mask)
    pmean = _prefix_mean_coherence_scores_fast(input_ids, similarity_matrix, attention_mask)
    return alpha * seq + (1.0 - alpha) * pmean


@torch.no_grad()
def compute_ndcg_at_k(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k_values: list[int],
) -> dict[int, float]:
    valid_mask = labels != -100
    n_valid = int(valid_mask.sum().item())
    if n_valid == 0:
        return {k: 0.0 for k in k_values}

    labels_clamped = labels.clamp(min=0)
    true_scores = logits.gather(2, labels_clamped.unsqueeze(-1)).squeeze(-1)
    ranks = (logits > true_scores.unsqueeze(-1)).sum(dim=-1) + 1

    results: dict[int, float] = {}
    for k in k_values:
        in_top_k = valid_mask & (ranks <= k)
        dcg = torch.where(
            in_top_k,
            1.0 / torch.log2(ranks.float() + 1.0),
            torch.zeros_like(ranks, dtype=torch.float),
        )
        results[k] = dcg.sum().item() / n_valid
    return results


@torch.no_grad()
def compute_eval_coherence(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    similarity_matrix: torch.Tensor,
) -> float:
    # Always sequential regardless of training mode, so the eval metric is
    # comparable across ablations.
    valid_mask = labels != -100
    n_valid = int(valid_mask.sum().item())
    if n_valid == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    sim   = similarity_matrix[input_ids, preds]
    return (sim * valid_mask.float()).sum().item() / n_valid


@torch.no_grad()
def evaluate(
    model: DecodeOnlyTransformer,
    loader,
    similarity_matrix: torch.Tensor,
    coherence_weight: float,
    coherence_temperature: float,
    coherence_mode: str,
    coherence_alpha: float,
    k_values: list[int],
    device: torch.device,
) -> dict:
    model.eval()

    ce_sum       = 0.0
    coh_sum      = 0.0
    ndcg_sums    = {k: 0.0 for k in k_values}
    eval_coh_sum = 0.0
    n_tokens     = 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        labels         = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits = model(input_ids, attention_mask)

        coherence_scores = _get_coherence_scores(
            input_ids, similarity_matrix, attention_mask, coherence_mode, coherence_alpha,
        )

        loss_dict = combined_ce_coherence_loss(
            logits=logits,
            labels=labels,
            coherence_scores=coherence_scores,
            attention_mask=attention_mask,
            ce_weight=1.0,
            coherence_weight=coherence_weight,
            coherence_temperature=coherence_temperature,
        )

        n_valid   = int((labels != -100).sum().item())
        n_tokens += n_valid
        ce_sum   += float(loss_dict['ce_loss'])        * n_valid
        coh_sum  += float(loss_dict['coherence_loss']) * n_valid

        ndcg = compute_ndcg_at_k(logits, labels, k_values)
        for k in k_values:
            ndcg_sums[k] += ndcg[k] * n_valid

        eval_coh_sum += compute_eval_coherence(
            logits, input_ids, labels, similarity_matrix
        ) * n_valid

    d = max(n_tokens, 1)
    return {
        'ce_loss':        ce_sum        / d,
        'coh_loss':       coh_sum       / d,
        'eval_coherence': eval_coh_sum  / d,
        **{f'ndcg_{k}':   ndcg_sums[k] / d for k in k_values},
    }


def train_epoch(
    model: DecodeOnlyTransformer,
    loader,
    optimizer: torch.optim.Optimizer,
    similarity_matrix: torch.Tensor,
    coherence_weight: float,
    coherence_temperature: float,
    coherence_mode: str,
    coherence_alpha: float,
    max_grad_norm: float,
    device: torch.device,
) -> dict:
    model.train()

    loss_sum      = 0.0
    ce_sum        = 0.0
    coh_sum       = 0.0
    n_tokens      = 0
    grad_norm_sum = 0.0
    n_batches     = 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        labels         = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        coherence_scores = _get_coherence_scores(
            input_ids, similarity_matrix, attention_mask, coherence_mode, coherence_alpha,
        )

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        loss_dict = combined_ce_coherence_loss(
            logits=logits,
            labels=labels,
            coherence_scores=coherence_scores,
            attention_mask=attention_mask,
            ce_weight=1.0,
            coherence_weight=coherence_weight,
            coherence_temperature=coherence_temperature,
        )

        loss_dict['loss'].backward()

        # Log pre-clip norm to track how much the coherence gradient competes
        # with CE as lambda grows.
        pre_clip_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                pre_clip_norm += p.grad.data.norm(2).item() ** 2
        grad_norm_sum += pre_clip_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        n_valid   = int((labels != -100).sum().item())
        n_tokens  += n_valid
        loss_sum  += loss_dict['loss'].detach().item()           * n_valid
        ce_sum    += loss_dict['ce_loss'].detach().item()        * n_valid
        coh_sum   += loss_dict['coherence_loss'].detach().item() * n_valid
        n_batches += 1

    d = max(n_tokens, 1)
    return {
        'loss':      loss_sum / d,
        'ce_loss':   ce_sum   / d,
        'coh_loss':  coh_sum  / d,
        'grad_norm': grad_norm_sum / max(n_batches, 1),
    }


def run_single(
    coherence_weight: float,
    train_loader,
    val_loader,
    test_loader,
    similarity_matrix: torch.Tensor,
    model_config: ModelConfig,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    log_dir: Path,
) -> dict:
    seed_everything(args.seed)
    model     = DecodeOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    k_values      = args.k_values
    primary_k     = max(k_values)
    best_val_ndcg = -1.0
    ckpt_path     = output_dir / f'ckpt_{args.coherence_mode}_lambda_{coherence_weight:.4f}.pt'
    epoch_logs: list[dict] = []

    eval_kwargs = dict(
        similarity_matrix=similarity_matrix,
        coherence_weight=coherence_weight,
        coherence_temperature=args.coherence_temperature,
        coherence_mode=args.coherence_mode,
        coherence_alpha=args.coherence_alpha,
        k_values=k_values,
        device=device,
    )

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        train_m = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            similarity_matrix=similarity_matrix,
            coherence_weight=coherence_weight,
            coherence_temperature=args.coherence_temperature,
            coherence_mode=args.coherence_mode,
            coherence_alpha=args.coherence_alpha,
            max_grad_norm=args.max_grad_norm,
            device=device,
        )

        val_m = evaluate(model=model, loader=val_loader, **eval_kwargs)

        epoch_log = {
            'epoch':           epoch,
            'train_loss':      train_m['loss'],
            'train_ce_loss':   train_m['ce_loss'],
            'train_coh_loss':  train_m['coh_loss'],
            'train_grad_norm': train_m['grad_norm'],
            **{f'val_{k}': v for k, v in val_m.items()},
            'elapsed_s':       round(time.time() - t0, 2),
        }
        epoch_logs.append(epoch_log)

        val_ndcg_primary = float(val_m.get(f'ndcg_{primary_k}', 0.0))
        if val_ndcg_primary > best_val_ndcg:
            best_val_ndcg = val_ndcg_primary
            torch.save(model.state_dict(), ckpt_path)

        print(
            f'  lambda={coherence_weight:.3f} [{args.coherence_mode}]'
            f' | ep {epoch:02d}/{args.num_epochs}'
            f' | train_ce={train_m["ce_loss"]:.4f}'
            f' | train_coh={train_m["coh_loss"]:.5f}'
            f' | grad_norm={train_m["grad_norm"]:.3f}'
            f' | val_nDCG@{primary_k}={val_ndcg_primary:.4f}'
            f' | val_coh={val_m["eval_coherence"]:.4f}'
        )

    final_test_m = evaluate(model=model, loader=test_loader, **eval_kwargs)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    best_test_m = evaluate(model=model, loader=test_loader, **eval_kwargs)

    run_result = {
        'coherence_weight': coherence_weight,
        'coherence_mode':   args.coherence_mode,
        'coherence_alpha':  args.coherence_alpha,
        'epochs':           epoch_logs,
        'best_val_ndcg':    best_val_ndcg,
        'test_best':        {f'test_{k}': v for k, v in best_test_m.items()},
        'test_final':       {f'test_{k}': v for k, v in final_test_m.items()},
        'checkpoint':       str(ckpt_path),
    }

    write_log(run_result, subdir_name=output_dir.name, log_dir=log_dir)

    return run_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Pareto frontier sweep: nDCG@k vs. sequential coherence'
    )

    p.add_argument('--data_dir',            type=Path, default=None)
    p.add_argument('--max_train_playlists', type=int,  default=5000)
    p.add_argument('--min_track_freq',      type=int,  default=3)
    p.add_argument('--max_seq_len',         type=int,  default=64)
    p.add_argument('--batch_size',          type=int,  default=64)
    p.add_argument('--num_workers',         type=int,  default=0)

    p.add_argument('--d_model',  type=int,   default=128)
    p.add_argument('--n_heads',  type=int,   default=4)
    p.add_argument('--n_layers', type=int,   default=2)
    p.add_argument('--d_ff',     type=int,   default=256)
    p.add_argument('--dropout',  type=float, default=0.1)

    p.add_argument('--num_epochs',    type=int,   default=20)
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-2)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--seed',          type=int,   default=42)

    p.add_argument(
        '--coherence_weights', type=float, nargs='+',
        default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    )
    p.add_argument('--coherence_temperature', type=float, default=1.0)
    p.add_argument(
        '--coherence_mode', type=str, default='sequential',
        choices=['sequential', 'prefix_mean', 'combined'],
    )
    p.add_argument('--coherence_alpha', type=float, default=0.7)

    p.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 20])

    p.add_argument(
        '--output_dir', type=Path,
        default=Path('saved_models') / 'pareto_sweep',
    )
    p.add_argument('--log_dir', type=Path, default=None)
    p.add_argument('--device',  type=str,  default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(
        args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'device: {device}')

    project_root = Path(__file__).resolve().parent.parent
    data_dir = args.data_dir or (project_root / 'datasets' / 'MPD' / 'data')
    if not data_dir.exists():
        print(f'ERROR: data directory not found: {data_dir}', file=sys.stderr)
        sys.exit(1)

    log_dir = args.log_dir or (project_root / 'logs')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print('\nloading data...')
    mpd_config = MPDConfig(
        data_dir=data_dir,
        min_track_freq=args.min_track_freq,
        max_seq_len=args.max_seq_len,
        max_train_playlists=args.max_train_playlists,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader, vocab, train_sequences = make_mpd_loaders(
        config=mpd_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    vocab_size = len(vocab)
    matrix_mb  = vocab_size ** 2 * 4 / 1024 ** 2
    print(f'vocab size: {vocab_size:,}  |  similarity matrix: ~{matrix_mb:.0f} MB')
    if matrix_mb > 4096:
        print(
            'WARNING: similarity matrix exceeds 4 GB. '
            'Raise --min_track_freq or lower --max_train_playlists.',
            file=sys.stderr,
        )

    print('\nbuilding co-occurrence store...')
    store             = build_cooccurrence_store(sequences=train_sequences, vocab=vocab)
    similarity_matrix = build_dense_similarity_matrix(store=store, device=device)

    model_config = ModelConfig(
        num_tracks=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        pad_idx=vocab.pad_idx,
        tie_weights=True,
    )
    n_params = sum(p.numel() for p in DecodeOnlyTransformer(model_config).parameters())
    print(f'model parameters: {n_params:,}')

    print(f'\nsweeping {len(args.coherence_weights)} lambda values (mode: {args.coherence_mode})')
    print(f'lambdas: {args.coherence_weights}\n')

    all_runs: list[dict] = []
    for lam in args.coherence_weights:
        print(f'\nlambda={lam}  mode={args.coherence_mode}')
        run = run_single(
            coherence_weight=lam,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            similarity_matrix=similarity_matrix,
            model_config=model_config,
            args=args,
            device=device,
            output_dir=args.output_dir,
            log_dir=log_dir,
        )
        all_runs.append(run)

    results = {
        'config': {
            'max_train_playlists':   args.max_train_playlists,
            'min_track_freq':        args.min_track_freq,
            'max_seq_len':           args.max_seq_len,
            'vocab_size':            vocab_size,
            'd_model':               args.d_model,
            'n_heads':               args.n_heads,
            'n_layers':              args.n_layers,
            'd_ff':                  args.d_ff,
            'dropout':               args.dropout,
            'num_epochs':            args.num_epochs,
            'lr':                    args.lr,
            'weight_decay':          args.weight_decay,
            'max_grad_norm':         args.max_grad_norm,
            'coherence_temperature': args.coherence_temperature,
            'coherence_mode':        args.coherence_mode,
            'coherence_alpha':       args.coherence_alpha,
            'seed':                  args.seed,
            'k_values':              args.k_values,
            'coherence_weights':     args.coherence_weights,
        },
        'runs': all_runs,
    }
    results_path = args.output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nresults saved -> {results_path}')

    total_elapsed = sum(e['elapsed_s'] for run in all_runs for e in run['epochs'])
    minutes, seconds = divmod(int(total_elapsed), 60)
    print(f'total training time: {minutes}m {seconds}s')

    primary_k = max(args.k_values)
    col    = f'nDCG@{primary_k}'
    header = (f'{"lambda":>8}  {col:>10}  {"coherence":>10}'
              f'  {"CE loss":>10}  {"grad_norm_p50":>14}')
    sep = '-' * len(header)

    def _row(run: dict, key: str) -> str:
        lam    = run['coherence_weight']
        ndcg   = run[key].get(f'test_ndcg_{primary_k}', float('nan'))
        coh    = run[key].get('test_eval_coherence',    float('nan'))
        ce     = run[key].get('test_ce_loss',           float('nan'))
        norms  = [e['train_grad_norm'] for e in run['epochs']]
        med_gn = sorted(norms)[len(norms) // 2]
        return (f'{lam:>8.3f}  {ndcg:>10.4f}  {coh:>10.4f}'
                f'  {ce:>10.4f}  {med_gn:>14.3f}')

    print(f'\npareto frontier (final epoch):')
    print(header)
    print(sep)
    for run in all_runs:
        print(_row(run, 'test_final'))

    print(f'\npareto frontier (best val nDCG@{primary_k} checkpoint):')
    print(header)
    print(sep)
    for run in all_runs:
        print(_row(run, 'test_best'))


if __name__ == '__main__':
    main()
