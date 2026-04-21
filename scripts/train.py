#!/usr/bin/env python3
"""
Pareto-frontier training script.

Sweeps coherence weight lambda across a configurable range. For each value
trains a fresh model and records both nDCG@k (accuracy) and greedy-prediction
sequential coherence (quality) on the held-out test set. The joint results
are written to --output_dir/results.json.

Gradient norms are logged every epoch so you can inspect how each lambda
value affects optimization stability.

Usage (quick smoke-test, ~128 playlists):
    python scripts/train.py --max_train_playlists 128 --num_epochs 5

Typical evaluation run (~5k playlists):
    python scripts/train.py \
        --max_train_playlists 5000 \
        --num_epochs 20 \
        --coherence_weights 0.0 0.01 0.05 0.1 0.2 0.5 1.0
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow running from any directory
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


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ndcg_at_k(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k_values: list[int],
) -> dict[int, float]:
    """
    nDCG@k for next-track prediction.

    Because only one item is relevant per position, IDCG = 1/log2(2) = 1,
    so nDCG@k = 1/log2(rank+1) when rank <= k, else 0.

    Returns a dict mapping each k to the mean nDCG@k over all valid positions
    in the batch. Accumulate the returned values weighted by n_valid across
    batches to get the dataset-level mean.
    """
    valid_mask = labels != -100                               # (B, T)
    n_valid = int(valid_mask.sum().item())
    if n_valid == 0:
        return {k: 0.0 for k in k_values}

    labels_clamped = labels.clamp(min=0)

    # Score the true next track at every position
    true_scores = logits.gather(
        2, labels_clamped.unsqueeze(-1)
    ).squeeze(-1)                                             # (B, T)

    # Rank = #tracks scored strictly higher + 1  (tied scores get best rank)
    ranks = (logits > true_scores.unsqueeze(-1)).sum(dim=-1) + 1  # (B, T)

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
    """
    Mean sequential coherence of the model's greedy predictions.

    At each valid position t, measures similarity(greedy_pred[t], input_ids[t]).
    A position is valid when labels[t] != -100 (non-padding, non-BOS-target).
    Special tokens (PAD, BOS, EOS) have zero rows in the similarity matrix so
    they contribute 0 without requiring explicit filtering.
    """
    valid_mask = (labels != -100)
    n_valid = int(valid_mask.sum().item())
    if n_valid == 0:
        return 0.0

    preds = logits.argmax(dim=-1)                             # (B, T)

    # similarity_matrix[i, j] = sim(track i, track j)
    sim = similarity_matrix[input_ids, preds]                 # (B, T)
    return (sim * valid_mask.float()).sum().item() / n_valid


# ---------------------------------------------------------------------------
# One evaluation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: DecodeOnlyTransformer,
    loader,
    similarity_matrix: torch.Tensor,
    coherence_weight: float,
    coherence_temperature: float,
    k_values: list[int],
    device: torch.device,
) -> dict:
    model.eval()

    ce_sum = 0.0
    coh_sum = 0.0
    ndcg_sums = {k: 0.0 for k in k_values}
    eval_coh_sum = 0.0
    n_tokens = 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        labels         = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits = model(input_ids, attention_mask)

        coherence_scores = sequential_coherence_scores_fast(
            input_ids=input_ids,
            similarity_matrix=similarity_matrix,
            attention_mask=attention_mask,
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

        n_valid = int((labels != -100).sum().item())
        n_tokens  += n_valid
        ce_sum    += float(loss_dict['ce_loss'])  * n_valid
        coh_sum   += float(loss_dict['coherence_loss']) * n_valid

        ndcg = compute_ndcg_at_k(logits, labels, k_values)
        for k in k_values:
            ndcg_sums[k] += ndcg[k] * n_valid

        eval_coh_sum += compute_eval_coherence(
            logits, input_ids, labels, similarity_matrix
        ) * n_valid

    d = max(n_tokens, 1)
    return {
        'ce_loss':        ce_sum  / d,
        'coh_loss':       coh_sum / d,
        'eval_coherence': eval_coh_sum / d,
        **{f'ndcg_{k}':   ndcg_sums[k] / d for k in k_values},
    }


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: DecodeOnlyTransformer,
    loader,
    optimizer: torch.optim.Optimizer,
    similarity_matrix: torch.Tensor,
    coherence_weight: float,
    coherence_temperature: float,
    max_grad_norm: float,
    device: torch.device,
) -> dict:
    model.train()

    loss_sum = 0.0
    ce_sum   = 0.0
    coh_sum  = 0.0
    n_tokens = 0
    grad_norm_sum = 0.0
    n_batches = 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        labels         = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        coherence_scores = sequential_coherence_scores_fast(
            input_ids=input_ids,
            similarity_matrix=similarity_matrix,
            attention_mask=attention_mask,
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

        # Record pre-clip gradient norm — this is the stability signal the
        # reviewers asked about: at high lambda the coherence gradient can
        # dwarf the CE gradient and destabilize training.
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


# ---------------------------------------------------------------------------
# Full run for one lambda value
# ---------------------------------------------------------------------------

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
) -> dict:
    seed_everything(args.seed)
    model     = DecodeOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    k_values       = args.k_values
    primary_k      = max(k_values)
    best_val_ndcg  = -1.0
    ckpt_path      = output_dir / f'ckpt_lambda_{coherence_weight:.4f}.pt'
    epoch_logs: list[dict] = []

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        train_m = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            similarity_matrix=similarity_matrix,
            coherence_weight=coherence_weight,
            coherence_temperature=args.coherence_temperature,
            max_grad_norm=args.max_grad_norm,
            device=device,
        )

        val_m = evaluate(
            model=model,
            loader=val_loader,
            similarity_matrix=similarity_matrix,
            coherence_weight=coherence_weight,
            coherence_temperature=args.coherence_temperature,
            k_values=k_values,
            device=device,
        )

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
            f'  λ={coherence_weight:.3f} | ep {epoch:02d}/{args.num_epochs}'
            f' | train_ce={train_m["ce_loss"]:.4f}'
            f' | train_coh={train_m["coh_loss"]:.5f}'
            f' | grad_norm={train_m["grad_norm"]:.3f}'
            f' | val_nDCG@{primary_k}={val_ndcg_primary:.4f}'
            f' | val_coh={val_m["eval_coherence"]:.4f}'
        )

    # Evaluate the final epoch model on test while it's still in memory.
    # This is the Pareto-relevant result: it shows where each lambda actually
    # lands after full training, not just the accuracy-maximizing checkpoint.
    final_test_m = evaluate(
        model=model,
        loader=test_loader,
        similarity_matrix=similarity_matrix,
        coherence_weight=coherence_weight,
        coherence_temperature=args.coherence_temperature,
        k_values=k_values,
        device=device,
    )

    # Also evaluate the best-val-nDCG checkpoint for completeness.
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    best_test_m = evaluate(
        model=model,
        loader=test_loader,
        similarity_matrix=similarity_matrix,
        coherence_weight=coherence_weight,
        coherence_temperature=args.coherence_temperature,
        k_values=k_values,
        device=device,
    )

    return {
        'coherence_weight': coherence_weight,
        'epochs':           epoch_logs,
        'best_val_ndcg':    best_val_ndcg,
        'test_best':        {f'test_{k}': v for k, v in best_test_m.items()},
        'test_final':       {f'test_{k}': v for k, v in final_test_m.items()},
        'checkpoint':       str(ckpt_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Pareto frontier sweep: nDCG@k vs. sequential coherence'
    )

    # Data
    p.add_argument('--data_dir',             type=Path,  default=None,
                   help='Path to MPD data/ directory (auto-detected if omitted)')
    p.add_argument('--max_train_playlists',  type=int,   default=5000)
    p.add_argument('--min_track_freq',       type=int,   default=3,
                   help='Minimum playlist appearances to keep a track in vocab')
    p.add_argument('--max_seq_len',          type=int,   default=64)
    p.add_argument('--batch_size',           type=int,   default=64)
    p.add_argument('--num_workers',          type=int,   default=0)

    # Model
    p.add_argument('--d_model',   type=int,   default=128)
    p.add_argument('--n_heads',   type=int,   default=4)
    p.add_argument('--n_layers',  type=int,   default=2)
    p.add_argument('--d_ff',      type=int,   default=256)
    p.add_argument('--dropout',   type=float, default=0.1)

    # Training
    p.add_argument('--num_epochs',    type=int,   default=20)
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-2)
    p.add_argument('--max_grad_norm', type=float, default=1.0,
                   help='Pre-clip gradient norm; also logged raw for stability analysis')
    p.add_argument('--seed',          type=int,   default=42)

    # Coherence sweep
    p.add_argument(
        '--coherence_weights', type=float, nargs='+',
        default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        help='Lambda values to sweep. 0.0 is the pure CE baseline.',
    )
    p.add_argument('--coherence_temperature', type=float, default=1.0)

    # Evaluation
    p.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 20])

    # Output
    p.add_argument(
        '--output_dir', type=Path,
        default=Path('saved_models') / 'pareto_sweep',
    )
    p.add_argument('--device', type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(
        args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'Device: {device}')

    # Resolve data directory relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_dir = args.data_dir or (project_root / 'datasets' / 'MPD' / 'data')
    if not data_dir.exists():
        print(f'ERROR: data directory not found: {data_dir}', file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    print('\n=== Loading data ===')
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
    print(f'\nVocab size: {vocab_size:,}  |  similarity matrix: ~{matrix_mb:.0f} MB')
    if matrix_mb > 4096:
        print(
            'WARNING: estimated similarity matrix exceeds 4 GB. '
            'Raise --min_track_freq or lower --max_train_playlists to reduce it.',
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ coherence
    print('\n=== Building co-occurrence store & similarity matrix ===')
    store             = build_cooccurrence_store(sequences=train_sequences, vocab=vocab)
    similarity_matrix = build_dense_similarity_matrix(store=store, device=device)

    # ------------------------------------------------------------------ model
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
    print(f'Model parameters: {n_params:,}')

    # ------------------------------------------------------------------ sweep
    print(f'\n=== Sweeping {len(args.coherence_weights)} lambda values ===')
    print(f'λ: {args.coherence_weights}\n')

    all_runs: list[dict] = []
    for lam in args.coherence_weights:
        print(f'\n--- λ = {lam} ---')
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
        )
        all_runs.append(run)

    # ------------------------------------------------------------------ save
    results = {
        'config': {
            'max_train_playlists':  args.max_train_playlists,
            'min_track_freq':       args.min_track_freq,
            'max_seq_len':          args.max_seq_len,
            'vocab_size':           vocab_size,
            'd_model':              args.d_model,
            'n_heads':              args.n_heads,
            'n_layers':             args.n_layers,
            'd_ff':                 args.d_ff,
            'dropout':              args.dropout,
            'num_epochs':           args.num_epochs,
            'lr':                   args.lr,
            'weight_decay':         args.weight_decay,
            'max_grad_norm':        args.max_grad_norm,
            'coherence_temperature': args.coherence_temperature,
            'seed':                 args.seed,
            'k_values':             args.k_values,
            'coherence_weights':    args.coherence_weights,
        },
        'runs': all_runs,
    }
    results_path = args.output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved → {results_path}')

    # ------------------------------------------------------------------ summary
    total_elapsed = sum(
        e['elapsed_s'] for run in all_runs for e in run['epochs']
    )
    minutes, seconds = divmod(int(total_elapsed), 60)
    print(f'\nTotal training time: {minutes}m {seconds}s')

    primary_k = max(args.k_values)
    col = f'nDCG@{primary_k}'
    header = f'{"lambda":>8}  {col:>10}  {"coherence":>10}  {"CE loss":>10}  {"grad_norm_p50":>14}'
    sep = '-' * len(header)

    def _row(run: dict, key: str) -> str:
        lam  = run['coherence_weight']
        ndcg = run[key].get(f'test_ndcg_{primary_k}', float('nan'))
        coh  = run[key].get('test_eval_coherence', float('nan'))
        ce   = run[key].get('test_ce_loss', float('nan'))
        grad_norms = [e['train_grad_norm'] for e in run['epochs']]
        med_gn = sorted(grad_norms)[len(grad_norms) // 2]
        return f'{lam:>8.3f}  {ndcg:>10.4f}  {coh:>10.4f}  {ce:>10.4f}  {med_gn:>14.3f}'

    print('\n=== Pareto Frontier — final epoch (tradeoff at convergence) ===')
    print(header)
    print(sep)
    for run in all_runs:
        print(_row(run, 'test_final'))

    print(f'\n=== Pareto Frontier — best val nDCG@{primary_k} checkpoint (accuracy-optimised) ===')
    print(header)
    print(sep)
    for run in all_runs:
        print(_row(run, 'test_best'))


if __name__ == '__main__':
    main()
