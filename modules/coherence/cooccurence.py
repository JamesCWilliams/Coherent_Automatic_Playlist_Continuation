from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass
import math
from typing import Optional
import torch


@dataclass
class CoOccurrenceStore:

    track_counts: dict[int, int]
    pair_counts: dict[tuple[int, int], int]
    valid_track_mask: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return int(self.valid_track_mask.numel())


def _ordered_pair(
    a: int,
    b: int,
) -> tuple[int, int]:
    
    return (a, b) if a < b else (b, a)


def build_cooccurrence_store(
    sequences: list[list[int]],
    vocab,
) -> CoOccurrenceStore:

    special_indices = set()
    for attr in ['pad_idx', 'bos_idx', 'eos_idx', 'unk_idx', 'msk_idx']:
        if hasattr(vocab, attr):
            special_indices.add(getattr(vocab, attr))

    valid_track_mask = torch.ones(len(vocab), dtype=torch.bool)
    for idx in special_indices:
        if 0 <= idx < len(vocab):
            valid_track_mask[idx] = False

    track_counts = defaultdict(int)
    pair_counts = defaultdict(int)

    for seq in sequences:
        uniq_tracks = sorted({t for t in seq if 0 <= t < len(vocab) and valid_track_mask[t]})

        if not uniq_tracks:
            continue

        for t in uniq_tracks:
            track_counts[t] += 1

        for a, b in combinations(uniq_tracks, 2):
            pair_counts[(a, b)] += 1

    return CoOccurrenceStore(
        track_counts=dict(track_counts),
        pair_counts=dict(pair_counts),
        valid_track_mask=valid_track_mask,
    )


def track_similarity(
    track_a: int,
    track_b: int,
    store: CoOccurrenceStore,
) -> float:

    if not store.valid_track_mask[track_a] or not store.valid_track_mask[track_b]:
        return 0.0

    if track_a == track_b:
        return 1.0

    a, b = _ordered_pair(track_a, track_b)
    cij = store.pair_counts.get((a, b), 0)
    ci = store.track_counts.get(track_a, 0)
    cj = store.track_counts.get(track_b, 0)

    if ci == 0 or cj == 0:
        return 0.0

    return cij / math.sqrt(ci * cj)


def similarity_row(
    anchor_track: int,
    store: CoOccurrenceStore,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    vocab_size = store.vocab_size
    scores = torch.zeros(vocab_size, dtype=dtype)

    if not store.valid_track_mask[anchor_track]:
        if device is not None:
            scores = scores.to(device)
        return scores

    ci = store.track_counts.get(anchor_track, 0)
    if ci == 0:
        if device is not None:
            scores = scores.to(device)
        return scores

    scores[anchor_track] = 1.0

    for other_track in range(vocab_size):
        if other_track == anchor_track:
            continue
        if not store.valid_track_mask[other_track]:
            continue

        cj = store.track_counts.get(other_track, 0)
        if cj == 0:
            continue

        a, b = _ordered_pair(anchor_track, other_track)
        cij = store.pair_counts.get((a, b), 0)
        if cij == 0:
            continue

        scores[other_track] = cij / math.sqrt(ci * cj)

    scores = scores * store.valid_track_mask.to(dtype)

    if device is not None:
        scores = scores.to(device)

    return scores


def sequential_coherence_scores(
    input_ids: torch.Tensor,
    store: CoOccurrenceStore,
    attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    batch_size, seq_len = input_ids.shape
    vocab_size = store.vocab_size

    if device is None:
        device = input_ids.device

    scores = torch.zeros(batch_size, seq_len, vocab_size, dtype=dtype, device=device)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.to(torch.bool)

    for b in range(batch_size):
        for t in range(seq_len):
            if not attention_mask[b, t]:
                continue

            anchor = int(input_ids[b, t].item())
            scores[b, t] = similarity_row(anchor, store, device=device, dtype=dtype)

    return scores


def prefix_mean_coherence_scores(
    input_ids: torch.Tensor,
    store: CoOccurrenceStore,
    attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    batch_size, seq_len = input_ids.shape
    vocab_size = store.vocab_size

    if device is None:
        device = input_ids.device

    scores = torch.zeros(batch_size, seq_len, vocab_size, dtype=dtype, device=device)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.to(torch.bool)

    valid_track_mask = store.valid_track_mask.to(device)

    for b in range(batch_size):
        prefix_tracks: list[int] = []

        for t in range(seq_len):
            if not attention_mask[b, t]:
                continue

            current_track = int(input_ids[b, t].item())
            if 0 <= current_track < vocab_size and bool(valid_track_mask[current_track]):
                prefix_tracks.append(current_track)

            if not prefix_tracks:
                continue

            row_sum = torch.zeros(vocab_size, dtype=dtype, device=device)
            count = 0

            for anchor in prefix_tracks:
                row_sum += similarity_row(anchor, store, device=device, dtype=dtype)
                count += 1

            scores[b, t] = row_sum / max(count, 1)

    return scores


def combined_coherence_scores(
    input_ids: torch.Tensor,
    store: CoOccurrenceStore,
    attention_mask: Optional[torch.Tensor] = None,
    alpha: float = 0.7,
    use_sequential: bool = True,
    use_prefix_mean: bool = True,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    if not use_sequential and not use_prefix_mean:
        raise ValueError('At least one of use_sequential or use_prefix_mean must be True.')

    if use_sequential and not use_prefix_mean:
        return sequential_coherence_scores(
            input_ids=input_ids,
            store=store,
            attention_mask=attention_mask,
            device=device,
            dtype=dtype,
        )

    if use_prefix_mean and not use_sequential:
        return prefix_mean_coherence_scores(
            input_ids=input_ids,
            store=store,
            attention_mask=attention_mask,
            device=device,
            dtype=dtype,
        )

    seq_scores = sequential_coherence_scores(
        input_ids=input_ids,
        store=store,
        attention_mask=attention_mask,
        device=device,
        dtype=dtype,
    )

    mean_scores = prefix_mean_coherence_scores(
        input_ids=input_ids,
        store=store,
        attention_mask=attention_mask,
        device=device,
        dtype=dtype,
    )

    return alpha * seq_scores + (1.0 - alpha) * mean_scores


def build_dense_similarity_matrix(
    store: CoOccurrenceStore,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    vocab_size = store.vocab_size
    S = torch.zeros(vocab_size, vocab_size, dtype=dtype)

    valid_mask = store.valid_track_mask

    for i in range(vocab_size):
        if not valid_mask[i]:
            continue

        ci = store.track_counts.get(i, 0)
        if ci == 0:
            continue

        S[i, i] = 1.0

    for (a, b), cij in store.pair_counts.items():
        ci = store.track_counts.get(a, 0)
        cj = store.track_counts.get(b, 0)
        if ci == 0 or cj == 0:
            continue

        sim = cij / math.sqrt(ci * cj)
        S[a, b] = sim
        S[b, a] = sim

    S = S * valid_mask.to(dtype).unsqueeze(0)
    S = S * valid_mask.to(dtype).unsqueeze(1)

    if device is not None:
        S = S.to(device)

    return S


def sequential_coherence_scores_fast(
    input_ids: torch.Tensor,
    similarity_matrix: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    scores = similarity_matrix[input_ids]  # (B, T, V)

    if attention_mask is not None:
        scores = scores * attention_mask.to(scores.dtype).unsqueeze(-1)

    return scores
