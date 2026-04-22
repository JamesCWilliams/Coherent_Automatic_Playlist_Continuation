from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass
import math
import torch


@dataclass
class CoOccurrenceStore:
    track_counts: dict[int, int]
    pair_counts: dict[tuple[int, int], int]
    valid_track_mask: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return int(self.valid_track_mask.numel())


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


def build_dense_similarity_matrix(
    store: CoOccurrenceStore,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    vocab_size = store.vocab_size
    mat = torch.zeros(vocab_size, vocab_size, dtype=dtype)

    valid_mask = store.valid_track_mask

    for i in range(vocab_size):
        if not valid_mask[i]:
            continue
        ci = store.track_counts.get(i, 0)
        if ci == 0:
            continue
        mat[i, i] = 1.0

    for (a, b), cij in store.pair_counts.items():
        ci = store.track_counts.get(a, 0)
        cj = store.track_counts.get(b, 0)
        if ci == 0 or cj == 0:
            continue
        sim = cij / math.sqrt(ci * cj)
        mat[a, b] = sim
        mat[b, a] = sim

    mat = mat * valid_mask.to(dtype).unsqueeze(0)
    mat = mat * valid_mask.to(dtype).unsqueeze(1)

    if device is not None:
        mat = mat.to(device)

    return mat


def sequential_coherence_scores_fast(
    input_ids: torch.Tensor,
    similarity_matrix: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:

    scores = similarity_matrix[input_ids]  # (B, T, V)

    if attention_mask is not None:
        scores = scores * attention_mask.to(scores.dtype).unsqueeze(-1)

    return scores
