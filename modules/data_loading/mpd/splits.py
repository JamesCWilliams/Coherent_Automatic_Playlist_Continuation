import random
from typing import Iterable


def split_playlists_by_pid(
    playlists: Iterable[dict],
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    
    playlists = list(playlists)

    val_frac = (1 - train_frac) / 2

    rng = random.Random(seed)
    rng.shuffle(playlists)

    n = len(playlists)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = playlists[:n_train]
    val = playlists[n_train:n_train + n_val]
    test = playlists[n_train + n_val:]

    return train, val, test
