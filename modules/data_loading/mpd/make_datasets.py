import torch
from torch.utils.data import DataLoader

from .reader import MPDConfig, iter_playlists
from .vocab import build_track_vocab
from .encoding import collect_encoded_playlists
from .dataset import MPDDataset


def next_track_collate_fn(
        batch: list[dict[str, torch.Tensor]],
        pad_idx: int
) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(item['input_ids'].size(0) for item in batch)

    input_ids = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        input_ids[i, :seq_len] = item['input_ids']
        labels[i, :seq_len] = item['labels']
        mask[i, :seq_len] = item['attention_mask']

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': mask,
    }


def _assign_split_from_pid(pid: int, train_frac: float) -> str:
    if not (0.0 < train_frac < 1.0):
        raise ValueError('train_frac must be between 0 and 1.')

    train_cutoff = int(train_frac * 100)
    remaining = 100 - train_cutoff
    val_cutoff = train_cutoff + (remaining // 2)

    bucket = pid % 100

    if bucket < train_cutoff:
        return 'train'
    elif bucket < val_cutoff:
        return 'val'
    else:
        return 'test'


def collect_split_playlists_streaming(
    config: MPDConfig,
) -> tuple[list[dict], list[dict], list[dict]]:
    train_playlists, val_playlists, test_playlists = [], [], []

    max_train = config.max_train_playlists
    if max_train is not None:
        max_val = int(max_train * (1 - config.train) / 2)
        max_test = int(max_train * (1 - config.train) / 2)
    else:
        max_val = None
        max_test = None

    for playlist in iter_playlists(config.data_dir):
        pid = playlist.get('pid')
        if pid is None:
            continue

        split = _assign_split_from_pid(pid, config.train)

        if split == 'train':
            if max_train is None or len(train_playlists) < max_train:
                train_playlists.append(playlist)
        elif split == 'val':
            if max_val is None or len(val_playlists) < max_val:
                val_playlists.append(playlist)
        else:
            if max_test is None or len(test_playlists) < max_test:
                test_playlists.append(playlist)

        if (
            max_train is not None
            and len(train_playlists) >= max_train
            and len(val_playlists) >= max_val
            and len(test_playlists) >= max_test
        ):
            break

    return train_playlists, val_playlists, test_playlists


def make_mpd_loaders(
    config: MPDConfig | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    if config is None:
        config = MPDConfig()

    train_playlists, val_playlists, test_playlists = collect_split_playlists_streaming(config)

    print(f'Train playlists collected: {len(train_playlists)}')
    print(f'Val playlists collected: {len(val_playlists)}')
    print(f'Test playlists collected: {len(test_playlists)}')

    vocab = build_track_vocab(config, train_playlists)

    print(f'Vocab size: {len(vocab)}')

    train_sequences = collect_encoded_playlists(config, vocab, train_playlists)
    val_sequences = collect_encoded_playlists(config, vocab, val_playlists)
    test_sequences = collect_encoded_playlists(config, vocab, test_playlists)

    print(f'Train sequences kept: {len(train_sequences)}')
    print(f'Val sequences kept: {len(val_sequences)}')
    print(f'Test sequences kept: {len(test_sequences)}')

    train_dataset = MPDDataset(train_sequences, pad_idx=vocab.pad_idx)
    val_dataset = MPDDataset(val_sequences, pad_idx=vocab.pad_idx)
    test_dataset = MPDDataset(test_sequences, pad_idx=vocab.pad_idx)

    def collate_fn(batch):
        return next_track_collate_fn(batch, pad_idx=vocab.pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, vocab, train_sequences
