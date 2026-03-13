"""
Turn the MPD into a DataLoader.
"""

import json
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Iterator
import time
import torch
from torch.utils.data import Dataset, DataLoader



@dataclass
class MPDConfig:
    """
    Container class for reading the Spotify Million Playlist Dataset.
    """
    data_dir: Path | None = None
    min_track_freq: int = 3
    max_seq_len: int = 128
    min_playlist_len: int = 2

    pad_token: str = '[PAD]'
    bos_token: str = '[BOS]'
    eos_token: str = '[EOS]'
    unk_token: str = '[UNK]'
    msk_token: str = '[MSK]'


def iter_mpd_slice_files(
        data_dir: Path | None = None
) -> Iterator[Path]:
    """
    Dataset is huge; get slices on-demand (paths).
    """

    if data_dir is None:
        data_dir = Path.cwd() / 'datasets' / 'MPD' / 'data'

    for path in sorted(data_dir.glob('*.json')):
        yield path


def iter_playlists(
        data_dir: Path | None = None
) -> Iterator[dict]:
    """
    Dataset is huge; get slices on-demand (playlists)
    """

    if data_dir is None:
        data_dir = Path.cwd() / 'datasets' / 'MPD' / 'data'

    for this_slice in iter_mpd_slice_files(data_dir):
        with open(this_slice, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        yield from obj.get('playlists', [])


def playlist_to_track_sequence(
        playlist: dict
) -> list[str]:
    """
    Convert one playlist dict per iter_playlists() into an ordered list of track_uri tokens.
    """

    tracks = playlist.get('tracks', [])
    seq = [track['track_uri'] for track in tracks if 'track_uri' in track]
    return seq


class TrackVocab:
    def __init__(
            self,
            stoi: dict[str, int],
            itos: list[str],
            pad_idx: int,
            bos_idx: int,
            eos_idx: int,
            unk_idx: int,
            msk_idx: int
    ):
        """
        Helper for tokenization.
        """
        self.stoi = stoi
        self.itos = itos
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx
        self.msk_idx = msk_idx

    def __len__(self):
        return len(self.itos)
    
    def encode_token(self, token: str) -> int:
        return self.stoi.get(token, self.unk_idx)
    
    def decode_token(self, idx: int) -> str:
        return self.itos[idx]


def build_track_vocab(config: MPDConfig) -> TrackVocab:
    """
    Build vocabulary from tracks if track_uri is present more than some threshold.
    """

    counter = Counter()
    for playlist in iter_playlists(config.data_dir):
        seq = playlist_to_track_sequence(playlist)
        counter.update(seq)
    
    specials = [
        config.pad_token,
        config.bos_token,
        config.eos_token,
        config.unk_token,
        config.msk_token
    ]

    track_tokens = [track_uri for track_uri, freq in counter.items() if freq >= config.min_track_freq]
    track_tokens.sort()

    itos = specials + track_tokens
    stoi = {tok: i for i, tok in enumerate(itos)}

    return TrackVocab(
        stoi=stoi,
        itos=itos,
        pad_idx=stoi[config.pad_token],
        bos_idx=stoi[config.bos_token],
        eos_idx=stoi[config.eos_token],
        unk_idx=stoi[config.unk_token],
        msk_idx=stoi[config.msk_token]
    )


def encode_playlist(
        playlist: dict,
        vocab: TrackVocab,
        max_seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True
) -> list[int]:
    """
    Turn a playlist into tokens per TrackVocab.
    """

    track_seq = playlist_to_track_sequence(playlist)
    ids = [vocab.encode_token(t) for t in track_seq]

    if add_bos:
        ids = [vocab.bos_idx] + ids
    if add_eos:
        ids = [vocab.eos_idx] + ids

    return ids[:max_seq_len]


def collect_encoded_playlists(
        config: MPDConfig,
        vocab: TrackVocab
) -> list[list[int]]:
    """
    Convert playlists to encoded integer sequences.
    """

    encoded = []
    for playlist in iter_playlists(config.data_dir):
        raw_len = len(playlist_to_track_sequence(playlist))
        if raw_len < config.min_playlist_len:
            continue

        ids = encode_playlist(
            playlist,
            vocab,
            config.max_seq_len,
        )

        if len(ids) >= 3: # beginning token, one item, ending token
            encoded.append(ids)

    return encoded


class MPDDataset(Dataset):
    """
    Takes Spotify Million-Playlist Dataset playlists (already encoded) and turns them into tensors for next-track prediction.
    """

    def __init__(
            self,
            sequences: list[list[int]],
            pad_idx: int
    ):
        
        self.sequences = [seq for seq in sequences if len(seq) >= 2]
        self.pad_idx = pad_idx

    def __len__(self) -> int:

        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)
        mask = torch.ones_like(input_ids, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': mask
        }


def next_track_collate_fn(
        batch: list[dict[str, torch.Tensor]],
        pad_idx: int
) -> dict[str, torch.Tensor]:
    """
    Sequences are of variable length, pad them so they are the same length.
    """

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
        'attention_mask': mask
    }


def make_mpd_dataloader(
        config: MPDConfig | None = None,
        batch_size: int = 32
) -> DataLoader:
    """
    Get a DataLoader straight from the raw MPD.
    """

    if config is None:
        config = MPDConfig()

    print('Building vocab...')
    build_start = time.perf_counter()
    vocab = build_track_vocab(config)
    build_finish = time.perf_counter()
    print(f'Vocab size: {len(vocab)}')
    print(f'Built vocab in {build_finish - build_start} seconds.')

    def collate_fn(batch):
        return next_track_collate_fn(batch, pad_idx=vocab.pad_idx)

    print('Encoding playlists...')
    seq_start = time.perf_counter()
    sequences = collect_encoded_playlists(config, vocab)
    seq_finish = time.perf_counter()
    print(f'Playlists kept: {len(sequences)}')
    print(f'Got sequences in {seq_finish - seq_start} seconds.')

    dataset = MPDDataset(sequences, pad_idx=vocab.pad_idx)
    dataloader = DataLoader(
        dataset,
        batch_size,
        collate_fn=collate_fn
    )

    return dataloader
