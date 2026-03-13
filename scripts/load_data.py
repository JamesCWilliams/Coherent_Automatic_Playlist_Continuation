import json
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, List, Dict, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class MPDConfig:
    """
    Container class for reading the Spotify Million Playlist Dataset.
    """
    data_dir: str
    min_track_freq: int = 1
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
        data_dir = Path.cwd().parent / 'datasets' / 'MPD' / 'data'

    for path in sorted(data_dir.glob('*.json')):
        yield path


def iter_playlists(
        data_dir: Path | None = None
) -> Iterator[dict]:
    """
    Dataset is huge; get slices on-demand (playlists)
    """

    if data_dir is None:
        data_dir = Path.cwd().parent / 'datasets' / 'MPD' / 'data'

    for slice in iter_mpd_slice_files(data_dir):
        with open(slice, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        for playlist in obj.get('playlists', []):
            yield playlist


def playlist_to_track_sequence(
        playlist: dict
) -> List[str]:
    """
    Convert one playlist dict per iter_playlists() into an ordered list of track_uri tokens.
    """

    tracks = playlist.get('tracks', [])
    seq = [track['track_uri'] for track in tracks if 'track_uri' in track]
    return seq


class TrackVocab:
    def __init__(
            self,
            stoi: Dict[str, int],
            itos: List[str],
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




if __name__ == '__main__':
