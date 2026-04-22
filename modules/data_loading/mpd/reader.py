import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator


@dataclass
class MPDConfig:
    data_dir: Path | None = None
    min_track_freq: int = 2
    max_seq_len: int = 128
    min_playlist_len: int = 2

    pad_token: str = '[PAD]'
    bos_token: str = '[BOS]'
    eos_token: str = '[EOS]'
    unk_token: str = '[UNK]'
    msk_token: str = '[MSK]'

    train: float = 0.8
    max_train_playlists: int | None = None
    seed: int = 10


def iter_mpd_slice_files(data_dir: Path | None = None) -> Iterator[Path]:
    if data_dir is None:
        data_dir = Path.cwd() / 'datasets' / 'MPD' / 'data'
    yield from sorted(data_dir.glob('*.json'))


def iter_playlists(data_dir: Path | None = None) -> Iterator[dict]:
    if data_dir is None:
        data_dir = Path.cwd() / 'datasets' / 'MPD' / 'data'

    for this_slice in iter_mpd_slice_files(data_dir):
        with open(this_slice, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        yield from obj.get('playlists', [])


def playlist_to_track_sequence(playlist: dict) -> list[str]:
    tracks = sorted(playlist.get('tracks', []), key=lambda x: x.get('pos', 0))
    return [track['track_uri'] for track in tracks if 'track_uri' in track]
