from collections import Counter
from tqdm.auto import tqdm
from .reader import playlist_to_track_sequence, MPDConfig


class TrackVocab:
    def __init__(
            self,
            str_to_idx: dict[str, int],
            idx_to_str: list[str],
            pad_idx: int,
            bos_idx: int,
            eos_idx: int,
            unk_idx: int,
            msk_idx: int
    ):
        self.str_to_idx = str_to_idx
        self.idx_to_str = idx_to_str
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx
        self.msk_idx = msk_idx

    def __len__(self):
        return len(self.idx_to_str)

    def encode_token(self, token: str) -> int:
        return self.str_to_idx.get(token, self.unk_idx)

    def decode_token(self, idx: int) -> str:
        return self.idx_to_str[idx]


def build_track_vocab(config: MPDConfig, playlists) -> TrackVocab:
    counter = Counter()
    for playlist in tqdm(playlists):
        seq = playlist_to_track_sequence(playlist)
        counter.update(seq)

    specials = [
        config.pad_token,
        config.bos_token,
        config.eos_token,
        config.unk_token,
        config.msk_token,
    ]

    track_tokens = sorted(
        uri for uri, freq in counter.items() if freq >= config.min_track_freq
    )

    idx_to_str = specials + track_tokens
    str_to_idx = {tok: i for i, tok in enumerate(idx_to_str)}

    return TrackVocab(
        str_to_idx=str_to_idx,
        idx_to_str=idx_to_str,
        pad_idx=str_to_idx[config.pad_token],
        bos_idx=str_to_idx[config.bos_token],
        eos_idx=str_to_idx[config.eos_token],
        unk_idx=str_to_idx[config.unk_token],
        msk_idx=str_to_idx[config.msk_token],
    )
