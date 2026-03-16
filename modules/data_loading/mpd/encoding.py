from tqdm.auto import tqdm
from .vocab import TrackVocab
from .reader import playlist_to_track_sequence, MPDConfig


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
        ids = ids + [vocab.eos_idx]

    return ids[:max_seq_len]


def collect_encoded_playlists(
        config: MPDConfig,
        vocab: TrackVocab,
        playlists
) -> list[list[int]]:
    """
    Convert playlists to encoded integer sequences.
    """

    encoded = []
    for playlist in tqdm(playlists):
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
