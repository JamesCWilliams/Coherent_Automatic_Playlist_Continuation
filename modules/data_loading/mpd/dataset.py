import torch
from torch.utils.data import Dataset


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
