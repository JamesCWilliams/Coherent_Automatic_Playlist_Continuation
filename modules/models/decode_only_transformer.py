from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    num_tracks: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 128
    pad_idx: int = 0
    tie_weights: bool = True


class PosEmbedding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        return self.embedding(positions)


class CausalTransformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class DecodeOnlyTransformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.track_embedding = nn.Embedding(
            config.num_tracks,
            config.d_model,
            padding_idx=config.pad_idx,
        )
        self.pos_embedding = PosEmbedding(config.max_seq_len, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        self.final_ln = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.num_tracks, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.track_embedding.weight

    def make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def encode_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f'Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}'
            )

        x = self.track_embedding(input_ids) + self.pos_embedding(input_ids)
        x = self.input_dropout(x)
        causal_mask = self.make_causal_mask(seq_len, input_ids.device)

        if attention_mask is None:
            key_padding_mask = (input_ids == self.config.pad_idx)
        else:
            key_padding_mask = (attention_mask == 0)

        for block in self.blocks:
            x = block(x, causal_mask=causal_mask, key_padding_mask=key_padding_mask)

        return self.final_ln(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.lm_head(self.encode_tokens(input_ids, attention_mask))
