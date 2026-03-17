from typing import Optional

import torch
import torch.nn.functional as F


def expected_coherence_loss(
    logits: torch.Tensor,
    coherence_scores: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    reduction: str = 'mean',
) -> torch.Tensor:

    if logits.shape != coherence_scores.shape:
        raise ValueError(
            f'logits shape {logits.shape} must match coherence_scores shape {coherence_scores.shape}.'
        )
    probs = F.softmax(logits / temperature, dim=-1)
    per_position = -(probs * coherence_scores).sum(dim=-1)  # (B, T)
    valid_positions = torch.ones_like(per_position, dtype=torch.bool)
    if attention_mask is not None:
        valid_positions = valid_positions & attention_mask.to(torch.bool)
    if labels is not None:
        valid_positions = valid_positions & (labels != -100)
    per_position = per_position * valid_positions
    if reduction == 'none':
        return per_position
    denom = valid_positions.sum().clamp_min(1)
    if reduction == 'sum':
        return per_position.sum()
    if reduction == 'mean':
        return per_position.sum() / denom
    raise ValueError(f'Unsupported reduction: {reduction}')


def combined_ce_coherence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coherence_scores: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ce_weight: float = 1.0,
    coherence_weight: float = 0.1,
    coherence_temperature: float = 1.0,
) -> dict[str, torch.Tensor]:

    vocab_size = logits.size(-1)

    ce_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )

    coh_loss = expected_coherence_loss(
        logits=logits,
        coherence_scores=coherence_scores,
        attention_mask=attention_mask,
        labels=labels,
        temperature=coherence_temperature,
        reduction='mean',
    )

    total_loss = ce_weight * ce_loss + coherence_weight * coh_loss

    return {
        'loss': total_loss,
        'ce_loss': ce_loss,
        'coherence_loss': coh_loss,
    }
