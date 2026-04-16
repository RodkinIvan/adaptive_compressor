"""Shared modules and utilities for adaptive compressor models."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..routing import RoutingInfo, broadcast_parent_to_child


@dataclass
class AdaptiveCompressorConfig:
    vocab_size: int = 257
    byte_eos_id: int = 256
    hidden_size: int = 128
    num_levels: int = 3
    threshold: float = 0.1
    border_mode: str = "uncertainty"
    byte_entropy_threshold: float = 20.0
    meta_uncertainty_threshold: float = 1.0
    entropy_floor: float = 0.0
    entropy_reg_weight: float = 0.001
    dropout: float = 0.0
    encoder_loss_weight: float = 1.0
    decoder_loss_weight: float = 1.0
    meta_loss_weight: float = 1.0
    uncertainty_loss_weight: float = 0.1
    level_thresholds: list[float] = field(default_factory=list)

    def thresholds(self) -> list[float]:
        if self.level_thresholds:
            return self.level_thresholds
        return [
            self.meta_uncertainty_threshold for _ in range(max(self.num_levels - 1, 0))
        ]


class GRUBlock(nn.Module):
    """Small wrapper around a 2-layer GRU with optional packed inputs."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        if lengths is None:
            outputs, _ = self.gru(inputs)
            return outputs

        if int(lengths.max().item()) == inputs.size(1):
            outputs, _ = self.gru(inputs)
            return outputs

        packed = pack_padded_sequence(
            inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.gru(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=inputs.size(1)
        )
        return outputs


class ResidualGRUBlock(nn.Module):
    """Two-layer GRU block with a residual stream."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.gru_block = GRUBlock(hidden_size=hidden_size, dropout=dropout)

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        return inputs + self.gru_block(inputs, lengths=lengths)


class DecoderBlock(nn.Module):
    """Child-level decoder conditioned by broadcast parent states."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.parent_projection = nn.Linear(hidden_size, hidden_size)
        self.gru = GRUBlock(hidden_size=hidden_size, dropout=dropout)

    def forward(
        self,
        child_inputs: torch.Tensor,
        parent_states: torch.Tensor,
        routing: RoutingInfo | None = None,
        child_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if routing is None:
            parent_broadcast = parent_states
        else:
            parent_broadcast = broadcast_parent_to_child(
                parent_states,
                routing,
                child_inputs.size(1),
                child_lengths=child_lengths,
            )
        conditioned = child_inputs + self.parent_projection(parent_broadcast)
        return self.gru(conditioned, lengths=child_lengths)


def cumulative_border_mask(
    scores: torch.Tensor,
    threshold: float,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Emit a border when cumulative span uncertainty crosses a threshold."""

    batch_size, seq_len = scores.shape
    border_mask = torch.zeros_like(scores, dtype=torch.bool)

    if lengths is None:
        lengths = torch.full(
            (batch_size,), seq_len, device=scores.device, dtype=torch.long
        )

    for batch_idx in range(batch_size):
        length = int(lengths[batch_idx].item())
        if length <= 0:
            continue
        border_mask[batch_idx, 0] = True
        running_score = scores.new_zeros(())
        for position in range(1, length):
            running_score = running_score + scores[batch_idx, position]
            if running_score >= threshold:
                border_mask[batch_idx, position] = True
                running_score = scores.new_zeros(())

    return border_mask


def entropy_regularizer(entropy: torch.Tensor, floor: float) -> torch.Tensor:
    """Penalize entropy values that fall below a small floor."""

    if floor <= 0:
        return entropy.new_zeros(())
    return (floor - entropy).clamp_min(0.0).pow(2).mean()


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in a module."""

    return sum(parameter.numel() for parameter in module.parameters())
