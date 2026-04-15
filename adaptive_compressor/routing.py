"""Utilities for adaptive border selection and span routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RoutingInfo:
    """Mapping between a child sequence and its compressed parent sequence."""

    parent_positions: torch.Tensor
    parent_lengths: torch.Tensor
    child_to_parent: torch.Tensor


def make_byte_targets(byte_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Shift bytes left and use EOS as the last target."""

    targets = torch.empty_like(byte_ids)
    targets[:, :-1] = byte_ids[:, 1:]
    targets[:, -1] = eos_id
    return targets


def build_routing(
    border_mask: torch.Tensor, lengths: torch.Tensor | None = None
) -> RoutingInfo:
    """Build parent token positions and child-to-parent span assignments.

    Position 0 is always forced to be a border for every non-empty sequence.
    """

    batch_size, seq_len = border_mask.shape
    device = border_mask.device

    if lengths is None:
        lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)

    masks = border_mask.clone()
    parent_positions_list: list[torch.Tensor] = []
    child_to_parent = torch.zeros(
        (batch_size, seq_len), device=device, dtype=torch.long
    )
    max_parent_len = 0

    for batch_idx in range(batch_size):
        length = int(lengths[batch_idx].item())
        if length <= 0:
            parent_positions = torch.empty(0, device=device, dtype=torch.long)
            parent_positions_list.append(parent_positions)
            continue

        masks[batch_idx, length:] = False
        masks[batch_idx, 0] = True
        parent_positions = torch.nonzero(
            masks[batch_idx, :length], as_tuple=False
        ).squeeze(-1)
        parent_positions_list.append(parent_positions)
        max_parent_len = max(max_parent_len, int(parent_positions.numel()))

        child_parent_ids = (
            torch.bucketize(
                torch.arange(length, device=device), parent_positions, right=True
            )
            - 1
        )
        child_to_parent[batch_idx, :length] = child_parent_ids.clamp_min(0)

    padded_positions = torch.zeros(
        (batch_size, max_parent_len), device=device, dtype=torch.long
    )
    parent_lengths = torch.zeros((batch_size,), device=device, dtype=torch.long)

    for batch_idx, parent_positions in enumerate(parent_positions_list):
        parent_len = int(parent_positions.numel())
        parent_lengths[batch_idx] = parent_len
        if parent_len > 0:
            padded_positions[batch_idx, :parent_len] = parent_positions

    return RoutingInfo(
        parent_positions=padded_positions,
        parent_lengths=parent_lengths,
        child_to_parent=child_to_parent,
    )


def gather_parent_tokens(
    child_states: torch.Tensor, routing: RoutingInfo
) -> torch.Tensor:
    """Select parent token states from child states using routing positions."""

    batch_size, _, hidden_dim = child_states.shape
    max_parent_len = routing.parent_positions.size(1)
    gathered = child_states.new_zeros((batch_size, max_parent_len, hidden_dim))

    for batch_idx in range(batch_size):
        parent_len = int(routing.parent_lengths[batch_idx].item())
        if parent_len == 0:
            continue
        positions = routing.parent_positions[batch_idx, :parent_len]
        gathered[batch_idx, :parent_len] = child_states[batch_idx, positions]

    return gathered


def make_meta_targets(
    inputs: torch.Tensor, lengths: torch.Tensor, eos_embedding: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift compressed embeddings left and use EOS embedding as final target."""

    batch_size, seq_len, hidden_dim = inputs.shape
    targets = inputs.new_zeros((batch_size, seq_len, hidden_dim))
    valid_mask = torch.zeros(
        (batch_size, seq_len), device=inputs.device, dtype=torch.bool
    )

    for batch_idx in range(batch_size):
        length = int(lengths[batch_idx].item())
        if length <= 0:
            continue
        if length > 1:
            targets[batch_idx, : length - 1] = inputs[batch_idx, 1:length]
        targets[batch_idx, length - 1] = eos_embedding
        valid_mask[batch_idx, :length] = True

    return targets, valid_mask


def broadcast_parent_to_child(
    parent_states: torch.Tensor, routing: RoutingInfo, child_length: int
) -> torch.Tensor:
    """Broadcast each parent state across the child span it owns."""

    batch_size, _, hidden_dim = parent_states.shape
    broadcast = parent_states.new_zeros((batch_size, child_length, hidden_dim))

    for batch_idx in range(batch_size):
        parent_len = int(routing.parent_lengths[batch_idx].item())
        if parent_len == 0:
            continue
        child_parent_ids = routing.child_to_parent[batch_idx, :child_length]
        broadcast[batch_idx] = parent_states[batch_idx, child_parent_ids]

    return broadcast


def select_level0_borders(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Promote byte positions whose next-byte prediction is wrong."""

    predictions = logits.argmax(dim=-1)
    return predictions.ne(targets)


def select_meta_borders(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Promote compressed positions whose next-embedding prediction MSE is high."""

    mse = (predictions - targets).pow(2).mean(dim=-1)
    border_mask = mse.gt(threshold)
    return border_mask & valid_mask
