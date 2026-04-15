"""Minimal PyTorch implementation of the adaptive compressor architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .routing import (
    RoutingInfo,
    broadcast_parent_to_child,
    build_routing,
    ensure_nontrivial_borders,
    gather_parent_tokens,
    make_byte_targets,
    make_meta_targets,
    select_level0_borders,
    select_meta_borders,
    select_meta_uncertainty_borders,
)


@dataclass
class AdaptiveCompressorConfig:
    vocab_size: int = 257
    byte_eos_id: int = 256
    hidden_size: int = 128
    num_levels: int = 3
    threshold: float = 0.1
    border_mode: str = "uncertainty"
    byte_entropy_threshold: float = 0.0
    meta_uncertainty_threshold: float = 0.0
    min_border_fraction: float = 0.25
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


def normalize_sequence_scores(
    scores: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize scores per sequence to make thresholding scale-robust."""

    batch_size, seq_len = scores.shape
    normalized = scores.new_zeros(scores.shape)

    if lengths is None:
        lengths = torch.full(
            (batch_size,), seq_len, device=scores.device, dtype=torch.long
        )

    for batch_idx in range(batch_size):
        length = int(lengths[batch_idx].item())
        if length <= 1:
            continue
        valid_scores = scores[batch_idx, 1:length]
        mean = valid_scores.mean()
        std = valid_scores.std(unbiased=False).clamp_min(1e-6)
        normalized[batch_idx, 1:length] = (valid_scores - mean) / std

    return normalized


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


class SimpleByteGRUBaseline(nn.Module):
    """Plain byte-level baseline with one stacked GRU and one LM loss."""

    def __init__(self, config: AdaptiveCompressorConfig) -> None:
        super().__init__()
        if config.num_levels < 1:
            raise ValueError("num_levels must be at least 1")

        self.config = config
        self.byte_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2 * config.num_levels,
            batch_first=True,
            dropout=config.dropout,
        )
        self.byte_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self, byte_ids: torch.Tensor
    ) -> dict[str, torch.Tensor | list[int] | list[torch.Tensor]]:
        byte_targets = make_byte_targets(byte_ids, eos_id=self.config.byte_eos_id)
        embeddings = self.byte_embedding(byte_ids)
        hidden_states, _ = self.gru(embeddings)
        byte_logits = self.byte_head(hidden_states)
        byte_loss = F.cross_entropy(
            byte_logits.reshape(-1, self.config.vocab_size),
            byte_targets.reshape(-1),
        )

        return {
            "loss": byte_loss,
            "byte_encoder_loss": byte_loss,
            "byte_decoder_loss": byte_loss,
            "meta_loss": byte_loss.new_zeros(()),
            "uncertainty_loss": byte_loss.new_zeros(()),
            "byte_encoder_logits": byte_logits,
            "byte_decoder_logits": byte_logits,
            "border_counts": [],
            "level_inputs": [embeddings],
            "level_hidden_states": [hidden_states],
        }


class AdaptiveCompressor(nn.Module):
    """Hierarchical adaptive compressor with mirrored decoder."""

    def __init__(self, config: AdaptiveCompressorConfig) -> None:
        super().__init__()
        if config.num_levels < 1:
            raise ValueError("num_levels must be at least 1")
        if config.border_mode not in {"uncertainty", "teacher_forced"}:
            raise ValueError("border_mode must be 'uncertainty' or 'teacher_forced'")
        if config.border_mode == "teacher_forced":
            warnings.warn(
                "border_mode='teacher_forced' uses target-dependent routing and leaks future information. "
                "Keep it for prefill-style experiments only.",
                stacklevel=2,
            )

        self.config = config
        self.byte_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder_blocks = nn.ModuleList(
            [
                GRUBlock(config.hidden_size, config.dropout)
                for _ in range(config.num_levels)
            ]
        )
        self.meta_predictors = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(config.num_levels - 1)
            ]
        )
        self.meta_uncertainty_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, 1) for _ in range(config.num_levels - 1)]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(config.hidden_size, config.dropout)
                for _ in range(config.num_levels - 1)
            ]
        )
        self.byte_encoder_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.byte_decoder_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.meta_eos_embeddings = nn.ParameterList(
            [
                nn.Parameter(torch.randn(config.hidden_size) * 0.02)
                for _ in range(config.num_levels - 1)
            ]
        )

    def forward(
        self, byte_ids: torch.Tensor
    ) -> dict[str, torch.Tensor | list[int] | list[torch.Tensor]]:
        thresholds = self.config.thresholds()
        byte_targets = make_byte_targets(byte_ids, eos_id=self.config.byte_eos_id)

        level_inputs: list[torch.Tensor] = []
        level_hidden_states: list[torch.Tensor] = []
        level_lengths: list[torch.Tensor] = []
        routings: list[RoutingInfo] = []
        border_counts: list[int] = []
        meta_losses: list[torch.Tensor] = []
        uncertainty_losses: list[torch.Tensor] = []

        level0_inputs = self.byte_embedding(byte_ids)
        level_inputs.append(level0_inputs)
        level_lengths.append(
            torch.full(
                (byte_ids.size(0),),
                byte_ids.size(1),
                device=byte_ids.device,
                dtype=torch.long,
            )
        )

        level0_hidden = self.encoder_blocks[0](level0_inputs)
        level_hidden_states.append(level0_hidden)

        byte_encoder_logits = self.byte_encoder_head(level0_hidden)
        byte_encoder_loss = F.cross_entropy(
            byte_encoder_logits.reshape(-1, self.config.vocab_size),
            byte_targets.reshape(-1),
        )

        if self.config.num_levels > 1:
            if self.config.border_mode == "teacher_forced":
                border_mask = select_level0_borders(byte_encoder_logits, byte_targets)
            else:
                probabilities = byte_encoder_logits.softmax(dim=-1)
                entropy_scores = -(
                    probabilities * probabilities.clamp_min(1e-8).log()
                ).sum(dim=-1)
                normalized_entropy = normalize_sequence_scores(entropy_scores)
                border_mask = normalized_entropy.gt(self.config.byte_entropy_threshold)
                border_mask = ensure_nontrivial_borders(
                    border_mask,
                    normalized_entropy,
                    min_border_fraction=self.config.min_border_fraction,
                )
            routing = build_routing(border_mask)
            routings.append(routing)
            compressed_inputs = gather_parent_tokens(level0_hidden, routing)
            border_counts.append(int(routing.parent_lengths.float().mean().item()))
        else:
            compressed_inputs = level0_hidden

        for level_idx in range(1, self.config.num_levels):
            current_lengths = routings[level_idx - 1].parent_lengths
            level_inputs.append(compressed_inputs)
            level_lengths.append(current_lengths)

            hidden_states = self.encoder_blocks[level_idx](
                compressed_inputs, lengths=current_lengths
            )
            level_hidden_states.append(hidden_states)

            meta_predictions = self.meta_predictors[level_idx - 1](hidden_states)
            predicted_uncertainty = F.softplus(
                self.meta_uncertainty_heads[level_idx - 1](hidden_states)
            )
            meta_targets, valid_mask = make_meta_targets(
                compressed_inputs,
                lengths=current_lengths,
                eos_embedding=self.meta_eos_embeddings[level_idx - 1],
            )
            mse_per_position = (meta_predictions - meta_targets).pow(2).mean(dim=-1)
            meta_loss = mse_per_position[valid_mask].mean()
            meta_losses.append(meta_loss)
            uncertainty_loss = F.mse_loss(
                predicted_uncertainty.squeeze(-1)[valid_mask],
                mse_per_position.detach()[valid_mask],
            )
            uncertainty_losses.append(uncertainty_loss)

            if level_idx < self.config.num_levels - 1:
                if self.config.border_mode == "teacher_forced":
                    border_mask = select_meta_borders(
                        meta_predictions,
                        meta_targets,
                        valid_mask,
                        threshold=self.config.threshold,
                    )
                else:
                    normalized_uncertainty = normalize_sequence_scores(
                        predicted_uncertainty.squeeze(-1),
                        lengths=current_lengths,
                    )
                    border_mask = select_meta_uncertainty_borders(
                        normalized_uncertainty.unsqueeze(-1),
                        valid_mask,
                        threshold=thresholds[level_idx - 1],
                    )
                    border_mask = ensure_nontrivial_borders(
                        border_mask,
                        normalized_uncertainty,
                        lengths=current_lengths,
                        min_border_fraction=self.config.min_border_fraction,
                    )
                routing = build_routing(border_mask, lengths=current_lengths)
                routings.append(routing)
                compressed_inputs = gather_parent_tokens(hidden_states, routing)
                border_counts.append(int(routing.parent_lengths.float().mean().item()))

        reconstructed = level_hidden_states[-1]

        for child_level in reversed(range(self.config.num_levels - 1)):
            child_inputs = level_inputs[child_level]
            child_lengths = level_lengths[child_level] if child_level > 0 else None
            reconstructed = self.decoder_blocks[child_level](
                child_inputs=child_inputs,
                parent_states=reconstructed,
                routing=routings[child_level],
                child_lengths=child_lengths,
            )

        byte_decoder_logits = self.byte_decoder_head(reconstructed)
        byte_decoder_loss = F.cross_entropy(
            byte_decoder_logits.reshape(-1, self.config.vocab_size),
            byte_targets.reshape(-1),
        )

        total_meta_loss = (
            torch.stack(meta_losses).mean()
            if meta_losses
            else byte_decoder_loss.new_zeros(())
        )
        total_uncertainty_loss = (
            torch.stack(uncertainty_losses).mean()
            if uncertainty_losses
            else byte_decoder_loss.new_zeros(())
        )
        total_loss = (
            self.config.encoder_loss_weight * byte_encoder_loss
            + self.config.decoder_loss_weight * byte_decoder_loss
            + self.config.meta_loss_weight * total_meta_loss
            + self.config.uncertainty_loss_weight * total_uncertainty_loss
        )

        return {
            "loss": total_loss,
            "byte_encoder_loss": byte_encoder_loss,
            "byte_decoder_loss": byte_decoder_loss,
            "meta_loss": total_meta_loss,
            "uncertainty_loss": total_uncertainty_loss,
            "byte_encoder_logits": byte_encoder_logits,
            "byte_decoder_logits": byte_decoder_logits,
            "border_counts": border_counts,
            "level_inputs": level_inputs,
            "level_hidden_states": level_hidden_states,
        }


def build_model(
    model_type: str,
    config: AdaptiveCompressorConfig,
) -> AdaptiveCompressor | SimpleByteGRUBaseline:
    if model_type == "adaptive":
        return AdaptiveCompressor(config)
    if model_type == "baseline":
        return SimpleByteGRUBaseline(config)
    raise ValueError(f"Unknown model_type: {model_type}")
