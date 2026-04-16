"""Adaptive hierarchical model."""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from ..routing import (
    RoutingInfo,
    build_routing,
    gather_parent_tokens,
    make_byte_targets,
    make_meta_targets,
    select_level0_borders,
    select_meta_borders,
)
from .modules import (
    AdaptiveCompressorConfig,
    DecoderBlock,
    GRUBlock,
    cumulative_border_mask,
    entropy_regularizer,
)


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
        entropy_regularizers: list[torch.Tensor] = []

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
        byte_probabilities = byte_encoder_logits.softmax(dim=-1)
        byte_entropy = -(
            byte_probabilities * byte_probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        entropy_regularizers.append(
            entropy_regularizer(byte_entropy, floor=self.config.entropy_floor)
        )

        if self.config.num_levels > 1:
            if self.config.border_mode == "teacher_forced":
                border_mask = select_level0_borders(byte_encoder_logits, byte_targets)
            else:
                border_mask = cumulative_border_mask(
                    byte_entropy,
                    threshold=self.config.byte_entropy_threshold,
                    lengths=level_lengths[0],
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
                    raw_uncertainty = predicted_uncertainty.squeeze(-1)
                    border_mask = cumulative_border_mask(
                        raw_uncertainty,
                        threshold=thresholds[level_idx - 1],
                        lengths=current_lengths,
                    )
                    border_mask = border_mask & valid_mask
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
        total_entropy_reg = (
            torch.stack(entropy_regularizers).mean()
            if entropy_regularizers
            else byte_decoder_loss.new_zeros(())
        )
        total_loss = (
            self.config.encoder_loss_weight * byte_encoder_loss
            + self.config.decoder_loss_weight * byte_decoder_loss
            + self.config.meta_loss_weight * total_meta_loss
            + self.config.uncertainty_loss_weight * total_uncertainty_loss
            + self.config.entropy_reg_weight * total_entropy_reg
        )

        return {
            "loss": total_loss,
            "byte_encoder_loss": byte_encoder_loss,
            "byte_decoder_loss": byte_decoder_loss,
            "meta_loss": total_meta_loss,
            "uncertainty_loss": total_uncertainty_loss,
            "entropy_reg_loss": total_entropy_reg,
            "byte_encoder_logits": byte_encoder_logits,
            "byte_decoder_logits": byte_decoder_logits,
            "border_counts": border_counts,
            "level_inputs": level_inputs,
            "level_hidden_states": level_hidden_states,
        }
