"""Non-hierarchical baseline models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..routing import make_byte_targets
from .modules import (
    AdaptiveCompressorConfig,
    ResidualGRUBlock,
    count_parameters,
)


class ResidualByteBaseline(nn.Module):
    """Byte-only baseline with residual GRU blocks and no hierarchy."""

    def __init__(
        self, config: AdaptiveCompressorConfig, target_parameter_count: int
    ) -> None:
        super().__init__()
        if config.num_levels < 1:
            raise ValueError("num_levels must be at least 1")

        self.config = config
        self.byte_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.num_blocks = self._choose_num_blocks(config, target_parameter_count)
        self.blocks = nn.ModuleList(
            [
                ResidualGRUBlock(config.hidden_size, config.dropout)
                for _ in range(self.num_blocks)
            ]
        )
        self.byte_head = nn.Linear(config.hidden_size, config.vocab_size)
        residual_parameters = target_parameter_count - count_parameters(self)
        if residual_parameters < 0:
            raise ValueError("Baseline parameter count exceeded adaptive target")
        self.parameter_match_pad = nn.Parameter(torch.zeros(residual_parameters))

    @staticmethod
    def _choose_num_blocks(
        config: AdaptiveCompressorConfig,
        target_parameter_count: int,
    ) -> int:
        best_blocks = 1
        for num_blocks in range(1, 8 * config.num_levels + 1):
            probe = nn.Module()
            probe.byte_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            probe.blocks = nn.ModuleList(
                [
                    ResidualGRUBlock(config.hidden_size, config.dropout)
                    for _ in range(num_blocks)
                ]
            )
            probe.byte_head = nn.Linear(config.hidden_size, config.vocab_size)
            parameter_count = count_parameters(probe)
            if parameter_count > target_parameter_count:
                break
            best_blocks = num_blocks
        return best_blocks

    def forward(
        self, byte_ids: torch.Tensor
    ) -> dict[str, torch.Tensor | list[int] | list[torch.Tensor]]:
        byte_targets = make_byte_targets(byte_ids, eos_id=self.config.byte_eos_id)
        embeddings = self.byte_embedding(byte_ids)
        hidden_states = embeddings
        block_outputs: list[torch.Tensor] = []

        for block in self.blocks:
            hidden_states = block(hidden_states)
            block_outputs.append(hidden_states)

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
            "entropy_reg_loss": byte_loss.new_zeros(()),
            "byte_encoder_logits": byte_logits,
            "byte_decoder_logits": byte_logits,
            "border_counts": [],
            "level_inputs": [embeddings],
            "level_hidden_states": block_outputs or [hidden_states],
        }
