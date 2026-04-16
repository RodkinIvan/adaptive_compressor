"""Model factory helpers."""

from __future__ import annotations

from .adaptive import AdaptiveCompressor
from .baseline import ResidualByteBaseline
from .modules import AdaptiveCompressorConfig, count_parameters


def build_model(
    model_type: str,
    config: AdaptiveCompressorConfig,
) -> AdaptiveCompressor | ResidualByteBaseline:
    if model_type == "adaptive":
        return AdaptiveCompressor(config)
    if model_type == "baseline":
        target_parameter_count = count_parameters(AdaptiveCompressor(config))
        return ResidualByteBaseline(
            config, target_parameter_count=target_parameter_count
        )
    raise ValueError(f"Unknown model_type: {model_type}")
