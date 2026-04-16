"""Model package exports."""

from .adaptive import AdaptiveCompressor
from .baseline import ResidualByteBaseline
from .factory import build_model
from .modules import (
    AdaptiveCompressorConfig,
    GRUBlock,
    ResidualGRUBlock,
    cumulative_border_mask,
    count_parameters,
)

SimpleByteGRUBaseline = ResidualByteBaseline

__all__ = [
    "AdaptiveCompressor",
    "AdaptiveCompressorConfig",
    "ResidualByteBaseline",
    "SimpleByteGRUBaseline",
    "GRUBlock",
    "ResidualGRUBlock",
    "cumulative_border_mask",
    "count_parameters",
    "build_model",
]
