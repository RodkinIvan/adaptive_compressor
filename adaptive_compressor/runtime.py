"""Runtime helpers shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from .models import AdaptiveCompressorConfig


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "mps":
        return torch.device("mps")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(config_payload: object) -> AdaptiveCompressorConfig:
    if isinstance(config_payload, AdaptiveCompressorConfig):
        return config_payload
    if is_dataclass(config_payload):
        return AdaptiveCompressorConfig(**asdict(config_payload))
    if isinstance(config_payload, dict):
        return AdaptiveCompressorConfig(**config_payload)
    raise TypeError(f"Unsupported config payload type: {type(config_payload)!r}")


def load_checkpoint(checkpoint_path: Path) -> dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
