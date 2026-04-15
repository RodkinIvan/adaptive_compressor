"""Autoregressive inference utilities for adaptive compressor models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from .model import AdaptiveCompressorConfig, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--max-new-bytes", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )
    parser.add_argument("--check-causality", action="store_true")
    parser.add_argument("--causality-max-positions", type=int, default=64)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
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


def load_prompt(args: argparse.Namespace) -> bytes:
    if args.prompt_file is not None:
        return args.prompt_file.read_bytes()
    return args.prompt.encode("utf-8")


def load_config(config_payload: object) -> AdaptiveCompressorConfig:
    if isinstance(config_payload, AdaptiveCompressorConfig):
        return config_payload
    if is_dataclass(config_payload):
        return AdaptiveCompressorConfig(**asdict(config_payload))
    if isinstance(config_payload, dict):
        return AdaptiveCompressorConfig(**config_payload)
    raise TypeError(f"Unsupported config payload type: {type(config_payload)!r}")


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled_logits = logits / temperature
    if top_k > 0:
        top_values, top_indices = torch.topk(
            scaled_logits, k=min(top_k, scaled_logits.size(-1))
        )
        probabilities = torch.softmax(top_values, dim=-1)
        choice = torch.multinomial(probabilities, num_samples=1)
        return int(top_indices[choice].item())

    probabilities = torch.softmax(scaled_logits, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def next_byte_logits(model: torch.nn.Module, prefix: torch.Tensor) -> torch.Tensor:
    outputs = model(prefix)
    return outputs["byte_decoder_logits"][:, -1, :]


def generate_bytes(
    model: torch.nn.Module,
    prompt_bytes: bytes,
    max_new_bytes: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    eos_id: int,
) -> bytes:
    generated = list(prompt_bytes)
    if not generated:
        raise ValueError(
            "Prompt must contain at least one byte for autoregressive generation"
        )

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_bytes):
            prefix = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(
                0
            )
            logits = next_byte_logits(model, prefix)[0]
            next_token = sample_next_token(logits, temperature=temperature, top_k=top_k)
            if next_token == eos_id:
                break
            generated.append(next_token)

    return bytes(generated)


def check_causality(
    model: torch.nn.Module,
    prompt_bytes: bytes,
    max_positions: int,
    device: torch.device,
) -> tuple[int, float]:
    if len(prompt_bytes) < 2:
        return 0, 0.0

    token_count = min(len(prompt_bytes), max_positions)
    tokens = torch.tensor(
        list(prompt_bytes[:token_count]), dtype=torch.long, device=device
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        full_logits = model(tokens)["byte_decoder_logits"][0, :token_count]
        max_difference = 0.0
        worst_position = 0

        for position in range(token_count):
            prefix = tokens[:, : position + 1]
            prefix_logits = model(prefix)["byte_decoder_logits"][0, -1]
            difference = (prefix_logits - full_logits[position]).abs().max().item()
            if difference > max_difference:
                max_difference = difference
                worst_position = position

    return worst_position, max_difference


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    config = load_config(checkpoint["config"])
    train_args = checkpoint.get("args", {})
    model_type = train_args.get("model_type", "adaptive")
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    prompt_bytes = load_prompt(args)
    if args.check_causality:
        worst_position, max_difference = check_causality(
            model=model,
            prompt_bytes=prompt_bytes,
            max_positions=args.causality_max_positions,
            device=device,
        )
        print(
            f"causality_check worst_position={worst_position} max_abs_diff={max_difference:.8f}"
        )

    generated = generate_bytes(
        model=model,
        prompt_bytes=prompt_bytes,
        max_new_bytes=args.max_new_bytes,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        eos_id=config.byte_eos_id,
    )
    print(generated.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
