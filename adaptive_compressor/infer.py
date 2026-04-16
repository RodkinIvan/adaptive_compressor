"""Autoregressive inference utilities for adaptive compressor models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import (
    AdaptiveCompressor,
    AdaptiveCompressorConfig,
    SimpleByteGRUBaseline,
    build_model,
)


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
    parser.add_argument(
        "--inference-mode",
        choices=["recompute", "cached"],
        default="cached",
    )
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


def step_gru(
    gru: torch.nn.GRU,
    hidden: torch.Tensor,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs, next_hidden = gru(inputs.view(1, 1, -1), hidden)
    return outputs[0, 0], next_hidden


class CachedBaselineGenerator:
    """Exact linear-time generator for the plain byte GRU baseline."""

    def __init__(self, model: SimpleByteGRUBaseline, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.hidden = torch.zeros(
            model.gru.num_layers,
            1,
            model.config.hidden_size,
            device=device,
        )

    def step(self, token_id: int) -> torch.Tensor:
        token = torch.tensor([token_id], device=self.device)
        embeddings = self.model.byte_embedding(token)[0]
        hidden_state, self.hidden = step_gru(self.model.gru, self.hidden, embeddings)
        return self.model.byte_head(hidden_state)


class CachedAdaptiveGenerator:
    """Event-driven cached generator for the adaptive model.

    This follows the same cumulative-uncertainty routing rule used during training.
    """

    def __init__(self, model: AdaptiveCompressor, device: torch.device) -> None:
        if model.config.border_mode != "uncertainty":
            raise ValueError(
                "Cached adaptive inference only supports border_mode='uncertainty'"
            )

        self.model = model
        self.device = device
        hidden_size = model.config.hidden_size
        self.encoder_hidden = [
            torch.zeros(2, 1, hidden_size, device=device)
            for _ in range(model.config.num_levels)
        ]
        self.decoder_hidden = [
            torch.zeros(2, 1, hidden_size, device=device)
            for _ in range(model.config.num_levels - 1)
        ]
        self.active_parent = [
            torch.zeros(hidden_size, device=device)
            for _ in range(model.config.num_levels - 1)
        ]
        self.level_token_counts = [0 for _ in range(model.config.num_levels)]
        self.emitted_token_counts = [0 for _ in range(model.config.num_levels)]
        self.running_uncertainty = [
            torch.zeros((), device=device) for _ in range(model.config.num_levels - 1)
        ]

    def _should_emit_border(self, level_idx: int, raw_score: torch.Tensor) -> bool:
        self.level_token_counts[level_idx] += 1
        if self.level_token_counts[level_idx] == 1:
            self.running_uncertainty[level_idx] = torch.zeros((), device=self.device)
            return True

        if level_idx == 0:
            threshold = self.model.config.byte_entropy_threshold
        else:
            threshold = self.model.config.thresholds()[level_idx - 1]

        self.running_uncertainty[level_idx] = (
            self.running_uncertainty[level_idx] + raw_score
        )
        border = bool(self.running_uncertainty[level_idx].item() >= threshold)
        if border:
            self.running_uncertainty[level_idx] = torch.zeros((), device=self.device)
        return border

    def _step_decoder(
        self,
        level_idx: int,
        child_input: torch.Tensor,
        parent_state: torch.Tensor,
    ) -> torch.Tensor:
        decoder = self.model.decoder_blocks[level_idx]
        conditioned = child_input + decoder.parent_projection(parent_state)
        output, next_hidden = step_gru(
            decoder.gru.gru, self.decoder_hidden[level_idx], conditioned
        )
        self.decoder_hidden[level_idx] = next_hidden
        return output

    def _emit_level_event(self, level_idx: int, token_input: torch.Tensor) -> None:
        self.emitted_token_counts[level_idx] += 1
        encoder = self.model.encoder_blocks[level_idx].gru
        encoder_output, next_hidden = step_gru(
            encoder,
            self.encoder_hidden[level_idx],
            token_input,
        )
        self.encoder_hidden[level_idx] = next_hidden

        is_border = False
        if level_idx < self.model.config.num_levels - 1:
            predicted_uncertainty = F.softplus(
                self.model.meta_uncertainty_heads[level_idx](encoder_output)
            ).squeeze(-1)
            is_border = self._should_emit_border(level_idx, predicted_uncertainty)

        if level_idx == self.model.config.num_levels - 1:
            if self.emitted_token_counts[level_idx] > 1:
                self.active_parent[level_idx - 1] = encoder_output
        else:
            parent_for_current = (
                torch.zeros_like(self.active_parent[level_idx])
                if is_border
                else self.active_parent[level_idx]
            )
            decoded_output = self._step_decoder(
                level_idx, token_input, parent_for_current
            )
            if self.emitted_token_counts[level_idx] > 1:
                self.active_parent[level_idx - 1] = decoded_output

        if level_idx >= self.model.config.num_levels - 1:
            return

        if is_border:
            self._emit_level_event(level_idx + 1, encoder_output)

    def step(self, token_id: int) -> torch.Tensor:
        token = torch.tensor([token_id], device=self.device)
        byte_embedding = self.model.byte_embedding(token)[0]

        level0_output, next_hidden = step_gru(
            self.model.encoder_blocks[0].gru,
            self.encoder_hidden[0],
            byte_embedding,
        )
        self.encoder_hidden[0] = next_hidden

        byte_encoder_logits = self.model.byte_encoder_head(level0_output)
        probabilities = byte_encoder_logits.softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum()
        is_level0_border = False
        if self.model.config.num_levels > 1:
            is_level0_border = self._should_emit_border(0, entropy)

        if self.model.config.num_levels == 1:
            decoder_output = level0_output
        else:
            parent_for_current = (
                torch.zeros_like(self.active_parent[0])
                if is_level0_border
                else self.active_parent[0]
            )
            decoder_output = self._step_decoder(0, byte_embedding, parent_for_current)
        logits = self.model.byte_decoder_head(decoder_output)

        if self.model.config.num_levels > 1 and is_level0_border:
            self._emit_level_event(1, level0_output)

        return logits


def build_cached_generator(
    model: torch.nn.Module,
    device: torch.device,
) -> CachedAdaptiveGenerator | CachedBaselineGenerator:
    if isinstance(model, AdaptiveCompressor):
        return CachedAdaptiveGenerator(model, device=device)
    if isinstance(model, SimpleByteGRUBaseline):
        return CachedBaselineGenerator(model, device=device)
    raise TypeError(f"Unsupported model type for cached inference: {type(model)!r}")


def generate_bytes(
    model: torch.nn.Module,
    prompt_bytes: bytes,
    max_new_bytes: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    eos_id: int,
    inference_mode: str,
) -> bytes:
    generated = list(prompt_bytes)
    if not generated:
        raise ValueError(
            "Prompt must contain at least one byte for autoregressive generation"
        )

    model.eval()
    with torch.no_grad():
        cached_generator = None
        if inference_mode == "cached":
            cached_generator = build_cached_generator(model, device=device)

        last_logits = None
        for byte_value in generated:
            if cached_generator is not None:
                last_logits = cached_generator.step(byte_value)

        for _ in range(max_new_bytes):
            if cached_generator is None:
                prefix = torch.tensor(
                    generated, dtype=torch.long, device=device
                ).unsqueeze(0)
                logits = next_byte_logits(model, prefix)[0]
            else:
                if last_logits is None:
                    raise RuntimeError("Cached inference requires a non-empty prompt")
                logits = last_logits
            next_token = sample_next_token(logits, temperature=temperature, top_k=top_k)
            if next_token == eos_id:
                break
            generated.append(next_token)
            if cached_generator is not None:
                last_logits = cached_generator.step(next_token)

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
        inference_mode=args.inference_mode,
    )
    print(generated.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
