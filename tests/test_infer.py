import unittest

import torch

from adaptive_compressor.infer import build_cached_generator, next_byte_logits
from adaptive_compressor.model import AdaptiveCompressorConfig, build_model


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_cached_baseline_matches_recompute(self) -> None:
        config = AdaptiveCompressorConfig(hidden_size=24, num_levels=3)
        model = build_model("baseline", config)
        model.eval()

        tokens = list(b"Hello")
        generator = build_cached_generator(model, torch.device("cpu"))
        cached_logits = None
        for token in tokens:
            cached_logits = generator.step(token)

        prefix = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        recompute_logits = next_byte_logits(model, prefix)[0]

        self.assertIsNotNone(cached_logits)
        self.assertLess((cached_logits - recompute_logits).abs().max().item(), 1e-6)

    def test_cached_adaptive_matches_recompute(self) -> None:
        config = AdaptiveCompressorConfig(
            hidden_size=24,
            num_levels=3,
            border_mode="uncertainty",
            byte_entropy_threshold=20.0,
            meta_uncertainty_threshold=1.0,
        )
        model = build_model("adaptive", config)
        model.eval()

        tokens = list(b"Hello")
        generator = build_cached_generator(model, torch.device("cpu"))
        cached_logits = None
        for token in tokens:
            cached_logits = generator.step(token)

        prefix = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        recompute_logits = next_byte_logits(model, prefix)[0]

        self.assertIsNotNone(cached_logits)
        self.assertLess((cached_logits - recompute_logits).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
