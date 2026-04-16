import math
import unittest

import torch

from adaptive_compressor.model import AdaptiveCompressorConfig, build_model


class ModelSanityTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_baseline_matches_adaptive_parameter_count(self) -> None:
        config = AdaptiveCompressorConfig(hidden_size=32, num_levels=3)
        adaptive = build_model("adaptive", config)
        baseline = build_model("baseline", config)

        adaptive_parameters = sum(p.numel() for p in adaptive.parameters())
        baseline_parameters = sum(p.numel() for p in baseline.parameters())

        self.assertEqual(adaptive_parameters, baseline_parameters)

    def test_adaptive_forward_returns_finite_losses_and_expected_shapes(self) -> None:
        config = AdaptiveCompressorConfig(
            hidden_size=32,
            num_levels=3,
            border_mode="uncertainty",
            byte_entropy_threshold=20.0,
            meta_uncertainty_threshold=1.0,
        )
        model = build_model("adaptive", config)
        model.eval()

        byte_ids = torch.randint(0, 256, (2, 16))
        outputs = model(byte_ids)

        self.assertIn("loss", outputs)
        self.assertIn("meta_loss", outputs)
        self.assertIn("uncertainty_loss", outputs)
        self.assertIn("entropy_reg_loss", outputs)
        self.assertEqual(
            outputs["byte_decoder_logits"].shape, (2, 16, config.vocab_size)
        )
        self.assertEqual(len(outputs["border_counts"]), config.num_levels - 1)
        self.assertTrue(math.isfinite(outputs["loss"].item()))
        self.assertTrue(math.isfinite(outputs["byte_encoder_loss"].item()))
        self.assertTrue(math.isfinite(outputs["byte_decoder_loss"].item()))


if __name__ == "__main__":
    unittest.main()
