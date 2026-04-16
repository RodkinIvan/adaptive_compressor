import unittest

import torch

from adaptive_compressor.model import cumulative_border_mask
from adaptive_compressor.routing import build_routing, gather_parent_tokens


class RoutingTests(unittest.TestCase):
    def test_cumulative_border_mask_resets_after_threshold(self) -> None:
        scores = torch.tensor([[1.0, 4.0, 7.0, 3.0, 9.0]])
        mask = cumulative_border_mask(scores, threshold=10.0)

        expected = torch.tensor([[True, False, True, False, True]])
        self.assertTrue(torch.equal(mask, expected))

    def test_build_routing_and_gather_parent_tokens_follow_borders(self) -> None:
        child_states = torch.arange(1, 1 + 5 * 3, dtype=torch.float32).view(1, 5, 3)
        border_mask = torch.tensor([[True, False, True, False, True]])

        routing = build_routing(border_mask)
        gathered = gather_parent_tokens(child_states, routing)

        self.assertEqual(routing.parent_lengths.tolist(), [3])
        self.assertEqual(routing.parent_positions[0, :3].tolist(), [0, 2, 4])
        self.assertTrue(torch.equal(gathered[0, 0], child_states[0, 0]))
        self.assertTrue(torch.equal(gathered[0, 1], child_states[0, 2]))
        self.assertTrue(torch.equal(gathered[0, 2], child_states[0, 4]))


if __name__ == "__main__":
    unittest.main()
