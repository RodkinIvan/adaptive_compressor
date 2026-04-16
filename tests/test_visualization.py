import unittest
from pathlib import Path


class VisualizationTests(unittest.TestCase):
    def test_visualization_file_exists_and_mentions_core_figures(self) -> None:
        visualization_path = Path("docs/model_visualization.html")
        self.assertTrue(visualization_path.exists())

        content = visualization_path.read_text(encoding="utf-8")
        self.assertIn("Figure A. Full Model", content)
        self.assertIn("Figure B. Cumulative Border Rule", content)
        self.assertIn("Figure C. Exact Streaming Inference", content)


if __name__ == "__main__":
    unittest.main()
