"""Kiểm thử nhanh các hàm đánh giá cốt lõi (không cần tải mô hình)."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import classification_metrics

class MetricsTests(unittest.TestCase):
  def test_perfect_prediction(self):
    result = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    self.assertAlmostEqual(result["accuracy"], 1.0)
    self.assertAlmostEqual(result["macro_f1"], 1.0)
    self.assertAlmostEqual(result["qwk"], 1.0)

  def test_handles_missing_predicted_class(self):
    result = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0])
    self.assertTrue(0.0 <= result["macro_f1"] <= 1.0)
