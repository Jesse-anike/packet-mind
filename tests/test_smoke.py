import unittest
from pathlib import Path

from main import analyze, parse_dataset

ROOT = Path(__file__).resolve().parents[1]


class NetworkTrafficClassifierTests(unittest.TestCase):
    def test_sample_has_numeric_features_and_two_classes(self):
        sample = (ROOT / "sample.csv").read_text(encoding="utf-8")
        parsed = parse_dataset(sample)
        self.assertNotIn("error", parsed)
        self.assertEqual(len(parsed["features"]), 30)
        self.assertEqual(set(parsed["labels"]), {"BENIGN", "ATTACK"})

    def test_models_train_and_return_metrics(self):
        sample = (ROOT / "sample.csv").read_text(encoding="utf-8")
        report = analyze(sample)
        self.assertNotIn("error", report)
        self.assertIn("decision_tree", report["models"])
        self.assertIn("random_forest", report["models"])
        self.assertIn("accuracy", report["models"]["decision_tree"])

    def test_missing_label_column_is_reported(self):
        report = analyze("feature_a,category\n1,x\n", label_column="Label")
        self.assertIn("error", report)


if __name__ == "__main__":
    unittest.main()
