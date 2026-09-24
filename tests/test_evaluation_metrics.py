import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from mia.evaluation import evaluate_MIA


class TestEvaluationMetrics(unittest.TestCase):
    def test_evaluate_mia_includes_tpr_at_fixed_fpr_metrics(self):
        score = np.array([0.95, 0.9, 0.8, 0.2, 0.1, 0.05], dtype=float)
        ground_truth = np.array([1, 1, 1, 0, 0, 0], dtype=int)

        metrics = evaluate_MIA(score, ground_truth)

        self.assertIn("AUC", metrics)
        self.assertIn("pAUC@1%FPR", metrics)
        self.assertIn("TPR@1%FPR", metrics)
        self.assertIn("TPR@0.1%FPR", metrics)
        self.assertIn("TPR@0.01%FPR", metrics)
        self.assertIn("score", metrics)
        self.assertIn("ground_truth", metrics)
        np.testing.assert_array_equal(metrics["score"], score)
        np.testing.assert_array_equal(metrics["ground_truth"], ground_truth)
        self.assertAlmostEqual(
            metrics["pAUC@1%FPR"],
            roc_auc_score(y_true=ground_truth, y_score=score, max_fpr=1e-2),
        )
        self.assertGreaterEqual(metrics["TPR@1%FPR"], 0.0)
        self.assertLessEqual(metrics["TPR@1%FPR"], 1.0)
        self.assertGreaterEqual(metrics["TPR@0.1%FPR"], 0.0)
        self.assertLessEqual(metrics["TPR@0.1%FPR"], 1.0)
        self.assertGreaterEqual(metrics["TPR@0.01%FPR"], 0.0)
        self.assertLessEqual(metrics["TPR@0.01%FPR"], 1.0)


if __name__ == "__main__":
    unittest.main()
