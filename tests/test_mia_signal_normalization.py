import unittest

import torch

from mia import run_audit
from mia.utils import standardize_signals

class TestSignalNormalization(unittest.TestCase):
    def test_pretransformation_runs_before_per_model_standardization(self):
        '''Check log and identity transforms with independent model statistics.'''
        signals = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                [[11.0, 12.0], [13.0, 14.0]]])
        for pretransformation, expected in (("log", torch.log(signals)), ("none", signals)):
            with self.subTest(pretransformation=pretransformation):
                transformed = standardize_signals(signals, pretransformation)
                expected_mean = expected.mean(dim=(1, 2), keepdim=True)
                expected_std = expected.std(dim=(1, 2), unbiased=False, keepdim=True)
                standardized = (expected - expected_mean) / expected_std
                self.assertTrue(torch.allclose(transformed, standardized))
                single_model = standardize_signals(signals[0], pretransformation)
                self.assertTrue(torch.allclose(transformed[0], single_model))

    def test_audit_and_single_model_paths_use_same_log_normalization(self):
        '''Check the audit wrapper agrees with direct single-model standardization.'''
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        shadows = torch.stack((target * 2, target * 4))
        audit_target, audit_shadows = run_audit.normalize_loss_signals(target, shadows, "log_standardized")
        self.assertTrue(torch.allclose(audit_target, standardize_signals(target, "log")))
        self.assertTrue(torch.allclose(audit_shadows[0], standardize_signals(shadows[0], "log")))
        self.assertTrue(torch.allclose(audit_shadows[1], standardize_signals(shadows[1], "log")))
        self.assertTrue(torch.equal(run_audit.normalize_loss_signals(target, shadows, "none")[0], target))

    def test_rejects_modes_and_values_that_would_silently_corrupt_results(self):
        '''Reject unknown modes, invalid shapes, and non-finite transformed values.'''
        with self.assertRaisesRegex(ValueError, "non-finite"):
            standardize_signals(torch.tensor([[0.0, 1.0]]), "log")
        with self.assertRaisesRegex(ValueError, "Unsupported pretransformation"):
            standardize_signals(torch.ones((2, 2)), "square")
        with self.assertRaisesRegex(ValueError, "shape"):
            standardize_signals(torch.ones((2, 2, 2, 2)), "none")
        with self.assertRaisesRegex(ValueError, "Unsupported loss normalization"):
            run_audit.normalize_loss_signals(torch.ones((2, 2)), torch.ones((1, 2, 2)), "model_zscore_log")


if __name__ == "__main__":
    unittest.main()
