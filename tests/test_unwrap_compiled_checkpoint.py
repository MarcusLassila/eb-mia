import tempfile
import unittest
from pathlib import Path

import torch

from utils import has_torch_compile_wrapped_state_dict, load_checkpoint, unwrap_checkpoint_state_dicts, unwrap_torch_compile_state_dict
from utils.unwrap_compiled_checkpoint import unwrap_compiled_checkpoint


class TestUnwrapCompiledCheckpoint(unittest.TestCase):
    def test_unwrap_torch_compile_state_dict_removes_prefix_only(self):
        state_dict = {
            "_orig_mod.weight": torch.tensor([1.0]),
            "block._orig_mod.bias": torch.tensor([2.0]),
        }
        unwrapped_state_dict = unwrap_torch_compile_state_dict(state_dict)
        self.assertIn("weight", unwrapped_state_dict)
        self.assertIn("block._orig_mod.bias", unwrapped_state_dict)
        self.assertNotIn("_orig_mod.weight", unwrapped_state_dict)
        self.assertTrue(torch.equal(unwrapped_state_dict["weight"], torch.tensor([1.0])))

    def test_unwrap_torch_compile_state_dict_collision_raises(self):
        state_dict = {
            "weight": torch.tensor([1.0]),
            "_orig_mod.weight": torch.tensor([2.0]),
        }
        with self.assertRaisesRegex(ValueError, "collision"):
            unwrap_torch_compile_state_dict(state_dict)

    def test_unwrap_checkpoint_state_dicts(self):
        checkpoint = {
            "network_state_dict": {"_orig_mod.weight": torch.tensor([1.0])},
            "raw_network_state_dict": {"_orig_mod.bias": torch.tensor([2.0])},
            "ema_network_state_dict": {"_orig_mod.buffer": torch.tensor([3.0])},
            "optimizer_state_dict": {"state": {}},
        }
        unwrapped_checkpoint = unwrap_checkpoint_state_dicts(checkpoint)
        self.assertIn("weight", unwrapped_checkpoint["network_state_dict"])
        self.assertIn("bias", unwrapped_checkpoint["raw_network_state_dict"])
        self.assertIn("buffer", unwrapped_checkpoint["ema_network_state_dict"])
        self.assertEqual(unwrapped_checkpoint["optimizer_state_dict"], checkpoint["optimizer_state_dict"])
        self.assertFalse(has_torch_compile_wrapped_state_dict(unwrapped_checkpoint["network_state_dict"]))

    def test_unwrap_compiled_checkpoint_overwrites_input_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            checkpoint = {
                "network_state_dict": {"_orig_mod.weight": torch.tensor([1.0])},
                "raw_network_state_dict": {"_orig_mod.bias": torch.tensor([2.0])},
                "ema_network_state_dict": {"_orig_mod.buffer": torch.tensor([3.0])},
                "train_indices": torch.tensor([0, 1]),
            }
            torch.save(checkpoint, checkpoint_path)
            output_path = unwrap_compiled_checkpoint(checkpoint_path)
            self.assertEqual(output_path, checkpoint_path)
            loaded = load_checkpoint(str(checkpoint_path), torch.device("cpu"))
            self.assertIn("weight", loaded["network_state_dict"])
            self.assertIn("bias", loaded["raw_network_state_dict"])
            self.assertIn("buffer", loaded["ema_network_state_dict"])
            self.assertFalse(has_torch_compile_wrapped_state_dict(loaded["network_state_dict"]))


if __name__ == "__main__":
    unittest.main()
