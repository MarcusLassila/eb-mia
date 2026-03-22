import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import evaluation.quality_metrics as qm


class FakeFID:
    last_instance = None

    def __init__(self, feature=2048, normalize=True):
        self.feature = feature
        self.normalize = normalize
        self.updates = []
        FakeFID.last_instance = self

    def to(self, device):
        self.device = device
        return self

    def set_dtype(self, dtype):
        self.dtype = dtype
        return self

    def update(self, samples, real):
        self.updates.append((samples.detach().cpu(), real))

    def compute(self):
        return torch.tensor(123.0)


class FakeIS:
    last_instance = None

    def __init__(self, normalize=True):
        self.normalize = normalize
        self.updates = []
        FakeIS.last_instance = self

    def to(self, device):
        self.device = device
        return self

    def update(self, samples):
        self.updates.append(samples.detach().cpu())

    def compute(self):
        return torch.tensor(1.5), torch.tensor(0.1)


class FakeModel:
    def __init__(self, outputs):
        self.outputs = outputs
        self.call_index = 0

    def sample(self, batch_size, **kwargs):
        output = self.outputs[self.call_index]
        self.call_index += 1
        return output


class TestQualityMetrics(unittest.TestCase):
    def test_format_metric_result_includes_checkpoint_stem_and_metric_label(self):
        result = qm.format_metric_result(
            "fid",
            "/tmp/DDPM-CelebA-ent-f0p5-p0p5-sz64-s0-epoch500.pth",
            "6.78",
        )

        self.assertEqual(
            result,
            "DDPM-CelebA-ent-f0p5-p0p5-sz64-s0-epoch500\nFID: 6.78",
        )

    def test_save_metric_result_writes_expected_file_and_content(self):
        checkpoint = "DDPM-CelebA-ent-f0p5-p0p5-sz64-s0-epoch500.pth"
        with TemporaryDirectory() as tmp_dir:
            output_path = qm.save_metric_result("is", checkpoint, "1.5 +- 0.1", output_dir=tmp_dir)

            self.assertEqual(output_path, Path(tmp_dir) / "IS-DDPM-CelebA-ent-f0p5-p0p5-sz64-s0-epoch500.txt")
            self.assertTrue(output_path.exists())
            self.assertEqual(
                output_path.read_text(),
                "DDPM-CelebA-ent-f0p5-p0p5-sz64-s0-epoch500\nIS: 1.5 +- 0.1\n",
            )

    def test_fid_score_normalizes_real_samples_and_checks_first_batch_only(self):
        dataset = [
            torch.full((1, 1, 1), -1.0),
            torch.full((1, 1, 1), 1.0),
            torch.full((1, 1, 1), 2.0),
        ]
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        gen_batch1 = torch.full((2, 1, 1, 1), 0.2)
        gen_batch2 = torch.full((1, 1, 1, 1), 1.2)
        model = FakeModel([gen_batch1, gen_batch2])

        with patch.object(qm, "FrechetInceptionDistance", FakeFID):
            score = qm.fid_score(model, dataloader, torch.device("cpu"), disable_tqdm=True)

        self.assertTrue(torch.equal(score, torch.tensor(123.0)))
        updates = FakeFID.last_instance.updates
        self.assertEqual(len(updates), 4)
        expected_real_batch1 = torch.tensor([[[[0.0]]], [[[1.0]]]])
        self.assertTrue(torch.equal(updates[0][0], expected_real_batch1))
        self.assertTrue(torch.equal(updates[1][0], gen_batch1))
        expected_real_batch2 = torch.tensor([[[[1.5]]]])
        self.assertTrue(torch.equal(updates[2][0], expected_real_batch2))
        self.assertTrue(torch.equal(updates[3][0], gen_batch2))

    def test_inception_score_checks_first_batch_only(self):
        gen_batch1 = torch.full((2, 1, 1, 1), 0.3)
        gen_batch2 = torch.full((1, 1, 1, 1), 1.3)
        model = FakeModel([gen_batch1, gen_batch2])

        with patch.object(qm, "InceptionScore", FakeIS):
            mean, std = qm.inception_score(model, 3, 2, torch.device("cpu"), disable_tqdm=True)

        self.assertTrue(torch.equal(mean, torch.tensor(1.5)))
        self.assertTrue(torch.equal(std, torch.tensor(0.1)))
        updates = FakeIS.last_instance.updates
        self.assertEqual(len(updates), 2)
        self.assertTrue(torch.equal(updates[0], gen_batch1))
        self.assertTrue(torch.equal(updates[1], gen_batch2))


if __name__ == "__main__":
    unittest.main()
