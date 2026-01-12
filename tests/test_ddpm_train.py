import tempfile
import unittest

import torch
from torch.utils.data import Dataset, Subset

from ddpm.ddpm import DDPM


class _TensorImageDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx]


class TestDDPMTrain(unittest.TestCase):
    def test_train_runs_two_epochs(self):
        torch.manual_seed(0)
        data = torch.randn(8, 1, 4, 4)
        dataset = _TensorImageDataset(data)
        train_dataset = Subset(dataset, indices=torch.arange(0, 6))
        val_dataset = Subset(dataset, indices=torch.arange(6, 8))

        beta = torch.linspace(start=1e-4, end=0.02, steps=10)
        model = DDPM(
            beta=beta,
            channel_mult=(1,),
            image_dim=(1, 4, 4),
            base_channels=32,
            dropout=0.0,
            resample_with_conv=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = f"{tmpdir}/ddpm_test.pth"
            model.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=2,
                lr=1e-3,
                n_epochs=2,
                savepath=savepath,
                simul_batch_size=2,
                grad_clip=0.0,
                epochs_per_checkpoint=1,
                autocast_dtype="bfloat16",
            )


if __name__ == "__main__":
    unittest.main()
