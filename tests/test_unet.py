import unittest

import torch

from unet.unet import UNet


class TestUNetForward(unittest.TestCase):
    def test_forward_shapes_match(self):
        cases = [
            {
                "batch_size": 2,
                "image_size": 32,
                "in_channels": 3,
                "out_channels": 3,
                "base_channels": 32,
                "channel_mult": (1, 2, 2),
                "n_attention_heads": 4,
            },
            {
                "batch_size": 4,
                "image_size": 64,
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 32,
                "channel_mult": (1, 2, 4),
                "n_attention_heads": 2,
            },
        ]

        for case in cases:
            with self.subTest(case=case):
                model = UNet(
                    image_size=case["image_size"],
                    in_channels=case["in_channels"],
                    out_channels=case["out_channels"],
                    base_channels=case["base_channels"],
                    channel_mult=case["channel_mult"],
                    n_attention_heads=case["n_attention_heads"],
                )
                x = torch.randn(
                    case["batch_size"],
                    case["in_channels"],
                    case["image_size"],
                    case["image_size"],
                )
                t = torch.randint(0, 1000, (case["batch_size"],), dtype=torch.long)
                y = model(x, t)
                self.assertEqual(
                    y.shape,
                    (
                        case["batch_size"],
                        case["out_channels"],
                        case["image_size"],
                        case["image_size"],
                    ),
                )


if __name__ == "__main__":
    unittest.main()
