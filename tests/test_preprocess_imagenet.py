from io import BytesIO
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from data import preprocess_imagenet

class _FakeImageNetSplit:

    def __init__(self, colors):
        self.colors = colors

    def cast_column(self, column_name, feature):
        self.column_name = column_name
        self.feature = feature
        return self

    def __iter__(self):
        for index, color in enumerate(self.colors):
            image = Image.new("RGB", (32 + index, 40 + index), color=color)
            image_buffer = BytesIO()
            image.save(image_buffer, format="JPEG")
            yield {
                "image": {
                    "bytes": image_buffer.getvalue(),
                    "path": None,
                },
                "label": index,
            }

    def __len__(self):
        return len(self.colors)

class TestPreprocessImageNet(unittest.TestCase):

    def test_main_saves_ordered_rgb_images_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "processed"
            train_split = _FakeImageNetSplit([(255, 0, 0), (0, 255, 0), (255, 255, 0)])
            validation_split = _FakeImageNetSplit([(0, 0, 255)])

            def fake_load_dataset(name, split, cache_dir, token):
                self.assertEqual(name, "ILSVRC/imagenet-1k")
                self.assertEqual(cache_dir, tmpdir)
                self.assertTrue(token)
                if split == "train":
                    return train_split
                if split == "validation":
                    return validation_split
                raise AssertionError(f"Unexpected split: {split}")

            with patch.object(preprocess_imagenet, "load_dataset", side_effect=fake_load_dataset):
                manifest_path = preprocess_imagenet.main([
                    "--data-dir",
                    tmpdir,
                    "--output-dir",
                    str(output_dir),
                    "--size",
                    "16",
                    "--quality",
                    "90",
                    "--images-per-shard",
                    "2",
                    "--log-every",
                    "0",
                ])

            with manifest_path.open("r") as file:
                manifest = json.load(file)

            self.assertEqual(manifest["size"], 16)
            self.assertEqual(manifest["color"], "RGB")
            self.assertEqual(
                manifest["splits"],
                [
                    {"name": "train", "length": 3},
                    {"name": "validation", "length": 1},
                ],
            )
            expected_shards = [
                "shards/imagenet-rgb-sz16-000000.zip",
                "shards/imagenet-rgb-sz16-000001.zip",
            ]
            shard_paths = output_dir.glob("shards/*.zip")
            actual_shards = sorted(str(path.relative_to(output_dir)) for path in shard_paths)
            self.assertEqual(actual_shards, expected_shards)
            expected_members = {
                expected_shards[0]: ["000000000.jpg", "000000001.jpg"],
                expected_shards[1]: ["000000002.jpg", "000000003.jpg"],
            }
            for shard_name, member_names in expected_members.items():
                shard_path = output_dir / shard_name
                with zipfile.ZipFile(shard_path, "r") as shard_file:
                    self.assertEqual(shard_file.namelist(), member_names)
                    for member_name in member_names:
                        with shard_file.open(member_name, "r") as image_file:
                            image_bytes = image_file.read()
                        with Image.open(BytesIO(image_bytes)) as image:
                            self.assertEqual(image.mode, "RGB")
                            self.assertEqual(image.size, (16, 16))

    def test_prepare_output_dir_rejects_existing_cache_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with (output_dir / "manifest.json").open("w") as file:
                json.dump({}, file)

            with self.assertRaises(FileExistsError):
                preprocess_imagenet.prepare_output_dir(output_dir, overwrite=False)

if __name__ == "__main__":
    unittest.main()
