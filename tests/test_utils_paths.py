import unittest
from pathlib import Path

from utils import resolve_path


class TestUtilsPaths(unittest.TestCase):
    def test_resolve_path_uses_root_for_relative_path(self):
        root = Path("/tmp/repo")
        self.assertEqual(resolve_path("mia/configs/config_audit.yaml", root), root / "mia/configs/config_audit.yaml")

    def test_resolve_path_keeps_absolute_path(self):
        path = Path("/tmp/config.yaml")
        self.assertEqual(resolve_path(path, Path("/tmp/repo")), path)

    def test_resolve_path_keeps_relative_path_when_root_is_none(self):
        path = Path("training/configs/test.yaml")
        self.assertEqual(resolve_path(path, None), path)


if __name__ == "__main__":
    unittest.main()
