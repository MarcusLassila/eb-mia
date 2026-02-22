import tempfile
import unittest
from unittest import mock

from hf_scripts import download, upload


class TestHFDownloadScript(unittest.TestCase):
    def test_download_directory_downloads_remote_directory_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(download, "snapshot_download", return_value="/tmp/out") as mock_snapshot:
                output = download.download_directory(tmpdir, "checkpoints/run1", repo_id="owner/repo", repo_type="model")
        self.assertEqual(output, "/tmp/out")
        self.assertEqual(mock_snapshot.call_args.kwargs["allow_patterns"], "checkpoints/run1/**")

    def test_download_directory_filters_with_file_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(download, "snapshot_download", return_value="/tmp/out") as mock_snapshot:
                download.download_directory(
                    tmpdir,
                    "checkpoints/run1",
                    file_pattern="*-epoch10.pth",
                    repo_id="owner/repo",
                    repo_type="model",
                )
        self.assertEqual(mock_snapshot.call_args.kwargs["allow_patterns"], "checkpoints/run1/*-epoch10.pth")


class TestHFUploadScript(unittest.TestCase):
    def test_upload_directory_uploads_all_files_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(upload, "upload_folder", return_value="commit-url") as mock_upload:
                output = upload.upload_directory(tmpdir, "checkpoints/run1", repo_id="owner/repo", repo_type="model")
        self.assertEqual(output, "commit-url")
        self.assertIsNone(mock_upload.call_args.kwargs["allow_patterns"])

    def test_upload_directory_filters_with_file_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(upload, "upload_folder", return_value="commit-url") as mock_upload:
                upload.upload_directory(
                    tmpdir,
                    "checkpoints/run1",
                    file_pattern="*-epoch10.pth",
                    repo_id="owner/repo",
                    repo_type="model",
                )
        self.assertEqual(mock_upload.call_args.kwargs["allow_patterns"], "*-epoch10.pth")


if __name__ == "__main__":
    unittest.main()
