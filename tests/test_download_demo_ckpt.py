"""Demo checkpoint downloader: local copy, skip-if-exists, env URL override."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_demo_ckpt import download_checkpoint, main, resolve_url


class TestDownloadDemoCkpt(unittest.TestCase):
    def test_copy_local_path_and_skip_if_exists(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src.pt"
            src.write_bytes(b"ckpt-bytes")
            dest = root / "out" / "ckpt.pt"
            got = download_checkpoint(str(src), dest)
            self.assertEqual(got, dest)
            self.assertEqual(dest.read_bytes(), b"ckpt-bytes")

            src.write_bytes(b"newer")
            skipped = download_checkpoint(str(src), dest, force=False)
            self.assertEqual(skipped.read_bytes(), b"ckpt-bytes")

            forced = download_checkpoint(str(src), dest, force=True)
            self.assertEqual(forced.read_bytes(), b"newer")

    def test_file_uri_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src.pt"
            src.write_bytes(b"from-uri")
            dest = root / "ckpt.pt"
            uri = src.resolve().as_uri()
            download_checkpoint(uri, dest)
            self.assertEqual(dest.read_bytes(), b"from-uri")

    def test_resolve_url_env_override(self):
        with patch.dict(os.environ, {"NANOGPT_DEMO_CKPT_URL": "C:/local/ckpt.pt"}):
            self.assertEqual(resolve_url("https://example.com/ckpt.pt"), "C:/local/ckpt.pt")

    def test_main_local_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src.pt"
            src.write_bytes(b"cli")
            dest = root / "dest.pt"
            main(["--url", str(src), "--out", str(dest), "--no_deps"])
            self.assertEqual(dest.read_bytes(), b"cli")


if __name__ == "__main__":
    unittest.main()
