"""prepare_data writes train/val/vocab into a temp dir (no network)."""

import tempfile
import unittest
from pathlib import Path

import torch

from data import prepare_data


CORPUS = "abcdefghijklmnopqrstuvwxyz\n" * 20


class TestPrepareData(unittest.TestCase):
    def test_prepare_data_tempdir_char(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "corpus.txt"
            src.write_text(CORPUS, encoding="utf-8")
            out = root / "data"
            result = prepare_data(
                data_dir=str(out),
                input_file=str(src),
                tokenizer_type="char",
                train_split=0.9,
            )
            self.assertTrue((out / "train.pt").exists())
            self.assertTrue((out / "val.pt").exists())
            self.assertTrue((out / "vocab.pt").exists())

            n_tokens = len(result["train_data"]) + len(result["val_data"])
            self.assertEqual(n_tokens, len(CORPUS))
            self.assertGreater(len(result["val_data"]), 0)
            self.assertEqual(result["vocab_size"], result["tokenizer"].vocab_size)
            self.assertEqual(result["tokenizer"].type, "char")

            sample = CORPUS[:40]
            self.assertEqual(result["decode"](result["encode"](sample)), sample)

            train = torch.load(out / "train.pt", weights_only=False)
            val = torch.load(out / "val.pt", weights_only=False)
            vocab = torch.load(out / "vocab.pt", weights_only=False)
            self.assertEqual(train.shape, result["train_data"].shape)
            self.assertEqual(val.shape, result["val_data"].shape)
            self.assertEqual(vocab["tokenizer_type"], "char")
            self.assertEqual(vocab["vocab_size"], result["vocab_size"])

            # Should not download Shakespeare or write into the repo
            self.assertFalse((out / "input.txt").exists())

    def test_missing_input_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                prepare_data(
                    data_dir=td,
                    input_file=str(Path(td) / "nope.txt"),
                    tokenizer_type="char",
                )


if __name__ == "__main__":
    unittest.main()
