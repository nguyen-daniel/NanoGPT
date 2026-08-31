"""BPE / char tokenizer roundtrip tests (no GPU)."""

import tempfile
import unittest
from pathlib import Path

from tokenizer import BPETokenizer, CharTokenizer, load_tokenizer


SAMPLE = "To be, or not to be, that is the question:\nWhether 'tis nobler"


class TestCharTokenizer(unittest.TestCase):
    def test_roundtrip(self):
        tok = CharTokenizer.train(SAMPLE)
        self.assertEqual(tok.decode(tok.encode(SAMPLE)), SAMPLE)

    def test_save_load(self):
        tok = CharTokenizer.train(SAMPLE)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vocab.pt"
            tok.save(path)
            loaded = load_tokenizer(path)
            self.assertEqual(loaded.type, "char")
            self.assertEqual(loaded.decode(loaded.encode(SAMPLE)), SAMPLE)


class TestBPETokenizer(unittest.TestCase):
    def test_roundtrip(self):
        tok = BPETokenizer.train(SAMPLE, vocab_size=280)
        self.assertEqual(tok.decode(tok.encode(SAMPLE)), SAMPLE)

    def test_shorter_than_utf8_bytes_after_merges(self):
        tok = BPETokenizer.train(SAMPLE * 20, vocab_size=300)
        encoded = tok.encode(SAMPLE * 20)
        self.assertLess(len(encoded), len((SAMPLE * 20).encode("utf-8")))

    def test_save_load(self):
        tok = BPETokenizer.train(SAMPLE, vocab_size=280)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vocab.pt"
            tok.save(path)
            loaded = load_tokenizer(path)
            self.assertEqual(loaded.type, "bpe")
            self.assertEqual(loaded.decode(loaded.encode(SAMPLE)), SAMPLE)


if __name__ == "__main__":
    unittest.main()
