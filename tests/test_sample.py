"""sample.py checkpoint load + generate shape (CPU, tiny model)."""

import tempfile
import unittest
from pathlib import Path

import torch

from data import prepare_data
from model import GPT, GPTConfig, filter_logits
from sample import generate_text, load_checkpoint
from train import make_checkpoint


CORPUS = "hello world, this is a tiny test corpus.\n" * 15


class TestSample(unittest.TestCase):
    def _tiny_run(self, td: Path):
        src = td / "corpus.txt"
        src.write_text(CORPUS, encoding="utf-8")
        data_dir = td / "data"
        prepared = prepare_data(
            data_dir=str(data_dir),
            input_file=str(src),
            tokenizer_type="char",
            train_split=0.9,
        )
        config = GPTConfig(
            block_size=12,
            vocab_size=prepared["vocab_size"],
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            use_sdpa=True,
        )
        torch.manual_seed(0)
        model = GPT(config)
        ckpt_path = td / "ckpt.pt"
        ckpt = make_checkpoint(
            model,
            torch.optim.AdamW(model.parameters(), lr=1e-3),
            config,
            iter_num=0,
            best_val_loss=float("inf"),
            scaler=None,
            seed=0,
            tokenizer_type=prepared["tokenizer"].type,
            argv=["train.py", "--device", "cpu"],
        )
        torch.save(ckpt, ckpt_path)
        return ckpt_path, data_dir, config, model

    def test_load_and_generate_shape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt_path, _data_dir, config, _model = self._tiny_run(root)
            loaded, loaded_config = load_checkpoint(ckpt_path, torch.device("cpu"))
            self.assertEqual(loaded_config.vocab_size, config.vocab_size)
            self.assertEqual(loaded_config.dropout, 0.0)

            prompt_len = 4
            new_tokens = 6
            idx = torch.randint(0, config.vocab_size, (1, prompt_len))
            with torch.no_grad():
                out = loaded.generate(idx, max_new_tokens=new_tokens)
            self.assertEqual(out.shape, (1, prompt_len + new_tokens))
            self.assertEqual(out.dtype, torch.long)

    def test_generate_text_returns_string(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt_path, data_dir, _config, _model = self._tiny_run(root)
            text = generate_text(
                checkpoint_path=str(ckpt_path),
                prompt="h",
                num_tokens=8,
                temperature=1.0,
                data_dir=str(data_dir),
                device="cpu",
            )
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)
            self.assertTrue(text.startswith("h"))


class TestTopKTopP(unittest.TestCase):
    def test_top_k_masks_below_kth(self):
        logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
        original = logits.clone()
        out = filter_logits(logits, top_k=2)
        self.assertTrue(torch.equal(logits, original))
        self.assertTrue(torch.isneginf(out[0, 0]))
        self.assertEqual(out[0, 1].item(), 3.0)
        self.assertEqual(out[0, 2].item(), 2.0)
        self.assertTrue(torch.isneginf(out[0, 3]))

    def test_top_p_keeps_at_least_one(self):
        logits = torch.tensor([[10.0, 0.0, -10.0, -20.0]])
        out = filter_logits(logits, top_p=0.0)
        self.assertEqual(int((~torch.isneginf(out)).sum()), 1)
        self.assertEqual(out[0, 0].item(), 10.0)

    def test_top_k_then_top_p_sequential(self):
        logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        out = filter_logits(logits, top_k=3, top_p=0.5)
        # top-k drops the two smallest first
        self.assertTrue(torch.isneginf(out[0, 3]))
        self.assertTrue(torch.isneginf(out[0, 4]))
        # top-p then restricts the remaining nucleus
        self.assertFalse(torch.isneginf(out[0, 0]))
        self.assertGreaterEqual(int(torch.isneginf(out).sum()), 2)

    def test_generate_top_k_one_is_deterministic(self):
        config = GPTConfig(
            block_size=12,
            vocab_size=20,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        torch.manual_seed(0)
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (1, 4))
        torch.manual_seed(1)
        a = model.generate(idx.clone(), max_new_tokens=6, top_k=1, temperature=1.0)
        torch.manual_seed(99)
        b = model.generate(idx.clone(), max_new_tokens=6, top_k=1, temperature=1.0)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.shape, (1, 10))
        self.assertTrue((a >= 0).all() and (a < config.vocab_size).all())

    def test_generate_top_k_and_top_p_together(self):
        config = GPTConfig(
            block_size=12,
            vocab_size=20,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        torch.manual_seed(0)
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (1, 3))
        out = model.generate(idx, max_new_tokens=5, top_k=5, top_p=0.9, temperature=0.8)
        self.assertEqual(out.shape, (1, 8))
        self.assertTrue((out >= 0).all() and (out < config.vocab_size).all())


if __name__ == "__main__":
    unittest.main()
