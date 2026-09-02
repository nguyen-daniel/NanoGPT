"""sample.py checkpoint load + generate shape (CPU, tiny model)."""

import tempfile
import unittest
from pathlib import Path

import torch

from data import prepare_data
from model import GPT, GPTConfig
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


if __name__ == "__main__":
    unittest.main()
