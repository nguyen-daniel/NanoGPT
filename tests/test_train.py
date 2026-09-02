"""Training helpers: get_batch, get_lr, checkpoint roundtrip (CPU, tiny models)."""

import math
import tempfile
import unittest
from pathlib import Path

import torch

from model import GPT, GPTConfig
from sample import load_checkpoint
from train import collect_run_metadata, get_batch, get_lr, make_checkpoint


class TestGetBatch(unittest.TestCase):
    def test_shapes_and_shift(self):
        data = torch.arange(100, dtype=torch.long)
        block_size, batch_size = 8, 4
        x, y = get_batch(data, block_size, batch_size, torch.device("cpu"))
        self.assertEqual(x.shape, (batch_size, block_size))
        self.assertEqual(y.shape, (batch_size, block_size))
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(x.device.type, "cpu")
        # data is arange, so each target token is the input token + 1
        self.assertTrue(torch.equal(y, x + 1))
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))

    def test_values_come_from_data(self):
        data = torch.arange(50, dtype=torch.long)
        x, y = get_batch(data, block_size=5, batch_size=3, device=torch.device("cpu"))
        self.assertTrue((x >= 0).all() and (x < 50).all())
        self.assertTrue((y >= 0).all() and (y < 50).all())
        for row in range(x.size(0)):
            start = int(x[row, 0])
            expected_x = data[start : start + 5]
            expected_y = data[start + 1 : start + 6]
            self.assertTrue(torch.equal(x[row], expected_x))
            self.assertTrue(torch.equal(y[row], expected_y))


class TestGetLr(unittest.TestCase):
    def test_warmup_linear(self):
        lr, warmup, max_iters, min_lr = 3e-4, 100, 5000, 0.1
        self.assertAlmostEqual(get_lr(0, lr, warmup, max_iters, min_lr), lr * 1 / warmup)
        self.assertAlmostEqual(get_lr(49, lr, warmup, max_iters, min_lr), lr * 50 / warmup)
        self.assertAlmostEqual(get_lr(warmup - 1, lr, warmup, max_iters, min_lr), lr)

    def test_warmup_increases(self):
        lrs = [get_lr(i, 1.0, 10, 100, 0.1) for i in range(10)]
        self.assertTrue(all(lrs[i] < lrs[i + 1] for i in range(9)))

    def test_cosine_at_boundaries(self):
        lr, warmup, max_iters, min_lr = 1.0, 10, 100, 0.1
        # First cosine step equals peak LR
        self.assertAlmostEqual(get_lr(warmup, lr, warmup, max_iters, min_lr), lr)
        # Last scheduled step equals min_lr * learning_rate
        self.assertAlmostEqual(get_lr(max_iters, lr, warmup, max_iters, min_lr), min_lr * lr)
        self.assertAlmostEqual(get_lr(max_iters + 1, lr, warmup, max_iters, min_lr), min_lr * lr)

    def test_cosine_matches_formula(self):
        it, lr, warmup, max_iters, min_lr = 555, 3e-4, 100, 5000, 0.1
        decay_ratio = (it - warmup) / (max_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        expected = min_lr * lr + coeff * (lr - min_lr * lr)
        self.assertAlmostEqual(get_lr(it, lr, warmup, max_iters, min_lr), expected)

    def test_cosine_non_increasing(self):
        lrs = [get_lr(i, 1.0, 10, 100, 0.1) for i in range(10, 101)]
        self.assertTrue(all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1)))


class TestCheckpointRoundtrip(unittest.TestCase):
    def _tiny_config(self):
        return GPTConfig(
            block_size=16,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            use_sdpa=True,
        )

    def test_make_checkpoint_metadata_and_weights(self):
        torch.manual_seed(0)
        config = self._tiny_config()
        model = GPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        argv = ["train.py", "--seed", "1337", "--dropout", "0.0"]
        ckpt = make_checkpoint(
            model,
            optimizer,
            config,
            iter_num=7,
            best_val_loss=2.5,
            scaler=None,
            seed=1337,
            tokenizer_type="char",
            argv=argv,
        )
        self.assertEqual(ckpt["iter_num"], 7)
        self.assertEqual(ckpt["best_val_loss"], 2.5)
        self.assertEqual(ckpt["tokenizer_type"], "char")
        self.assertEqual(ckpt["seed"], 1337)
        self.assertEqual(ckpt["torch_version"], torch.__version__)
        self.assertEqual(ckpt["argv"], argv)
        self.assertTrue(ckpt["git_sha"] is None or isinstance(ckpt["git_sha"], str))
        self.assertIsNone(ckpt["scaler"])

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            torch.save(ckpt, path)
            loaded_model, loaded_config = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(loaded_config.n_layer, config.n_layer)
            self.assertEqual(loaded_config.dropout, 0.0)

            model.eval()
            loaded_model.eval()
            idx = torch.randint(0, config.vocab_size, (2, 8))
            with torch.no_grad():
                original = model(idx)
                restored = loaded_model(idx)
            self.assertEqual(restored.shape, original.shape)
            self.assertTrue(torch.allclose(original, restored, atol=1e-6, rtol=1e-5))

            disk = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(disk["tokenizer_type"], "char")
            self.assertEqual(disk["seed"], 1337)
            self.assertEqual(disk["argv"], argv)

    def test_collect_run_metadata_keys(self):
        meta = collect_run_metadata(seed=42, tokenizer_type="bpe", argv=["python", "train.py"])
        self.assertEqual(set(meta), {"tokenizer_type", "seed", "torch_version", "git_sha", "argv"})
        self.assertEqual(meta["tokenizer_type"], "bpe")
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["torch_version"], torch.__version__)
        self.assertEqual(meta["argv"], ["python", "train.py"])


class TestDropoutConfig(unittest.TestCase):
    def test_dropout_wired_from_config(self):
        cfg = GPTConfig(
            block_size=16,
            vocab_size=20,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.3,
        )
        model = GPT(cfg)
        self.assertEqual(model.config.dropout, 0.3)
        self.assertEqual(model.dropout.p, 0.3)
        block = model.blocks[0]
        self.assertEqual(block.attn.dropout, 0.3)
        self.assertEqual(block.attn.attn_dropout.p, 0.3)
        self.assertEqual(block.mlp.dropout.p, 0.3)


class TestTrainSavesMetadata(unittest.TestCase):
    def test_tiny_train_writes_ckpt_metadata(self):
        from data import prepare_data
        from train import train

        corpus = "abcdefghijklmnopqrstuvwxyz \n" * 40
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "corpus.txt"
            src.write_text(corpus, encoding="utf-8")
            data_dir = root / "data"
            out_dir = root / "out"
            prepare_data(
                data_dir=str(data_dir),
                input_file=str(src),
                tokenizer_type="char",
                train_split=0.9,
            )
            train(
                data_dir=str(data_dir),
                out_dir=str(out_dir),
                block_size=8,
                batch_size=2,
                n_layer=1,
                n_head=2,
                n_embd=16,
                max_iters=2,
                warmup_iters=1,
                eval_interval=1,
                eval_iters=1,
                device="cpu",
                use_compile=False,
                use_amp=False,
                seed=123,
                dropout=0.0,
            )
            ckpt_path = out_dir / "ckpt.pt"
            self.assertTrue(ckpt_path.exists())
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["tokenizer_type"], "char")
            self.assertEqual(ckpt["seed"], 123)
            self.assertEqual(ckpt["torch_version"], torch.__version__)
            self.assertEqual(ckpt["config"].dropout, 0.0)
            self.assertIn("argv", ckpt)
            self.assertTrue(ckpt["git_sha"] is None or isinstance(ckpt["git_sha"], str))
            loaded, _cfg = load_checkpoint(ckpt_path, torch.device("cpu"))
            idx = torch.randint(0, ckpt["config"].vocab_size, (1, 4))
            with torch.no_grad():
                out = loaded.generate(idx, max_new_tokens=3)
            self.assertEqual(out.shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
