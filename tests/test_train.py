"""Training helpers: get_batch, get_lr, checkpoint roundtrip (CPU, tiny models)."""

import math
import tempfile
import unittest
from pathlib import Path

import torch

from model import GPT, GPTConfig, strip_orig_mod_prefix
from sample import load_checkpoint
from train import collect_run_metadata, configure_optimizer, get_batch, get_lr, make_checkpoint


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
    def test_default_dropout_is_point_one(self):
        self.assertEqual(GPTConfig().dropout, 0.1)
        cfg = GPTConfig(block_size=16, vocab_size=32, n_layer=1, n_head=2, n_embd=16)
        self.assertEqual(cfg.dropout, 0.1)

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


def _tiny_corpus_dirs(td: Path):
    from data import prepare_data

    corpus = "abcdefghijklmnopqrstuvwxyz \n" * 40
    src = td / "corpus.txt"
    src.write_text(corpus, encoding="utf-8")
    data_dir = td / "data"
    out_dir = td / "out"
    prepare_data(
        data_dir=str(data_dir),
        input_file=str(src),
        tokenizer_type="char",
        train_split=0.9,
    )
    return data_dir, out_dir


def _tiny_train_kwargs(data_dir, out_dir, **overrides):
    kwargs = dict(
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
        grad_clip=1.0,
    )
    kwargs.update(overrides)
    return kwargs


class TestConfigureOptimizer(unittest.TestCase):
    def test_decay_and_nodecay_groups(self):
        config = GPTConfig(
            block_size=16,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        model = GPT(config)
        opt = configure_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
        decay = next(g for g in opt.param_groups if g["weight_decay"] > 0)
        nodecay = next(g for g in opt.param_groups if g["weight_decay"] == 0.0)
        self.assertEqual(decay["weight_decay"], 0.1)

        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), sum(1 for p in model.parameters() if p.requires_grad))

        decay_ids = {id(p) for p in decay["params"]}
        nodecay_ids = {id(p) for p in nodecay["params"]}
        for name, param in model.named_parameters(remove_duplicate=False):
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "embedding" in name or "ln_" in name or "norm" in name:
                self.assertIn(id(param), nodecay_ids, name)
            elif name.startswith("lm_head."):
                # Tied to token_embedding; classified once as no-decay
                self.assertIn(id(param), nodecay_ids, name)
            else:
                self.assertIn(id(param), decay_ids, name)


class TestResume(unittest.TestCase):
    def test_resume_continues_from_checkpoint(self):
        from train import train

        with tempfile.TemporaryDirectory() as td:
            data_dir, out_dir = _tiny_corpus_dirs(Path(td))
            train(**_tiny_train_kwargs(data_dir, out_dir, max_iters=2))
            ckpt_path = out_dir / "ckpt.pt"
            self.assertTrue(ckpt_path.exists())
            first = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            first_iter = first["iter_num"]
            first_weights = {k: v.clone() for k, v in first["model"].items()}
            # Tiny runs may not beat the saved val loss; force the next eval to write.
            first["best_val_loss"] = float("inf")
            torch.save(first, ckpt_path)

            train(**_tiny_train_kwargs(data_dir, out_dir, max_iters=4, resume=True))
            second = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.assertGreater(second["iter_num"], first_iter)
            self.assertFalse(
                all(torch.equal(first_weights[k], second["model"][k]) for k in first_weights)
            )


class TestOrigModStrip(unittest.TestCase):
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

    def test_strip_orig_mod_prefix_roundtrip(self):
        torch.manual_seed(0)
        config = self._tiny_config()
        model = GPT(config)
        state = model.state_dict()
        prefixed = {f"_orig_mod.{k}": v.clone() for k, v in state.items()}
        stripped = strip_orig_mod_prefix(prefixed)
        self.assertIsNot(stripped, prefixed)
        self.assertFalse(any(k.startswith("_orig_mod.") for k in stripped))
        self.assertEqual(set(stripped), set(state))

        other = GPT(config)
        other.load_state_dict(stripped)
        idx = torch.randint(0, config.vocab_size, (2, 8))
        model.eval()
        other.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(idx), other(idx), atol=1e-6, rtol=1e-5))

        unchanged = strip_orig_mod_prefix(state)
        self.assertIs(unchanged, state)

    def test_load_checkpoint_strips_orig_mod(self):
        torch.manual_seed(1)
        config = self._tiny_config()
        model = GPT(config)
        prefixed = {f"_orig_mod.{k}": v.clone() for k, v in model.state_dict().items()}
        ckpt = make_checkpoint(
            model,
            torch.optim.AdamW(model.parameters(), lr=1e-3),
            config,
            iter_num=0,
            best_val_loss=1.0,
            scaler=None,
            seed=1,
            tokenizer_type="char",
        )
        ckpt["model"] = prefixed
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            torch.save(ckpt, path)
            loaded, _ = load_checkpoint(path, torch.device("cpu"))
            idx = torch.randint(0, config.vocab_size, (1, 8))
            model.eval()
            loaded.eval()
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(idx), loaded(idx), atol=1e-6, rtol=1e-5))

    def test_resume_loads_compiled_checkpoint_prefix(self):
        from train import train

        with tempfile.TemporaryDirectory() as td:
            data_dir, out_dir = _tiny_corpus_dirs(Path(td))
            train(**_tiny_train_kwargs(data_dir, out_dir, max_iters=2))
            ckpt_path = out_dir / "ckpt.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt["model"] = {f"_orig_mod.{k}": v for k, v in ckpt["model"].items()}
            ckpt["best_val_loss"] = float("inf")
            torch.save(ckpt, ckpt_path)
            train(**_tiny_train_kwargs(data_dir, out_dir, max_iters=3, resume=True))
            resumed = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.assertFalse(any(k.startswith("_orig_mod.") for k in resumed["model"]))
            self.assertGreaterEqual(resumed["iter_num"], ckpt["iter_num"])


if __name__ == "__main__":
    unittest.main()
