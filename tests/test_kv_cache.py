"""Cached vs full-context logits on CPU (no GPU)."""

import unittest

import torch

from model import GPT, GPTConfig


class TestKVCache(unittest.TestCase):
    def _tiny(self, use_sdpa):
        cfg = GPTConfig(
            block_size=32,
            vocab_size=50,
            n_layer=2,
            n_head=4,
            n_embd=64,
            use_sdpa=use_sdpa,
        )
        return GPT(cfg).eval()

    def _incremental_logits(self, model, idx, prefix_len=1):
        """First step: prefix; later steps: one token, using the KV cache."""
        logits_0, cache = model(idx[:, :prefix_len], use_cache=True)
        parts = [logits_0]
        for t in range(prefix_len, idx.size(1)):
            logits_t, cache = model(idx[:, t : t + 1], past_kvs=cache, use_cache=True)
            parts.append(logits_t)
        return torch.cat(parts, dim=1)

    def _assert_cached_matches_full(self, use_sdpa, prefix_len):
        torch.manual_seed(0)
        model = self._tiny(use_sdpa)
        idx = torch.randint(0, 50, (2, 12))
        with torch.no_grad():
            full = model(idx)
            cached = self._incremental_logits(model, idx, prefix_len=prefix_len)
        self.assertEqual(full.shape, cached.shape)
        self.assertTrue(
            torch.allclose(full, cached, atol=1e-4, rtol=1e-4),
            f"max abs diff={ (full - cached).abs().max().item() }",
        )

    def test_cached_vs_full_sdpa_token_by_token(self):
        self._assert_cached_matches_full(use_sdpa=True, prefix_len=1)

    def test_cached_vs_full_sdpa_prefix_then_decode(self):
        self._assert_cached_matches_full(use_sdpa=True, prefix_len=5)

    def test_cached_vs_full_manual_token_by_token(self):
        self._assert_cached_matches_full(use_sdpa=False, prefix_len=1)

    def test_cached_vs_full_manual_prefix_then_decode(self):
        self._assert_cached_matches_full(use_sdpa=False, prefix_len=5)

    def test_generate_shape_uses_cache(self):
        torch.manual_seed(1)
        model = self._tiny(use_sdpa=True)
        idx = torch.randint(0, 50, (1, 4))
        out = model.generate(idx, max_new_tokens=6, temperature=1.0)
        self.assertEqual(out.shape, (1, 10))


if __name__ == "__main__":
    unittest.main()
