"""SDPA vs manual causal attention: shapes and eval-mode closeness."""

import unittest

import torch

from model import CausalSelfAttention, GPT, GPTConfig


class TestAttention(unittest.TestCase):
    def test_sdpa_and_manual_shapes_match(self):
        cfg_sdpa = GPTConfig(block_size=32, n_head=4, n_embd=64, use_flash_attn=True)
        cfg_man = GPTConfig(block_size=32, n_head=4, n_embd=64, use_flash_attn=False)
        sdpa = CausalSelfAttention(cfg_sdpa).eval()
        manual = CausalSelfAttention(cfg_man).eval()
        # Shared projection weights so the comparison is fair
        manual.load_state_dict(sdpa.state_dict(), strict=False)
        x = torch.randn(2, 16, 64)
        y_s = sdpa(x)
        y_m = manual(x)
        self.assertEqual(y_s.shape, y_m.shape)
        self.assertEqual(y_s.shape, (2, 16, 64))
        self.assertTrue(torch.isfinite(y_s).all())
        self.assertTrue(torch.isfinite(y_m).all())
        self.assertTrue(torch.allclose(y_s, y_m, atol=2e-4, rtol=2e-3))

    def test_gpt_logits_shape(self):
        cfg = GPTConfig(
            block_size=32,
            vocab_size=50,
            n_layer=2,
            n_head=4,
            n_embd=64,
            use_flash_attn=True,
        )
        model = GPT(cfg)
        idx = torch.randint(0, 50, (2, 8))
        logits = model(idx)
        self.assertEqual(logits.shape, (2, 8, 50))
        n = model.get_num_params()
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
