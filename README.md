# NanoGPT

[![CI](https://github.com/nguyen-daniel/NanoGPT/actions/workflows/main.yml/badge.svg)](https://github.com/nguyen-daniel/NanoGPT/actions/workflows/main.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Decoder-only GPT in PyTorch (Karpathy-style): train on Tiny Shakespeare or your own text, sample from a checkpoint. Character tokenizer by default; optional BPE. MIT licensed.

## What I built

- Decoder-only GPT: pre-norm blocks, causal attention, GELU MLP, weight tying (~10–15M params at default 6×6×384)
- Attention via **PyTorch SDPA** (`F.scaled_dot_product_attention`); `--no_sdpa` is the manual `QK^T` + causal-mask fallback. SDPA may dispatch a FlashAttention kernel when the backend provides one — this repo does not implement FlashAttention.
- **KV-cached sampling**: `generate()` forwards the full prompt once, then one new token per step. SDPA uses `is_causal=True` on the first step and `is_causal=False` over cached K/V + the new token. (No invented latency numbers.)
- **DirectML** device path for Windows + AMD (RX 7800 XT); AMP and `torch.compile` are skipped on DirectML (FP32). Long DML runs are checkpointed from a CPU re-eval of the weights.
- Vectorized `get_batch()` with optional device-resident tokens (no Python index loop)
- Character tokenizer by default; optional BPE (`python data.py --tokenizer bpe`)

## Run

```bash
git clone https://github.com/nguyen-daniel/NanoGPT.git
cd NanoGPT

# CPU / CI
pip install -r requirements.txt

python data.py
python train.py --max_iters 50 --eval_interval 50 --eval_iters 2 --no_amp --no_compile
python sample.py --prompt "ROMEO:"
python -m unittest discover -s tests -v
```

Windows + AMD (RX 7800 XT): Python 3.11, `pip install -r requirements-directml.txt`, then `python device.py` (expect DirectML / RX 7800 XT). Do not pre-install a newer torch.

## Proof

GPU device proof (DirectML, RX 7800 XT): [`results/gpu_proof.json`](results/gpu_proof.json) and [`results/train_run.json`](results/train_run.json). A 5000-iter **6×6×384** run did execute on `privateuseone:1` / RX 7800 XT. DirectML’s reported val loss (~1.47) does **not** match a CPU re-eval of the same checkpoint (~5.9). CPU-gated DML training stops improving around val 3.1–3.3. The Shakespeare-like sample below is therefore from a **CPU-trained 6×6×192** run (5000 iters, seed 1337, val **1.71**), not from the DirectML 6×6×384 weights.

Full sample: [`results/sample_shakespeare.txt`](results/sample_shakespeare.txt) (`ROMEO:`, temperature 0.8, top-k 40). Checkpoints are gitignored (`out/`, `*.pt`) and are not committed.

```
ROMEO:
Mow you remenour good or came about,
The revensed, and not into cannother of my reath
This this reconce are york.

Provost:
My say lord, let you this usainst his with heart apon
it to my sovereign madier.

DUKE VINCENTIO:
And he arrow'd hath
his love with him and in theme?
```

That is play-formatted, misspelled Shakespeare-like text — not the 3-iter `ROMEO:ent AR#go.` GPU smoke string.

## Results

Device proof measured on **AMD Radeon RX 7800 XT** (Windows, **DirectML** — not CUDA). Sample metrics: [`results/train_run.json`](results/train_run.json). BPE: `python benches/bench_bpe.py`.

| Topic | What is true | Artifact |
|-------|----------------|----------|
| GPU | Training tensors live on `privateuseone:1` / RX 7800 XT | `scripts/check_gpu.py`, train log `Proof: batch x.device=...`, `results/gpu_proof.json` |
| Sample | CPU 6×6×192, 5000 iters, seed 1337, val 1.71; speaker names + verse-ish lines | `results/sample_shakespeare.txt`, `results/train_run.json` |
| DirectML eval | Long DML runs report falling val; CPU re-eval of cloned weights does not follow | `results/train_run.json` (`cpu_reeval_of_that_ckpt_val_loss`) |
| Attention | Causal self-attention via **`F.scaled_dot_product_attention`** (PyTorch SDPA; may dispatch FlashAttention **when the backend provides it**). Manual attention is `--no_sdpa`. | `model.py`, `tests/test_attention.py` |
| KV cache | Incremental decode logits match a full-context forward (`allclose` on CPU) | `tests/test_kv_cache.py` |
| Size | Default 6×6×384 is ~10–15M parameters (vocab-dependent); printed at train start | train log |
| BPE | 2.50× sequence compression vs char on Tiny Shakespeare (vocab 1000) | `python benches/bench_bpe.py` → `results/bpe_compression.json` |
| Batching | Vectorized gather; optional device-resident tokens | `get_batch()` vs `get_batch_loop()` in `train.py` |

No 2–4× memory or 40% data-loading claims, and no invented KV-cache latency percentages. If you run `make bench`, treat the JSON as the only numbers.

## Architecture

Token + position embeddings → decoder blocks (pre-norm, causal attention, GELU MLP) → LM head with weight tying. SDPA when `use_sdpa` is on; otherwise explicit `QK^T` + causal mask. Sampling keeps a per-layer K/V cache after the first prompt step.

## AMD Radeon RX 7800 XT

ROCm is not available on native Windows. Use **torch-directml** (DirectX 12). AMP and `torch.compile` are skipped on DirectML (FP32).

Look for: `Device backend: directml`, `Device name: AMD Radeon RX 7800 XT`, `Proof: batch x.device=privateuseone:1` matching a model parameter — not `cpu`. Saved: `results/gpu_proof.json`.

DirectML training uses FP32, `AdamW(foreach=False)`, and grad clip 1.0. On this stack, DML val loss can fall while a CPU re-eval of the cloned `state_dict` does not — checkpoints are gated on that CPU number. Sampling loads the checkpoint on CPU then moves the model to DirectML (`sample.py`).
