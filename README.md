# NanoGPT

[![CI](https://github.com/nguyen-daniel/NanoGPT/actions/workflows/main.yml/badge.svg)](https://github.com/nguyen-daniel/NanoGPT/actions/workflows/main.yml)

Decoder-only GPT in PyTorch (Karpathy-style): train on Tiny Shakespeare or your own text, sample from a checkpoint. Character tokenizer by default; optional BPE.

## What I built

- Decoder-only GPT: pre-norm blocks, causal attention, GELU MLP, weight tying (~10–15M params at default 6×6×384)
- Attention via **PyTorch SDPA** (`F.scaled_dot_product_attention`); `--no_sdpa` is the manual `QK^T` + causal-mask fallback. SDPA may dispatch a FlashAttention kernel when the backend provides one — this repo does not implement FlashAttention.
- **DirectML** device path for Windows + AMD (RX 7800 XT); AMP and `torch.compile` are skipped on DirectML (FP32)
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

GPU device proof (DirectML, RX 7800 XT): [`results/gpu_proof.json`](results/gpu_proof.json). The 3-iter `ROMEO:ent AR#go.` string is GPU smoke, not a trained language model. A Shakespeare-like sample needs a full `python train.py` checkpoint (`out/ckpt.pt`); this repo does not ship one.

## Results

Measured on **AMD Radeon RX 7800 XT** (Windows, **DirectML** — not CUDA). Reproduce: `python device.py`, then `python train.py --max_iters 5 --no_amp --no_compile`.

| Topic | What is true | Artifact |
|-------|----------------|----------|
| GPU | Training tensors live on `privateuseone:1` / RX 7800 XT | `scripts/check_gpu.py`, train log `Proof: batch x.device=...` |
| Attention | Causal self-attention via **`F.scaled_dot_product_attention`** (PyTorch SDPA; may dispatch FlashAttention **when the backend provides it**). Manual attention is `--no_sdpa`. | `model.py`, `tests/test_attention.py` |
| Size | Default 6×6×384 is ~10–15M parameters (vocab-dependent); printed at train start | train log |
| BPE | Sequence length vs char is measured, not assumed | `python benches/bench_bpe.py` → `results/bpe_compression.json` |
| Batching | Vectorized gather; optional device-resident tokens | `get_batch()` vs `get_batch_loop()` in `train.py` |

No 2–4× memory or 40% data-loading claims. If you run `make bench`, treat the JSON as the only numbers.

## Architecture

Token + position embeddings → decoder blocks (pre-norm, causal attention, GELU MLP) → LM head with weight tying. SDPA when `use_sdpa` is on; otherwise explicit `QK^T` + causal mask.

## AMD Radeon RX 7800 XT

ROCm is not available on native Windows. Use **torch-directml** (DirectX 12). AMP and `torch.compile` are skipped on DirectML (FP32).

Look for: `Device backend: directml`, `Device name: AMD Radeon RX 7800 XT`, `Proof: batch x.device=privateuseone:1` matching a model parameter — not `cpu`. Saved: `results/gpu_proof.json`.

A 3-iteration smoke train is enough to prove the device; generated text will be noise until you run a full `python train.py`. Sampling loads the checkpoint on CPU then moves the model to DirectML (`sample.py`).
