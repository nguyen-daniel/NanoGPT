# NanoGPT

[![CI](https://github.com/nguyen-daniel/NanoGPT/actions/workflows/main.yml/badge.svg)](https://github.com/nguyen-daniel/NanoGPT/actions/workflows/main.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Decoder-only GPT in PyTorch (Karpathy-style): train on Tiny Shakespeare or your own text, sample from a checkpoint. Character tokenizer by default; optional BPE.

## Demo (no training)

One command → Shakespeare-ish text from a **CPU-trained** 6×6×192 checkpoint (val **1.71**). Downloads ~10–15MB weights from a GitHub Release if they are not already in `out_demo/`.

```bash
git clone https://github.com/nguyen-daniel/NanoGPT.git
cd NanoGPT
pip install -r requirements.txt   # torch + requests; skip if already installed
make demo
```

Windows without Make: `demo.bat`, or:

```bash
python scripts/download_demo_ckpt.py
python sample.py --checkpoint out_demo/ckpt.pt --prompt "ROMEO:" --num_tokens 200 --temperature 0.8 --top_k 40 --device cpu
```

That checkpoint stores its own 65-char vocab (Tiny Shakespeare **before** UNK). Do not decode it with a current 66+UNK `data/vocab.pt` — `sample.py` prefers the vocab in the file.

## What I built

- Decoder-only GPT: pre-norm blocks, causal attention, GELU MLP, weight tying (~10–15M params at default 6×6×384)
- Attention via **PyTorch SDPA** (`F.scaled_dot_product_attention`); `--no_sdpa` is the manual `QK^T` + causal-mask fallback. SDPA may dispatch a FlashAttention kernel when the backend provides one — this repo does not implement FlashAttention.
- **DirectML** device path for Windows + AMD (RX 7800 XT); AMP and `torch.compile` are skipped on DirectML (FP32). DirectML can train; its reported val loss is not trusted — always re-eval on CPU (`results/directml.md`).
- Vectorized `get_batch()` with optional device-resident tokens (no Python index loop)
- Character tokenizer by default (unknown chars map to UNK); optional BPE (`python data.py --tokenizer bpe`)
- Dropout on `GPTConfig` (`--dropout`, default 0.1); AdamW decay vs no-decay groups; `--grad_clip` (default 1.0)
- `ckpt.pt` stores tokenizer type, seed, torch version, git SHA, full argv, and the char `vocab` list (65 vs 66+UNK compatibility)

## Train from scratch

```bash
# CPU / CI  (torch>=2.0,<3 and requests>=2.28,<3; not torch-directml)
pip install -r requirements.txt

python data.py
python train.py --max_iters 50 --eval_interval 50 --eval_iters 2 --no_amp --no_compile
python sample.py --prompt "ROMEO:"
python -m unittest discover -s tests -v
# lint (optional): pip install ruff && ruff format --check . && ruff check .
```

Score any checkpoint on CPU (including one trained on DirectML):

```bash
python train.py --eval_only --eval_device cpu --checkpoint out/ckpt.pt
# after a DirectML train:
python train.py --device directml --cpu_eval --no_amp --no_compile
```

Windows + AMD (RX 7800 XT): Python 3.11, `pip install -r requirements-directml.txt`, then `python device.py` (expect DirectML / RX 7800 XT). Do not pre-install a newer torch.

## Proof

GPU device proof (DirectML, RX 7800 XT): [`results/gpu_proof.json`](results/gpu_proof.json). The 3-iter `ROMEO:ent AR#go.` string is GPU smoke, not a trained language model.

CPU-trained sample (6×6×192, 5000 iters, seed 1337, iter 4750, val **1.71**): [`results/sample_shakespeare.txt`](results/sample_shakespeare.txt), metrics in [`results/train_run.json`](results/train_run.json). Recovered from PR #6 / commit `5d3d217`; the same `out_cpu/ckpt.pt` is still on this machine (`best_val_loss=1.7092256546020508`).

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

That is play-formatted, misspelled Shakespeare-like text — not the 250-iter `ROMEO: ll ntr t utla…` smoke string.

DirectML honesty: a 6×6×384 DML run reported val **1.47**; CPU re-eval of those weights was **5.86**. Read [`results/directml.md`](results/directml.md). Always `--eval_device cpu`.

Eval artifact (endpoints only; per-iter CPU logs were not retained):

![CPU val 1.71 vs untrusted DirectML 1.47 vs CPU re-eval 5.86](results/loss_curve.svg)

JSON: [`results/loss_curve.json`](results/loss_curve.json).

## Results

Device proof measured on **AMD Radeon RX 7800 XT** (Windows, **DirectML** — not CUDA). Sample metrics: [`results/train_run.json`](results/train_run.json). BPE: [`results/bpe_compression.json`](results/bpe_compression.json) (`python benches/bench_bpe.py`).

| Topic | What is true | Artifact |
|-------|----------------|----------|
| GPU | Training tensors live on `privateuseone:1` / RX 7800 XT | `scripts/check_gpu.py`, train log `Proof: batch x.device=...` |
| Sample | CPU 6×6×192, 5000 iters, seed 1337, val 1.71; speaker names + verse-ish lines | `results/sample_shakespeare.txt`, `results/train_run.json`, `make demo` |
| DirectML eval | Long DML runs report falling val; CPU re-eval of cloned weights does not follow | `results/directml.md`, `results/train_run.json` (`cpu_reeval_of_that_ckpt_val_loss`) |
| Attention | Causal self-attention via **`F.scaled_dot_product_attention`** (PyTorch SDPA; may dispatch FlashAttention **when the backend provides it**). Manual attention is `--no_sdpa`. | `model.py`, `tests/test_attention.py` |
| Size | Default 6×6×384 is ~10–15M parameters (vocab-dependent); printed at train start | train log |
| BPE | 2.49× sequence compression vs char on Tiny Shakespeare (vocab 1000) | `python benches/bench_bpe.py` → `results/bpe_compression.json` |
| Batching | Vectorized gather; optional device-resident tokens | `get_batch()` vs `get_batch_loop()` in `train.py` |
| Train loop | `get_batch` / `get_lr` / checkpoint roundtrip / `--resume` / `_orig_mod` strip / `prepare_data` / sample generate + top-k/top-p / CPU-eval / demo download | `tests/test_train.py`, `tests/test_data.py`, `tests/test_sample.py`, `tests/test_download_demo_ckpt.py` |

No 2–4× memory or 40% data-loading claims, and no 100 iter/s or &lt;10ms/token claims. If you run `make bench`, treat the JSON as the only numbers.

## Architecture

Token + position embeddings → decoder blocks (pre-norm, causal attention, GELU MLP) → LM head with weight tying. SDPA when `use_sdpa` is on; otherwise explicit `QK^T` + causal mask.

## AMD Radeon RX 7800 XT

ROCm is not available on native Windows. Use **torch-directml** (DirectX 12). AMP and `torch.compile` are skipped on DirectML (FP32).

Look for: `Device backend: directml`, `Device name: AMD Radeon RX 7800 XT`, `Proof: batch x.device=privateuseone:1` matching a model parameter — not `cpu`. Saved: `results/gpu_proof.json`.

A 3-iteration smoke train is enough to prove the device; generated text will be noise until you run a full `python train.py` or use `make demo`. Sampling loads the checkpoint on CPU then moves the model to the requested device (`sample.py`). DirectML val is not ground truth — `python train.py --eval_only --eval_device cpu`.
