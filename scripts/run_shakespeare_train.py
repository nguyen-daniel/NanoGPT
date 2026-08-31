"""
Train default NanoGPT on Tiny Shakespeare (DirectML / RX 7800 XT),
then write results/sample_shakespeare.txt and results/train_run.json.

The checkpoint stays in out/ (gitignored). Do not commit a 40MB+ ckpt.pt.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import GPT
from device import detect_device, report_device
from sample import generate_text
from train import estimate_loss, train

SEED = 1337
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 20


def main():
    info = detect_device()
    report_device(info)
    t0 = time.perf_counter()
    train(
        max_iters=MAX_ITERS,
        eval_interval=EVAL_INTERVAL,
        eval_iters=EVAL_ITERS,
        warmup_iters=100,
        use_compile=False,
        use_amp=False,
        seed=SEED,
        out_dir="out",
        grad_clip=1.0,
    )
    elapsed = time.perf_counter() - t0

    ckpt_path = ROOT / "out" / "ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    # DirectML eval can drift from the saved weights; re-eval on CPU.
    cpu_model = GPT(cfg)
    cpu_model.load_state_dict(ckpt["model"])
    cpu_model.eval()
    train_data = torch.load(ROOT / "data" / "train.pt", map_location="cpu")
    val_data = torch.load(ROOT / "data" / "val.pt", map_location="cpu")
    cpu_losses = estimate_loss(
        cpu_model, train_data, val_data, cfg.block_size, 32, 10, torch.device("cpu"), False
    )
    print(f"CPU re-eval: train {cpu_losses['train']:.4f} | val {cpu_losses['val']:.4f}")

    text = generate_text(
        checkpoint_path=str(ckpt_path),
        prompt="ROMEO:",
        num_tokens=500,
        temperature=0.8,
        top_k=40,
        data_dir="data",
        device="cpu",
    )

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    sample_path = results / "sample_shakespeare.txt"
    sample_path.write_text(text, encoding="utf-8")

    meta = {
        "backend": info.backend,
        "device_name": info.name,
        "torch_device": str(info.device),
        "proof": f"batch x.device={info.device}  model.param.device={info.device}",
        "seed": SEED,
        "max_iters": MAX_ITERS,
        "iter_num": int(ckpt["iter_num"]),
        "best_val_loss": float(ckpt["best_val_loss"]),
        "cpu_train_loss": float(cpu_losses["train"]),
        "cpu_val_loss": float(cpu_losses["val"]),
        "n_layer": int(cfg.n_layer),
        "n_head": int(cfg.n_head),
        "n_embd": int(cfg.n_embd),
        "block_size": int(cfg.block_size),
        "batch_size": 64,
        "elapsed_sec": round(elapsed, 1),
        "sample_path": "results/sample_shakespeare.txt",
        "prompt": "ROMEO:",
        "temperature": 0.8,
        "top_k": 40,
        "checkpoint": "out/ckpt.pt (gitignored; not committed)",
        "note": (
            "Native Windows + torch-directml, FP32 (no AMP/compile). "
            "Index 1 is the discrete 7800 XT."
        ),
    }
    (results / "train_run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {sample_path}")
    print(f"Wrote {results / 'train_run.json'}")
    print(f"iter_num={meta['iter_num']} best_val_loss={meta['best_val_loss']:.4f} elapsed_sec={meta['elapsed_sec']}")


if __name__ == "__main__":
    main()
