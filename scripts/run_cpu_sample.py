"""CPU train for a CPU-verified Shakespeare sample (DirectML long-run eval diverges)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import GPT
from sample import generate_text
from train import estimate_loss, train

SEED = 1337
MAX_ITERS = 5000


def main():
    t0 = time.perf_counter()
    train(
        max_iters=MAX_ITERS,
        eval_interval=250,
        eval_iters=20,
        warmup_iters=100,
        use_compile=False,
        use_amp=False,
        seed=SEED,
        out_dir="out_cpu",
        device="cpu",
        batch_size=64,
        block_size=128,
        n_layer=6,
        n_head=6,
        n_embd=192,
        grad_clip=1.0,
    )
    elapsed = time.perf_counter() - t0

    ckpt_path = ROOT / "out_cpu" / "ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    train_data = torch.load(ROOT / "data" / "train.pt", map_location="cpu")
    val_data = torch.load(ROOT / "data" / "val.pt", map_location="cpu")
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    cpu_losses = estimate_loss(
        model, train_data, val_data, cfg.block_size, 32, 20, torch.device("cpu"), False
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
    (results / "sample_shakespeare.txt").write_text(text, encoding="utf-8")

    meta = {
        "sample_train": {
            "device": "cpu",
            "reason": (
                "DirectML 6x6x384 runs on the RX 7800 XT and the live DML val loss "
                "falls, but CPU re-eval of cloned weights stops improving around "
                "val 3.1–3.3 (unigram/bigram stage) then gets worse. The committed "
                "sample is from a CPU-trained 6x6x192 model whose val loss is measured "
                "on CPU."
            ),
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
        },
        "directml_7800xt": {
            "backend": "directml",
            "device_name": "AMD Radeon RX 7800 XT",
            "torch_device": "privateuseone:1",
            "proof": "batch x.device=privateuseone:1  model.param.device=privateuseone:1",
            "seed": 1337,
            "max_iters": 5000,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "block_size": 256,
            "batch_size": 64,
            "directml_reported_best_val_loss": 1.4659,
            "cpu_reeval_of_that_ckpt_val_loss": 5.8566,
            "cpu_gated_best_val_loss": 3.17,
            "cpu_gated_iter": 250,
            "note": (
                "Native Windows + torch-directml, FP32. AdamW foreach=False, grad_clip=1.0, "
                "CPU-cloned checkpoints. DML eval is not used as the save metric."
            ),
        },
        "sample_path": "results/sample_shakespeare.txt",
        "prompt": "ROMEO:",
        "temperature": 0.8,
        "top_k": 40,
        "checkpoint": "out_cpu/ckpt.pt (gitignored; not committed)",
    }
    (results / "train_run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {results / 'sample_shakespeare.txt'}")
    print(f"Wrote {results / 'train_run.json'}")


if __name__ == "__main__":
    main()
