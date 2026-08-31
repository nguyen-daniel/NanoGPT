"""
Peak-memory (and optional speed) comparison: SDPA vs manual causal attention.

Fair comparison: same batch, sequence length, heads, and embedding dim.
Writes measured ratios — does not assume 2–4x.

Usage:
    python benches/bench_attention.py
    python benches/bench_attention.py --block_size 512 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from device import detect_device, report_device  # noqa: E402
from model import SDPA_AVAILABLE, GPTConfig, CausalSelfAttention  # noqa: E402


def _sync(device):
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize()


def _cuda_peak_bytes():
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None


def _run_forward(attn, x, steps: int):
    y = None
    for _ in range(steps):
        y = attn(x)
        if y.requires_grad:
            y.sum().backward()
            attn.zero_grad(set_to_none=True)
    return y


def measure(attn, x, device, warmup: int, steps: int) -> dict:
    attn = attn.to(device)
    x = x.to(device)

    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    for _ in range(warmup):
        y = attn(x)
        if x.requires_grad:
            y.sum().backward()
            attn.zero_grad(set_to_none=True)
    _sync(device)

    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    y = _run_forward(attn, x, steps)
    _sync(device)
    elapsed = (time.perf_counter() - t0) / steps

    peak = _cuda_peak_bytes()
    # Theoretical attention matrix: B * H * T * T * 4 bytes (fp32 scores)
    b, t, _ = x.shape
    h = attn.n_head
    theoretical_attn_bytes = b * h * t * t * 4

    return {
        "seconds_per_fwd": elapsed,
        "cuda_peak_bytes": peak,
        "theoretical_attn_matrix_bytes": theoretical_attn_bytes,
        "output_device": str(y.device) if y is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    info = detect_device(args.device)
    report_device(info)
    device = info.device

    cfg_sdpa = GPTConfig(
        block_size=args.block_size,
        n_head=args.n_head,
        n_embd=args.n_embd,
        use_sdpa=True,
    )
    cfg_manual = GPTConfig(
        block_size=args.block_size,
        n_head=args.n_head,
        n_embd=args.n_embd,
        use_sdpa=False,
    )

    x = torch.randn(args.batch_size, args.block_size, args.n_embd, requires_grad=True)

    attn_sdpa = CausalSelfAttention(cfg_sdpa)
    attn_manual = CausalSelfAttention(cfg_manual)

    sdpa = measure(attn_sdpa, x, device, args.warmup, args.steps)
    # Fresh x so autograd graphs do not share
    x2 = torch.randn(args.batch_size, args.block_size, args.n_embd, requires_grad=True)
    manual = measure(attn_manual, x2, device, args.warmup, args.steps)

    ratio = None
    if sdpa["cuda_peak_bytes"] and manual["cuda_peak_bytes"] and sdpa["cuda_peak_bytes"] > 0:
        ratio = manual["cuda_peak_bytes"] / sdpa["cuda_peak_bytes"]

    result = {
        "hardware": {
            "os": f"{platform.system()} {platform.release()}",
            "backend": info.backend,
            "device_name": info.name,
            "torch_device": str(info.device),
            "sdpa_available": SDPA_AVAILABLE,
            "sdpa_impl": str(getattr(F, "scaled_dot_product_attention", None)),
        },
        "config": {
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
        },
        "sdpa": sdpa,
        "manual": manual,
        "manual_over_sdpa_peak_memory": ratio,
        "theoretical_attn_matrix_mib": sdpa["theoretical_attn_matrix_bytes"] / (1024 * 1024),
        "resume_claim": "2–4x memory reduction vs vanilla PyTorch attention",
        "claim_met": bool(ratio is not None and 2.0 <= ratio <= 8.0),
        "note": (
            "SDPA is F.scaled_dot_product_attention; the backend may dispatch FlashAttention. "
            "On CUDA, peak allocated bytes are measured. On DirectML/CPU, CUDA peak "
            "is unavailable — compare theoretical attn-matrix size and wall time. "
            "DirectML may execute SDPA as math attention, so the memory ratio can be ~1x."
        ),
    }

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "attention_bench.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== Attention memory/speed benchmark ===")
    print(f"SDPA   time: {sdpa['seconds_per_fwd'] * 1e3:.2f} ms   peak={sdpa['cuda_peak_bytes']}")
    print(f"Manual time: {manual['seconds_per_fwd'] * 1e3:.2f} ms   peak={manual['cuda_peak_bytes']}")
    print(f"Peak memory ratio (manual/SDPA): {ratio}")
    print(f"Theoretical attn matrix: {result['theoretical_attn_matrix_mib']:.2f} MiB")
    print(f"Resume 2–4x claim met (CUDA peak): {result['claim_met']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
