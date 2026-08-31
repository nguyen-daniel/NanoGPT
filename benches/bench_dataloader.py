"""
Compare the old Python-loop get_batch vs the vectorized gather path.

Measures host-side batch construction time (and H2D when the dataset is
kept on CPU). Does not invent a 40% figure — writes whatever this machine
actually measures.

Usage:
    python benches/bench_dataloader.py
    python benches/bench_dataloader.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from device import detect_device, report_device  # noqa: E402
from train import get_batch, get_batch_loop  # noqa: E402


def _sync(device):
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, iters: int, warmup: int, device) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    info = detect_device(args.device)
    report_device(info)
    device = info.device

    data_cpu = torch.randint(0, 1000, (args.n,), dtype=torch.long)

    def loop_cpu_then_to():
        return get_batch_loop(data_cpu, args.block_size, args.batch_size, device)

    def vec_cpu_then_to():
        return get_batch(data_cpu, args.block_size, args.batch_size, device)

    loop_s = time_fn(loop_cpu_then_to, args.iters, args.warmup, device)
    vec_s = time_fn(vec_cpu_then_to, args.iters, args.warmup, device)

    gpu_resident = None
    gpu_s = None
    try:
        data_gpu = data_cpu.to(device)
        x, y = get_batch(data_gpu, args.block_size, args.batch_size, device)
        assert str(x.device) == str(device) or x.device.type == getattr(device, "type", None)

        def vec_on_device():
            return get_batch(data_gpu, args.block_size, args.batch_size, device)

        gpu_s = time_fn(vec_on_device, args.iters, args.warmup, device)
        gpu_resident = True
    except Exception as exc:
        print(f"GPU-resident gather failed ({exc}); skipping that path.")
        gpu_resident = False

    reduction_vec_vs_loop = (loop_s - vec_s) / loop_s if loop_s > 0 else 0.0
    best_s = min(s for s in (vec_s, gpu_s) if s is not None)
    reduction_best_vs_loop = (loop_s - best_s) / loop_s if loop_s > 0 else 0.0

    result = {
        "hardware": {
            "os": f"{platform.system()} {platform.release()}",
            "backend": info.backend,
            "device_name": info.name,
            "torch_device": str(info.device),
        },
        "config": {
            "tokens": args.n,
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "iters": args.iters,
        },
        "seconds_per_batch": {
            "python_loop_plus_to_device": loop_s,
            "vectorized_cpu_then_to_device": vec_s,
            "vectorized_device_resident": gpu_s,
        },
        "overhead_reduction_vs_python_loop": {
            "vectorized_cpu_then_to_device": reduction_vec_vs_loop,
            "best_path": reduction_best_vs_loop,
        },
        "gpu_resident_gather_ok": gpu_resident,
        "resume_claim": "GPU-accelerated tensor ops reducing data loading overhead by 40%",
        "claim_met": reduction_best_vs_loop >= 0.40,
        "note": (
            "Measured batch-construction overhead only (not full training step). "
            "The 'python_loop' path is the historical get_batch implementation."
        ),
    }

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "dataloader_bench.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== Data-loading benchmark ===")
    print(f"loop+to(device):     {loop_s * 1e3:.3f} ms/batch")
    print(f"vectorized+to:       {vec_s * 1e3:.3f} ms/batch  ({reduction_vec_vs_loop * 100:.1f}% vs loop)")
    if gpu_s is not None:
        print(f"device-resident:     {gpu_s * 1e3:.3f} ms/batch  ({reduction_best_vs_loop * 100:.1f}% vs loop)")
    print(f"Resume 40% claim met: {result['claim_met']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
