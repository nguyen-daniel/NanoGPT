"""
Device detection for NanoGPT.

Priority:
  1. CUDA (NVIDIA)
  2. ROCm/HIP (AMD on Linux; appears as torch.cuda)
  3. DirectML (AMD/Intel/NVIDIA on native Windows)
  4. MPS (Apple Silicon)
  5. CPU

On this project the intended Windows+AMD path is torch-directml
(Radeon RX 7800 XT / gfx1101). CUDA will not work on AMD.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved training/inference device."""

    backend: str  # 'cuda' | 'rocm' | 'directml' | 'mps' | 'cpu'
    device: object  # torch.device
    name: str
    index: int | None = None

    def __str__(self) -> str:
        extra = f" index={self.index}" if self.index is not None else ""
        return f"{self.backend}:{self.name}{extra} ({self.device})"


def _directml_available():
    try:
        import torch_directml

        return torch_directml.is_available()
    except Exception:
        return False


def _pick_directml():
    """Prefer a discrete Radeon (e.g. 7800 XT) over the iGPU."""
    import torch_directml

    count = torch_directml.device_count()
    names = [torch_directml.device_name(i).replace("\x00", "").strip() for i in range(count)]

    preferred = None
    for i, name in enumerate(names):
        upper = name.upper()
        if any(token in upper for token in ("7800", "RX ", "RADEON RX", "DISCRETE")):
            preferred = i
            break
        if "RADEON" in upper and "GRAPHICS" not in upper and preferred is None:
            preferred = i

    if preferred is None:
        preferred = 0

    return torch_directml.device(preferred), names[preferred], preferred, names


def detect_device(requested: str | None = None) -> DeviceInfo:
    """
    Resolve the best available device.

    Args:
        requested: Optional override: 'cuda', 'rocm', 'dml', 'directml', 'mps', 'cpu', or None.

    Returns:
        DeviceInfo with backend name, torch.device, and human-readable GPU name.
    """
    import torch

    req = (requested or "").strip().lower() or None
    if req in ("dml", "directml"):
        req = "directml"
    if req == "rocm":
        req = "cuda"  # ROCm uses the cuda device type in PyTorch

    def _cuda_info() -> DeviceInfo:
        gpu_name = torch.cuda.get_device_name(0)
        hip = getattr(torch.version, "hip", None)
        is_amd = "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper() or bool(hip)
        backend = "rocm" if is_amd else "cuda"
        return DeviceInfo(backend=backend, device=torch.device("cuda"), name=gpu_name, index=0)

    if req is None:
        if torch.cuda.is_available():
            return _cuda_info()
        if _directml_available():
            dml_dev, name, idx, _ = _pick_directml()
            return DeviceInfo(backend="directml", device=dml_dev, name=name, index=idx)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceInfo(
                backend="mps", device=torch.device("mps"), name="Apple Silicon", index=0
            )
        return DeviceInfo(
            backend="cpu", device=torch.device("cpu"), name=platform.processor() or "CPU"
        )

    if req == "cuda":
        if torch.cuda.is_available():
            return _cuda_info()
        print("Warning: CUDA/ROCm requested but not available. Falling back.")
        return (
            detect_device(None)
            if _directml_available()
            else DeviceInfo(
                backend="cpu", device=torch.device("cpu"), name="CPU (CUDA unavailable)"
            )
        )

    if req == "directml":
        if _directml_available():
            dml_dev, name, idx, _ = _pick_directml()
            return DeviceInfo(backend="directml", device=dml_dev, name=name, index=idx)
        print(
            "Warning: DirectML requested but torch-directml is not installed. Falling back to CPU."
        )
        print("  Install (Python 3.11): pip install torch-directml")
        return DeviceInfo(
            backend="cpu", device=torch.device("cpu"), name="CPU (DirectML unavailable)"
        )

    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceInfo(
                backend="mps", device=torch.device("mps"), name="Apple Silicon", index=0
            )
        print("Warning: MPS requested but not available. Falling back to CPU.")
        return DeviceInfo(backend="cpu", device=torch.device("cpu"), name="CPU (MPS unavailable)")

    if req == "cpu":
        return DeviceInfo(
            backend="cpu", device=torch.device("cpu"), name=platform.processor() or "CPU"
        )

    print(f"Warning: Unknown device '{requested}'. Auto-detecting.")
    return detect_device(None)


def report_device(info: DeviceInfo) -> None:
    """Print a recruiter-auditable device line."""
    print(f"Device backend: {info.backend}")
    print(f"Device name:    {info.name}")
    print(f"torch.device:   {info.device}")
    if info.backend in ("cuda", "rocm"):
        import torch

        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory:     {props.total_memory / 1e9:.2f} GB")
        hip = getattr(torch.version, "hip", None)
        if hip:
            print(f"HIP/ROCm:       {hip}")


def set_seed(seed: int = 1337) -> None:
    """Seed Python, NumPy (if present), and PyTorch for reproducible runs."""
    import random

    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def supports_amp(info: DeviceInfo) -> bool:
    """Mixed precision is only reliable on CUDA/ROCm, not DirectML."""
    return info.backend in ("cuda", "rocm")


def supports_compile(info: DeviceInfo) -> bool:
    """torch.compile is most useful on Linux + CUDA/ROCm."""
    return platform.system() == "Linux" and info.backend in ("cuda", "rocm")


if __name__ == "__main__":
    info = detect_device()
    report_device(info)
    print(f"Resolved: {info}")
