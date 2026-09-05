"""
Download the CPU-trained Shakespeare demo checkpoint.

Weights are hosted as a GitHub Release asset (too large for git). The file
stores a 65-char pre-UNK vocab so sample.py does not need data/vocab.pt.

Usage:
    python scripts/download_demo_ckpt.py
    python scripts/download_demo_ckpt.py --out out_demo/ckpt.pt
    NANOGPT_DEMO_CKPT_URL=... python scripts/download_demo_ckpt.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = (
    "https://github.com/nguyen-daniel/NanoGPT/releases/download/demo-ckpt-cpu-6x6x192/ckpt.pt"
)
DEFAULT_OUT = ROOT / "out_demo" / "ckpt.pt"


def ensure_cpu_deps():
    """Install requirements.txt if torch or requests is missing."""
    missing = []
    for name in ("torch", "requests"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if not missing:
        return
    reqs = ROOT / "requirements.txt"
    print(f"Missing {missing}; installing {reqs} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(reqs)])


def resolve_url(url: str) -> str:
    return os.environ.get("NANOGPT_DEMO_CKPT_URL", url)


def _path_from_file_url(url: str) -> Path:
    parsed = urlparse(url)
    path_str = parsed.path
    if parsed.netloc:
        path_str = f"//{parsed.netloc}{parsed.path}"
    # urlparse('file:///C:/foo') -> '/C:/foo'
    if os.name == "nt" and path_str.startswith("/") and len(path_str) >= 3 and path_str[2] == ":":
        path_str = path_str[1:]
    return Path(path_str)


def is_local_source(url: str) -> Path | None:
    """Return a local path if url is a file:// URI or an existing file."""
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = _path_from_file_url(url)
        return path if path.exists() else None
    # Bare path, or Windows drive letter parsed as the URL scheme.
    if parsed.scheme == "" or (os.name == "nt" and len(parsed.scheme) == 1):
        path = Path(url)
        return path if path.exists() else None
    return None


def download_checkpoint(url: str, dest: Path, force: bool = False) -> Path:
    """
    Fetch url into dest. Local paths and file:// URIs are copied.
    Skips the download when dest already exists unless force=True.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"Checkpoint already present: {dest} ({dest.stat().st_size} bytes)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    source = is_local_source(url)
    if source is not None:
        print(f"Copying local checkpoint {source} -> {dest}")
        shutil.copy2(source, dest)
        return dest

    try:
        import requests
    except ImportError as exc:
        raise SystemExit("requests is required to download the demo checkpoint") from exc

    print(f"Downloading {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".partial")
    written = 0
    try:
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"Wrote {dest} ({written} bytes)")
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download the NanoGPT CPU demo checkpoint")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Release asset URL (or a local path). Overridden by NANOGPT_DEMO_CKPT_URL.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Destination path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if dest exists")
    parser.add_argument(
        "--no_deps",
        action="store_true",
        help="Do not pip-install requirements.txt when torch/requests are missing",
    )
    args = parser.parse_args(argv)

    if not args.no_deps:
        ensure_cpu_deps()

    dest = download_checkpoint(resolve_url(args.url), Path(args.out), force=args.force)
    print(dest)
    return dest


if __name__ == "__main__":
    main()
