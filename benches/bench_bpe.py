"""
Measure BPE vs character-level sequence length on Tiny Shakespeare.

Writes results/bpe_compression.json so the 2–4x claim is a file, not a rumor.

Usage:
    python benches/bench_bpe.py
    python benches/bench_bpe.py --vocab_size 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import download_shakespeare  # noqa: E402
from tokenizer import BPETokenizer, CharTokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--vocab_size", type=int, default=1000)
    args = parser.parse_args()

    text_path = download_shakespeare(args.data_dir)
    text = text_path.read_text(encoding="utf-8")
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))

    char_tok = CharTokenizer.train(text)
    char_tokens = char_tok.encode(text)

    bpe_tok = BPETokenizer.train(text, vocab_size=args.vocab_size)
    bpe_tokens = bpe_tok.encode(text)

    ratio_vs_char = n_chars / len(bpe_tokens)
    ratio_vs_bytes = n_bytes / len(bpe_tokens)

    result = {
        "corpus": "Tiny Shakespeare",
        "path": str(text_path),
        "characters": n_chars,
        "utf8_bytes": n_bytes,
        "char_tokenizer": {
            "vocab_size": char_tok.vocab_size,
            "tokens": len(char_tokens),
            "tokens_per_char": len(char_tokens) / n_chars,
        },
        "bpe_tokenizer": {
            "vocab_size": bpe_tok.vocab_size,
            "target_vocab_size": args.vocab_size,
            "tokens": len(bpe_tokens),
        },
        "compression_ratio_vs_char": ratio_vs_char,
        "compression_ratio_vs_utf8_bytes": ratio_vs_bytes,
        "resume_claim": "2–4x sequence compression vs char",
        "claim_met": 2.0 <= ratio_vs_char <= 4.5,
    }

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "bpe_compression.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== BPE compression (Tiny Shakespeare) ===")
    print(f"Characters:          {n_chars:,}")
    print(f"Char tokens:         {len(char_tokens):,}")
    print(f"BPE tokens (V={bpe_tok.vocab_size}): {len(bpe_tokens):,}")
    print(f"Compression vs char: {ratio_vs_char:.3f}x")
    print(f"Resume 2–4x claim met: {result['claim_met']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
