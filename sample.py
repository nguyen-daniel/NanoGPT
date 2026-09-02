"""
Text generation script for NanoGPT.
Loads a trained checkpoint and generates text from a prompt.
Automatically detects tokenizer type (character or BPE) from the data directory.
"""

import torch
import argparse
from pathlib import Path
from model import GPT, strip_orig_mod_prefix
from tokenizer import load_tokenizer
from device import detect_device, report_device


def load_checkpoint(checkpoint_path, device):
    """
    Load model checkpoint and configuration.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        model: Loaded GPT model
        config: GPTConfig used for the model
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    # DirectML rejects map_location=dml_device; load on CPU then .to(device).
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract configuration
    config = checkpoint['config']
    if not hasattr(config, 'dropout'):
        config.dropout = 0.1
    print(f"Model config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd")
    meta_keys = ('tokenizer_type', 'seed', 'torch_version', 'git_sha', 'argv')
    if any(k in checkpoint for k in meta_keys):
        print(
            "Run metadata: "
            f"tokenizer={checkpoint.get('tokenizer_type')} "
            f"seed={checkpoint.get('seed')} "
            f"torch={checkpoint.get('torch_version')} "
            f"git={checkpoint.get('git_sha')}"
        )
        argv = checkpoint.get('argv')
        if argv:
            print(f"  argv: {argv}")

    # Initialize model
    model = GPT(config)

    state_dict = checkpoint['model']
    stripped = strip_orig_mod_prefix(state_dict)
    if stripped is not state_dict:
        print("Detected compiled model checkpoint, stripping '_orig_mod.' prefix...")
        state_dict = stripped

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully ({model.get_num_params() / 1e6:.2f}M parameters)")
    return model, config


def load_vocabulary(data_dir='data'):
    """
    Load tokenizer from data directory.

    Automatically detects tokenizer type (character or BPE) from saved metadata.

    Args:
        data_dir: Directory containing vocabulary/tokenizer data

    Returns:
        Dictionary with 'encode', 'decode' functions and 'vocab_size'
    """
    data_path = Path(data_dir)
    vocab_path = data_path / 'vocab.pt'

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Run data.py first.")

    print(f"Loading tokenizer from {vocab_path}...")

    # Load tokenizer using the unified loader (auto-detects type)
    tokenizer = load_tokenizer(vocab_path)

    print(f"Tokenizer loaded: {tokenizer.type}, vocab_size={tokenizer.vocab_size}")

    return {
        'encode': tokenizer.encode,
        'decode': tokenizer.decode,
        'vocab_size': tokenizer.vocab_size,
        'tokenizer': tokenizer,
    }


def generate_text(
    checkpoint_path='out/ckpt.pt',
    prompt='\n',
    num_tokens=500,
    temperature=1.0,
    top_k=None,
    top_p=None,
    data_dir='data',
    device=None,  # None = auto-detect, or specify 'cuda', 'mps', 'cpu'
):
    """
    Generate text from a prompt using a trained model.

    Args:
        checkpoint_path: Path to the model checkpoint
        prompt: Starting text prompt
        num_tokens: Number of new tokens to generate
        temperature: Sampling temperature (1.0 = default, >1.0 = more random, <1.0 = more focused)
        top_k: If specified, only sample from the top-k most likely tokens
        top_p: If specified, nucleus sampling (keep tokens with cumulative prob >= top_p).
               Combined with top_k, top-k is applied first, then top-p.
        data_dir: Directory containing vocabulary data
        device: Device to run generation on (None = auto-detect, 'cuda', 'mps', or 'cpu')
    """
    info = detect_device(device)
    report_device(info)
    device_obj = info.device

    # Load vocabulary
    vocab = load_vocabulary(data_dir)
    encode = vocab['encode']
    decode = vocab['decode']

    # Load model
    model, config = load_checkpoint(checkpoint_path, device_obj)

    # Encode the prompt
    print(f"\nPrompt: {repr(prompt)}")
    prompt_tokens = encode(prompt)
    print(f"Prompt length: {len(prompt_tokens)} tokens")

    # Convert to tensor
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device_obj)

    # Generate new tokens
    print(f"\nGenerating {num_tokens} new tokens...")
    print("-" * 60)

    generated_idx = model.generate(
        idx, max_new_tokens=num_tokens, temperature=temperature, top_k=top_k, top_p=top_p
    )

    # Decode the generated tokens
    generated_tokens = generated_idx[0].tolist()
    generated_text = decode(generated_tokens)

    print("Generated text:")
    print("=" * 60)
    try:
        print(generated_text)
    except UnicodeEncodeError:
        print(generated_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
    print("=" * 60)

    return generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using a trained NanoGPT model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='out/ckpt.pt',
        help='Path to model checkpoint (default: out/ckpt.pt)',
    )
    parser.add_argument(
        '--prompt', type=str, default='\n', help='Starting prompt text (default: \\n)'
    )
    parser.add_argument(
        '--num_tokens', type=int, default=500, help='Number of tokens to generate (default: 500)'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0, help='Sampling temperature (default: 1.0)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Top-k sampling (default: None). Applied before top-p when both are set.',
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=None,
        help='Top-p / nucleus sampling (default: None). Applied after top-k when both are set.',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing vocabulary data (default: data)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/directml/mps/cpu, default: auto-detect)',
    )

    args = parser.parse_args()

    generate_text(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        data_dir=args.data_dir,
        device=args.device,
    )
