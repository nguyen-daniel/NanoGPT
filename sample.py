"""
Text generation script for NanoGPT.
Loads a trained checkpoint and generates text from a prompt.
"""

import torch
import argparse
from pathlib import Path
from model import GPT, GPTConfig


def get_device(device=None):
    """
    Get the best available device for inference.
    
    Supports:
    - CUDA (NVIDIA GPUs)
    - ROCm (AMD GPUs) - accessed via 'cuda' device string
    - MPS (Apple Silicon GPUs)
    - CPU (fallback)
    
    Args:
        device: Optional device string ('cuda', 'mps', 'cpu', or None for auto-detect)
    
    Returns:
        Device string and device object
    """
    if device is None:
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            
            # Detect if it's an AMD GPU (ROCm) or NVIDIA GPU
            is_amd = 'AMD' in gpu_name.upper() or 'Radeon' in gpu_name or 'ROCm' in str(torch.version.hip) if hasattr(torch.version, 'hip') else False
            
            if is_amd:
                print(f"ROCm available: Using AMD GPU ({gpu_name})")
            else:
                print(f"CUDA available: Using NVIDIA GPU ({gpu_name})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("MPS available: Using Apple Silicon GPU")
        else:
            device = 'cpu'
            print("Using CPU (no GPU available)")
    else:
        # Validate requested device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA/ROCm requested but not available. Falling back to CPU.")
            print("Note: For AMD GPUs, install PyTorch with ROCm support:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7")
            device = 'cpu'
        elif device == 'mps' and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = 'cpu'
        elif device not in ['cuda', 'mps', 'cpu']:
            print(f"Warning: Unknown device '{device}'. Using CPU.")
            device = 'cpu'
    
    return device, torch.device(device)


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
    # weights_only=False is safe here since we trust our own checkpoints
    # PyTorch 2.6+ defaults to weights_only=True for security
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration
    config = checkpoint['config']
    print(f"Model config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd")
    
    # Initialize model
    model = GPT(config)
    
    # Load state dict - handle both compiled and uncompiled checkpoints
    state_dict = checkpoint['model']
    
    # If checkpoint was saved from a compiled model, strip '_orig_mod.' prefix
    # torch.compile() wraps the model and adds this prefix to parameter names
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("Detected compiled model checkpoint, stripping '_orig_mod.' prefix...")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully ({model.get_num_params() / 1e6:.2f}M parameters)")
    return model, config


def load_vocabulary(data_dir='data'):
    """
    Load vocabulary and encoding/decoding functions from data.py.
    
    Args:
        data_dir: Directory containing vocabulary metadata
    
    Returns:
        Dictionary with 'encode' and 'decode' functions
    """
    data_path = Path(data_dir)
    vocab_path = data_path / 'vocab.pt'
    
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Run data.py first.")
    
    print(f"Loading vocabulary from {vocab_path}...")
    # weights_only=False is safe here since we trust our own vocabulary files
    vocab_metadata = torch.load(vocab_path, weights_only=False)
    
    # Create encode and decode functions
    char_to_int = vocab_metadata['char_to_int']
    int_to_char = vocab_metadata['int_to_char']
    
    def encode(text):
        """Convert string to list of integers."""
        return [char_to_int[ch] for ch in text]
    
    def decode(integers):
        """Convert list of integers to string."""
        return ''.join([int_to_char[i] for i in integers])
    
    vocab_size = vocab_metadata['vocab_size']
    print(f"Vocabulary loaded: {vocab_size} unique characters")
    
    return {'encode': encode, 'decode': decode, 'vocab_size': vocab_size}


def generate_text(
    checkpoint_path='out/ckpt.pt',
    prompt='\n',
    num_tokens=500,
    temperature=1.0,
    top_k=None,
    data_dir='data',
    device=None  # None = auto-detect, or specify 'cuda', 'mps', 'cpu'
):
    """
    Generate text from a prompt using a trained model.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        prompt: Starting text prompt
        num_tokens: Number of new tokens to generate
        temperature: Sampling temperature (1.0 = default, >1.0 = more random, <1.0 = more focused)
        top_k: If specified, only sample from top-k most likely tokens
        data_dir: Directory containing vocabulary data
        device: Device to run generation on (None = auto-detect, 'cuda', 'mps', or 'cpu')
    """
    # Determine device
    device_str, device_obj = get_device(device)
    
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
        idx,
        max_new_tokens=num_tokens,
        temperature=temperature,
        top_k=top_k
    )
    
    # Decode the generated tokens
    generated_tokens = generated_idx[0].tolist()
    generated_text = decode(generated_tokens)
    
    # Display results
    print("Generated text:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)
    
    return generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using a trained NanoGPT model')
    parser.add_argument('--checkpoint', type=str, default='out/ckpt.pt',
                        help='Path to model checkpoint (default: out/ckpt.pt)')
    parser.add_argument('--prompt', type=str, default='\n',
                        help='Starting prompt text (default: \\n)')
    parser.add_argument('--num_tokens', type=int, default=500,
                        help='Number of tokens to generate (default: 500)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k sampling (default: None, no filtering)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing vocabulary data (default: data)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')
    
    args = parser.parse_args()
    
    generate_text(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        data_dir=args.data_dir,
        device=args.device
    )

