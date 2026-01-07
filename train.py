"""
Training script for NanoGPT.
Implements the training loop with AdamW optimizer, batch sampling, and checkpointing.
Includes cosine learning rate decay with warmup, torch.compile, and mixed-precision training.
Supports resuming from checkpoints and optional TensorBoard logging.
"""

import os
import platform
import math
import torch
import torch.nn as nn
from pathlib import Path
from model import GPT, GPTConfig
from data import prepare_data

# Optional TensorBoard support (built into PyTorch, no extra install needed)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def get_device(device=None):
    """
    Get the best available device for training.
    
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
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Detect if it's an AMD GPU (ROCm) or NVIDIA GPU
            # ROCm devices typically show up with AMD in the name or via backend
            is_amd = 'AMD' in gpu_name.upper() or 'Radeon' in gpu_name or 'ROCm' in str(torch.version.hip) if hasattr(torch.version, 'hip') else False
            
            if is_amd:
                print(f"ROCm available: Using AMD GPU ({gpu_name})")
                print(f"GPU Memory: {gpu_memory:.2f} GB")
            else:
                print(f"CUDA available: Using NVIDIA GPU ({gpu_name})")
                print(f"GPU Memory: {gpu_memory:.2f} GB")
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


def get_batch(data, block_size, batch_size, device):
    """
    Sample a random batch of data for training.
    
    Args:
        data: Tensor of shape (N,) containing token indices
        block_size: Context length (sequence length)
        batch_size: Number of sequences in the batch
        device: Device to place tensors on
    
    Returns:
        x: Input sequences of shape (batch_size, block_size)
        y: Target sequences of shape (batch_size, block_size)
    """
    # Sample random starting indices for each sequence in the batch
    # We need to ensure we don't go out of bounds, so max index is len(data) - block_size
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Extract sequences of length block_size starting from each random index
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Targets are the same sequences shifted by 1 position
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Move to device
    x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it, learning_rate, warmup_iters, max_iters, min_lr):
    """
    Get learning rate with linear warmup and cosine decay.
    
    Args:
        it: Current iteration number
        learning_rate: Maximum learning rate
        warmup_iters: Number of warmup iterations
        max_iters: Maximum number of iterations
        min_lr: Minimum learning rate (as fraction of max_lr)
    
    Returns:
        Current learning rate
    """
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    
    # Cosine decay
    if it > max_iters:
        return min_lr * learning_rate
    
    # Cosine decay from max_lr to min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr * learning_rate + coeff * (learning_rate - min_lr * learning_rate)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device, use_amp=False):
    """
    Estimate the loss on train and validation sets.
    
    Args:
        model: GPT model
        train_data: Training data tensor
        val_data: Validation data tensor
        block_size: Context length
        batch_size: Batch size for evaluation
        eval_iters: Number of iterations to average over
        device: Device to run evaluation on
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Dictionary with 'train' and 'val' loss values
    """
    model.eval()
    out = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    _, loss = model(X, Y)
            else:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    
    model.train()
    return out


def train(
    data_dir='data',
    block_size=256,
    batch_size=64,
    n_layer=6,
    n_head=6,
    n_embd=384,
    learning_rate=3e-4,
    max_iters=5000,
    warmup_iters=100,
    min_lr=0.1,
    eval_interval=500,
    eval_iters=200,
    device=None,  # None = auto-detect, or specify 'cuda', 'mps', 'cpu'
    out_dir='out',
    use_compile=True,
    use_amp=True,
    resume=False,
    use_tensorboard=False,
    use_flash_attn=True
):
    """
    Main training function.
    
    Args:
        data_dir: Directory containing processed data
        block_size: Context length (sequence length)
        batch_size: Batch size for training
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        learning_rate: Maximum learning rate for AdamW optimizer
        max_iters: Maximum number of training iterations
        warmup_iters: Number of warmup iterations for learning rate
        min_lr: Minimum learning rate (as fraction of max_lr)
        eval_interval: Evaluate and print loss every N iterations
        eval_iters: Number of iterations to average loss over during evaluation
        device: Device to train on (None = auto-detect, 'cuda', 'mps', or 'cpu')
        out_dir: Directory to save checkpoints
        use_compile: Whether to use torch.compile (only on Linux with CUDA)
        use_amp: Whether to use automatic mixed precision training (CUDA only)
        resume: Whether to resume training from the latest checkpoint
        use_tensorboard: Whether to log metrics to TensorBoard
        use_flash_attn: Whether to use Flash Attention (PyTorch SDPA) for memory efficiency
    """
    # Determine device
    device_str, device_obj = get_device(device)
    
    # Check if AMD GPU for later use
    is_amd_gpu = False
    if device_str == 'cuda' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        is_amd_gpu = 'AMD' in gpu_name.upper() or 'Radeon' in gpu_name or (hasattr(torch.version, 'hip') and 'ROCm' in str(torch.version.hip))
    
    # Load data
    print("Loading data...")
    data_path = Path(data_dir)
    
    # Check if data files exist, if not prepare them
    if not (data_path / 'train.pt').exists() or not (data_path / 'vocab.pt').exists():
        print("Data files not found. Preparing data...")
        data_dict = prepare_data(data_dir)
        train_data = data_dict['train_data']
        val_data = data_dict['val_data']
        vocab_size = data_dict['vocab_size']
    else:
        print("Loading preprocessed data...")
        train_data = torch.load(data_path / 'train.pt')
        val_data = torch.load(data_path / 'val.pt')
        vocab_metadata = torch.load(data_path / 'vocab.pt')
        vocab_size = vocab_metadata['vocab_size']
    
    print(f"Training data: {len(train_data):,} tokens")
    print(f"Validation data: {len(val_data):,} tokens")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize TensorBoard writer if requested
    writer = None
    if use_tensorboard:
        if TENSORBOARD_AVAILABLE:
            log_dir = Path(out_dir) / 'runs'
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging enabled: {log_dir}")
            print(f"  View with: tensorboard --logdir {log_dir}")
        else:
            print("Warning: TensorBoard requested but torch.utils.tensorboard not available")
    
    # Initialize model
    print("\nInitializing model...")
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        use_flash_attn=use_flash_attn
    )
    model = GPT(config)
    model = model.to(device_obj)
    
    # Print model parameters
    n_params = model.get_num_params() / 1e6
    print(f"Model initialized with {n_params:.2f}M parameters")
    
    # Report Flash Attention status
    from model import FLASH_ATTN_AVAILABLE
    if use_flash_attn and FLASH_ATTN_AVAILABLE:
        print("Flash Attention enabled (PyTorch SDPA)")
    elif use_flash_attn and not FLASH_ATTN_AVAILABLE:
        print("Flash Attention requested but not available (requires PyTorch 2.0+)")
    else:
        print("Flash Attention disabled (using manual attention)")
    
    # Apply torch.compile if on Linux and requested
    if use_compile and platform.system() == 'Linux' and device_str == 'cuda':
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled successfully")
    elif use_compile and platform.system() != 'Linux':
        print("Warning: torch.compile is only supported on Linux. Skipping compilation.")
    elif use_compile and device_str != 'cuda':
        print("Warning: torch.compile is most effective on CUDA. Skipping compilation.")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Mixed precision scaler (only needed for CUDA)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device_str == 'cuda') else None
    
    # Resume from checkpoint if requested
    start_iter = 0
    best_val_loss = float('inf')
    
    if resume:
        checkpoint_path = Path(out_dir) / 'ckpt.pt'
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
            
            # Load model state - handle torch.compile prefix if present
            state_dict = checkpoint['model']
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                print("  Stripping '_orig_mod.' prefix from compiled checkpoint...")
                state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
            
            # Load into model (may need to unwrap if compiled)
            if hasattr(model, '_orig_mod'):
                model._orig_mod.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load scaler state if available
            if scaler is not None and checkpoint.get('scaler') is not None:
                scaler.load_state_dict(checkpoint['scaler'])
            
            # Resume from next iteration
            start_iter = checkpoint['iter_num'] + 1
            best_val_loss = checkpoint['best_val_loss']
            
            print(f"  Resumed at iteration {start_iter}")
            print(f"  Best validation loss so far: {best_val_loss:.4f}")
        else:
            print(f"\nWarning: --resume specified but no checkpoint found at {checkpoint_path}")
            print("  Starting training from scratch...")
    
    # Training loop
    print(f"\nStarting training on {device_str.upper()}...")
    if device_str == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Max iterations: {max_iters}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Evaluation interval: {eval_interval} iterations")
    if use_amp and device_str == 'cuda':
        gpu_type = "AMD GPU (ROCm)" if is_amd_gpu else "NVIDIA GPU"
        print(f"Using mixed-precision training (FP16) on {gpu_type}")
    if resume and start_iter > 0:
        print(f"Resuming from iteration {start_iter}")
    print("-" * 60)
    
    iter_num = start_iter
    
    while iter_num < max_iters:
        # Update learning rate with warmup and cosine decay
        lr = get_lr(iter_num, learning_rate, warmup_iters, max_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Sample a batch of data
        x, y = get_batch(train_data, block_size, batch_size, device_obj)
        
        # Forward and backward pass with mixed precision if enabled
        if use_amp and device_str == 'cuda' and scaler is not None:
            # Mixed precision forward pass
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = model(x, y)
            
            # Mixed precision backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision forward and backward pass
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate periodically
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device_obj, use_amp and device_str == 'cuda')
            current_lr = optimizer.param_groups[0]['lr']
            print(f"iter {iter_num:5d} | lr {current_lr:.2e} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('Loss/train', losses['train'], iter_num)
                writer.add_scalar('Loss/val', losses['val'], iter_num)
                writer.add_scalar('LearningRate', current_lr, iter_num)
            
            # Save checkpoint if validation loss improved
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'scaler': scaler.state_dict() if scaler is not None else None,
                }
                checkpoint_path = Path(out_dir) / 'ckpt.pt'
                torch.save(checkpoint, checkpoint_path)
                print(f"  -> Checkpoint saved (val loss: {best_val_loss:.4f})")
        
        iter_num += 1
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {Path(out_dir) / 'ckpt.pt'}")
    if writer is not None:
        print(f"TensorBoard logs saved to: {Path(out_dir) / 'runs'}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NanoGPT model')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/mps/cpu, default: auto-detect)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing processed data (default: data)')
    parser.add_argument('--block_size', type=int, default=256,
                        help='Context length (default: 256)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--n_layer', type=int, default=6,
                        help='Number of transformer layers (default: 6)')
    parser.add_argument('--n_head', type=int, default=6,
                        help='Number of attention heads (default: 6)')
    parser.add_argument('--n_embd', type=int, default=384,
                        help='Embedding dimension (default: 384)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Maximum learning rate (default: 3e-4)')
    parser.add_argument('--max_iters', type=int, default=5000,
                        help='Maximum training iterations (default: 5000)')
    parser.add_argument('--warmup_iters', type=int, default=100,
                        help='Warmup iterations (default: 100)')
    parser.add_argument('--min_lr', type=float, default=0.1,
                        help='Minimum learning rate as fraction of max_lr (default: 0.1)')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluate every N iterations (default: 500)')
    parser.add_argument('--eval_iters', type=int, default=200,
                        help='Number of iterations to average loss over (default: 200)')
    parser.add_argument('--out_dir', type=str, default='out',
                        help='Output directory for checkpoints (default: out)')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile (Linux CUDA only)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed-precision training (CUDA only)')
    parser.add_argument('--no_flash_attn', action='store_true',
                        help='Disable Flash Attention (PyTorch SDPA)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging (logs to out_dir/runs/)')
    
    args = parser.parse_args()
    
    # Training hyperparameters
    train(
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        min_lr=args.min_lr,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        device=args.device,  # Auto-detect if None
        out_dir=args.out_dir,
        use_compile=not args.no_compile,
        use_amp=not args.no_amp,
        resume=args.resume,
        use_tensorboard=args.tensorboard,
        use_flash_attn=not args.no_flash_attn
    )

