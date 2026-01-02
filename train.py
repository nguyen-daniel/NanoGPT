"""
Training script for NanoGPT.
Implements the training loop with AdamW optimizer, batch sampling, and checkpointing.
Includes cosine learning rate decay with warmup, torch.compile, and mixed-precision training.
"""

import os
import platform
import math
import torch
import torch.nn as nn
from pathlib import Path
from model import GPT, GPTConfig
from data import prepare_data


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
    device='cuda' if torch.cuda.is_available() else 'cpu',
    out_dir='out',
    use_compile=True,
    use_amp=True
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
        device: Device to train on ('cuda' or 'cpu')
        out_dir: Directory to save checkpoints
        use_compile: Whether to use torch.compile (only on Linux)
        use_amp: Whether to use automatic mixed precision training
    """
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
    
    # Initialize model
    print("\nInitializing model...")
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )
    model = GPT(config)
    model = model.to(device)
    
    # Print model parameters
    n_params = model.get_num_params() / 1e6
    print(f"Model initialized with {n_params:.2f}M parameters")
    
    # Apply torch.compile if on Linux and requested
    if use_compile and platform.system() == 'Linux' and device == 'cuda':
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled successfully")
    elif use_compile and platform.system() != 'Linux':
        print("Warning: torch.compile is only supported on Linux. Skipping compilation.")
    elif use_compile and device != 'cuda':
        print("Warning: torch.compile is most effective on CUDA. Skipping compilation.")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Mixed precision scaler (only needed for CUDA)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device == 'cuda') else None
    
    # Training loop
    print(f"\nStarting training on {device}...")
    print(f"Max iterations: {max_iters}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Evaluation interval: {eval_interval} iterations")
    if use_amp and device == 'cuda':
        print("Using mixed-precision training (FP16)")
    print("-" * 60)
    
    best_val_loss = float('inf')
    iter_num = 0
    
    while iter_num < max_iters:
        # Update learning rate with warmup and cosine decay
        lr = get_lr(iter_num, learning_rate, warmup_iters, max_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Sample a batch of data
        x, y = get_batch(train_data, block_size, batch_size, device)
        
        # Forward and backward pass with mixed precision if enabled
        if use_amp and device == 'cuda' and scaler is not None:
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
            losses = estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device, use_amp and device == 'cuda')
            current_lr = optimizer.param_groups[0]['lr']
            print(f"iter {iter_num:5d} | lr {current_lr:.2e} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            
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
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {Path(out_dir) / 'ckpt.pt'}")


if __name__ == '__main__':
    # Training hyperparameters
    # These are reasonable defaults for a small model on Shakespeare
    train(
        data_dir='data',
        block_size=256,      # Context length
        batch_size=64,       # Batch size
        n_layer=6,           # Number of transformer layers
        n_head=6,            # Number of attention heads
        n_embd=384,          # Embedding dimension
        learning_rate=3e-4,  # Maximum learning rate
        max_iters=5000,      # Maximum training iterations
        warmup_iters=100,    # Warmup iterations for learning rate
        min_lr=0.1,          # Minimum learning rate (as fraction of max_lr)
        eval_interval=500,   # Evaluate every N iterations
        eval_iters=200,      # Number of iterations to average loss over
        device='cuda' if torch.cuda.is_available() else 'cpu',
        out_dir='out',
        use_compile=True,   # Use torch.compile (Linux only)
        use_amp=True        # Use mixed-precision training
    )

