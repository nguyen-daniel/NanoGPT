"""
Training script for NanoGPT.
Implements the training loop with AdamW optimizer, batch sampling, and checkpointing.
Includes cosine learning rate decay with warmup, torch.compile, and mixed-precision training.
Supports resuming from checkpoints and optional TensorBoard logging.
"""

import os
import sys
import subprocess
import platform
import math
import torch
from pathlib import Path
from model import GPT, GPTConfig, strip_orig_mod_prefix
from data import prepare_data
from device import detect_device, report_device, set_seed, supports_amp, supports_compile

# Optional TensorBoard support (built into PyTorch, no extra install needed)
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def configure_optimizer(model, learning_rate, weight_decay=0.1):
    """
    AdamW with decay on matmul weights and no decay on norms, biases, embeddings.

    The tied lm_head shares the token-embedding Parameter, so it is classified
    once (as an embedding: no decay). Unique parameters only — weight tying
    must not put the same tensor in both groups.
    """
    inner = getattr(model, "_orig_mod", model)
    no_decay_tokens = ("bias", "ln_", "norm", "embedding")

    id_to_names = {}
    id_to_param = {}
    for name, param in inner.named_parameters(remove_duplicate=False):
        if not param.requires_grad:
            continue
        pid = id(param)
        id_to_names.setdefault(pid, []).append(name)
        id_to_param[pid] = param

    decay, no_decay = [], []
    for pid, param in id_to_param.items():
        names = id_to_names[pid]
        if param.ndim < 2 or any(any(token in name for token in no_decay_tokens) for name in names):
            no_decay.append(param)
        else:
            decay.append(param)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=learning_rate)


def get_batch_loop(data, block_size, batch_size, device):
    """
    Historical batch path: Python loop + per-batch .to(device).

    Kept so benches/bench_dataloader.py can measure the new gather path
    against the implementation the resume previously implied.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_batch(data, block_size, batch_size, device):
    """
    Vectorized batch gather.

    If `data` already lives on `device`, the gather stays on that device
    (the AMD/Windows path: keep Tiny Shakespeare tokens on DirectML).
    Otherwise gather on the data's device and copy once with non_blocking
    when the destination is CUDA.
    """
    n = data.size(0) - block_size
    ix = torch.randint(n, (batch_size,), device=data.device)
    offsets = torch.arange(block_size, device=data.device)
    x = data[ix.unsqueeze(1) + offsets]
    y = data[ix.unsqueeze(1) + offsets + 1]
    if x.device != device:
        non_blocking = getattr(device, "type", None) == "cuda"
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
    return x, y


def _git_sha():
    """Best-effort HEAD SHA for the checkpoint; None if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_run_metadata(seed, tokenizer_type, argv=None):
    """
    Fields stored in ckpt.pt so a run can be reproduced from the file alone.

    Returns tokenizer type, seed, torch version, git SHA, and the full argv.
    """
    if argv is None:
        argv = sys.argv
    return {
        "tokenizer_type": tokenizer_type,
        "seed": seed,
        "torch_version": torch.__version__,
        "git_sha": _git_sha(),
        "argv": list(argv),
    }


def make_checkpoint(
    model,
    optimizer,
    config,
    iter_num,
    best_val_loss,
    scaler,
    seed,
    tokenizer_type,
    argv=None,
    vocab=None,
):
    """Build the dict written to out/ckpt.pt (weights + optimizer + run metadata)."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    checkpoint.update(collect_run_metadata(seed, tokenizer_type, argv=argv))
    # Char vocab must live in the ckpt: current data/vocab.pt is 66 (UNK) and
    # older Tiny Shakespeare runs used 65. sample.py prefers this list.
    if vocab is not None:
        checkpoint["vocab"] = list(vocab)
    return checkpoint


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
def estimate_loss(
    model, train_data, val_data, block_size, batch_size, eval_iters, device, use_amp=False
):
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


def _eval_tensors_for_checkpoint(checkpoint, tokenizer, data_dir, train_split=0.9):
    """
    Load train/val token tensors that match the checkpoint's vocab.

    Current CharTokenizer.train() adds UNK (vocab 66). Older Tiny Shakespeare
    checkpoints used 65 chars and a different id map — data/train.pt from a
    66-char prepare is not valid for those weights.
    """
    data_path = Path(data_dir)
    ckpt_vocab = checkpoint.get("vocab")
    disk_vocab = None
    if (data_path / "vocab.pt").exists():
        disk_meta = torch.load(data_path / "vocab.pt", map_location="cpu", weights_only=False)
        disk_vocab = disk_meta.get("vocab")
        same_size = disk_meta.get("vocab_size") == checkpoint["config"].vocab_size
        same_list = ckpt_vocab is None or disk_vocab == ckpt_vocab
        if (
            same_size
            and same_list
            and (data_path / "train.pt").exists()
            and (data_path / "val.pt").exists()
        ):
            return (
                torch.load(data_path / "train.pt", map_location="cpu", weights_only=False),
                torch.load(data_path / "val.pt", map_location="cpu", weights_only=False),
            )

    text_path = data_path / "input.txt"
    if not text_path.exists():
        raise FileNotFoundError(
            f"Cannot build eval tensors for this checkpoint: {text_path} is missing "
            "and data/train.pt does not match the checkpoint vocab. "
            "Run data.py or pass a data_dir that has input.txt."
        )
    text = text_path.read_text(encoding="utf-8")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_idx = int(train_split * len(tokens))
    if ckpt_vocab is not None and disk_vocab is not None and ckpt_vocab != disk_vocab:
        print(
            "Re-encoded input.txt with the checkpoint vocab "
            f"(ckpt {len(ckpt_vocab)} chars vs data/vocab.pt {len(disk_vocab)})."
        )
    return tokens[:split_idx], tokens[split_idx:]


def eval_checkpoint(
    checkpoint_path,
    data_dir="data",
    eval_device="cpu",
    eval_iters=20,
    batch_size=32,
    train_split=0.9,
):
    """
    Score a checkpoint on eval_device (prefer CPU).

    DirectML-reported val loss has been untrustworthy on this Windows + AMD
    stack — see results/directml.md. This path always builds a fresh model on
    eval_device from the saved state_dict.
    """
    from sample import load_tokenizer_for_checkpoint, model_from_checkpoint, read_checkpoint

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    info = detect_device(eval_device)
    report_device(info)
    if info.backend == "directml":
        print(
            "Warning: scoring on DirectML. Reported DML val is not ground truth; "
            "re-run with --eval_device cpu."
        )

    checkpoint = read_checkpoint(checkpoint_path)
    tokenizer = load_tokenizer_for_checkpoint(checkpoint, data_dir)
    model, config = model_from_checkpoint(checkpoint, info.device)
    train_data, val_data = _eval_tensors_for_checkpoint(
        checkpoint, tokenizer, data_dir, train_split=train_split
    )
    try:
        train_data = train_data.to(info.device)
        val_data = val_data.to(info.device)
    except Exception as exc:
        print(f"Warning: could not move eval tensors to {info.device}: {exc}")

    block_size = config.block_size
    losses = estimate_loss(
        model,
        train_data,
        val_data,
        block_size,
        min(batch_size, 32),
        eval_iters,
        info.device,
        use_amp=False,
    )
    ckpt_best = checkpoint.get("best_val_loss")
    print(
        f"CPU-eval path on {info.backend} ({info.name}): "
        f"train {losses['train']:.4f} | val {losses['val']:.4f}"
        + (f" | ckpt best_val_loss {ckpt_best:.4f}" if ckpt_best is not None else "")
    )
    if info.backend != "cpu":
        print("Treat this as a device-side number, not a CPU re-eval.")
    return losses


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
    device=None,  # None = auto-detect, or 'cuda' / 'directml' / 'mps' / 'cpu'
    seed=1337,
    out_dir='out',
    use_compile=True,
    use_amp=True,
    resume=False,
    use_tensorboard=False,
    use_sdpa=True,
    gradient_checkpointing=False,
    dropout=0.1,
    grad_clip=1.0,
    weight_decay=0.1,
    cpu_eval=False,
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
        device: Device to train on (None = auto-detect, 'cuda', 'directml', 'mps', or 'cpu')
        seed: RNG seed for reproducible runs
        out_dir: Directory to save checkpoints
        use_compile: Whether to use torch.compile (only on Linux with CUDA)
        use_amp: Whether to use automatic mixed precision training (CUDA only)
        resume: Whether to resume training from the latest checkpoint
        use_tensorboard: Whether to log metrics to TensorBoard
        use_sdpa: Whether to use PyTorch SDPA (F.scaled_dot_product_attention)
        gradient_checkpointing: Whether to use gradient checkpointing to save memory (trades compute for memory)
        dropout: Dropout probability for embeddings, attention, and MLP
        grad_clip: Max gradient norm; 0 disables clipping
        weight_decay: AdamW decay for matmul weights (norms/biases/embeddings use 0)
        cpu_eval: After training, re-score the saved checkpoint on CPU
    """
    set_seed(seed)
    info = detect_device(device)
    report_device(info)
    device_str = info.backend
    device_obj = info.device
    amp_ok = use_amp and supports_amp(info)
    compile_ok = use_compile and supports_compile(info)
    is_amd_gpu = (
        info.backend in ("rocm", "directml")
        or "radeon" in info.name.lower()
        or "amd" in info.name.lower()
    )

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
        tokenizer_type = data_dict['tokenizer'].type
        vocab_list = getattr(data_dict['tokenizer'], 'vocab', None)
    else:
        print("Loading preprocessed data...")
        train_data = torch.load(data_path / 'train.pt')
        val_data = torch.load(data_path / 'val.pt')
        vocab_metadata = torch.load(data_path / 'vocab.pt')
        vocab_size = vocab_metadata['vocab_size']
        tokenizer_type = vocab_metadata.get('tokenizer_type', 'char')
        vocab_list = vocab_metadata.get('vocab')

    # Keep tokens on the training device so get_batch() is a GPU gather,
    # not a Python loop + H2D copy. Tiny Shakespeare fits in a few MB.
    try:
        train_data = train_data.to(device_obj)
        val_data = val_data.to(device_obj)
        print(f"Token tensors on {train_data.device} (device-resident gather)")
    except Exception as exc:
        print(f"Warning: could not move token tensors to {device_obj}: {exc}")
        print("  Falling back to host-side gather + copy.")

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
        use_sdpa=use_sdpa,
        gradient_checkpointing=gradient_checkpointing,
        dropout=dropout,
    )
    model = GPT(config)
    model = model.to(device_obj)

    # Print model parameters
    n_params = model.get_num_params() / 1e6
    print(f"Model initialized with {n_params:.2f}M parameters")
    print(f"Tokenizer: {tokenizer_type}; dropout={dropout}")

    from model import SDPA_AVAILABLE

    if use_sdpa and SDPA_AVAILABLE:
        print("SDPA enabled (F.scaled_dot_product_attention; backend may dispatch FlashAttention)")
    elif use_sdpa and not SDPA_AVAILABLE:
        print("SDPA requested but not available (requires PyTorch 2.0+); using manual attention")
    else:
        print("SDPA disabled (using manual attention)")

    # Report gradient checkpointing status
    if gradient_checkpointing:
        print("Gradient checkpointing enabled (trading compute for ~50% memory reduction)")

    # Apply torch.compile only on Linux CUDA/ROCm
    if compile_ok:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled successfully")
    elif use_compile:
        print(f"Warning: torch.compile skipped on {platform.system()} / {device_str}.")

    optimizer = configure_optimizer(model, learning_rate, weight_decay=weight_decay)
    n_decay = sum(
        p.numel() for g in optimizer.param_groups if g["weight_decay"] > 0 for p in g["params"]
    )
    n_nodecay = sum(
        p.numel() for g in optimizer.param_groups if g["weight_decay"] == 0 for p in g["params"]
    )
    print(
        f"AdamW: {n_decay:,} decay / {n_nodecay:,} no-decay params; weight_decay={weight_decay}; "
        f"grad_clip={grad_clip}"
    )

    # Mixed precision: CUDA/ROCm only. DirectML is FP32.
    scaler = torch.cuda.amp.GradScaler() if amp_ok else None

    # Resume from checkpoint if requested
    start_iter = 0
    best_val_loss = float('inf')

    if resume:
        checkpoint_path = Path(out_dir) / 'ckpt.pt'
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            state_dict = checkpoint['model']
            stripped = strip_orig_mod_prefix(state_dict)
            if stripped is not state_dict:
                print("  Stripping '_orig_mod.' prefix from compiled checkpoint...")
                state_dict = stripped

            # Load into model (may need to unwrap if compiled)
            if hasattr(model, '_orig_mod'):
                model._orig_mod.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)

            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except (ValueError, KeyError) as exc:
                print(f"  Warning: could not load optimizer state ({exc}); using a fresh AdamW")

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
    print(f"\nStarting training on {device_str} ({info.name})...")
    print(f"Max iterations: {max_iters}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Evaluation interval: {eval_interval} iterations")
    if amp_ok:
        gpu_type = "AMD GPU (ROCm)" if is_amd_gpu else "NVIDIA GPU"
        print(f"Using mixed-precision training (FP16) on {gpu_type}")
    elif use_amp and not amp_ok:
        print("Mixed precision disabled (not supported on this backend; using FP32)")
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
        if iter_num == start_iter:
            param_dev = next(model.parameters()).device
            print(f"Proof: batch x.device={x.device}  model.param.device={param_dev}")
            if x.device != param_dev:
                print("Warning: batch and model are on different devices")

        # Forward and backward pass with mixed precision if enabled
        if amp_ok and scaler is not None:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = model(x, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Evaluate periodically
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data, block_size, batch_size, eval_iters, device_obj, amp_ok
            )
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"iter {iter_num:5d} | lr {current_lr:.2e} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
            )

            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('Loss/train', losses['train'], iter_num)
                writer.add_scalar('Loss/val', losses['val'], iter_num)
                writer.add_scalar('LearningRate', current_lr, iter_num)

            # Save checkpoint if validation loss improved
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = make_checkpoint(
                    model,
                    optimizer,
                    config,
                    iter_num,
                    best_val_loss,
                    scaler,
                    seed,
                    tokenizer_type,
                    vocab=vocab_list,
                )
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
    if cpu_eval:
        ckpt_path = Path(out_dir) / "ckpt.pt"
        if ckpt_path.exists():
            print("\nCPU re-eval of saved checkpoint (DirectML val is not ground truth):")
            eval_checkpoint(
                ckpt_path,
                data_dir=data_dir,
                eval_device="cpu",
                eval_iters=min(eval_iters, 20),
                batch_size=min(batch_size, 32),
            )
        else:
            print("CPU re-eval skipped (no checkpoint written).")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NanoGPT model')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/directml/mps/cpu, default: auto-detect)',
    )
    parser.add_argument('--seed', type=int, default=1337, help='RNG seed (default: 1337)')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing processed data (default: data)',
    )
    parser.add_argument('--block_size', type=int, default=256, help='Context length (default: 256)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument(
        '--n_layer', type=int, default=6, help='Number of transformer layers (default: 6)'
    )
    parser.add_argument(
        '--n_head', type=int, default=6, help='Number of attention heads (default: 6)'
    )
    parser.add_argument(
        '--n_embd', type=int, default=384, help='Embedding dimension (default: 384)'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=3e-4, help='Maximum learning rate (default: 3e-4)'
    )
    parser.add_argument(
        '--max_iters', type=int, default=5000, help='Maximum training iterations (default: 5000)'
    )
    parser.add_argument(
        '--warmup_iters', type=int, default=100, help='Warmup iterations (default: 100)'
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        default=0.1,
        help='Minimum learning rate as fraction of max_lr (default: 0.1)',
    )
    parser.add_argument(
        '--eval_interval', type=int, default=500, help='Evaluate every N iterations (default: 500)'
    )
    parser.add_argument(
        '--eval_iters',
        type=int,
        default=200,
        help='Number of iterations to average loss over (default: 200)',
    )
    parser.add_argument(
        '--out_dir', type=str, default='out', help='Output directory for checkpoints (default: out)'
    )
    parser.add_argument(
        '--no_compile', action='store_true', help='Disable torch.compile (Linux CUDA only)'
    )
    parser.add_argument(
        '--no_amp', action='store_true', help='Disable mixed-precision training (CUDA only)'
    )
    parser.add_argument(
        '--no_sdpa',
        action='store_true',
        help='Disable PyTorch SDPA (use manual QK^T + causal mask)',
    )
    parser.add_argument(
        '--resume', action='store_true', help='Resume training from the latest checkpoint'
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Enable TensorBoard logging (logs to out_dir/runs/)',
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='Dropout probability (default: 0.1)'
    )
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=1.0,
        help='Max gradient L2 norm (default: 1.0; 0 disables)',
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.1,
        help='AdamW weight decay for matmul weights (default: 0.1; norms/biases/embeddings use 0)',
    )
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable gradient checkpointing (trades compute for ~50%% memory reduction)',
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Score a checkpoint and exit (no training). Prefer --eval_device cpu.',
    )
    parser.add_argument(
        '--eval_device',
        type=str,
        default=None,
        help='Device for --eval_only (default: cpu). DirectML val is not ground truth.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint path for --eval_only (default: out_dir/ckpt.pt)',
    )
    parser.add_argument(
        '--cpu_eval',
        action='store_true',
        help='After training, re-eval the saved checkpoint on CPU',
    )

    args = parser.parse_args()

    if args.eval_only:
        ckpt = args.checkpoint or str(Path(args.out_dir) / 'ckpt.pt')
        eval_checkpoint(
            ckpt,
            data_dir=args.data_dir,
            eval_device=args.eval_device or 'cpu',
            eval_iters=args.eval_iters,
            batch_size=args.batch_size,
        )
        raise SystemExit(0)

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
        seed=args.seed,
        out_dir=args.out_dir,
        use_compile=not args.no_compile,
        use_amp=not args.no_amp,
        resume=args.resume,
        use_tensorboard=args.tensorboard,
        use_sdpa=not args.no_sdpa,
        gradient_checkpointing=args.gradient_checkpointing,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        cpu_eval=args.cpu_eval,
    )
