# NanoGPT

A clean, educational implementation of GPT (Generative Pre-trained Transformer) from scratch in PyTorch, following Andrej Karpathy's architecture. This project implements a character-level language model trained on the Tiny Shakespeare dataset.

## Project Structure

- **`data.py`** - Downloads and preprocesses the Tiny Shakespeare dataset, creates vocabulary and encoding/decoding functions
- **`model.py`** - Implements the GPT architecture with transformer blocks, attention, and MLP layers
- **`train.py`** - Training script with AdamW optimizer, learning rate scheduling, mixed-precision training, and checkpointing
- **`sample.py`** - Text generation script that loads a trained model and generates text from prompts

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd NanoGPT

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch requests

# For AMD GPU support (ROCm), see "AMD GPU Setup" section below
```

### GPU Support

The code automatically detects and uses available GPUs:

- **NVIDIA GPUs**: Uses CUDA (default PyTorch installation)
- **AMD GPUs**: Uses ROCm (requires PyTorch with ROCm support) - see "AMD GPU Setup" below
- **Apple Silicon**: Uses MPS (Metal Performance Shaders)
- **CPU**: Automatic fallback if no GPU is available

The training script will automatically detect and use your GPU if available. You can also force a specific device:
```bash
python train.py --device cuda    # Use GPU (NVIDIA or AMD with ROCm)
python train.py --device cpu     # Force CPU
```

### AMD GPU Setup

For AMD GPUs, you need to install PyTorch with ROCm support. Follow these steps:

#### 1. Check Your ROCm Version (if already installed)

If you have ROCm installed, check the version:

```bash
# Method 1: Using rocm-smi
rocm-smi --version

# Method 2: Check ROCm installation directory
cat /opt/rocm/.info/version-* 2>/dev/null

# Method 3: Check package manager (Debian/Ubuntu)
dpkg -l | grep rocm

# Method 4: Check package manager (Arch Linux)
pacman -Q | grep rocm
```

#### 2. Install PyTorch with ROCm Support

**Option A: If you have ROCm installed**, install PyTorch matching your ROCm version:

```bash
# For ROCm 5.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6

# For ROCm 5.7
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# For ROCm 6.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

**Note:** `torchaudio` may not be available for all ROCm versions. You can install just `torch` and `torchvision` if needed.

**Option B: If you don't have ROCm installed**, you can install PyTorch with ROCm support directly (PyTorch will work with compatible ROCm versions):

```bash
# Try the latest ROCm version (usually 5.7 or 6.0)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

Check the [PyTorch ROCm installation guide](https://pytorch.org/get-started/locally/) for the latest supported versions.

#### 3. Verify AMD GPU Detection

After installation, verify PyTorch can see your AMD GPU:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

You should see output like:
```
CUDA available: True
GPU: AMD Radeon RX 7900 XTX
```

#### 4. Run Training

The training script will automatically detect and use your AMD GPU:

```bash
python train.py
```

You should see output like:
```
ROCm available: Using AMD GPU (AMD Radeon RX 7900 XTX)
GPU Memory: 24.00 GB
Starting training on CUDA...
```

**Note:** AMD GPUs are accessed via the same `cuda` device string in PyTorch (ROCm is CUDA-compatible), so the code will show "CUDA" but it's actually using ROCm.

#### Troubleshooting

- **"CUDA not available"**: Make sure PyTorch with ROCm is installed correctly
- **"torchaudio not found"**: This is normal for some ROCm versions. Install only `torch` and `torchvision`
- **Performance issues**: Ensure ROCm drivers are properly installed and your GPU is supported
- **Check GPU compatibility**: See [ROCm documentation](https://rocm.docs.amd.com/) for supported GPUs

## Usage

### 1. Prepare the Dataset

#### Default: Tiny Shakespeare

Download and preprocess the Tiny Shakespeare dataset:

```bash
python data.py
```

This will:
- Download the dataset to `data/input.txt`
- Create vocabulary from unique characters
- Generate `data/train.pt`, `data/val.pt`, and `data/vocab.pt`
- Split data into 90% training and 10% validation sets

#### Custom Dataset

Train on your own text file:

```bash
# Prepare your custom dataset
python data.py --input_file my_corpus.txt --data_dir my_data

# Train on the custom data
python train.py --data_dir my_data
```

Options:
- `--input_file`: Path to your text file
- `--data_dir`: Directory to save processed data (default: `data`)
- `--train_split`: Fraction of data for training (default: `0.9`)

### 2. Train the Model

Train the GPT model:

```bash
python train.py
```

The training script supports various hyperparameters. You can modify them in `train.py` or extend the script to accept command-line arguments:

- `block_size`: Context length (default: 256)
- `batch_size`: Batch size (default: 64)
- `n_layer`: Number of transformer layers (default: 6)
- `n_head`: Number of attention heads (default: 6)
- `n_embd`: Embedding dimension (default: 384)
- `learning_rate`: Maximum learning rate (default: 3e-4)
- `max_iters`: Maximum training iterations (default: 5000)
- `warmup_iters`: Warmup iterations (default: 100)
- `min_lr`: Minimum learning rate as fraction of max_lr (default: 0.1)

The script will:
- Use cosine learning rate decay with linear warmup
- Apply mixed-precision training (FP16) on CUDA if available
- Use Flash Attention (PyTorch SDPA) for memory-efficient attention
- Compile the model with `torch.compile` on Linux for faster training
- Save checkpoints to `out/ckpt.pt` when validation loss improves

#### Flash Attention

Flash Attention is enabled by default on PyTorch 2.0+ for significant memory savings and faster training on long sequences. To disable:

```bash
python train.py --no_flash_attn
```

#### Resume Training

If training is interrupted, you can resume from the last checkpoint:

```bash
python train.py --resume
```

This will load the model, optimizer, and scaler state from the checkpoint and continue training.

#### TensorBoard Logging

Enable TensorBoard to visualize training metrics:

```bash
python train.py --tensorboard
```

View the logs with:
```bash
tensorboard --logdir out/runs
```

### 3. Generate Text

Generate text from a trained model:

```bash
python sample.py
```

Options:
```bash
python sample.py \
    --checkpoint out/ckpt.pt \
    --prompt "\n" \
    --num_tokens 500 \
    --temperature 1.0 \
    --top_k 40
```

Arguments:
- `--checkpoint`: Path to model checkpoint (default: `out/ckpt.pt`)
- `--prompt`: Starting text prompt (default: `\n`)
- `--num_tokens`: Number of tokens to generate (default: 500)
- `--temperature`: Sampling temperature - higher = more random (default: 1.0)
- `--top_k`: Top-k sampling - only sample from top-k most likely tokens (default: None)
- `--top_p`: Top-p (nucleus) sampling - keep tokens with cumulative probability >= p (default: None)

#### Sampling Strategies

**Temperature** controls randomness:
- `temperature=0.8` — More focused, coherent text
- `temperature=1.0` — Default, balanced
- `temperature=1.2` — More creative, varied

**Top-k** limits vocabulary to k most likely tokens:
```bash
python sample.py --top_k 40 --temperature 0.8
```

**Top-p (nucleus)** keeps tokens until cumulative probability reaches p:
```bash
python sample.py --top_p 0.9 --temperature 0.8
```

Top-p often produces better quality than top-k as it adapts to the confidence distribution.

## Architecture Overview

The model follows the standard GPT architecture:

1. **Token and Position Embeddings**: Convert token indices to dense vectors and add positional information
2. **Transformer Blocks**: Stack of decoder blocks, each containing:
   - **Causal Self-Attention**: Multi-head attention with causal masking to prevent looking at future tokens
   - **MLP**: Two-layer feedforward network with GELU activation
   - **Residual Connections**: Around both attention and MLP layers
   - **Layer Normalization**: Applied before attention and MLP (pre-norm architecture)
3. **Language Modeling Head**: Projects hidden states to vocabulary logits
4. **Weight Tying**: Shares weights between token embedding and output projection

## Key Features

- **Clean, Modular Code**: Each component is well-separated and educational
- **Efficient Training**: Mixed-precision training and torch.compile support for faster training
- **Flash Attention**: Memory-efficient attention via PyTorch SDPA (2-4x memory reduction)
- **Custom Datasets**: Train on any text file, not just Shakespeare
- **Learning Rate Scheduling**: Cosine decay with linear warmup for stable training
- **Checkpointing**: Saves best model based on validation loss; supports resuming training
- **TensorBoard Integration**: Visualize training metrics in real-time
- **Text Generation**: Autoregressive generation with temperature, top-k, and top-p sampling

## Model Size

The default configuration creates a model with approximately:
- **6 transformer layers**
- **6 attention heads per layer**
- **384 embedding dimensions**
- **~10-15M parameters** (depending on vocabulary size)

This is a small model suitable for educational purposes and can train on a single GPU in reasonable time.

## Future Improvements

- [ ] Add support for BPE (Byte Pair Encoding) tokenization
- [ ] Implement gradient checkpointing for larger models
- [ ] Add support for multi-GPU training (DDP)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT) - Inspiration for this implementation

## License

This project is for educational purposes.
