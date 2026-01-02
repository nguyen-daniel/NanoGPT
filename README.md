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

# Install dependencies
pip install torch requests
```

## Usage

### 1. Prepare the Dataset

First, download and preprocess the Tiny Shakespeare dataset:

```bash
python data.py
```

This will:
- Download the dataset to `data/input.txt`
- Create vocabulary from unique characters
- Generate `data/train.pt`, `data/val.pt`, and `data/vocab.pt`
- Split data into 90% training and 10% validation sets

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
- Compile the model with `torch.compile` on Linux for faster training
- Save checkpoints to `out/ckpt.pt` when validation loss improves

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
- **Learning Rate Scheduling**: Cosine decay with linear warmup for stable training
- **Checkpointing**: Saves best model based on validation loss
- **Text Generation**: Autoregressive generation with temperature and top-k sampling

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
- [ ] Add support for multi-GPU training
- [ ] Implement more sophisticated sampling strategies (nucleus/top-p sampling)
- [ ] Add training metrics visualization (TensorBoard integration)
- [ ] Support for resuming training from checkpoints

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT) - Inspiration for this implementation

## License

This project is for educational purposes.
