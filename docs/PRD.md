# NanoGPT — Product Requirements Document

**Version:** 1.0  
**Date:** January 6, 2026  
**Status:** Living Document

---

## 1. Executive Summary

NanoGPT is an educational, minimalist implementation of the GPT (Generative Pre-trained Transformer) architecture. Built from scratch in PyTorch, it serves as both a learning resource and a functional character-level language model. The project prioritizes code clarity, modularity, and pedagogical value over production-scale performance.

---

## 2. Vision & Goals

### 2.1 Vision Statement

To provide the clearest possible implementation of a GPT-style transformer, enabling developers and researchers to understand the core mechanics of modern language models through hands-on experimentation.

### 2.2 Primary Goals

| Goal | Description | Success Metric |
|------|-------------|----------------|
| **Educational Clarity** | Every component is readable and well-documented | New developers can understand the architecture in < 1 hour |
| **Functional Completeness** | Full training and inference pipeline | Model generates coherent Shakespeare-style text |
| **Hardware Flexibility** | Support NVIDIA, AMD, Apple Silicon, and CPU | Training works on all major platforms |
| **Minimal Dependencies** | Only essential libraries required | `torch` and `requests` are the only requirements |

### 2.3 Non-Goals

- Production-scale training (billions of parameters)
- Distributed multi-node training
- Enterprise deployment features
- Fine-tuning on arbitrary datasets (initial scope)

---

## 3. Target Users

### 3.1 Primary Personas

**The Curious Developer**
- Background: Software engineer learning ML/AI
- Need: Understand how transformers work at the code level
- Pain Point: Research papers and production codebases are too complex

**The ML Student**
- Background: CS student taking ML courses
- Need: Practical implementation to complement theory
- Pain Point: Textbook examples don't show real training loops

**The Researcher**
- Background: Academic or industry researcher
- Need: Clean baseline for experimentation
- Pain Point: Existing implementations have too many abstractions

---

## 4. Technical Architecture

### 4.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         NanoGPT System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  data.py │───▶│ model.py │───▶│ train.py │───▶│sample.py │  │
│  │          │    │          │    │          │    │          │  │
│  │ Download │    │   GPT    │    │ Training │    │   Text   │  │
│  │ Preproc. │    │   Arch   │    │   Loop   │    │   Gen    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │              │               │               │         │
│       ▼              ▼               ▼               ▼         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    data/ directory                        │  │
│  │  input.txt  │  train.pt  │  val.pt  │  vocab.pt          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    out/ directory                         │  │
│  │                       ckpt.pt                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Core Components

#### 4.2.1 Data Pipeline (`data.py`)

| Component | Responsibility |
|-----------|----------------|
| `download_shakespeare()` | Fetch Tiny Shakespeare dataset from GitHub |
| `get_vocabulary()` | Extract unique characters for tokenization |
| `create_encoder_decoder()` | Build char↔int mapping functions |
| `prepare_data()` | Orchestrate full pipeline, save train/val splits |

**Data Flow:**
```
URL → input.txt → vocabulary extraction → encode → train.pt + val.pt
```

#### 4.2.2 Model Architecture (`model.py`)

| Class | Purpose |
|-------|---------|
| `GPTConfig` | Dataclass holding all hyperparameters |
| `CausalSelfAttention` | Multi-head attention with causal masking |
| `MLP` | Feedforward network (GELU activation) |
| `Block` | Transformer block (attention + MLP + residuals) |
| `GPT` | Full model: embeddings → blocks → lm_head |

**Architecture Details:**
- Pre-LayerNorm (more stable than post-norm)
- Weight tying between token embedding and output projection
- Learnable positional embeddings (not sinusoidal)

#### 4.2.3 Training Loop (`train.py`)

| Feature | Implementation |
|---------|----------------|
| Optimizer | AdamW with weight decay |
| LR Schedule | Linear warmup → cosine decay |
| Mixed Precision | FP16 via `torch.amp` (CUDA only) |
| Compilation | `torch.compile()` (Linux + CUDA only) |
| Checkpointing | Save best model by validation loss |

#### 4.2.4 Inference (`sample.py`)

| Feature | Implementation |
|---------|----------------|
| Sampling | Multinomial with temperature |
| Top-k Filtering | Optional nucleus-like sampling |
| Device Support | Auto-detect or manual override |

### 4.3 Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 256 | Context window length |
| `batch_size` | 64 | Training batch size |
| `n_layer` | 6 | Transformer layers |
| `n_head` | 6 | Attention heads per layer |
| `n_embd` | 384 | Embedding dimension |
| `learning_rate` | 3e-4 | Peak learning rate |
| `max_iters` | 5000 | Training iterations |
| `warmup_iters` | 100 | LR warmup iterations |
| `dropout` | 0.1 | Dropout rate |

**Resulting Model Size:** ~10-15M parameters

---

## 5. Functional Requirements

### 5.1 Must Have (P0)

| ID | Requirement | Status |
|----|-------------|--------|
| FR-01 | Download and preprocess Tiny Shakespeare dataset | ✅ Complete |
| FR-02 | Implement GPT architecture with configurable size | ✅ Complete |
| FR-03 | Train model with loss tracking | ✅ Complete |
| FR-04 | Save/load model checkpoints | ✅ Complete |
| FR-05 | Generate text from trained model | ✅ Complete |
| FR-06 | Support CUDA/ROCm/MPS/CPU devices | ✅ Complete |

### 5.2 Should Have (P1)

| ID | Requirement | Status |
|----|-------------|--------|
| FR-07 | Cosine LR decay with warmup | ✅ Complete |
| FR-08 | Mixed-precision training | ✅ Complete |
| FR-09 | `torch.compile()` optimization | ✅ Complete |
| FR-10 | Command-line argument support | ✅ Complete |
| FR-11 | Temperature-based sampling | ✅ Complete |
| FR-12 | Top-k sampling | ✅ Complete |

### 5.3 Could Have (P2) — Future Roadmap

| ID | Requirement | Status |
|----|-------------|--------|
| FR-13 | BPE tokenization (byte-pair encoding) | 🔲 Planned |
| FR-14 | Gradient checkpointing for larger models | 🔲 Planned |
| FR-15 | Multi-GPU training (DDP) | 🔲 Planned |
| FR-16 | Top-p (nucleus) sampling | 🔲 Planned |
| FR-17 | TensorBoard integration | 🔲 Planned |
| FR-18 | Resume training from checkpoint | 🔲 Planned |
| FR-19 | Custom dataset support | 🔲 Planned |
| FR-20 | Flash Attention integration | 🔲 Planned |

---

## 6. Non-Functional Requirements

### 6.1 Performance

| Metric | Target | Current |
|--------|--------|---------|
| Training throughput | > 100 iter/s on modern GPU | ✅ Achieved |
| Memory efficiency | Train on 8GB VRAM | ✅ Achieved |
| Inference latency | < 10ms per token on GPU | ✅ Achieved |

### 6.2 Compatibility

| Platform | Support Level |
|----------|---------------|
| Linux (NVIDIA CUDA) | Full (compile + AMP) |
| Linux (AMD ROCm) | Full (compile + AMP) |
| macOS (Apple Silicon MPS) | Partial (no compile/AMP) |
| Windows (NVIDIA CUDA) | Partial (no compile) |
| CPU (any OS) | Baseline (slower) |

### 6.3 Code Quality

- **Documentation:** Every function has docstrings
- **Type Hints:** Used in configuration classes
- **Modularity:** Single responsibility per file
- **Naming:** Descriptive, following PyTorch conventions

---

## 7. User Workflows

### 7.1 First-Time Setup

```bash
# Clone and setup
git clone <repo-url>
cd NanoGPT
python -m venv venv
source venv/bin/activate
pip install torch requests

# Prepare data
python data.py

# Train model
python train.py

# Generate text
python sample.py --prompt "ROMEO:"
```

### 7.2 Experimentation Workflow

```bash
# Train with custom hyperparameters
python train.py \
    --n_layer 8 \
    --n_head 8 \
    --n_embd 512 \
    --max_iters 10000 \
    --learning_rate 1e-4

# Generate with different sampling
python sample.py \
    --temperature 0.8 \
    --top_k 40 \
    --num_tokens 1000
```

---

## 8. Success Criteria

### 8.1 Quantitative Metrics

| Metric | Target |
|--------|--------|
| Validation loss | < 1.5 after 5000 iterations |
| Generated text coherence | Recognizable Shakespeare-like structure |
| Setup time (new user) | < 10 minutes |

### 8.2 Qualitative Metrics

- Generated text maintains character dialogue format
- Model learns iambic-like rhythm in longer generations
- Users report understanding transformer mechanics

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| PyTorch API changes | Medium | Pin versions in requirements.txt |
| ROCm compatibility issues | Medium | Document tested versions |
| Memory OOM on small GPUs | Low | Provide smaller config presets |
| Dataset URL becomes unavailable | Low | Include backup or bundled option |

---

## 10. Appendix

### 10.1 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer paper
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — Inspiration

### 10.2 Glossary

| Term | Definition |
|------|------------|
| **Block Size** | Maximum context length the model can process |
| **Causal Mask** | Prevents attention to future tokens |
| **Token Embedding** | Learned vector representation of each vocabulary item |
| **Position Embedding** | Learned encoding of sequence position |
| **Weight Tying** | Sharing weights between input embeddings and output projection |

---

*This PRD is a living document. Update as the project evolves.*

