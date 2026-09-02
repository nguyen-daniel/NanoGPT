"""
GPT model architecture for NanoGPT.
Implements a decoder-only transformer following the GPT architecture.

© 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from dataclasses import dataclass

# PyTorch 2.0+ SDPA; the backend may dispatch FlashAttention when it provides one.
SDPA_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')

# torch.compile() wraps modules and prefixes parameter names with this.
ORIG_MOD_PREFIX = "_orig_mod."


def strip_orig_mod_prefix(state_dict):
    """
    Remove torch.compile() `_orig_mod.` prefixes from a checkpoint state dict.

    Returns the original mapping when no keys need stripping.
    """
    if not any(key.startswith(ORIG_MOD_PREFIX) for key in state_dict):
        return state_dict
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith(ORIG_MOD_PREFIX):
            stripped[key[len(ORIG_MOD_PREFIX) :]] = value
        else:
            stripped[key] = value
    return stripped


def filter_logits(logits, top_k=None, top_p=None):
    """
    Filter next-token logits before softmax.

    Top-k and top-p may be combined. When both are set they apply in order:
    top-k first (keep the k largest logits), then top-p / nucleus on what remains.
    Typical top-p values are 0.9–0.95. The nucleus always keeps at least one token.
    """
    if top_k is None and top_p is None:
        return logits

    filtered = logits.clone()

    if top_k is not None:
        k = min(int(top_k), filtered.size(-1))
        kth = torch.topk(filtered, k, dim=-1).values[..., -1:]
        filtered = filtered.masked_fill(filtered < kth, float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered = filtered.masked_fill(indices_to_remove, float("-inf"))

    return filtered


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model hyperparameters.

    Attributes:
        block_size: Maximum context length (sequence length)
        vocab_size: Size of the vocabulary (number of unique tokens)
        n_layer: Number of transformer decoder layers
        n_head: Number of attention heads in multi-head attention
        n_embd: Embedding dimension (size of token embeddings)
        use_sdpa: Whether to use PyTorch SDPA (F.scaled_dot_product_attention)
        gradient_checkpointing: Whether to use gradient checkpointing to save memory
        dropout: Dropout probability for embeddings, attention, and MLP
    """

    block_size: int = 1024  # context length
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12  # number of transformer blocks
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension
    use_sdpa: bool = (
        True  # F.scaled_dot_product_attention when available (backend may use FlashAttention)
    )
    gradient_checkpointing: bool = False  # trade compute for memory during training
    dropout: float = 0.1  # dropout probability (0.0 disables)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.

    Implements scaled dot-product attention with a causal mask to prevent
    the model from looking at future tokens during autoregressive generation.

    When use_sdpa is set and SDPA is available (PyTorch 2.0+), uses
    F.scaled_dot_product_attention. The backend may dispatch FlashAttention
    when it provides that kernel; this is not a custom FlashAttention impl.
    """

    def __init__(self, config: GPTConfig):
        """
        Initialize the causal self-attention layer.

        Args:
            config: GPTConfig instance containing model hyperparameters
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.use_sdpa = config.use_sdpa and SDPA_AVAILABLE

        # Key, Query, Value projections for all heads
        # We use a single linear layer and split it into heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Dropout for regularization (only used in manual attention path)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Causal mask: prevents attention to future positions
        # Only needed for manual attention (SDPA uses is_causal=True)
        if not self.use_sdpa:
            self.register_buffer(
                'bias',
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x, past_kv=None, use_cache=False):
        """
        Forward pass of causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            past_kv: Optional (past_k, past_v), each (batch, n_head, T_past, head_size)
            use_cache: If True, also return present (k, v) for the next decode step

        Returns:
            y of shape (batch_size, seq_len, n_embd), or (y, present_kv) if use_cache
        """
        batch_size, seq_len, n_embd = x.shape

        # Compute Q, K, V for all heads
        # Shape: (batch_size, seq_len, 3 * n_embd)
        qkv = self.c_attn(x)

        # Split into Q, K, V
        # Shape: each is (batch_size, seq_len, n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to separate heads
        # Shape: (batch_size, seq_len, n_head, head_size)
        q = q.view(batch_size, seq_len, self.n_head, self.head_size)
        k = k.view(batch_size, seq_len, self.n_head, self.head_size)
        v = v.view(batch_size, seq_len, self.n_head, self.head_size)

        # Transpose to get (batch_size, n_head, seq_len, head_size)
        # This allows efficient batch matrix multiplication
        q = q.transpose(1, 2)  # (batch_size, n_head, seq_len, head_size)
        k = k.transpose(1, 2)  # (batch_size, n_head, seq_len, head_size)
        v = v.transpose(1, 2)  # (batch_size, n_head, seq_len, head_size)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        t_q = q.size(2)
        t_k = k.size(2)

        if self.use_sdpa:
            # PyTorch SDPA; backend may dispatch FlashAttention when available.
            # First step (no cache): full prompt, is_causal=True.
            # Later steps: one new token over cached K/V + new; is_causal=False.
            if past_kv is None:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
            elif t_q == 1:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
            else:
                # Prefix cache + several new tokens: attend to all past, causal among new.
                # Additive -inf mask (not a bool mask) so the meaning is unambiguous.
                past_len = t_k - t_q
                q_pos = torch.arange(t_q, device=q.device)
                k_pos = torch.arange(t_k, device=q.device)
                disallowed = k_pos.unsqueeze(0) > (past_len + q_pos).unsqueeze(1)
                attn_mask = torch.zeros(t_q, t_k, device=q.device, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(disallowed, float("-inf"))
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
        else:
            # Manual attention computation (fallback for older PyTorch versions)
            # Scaled dot-product attention
            # Compute attention scores: Q @ K^T / sqrt(head_size)
            # Shape: (batch_size, n_head, t_q, t_k)
            att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size**0.5))

            if past_kv is None:
                att = att.masked_fill(self.bias[:, :, :t_q, :t_k] == 0, float("-inf"))
            elif t_q > 1:
                past_len = t_k - t_q
                q_pos = torch.arange(t_q, device=att.device)
                k_pos = torch.arange(t_k, device=att.device)
                allowed = k_pos.unsqueeze(0) <= (past_len + q_pos).unsqueeze(1)
                att = att.masked_fill(~allowed, float("-inf"))
            # t_q == 1 and past exists: attend to all cached K/V + new token (no mask)

            # Softmax to get attention weights
            att = torch.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Apply attention to values
            # Shape: (batch_size, n_head, seq_len, head_size)
            y = att @ v

        # Concatenate heads
        # Shape: (batch_size, seq_len, n_head, head_size)
        y = y.transpose(1, 2).contiguous()
        # Shape: (batch_size, seq_len, n_embd)
        y = y.view(batch_size, seq_len, n_embd)

        # Output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        if use_cache:
            return y, (k, v)
        return y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (feedforward network) for transformer blocks.

    Consists of two linear layers with GELU activation in between.
    """

    def __init__(self, config: GPTConfig):
        """
        Initialize the MLP.

        Args:
            config: GPTConfig instance containing model hyperparameters
        """
        super().__init__()
        # Expand to 4x the embedding dimension (standard in transformers)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block combining attention and MLP with residual connections.

    Architecture:
        x -> LayerNorm -> Attention -> + (residual) -> LayerNorm -> MLP -> + (residual) -> output
    """

    def __init__(self, config: GPTConfig):
        """
        Initialize the transformer block.

        Args:
            config: GPTConfig instance containing model hyperparameters
        """
        super().__init__()
        # Pre-attention layer normalization
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # Self-attention
        self.attn = CausalSelfAttention(config)
        # Pre-MLP layer normalization
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # Feedforward MLP
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False):
        """
        Forward pass of the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            past_kv: Optional cached (k, v) for this layer's attention
            use_cache: If True, also return this layer's present (k, v)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd),
            or (output, present_kv) if use_cache
        """
        if use_cache:
            attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, present_kv

        # Self-attention with residual connection
        x = x + self.attn(self.ln_1(x))

        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model.

    A decoder-only transformer architecture for autoregressive language modeling.
    """

    def __init__(self, config: GPTConfig):
        """
        Initialize the GPT model.

        Args:
            config: GPTConfig instance containing model hyperparameters
        """
        super().__init__()
        self.config = config

        # Token embedding layer: maps token indices to dense vectors
        # Input: (batch_size, seq_len) of token indices
        # Output: (batch_size, seq_len, n_embd)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embedding layer: encodes position information
        # Input: (batch_size, seq_len) of position indices
        # Output: (batch_size, seq_len, n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language modeling head: projects hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output projection
        # This is a common technique that reduces parameters and can improve performance
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False):
        """
        Forward pass of the GPT model.

        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            targets: Target token indices for training, shape (batch_size, seq_len)
                    If None, model is in inference mode.
            past_kvs: Optional list of per-layer (k, v) caches from a previous step
            use_cache: If True, also return present K/V for each layer

        Returns:
            If targets is None: logits of shape (batch_size, seq_len, vocab_size)
            If targets is provided: (logits, loss) tuple
            If use_cache: appends a list of present (k, v) per layer
        """
        batch_size, seq_len = idx.shape

        past_len = 0
        if past_kvs is not None:
            past_len = past_kvs[0][0].size(2)
        if past_len + seq_len > self.config.block_size:
            raise ValueError(
                f"past_len ({past_len}) + seq_len ({seq_len}) exceeds "
                f"block_size ({self.config.block_size})"
            )

        # Get token embeddings
        # Shape: (batch_size, seq_len, n_embd)
        token_embeddings = self.token_embedding(idx)

        # Positions continue after the cached prefix (0..T-1 on the first step)
        position_indices = torch.arange(past_len, past_len + seq_len, device=idx.device)

        # Get position embeddings
        # Shape: (seq_len, n_embd) -> broadcasted to (batch_size, seq_len, n_embd)
        position_embeddings = self.position_embedding(position_indices)

        # Combine token and position embeddings
        # Shape: (batch_size, seq_len, n_embd)
        x = token_embeddings + position_embeddings

        # Apply dropout
        x = self.dropout(x)

        # Pass through transformer blocks
        # Gradient checkpointing trades compute for memory by recomputing
        # activations during the backward pass instead of storing them
        presents = [] if use_cache else None
        if self.config.gradient_checkpointing and self.training and not use_cache:
            for block in self.blocks:
                # use_reentrant=False is recommended for new code (PyTorch 2.0+)
                # preserve_rng_state=True ensures dropout is consistent
                x = gradient_checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
        else:
            for i, block in enumerate(self.blocks):
                past = past_kvs[i] if past_kvs is not None else None
                if use_cache:
                    x, present = block(x, past_kv=past, use_cache=True)
                    presents.append(present)
                else:
                    x = block(x)

        # Apply final layer normalization
        x = self.ln_f(x)

        # Generate logits through language modeling head
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape logits and targets for cross-entropy loss
            # Cross-entropy expects (N, C) for logits and (N,) for targets
            # where N is the number of samples and C is the number of classes
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            targets_flat = targets.view(batch_size * seq_len)

            # Calculate cross-entropy loss
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        if use_cache:
            return (logits, presents) if loss is None else (logits, loss, presents)
        return logits if loss is None else (logits, loss)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate new tokens given a starting sequence.

        First step forwards the full (cropped) prompt and stores K/V.
        Later steps forward one new token and attend over the cached K/V.

        Args:
            idx: Starting token indices of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = no change, >1.0 = more random, <1.0 = more focused)
            top_k: If specified, only sample from the top-k most likely tokens (None = no top-k)
            top_p: If specified, nucleus sampling: keep the smallest set of tokens whose
                   cumulative probability is >= top_p (None = no nucleus filter).
                   Typical values: 0.9-0.95. May be combined with top_k; top-k is applied first.

        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()

        def _sample_next(step_logits):
            # Focus only on the last time step
            # Shape: (batch_size, vocab_size)
            step_logits = step_logits[:, -1, :] / temperature
            step_logits = filter_logits(step_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(step_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        # First step: full prompt (cropped), populate the KV cache
        idx_cond = idx[:, -self.config.block_size :]
        logits, past_kvs = self(idx_cond, targets=None, use_cache=True)

        for _ in range(max_new_tokens):
            idx_next = _sample_next(logits)
            idx = torch.cat((idx, idx_next), dim=1)

            cache_len = past_kvs[0][0].size(2)
            if cache_len >= self.config.block_size:
                # Sliding window: recompute last block_size tokens (positions 0..T-1)
                idx_cond = idx[:, -self.config.block_size :]
                logits, past_kvs = self(idx_cond, targets=None, use_cache=True)
            else:
                # One new token; attend over cached K/V + this token (is_causal=False)
                logits, past_kvs = self(idx_next, targets=None, past_kvs=past_kvs, use_cache=True)

        self.train()
        return idx

    def get_num_params(self, non_embedding=False):
        """
        Calculate the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters from count

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract embedding parameters
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        return n_params
