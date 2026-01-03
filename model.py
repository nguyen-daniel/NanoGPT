"""
GPT model architecture for NanoGPT.
Implements a decoder-only transformer following the GPT architecture.

© 2026
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


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
    """
    block_size: int = 1024  # context length
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12  # number of transformer blocks
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.
    
    Implements scaled dot-product attention with a causal mask to prevent
    the model from looking at future tokens during autoregressive generation.
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
        
        # Key, Query, Value projections for all heads
        # We use a single linear layer and split it into heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Dropout for regularization
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        
        # Causal mask: prevents attention to future positions
        # Register as buffer so it's not a trainable parameter
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                      .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        """
        Forward pass of causal self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
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
        
        # Scaled dot-product attention
        # Compute attention scores: Q @ K^T / sqrt(head_size)
        # Shape: (batch_size, n_head, seq_len, seq_len)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        
        # Apply causal mask: set future positions to -inf
        # The mask has 1s for positions we can attend to, 0s for positions we cannot
        att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        
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
        self.dropout = nn.Dropout(0.1)
    
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
    
    def forward(self, x):
        """
        Forward pass of the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
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
        self.dropout = nn.Dropout(0.1)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Language modeling head: projects hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and output projection
        # This is a common technique that reduces parameters and can improve performance
        self.token_embedding.weight = self.lm_head.weight
        
    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT model.
        
        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            targets: Target token indices for training, shape (batch_size, seq_len)
                    If None, model is in inference mode.
        
        Returns:
            If targets is None: logits of shape (batch_size, seq_len, vocab_size)
            If targets is provided: (logits, loss) tuple
        """
        batch_size, seq_len = idx.shape
        
        # Get token embeddings
        # Shape: (batch_size, seq_len, n_embd)
        token_embeddings = self.token_embedding(idx)
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len,)
        position_indices = torch.arange(seq_len, device=idx.device)
        
        # Get position embeddings
        # Shape: (seq_len, n_embd) -> broadcasted to (batch_size, seq_len, n_embd)
        position_embeddings = self.position_embedding(position_indices)
        
        # Combine token and position embeddings
        # Shape: (batch_size, seq_len, n_embd)
        x = token_embeddings + position_embeddings
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
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
        
        return logits if loss is None else (logits, loss)
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens given a starting sequence.
        
        Args:
            idx: Starting token indices of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = no change, >1.0 = more random, <1.0 = more focused)
            top_k: If specified, only sample from top-k most likely tokens (None = no filtering)
        
        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx[:, -self.config.block_size:]
            
            # Forward pass to get logits
            logits = self(idx_cond, targets=None)
            
            # Focus only on the last time step
            # Shape: (batch_size, vocab_size)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
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

