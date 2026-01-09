"""
Tokenizer implementations for NanoGPT.
Supports character-level tokenization (default) and BPE (Byte Pair Encoding).

Usage:
    # Character tokenizer (default)
    tokenizer = CharTokenizer.train(text)
    
    # BPE tokenizer
    tokenizer = BPETokenizer.train(text, vocab_size=1000)
    
    # Encode/decode
    tokens = tokenizer.encode("Hello world")
    text = tokenizer.decode(tokens)
    
    # Save/load
    tokenizer.save("vocab.pt")
    tokenizer = load_tokenizer("vocab.pt")

© 2026
"""

import torch
from pathlib import Path
from abc import ABC, abstractmethod
from collections import Counter


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @property
    @abstractmethod
    def type(self) -> str:
        """Return tokenizer type identifier."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert text to list of token IDs."""
        pass
    
    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Convert list of token IDs to text."""
        pass
    
    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save tokenizer to file."""
        pass
    
    @classmethod
    @abstractmethod
    def from_file(cls, path: str | Path) -> 'BaseTokenizer':
        """Load tokenizer from file."""
        pass


class CharTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.
    
    Maps each unique character in the corpus to an integer ID.
    Simple and works well for small vocabularies and educational purposes.
    """
    
    def __init__(self, vocab: list[str]):
        """
        Initialize character tokenizer with vocabulary.
        
        Args:
            vocab: List of unique characters (sorted)
        """
        self._vocab = vocab
        self._char_to_int = {ch: i for i, ch in enumerate(vocab)}
        self._int_to_char = {i: ch for i, ch in enumerate(vocab)}
    
    @property
    def type(self) -> str:
        return 'char'
    
    @property
    def vocab_size(self) -> int:
        return len(self._vocab)
    
    @property
    def vocab(self) -> list[str]:
        """Return vocabulary as list of characters."""
        return self._vocab
    
    def encode(self, text: str) -> list[int]:
        """Convert text to list of character IDs."""
        return [self._char_to_int[ch] for ch in text]
    
    def decode(self, tokens: list[int]) -> str:
        """Convert list of character IDs to text."""
        return ''.join([self._int_to_char[i] for i in tokens])
    
    def save(self, path: str | Path) -> None:
        """Save tokenizer to file."""
        metadata = {
            'tokenizer_type': 'char',
            'vocab': self._vocab,
            'vocab_size': self.vocab_size,
            'char_to_int': self._char_to_int,
            'int_to_char': self._int_to_char
        }
        torch.save(metadata, path)
    
    @classmethod
    def from_file(cls, path: str | Path) -> 'CharTokenizer':
        """Load character tokenizer from file."""
        metadata = torch.load(path, weights_only=False)
        return cls(vocab=metadata['vocab'])
    
    @classmethod
    def from_metadata(cls, metadata: dict) -> 'CharTokenizer':
        """Create tokenizer from metadata dictionary."""
        return cls(vocab=metadata['vocab'])
    
    @classmethod
    def train(cls, text: str) -> 'CharTokenizer':
        """
        Train character tokenizer on text.
        
        Args:
            text: Training text
            
        Returns:
            Trained CharTokenizer
        """
        vocab = sorted(list(set(text)))
        return cls(vocab=vocab)


class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer.
    
    Starts with byte-level tokens and iteratively merges the most frequent
    adjacent pairs until reaching the target vocabulary size. This creates
    a subword vocabulary that balances between character and word-level.
    
    Benefits over character tokenization:
    - Shorter sequences (better for long context)
    - Better handling of rare words
    - More semantic token boundaries
    """
    
    def __init__(self, merges: list[tuple[int, int]], vocab: dict[int, bytes]):
        """
        Initialize BPE tokenizer with trained merges and vocabulary.
        
        Args:
            merges: List of (token_a, token_b) merge pairs in order they were learned
            vocab: Mapping from token ID to bytes representation
        """
        self._merges = merges
        self._vocab = vocab  # id -> bytes
        self._vocab_inverse = {v: k for k, v in vocab.items()}  # bytes -> id
        
        # Build merge lookup for fast encoding
        self._merge_ranks = {pair: i for i, pair in enumerate(merges)}
    
    @property
    def type(self) -> str:
        return 'bpe'
    
    @property
    def vocab_size(self) -> int:
        return len(self._vocab)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text to list of BPE token IDs.
        
        Algorithm:
        1. Convert text to bytes
        2. Start with byte-level tokens
        3. Iteratively apply merges in learned order
        """
        # Start with byte-level tokens (0-255)
        tokens = list(text.encode('utf-8'))
        
        # Apply merges iteratively
        while len(tokens) >= 2:
            # Find the pair with lowest merge rank (learned earliest)
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            
            # Find best pair to merge (lowest rank = learned first)
            best_pair = None
            best_rank = float('inf')
            for pair in pairs:
                rank = self._merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            
            if best_pair is None:
                break  # No more merges applicable
            
            # Apply the merge
            new_token = 256 + best_rank  # Token ID for merged pair
            tokens = self._merge_tokens(tokens, best_pair, new_token)
        
        return tokens
    
    def _merge_tokens(self, tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
        """Replace all occurrences of pair with new_token."""
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result
    
    def decode(self, tokens: list[int]) -> str:
        """
        Decode list of BPE token IDs to text.
        
        Converts each token to its bytes representation and decodes as UTF-8.
        """
        byte_list = b''.join(self._vocab[t] for t in tokens)
        # errors='replace' handles any invalid UTF-8 sequences gracefully
        return byte_list.decode('utf-8', errors='replace')
    
    def save(self, path: str | Path) -> None:
        """Save tokenizer to file."""
        metadata = {
            'tokenizer_type': 'bpe',
            'merges': self._merges,
            'vocab': self._vocab,
            'vocab_size': self.vocab_size
        }
        torch.save(metadata, path)
    
    @classmethod
    def from_file(cls, path: str | Path) -> 'BPETokenizer':
        """Load BPE tokenizer from file."""
        metadata = torch.load(path, weights_only=False)
        return cls(merges=metadata['merges'], vocab=metadata['vocab'])
    
    @classmethod
    def from_metadata(cls, metadata: dict) -> 'BPETokenizer':
        """Create tokenizer from metadata dictionary."""
        return cls(merges=metadata['merges'], vocab=metadata['vocab'])
    
    @classmethod
    def train(cls, text: str, vocab_size: int = 1000) -> 'BPETokenizer':
        """
        Train BPE tokenizer on text.
        
        Args:
            text: Training text
            vocab_size: Target vocabulary size (minimum 256 for byte-level)
            
        Returns:
            Trained BPETokenizer
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (byte-level tokens)")
        
        num_merges = vocab_size - 256  # Number of merge operations
        
        print(f"Training BPE tokenizer with vocab_size={vocab_size} ({num_merges} merges)...")
        
        # Start with byte-level tokens
        tokens = list(text.encode('utf-8'))
        print(f"Initial tokens: {len(tokens):,} bytes")
        
        # Initialize vocabulary with byte-level tokens
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []
        
        for i in range(num_merges):
            # Count all adjacent pairs
            pair_counts = Counter()
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1])
                pair_counts[pair] += 1
            
            if not pair_counts:
                print(f"  No more pairs to merge at iteration {i}")
                break
            
            # Find most common pair
            best_pair = pair_counts.most_common(1)[0][0]
            best_count = pair_counts[best_pair]
            
            # Create new token
            new_token = 256 + i
            
            # Record the merge
            merges.append(best_pair)
            
            # Update vocabulary: new token = concatenation of pair bytes
            vocab[new_token] = vocab[best_pair[0]] + vocab[best_pair[1]]
            
            # Apply merge to token sequence
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == best_pair[0] and tokens[j + 1] == best_pair[1]:
                    new_tokens.append(new_token)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
            
            # Progress update every 10% of merges
            if (i + 1) % max(1, num_merges // 10) == 0 or i == num_merges - 1:
                compression = len(text.encode('utf-8')) / len(tokens)
                print(f"  Merge {i + 1}/{num_merges}: "
                      f"pair {best_pair} (count={best_count}), "
                      f"tokens={len(tokens):,}, "
                      f"compression={compression:.2f}x")
        
        final_compression = len(text.encode('utf-8')) / len(tokens)
        print(f"Training complete: {len(vocab)} tokens, {final_compression:.2f}x compression")
        
        return cls(merges=merges, vocab=vocab)


def load_tokenizer(path: str | Path) -> BaseTokenizer:
    """
    Load tokenizer from file, auto-detecting type.
    
    Args:
        path: Path to tokenizer file (vocab.pt)
        
    Returns:
        Appropriate tokenizer instance (CharTokenizer or BPETokenizer)
    """
    metadata = torch.load(path, weights_only=False)
    
    # Determine tokenizer type (default to 'char' for backward compatibility)
    tokenizer_type = metadata.get('tokenizer_type', 'char')
    
    if tokenizer_type == 'char':
        return CharTokenizer.from_metadata(metadata)
    elif tokenizer_type == 'bpe':
        return BPETokenizer.from_metadata(metadata)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


if __name__ == '__main__':
    # Simple demo/test
    import argparse
    
    parser = argparse.ArgumentParser(description='Test tokenizers')
    parser.add_argument('--type', choices=['char', 'bpe'], default='char',
                        help='Tokenizer type to test')
    parser.add_argument('--vocab_size', type=int, default=300,
                        help='Vocabulary size for BPE (default: 300)')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to tokenize (default: sample text)')
    
    args = parser.parse_args()
    
    # Sample text for testing
    test_text = args.text or """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them.
    """
    
    print(f"Testing {args.type} tokenizer")
    print("=" * 50)
    
    if args.type == 'char':
        tokenizer = CharTokenizer.train(test_text)
    else:
        tokenizer = BPETokenizer.train(test_text, vocab_size=args.vocab_size)
    
    print(f"\nTokenizer type: {tokenizer.type}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Test encode/decode roundtrip
    sample = test_text[:100]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nSample text: {repr(sample)}")
    print(f"Encoded ({len(encoded)} tokens): {encoded[:20]}...")
    print(f"Decoded: {repr(decoded)}")
    print(f"Roundtrip match: {sample == decoded}")
    
    # Test save/load
    test_path = Path('/tmp/test_tokenizer.pt')
    tokenizer.save(test_path)
    loaded = load_tokenizer(test_path)
    print(f"\nSave/load test: type={loaded.type}, vocab_size={loaded.vocab_size}")
    
    # Verify loaded tokenizer works
    encoded2 = loaded.encode(sample)
    decoded2 = loaded.decode(encoded2)
    print(f"Loaded tokenizer roundtrip match: {sample == decoded2}")

