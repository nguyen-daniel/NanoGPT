"""
Data preparation script for NanoGPT.
Supports custom text files or downloads the Tiny Shakespeare dataset.
Supports both character-level and BPE tokenization.

Usage:
    # Default: Download and prepare Tiny Shakespeare with character tokenization
    python data.py
    
    # Custom dataset: Prepare your own text file
    python data.py --input_file my_corpus.txt
    python data.py --input_file my_corpus.txt --data_dir my_data
    
    # BPE tokenization (subword)
    python data.py --tokenizer bpe --vocab_size 1000
    python data.py --input_file my_corpus.txt --tokenizer bpe --vocab_size 2000

© 2026
"""

import os
import argparse
import torch
import requests
from pathlib import Path
from tokenizer import CharTokenizer, BPETokenizer, load_tokenizer


def download_shakespeare(data_dir='data'):
    """
    Download the Tiny Shakespeare dataset if it doesn't exist.
    
    Args:
        data_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded text file
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    input_file = data_dir / 'input.txt'
    
    if not input_file.exists():
        print("Downloading Tiny Shakespeare dataset...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        response = requests.get(url)
        response.raise_for_status()
        
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded dataset to {input_file}")
    else:
        print(f"Dataset already exists at {input_file}")
    
    return input_file


def load_custom_file(input_file):
    """
    Load a custom text file for training.
    
    Args:
        input_file: Path to the input text file
        
    Returns:
        Path to the text file (validated)
    
    Raises:
        FileNotFoundError: If the input file doesn't exist
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Using custom input file: {input_path}")
    return input_path


def get_vocabulary(text):
    """
    Extract unique characters from text to create vocabulary.
    
    Args:
        text: Input text string
        
    Returns:
        sorted list of unique characters
    """
    chars = sorted(list(set(text)))
    return chars


def create_encoder_decoder(vocab):
    """
    Create encode and decode functions for character-to-integer mapping.
    
    Args:
        vocab: List of unique characters
        
    Returns:
        encode: Function that maps characters to integers
        decode: Function that maps integers to characters
    """
    # Create mapping dictionaries
    char_to_int = {ch: i for i, ch in enumerate(vocab)}
    int_to_char = {i: ch for i, ch in enumerate(vocab)}
    
    def encode(text):
        """Convert string to list of integers."""
        return [char_to_int[ch] for ch in text]
    
    def decode(integers):
        """Convert list of integers to string."""
        return ''.join([int_to_char[i] for i in integers])
    
    return encode, decode


def prepare_data(data_dir='data', train_split=0.9, input_file=None, 
                 tokenizer_type='char', vocab_size=None):
    """
    Main function to prepare the dataset.
    
    Args:
        data_dir: Directory to save/load data
        train_split: Fraction of data to use for training (default 0.9)
        input_file: Path to custom text file (optional). If None, downloads Shakespeare.
        tokenizer_type: Type of tokenizer to use ('char' or 'bpe')
        vocab_size: Target vocabulary size for BPE (ignored for char tokenizer)
        
    Returns:
        Dictionary containing:
            - train_data: Training tensor
            - val_data: Validation tensor
            - vocab_size: Size of vocabulary
            - tokenizer: Tokenizer instance
            - encode: Encoding function
            - decode: Decoding function
    """
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Get input file - either custom or download Shakespeare
    if input_file is not None:
        text_file = load_custom_file(input_file)
    else:
        text_file = download_shakespeare(data_dir)
    
    # Read text
    print("Reading text file...")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset contains {len(text):,} characters")
    
    # Create tokenizer based on type
    print(f"\nTraining {tokenizer_type} tokenizer...")
    if tokenizer_type == 'char':
        tokenizer = CharTokenizer.train(text)
        print(f"Vocabulary size: {tokenizer.vocab_size} unique characters")
        if tokenizer.vocab_size <= 100:
            print(f"Vocabulary: {''.join(tokenizer.vocab)}")
    elif tokenizer_type == 'bpe':
        if vocab_size is None:
            vocab_size = 1000  # Default BPE vocab size
        tokenizer = BPETokenizer.train(text, vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    # Encode entire dataset
    print("\nEncoding dataset...")
    tokens = tokenizer.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)
    print(f"Encoded dataset: {len(data):,} tokens")
    
    # Calculate compression ratio for BPE
    if tokenizer_type == 'bpe':
        char_count = len(text)
        compression = char_count / len(data)
        print(f"Compression ratio: {compression:.2f}x (vs character-level)")
    
    # Split into train and validation
    n = len(data)
    split_idx = int(train_split * n)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"\nTraining data: {len(train_data):,} tokens ({len(train_data)/n*100:.1f}%)")
    print(f"Validation data: {len(val_data):,} tokens ({len(val_data)/n*100:.1f}%)")
    
    # Save processed data
    print("\nSaving processed data...")
    output_dir = Path(data_dir)
    torch.save(train_data, output_dir / 'train.pt')
    torch.save(val_data, output_dir / 'val.pt')
    
    # Save tokenizer (includes vocab metadata)
    tokenizer.save(output_dir / 'vocab.pt')
    
    print(f"Saved processed data to {output_dir}/")
    print("Files created:")
    print(f"  - {output_dir / 'train.pt'}")
    print(f"  - {output_dir / 'val.pt'}")
    print(f"  - {output_dir / 'vocab.pt'} (tokenizer: {tokenizer_type})")
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'vocab_size': tokenizer.vocab_size,
        'tokenizer': tokenizer,
        'encode': tokenizer.encode,
        'decode': tokenizer.decode
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for NanoGPT training')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to custom text file (default: download Tiny Shakespeare)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to save processed data (default: data)')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Fraction of data for training (default: 0.9)')
    parser.add_argument('--tokenizer', type=str, default='char', choices=['char', 'bpe'],
                        help='Tokenizer type: char (character-level) or bpe (subword) (default: char)')
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='Vocabulary size for BPE tokenizer (default: 1000, ignored for char)')
    
    args = parser.parse_args()
    
    # Prepare the dataset
    data = prepare_data(
        data_dir=args.data_dir,
        train_split=args.train_split,
        input_file=args.input_file,
        tokenizer_type=args.tokenizer,
        vocab_size=args.vocab_size
    )
    
    # Test encode/decode with a sample from the actual corpus
    print("\n" + "="*50)
    print("Testing encode/decode functions:")
    # Use first 50 tokens from the corpus for testing
    test_tokens = data['train_data'][:50].tolist()
    test_string = data['decode'](test_tokens)
    encoded = data['encode'](test_string)
    decoded = data['decode'](encoded)
    print(f"Sample text: {repr(test_string[:100])}")
    print(f"Encoded length: {len(encoded)} tokens")
    print(f"Roundtrip match: {test_string == decoded}")

