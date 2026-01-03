"""
Data preparation script for NanoGPT.
Downloads and processes the Tiny Shakespeare dataset for character-level language modeling.

© 2026
"""

import os
import torch
import requests
from pathlib import Path


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


def prepare_data(data_dir='data', train_split=0.9):
    """
    Main function to prepare the dataset.
    
    Args:
        data_dir: Directory to save/load data
        train_split: Fraction of data to use for training (default 0.9)
        
    Returns:
        Dictionary containing:
            - train_data: Training tensor
            - val_data: Validation tensor
            - vocab: Vocabulary (list of characters)
            - vocab_size: Size of vocabulary
            - encode: Encoding function
            - decode: Decoding function
    """
    # Download dataset
    input_file = download_shakespeare(data_dir)
    
    # Read text
    print("Reading text file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset contains {len(text):,} characters")
    
    # Create vocabulary
    print("Creating vocabulary...")
    vocab = get_vocabulary(text)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Vocabulary: {''.join(vocab)}")
    
    # Create encoder/decoder
    encode, decode = create_encoder_decoder(vocab)
    
    # Encode entire dataset
    print("Encoding dataset...")
    data = torch.tensor(encode(text), dtype=torch.long)
    print(f"Encoded dataset shape: {data.shape}")
    
    # Split into train and validation
    n = len(data)
    split_idx = int(train_split * n)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Training data: {len(train_data):,} tokens ({len(train_data)/n*100:.1f}%)")
    print(f"Validation data: {len(val_data):,} tokens ({len(val_data)/n*100:.1f}%)")
    
    # Save processed data
    print("Saving processed data...")
    output_dir = Path(data_dir)
    torch.save(train_data, output_dir / 'train.pt')
    torch.save(val_data, output_dir / 'val.pt')
    
    # Save vocabulary metadata
    metadata = {
        'vocab': vocab,
        'vocab_size': vocab_size,
        'char_to_int': {ch: i for i, ch in enumerate(vocab)},
        'int_to_char': {i: ch for i, ch in enumerate(vocab)}
    }
    torch.save(metadata, output_dir / 'vocab.pt')
    
    print(f"Saved processed data to {output_dir}/")
    print("Files created:")
    print(f"  - {output_dir / 'train.pt'}")
    print(f"  - {output_dir / 'val.pt'}")
    print(f"  - {output_dir / 'vocab.pt'}")
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'vocab': vocab,
        'vocab_size': vocab_size,
        'encode': encode,
        'decode': decode
    }


if __name__ == '__main__':
    # Prepare the dataset
    data = prepare_data()
    
    # Test encode/decode
    print("\n" + "="*50)
    print("Testing encode/decode functions:")
    test_string = "Hello, World!"
    encoded = data['encode'](test_string)
    decoded = data['decode'](encoded)
    print(f"Original: {test_string}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    print(f"Match: {test_string == decoded}")

