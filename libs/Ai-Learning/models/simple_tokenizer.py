"""
Simple Tokenizer for LLM Training
=================================

A basic tokenizer that converts text to token IDs and vice versa.
This tokenizer includes:

1. Text preprocessing (lowercase, basic cleaning)
2. Word-level tokenization
3. Vocabulary building
4. Token ID conversion
5. Special tokens (PAD, UNK, BOS, EOS)

This is a simplified version - in practice, you'd use more sophisticated
tokenizers like BPE, WordPiece, or SentencePiece.

Author: AI Learning Course
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter
import pickle
import os

class SimpleTokenizer:
    """
    A simple word-level tokenizer for training language models.
    
    This tokenizer:
    - Converts text to lowercase
    - Splits on whitespace and punctuation
    - Builds vocabulary from training data
    - Handles special tokens
    """
    
    def __init__(self, vocab_size: int = 10000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>"  # Beginning of sequence
        self.EOS_TOKEN = "<EOS>"  # End of sequence
        
        # Vocabulary
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size_actual = 0
        
        # Statistics
        self.total_tokens = 0
        self.unique_tokens = 0
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Add spaces around punctuation for better tokenization
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        text = re.sub(r'["\']', r' \1 ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization
        tokens = text.split()
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of training texts
        """
        print("Building vocabulary...")
        
        # Count word frequencies
        word_counts = Counter()
        
        for text in texts:
            preprocessed = self.preprocess_text(text)
            tokens = self.tokenize(preprocessed)
            word_counts.update(tokens)
        
        # Add special tokens first
        self.word_to_id[self.PAD_TOKEN] = 0
        self.word_to_id[self.UNK_TOKEN] = 1
        self.word_to_id[self.BOS_TOKEN] = 2
        self.word_to_id[self.EOS_TOKEN] = 3
        
        # Add most frequent words
        most_common = word_counts.most_common(self.vocab_size - 4)  # Reserve space for special tokens
        
        for word, count in most_common:
            if count >= self.min_frequency:
                self.word_to_id[word] = len(self.word_to_id)
        
        # Create reverse mapping
        self.id_to_word = {id: word for word, id in self.word_to_id.items()}
        self.vocab_size_actual = len(self.word_to_id)
        
        # Update statistics
        self.total_tokens = sum(word_counts.values())
        self.unique_tokens = len(word_counts)
        
        print(f"Vocabulary built: {self.vocab_size_actual} tokens")
        print(f"Total tokens in corpus: {self.total_tokens:,}")
        print(f"Unique tokens: {self.unique_tokens:,}")
        print(f"Coverage: {self.vocab_size_actual / self.unique_tokens * 100:.1f}%")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        preprocessed = self.preprocess_text(text)
        tokens = self.tokenize(preprocessed)
        
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.word_to_id[self.BOS_TOKEN])
        
        for token in tokens:
            token_id = self.word_to_id.get(token, self.word_to_id[self.UNK_TOKEN])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.word_to_id[self.EOS_TOKEN])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_word:
                word = self.id_to_word[token_id]
                
                if skip_special_tokens and word in [self.PAD_TOKEN, self.UNK_TOKEN, 
                                                   self.BOS_TOKEN, self.EOS_TOKEN]:
                    continue
                
                tokens.append(word)
            else:
                if not skip_special_tokens:
                    tokens.append(f"<UNK_{token_id}>")
        
        # Join tokens and clean up spacing
        text = " ".join(tokens)
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'(["\'])\s+', r'\1', text)     # Remove space after opening quotes
        text = re.sub(r'\s+(["\'])', r'\1', text)     # Remove space before closing quotes
        
        return text.strip()
    
    def get_vocab_size(self) -> int:
        """Get the actual vocabulary size"""
        return self.vocab_size_actual
    
    def save(self, filepath: str) -> None:
        """Save tokenizer to file"""
        tokenizer_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'vocab_size_actual': self.vocab_size_actual,
            'total_tokens': self.total_tokens,
            'unique_tokens': self.unique_tokens
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load tokenizer from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        self.word_to_id = tokenizer_data['word_to_id']
        self.id_to_word = {int(k): v for k, v in tokenizer_data['id_to_word'].items()}
        self.vocab_size = tokenizer_data['vocab_size']
        self.min_frequency = tokenizer_data['min_frequency']
        self.vocab_size_actual = tokenizer_data['vocab_size_actual']
        self.total_tokens = tokenizer_data['total_tokens']
        self.unique_tokens = tokenizer_data['unique_tokens']
        
        print(f"Tokenizer loaded from {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get tokenizer statistics"""
        return {
            'vocab_size': self.vocab_size_actual,
            'total_tokens': self.total_tokens,
            'unique_tokens': self.unique_tokens,
            'coverage': self.vocab_size_actual / self.unique_tokens * 100 if self.unique_tokens > 0 else 0,
            'special_tokens': {
                'PAD': self.word_to_id.get(self.PAD_TOKEN, -1),
                'UNK': self.word_to_id.get(self.UNK_TOKEN, -1),
                'BOS': self.word_to_id.get(self.BOS_TOKEN, -1),
                'EOS': self.word_to_id.get(self.EOS_TOKEN, -1)
            }
        }

def create_sample_dataset() -> List[str]:
    """
    Create a sample dataset for testing the tokenizer.
    In practice, you would load your own training data.
    """
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language for machine learning.",
        "Artificial intelligence will change the world in many ways.",
        "Deep learning models can understand complex patterns in data.",
        "Natural language processing helps computers understand human language.",
        "Machine learning algorithms learn from data without explicit programming.",
        "Neural networks are inspired by the structure of the human brain.",
        "Computer vision enables machines to interpret and understand visual information.",
        "Reinforcement learning teaches agents to make decisions through trial and error.",
        "Data science combines statistics, programming, and domain expertise.",
        "The future of technology lies in artificial intelligence and automation.",
        "Programming is both an art and a science that requires creativity and logic.",
        "Open source software has revolutionized the way we develop technology.",
        "Cloud computing provides scalable and flexible computing resources.",
        "Cybersecurity is crucial for protecting digital information and systems.",
        "Blockchain technology offers secure and transparent ways to store data.",
        "The internet has connected people and information across the globe.",
        "Mobile applications have transformed how we interact with technology.",
        "User experience design focuses on creating intuitive and enjoyable interfaces.",
        "Software engineering principles help build reliable and maintainable systems."
    ]
    
    return sample_texts

# Example usage and testing
if __name__ == "__main__":
    print("🔤 Simple Tokenizer for LLM")
    print("=" * 40)
    
    # Create sample dataset
    texts = create_sample_dataset()
    print(f"Created sample dataset with {len(texts)} texts")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000, min_frequency=1)
    
    # Build vocabulary
    tokenizer.build_vocabulary(texts)
    
    # Test encoding and decoding
    test_text = "Python is a powerful programming language for machine learning."
    print(f"\nOriginal text: {test_text}")
    
    # Encode
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded_text}")
    
    # Test without special tokens
    token_ids_no_special = tokenizer.encode(test_text, add_special_tokens=False)
    decoded_no_special = tokenizer.decode(token_ids_no_special, skip_special_tokens=True)
    print(f"Without special tokens: {decoded_no_special}")
    
    # Show statistics
    stats = tokenizer.get_statistics()
    print(f"\nTokenizer Statistics:")
    for key, value in stats.items():
        if key != 'special_tokens':
            print(f"  {key}: {value}")
    
    print(f"\nSpecial Tokens:")
    for token, id in stats['special_tokens'].items():
        print(f"  {token}: {id}")
    
    # Test with unknown words
    unknown_text = "This is a completely unknown word: xyzabc123"
    print(f"\nTesting with unknown words: {unknown_text}")
    unknown_ids = tokenizer.encode(unknown_text)
    print(f"Token IDs: {unknown_ids}")
    decoded_unknown = tokenizer.decode(unknown_ids)
    print(f"Decoded: {decoded_unknown}")
    
    print("\n✅ Tokenizer is ready for training!")
    print("\nNext steps:")
    print("1. Load your training data")
    print("2. Build vocabulary from your data")
    print("3. Use tokenizer to prepare training sequences")
    print("4. Train your language model")
