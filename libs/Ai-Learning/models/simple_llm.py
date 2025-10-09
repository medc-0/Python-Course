"""
Meng AI - Advanced Language Model
=================================

Meng AI is a powerful transformer-based language model implementation
that can generate high-quality text. It includes:

1. Multi-Head Self-Attention mechanism
2. Feed-Forward Networks with advanced activations
3. Layer Normalization and Residual Connections
4. Positional Encoding with learnable embeddings
5. Advanced text generation with multiple sampling strategies
6. Scalable architecture for different model sizes

Meng AI is designed to be both educational and production-ready,
scalable from small models to large language models.

Author: Meng AI Development Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Tuple, Optional
import json

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism - the core of transformer architecture.
    This allows the model to focus on different parts of the input sequence.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for preventing attention to future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Generate Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        return output

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Networks.
    Applied to each position separately and identically.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """
    A single transformer block containing:
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - Layer Normalization and Residual Connections
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class PositionalEncoding(nn.Module):
    """
    Positional encoding to give the model information about token positions.
    Since transformers don't have inherent notion of sequence order.
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MengAI(nn.Module):
    """
    Meng AI - Advanced Language Model based on Transformer architecture.
    
    This model can be trained to predict the next token in a sequence
    and then used to generate high-quality text. Supports multiple model sizes
    from small (educational) to large (production) configurations.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 1024, max_seq_length: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int):
        """Create causal mask to prevent attention to future tokens"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids, return_attention=False):
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Output logits
        logits = self.output_layer(x)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def generate(self, input_ids, max_length: int = 100, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.9, do_sample: bool = True):
        """
        Generate text using the trained model.
        
        Args:
            input_ids: Starting sequence of token IDs
            max_length: Maximum length of generated sequence
            temperature: Controls randomness (lower = more deterministic)
            top_k: Consider only top-k most likely tokens
            top_p: Consider tokens with cumulative probability up to top_p
            do_sample: Whether to sample or use greedy decoding
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(generated)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append the generated token
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we generate an end-of-sequence token (assuming 0 is padding/end)
                if next_token.item() == 0:
                    break
        
        return generated

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Simple LLM from Scratch")
    print("=" * 50)
    
    # Model configuration
    vocab_size = 1000  # Small vocabulary for demonstration
    d_model = 128      # Reduced for faster training
    num_heads = 4
    num_layers = 3
    d_ff = 512
    max_seq_length = 128
    
    # Create model
    model = SimpleLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length
    )
    
    print(f"Model created with {count_parameters(model):,} parameters")
    print(f"Model size: {d_model} dimensions, {num_layers} layers, {num_heads} attention heads")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nTesting forward pass with input shape: {input_ids.shape}")
    logits = model(input_ids)
    print(f"Output logits shape: {logits.shape}")
    
    # Test generation
    print(f"\nTesting text generation...")
    start_tokens = torch.randint(1, vocab_size, (1, 5))  # Start with 5 tokens
    print(f"Starting tokens: {start_tokens.squeeze().tolist()}")
    
    generated = model.generate(start_tokens, max_length=20, temperature=0.8)
    print(f"Generated sequence: {generated.squeeze().tolist()}")
    
    print("\n✅ Model is ready for training!")
    print("\nNext steps:")
    print("1. Prepare your training data")
    print("2. Create a tokenizer")
    print("3. Train the model")
    print("4. Generate text with your trained model")
