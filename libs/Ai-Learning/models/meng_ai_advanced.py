"""
Meng AI - Advanced Language Model with Scaling Configurations
============================================================

This is the advanced version of Meng AI with:
1. Multiple model size configurations (Small, Medium, Large, XL)
2. Advanced training features (mixed precision, gradient accumulation)
3. Better optimization strategies
4. Advanced sampling techniques
5. Production-ready features

Author: Meng AI Development Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ModelSize(Enum):
    """Predefined model sizes for Meng AI"""
    SMALL = "small"      # ~10M parameters - Fast training, good for learning
    MEDIUM = "medium"    # ~50M parameters - Balanced performance
    LARGE = "large"      # ~200M parameters - High quality
    XL = "xl"           # ~500M parameters - Production quality

@dataclass
class MengAIConfig:
    """Configuration class for Meng AI models"""
    vocab_size: int = 50000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 12
    d_ff: int = 2048
    max_seq_length: int = 1024
    dropout: float = 0.1
    use_learnable_pos_emb: bool = True
    use_rotary_emb: bool = False
    activation: str = "gelu"  # gelu, relu, swish
    layer_norm_eps: float = 1e-5
    use_gradient_checkpointing: bool = False
    
    @classmethod
    def from_size(cls, size: ModelSize, vocab_size: int = 50000):
        """Create configuration based on predefined model size"""
        configs = {
            ModelSize.SMALL: cls(
                vocab_size=vocab_size,
                d_model=256,
                num_heads=4,
                num_layers=6,
                d_ff=1024,
                max_seq_length=512,
                dropout=0.1
            ),
            ModelSize.MEDIUM: cls(
                vocab_size=vocab_size,
                d_model=512,
                num_heads=8,
                num_layers=12,
                d_ff=2048,
                max_seq_length=1024,
                dropout=0.1
            ),
            ModelSize.LARGE: cls(
                vocab_size=vocab_size,
                d_model=1024,
                num_heads=16,
                num_layers=24,
                d_ff=4096,
                max_seq_length=2048,
                dropout=0.1,
                use_rotary_emb=True
            ),
            ModelSize.XL: cls(
                vocab_size=vocab_size,
                d_model=1536,
                num_heads=24,
                num_layers=36,
                d_ff=6144,
                max_seq_length=4096,
                dropout=0.1,
                use_rotary_emb=True,
                use_gradient_checkpointing=True
            )
        }
        return configs[size]

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position encoding"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embedding to query and key tensors"""
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class AdvancedMultiHeadAttention(nn.Module):
    """Advanced Multi-Head Attention with RoPE support"""
    
    def __init__(self, config: MengAIConfig):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_model // config.num_heads
        self.use_rotary_emb = config.use_rotary_emb
        
        # Linear transformations
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Rotary embedding
        if self.use_rotary_emb:
            self.rotary_emb = RotaryEmbedding(self.d_k, config.max_seq_length)
        
        # Attention scaling
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def forward(self, x, mask=None, position_ids=None):
        batch_size, seq_len, d_model = x.size()
        
        # Generate Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply rotary embedding if enabled
        if self.use_rotary_emb and position_ids is not None:
            cos, sin = self.rotary_emb(x, seq_len)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        return output

class AdvancedFeedForward(nn.Module):
    """Advanced Feed-Forward Network with configurable activation"""
    
    def __init__(self, config: MengAIConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Configurable activation
        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class AdvancedTransformerBlock(nn.Module):
    """Advanced Transformer Block with gradient checkpointing"""
    
    def __init__(self, config: MengAIConfig):
        super().__init__()
        self.attention = AdvancedMultiHeadAttention(config)
        self.feed_forward = AdvancedFeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
    
    def forward(self, x, mask=None, position_ids=None):
        # Self-attention with residual connection
        if self.use_gradient_checkpointing and self.training:
            attn_output = torch.utils.checkpoint.checkpoint(
                self.attention, x, mask, position_ids
            )
        else:
            attn_output = self.attention(x, mask, position_ids)
        
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        if self.use_gradient_checkpointing and self.training:
            ff_output = torch.utils.checkpoint.checkpoint(self.feed_forward, x)
        else:
            ff_output = self.feed_forward(x)
        
        x = self.norm2(x + self.dropout(ff_output))
        return x

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        
    def forward(self, x, position_ids=None):
        seq_len = x.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        pos_emb = self.pos_embedding(position_ids)
        return x + pos_emb

class MengAIAdvanced(nn.Module):
    """
    Meng AI Advanced - Production-ready language model
    
    Features:
    - Multiple model size configurations
    - Advanced attention mechanisms (RoPE)
    - Learnable positional embeddings
    - Gradient checkpointing for memory efficiency
    - Configurable activations
    - Advanced sampling strategies
    """
    
    def __init__(self, config: MengAIConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.max_seq_length = config.max_seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding
        if config.use_learnable_pos_emb:
            self.positional_encoding = LearnablePositionalEncoding(
                config.d_model, config.max_seq_length
            )
        else:
            # Use sinusoidal encoding (from original implementation)
            pe = torch.zeros(config.max_seq_length, config.d_model)
            position = torch.arange(0, config.max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * 
                               (-math.log(10000.0) / config.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
            self.positional_encoding = None
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            AdvancedTransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Advanced weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int):
        """Create causal mask to prevent attention to future tokens"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids, position_ids=None, return_attention=False):
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, position_ids)
        else:
            x = x + self.pe[:seq_len, :].transpose(0, 1)
        
        x = self.dropout(x)
        
        # Pass through transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask, position_ids)
        
        # Output logits
        logits = self.output_layer(x)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def generate(self, input_ids, max_length: int = 100, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.9, do_sample: bool = True,
                 repetition_penalty: float = 1.0, length_penalty: float = 1.0):
        """
        Advanced text generation with multiple sampling strategies
        
        Args:
            input_ids: Starting sequence of token IDs
            max_length: Maximum length of generated sequence
            temperature: Controls randomness
            top_k: Consider only top-k most likely tokens
            top_p: Consider tokens with cumulative probability up to top_p
            do_sample: Whether to sample or use greedy decoding
            repetition_penalty: Penalty for repeating tokens
            length_penalty: Penalty for sequence length
        """
        self.eval()
        generated = input_ids.clone()
        used_tokens = set()
        
        with torch.no_grad():
            for step in range(max_length):
                # Forward pass
                logits = self.forward(generated)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in used_tokens:
                        if next_token_logits[0, token_id] < 0:
                            next_token_logits[0, token_id] *= repetition_penalty
                        else:
                            next_token_logits[0, token_id] /= repetition_penalty
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
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
                
                # Apply length penalty
                if length_penalty != 1.0:
                    next_token_logits *= (1.0 + step * (length_penalty - 1.0))
                
                # Append the generated token
                generated = torch.cat([generated, next_token], dim=1)
                used_tokens.add(next_token.item())
                
                # Stop if we generate an end-of-sequence token
                if next_token.item() == 0:  # Assuming 0 is PAD/EOS
                    break
        
        return generated
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'config': self.config
        }
    
    def save_config(self, filepath: str):
        """Save model configuration"""
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'd_model': self.config.d_model,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'd_ff': self.config.d_ff,
            'max_seq_length': self.config.max_seq_length,
            'dropout': self.config.dropout,
            'use_learnable_pos_emb': self.config.use_learnable_pos_emb,
            'use_rotary_emb': self.config.use_rotary_emb,
            'activation': self.config.activation,
            'layer_norm_eps': self.config.layer_norm_eps,
            'use_gradient_checkpointing': self.config.use_gradient_checkpointing
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_config_file(cls, filepath: str):
        """Load model from configuration file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = MengAIConfig(**config_dict)
        return cls(config)

def create_meng_ai_model(size: ModelSize, vocab_size: int = 50000) -> MengAIAdvanced:
    """Factory function to create Meng AI models of different sizes"""
    config = MengAIConfig.from_size(size, vocab_size)
    model = MengAIAdvanced(config)
    
    print(f"Created Meng AI {size.value.upper()} model:")
    size_info = model.get_model_size()
    print(f"  Parameters: {size_info['total_parameters']:,}")
    print(f"  Model Size: {size_info['model_size_mb']:.1f} MB")
    print(f"  Layers: {config.num_layers}")
    print(f"  Dimensions: {config.d_model}")
    print(f"  Attention Heads: {config.num_heads}")
    
    return model

# Example usage
if __name__ == "__main__":
    print("🚀 Meng AI Advanced - Model Creation Test")
    print("=" * 50)
    
    # Test different model sizes
    for size in [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]:
        print(f"\nCreating {size.value.upper()} model...")
        model = create_meng_ai_model(size, vocab_size=1000)
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        with torch.no_grad():
            logits = model(input_ids)
            print(f"  Output shape: {logits.shape}")
        
        # Test generation
        start_tokens = torch.randint(1, 1000, (1, 3))
        generated = model.generate(start_tokens, max_length=10, temperature=0.8)
        print(f"  Generated sequence length: {generated.shape[1]}")
    
    print(f"\n✅ All Meng AI models created successfully!")
    print(f"\nNext steps:")
    print(f"1. Choose your model size based on available resources")
    print(f"2. Train with advanced training pipeline")
    print(f"3. Use advanced sampling for better text generation")
    print(f"4. Scale up to production deployment")
