"""
LLM Training Script
==================

This script provides a complete training pipeline for the Simple LLM.
It includes:

1. Data loading and preprocessing
2. Dataset creation with sliding windows
3. Training loop with loss calculation
4. Model evaluation and metrics
5. Checkpointing and model saving
6. Learning rate scheduling
7. Gradient clipping

Author: AI Learning Course
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from typing import List, Tuple, Dict, Optional
import time
from datetime import datetime

from simple_llm import SimpleLLM
from simple_tokenizer import SimpleTokenizer, create_sample_dataset

class TextDataset(Dataset):
    """
    Dataset class for training the language model.
    Creates sliding window sequences from tokenized text.
    """
    
    def __init__(self, token_ids: List[int], sequence_length: int = 128):
        self.token_ids = token_ids
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.token_ids) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of tokens
        sequence = self.token_ids[idx:idx + self.sequence_length + 1]
        
        # Split into input and target
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids

class LLMTrainer:
    """
    Complete trainer for the Simple LLM.
    Handles training, evaluation, and model management.
    """
    
    def __init__(self, model: SimpleLLM, tokenizer: SimpleTokenizer, 
                 device: str = 'auto'):
        self.model = model
        self.tokenizer = tokenizer
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def prepare_data(self, texts: List[str], sequence_length: int = 128,
                    train_split: float = 0.8, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.
        
        Args:
            texts: List of training texts
            sequence_length: Length of input sequences
            train_split: Fraction of data for training
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        print("Preparing data...")
        
        # Tokenize all texts
        all_token_ids = []
        for text in texts:
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            all_token_ids.extend(token_ids)
        
        print(f"Total tokens: {len(all_token_ids):,}")
        
        # Split into train and validation
        split_idx = int(len(all_token_ids) * train_split)
        train_tokens = all_token_ids[:split_idx]
        val_tokens = all_token_ids[split_idx:]
        
        print(f"Train tokens: {len(train_tokens):,}")
        print(f"Validation tokens: {len(val_tokens):,}")
        
        # Create datasets
        train_dataset = TextDataset(train_tokens, sequence_length)
        val_dataset = TextDataset(val_tokens, sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def setup_optimizer(self, learning_rate: float = 1e-4, weight_decay: float = 0.01,
                       scheduler_type: str = 'cosine', warmup_steps: int = 1000):
        """
        Setup optimizer and learning rate scheduler.
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
            warmup_steps: Number of warmup steps
        """
        # Use AdamW optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Setup scheduler
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10000, eta_min=learning_rate * 0.1
            )
        elif scheduler_type == 'linear':
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=warmup_steps
            )
        else:
            self.scheduler = None
        
        print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"Scheduler: {scheduler_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            
            # Calculate loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Store learning rate
            self.learning_rates.append(current_lr)
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(val_loader, desc="Validation"):
                # Move to device
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Calculate loss
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def generate_sample(self, prompt: str = "", max_length: int = 50, 
                       temperature: float = 0.8) -> str:
        """
        Generate a sample text from the model.
        
        Args:
            prompt: Starting prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        with torch.no_grad():
            if prompt:
                # Encode prompt
                input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            else:
                # Start with BOS token
                input_tensor = torch.tensor([[self.tokenizer.word_to_id[self.tokenizer.BOS_TOKEN]]], 
                                          dtype=torch.long).to(self.device)
            
            # Generate
            generated = self.model.generate(
                input_tensor, 
                max_length=max_length, 
                temperature=temperature,
                do_sample=True
            )
            
            # Decode
            generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
            
        return generated_text
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 10, save_dir: str = "checkpoints",
              save_every: int = 2, generate_every: int = 1) -> Dict:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            generate_every: Generate sample every N epochs
            
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update history
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['learning_rates'].extend(self.learning_rates[-len(train_loader):])
            history['epochs'].append(epoch + 1)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Generate sample
            if (epoch + 1) % generate_every == 0:
                print("\nGenerated Sample:")
                sample = self.generate_sample("The future of", max_length=30, temperature=0.8)
                print(f"'{sample}'")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_dir, epoch + 1, val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(save_dir, "best", val_loss)
                print(f"New best model saved! (Val Loss: {val_loss:.4f})")
        
        return history
    
    def save_checkpoint(self, save_dir: str, epoch: str, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'tokenizer_vocab': self.tokenizer.word_to_id,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'max_seq_length': self.model.max_seq_length
            }
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {filepath}")
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['epochs'], history['train_losses'], label='Train Loss', marker='o')
        ax1.plot(history['epochs'], history['val_losses'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate plot
        ax2.plot(history['learning_rates'], label='Learning Rate', alpha=0.7)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved: {save_path}")
        
        plt.show()

def main():
    """Main training function"""
    print("🚀 LLM Training Pipeline")
    print("=" * 50)
    
    # Configuration
    config = {
        'vocab_size': 2000,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'max_seq_length': 128,
        'sequence_length': 128,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 5,
        'train_split': 0.8
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create sample dataset (in practice, load your own data)
    print("\nLoading data...")
    texts = create_sample_dataset()
    print(f"Loaded {len(texts)} texts")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'], min_frequency=1)
    tokenizer.build_vocabulary(texts)
    
    # Initialize model
    print("\nInitializing model...")
    model = SimpleLLM(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = LLMTrainer(model, tokenizer)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        texts, 
        sequence_length=config['sequence_length'],
        train_split=config['train_split'],
        batch_size=config['batch_size']
    )
    
    # Setup optimizer
    trainer.setup_optimizer(learning_rate=config['learning_rate'])
    
    # Train model
    print(f"\nStarting training...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=config['num_epochs'],
        save_dir="checkpoints",
        save_every=1,
        generate_every=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Plot training history
    trainer.plot_training_history(history, "training_history.png")
    
    # Final generation test
    print("\nFinal Generation Test:")
    prompts = ["The future of", "Python is", "Artificial intelligence"]
    
    for prompt in prompts:
        generated = trainer.generate_sample(prompt, max_length=40, temperature=0.8)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()
    
    print("✅ Training completed successfully!")
    print("\nNext steps:")
    print("1. Load the best checkpoint")
    print("2. Generate more text samples")
    print("3. Fine-tune on your specific domain")
    print("4. Deploy for inference")

if __name__ == "__main__":
    main()
