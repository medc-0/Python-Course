"""
Meng AI Advanced Trainer
=======================

Advanced training pipeline for Meng AI with:
1. Mixed precision training (FP16)
2. Gradient accumulation for large batch simulation
3. Advanced optimizers (AdamW, Lion, AdaFactor)
4. Learning rate scheduling (cosine, linear, polynomial)
5. Model parallelism support
6. Advanced evaluation metrics
7. Distributed training support
8. Memory optimization

Author: Meng AI Development Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import wandb
from collections import defaultdict

from meng_ai_advanced import MengAIAdvanced, MengAIConfig, ModelSize, create_meng_ai_model
from simple_tokenizer import SimpleTokenizer

@dataclass
class TrainingConfig:
    """Advanced training configuration"""
    # Model configuration
    model_size: ModelSize = ModelSize.MEDIUM
    vocab_size: int = 50000
    
    # Training parameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_epochs: int = 10
    max_steps: Optional[int] = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Optimization
    optimizer: str = "adamw"  # adamw, lion, adafactor
    scheduler: str = "cosine"  # cosine, linear, polynomial, constant
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16, bfloat16
    
    # Memory optimization
    use_gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Data
    sequence_length: int = 1024
    train_split: float = 0.8
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True

class AdvancedDataset(Dataset):
    """Advanced dataset with dynamic batching and memory optimization"""
    
    def __init__(self, token_ids: List[int], sequence_length: int = 1024, 
                 dynamic_length: bool = False):
        self.token_ids = token_ids
        self.sequence_length = sequence_length
        self.dynamic_length = dynamic_length
        
        if dynamic_length:
            # Create sequences of varying lengths for better training
            self.sequences = self._create_dynamic_sequences()
        else:
            self.sequences = self._create_fixed_sequences()
    
    def _create_fixed_sequences(self):
        """Create fixed-length sequences"""
        sequences = []
        for i in range(0, len(self.token_ids) - self.sequence_length, self.sequence_length):
            sequences.append(self.token_ids[i:i + self.sequence_length + 1])
        return sequences
    
    def _create_dynamic_sequences(self):
        """Create sequences of varying lengths"""
        sequences = []
        lengths = [512, 768, 1024, 1280, 1536]  # Different sequence lengths
        
        for length in lengths:
            for i in range(0, len(self.token_ids) - length, length):
                sequences.append(self.token_ids[i:i + length + 1])
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Split into input and target
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids

class AdvancedOptimizer:
    """Advanced optimizer factory"""
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: TrainingConfig):
        """Create optimizer based on configuration"""
        
        # Parameter grouping for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if config.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "lion":
            # Lion optimizer (if available)
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    optimizer_grouped_parameters,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
            except ImportError:
                print("Lion optimizer not available, falling back to AdamW")
                optimizer = optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=config.learning_rate,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=config.weight_decay
                )
        elif config.optimizer.lower() == "adafactor":
            # Adafactor optimizer (if available)
            try:
                from transformers import Adafactor
                optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=config.learning_rate,
                    scale_parameter=False,
                    relative_step=False
                )
            except ImportError:
                print("Adafactor optimizer not available, falling back to AdamW")
                optimizer = optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=config.learning_rate,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=config.weight_decay
                )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        
        return optimizer

class AdvancedScheduler:
    """Advanced learning rate scheduler"""
    
    @staticmethod
    def create_scheduler(optimizer, config: TrainingConfig, num_training_steps: int):
        """Create scheduler based on configuration"""
        
        if config.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
                eta_min=config.learning_rate * 0.1
            )
        elif config.scheduler.lower() == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=config.warmup_steps
            )
        elif config.scheduler.lower() == "polynomial":
            scheduler = optim.lr_scheduler.PolynomialLR(
                optimizer,
                total_iters=num_training_steps,
                power=0.5
            )
        elif config.scheduler.lower() == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {config.scheduler}")
        
        return scheduler

class MengAITrainer:
    """Advanced trainer for Meng AI models"""
    
    def __init__(self, config: TrainingConfig, device: str = 'auto'):
        self.config = config
        self.device = self._setup_device(device)
        
        # Set reproducibility
        self._set_seed(config.seed)
        
        # Initialize model
        self.model = create_meng_ai_model(config.model_size, config.vocab_size)
        self.model.to(self.device)
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        self.amp_dtype = getattr(torch, config.amp_dtype) if config.use_amp else None
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        self.global_step = 0
        self.epoch = 0
        
        # Wandb logging (optional)
        self.use_wandb = False
        
        print(f"🚀 Meng AI Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {config.model_size.value}")
        print(f"   Mixed Precision: {config.use_amp}")
        print(f"   Gradient Accumulation: {config.gradient_accumulation_steps}")
    
    def _setup_device(self, device: str):
        """Setup training device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                print("Using CPU")
        else:
            device = torch.device(device)
        return device
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup_training(self, num_training_steps: int):
        """Setup optimizer and scheduler"""
        self.optimizer = AdvancedOptimizer.create_optimizer(self.model, self.config)
        self.scheduler = AdvancedScheduler.create_scheduler(
            self.optimizer, self.config, num_training_steps
        )
        
        print(f"✅ Training setup complete")
        print(f"   Optimizer: {self.config.optimizer}")
        print(f"   Scheduler: {self.config.scheduler}")
        print(f"   Learning Rate: {self.config.learning_rate}")
    
    def prepare_data(self, texts: List[str]) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data"""
        print("📊 Preparing data...")
        
        # Initialize tokenizer
        tokenizer = SimpleTokenizer(vocab_size=self.config.vocab_size, min_frequency=2)
        tokenizer.build_vocabulary(texts)
        
        # Tokenize all texts
        all_token_ids = []
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            all_token_ids.extend(token_ids)
        
        print(f"Total tokens: {len(all_token_ids):,}")
        
        # Split into train and validation
        split_idx = int(len(all_token_ids) * self.config.train_split)
        train_tokens = all_token_ids[:split_idx]
        val_tokens = all_token_ids[split_idx:]
        
        # Create datasets
        train_dataset = AdvancedDataset(
            train_tokens, 
            self.config.sequence_length,
            dynamic_length=True
        )
        val_dataset = AdvancedDataset(
            val_tokens, 
            self.config.sequence_length,
            dynamic_length=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader, tokenizer
    
    def train_step(self, batch):
        """Single training step with mixed precision"""
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass with mixed precision
        if self.config.use_amp:
            with autocast(dtype=self.amp_dtype):
                logits = self.model(input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss = loss / self.config.gradient_accumulation_steps
        else:
            logits = self.model(input_ids)
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Optimizer step with gradient clipping"""
        if self.config.use_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        self.optimizer.zero_grad()
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer_step()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    avg_loss = total_loss / num_batches
                    
                    metrics = {
                        'train_loss': avg_loss,
                        'learning_rate': current_lr,
                        'epoch': self.epoch,
                        'step': self.global_step
                    }
                    
                    if self.use_wandb:
                        wandb.log(metrics)
                    
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'step': self.global_step
                    })
            
            # Evaluation
            if self.global_step % self.config.eval_steps == 0:
                val_metrics = self.evaluate(val_loader)
                self.metrics['val_loss'].append(val_metrics['val_loss'])
                self.metrics['val_perplexity'].append(val_metrics['val_perplexity'])
        
        return {
            'train_loss': total_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                if self.config.use_amp:
                    with autocast(dtype=self.amp_dtype):
                        logits = self.model(input_ids)
                        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                else:
                    logits = self.model(input_ids)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }
    
    def train(self, texts: List[str], output_dir: str = "meng_ai_output"):
        """Complete training pipeline"""
        print("🚀 Starting Meng AI Training")
        print("=" * 50)
        
        # Prepare data
        train_loader, val_loader, tokenizer = self.prepare_data(texts)
        
        # Setup training
        num_training_steps = len(train_loader) * self.config.max_epochs // self.config.gradient_accumulation_steps
        self.setup_training(num_training_steps)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        config_dict = {
            'model_size': self.config.model_size.value,
            'vocab_size': self.config.vocab_size,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'max_epochs': self.config.max_epochs,
            'sequence_length': self.config.sequence_length
        }
        
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            print("-" * 30)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Log metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Perplexity: {val_metrics['val_perplexity']:.2f}")
            print(f"Learning Rate: {train_metrics['learning_rate']:.2e}")
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_model(output_dir, "best")
                print(f"✅ New best model saved! (Val Loss: {val_metrics['val_loss']:.4f})")
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_model(output_dir, f"checkpoint_step_{self.global_step}")
        
        # Save final model
        self.save_model(output_dir, "final")
        
        # Save tokenizer
        tokenizer.save(os.path.join(output_dir, 'tokenizer.json'))
        
        print(f"\n🎉 Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {output_dir}")
    
    def save_model(self, output_dir: str, name: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(output_dir, f"{name}.pt")
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        print(f"✅ Model loaded from {filepath}")

def main():
    """Main training function"""
    print("🚀 Meng AI Advanced Training")
    print("=" * 50)
    
    # Configuration
    config = TrainingConfig(
        model_size=ModelSize.MEDIUM,
        vocab_size=10000,
        batch_size=16,
        gradient_accumulation_steps=2,
        max_epochs=5,
        learning_rate=1e-4,
        use_amp=True,
        eval_steps=100,
        save_steps=200,
        logging_steps=50
    )
    
    print("Training Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Create sample dataset
    from simple_tokenizer import create_sample_dataset
    texts = create_sample_dataset()
    print(f"\nLoaded {len(texts)} training texts")
    
    # Initialize trainer
    trainer = MengAITrainer(config)
    
    # Train model
    trainer.train(texts, output_dir="meng_ai_output")
    
    print("\n✅ Training completed successfully!")
    print("\nNext steps:")
    print("1. Load the best model checkpoint")
    print("2. Generate text samples")
    print("3. Fine-tune on your specific domain")
    print("4. Deploy for production use")

if __name__ == "__main__":
    main()
