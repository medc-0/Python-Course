"""
LLM Demo Script
===============

A simple demo script to quickly train and test the Simple LLM.
This script will:

1. Create a small dataset
2. Train a mini LLM (fast training)
3. Generate text samples
4. Show the model's capabilities

Run this script to see your LLM in action!

Author: AI Learning Course
"""

import torch
import torch.nn as nn
from simple_llm import SimpleLLM
from simple_tokenizer import SimpleTokenizer
from train_llm import LLMTrainer, TextDataset
from torch.utils.data import DataLoader
import time
import random

def create_demo_dataset():
    """Create a small demo dataset for quick training"""
    demo_texts = [
        "The cat sat on the mat and purred softly.",
        "Python programming is fun and powerful for data science.",
        "Machine learning algorithms can predict future trends.",
        "Artificial intelligence will transform many industries.",
        "Deep learning models understand complex patterns in data.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to see and interpret images.",
        "Data science combines statistics, programming, and domain knowledge.",
        "Neural networks are inspired by biological brain structures.",
        "Reinforcement learning teaches agents through trial and error.",
        "The future of technology lies in artificial intelligence.",
        "Programming requires both creativity and logical thinking.",
        "Open source software has revolutionized software development.",
        "Cloud computing provides scalable computing resources.",
        "Cybersecurity protects digital information and systems.",
        "Blockchain technology offers secure data storage solutions.",
        "The internet connects people and information globally.",
        "Mobile apps have changed how we use technology.",
        "User experience design focuses on intuitive interfaces.",
        "Software engineering builds reliable and maintainable systems.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables human-computer communication.",
        "Computer vision processes and analyzes visual information.",
        "Data mining extracts valuable insights from large datasets.",
        "Predictive analytics forecasts future events and trends.",
        "Big data technologies handle massive amounts of information.",
        "The internet of things connects everyday devices to networks.",
        "Quantum computing promises revolutionary computational power.",
        "Robotics combines mechanical engineering with artificial intelligence.",
        "Virtual reality creates immersive digital experiences.",
        "Augmented reality overlays digital information on real world.",
        "Cryptocurrency uses blockchain technology for digital transactions.",
        "Smart contracts automate agreements using blockchain technology.",
        "Edge computing processes data closer to the source.",
        "5G networks provide faster and more reliable connectivity.",
        "Autonomous vehicles use AI for self-driving capabilities.",
        "Smart cities integrate technology to improve urban living.",
        "Digital transformation modernizes business processes and operations."
    ]
    
    return demo_texts

def quick_train_demo():
    """Quick training demo with a small model"""
    print("🚀 Quick LLM Demo")
    print("=" * 40)
    
    # Configuration for quick training
    config = {
        'vocab_size': 1000,
        'd_model': 128,      # Small model for fast training
        'num_heads': 4,
        'num_layers': 3,     # Fewer layers for speed
        'd_ff': 256,
        'max_seq_length': 64,
        'sequence_length': 32,
        'batch_size': 8,
        'learning_rate': 1e-3,  # Higher learning rate for faster convergence
        'num_epochs': 3,        # Few epochs for demo
        'train_split': 0.8
    }
    
    print("Demo Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create dataset
    print(f"\n📚 Creating demo dataset...")
    texts = create_demo_dataset()
    print(f"Created {len(texts)} training texts")
    
    # Initialize tokenizer
    print(f"\n🔤 Building tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'], min_frequency=1)
    tokenizer.build_vocabulary(texts)
    
    # Show some tokenization examples
    sample_text = "Python programming is fun and powerful for data science."
    print(f"\nTokenization example:")
    print(f"Original: {sample_text}")
    token_ids = tokenizer.encode(sample_text)
    print(f"Token IDs: {token_ids}")
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: {decoded}")
    
    # Initialize model
    print(f"\n🧠 Creating model...")
    model = SimpleLLM(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length']
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created with {param_count:,} parameters")
    
    # Initialize trainer
    trainer = LLMTrainer(model, tokenizer)
    
    # Prepare data
    print(f"\n📊 Preparing data...")
    train_loader, val_loader = trainer.prepare_data(
        texts, 
        sequence_length=config['sequence_length'],
        train_split=config['train_split'],
        batch_size=config['batch_size']
    )
    
    # Setup optimizer
    trainer.setup_optimizer(learning_rate=config['learning_rate'])
    
    # Quick training
    print(f"\n🏋️ Starting quick training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train one epoch
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Generate sample after each epoch
        print(f"\nGenerated text after epoch {epoch + 1}:")
        sample = trainer.generate_sample("The future", max_length=25, temperature=0.8)
        print(f"'{sample}'")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return trainer

def interactive_generation(trainer):
    """Interactive text generation"""
    print(f"\n🎯 Interactive Text Generation")
    print("=" * 40)
    print("Enter prompts to generate text (type 'quit' to exit)")
    print("Try prompts like: 'The future of', 'Python is', 'Machine learning'")
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            # Generate text
            print("Generating...")
            generated = trainer.generate_sample(
                prompt, 
                max_length=50, 
                temperature=0.8
            )
            
            print(f"Generated: '{generated}'")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye! 👋")

def test_different_temperatures(trainer):
    """Test generation with different temperatures"""
    print(f"\n🌡️ Testing Different Temperatures")
    print("=" * 40)
    
    prompt = "The future of"
    temperatures = [0.1, 0.5, 0.8, 1.2]
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        generated = trainer.generate_sample(prompt, max_length=30, temperature=temp)
        print(f"Generated: '{generated}'")

def test_different_prompts(trainer):
    """Test generation with different prompts"""
    print(f"\n💭 Testing Different Prompts")
    print("=" * 40)
    
    prompts = [
        "The future of",
        "Python is",
        "Machine learning",
        "Artificial intelligence",
        "The cat",
        "Programming",
        "Data science",
        "Technology will"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = trainer.generate_sample(prompt, max_length=35, temperature=0.8)
        print(f"Generated: '{generated}'")

def main():
    """Main demo function"""
    print("🎉 Welcome to the Simple LLM Demo!")
    print("This demo will train a small language model and show you how it works.")
    print()
    
    # Quick training
    trainer = quick_train_demo()
    
    # Test different scenarios
    test_different_temperatures(trainer)
    test_different_prompts(trainer)
    
    # Interactive generation
    interactive_generation(trainer)
    
    print(f"\n🎊 Demo completed!")
    print(f"\nWhat you've learned:")
    print(f"✅ How to build a transformer-based language model")
    print(f"✅ How to train it on text data")
    print(f"✅ How to generate text with different sampling strategies")
    print(f"✅ How temperature affects text generation")
    print(f"✅ How prompts influence the generated content")
    
    print(f"\nNext steps:")
    print(f"🔹 Train on larger datasets for better results")
    print(f"🔹 Experiment with different model architectures")
    print(f"🔹 Try fine-tuning on specific domains")
    print(f"🔹 Implement more advanced sampling techniques")
    print(f"🔹 Add attention visualization")
    print(f"🔹 Build a web interface for text generation")

if __name__ == "__main__":
    main()
