# Simple LLM from Scratch 🚀

A fully functional Large Language Model (LLM) built from scratch using PyTorch. This implementation includes all the essential components of modern language models:

- **Multi-Head Self-Attention** mechanism
- **Transformer architecture** with residual connections
- **Positional encoding** for sequence understanding
- **Text generation** with sampling strategies
- **Complete training pipeline** with optimization

## 🎯 What You'll Learn

- How transformer architectures work internally
- How attention mechanisms enable language understanding
- How to train language models from scratch
- How to generate coherent text with different sampling strategies
- How to build a complete ML pipeline

## 📁 Files Overview

### Core Components
- **`simple_llm.py`** - The main transformer model implementation
- **`simple_tokenizer.py`** - Text preprocessing and tokenization
- **`train_llm.py`** - Complete training pipeline with optimization
- **`demo_llm.py`** - Quick demo script to see the model in action

### Configuration
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This documentation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Demo
```bash
python demo_llm.py
```

This will:
- Train a small LLM on sample data
- Generate text samples
- Show interactive text generation
- Demonstrate different temperature settings

### 3. Train Your Own Model
```bash
python train_llm.py
```

## 🧠 Model Architecture

### Transformer Components

1. **Multi-Head Self-Attention**
   - Allows the model to focus on different parts of the input
   - Uses scaled dot-product attention
   - Includes causal masking for language modeling

2. **Feed-Forward Networks**
   - Position-wise fully connected layers
   - Applied to each position independently
   - Uses ReLU activation

3. **Layer Normalization & Residual Connections**
   - Stabilizes training
   - Enables deeper networks
   - Improves gradient flow

4. **Positional Encoding**
   - Adds position information to embeddings
   - Uses sinusoidal functions
   - Enables sequence understanding

### Model Configuration

```python
model = SimpleLLM(
    vocab_size=2000,      # Vocabulary size
    d_model=256,          # Model dimension
    num_heads=8,          # Number of attention heads
    num_layers=6,         # Number of transformer layers
    d_ff=1024,           # Feed-forward dimension
    max_seq_length=512,   # Maximum sequence length
    dropout=0.1          # Dropout rate
)
```

## 🔤 Tokenization

The tokenizer handles:
- Text preprocessing (lowercase, cleaning)
- Word-level tokenization
- Vocabulary building
- Special tokens (PAD, UNK, BOS, EOS)
- Encoding/decoding between text and token IDs

```python
tokenizer = SimpleTokenizer(vocab_size=10000, min_frequency=2)
tokenizer.build_vocabulary(training_texts)

# Encode text to token IDs
token_ids = tokenizer.encode("Hello world!")

# Decode token IDs back to text
text = tokenizer.decode(token_ids)
```

## 🏋️ Training

The training pipeline includes:
- **Data preparation** with sliding windows
- **Optimization** with AdamW optimizer
- **Learning rate scheduling** (cosine annealing)
- **Gradient clipping** for stability
- **Validation** and metrics tracking
- **Checkpointing** for model saving

### Training Configuration

```python
config = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'sequence_length': 128,
    'weight_decay': 0.01
}
```

## 🎲 Text Generation

The model supports various generation strategies:

### Sampling Parameters
- **Temperature**: Controls randomness (0.1 = deterministic, 1.0 = random)
- **Top-k**: Consider only top-k most likely tokens
- **Top-p**: Consider tokens with cumulative probability up to top_p
- **Greedy**: Always choose the most likely token

### Example Generation

```python
# Generate with different temperatures
generated = model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,    # Balanced creativity
    top_k=50,          # Consider top 50 tokens
    top_p=0.9,         # Nucleus sampling
    do_sample=True     # Use sampling
)
```

## 📊 Results and Metrics

The model tracks:
- **Training loss** (Cross-entropy)
- **Validation loss** (Perplexity)
- **Learning rate** schedule
- **Generation quality** (qualitative)

### Expected Performance
- **Small model** (128 dim, 3 layers): ~2-4 hours training
- **Medium model** (256 dim, 6 layers): ~8-12 hours training
- **Large model** (512 dim, 12 layers): ~1-2 days training

## 🎯 Use Cases

This LLM can be used for:
- **Text completion** and continuation
- **Creative writing** assistance
- **Code generation** (with appropriate training)
- **Question answering** (with fine-tuning)
- **Language understanding** research
- **Educational purposes** (learning transformers)

## 🔧 Customization

### Training on Your Data
1. Replace the sample dataset with your text data
2. Adjust vocabulary size based on your corpus
3. Tune hyperparameters for your use case
4. Monitor training metrics

### Model Scaling
- Increase `d_model` for more capacity
- Add more `num_layers` for depth
- Use more `num_heads` for attention diversity
- Adjust `d_ff` for feed-forward capacity

### Advanced Features
- Add **beam search** for better generation
- Implement **attention visualization**
- Add **model parallelism** for larger models
- Include **mixed precision** training

## 🚨 Limitations

This is a simplified implementation:
- **Small vocabulary** compared to production models
- **Limited context length** (512 tokens)
- **Basic tokenization** (word-level, not subword)
- **No pre-training** on large corpora
- **Single GPU** training only

## 🎓 Learning Path

1. **Start with the demo** (`demo_llm.py`)
2. **Understand the architecture** (`simple_llm.py`)
3. **Learn tokenization** (`simple_tokenizer.py`)
4. **Study training** (`train_llm.py`)
5. **Experiment with parameters**
6. **Train on your own data**
7. **Implement improvements**

## 🤝 Contributing

Feel free to:
- Add new features
- Improve documentation
- Fix bugs
- Share your results
- Suggest improvements

## 📚 Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [Training Tips for Large Language Models](https://arxiv.org/abs/2303.14374) - Modern training practices

## 🎉 Have Fun!

This is your journey into understanding how modern language models work. Experiment, learn, and most importantly - have fun building AI! 🚀

---

**Happy Coding!** 👨‍💻👩‍💻
