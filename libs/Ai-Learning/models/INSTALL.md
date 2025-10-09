# Installation Guide 🚀

## Quick Start

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Demo
```bash
python demo_llm.py
```

### 3. Train Your Own Model
```bash
python train_llm.py
```

## Detailed Installation

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **GPU**: Optional but recommended for faster training

### Step-by-Step Installation

#### 1. Clone or Download
```bash
# If you have git
git clone <repository-url>
cd Ai-Learning/models

# Or download and extract the files
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv llm_env

# Activate virtual environment
# On Windows:
llm_env\Scripts\activate
# On macOS/Linux:
source llm_env/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
python test_structure.py
```

### GPU Support (Optional)

For faster training, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'torch'
```bash
pip install torch torchvision torchaudio
```

#### 2. CUDA out of memory
- Reduce batch size in training configuration
- Use smaller model dimensions
- Enable gradient checkpointing

#### 3. Slow training
- Use GPU if available
- Reduce model size for faster training
- Use smaller sequence lengths

#### 4. Import errors
```bash
# Make sure you're in the correct directory
cd libs/Ai-Learning/models

# Check Python path
python -c "import sys; print(sys.path)"
```

### Performance Tips

#### For CPU Training
- Use smaller models (d_model=128, num_layers=3)
- Reduce batch size (8-16)
- Use shorter sequences (64-128 tokens)

#### For GPU Training
- Use larger models (d_model=256, num_layers=6)
- Increase batch size (32-64)
- Use longer sequences (128-256 tokens)

## Next Steps

After successful installation:

1. **Run the demo**: `python demo_llm.py`
2. **Read the code**: Start with `simple_llm.py`
3. **Train your model**: `python train_llm.py`
4. **Experiment**: Modify parameters and see the results
5. **Scale up**: Train on larger datasets

## Getting Help

If you encounter issues:

1. Check the error messages carefully
2. Verify your Python version: `python --version`
3. Check if all dependencies are installed: `pip list`
4. Try running the test script: `python test_structure.py`
5. Check the README.md for more information

## Happy Learning! 🎓

You're now ready to build and train your own language model!
