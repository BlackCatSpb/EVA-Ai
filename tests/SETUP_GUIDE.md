# EVA-AI System Requirements

## Python Version
**Python 3.12+** (Required for CUDA support)

## Quick Install

### Option 1: Automatic Setup
```batch
run_as_admin setup_eva.bat
```

### Option 2: Manual Setup

1. **Install Python 3.12**
   - Download: https://www.python.org/downloads/
   - During installation: Check "Add Python to PATH"

2. **Install CUDA 12.1** (Optional but recommended)
   - Download: https://developer.nvidia.com/cuda-downloads

3. **Install Dependencies**
   ```bash
   # PyTorch with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # OpenVINO
   pip install openvino openvino-genai
   
   # Core dependencies
   pip install transformers sentence-transformers numpy faiss-cpu safetensors
   pip install flask flask-cors nltk psutil Pillow matplotlib PyYAML
   ```

4. **Download spaCy Russian model**
   ```bash
   python -m spacy download ru_core_news_sm
   ```

## Required Packages

| Package | Version | Purpose |
|--------|--------|---------|
| torch | 2.5.1+cu121 | ML framework with CUDA |
| transformers | 4.30+ | HuggingFace models |
| sentence-transformers | 2.2+ | Embeddings |
| faiss-cpu | 1.7+ | Vector search |
| openvino-genai | latest | LLM inference |
| nltk | 3.8+ | NLP processing |
| flask | 3.0+ | Web GUI |

## Models

### LLM Models (GGUF)
Download to: `eva_pie_architecture/models/gguf_models/`

- **RuadaptQwen3-4B** (Q4_K_M.gguf) - ~2.4GB
  - Logic and Context model
  
### Embedding Models
Automatically downloaded to HF cache on first use:
- `intfloat/multilingual-e5-base` - ~1GB

### NLI Models
Automatically downloaded:
- `facebook/bart-large-mnli` - for contradiction detection

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| GPU VRAM | - | 2GB+ (for embeddings) |
| Storage | 10GB | 20GB+ |

## GPU Support

- **MX550**: Works with CUDA 12.1
- **CPU Fallback**: All features work without GPU

## Verification

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Troubleshooting

### "No module named 'torch'"
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### "OpenVINO GenAI not available"
```bash
pip install openvino openvino-genai
```

### "FAISS not available"
```bash
pip install faiss-cpu
```

## Startup

```bash
# Method 1: Batch file
start_eva.bat

# Method 2: Direct
python -m eva_ai
```
