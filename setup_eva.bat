@echo off
REM ==========================================
REM EVA-AI System Setup Script
REM ==========================================
REM Run as Administrator
REM ==========================================

echo ==========================================
echo EVA-AI System Setup
echo ==========================================
echo.

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Please install Python 3.12 from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, check "Add Python to PATH"
    pause
    exit /b 1
)

REM Get Python version
python --version

REM ==========================================
REM Step 1: Install OpenVINO
REM ==========================================
echo.
echo [Step 1/5] Installing OpenVINO...
pip install openvino openvino-genai

REM ==========================================
REM Step 2: Upgrade pip
REM ==========================================
echo.
echo [Step 2/5] Upgrading pip...
python -m pip install --upgrade pip

REM ==========================================
REM Step 3: Install PyTorch with CUDA
REM ==========================================
echo.
echo [Step 3/5] Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM ==========================================
REM Step 4: Install core dependencies
REM ==========================================
echo.
echo [Step 4/5] Installing core dependencies...
pip install transformers sentence-transformers numpy faiss-cpu safetensors tokenizers

REM ==========================================
REM Step 5: Install other dependencies
REM ==========================================
echo.
echo [Step 5/5] Installing other dependencies...
pip install flask flask-cors nltk psutil Pillow matplotlib PyYAML python-dotenv tqdm loguru pandas scikit-learn scipy pydantic pydantic-core

REM ==========================================
REM Download spaCy Russian model
REM ==========================================
echo.
echo [Extra] Downloading spaCy Russian model...
python -m spacy download ru_core_news_sm

REM ==========================================
REM Download NLTK data
REM ==========================================
echo.
echo [Extra] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Download GGUF models to eva_pie_architecture/models/gguf_models/
echo 2. Download embedding model (will be cached automatically)
echo 3. Run EVA: start_eva.bat
echo.
pause
