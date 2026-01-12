#!/bin/bash
# =====================================================
# MMS-TTS Karakalpak Fine-tuning Setup Script
# For Vast.ai with PyTorch Template
# =====================================================

set -e  # Exit on any error

echo "==========================================="
echo "   MMS-TTS Karakalpak Fine-tuning Setup"
echo "==========================================="

# -----------------------------------------------------
# 0. Check Python Version
# -----------------------------------------------------
echo ""
echo "[0/6] Checking Python version..."
python3 --version

# Check if we're in the base conda environment or need venv
if command -v conda &> /dev/null; then
    echo "✓ Conda detected, using conda environment"
    USE_VENV=false
else
    echo "✓ Using Python venv"
    USE_VENV=true
fi

# -----------------------------------------------------
# 1. System Update & Dependencies
# -----------------------------------------------------
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update -y
apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    espeak-ng \
    espeak-ng-data \
    git \
    ffmpeg \
    wget \
    build-essential

# Verify espeak-ng
espeak-ng --version || echo "Warning: espeak-ng not fully installed"

# -----------------------------------------------------
# 2. Create Virtual Environment (if needed)
# -----------------------------------------------------
echo ""
echo "[2/6] Setting up Python environment..."

if [ "$USE_VENV" = true ]; then
    # Create virtual environment with python3
    python3 -m venv venv_tts
    source venv_tts/bin/activate
    echo "✓ Virtual environment created: venv_tts"
    echo "✓ Activated: $(which python)"
else
    echo "✓ Using existing conda/system Python"
fi

# Upgrade pip
python3 -m pip install --upgrade pip

# -----------------------------------------------------
# 3. Install Python Requirements
# -----------------------------------------------------
echo ""
echo "[3/6] Installing Python packages..."

# Install core packages
python3 -m pip install --upgrade \
    numpy \
    scipy \
    librosa \
    soundfile \
    pandas \
    tqdm \
    tensorboard

# Install Coqui TTS
python3 -m pip install TTS

# Install HuggingFace Hub
python3 -m pip install huggingface_hub

# Verify torch (should be pre-installed in PyTorch template)
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# -----------------------------------------------------
# 4. Download MMS Model from HuggingFace
# -----------------------------------------------------
echo ""
echo "[4/6] Downloading MMS model from HuggingFace..."

python3 download_hf_model.py

# -----------------------------------------------------
# 5. Verify Installation
# -----------------------------------------------------
echo ""
echo "[5/6] Verifying installation..."

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import TTS
print(f"Coqui TTS: {TTS.__version__}")

import librosa
print(f"Librosa: {librosa.__version__}")

import os
if os.path.exists("mms_kaa_hf/vocab.json"):
    import json
    with open("mms_kaa_hf/vocab.json") as f:
        vocab = json.load(f)
    print(f"MMS Vocabulary: {len(vocab)} characters")
EOF

# -----------------------------------------------------
# 6. Check MMS Model Files
# -----------------------------------------------------
echo ""
echo "[6/6] Checking MMS model files..."

if [ -f "mms_kaa_hf/pytorch_model.bin" ]; then
    echo "✓ Model weights: mms_kaa_hf/pytorch_model.bin"
else
    echo "✗ Model weights not found!"
fi

if [ -f "mms_kaa_hf/config.json" ]; then
    echo "✓ Config: mms_kaa_hf/config.json"
else
    echo "✗ Config not found!"
fi

if [ -f "mms_kaa_hf/vocab.json" ]; then
    echo "✓ Vocabulary: mms_kaa_hf/vocab.json"
else
    echo "✗ Vocabulary not found!"
fi

# -----------------------------------------------------
# Done!
# -----------------------------------------------------
echo ""
echo "==========================================="
echo "   ✅ Setup Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo ""
if [ "$USE_VENV" = true ]; then
    echo "  1. Activate environment:"
    echo "     source venv_tts/bin/activate"
    echo ""
fi
echo "  2. Edit prepare_dataset.py with your HuggingFace token:"
echo "     nano prepare_dataset.py"
echo ""
echo "  3. Download and prepare your dataset:"
echo "     python3 prepare_dataset.py"
echo ""
echo "  4. Clean vocabulary (IMPORTANT!):"
echo "     python3 quick_fix_vocab.py"
echo ""
echo "  5. Start training:"
echo "     python3 train.py"
echo ""
