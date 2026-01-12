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
# 1. Install Python 3.10 venv
# -----------------------------------------------------
echo ""
echo "[1/6] Installing python3.10-venv..."
apt update -y
apt install python3.10-venv -y

# -----------------------------------------------------
# 2. Create Virtual Environment
# -----------------------------------------------------
echo ""
echo "[2/6] Creating virtual environment with Python 3.10..."
python3.10 -m venv venv_tts
source venv_tts/bin/activate

echo "✓ Virtual environment created: venv_tts"
echo "✓ Python: $(python --version)"
echo "✓ Path: $(which python)"

# Upgrade pip
pip install --upgrade pip

# -----------------------------------------------------
# 3. Install System Dependencies
# -----------------------------------------------------
echo ""
echo "[3/6] Installing system dependencies..."
apt install -y \
    libsndfile1 \
    libsndfile1-dev \
    espeak-ng \
    espeak-ng-data \
    git \
    ffmpeg \
    wget \
    build-essential

# -----------------------------------------------------
# 4. Install Python Packages
# -----------------------------------------------------
echo ""
echo "[4/6] Installing Python packages..."

# Core packages
pip install pandas librosa soundfile tqdm numpy scipy tensorboard

# HuggingFace Hub
pip install huggingface_hub

# TorchCodec (required by torchaudio for audio loading)
pip install torchcodec

# Coqui TTS (case sensitive!)
pip install TTS

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import TTS; print(f'Coqui TTS: {TTS.__version__}')"
python -c "import torchcodec; print('TorchCodec: OK')"

# -----------------------------------------------------
# 5. Download MMS Model from HuggingFace
# -----------------------------------------------------
echo ""
echo "[5/6] Downloading MMS Karakalpak model from HuggingFace..."
python download_hf_model.py

# -----------------------------------------------------
# 6. Verify Setup
# -----------------------------------------------------
echo ""
echo "[6/6] Final verification..."

python << 'EOF'
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

import os
if os.path.exists("mms_kaa_hf/vocab.json"):
    import json
    with open("mms_kaa_hf/vocab.json") as f:
        vocab = json.load(f)
    print(f"MMS Vocabulary: {len(vocab)} characters ✓")
else:
    print("⚠️  MMS model not downloaded yet")
EOF

# Check model files
echo ""
if [ -f "mms_kaa_hf/pytorch_model.bin" ]; then
    echo "✓ Model: mms_kaa_hf/pytorch_model.bin"
fi
if [ -f "mms_kaa_hf/config.json" ]; then
    echo "✓ Config: mms_kaa_hf/config.json"
fi
if [ -f "mms_kaa_hf/vocab.json" ]; then
    echo "✓ Vocab: mms_kaa_hf/vocab.json (47 chars)"
fi

# -----------------------------------------------------
# Done!
# -----------------------------------------------------
echo ""
echo "==========================================="
echo "   ✅ Setup Complete!"
echo "==========================================="
echo ""
echo "IMPORTANT: Activate environment before each session:"
echo "  source venv_tts/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Edit prepare_dataset.py with your HuggingFace token"
echo "  2. python prepare_dataset.py"
echo "  3. python quick_fix_vocab.py  # CRITICAL: fixes punctuation"
echo "  4. python train.py"
echo ""
