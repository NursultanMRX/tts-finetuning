#!/bin/bash
# =====================================================
# MMS-TTS Karakalpak Fine-tuning Setup Script
# For Vast.ai environment with NVIDIA RTX 3090/4090
# =====================================================

set -e  # Exit on any error

echo "==========================================="
echo "   MMS-TTS Karakalpak Fine-tuning Setup"
echo "==========================================="

# -----------------------------------------------------
# 1. System Update
# -----------------------------------------------------
echo "[1/5] Updating system packages..."
apt-get update -y
apt-get upgrade -y

# -----------------------------------------------------
# 2. Install System Dependencies
# -----------------------------------------------------
echo "[2/5] Installing system dependencies..."
apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    espeak-ng \
    espeak-ng-data \
    git \
    ffmpeg \
    wget \
    tar \
    build-essential

# Verify espeak-ng installation
echo "Verifying espeak-ng installation..."
espeak-ng --version

# -----------------------------------------------------
# 3. Install Python Requirements
# -----------------------------------------------------
echo "[3/5] Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Install specific PyTorch version for CUDA 11.8
# Uncomment if needed:
# pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# -----------------------------------------------------
# 4. Download MMS Karakalpak Checkpoint
# -----------------------------------------------------
echo "[4/5] Downloading MMS Karakalpak checkpoint..."

# Create directory for MMS model
mkdir -p mms_kaa

# Download the official Facebook MMS checkpoint for Karakalpak (kaa)
MMS_URL="https://dl.fbaipublicfiles.com/mms/tts/kaa.tar.gz"
CHECKPOINT_FILE="kaa.tar.gz"

echo "Downloading from: ${MMS_URL}"
wget -O ${CHECKPOINT_FILE} ${MMS_URL}

# Extract the checkpoint
echo "Extracting checkpoint..."
tar -xzf ${CHECKPOINT_FILE} -C mms_kaa

# Clean up the tar file
rm ${CHECKPOINT_FILE}

# List extracted files
echo "Extracted files in mms_kaa/:"
ls -la mms_kaa/

# -----------------------------------------------------
# 5. Verify Installation
# -----------------------------------------------------
echo "[5/5] Verifying installation..."

# Check Python packages
python -c "import TTS; print(f'TTS version: {TTS.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import librosa; print(f'Librosa version: {librosa.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"

# Check for checkpoint files
if [ -f "mms_kaa/G_100000.pth" ]; then
    echo "✓ Generator checkpoint found: mms_kaa/G_100000.pth"
elif [ -f "mms_kaa/G_50000.pth" ]; then
    echo "✓ Generator checkpoint found: mms_kaa/G_50000.pth"
else
    echo "⚠ Searching for generator checkpoint..."
    find mms_kaa -name "G_*.pth" -o -name "*.pth"
fi

if [ -f "mms_kaa/config.json" ]; then
    echo "✓ Config file found: mms_kaa/config.json"
fi

if [ -f "mms_kaa/vocab.txt" ]; then
    echo "✓ Vocab file found: mms_kaa/vocab.txt"
fi

echo ""
echo "==========================================="
echo "   Setup Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Run: python prepare_dataset.py"
echo "  2. Run: python train.py"
echo ""
