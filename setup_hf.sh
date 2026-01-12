#!/bin/bash
# =====================================================
# MMS-TTS Fine-tuning using HuggingFace finetune-hf-vits
# This is the OFFICIAL approach for fine-tuning MMS-TTS
# =====================================================

set -e

echo "==========================================="
echo "MMS-TTS Fine-tuning Setup (HuggingFace)"
echo "==========================================="

# -----------------------------------------------------
# 1. Clone finetune-hf-vits repo
# -----------------------------------------------------
echo ""
echo "[1/5] Cloning finetune-hf-vits repository..."

if [ -d "finetune-hf-vits" ]; then
    echo "  -> Directory exists, pulling latest..."
    cd finetune-hf-vits
    git pull
    cd ..
else
    git clone https://github.com/ylacombe/finetune-hf-vits.git
fi

# -----------------------------------------------------
# 2. Install requirements
# -----------------------------------------------------
echo ""
echo "[2/5] Installing requirements..."
cd finetune-hf-vits
pip install -r requirements.txt
pip install accelerate

# -----------------------------------------------------
# 3. Build monotonic alignment
# -----------------------------------------------------
echo ""
echo "[3/5] Building monotonic alignment search..."
cd monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd ..

# -----------------------------------------------------
# 4. Convert Karakalpak (kaa) discriminator
# -----------------------------------------------------
echo ""
echo "[4/5] Converting Karakalpak MMS discriminator for training..."

# Create local folder for converted model
mkdir -p ../mms_kaa_train

# Convert the discriminator (required for fine-tuning)
python convert_original_discriminator_checkpoint.py \
    --language_code kaa \
    --pytorch_dump_folder_path ../mms_kaa_train

echo "  -> Converted model saved to ../mms_kaa_train"

# -----------------------------------------------------
# 5. Verify installation
# -----------------------------------------------------
echo ""
echo "[5/5] Verifying installation..."

python -c "
import torch
from transformers import VitsModel, VitsTokenizer

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')

# Load Karakalpak MMS
tokenizer = VitsTokenizer.from_pretrained('facebook/mms-tts-kaa')
print(f'Tokenizer vocab size: {len(tokenizer.get_vocab())}')
print('✓ HuggingFace MMS setup complete!')
"

cd ..

echo ""
echo "==========================================="
echo "   ✅ Setup Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Prepare your dataset in HuggingFace format"
echo "  2. Create a training config file"
echo "  3. Run: accelerate launch finetune-hf-vits/run_vits_finetuning.py config.json"
echo ""
