#!/bin/bash
echo "============================================================"
echo "BARCHA CACHE FAYLLARINI TOZALASH"
echo "============================================================"

# 1. my_dataset ichidagi barcha cache fayllar
echo "1. my_dataset cache'ini tozalash..."
find my_dataset -name "*.pyc" -delete 2>/dev/null
find my_dataset -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find my_dataset -name ".preprocessed" -exec rm -rf {} + 2>/dev/null
find my_dataset -name "*.json" ! -name "metadata.csv" -delete 2>/dev/null
rm -rf my_dataset/.cache 2>/dev/null
rm -rf my_dataset/cache 2>/dev/null

# 2. output_finetune papkasi
echo "2. output_finetune tozalash..."
rm -rf output_finetune/* 2>/dev/null

# 3. Python cache
echo "3. Python cache tozalash..."
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 4. Formatter-specific cache
echo "4. Formatter cache tozalash..."
rm -rf my_dataset/metadata_train.txt 2>/dev/null
rm -rf my_dataset/metadata_val.txt 2>/dev/null

# 5. TTS cache (agar global cache bo'lsa)
echo "5. TTS global cache tozalash..."
rm -rf ~/.local/share/tts 2>/dev/null
rm -rf /tmp/tts_* 2>/dev/null

echo ""
echo "============================================================"
echo "✓ Barcha cache'lar tozalandi!"
echo "============================================================"
echo ""
echo "Endi train.py ni ishga tushiring:"
echo "  python train.py"
echo ""
