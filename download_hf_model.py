#!/usr/bin/env python3
"""
Hugging Face dan to'g'ri MMS modelini yuklab olish
"""

from huggingface_hub import snapshot_download
import json
import os
import glob

HF_REPO = "facebook/mms-tts-kaa"
LOCAL_DIR = "./mms_kaa_hf"

print("=" * 60)
print("HUGGING FACE MODELINI YUKLAB OLISH")
print("=" * 60)

# 1. Modelni yuklab olish
print(f"\n1. {HF_REPO} yuklanmoqda...")
try:
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=LOCAL_DIR,
        repo_type="model"
    )
    print(f"✓ Model yuklandi: {LOCAL_DIR}")
except Exception as e:
    print(f"✗ XATO: {e}")
    exit(1)

# 2. Yuklangan fayllarni ko'rsatish
print(f"\n2. Yuklangan fayllar:")
for root, dirs, files in os.walk(LOCAL_DIR):
    for file in files:
        filepath = os.path.join(root, file)
        filesize = os.path.getsize(filepath) / 1024 / 1024  # MB
        print(f"  - {filepath} ({filesize:.2f} MB)")

# 3. vocab.json dan vocab.txt yaratish
vocab_json_path = os.path.join(LOCAL_DIR, "vocab.json")
vocab_txt_path = os.path.join(LOCAL_DIR, "vocab.txt")

if os.path.exists(vocab_json_path):
    print(f"\n3. vocab.txt yaratilmoqda...")
    with open(vocab_json_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # Index bo'yicha sort
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])

    print(f"  Vocabulary: {len(vocab_sorted)} ta harf")

    # Karakalpak harflarini tekshirish
    karakalpak_chars = ['ғ', 'қ', 'ң', 'ү', 'ҳ', 'ә', 'ө', 'ў']
    print(f"  Karakalpak harflari:")
    for char in karakalpak_chars:
        char_dict = dict(vocab_sorted)
        if char in char_dict:
            print(f"    ✓ {char} (index: {char_dict[char]})")
        else:
            print(f"    ✗ {char} YO'Q!")

    # vocab.txt yaratish
    with open(vocab_txt_path, 'w', encoding='utf-8') as f:
        for char, _ in vocab_sorted:
            f.write(char + '\n')

    print(f"  ✓ vocab.txt yaratildi: {vocab_txt_path}")
else:
    print(f"\n✗ vocab.json topilmadi: {vocab_json_path}")

# 4. Checkpoint'ni topish
print(f"\n4. Checkpoint topish...")
ckpt_files = glob.glob(os.path.join(LOCAL_DIR, "*.pth"))
if ckpt_files:
    for ckpt in ckpt_files:
        size = os.path.getsize(ckpt) / 1024 / 1024
        print(f"  ✓ {ckpt} ({size:.2f} MB)")
else:
    print("  ✗ .pth checkpoint topilmadi!")
    # Boshqa formatlarni qidirish
    other_files = glob.glob(os.path.join(LOCAL_DIR, "*"))
    print(f"  Mavjud fayllar:")
    for f in other_files:
        if os.path.isfile(f):
            print(f"    - {os.path.basename(f)}")

# 5. config.json tekshirish
config_path = os.path.join(LOCAL_DIR, "config.json")
if os.path.exists(config_path):
    print(f"\n5. ✓ config.json topildi: {config_path}")
else:
    print(f"\n5. ✗ config.json topilmadi")

print("\n" + "=" * 60)
print("TRAIN.PY UCHUN YO'LLAR:")
print("=" * 60)

if ckpt_files:
    ckpt_name = os.path.basename(ckpt_files[0])
    print(f"""
MMS_CKPT = "mms_kaa_hf/{ckpt_name}"
MMS_CONFIG = "mms_kaa_hf/config.json"
VOCAB_FILE = "mms_kaa_hf/vocab.txt"
""")
else:
    print("\n✗ Checkpoint topilmadi, train.py ni yangilab bo'lmaydi")

print("=" * 60)
print("KEYINGI QADAMLAR:")
print("=" * 60)
print("""
1. train.py ni yangilang (yuqoridagi yo'llar bilan)
2. Cache'ni tozalang: bash clear_all_cache.sh
3. Training'ni boshlang: python train.py
""")
