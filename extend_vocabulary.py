#!/usr/bin/env python3
"""
MMS vocabulary'ni extend qilish - Karakalpak harflarini qo'shish
"""

import os
import torch
from collections import Counter

TRAIN_FILE = "./my_dataset/train.txt"
VOCAB_FILE = "mms_kaa/vocab.txt"
CHECKPOINT_FILE = "mms_kaa/G_100000.pth"

print("=" * 60)
print("VOCABULARY EXTENDER")
print("=" * 60)

# 1. Hozirgi vocabulary'ni o'qish
print("\n1. Hozirgi vocabulary o'qilmoqda...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    current_vocab = [line.strip() for line in f if line.strip()]

print(f"   Hozirgi harflar soni: {len(current_vocab)}")
print(f"   Birinchi 20 ta: {''.join(current_vocab[:20])}")

# 2. Dataset'dagi barcha harflarni topish
print("\n2. Dataset'dagi barcha harflar topilmoqda...")
all_chars = Counter()

with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) == 3:
            text = parts[2]
            for char in text:
                all_chars[char] += 1

print(f"   Dataset'da {len(all_chars)} xil harf topildi")

# 3. Yo'q bo'lgan harflarni topish
missing_chars = []
for char in all_chars:
    if char not in current_vocab:
        missing_chars.append((char, all_chars[char]))

missing_chars.sort(key=lambda x: x[1], reverse=True)

print(f"\n3. Vocabulary'da yo'q bo'lgan harflar:")
print(f"   Jami: {len(missing_chars)} ta harf")
if missing_chars:
    print(f"   Eng ko'p ishlatiladiganlar:")
    for char, count in missing_chars[:30]:
        if char.isprintable():
            print(f"      '{char}': {count} marta")
        else:
            print(f"      [U+{ord(char):04X}]: {count} marta")

# 4. Yangi vocabulary yaratish
if missing_chars:
    print(f"\n4. Yangi vocabulary yaratilmoqda...")

    new_vocab = current_vocab.copy()
    new_chars = [char for char, _ in missing_chars]
    new_vocab.extend(new_chars)

    # Backup yaratish
    backup_file = VOCAB_FILE + ".backup"
    with open(backup_file, 'w', encoding='utf-8') as f:
        for char in current_vocab:
            f.write(char + '\n')
    print(f"   ✓ Backup yaratildi: {backup_file}")

    # Yangi vocabulary'ni yozish
    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        for char in new_vocab:
            f.write(char + '\n')

    print(f"   ✓ Yangi vocabulary yozildi: {len(new_vocab)} ta harf")
    print(f"   ✓ Qo'shildi: {len(new_chars)} ta yangi harf")

    # 5. Checkpoint'ni yangilash (embedding layer resize)
    print(f"\n5. Checkpoint yangilanmoqda...")
    print(f"   OGOHLANTIRISH: Bu qadam vaqt talab qiladi...")

    checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu')

    # text_encoder embedding'ni topish
    embedding_key = None
    for key in checkpoint.keys():
        if 'emb' in key.lower() or 'embedding' in key.lower():
            if 'text' in key or 'enc_p' in key or 'encoder' in key:
                embedding_key = key
                break

    if embedding_key:
        old_embedding = checkpoint[embedding_key]
        old_vocab_size, embedding_dim = old_embedding.shape
        new_vocab_size = len(new_vocab)

        print(f"   Topilgan embedding: {embedding_key}")
        print(f"   Eski o'lcham: [{old_vocab_size}, {embedding_dim}]")
        print(f"   Yangi o'lcham: [{new_vocab_size}, {embedding_dim}]")

        # Yangi embedding yaratish
        new_embedding = torch.randn(new_vocab_size, embedding_dim) * 0.01
        new_embedding[:old_vocab_size] = old_embedding  # Eski ma'lumotlarni saqlash

        # Yangi checkpoint ga yozish
        checkpoint[embedding_key] = new_embedding

        # Backup checkpoint
        backup_ckpt = CHECKPOINT_FILE + ".backup"
        if not os.path.exists(backup_ckpt):
            torch.save(torch.load(CHECKPOINT_FILE), backup_ckpt)
            print(f"   ✓ Checkpoint backup: {backup_ckpt}")

        # Yangi checkpoint'ni saqlash
        torch.save(checkpoint, CHECKPOINT_FILE)
        print(f"   ✓ Checkpoint yangilandi!")
        print(f"   ✓ Yangi harflar uchun {len(new_chars)} ta embedding qo'shildi")
    else:
        print(f"   ✗ XATO: Text embedding topilmadi!")
        print(f"   Mavjud kalitlar:")
        for key in list(checkpoint.keys())[:10]:
            print(f"      - {key}")
        print("\n   Checkpoint'ni qo'lda yangilash kerak bo'lishi mumkin.")
else:
    print("\n✓ Barcha harflar vocabulary'da mavjud!")
    print("  Hech narsa qo'shish kerak emas.")

print("\n" + "=" * 60)
print("TAYYOR!")
print("=" * 60)
print("\nEndi training'ni qayta boshlang:")
print("  bash clear_all_cache.sh")
print("  python train.py")
