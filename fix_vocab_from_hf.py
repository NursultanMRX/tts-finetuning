#!/usr/bin/env python3
"""
Hugging Face vocab.json dan to'g'ri vocab.txt yaratish
"""

import json
import os

VOCAB_JSON = {
    " ": 46,
    "-": 40,
    "|": 0,
    "μ": 32,
    "а": 1,
    "б": 12,
    "в": 39,
    "г": 26,
    "д": 8,
    "е": 2,
    "ж": 24,
    "з": 18,
    "и": 3,
    "й": 13,
    "к": 19,
    "л": 6,
    "м": 11,
    "н": 5,
    "о": 14,
    "п": 16,
    "р": 7,
    "с": 9,
    "т": 10,
    "у": 17,
    "ф": 42,
    "х": 30,
    "ц": 45,
    "ш": 20,
    "ы": 4,
    "ь": 41,
    "э": 44,
    "ю": 43,
    "я": 34,
    "ѓ": 23,
    "є": 25,
    "њ": 21,
    "ќ": 15,
    "ў": 22,
    "ѳ": 36,
    "ғ": 33,
    "қ": 29,
    "ң": 31,
    "ү": 28,
    "ҳ": 27,
    "ә": 37,
    "ө": 38,
    "–": 35
}

VOCAB_FILE = "mms_kaa/vocab.txt"

print("=" * 60)
print("HUGGING FACE VOCABULARY FIX")
print("=" * 60)

# Index bo'yicha sort qilish
vocab_sorted = sorted(VOCAB_JSON.items(), key=lambda x: x[1])

print(f"\nJami harflar: {len(vocab_sorted)}")
print(f"\nBirinchi 10 ta:")
for char, idx in vocab_sorted[:10]:
    if char == ' ':
        print(f"  [{idx}]: [SPACE]")
    elif char == '\n':
        print(f"  [{idx}]: [NEWLINE]")
    else:
        print(f"  [{idx}]: {char}")

print(f"\nKarakalpak harflari:")
karakalpak_chars = ['ғ', 'қ', 'ң', 'ү', 'ҳ', 'ә', 'ө', 'ў']
for char in karakalpak_chars:
    if char in VOCAB_JSON:
        print(f"  [{VOCAB_JSON[char]}]: {char} ✓")
    else:
        print(f"  ✗ {char} topilmadi!")

# Backup yaratish
if os.path.exists(VOCAB_FILE):
    backup_file = VOCAB_FILE + ".old_wrong"
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        old_content = f.read()
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(old_content)
    print(f"\n✓ Eski vocab.txt backup: {backup_file}")

# Yangi vocab.txt yaratish
with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
    for char, idx in vocab_sorted:
        f.write(char + '\n')

print(f"✓ Yangi vocab.txt yaratildi: {VOCAB_FILE}")
print(f"✓ {len(vocab_sorted)} ta harf yozildi")

# Tekshirish
print(f"\n" + "=" * 60)
print("TEKSHIRISH")
print("=" * 60)

with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Jami qatorlar: {len(lines)}")
print(f"Birinchi 10 qator:")
for i, line in enumerate(lines[:10]):
    char = line.strip()
    if char == '':
        print(f"  [{i}]: [SPACE]")
    else:
        print(f"  [{i}]: {char}")

# Karakalpak harflarini tekshirish
print(f"\nKarakalpak harflari mavjudmi?")
content = ''.join([line.strip() for line in lines])
for char in karakalpak_chars:
    if char in content:
        print(f"  ✓ {char} - mavjud")
    else:
        print(f"  ✗ {char} - YO'Q!")

print("\n" + "=" * 60)
print("TAYYOR!")
print("=" * 60)
print("\nEndi:")
print("  1. Training'ni to'xtating (Ctrl+C)")
print("  2. Cache'ni tozalang: bash clear_all_cache.sh")
print("  3. Training'ni qayta boshlang: python train.py")
