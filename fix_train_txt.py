#!/usr/bin/env python3
"""
train.txt faylini tozalash - barcha dublikatlarni olib tashlash
"""

import os

TRAIN_FILE = "./my_dataset/train.txt"

print("=" * 60)
print("train.txt tozalash boshlandi...")
print("=" * 60)

# train.txt ni o'qish
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Jami qatorlar: {len(lines)}")

# Tuzatilgan qatorlar
fixed_lines = []
errors_found = 0

for i, line in enumerate(lines, 1):
    line = line.strip()
    if not line:
        continue

    parts = line.split("|")
    if len(parts) != 3:
        print(f"XATO - Qator {i}: Noto'g'ri format (3 ustun emas): {line[:50]}...")
        continue

    file_path, speaker, text = parts
    original_path = file_path

    # Dublikatlarni tuzatish
    # 1. wavs/wavs/ -> wavs/
    if "wavs/wavs/" in file_path:
        file_path = file_path.replace("wavs/wavs/", "")
        errors_found += 1
        print(f"Tuzatildi (qator {i}): wavs/wavs/ -> [fayl nomi]")

    # 2. wavs/ prefiksini OLIB TASHLASH (formatter o'zi qo'shadi!)
    if file_path.startswith("wavs/"):
        file_path = file_path.replace("wavs/", "", 1)
        errors_found += 1
        print(f"Tuzatildi (qator {i}): wavs/ prefiksi olib tashlandi")

    # 3. .wav.wav -> .wav
    if file_path.endswith(".wav.wav"):
        file_path = file_path[:-4]  # Oxirgi .wav ni olib tashlash
        errors_found += 1
        print(f"Tuzatildi (qator {i}): .wav.wav -> .wav")

    # 4. .wav kengaytmasi borligini tekshirish
    if not file_path.endswith(".wav"):
        file_path += ".wav"
        errors_found += 1
        print(f"Tuzatildi (qator {i}): .wav kengaytmasi qo'shildi")

    if original_path != file_path:
        print(f"  {original_path} -> {file_path}")

    fixed_lines.append(f"{file_path}|{speaker}|{text}")

print("\n" + "=" * 60)
print(f"Jami tuzatishlar: {errors_found}")
print(f"Yakuniy qatorlar: {len(fixed_lines)}")
print("=" * 60)

# Backup yaratish
backup_file = TRAIN_FILE + ".backup"
if os.path.exists(TRAIN_FILE):
    with open(backup_file, "w", encoding="utf-8") as f:
        with open(TRAIN_FILE, "r", encoding="utf-8") as orig:
            f.write(orig.read())
    print(f"✓ Backup yaratildi: {backup_file}")

# Tuzatilgan versiyani yozish
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for line in fixed_lines:
        f.write(line + "\n")

print(f"✓ train.txt yangilandi: {TRAIN_FILE}")

# Tekshirish
print("\n" + "=" * 60)
print("YAKUNIY TEKSHIRISH:")
print("=" * 60)

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    content = f.read()

    wavs_wavs_count = content.count("wavs/wavs/")
    wav_wav_count = content.count(".wav.wav")
    wavs_prefix_count = content.count("\nwavs/") + (1 if content.startswith("wavs/") else 0)

    if wavs_wavs_count > 0:
        print(f"✗ XATO: Hali ham {wavs_wavs_count} ta 'wavs/wavs/' topildi!")
    else:
        print("✓ OK: 'wavs/wavs/' dublikatlari yo'q")

    if wav_wav_count > 0:
        print(f"✗ XATO: Hali ham {wav_wav_count} ta '.wav.wav' topildi!")
    else:
        print("✓ OK: '.wav.wav' dublikatlari yo'q")

    if wavs_prefix_count > 0:
        print(f"✗ XATO: Hali ham {wavs_prefix_count} ta 'wavs/' prefiksi bor!")
        print("   (ljspeech formatter o'zi qo'shadi, shuning uchun olib tashlash kerak)")
    else:
        print("✓ OK: 'wavs/' prefikslari yo'q (to'g'ri!)")

print("\n" + "=" * 60)
print("Birinchi 5 qator:")
print("=" * 60)
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i <= 5:
            print(f"{i}: {line.strip()[:80]}...")

print("\n✓ Tayyor! Endi 'python train.py' ishga tushiring.")
