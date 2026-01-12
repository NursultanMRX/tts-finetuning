#!/usr/bin/env python3
"""
QUICK FIX: Immediately clean existing train.txt to fix CUDA index errors.

Run this on the server:
  cd /workspace/tts-finetuning
  python quick_fix_vocab.py
  python train.py
"""

import os
import re
import json

TRAIN_FILE = "./my_dataset/train.txt"
VOCAB_JSON = "mms_kaa_hf/vocab.json"

print("=" * 60)
print("QUICK VOCABULARY FIX")
print("=" * 60)

# Load vocabulary
print("\n1. Loading vocabulary...")
if not os.path.exists(VOCAB_JSON):
    print(f"   ERROR: {VOCAB_JSON} not found!")
    exit(1)

with open(VOCAB_JSON, 'r', encoding='utf-8') as f:
    vocab_dict = json.load(f)
vocab_set = set(vocab_dict.keys())
print(f"   ✓ {len(vocab_set)} characters loaded")

# Load train.txt
print("\n2. Loading train.txt...")
if not os.path.exists(TRAIN_FILE):
    print(f"   ERROR: {TRAIN_FILE} not found!")
    exit(1)

with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()
print(f"   ✓ {len(lines)} lines loaded")

# Clean texts
print("\n3. Cleaning out-of-vocabulary characters...")
cleaned_lines = []
modified = 0

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    parts = line.split('|')
    if len(parts) < 3:
        continue
    
    file_id, speaker, text = parts[0], parts[1], parts[2]
    
    # Clean: keep only vocab chars, replace others with space
    cleaned = []
    for char in text:
        if char in vocab_set:
            cleaned.append(char)
        else:
            cleaned.append(' ')
    
    cleaned_text = re.sub(r'\s+', ' ', ''.join(cleaned)).strip()
    
    if cleaned_text != text:
        modified += 1
    
    if cleaned_text:
        cleaned_lines.append(f"{file_id}|{speaker}|{cleaned_text}")

print(f"   ✓ {modified} lines modified")
print(f"   ✓ {len(cleaned_lines)} valid lines")

# Save
print("\n4. Saving cleaned train.txt...")
backup = TRAIN_FILE + ".backup_oov"
if not os.path.exists(backup):
    with open(backup, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"   ✓ Backup: {backup}")

with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
    for line in cleaned_lines:
        f.write(line + '\n')
print(f"   ✓ Saved: {TRAIN_FILE}")

# Verify
print("\n5. Verification...")
oov_found = 0
with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 3:
            for char in parts[2]:
                if char not in vocab_set:
                    oov_found += 1

if oov_found == 0:
    print("   ✓ SUCCESS: No out-of-vocabulary characters!")
else:
    print(f"   ✗ ERROR: Still {oov_found} OOV chars found!")

print("\n" + "=" * 60)
print("DONE! Now run: python train.py")
print("=" * 60)
