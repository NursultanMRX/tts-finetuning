#!/usr/bin/env python3
"""
CRITICAL FIX: Clean text in train.txt to match MMS vocabulary exactly.

Problem: MMS vocab.json has only 47 characters (no punctuation).
         Characters like , . ? ! are not in vocabulary.
         This causes CUDA index out of bounds errors during training.

Solution: Remove or replace all characters not in MMS vocabulary.
          For TTS, punctuation removal is acceptable (pauses are inferred).
"""

import os
import json
import re
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================
TRAIN_FILE = "./my_dataset/train.txt"
VOCAB_JSON = "mms_kaa_hf/vocab.json"

# Punctuation replacement strategy:
# None = remove entirely
# " " = replace with space
PUNCT_REPLACEMENT = " "  # Replace punctuation with space (natural pause)

print("=" * 70)
print("MMS VOCABULARY TEXT CLEANER")
print("=" * 70)

# ============================================================
# 1. LOAD MMS VOCABULARY
# ============================================================
print("\n[1/5] Loading MMS vocabulary...")

if not os.path.exists(VOCAB_JSON):
    print(f"   ✗ ERROR: {VOCAB_JSON} not found!")
    print("   Please run 'python download_hf_model.py' first.")
    exit(1)

with open(VOCAB_JSON, 'r', encoding='utf-8') as f:
    vocab_dict = json.load(f)

vocab_chars = set(vocab_dict.keys())
vocab_size = len(vocab_chars)

print(f"   ✓ Loaded {vocab_size} characters from MMS vocabulary")
print(f"   ✓ Characters: {''.join(sorted(vocab_dict.keys(), key=lambda x: vocab_dict[x]))}")

# Common punctuation that will be removed
COMMON_PUNCT = set('.,?!;:()[]{}"\'-–—…«»""''')
punct_in_vocab = COMMON_PUNCT & vocab_chars
punct_not_in_vocab = COMMON_PUNCT - vocab_chars

print(f"\n   Punctuation IN vocab:     {punct_in_vocab if punct_in_vocab else 'None'}")
print(f"   Punctuation NOT IN vocab: {punct_not_in_vocab if punct_not_in_vocab else 'None'}")

# ============================================================
# 2. LOAD AND ANALYZE TRAIN.TXT
# ============================================================
print("\n[2/5] Analyzing train.txt for out-of-vocabulary characters...")

if not os.path.exists(TRAIN_FILE):
    print(f"   ✗ ERROR: {TRAIN_FILE} not found!")
    print("   Please run 'python prepare_dataset.py' first.")
    exit(1)

with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"   ✓ Loaded {len(lines)} lines from train.txt")

# Find all out-of-vocabulary characters
oov_chars = Counter()  # out-of-vocabulary
all_chars = Counter()

for line in lines:
    parts = line.strip().split('|')
    if len(parts) >= 3:
        text = parts[2]
        for char in text:
            all_chars[char] += 1
            if char not in vocab_chars:
                oov_chars[char] += 1

print(f"\n   Dataset statistics:")
print(f"   - Total unique characters: {len(all_chars)}")
print(f"   - In vocabulary:          {len(all_chars) - len(oov_chars)}")
print(f"   - Out of vocabulary:      {len(oov_chars)}")

if oov_chars:
    print(f"\n   ⚠️  OUT-OF-VOCABULARY CHARACTERS (will be removed/replaced):")
    for char, count in oov_chars.most_common(30):
        if char.isprintable() and char != ' ':
            print(f"      '{char}' (U+{ord(char):04X}): {count} occurrences")
        else:
            print(f"      [U+{ord(char):04X}]: {count} occurrences")
else:
    print("\n   ✓ All characters are in vocabulary! No cleaning needed.")
    exit(0)

# ============================================================
# 3. CREATE CLEANING FUNCTION
# ============================================================
def clean_text_for_mms(text, vocab_set, replacement=" "):
    """
    Clean text to only contain characters in MMS vocabulary.
    
    Args:
        text: Original text
        vocab_set: Set of valid characters
        replacement: What to replace invalid chars with (None = remove)
    
    Returns:
        Cleaned text
    """
    cleaned = []
    for char in text:
        if char in vocab_set:
            cleaned.append(char)
        elif replacement is not None:
            cleaned.append(replacement)
        # else: skip the character (remove)
    
    # Clean up multiple spaces
    result = ''.join(cleaned)
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

# ============================================================
# 4. CLEAN ALL TEXTS
# ============================================================
print(f"\n[3/5] Cleaning texts (replacing OOV chars with '{PUNCT_REPLACEMENT or 'nothing'}')...")

cleaned_lines = []
changes_made = 0
chars_removed = 0

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    parts = line.split('|')
    if len(parts) < 3:
        continue
    
    file_id = parts[0]
    speaker = parts[1]
    text = parts[2]
    
    # Clean the text
    cleaned_text = clean_text_for_mms(text, vocab_chars, PUNCT_REPLACEMENT)
    
    if cleaned_text != text:
        changes_made += 1
        chars_removed += len(text) - len(cleaned_text.replace(' ', '').replace(text.replace(' ', ''), ''))
    
    # Skip if text becomes empty after cleaning
    if not cleaned_text.strip():
        print(f"   ⚠️  Skipping empty text after cleaning: {file_id}")
        continue
    
    cleaned_lines.append(f"{file_id}|{speaker}|{cleaned_text}")

print(f"   ✓ Processed {len(lines)} lines")
print(f"   ✓ Modified  {changes_made} lines")
print(f"   ✓ Final     {len(cleaned_lines)} valid lines")

# ============================================================
# 5. VERIFY CLEANED DATA
# ============================================================
print("\n[4/5] Verifying cleaned data...")

verify_oov = Counter()
for line in cleaned_lines:
    parts = line.split('|')
    if len(parts) >= 3:
        text = parts[2]
        for char in text:
            if char not in vocab_chars:
                verify_oov[char] += 1

if verify_oov:
    print(f"   ✗ ERROR: Still found {len(verify_oov)} OOV characters!")
    for char, count in verify_oov.most_common():
        print(f"      '{char}': {count}")
    exit(1)
else:
    print("   ✓ SUCCESS: All characters are now in vocabulary!")

# ============================================================
# 6. SAVE CLEANED DATA
# ============================================================
print("\n[5/5] Saving cleaned train.txt...")

# Backup original
backup_file = TRAIN_FILE + ".before_vocab_clean"
if not os.path.exists(backup_file):
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"   ✓ Backup saved: {backup_file}")

# Write cleaned version
with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
    for line in cleaned_lines:
        f.write(line + '\n')

print(f"   ✓ Cleaned train.txt saved: {TRAIN_FILE}")

# Show sample
print("\n" + "=" * 70)
print("SAMPLE OF CLEANED DATA (first 5 lines):")
print("=" * 70)
for i, line in enumerate(cleaned_lines[:5], 1):
    parts = line.split('|')
    if len(parts) >= 3:
        print(f"{i}. {parts[2][:70]}...")

print("\n" + "=" * 70)
print("✅ TEXT CLEANING COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("  1. Clear any cached data: bash clear_all_cache.sh (if available)")
print("  2. Start training:        python train.py")
print("\n⚠️  Note: Punctuation has been removed/replaced with spaces.")
print("   The TTS model will infer natural pauses from context.")
