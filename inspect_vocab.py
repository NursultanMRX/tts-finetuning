#!/usr/bin/env python3
"""Inspect vocabulary to debug tokenizer mismatch"""

import json

# Load vocab.json
with open('mms_kaa_hf/vocab.json', 'r', encoding='utf-8') as f:
    vocab_dict = json.load(f)

print(f"Total characters in vocab.json: {len(vocab_dict)}")
print("\nVocabulary mapping (character -> index):")

# Sort by index
vocab_sorted = sorted(vocab_dict.items(), key=lambda x: x[1])

for char, idx in vocab_sorted:
    # Show character representation
    if char == ' ':
        char_repr = '<space>'
    elif char == '\n':
        char_repr = '<newline>'
    elif char == '\t':
        char_repr = '<tab>'
    else:
        char_repr = char
    print(f"  {idx:2d}: '{char_repr}'")

# Show the vocab string that will be created
vocab_string = ''.join([char for char, _ in vocab_sorted])
print(f"\nVocabulary string length: {len(vocab_string)}")
print(f"Vocabulary string: {repr(vocab_string[:100])}...")

# Check for special tokens
special_tokens = ['_', '~', '^', '<pad>', '<blank>', '<eos>', '<bos>']
print("\nSpecial tokens found:")
for token in special_tokens:
    if token in vocab_dict:
        print(f"  '{token}' -> index {vocab_dict[token]}")
