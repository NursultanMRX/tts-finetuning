#!/usr/bin/env python3
"""
Debug script to check what load_tts_samples returns
"""
import os
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

DATASET_PATH = "./my_dataset"

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="train.txt",
    path=DATASET_PATH
)

print("=" * 60)
print("DATASET DEBUG")
print("=" * 60)
print(f"Dataset path: {DATASET_PATH}")
print(f"Train file: train.txt")
print(f"Full path: {os.path.join(DATASET_PATH, 'train.txt')}")
print("")

# Check train.txt exists
train_file_path = os.path.join(DATASET_PATH, "train.txt")
if os.path.exists(train_file_path):
    print("✓ train.txt topildi")
    with open(train_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"✓ Jami qatorlar: {len(lines)}")

    # Check for tumaris2_0012.wav
    print("\nSearching for tumaris2_0012.wav...")
    for i, line in enumerate(lines, 1):
        if "tumaris2_0012" in line:
            print(f"  Line {i}: {line.strip()}")
else:
    print("✗ train.txt topilmadi!")
    exit(1)

print("\n" + "=" * 60)
print("LOADING SAMPLES")
print("=" * 60)

try:
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_size=10
    )

    print(f"✓ Train samples: {len(train_samples)}")
    print(f"✓ Eval samples: {len(eval_samples)}")

    # Check samples for tumaris2_0012.wav
    print("\n" + "=" * 60)
    print("CHECKING SAMPLES FOR tumaris2_0012.wav")
    print("=" * 60)

    found = False
    for i, sample in enumerate(train_samples + eval_samples):
        if "tumaris2_0012" in sample["audio_file"]:
            print(f"\nSample {i}:")
            print(f"  audio_file: {sample['audio_file']}")
            print(f"  text: {sample['text'][:50]}...")

            # Check if file exists
            if os.path.exists(sample["audio_file"]):
                print(f"  ✓ File exists")
            else:
                print(f"  ✗ File NOT found!")
                print(f"  Looking for: {sample['audio_file']}")

                # Try to find the correct path
                correct_path = sample["audio_file"].replace("wavs/wavs/", "wavs/").replace(".wav.wav", ".wav")
                if os.path.exists(correct_path):
                    print(f"  ✓ Correct path exists: {correct_path}")
                else:
                    print(f"  ✗ Even correct path not found: {correct_path}")

            found = True

    if not found:
        print("✓ tumaris2_0012.wav not found in samples (might have been filtered out)")

    # Show first 5 samples
    print("\n" + "=" * 60)
    print("FIRST 5 SAMPLES")
    print("=" * 60)
    for i, sample in enumerate(train_samples[:5]):
        print(f"{i+1}. {sample['audio_file']}")
        if os.path.exists(sample["audio_file"]):
            print("   ✓ EXISTS")
        else:
            print("   ✗ NOT FOUND")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
