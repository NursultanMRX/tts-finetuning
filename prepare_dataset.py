#!/usr/bin/env python3
"""
=====================================================
MMS-TTS Dataset Preparation Script
=====================================================

This script handles:
1. Authentication with Hugging Face Hub
2. Downloading the private dataset
3. Converting metadata to LJSpeech format
4. Resampling all audio files to 16kHz (MMS native rate)

Author: Senior ML Engineer
Target: Facebook MMS-TTS Karakalpak Fine-tuning
=====================================================
"""

import os
import sys
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from huggingface_hub import snapshot_download, login
from tqdm import tqdm


# =====================================================
# CONFIGURATION - MODIFY THESE VALUES
# =====================================================

# Your Hugging Face Token (with read access to the private repo)
# Option 1: Set directly here
HF_TOKEN = ""  # e.g., "hf_xxxxxxxxxxxxxxxxxxxxx"

# Option 2: Or set via environment variable HF_TOKEN
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"

# Your private Hugging Face dataset repository
HF_DATASET_REPO = ""  # e.g., "username/my-karakalpak-dataset"

# Target sample rate for MMS models (DO NOT CHANGE)
TARGET_SAMPLE_RATE = 16000

# Output paths
DATASET_DIR = "./my_dataset"
OUTPUT_TRAIN_FILE = "./train.txt"


def get_hf_token():
    """Get Hugging Face token from variable or environment."""
    token = HF_TOKEN or os.environ.get("HF_TOKEN")
    
    if not token:
        print("\n" + "="*50)
        print("Hugging Face Token Required")
        print("="*50)
        token = input("Enter your Hugging Face token: ").strip()
    
    if not token:
        raise ValueError(
            "No Hugging Face token provided!\n"
            "Set HF_TOKEN variable in this script or as environment variable."
        )
    
    return token


def download_dataset(token: str, repo_id: str, local_dir: str):
    """
    Download private dataset from Hugging Face Hub.
    
    Args:
        token: HuggingFace authentication token
        repo_id: Repository ID (e.g., "username/dataset-name")
        local_dir: Local directory to save the dataset
    """
    print("\n" + "="*50)
    print("Downloading Dataset from Hugging Face")
    print("="*50)
    print(f"Repository: {repo_id}")
    print(f"Local directory: {local_dir}")
    
    # Login to Hugging Face
    print("\nAuthenticating with Hugging Face...")
    login(token=token)
    
    # Download the entire repository
    print("Downloading dataset (this may take a while)...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        token=token
    )
    
    print(f"✓ Dataset downloaded to: {local_dir}")


def convert_metadata_to_ljspeech(dataset_dir: str, output_file: str):
    """
    Convert metadata.csv to LJSpeech format (filename|text).
    
    Input CSV format: "file_name","text","speaker_name"
    Output format: filename|text (no header, pipe-separated)
    
    Args:
        dataset_dir: Directory containing metadata.csv and wavs/
        output_file: Path to output train.txt file
    """
    print("\n" + "="*50)
    print("Converting Metadata to LJSpeech Format")
    print("="*50)
    
    metadata_path = Path(dataset_dir) / "metadata.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at: {metadata_path}")
    
    # Read the CSV file
    print(f"Reading: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    print(f"Found {len(df)} entries in metadata.csv")
    print(f"Columns: {list(df.columns)}")
    
    # Display sample entries
    print("\nSample entries:")
    print(df.head(3).to_string())
    
    # =====================================================
    # DATA PREP LOGIC: Clean file paths
    # =====================================================
    # The metadata.csv contains paths like "wavs/audio_001.wav"
    # We need to remove the "wavs/" prefix because:
    # 1. LJSpeech format expects just the filename without extension
    # 2. Or filename with extension but we'll handle the path separately
    # 
    # For Coqui TTS LJSpeech formatter, the format should be:
    # basename_without_extension|text
    # =====================================================
    
    # Clean the file_name column
    df['clean_filename'] = df['file_name'].apply(lambda x: x.replace('wavs/', '').replace('.wav', ''))
    
    # Create LJSpeech format: filename|text
    lines = []
    for _, row in df.iterrows():
        filename = row['clean_filename']
        text = str(row['text']).strip()
        lines.append(f"{filename}|{text}")
    
    # Write to train.txt
    print(f"\nWriting LJSpeech format to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Created {output_file} with {len(lines)} entries")
    
    # Show sample output
    print("\nSample output (first 3 lines):")
    for line in lines[:3]:
        print(f"  {line}")
    
    return len(lines)


def resample_audio_files(dataset_dir: str, target_sr: int = 16000):
    """
    Resample all WAV files to the target sample rate.
    
    =====================================================
    AUDIO RESAMPLING LOGIC (CRUCIAL)
    =====================================================
    MMS models are trained on 16kHz audio. If your dataset
    has a different sample rate (e.g., 22050Hz, 44100Hz),
    training will fail with dimension mismatch errors.
    
    This function:
    1. Scans all .wav files in the wavs/ directory
    2. Checks each file's sample rate
    3. Resamples to 16kHz if necessary
    4. Overwrites the original file with the resampled version
    =====================================================
    
    Args:
        dataset_dir: Directory containing the wavs/ folder
        target_sr: Target sample rate (16000 for MMS)
    """
    print("\n" + "="*50)
    print(f"Resampling Audio Files to {target_sr}Hz")
    print("="*50)
    
    wavs_dir = Path(dataset_dir) / "wavs"
    
    if not wavs_dir.exists():
        raise FileNotFoundError(f"wavs/ directory not found at: {wavs_dir}")
    
    wav_files = list(wavs_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")
    
    resampled_count = 0
    skipped_count = 0
    error_count = 0
    
    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        try:
            # Load audio file
            audio, sr = librosa.load(wav_file, sr=None)  # sr=None preserves original
            
            if sr != target_sr:
                # Resample to target sample rate
                audio_resampled = librosa.resample(
                    audio, 
                    orig_sr=sr, 
                    target_sr=target_sr
                )
                
                # Save the resampled audio (overwrite original)
                sf.write(wav_file, audio_resampled, target_sr)
                resampled_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            print(f"\n⚠ Error processing {wav_file.name}: {e}")
            error_count += 1
    
    print(f"\n✓ Resampling complete:")
    print(f"  - Resampled: {resampled_count} files")
    print(f"  - Already {target_sr}Hz: {skipped_count} files")
    if error_count > 0:
        print(f"  - Errors: {error_count} files")


def verify_dataset(dataset_dir: str, train_file: str):
    """Verify the prepared dataset."""
    print("\n" + "="*50)
    print("Verifying Dataset")
    print("="*50)
    
    wavs_dir = Path(dataset_dir) / "wavs"
    
    # Count files
    wav_files = list(wavs_dir.glob("*.wav"))
    with open(train_file, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    
    print(f"WAV files in wavs/: {len(wav_files)}")
    print(f"Entries in train.txt: {len(train_lines)}")
    
    # Check sample rate of first file
    if wav_files:
        audio, sr = librosa.load(wav_files[0], sr=None)
        duration = len(audio) / sr
        print(f"\nSample file: {wav_files[0].name}")
        print(f"  Sample rate: {sr}Hz")
        print(f"  Duration: {duration:.2f}s")
    
    # Calculate total duration
    total_duration = 0
    for wav_file in tqdm(wav_files, desc="Calculating total duration"):
        try:
            audio, sr = librosa.load(wav_file, sr=None)
            total_duration += len(audio) / sr
        except Exception:
            pass
    
    hours = total_duration / 3600
    minutes = (total_duration % 3600) / 60
    
    print(f"\n✓ Total audio duration: {hours:.1f}h {minutes:.1f}m ({total_duration:.0f}s)")
    
    # Verify matching files
    missing_files = []
    for line in train_lines:
        filename = line.strip().split('|')[0]
        wav_path = wavs_dir / f"{filename}.wav"
        if not wav_path.exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n⚠ Warning: {len(missing_files)} files in train.txt not found in wavs/")
        print(f"  First few: {missing_files[:5]}")
    else:
        print(f"\n✓ All files in train.txt exist in wavs/")


def main():
    """Main entry point."""
    print("="*60)
    print("   MMS-TTS Dataset Preparation")
    print("   For Karakalpak (kaa) Fine-tuning")
    print("="*60)
    
    # Get configuration
    if not HF_DATASET_REPO:
        repo_id = input("Enter your HuggingFace dataset repo (e.g., username/dataset): ").strip()
    else:
        repo_id = HF_DATASET_REPO
    
    if not repo_id:
        raise ValueError("No dataset repository specified!")
    
    # Step 1: Get token and download dataset
    token = get_hf_token()
    download_dataset(token, repo_id, DATASET_DIR)
    
    # Step 2: Convert metadata to LJSpeech format
    convert_metadata_to_ljspeech(DATASET_DIR, OUTPUT_TRAIN_FILE)
    
    # Step 3: Resample audio files to 16kHz
    resample_audio_files(DATASET_DIR, TARGET_SAMPLE_RATE)
    
    # Step 4: Verify the prepared dataset
    verify_dataset(DATASET_DIR, OUTPUT_TRAIN_FILE)
    
    print("\n" + "="*60)
    print("   Dataset Preparation Complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - Dataset: {DATASET_DIR}/")
    print(f"  - Training file: {OUTPUT_TRAIN_FILE}")
    print(f"\nNext step: Run 'python train.py' to start fine-tuning")


if __name__ == "__main__":
    main()
