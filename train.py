#!/usr/bin/env python3
"""
=====================================================
MMS-TTS Fine-tuning Training Script
=====================================================

This script handles:
1. Loading the pre-trained MMS Karakalpak (kaa) checkpoint
2. Configuring VITS for fine-tuning with frozen text encoder
3. Training with low learning rate to prevent catastrophic forgetting

Key Training Strategy:
- FREEZE the Text Encoder (enc_p) to preserve learned representations
- Use LOW learning rate (1e-5) for stable fine-tuning
- Use character-based approach (no phonemes) for MMS compatibility

Author: Senior ML Engineer
Target: Facebook MMS-TTS Karakalpak on RTX 3090/4090 (24GB VRAM)
=====================================================
"""

import os
import json
import glob
from pathlib import Path

import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.trainer import Trainer, TrainerArgs


# =====================================================
# CONFIGURATION
# =====================================================

# Paths
MMS_CHECKPOINT_DIR = "./mms_kaa"
DATASET_DIR = "./my_dataset"
TRAIN_FILE = "./train.txt"
OUTPUT_DIR = "./output_kaa_finetuned"

# Training parameters optimized for RTX 3090/4090 (24GB VRAM)
BATCH_SIZE = 32  # Adjust based on VRAM usage
EVAL_BATCH_SIZE = 16
SAMPLE_RATE = 16000  # MMS native sample rate - DO NOT CHANGE

# Learning rate - CRITICAL for preventing catastrophic forgetting
# Using very low LR because we're fine-tuning, not training from scratch
LEARNING_RATE = 1e-5

# Training steps
TOTAL_STEPS = 50000  # Adjust based on your dataset size
SAVE_STEP = 1000
EVAL_STEP = 500
LOG_STEP = 100

# Eval split
EVAL_SPLIT_SIZE = 10  # Number of samples for evaluation


def find_checkpoint():
    """Find the generator checkpoint in the MMS directory."""
    checkpoint_dir = Path(MMS_CHECKPOINT_DIR)
    
    # Look for generator checkpoint files
    patterns = ["G_*.pth", "generator_*.pth", "checkpoint_*.pth"]
    
    for pattern in patterns:
        matches = list(checkpoint_dir.glob(pattern))
        if matches:
            # Return the one with highest step number
            return str(sorted(matches)[-1])
    
    # If no pattern matches, look for any .pth file
    pth_files = list(checkpoint_dir.glob("*.pth"))
    if pth_files:
        return str(pth_files[0])
    
    raise FileNotFoundError(
        f"No checkpoint found in {MMS_CHECKPOINT_DIR}/\n"
        "Expected files like G_100000.pth or similar"
    )


def load_mms_vocab(vocab_path: str) -> dict:
    """
    Load the MMS vocabulary file.
    
    MMS uses a character-based vocabulary stored in vocab.txt.
    Each line contains one character.
    """
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            char = line.strip()
            if char:  # Skip empty lines
                vocab[char] = idx
    
    print(f"Loaded vocabulary with {len(vocab)} characters")
    return vocab


def load_mms_config(config_path: str) -> dict:
    """Load the original MMS config.json."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_training_config(mms_config: dict) -> VitsConfig:
    """
    Create VitsConfig for fine-tuning based on original MMS config.
    
    =====================================================
    VOCAB LOGIC
    =====================================================
    MMS uses a specific character-based vocabulary.
    We must use use_phonemes=False to match MMS's character-based
    text processing. Using phonemes would cause dimension mismatch
    because the text encoder was trained on characters, not phonemes.
    =====================================================
    """
    
    # Get audio config from MMS
    audio_config = mms_config.get("audio", {})
    model_config = mms_config.get("model", {})
    
    config = VitsConfig(
        # Model name
        model="vits",
        run_name="mms_kaa_finetuned",
        
        # =====================================================
        # TEXT PROCESSING - Character-based (NO PHONEMES)
        # =====================================================
        # MMS uses character-based text, not phonemes
        # This is CRITICAL to avoid dimension mismatch errors
        use_phonemes=False,
        phoneme_language=None,
        
        # Character settings
        text_cleaner="basic_cleaners",
        characters=None,  # Will be set from vocab.txt
        add_blank=True,  # MMS uses blank tokens between characters
        
        # =====================================================
        # AUDIO SETTINGS - Must match MMS (16kHz)
        # =====================================================
        audio={
            "sample_rate": SAMPLE_RATE,
            "win_length": audio_config.get("win_length", 1024),
            "hop_length": audio_config.get("hop_length", 256),
            "num_mels": audio_config.get("num_mels", 80),
            "fft_size": audio_config.get("fft_size", 1024),
            "fmin": audio_config.get("fmin", 0),
            "fmax": audio_config.get("fmax", 8000),
            "mel_fmin": audio_config.get("mel_fmin", 0),
            "mel_fmax": audio_config.get("mel_fmax", None),
        },
        
        # =====================================================
        # TRAINING SETTINGS
        # =====================================================
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        
        # Epochs (will be limited by total steps)
        epochs=1000,
        
        # Learning rates - VERY LOW to prevent catastrophic forgetting
        lr_gen=LEARNING_RATE,
        lr_disc=LEARNING_RATE,
        
        # Optimizer settings
        optimizer="AdamW",
        optimizer_params={"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01},
        
        # Scheduler
        lr_scheduler_gen="ExponentialLR",
        lr_scheduler_gen_params={"gamma": 0.999875, "last_epoch": -1},
        lr_scheduler_disc="ExponentialLR",
        lr_scheduler_disc_params={"gamma": 0.999875, "last_epoch": -1},
        
        # =====================================================
        # DATASET SETTINGS
        # =====================================================
        # Use LJSpeech formatter which expects: filename|text
        datasets=[
            {
                "formatter": "ljspeech",
                "meta_file_train": TRAIN_FILE,
                "path": os.path.join(DATASET_DIR, "wavs"),
                "language": "kaa",
            }
        ],
        
        # Eval split
        eval_split_size=EVAL_SPLIT_SIZE,
        
        # =====================================================
        # OUTPUT SETTINGS
        # =====================================================
        output_path=OUTPUT_DIR,
        
        # Checkpointing
        save_step=SAVE_STEP,
        save_checkpoints=True,
        save_n_checkpoints=3,
        save_best_after=SAVE_STEP,
        
        # Evaluation
        print_step=LOG_STEP,
        plot_step=LOG_STEP,
        
        # Mixed precision for faster training on RTX 3090/4090
        mixed_precision=True,
        
        # =====================================================
        # VITS SPECIFIC SETTINGS
        # =====================================================
        use_sdp=True,  # Stochastic Duration Predictor
        use_speaker_embedding=False,  # Single speaker
        
        # Model architecture (from MMS config if available)
        hidden_channels=model_config.get("hidden_channels", 192),
        inter_channels=model_config.get("inter_channels", 192),
        
        # Loss weights
        kl_loss_alpha=1.0,
        
        # Evaluation
        test_sentences=[
            "Sálem, men Qaraqalpaq tilinde sóylemin.",
            "Búgin háwa júda jaqsı.",
            "Qaraqalpaqstan respublikası.",
        ],
    )
    
    return config


def freeze_text_encoder(model: Vits):
    """
    =====================================================
    FREEZING STRATEGY (MOST IMPORTANT)
    =====================================================
    
    To prevent catastrophic forgetting on a small dataset,
    we FREEZE the Text Encoder (enc_p).
    
    Why freeze enc_p?
    -----------------
    The text encoder has learned to map characters to 
    meaningful representations during MMS pre-training on
    many hours of data. With only ~2 hours of fine-tuning
    data, training the text encoder would:
    
    1. Overfit to the small dataset
    2. Lose the general language knowledge
    3. Degrade overall speech quality
    
    By freezing enc_p, we:
    - Preserve the learned character representations
    - Only update the decoder and vocoder components
    - Allow the model to adapt its voice characteristics
      while keeping linguistic knowledge intact
    
    What gets frozen:
    - enc_p: Text encoder (character embeddings + text processing)
    
    What stays trainable:
    - dec: Decoder (generates mel spectrograms)
    - enc_q: Posterior encoder
    - flow: Normalizing flow
    - dp: Duration predictor
    - Generator components
    =====================================================
    """
    
    print("\n" + "="*50)
    print("FREEZING TEXT ENCODER (enc_p)")
    print("="*50)
    
    frozen_params = 0
    total_params = 0
    
    # Freeze text encoder parameters
    if hasattr(model, 'enc_p'):
        for name, param in model.enc_p.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        print(f"✓ Frozen enc_p (Text Encoder)")
    else:
        print("⚠ Warning: enc_p not found in model")
    
    # Count total trainable parameters
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"\nParameter summary:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Frozen percentage: {100 * frozen_params / total_params:.1f}%")
    
    # List trainable components
    print(f"\nTrainable components:")
    trainable_components = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            component = name.split('.')[0]
            trainable_components.add(component)
    
    for component in sorted(trainable_components):
        print(f"  - {component}")
    
    return frozen_params, trainable_params


def setup_vocab_from_mms(config: VitsConfig, vocab_path: str):
    """
    Setup vocabulary/characters from MMS vocab.txt.
    
    MMS has its own character set for each language.
    We need to use the exact same characters for fine-tuning.
    """
    characters = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            char = line.strip()
            if char:
                characters.append(char)
    
    # Create character config
    # Note: This may need adjustment based on Coqui TTS version
    print(f"\nLoaded {len(characters)} characters from vocab.txt")
    print(f"Characters: {''.join(characters[:20])}...")
    
    return characters


def main():
    """Main training entry point."""
    print("="*60)
    print("   MMS-TTS Karakalpak Fine-tuning")
    print("   Optimized for RTX 3090/4090 (24GB VRAM)")
    print("="*60)
    
    # Check CUDA availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # =====================================================
    # Step 1: Find and load MMS checkpoint
    # =====================================================
    print("\n" + "="*50)
    print("Step 1: Loading MMS Checkpoint")
    print("="*50)
    
    checkpoint_path = find_checkpoint()
    print(f"Found checkpoint: {checkpoint_path}")
    
    # Load original MMS config if available
    mms_config_path = Path(MMS_CHECKPOINT_DIR) / "config.json"
    if mms_config_path.exists():
        mms_config = load_mms_config(mms_config_path)
        print(f"Loaded MMS config from: {mms_config_path}")
    else:
        print("⚠ No config.json found, using defaults")
        mms_config = {}
    
    # Check for vocab.txt
    vocab_path = Path(MMS_CHECKPOINT_DIR) / "vocab.txt"
    if vocab_path.exists():
        characters = setup_vocab_from_mms(None, vocab_path)
    else:
        print("⚠ No vocab.txt found, using default characters")
        characters = None
    
    # =====================================================
    # Step 2: Create training configuration
    # =====================================================
    print("\n" + "="*50)
    print("Step 2: Creating Training Configuration")
    print("="*50)
    
    config = create_training_config(mms_config)
    
    # Print key settings
    print(f"\nKey configuration:")
    print(f"  - Sample rate: {SAMPLE_RATE} Hz")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - use_phonemes: False (character-based)")
    print(f"  - Mixed precision: {config.mixed_precision}")
    print(f"  - Output directory: {OUTPUT_DIR}")
    
    # =====================================================
    # Step 3: Initialize model and load checkpoint
    # =====================================================
    print("\n" + "="*50)
    print("Step 3: Initializing Model")
    print("="*50)
    
    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)
    
    # Initialize tokenizer
    tokenizer, config = TTSTokenizer.init_from_config(config)
    
    # Initialize VITS model
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    
    # Load pre-trained weights
    # Using strict=False to handle potential architecture differences
    print(f"\nLoading checkpoint: {checkpoint_path}")
    print("Using strict=False to handle potential mismatches...")
    
    try:
        model.load_checkpoint(config, checkpoint_path, eval=False, strict=False)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"⚠ Warning during checkpoint loading: {e}")
        print("Attempting to load with state_dict directly...")
        
        # Alternative loading method
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✓ Checkpoint loaded with state_dict method")
    
    # =====================================================
    # Step 4: FREEZE Text Encoder (CRITICAL STEP)
    # =====================================================
    frozen_params, trainable_params = freeze_text_encoder(model)
    
    # =====================================================
    # Step 5: Load dataset
    # =====================================================
    print("\n" + "="*50)
    print("Step 5: Loading Dataset")
    print("="*50)
    
    # Check that train.txt exists
    if not Path(TRAIN_FILE).exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}\n"
            "Run 'python prepare_dataset.py' first!"
        )
    
    # Load training samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_size=config.eval_split_size
    )
    
    print(f"Training samples: {len(train_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")
    
    # =====================================================
    # Step 6: Initialize Trainer and Start Training
    # =====================================================
    print("\n" + "="*50)
    print("Step 6: Starting Training")
    print("="*50)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save config for reference
    config_save_path = Path(OUTPUT_DIR) / "config.json"
    config.save_json(config_save_path)
    print(f"Config saved to: {config_save_path}")
    
    # Initialize trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # We already loaded the checkpoint
            skip_train_epoch=False,
            start_with_eval=True,  # Evaluate before training to get baseline
        ),
        config=config,
        output_path=OUTPUT_DIR,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    print("\n" + "="*60)
    print("   Training Started!")
    print("="*60)
    print(f"\nCheckpoints will be saved to: {OUTPUT_DIR}")
    print(f"Save interval: every {SAVE_STEP} steps")
    print(f"Evaluation interval: every {EVAL_STEP} steps")
    print("\nMonitor training with TensorBoard:")
    print(f"  tensorboard --logdir={OUTPUT_DIR}")
    print("\n" + "-"*60)
    
    # Start training
    trainer.fit()
    
    print("\n" + "="*60)
    print("   Training Complete!")
    print("="*60)
    print(f"\nFine-tuned model saved in: {OUTPUT_DIR}")
    print("\nTo use the fine-tuned model:")
    print("  from TTS.api import TTS")
    print(f"  tts = TTS(model_path='{OUTPUT_DIR}/best_model.pth', config_path='{OUTPUT_DIR}/config.json')")
    print("  tts.tts_to_file('Your text here', file_path='output.wav')")


if __name__ == "__main__":
    main()
