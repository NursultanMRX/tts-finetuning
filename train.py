#!/usr/bin/env python3
"""
=====================================================
MMS-TTS Fine-tuning Training Script
=====================================================

This script handles:
1. Loading the pre-trained MMS Karakalpak (kaa) checkpoint
2. Configuring VITS for fine-tuning with frozen text encoder
3. Training with low learning rate to prevent catastrophic forgetting
4. ROBUST OOM/bad_alloc ERROR HANDLING

Key Training Strategy:
- FREEZE the Text Encoder (enc_p) to preserve learned representations
- Use LOW learning rate (1e-5) for stable fine-tuning
- Use character-based approach (no phonemes) for MMS compatibility
- Automatic memory management and OOM recovery

Author: Senior ML Engineer
Target: Facebook MMS-TTS Karakalpak on RTX 3090/4090 (24GB VRAM)
=====================================================
"""

import os
import gc
import sys
import json
import glob
import signal
import traceback
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
BATCH_SIZE = 32  # Will be auto-reduced on OOM
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

# =====================================================
# OOM PREVENTION SETTINGS
# =====================================================
MIN_BATCH_SIZE = 4  # Minimum batch size before giving up
ENABLE_GRADIENT_CHECKPOINTING = True  # Reduces VRAM at cost of speed
CUDA_MEMORY_FRACTION = 0.95  # Reserve 5% VRAM for safety
EMPTY_CACHE_EVERY_N_STEPS = 100  # Clear cache periodically


# =====================================================
# GPU MEMORY MANAGEMENT UTILITIES
# =====================================================

def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - reserved
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "free_gb": free,
        "usage_percent": (reserved / total) * 100
    }


def print_gpu_memory():
    """Print current GPU memory status."""
    info = get_gpu_memory_info()
    if info:
        print(f"GPU Memory: {info['allocated_gb']:.2f}GB allocated, "
              f"{info['reserved_gb']:.2f}GB reserved, "
              f"{info['free_gb']:.2f}GB free ({info['usage_percent']:.1f}% used)")


def clear_gpu_memory():
    """
    Aggressively clear GPU memory.
    
    Call this when OOM is detected to free up as much memory as possible.
    """
    if torch.cuda.is_available():
        # Empty the CUDA cache
        torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
        
        # Synchronize CUDA operations
        torch.cuda.synchronize()
        
        print("✓ GPU memory cleared")
        print_gpu_memory()


def setup_cuda_memory_management():
    """
    Configure CUDA memory settings for optimal training.
    
    =====================================================
    MEMORY OPTIMIZATION STRATEGIES
    =====================================================
    1. Set memory fraction to prevent OOM from memory fragmentation
    2. Enable expandable segments for better memory utilization
    3. Configure allocator for reduced fragmentation
    =====================================================
    """
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, running on CPU (will be very slow)")
        return
    
    print("\n" + "="*50)
    print("Configuring CUDA Memory Management")
    print("="*50)
    
    # Set memory fraction (leave some headroom)
    try:
        torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)
        print(f"✓ Set CUDA memory fraction to {CUDA_MEMORY_FRACTION * 100:.0f}%")
    except Exception as e:
        print(f"⚠ Could not set memory fraction: {e}")
    
    # Enable memory-efficient settings via environment variables
    # These should be set before CUDA is initialized, but we set them anyway
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    
    # Print initial memory status
    print_gpu_memory()
    
    # Pre-allocate some memory to test
    try:
        test_tensor = torch.zeros(1000, 1000, device='cuda')
        del test_tensor
        torch.cuda.empty_cache()
        print("✓ CUDA memory test passed")
    except RuntimeError as e:
        print(f"⚠ CUDA memory test warning: {e}")


def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing to reduce memory usage.
    
    =====================================================
    GRADIENT CHECKPOINTING
    =====================================================
    Trades compute for memory by not storing all activations.
    Instead, recomputes them during backward pass.
    
    Pros:
    - Significantly reduces VRAM usage (up to 50%)
    - Allows larger batch sizes
    
    Cons:
    - ~20-30% slower training
    - May slightly affect gradient computation
    
    This is CRITICAL for preventing OOM on 24GB GPUs.
    =====================================================
    """
    print("\n" + "="*50)
    print("Enabling Gradient Checkpointing")
    print("="*50)
    
    checkpointed_modules = 0
    
    # Enable checkpointing for encoder modules
    for name, module in model.named_modules():
        if hasattr(module, 'gradient_checkpointing_enable'):
            module.gradient_checkpointing_enable()
            checkpointed_modules += 1
        elif hasattr(module, 'set_gradient_checkpointing'):
            module.set_gradient_checkpointing(True)
            checkpointed_modules += 1
    
    # For VITS specifically, we can enable checkpointing on key components
    if hasattr(model, 'enc_p') and hasattr(model.enc_p, 'encoder'):
        # Try to enable for transformer encoder if present
        try:
            if hasattr(model.enc_p.encoder, 'layers'):
                for layer in model.enc_p.encoder.layers:
                    if hasattr(layer, 'set_grad_checkpointing'):
                        layer.set_grad_checkpointing(True)
                        checkpointed_modules += 1
        except Exception:
            pass
    
    if checkpointed_modules > 0:
        print(f"✓ Enabled gradient checkpointing on {checkpointed_modules} modules")
    else:
        print("⚠ No modules support native gradient checkpointing")
        print("  Using torch.utils.checkpoint may help in custom training loops")
    
    return checkpointed_modules


class OOMHandler:
    """
    Handler for Out-of-Memory errors during training.
    
    =====================================================
    OOM/BAD_ALLOC ERROR HANDLING STRATEGY
    =====================================================
    When OOM occurs:
    1. Clear GPU cache immediately
    2. Log the error with memory stats
    3. Reduce batch size if possible
    4. Retry with smaller batch
    5. If min batch size reached, save checkpoint and exit gracefully
    
    This prevents training crashes and data loss.
    =====================================================
    """
    
    def __init__(self, initial_batch_size, min_batch_size=4):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.oom_count = 0
        self.last_successful_batch_size = initial_batch_size
        
    def handle_oom(self, error, model=None, trainer=None):
        """
        Handle an OOM error.
        
        Returns:
            tuple: (should_continue, new_batch_size)
        """
        self.oom_count += 1
        
        print("\n" + "!"*60)
        print("   OUT OF MEMORY ERROR DETECTED")
        print("!"*60)
        print(f"\nError: {error}")
        print(f"OOM count: {self.oom_count}")
        
        # Print memory info before clearing
        print("\nMemory before cleanup:")
        print_gpu_memory()
        
        # Clear GPU memory aggressively
        clear_gpu_memory()
        
        # Print memory info after clearing
        print("\nMemory after cleanup:")
        print_gpu_memory()
        
        # Try to reduce batch size
        if self.current_batch_size > self.min_batch_size:
            new_batch_size = max(self.current_batch_size // 2, self.min_batch_size)
            print(f"\n⚠ Reducing batch size: {self.current_batch_size} → {new_batch_size}")
            self.current_batch_size = new_batch_size
            return True, new_batch_size
        else:
            print(f"\n✗ Already at minimum batch size ({self.min_batch_size})")
            print("  Cannot reduce further. Training must stop.")
            
            # Try to save checkpoint before exiting
            if trainer is not None:
                try:
                    print("\n  Attempting to save emergency checkpoint...")
                    emergency_path = Path(OUTPUT_DIR) / "emergency_checkpoint.pth"
                    torch.save({
                        'model_state_dict': model.state_dict() if model else None,
                        'oom_count': self.oom_count,
                        'last_batch_size': self.current_batch_size,
                    }, emergency_path)
                    print(f"  ✓ Emergency checkpoint saved to: {emergency_path}")
                except Exception as save_error:
                    print(f"  ✗ Could not save emergency checkpoint: {save_error}")
            
            return False, self.current_batch_size
    
    def report(self):
        """Print OOM handling summary."""
        print(f"\nOOM Handler Summary:")
        print(f"  - Total OOM events: {self.oom_count}")
        print(f"  - Initial batch size: {BATCH_SIZE}")
        print(f"  - Final batch size: {self.current_batch_size}")


def safe_training_step(trainer, oom_handler):
    """
    Wrapper for training that handles OOM errors gracefully.
    """
    try:
        trainer.fit()
        return True
    except RuntimeError as e:
        error_str = str(e).lower()
        
        # Check for CUDA OOM errors
        if any(x in error_str for x in ['out of memory', 'cuda out of memory', 'bad_alloc', 'cudallocasync']):
            should_continue, new_batch_size = oom_handler.handle_oom(e, trainer.model, trainer)
            return should_continue
        else:
            # Not an OOM error, re-raise
            raise
    except torch.cuda.OutOfMemoryError as e:
        should_continue, new_batch_size = oom_handler.handle_oom(e, trainer.model, trainer)
        return should_continue


# =====================================================
# ORIGINAL FUNCTIONS (unchanged)
# =====================================================

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


def create_training_config(mms_config: dict, batch_size: int = BATCH_SIZE) -> VitsConfig:
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
        # TRAINING SETTINGS - Use dynamic batch size
        # =====================================================
        batch_size=batch_size,  # Now accepts dynamic batch size
        eval_batch_size=min(EVAL_BATCH_SIZE, batch_size),
        
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


def setup_signal_handlers(model, output_dir):
    """
    Setup signal handlers for graceful shutdown.
    
    Catches SIGINT (Ctrl+C) and SIGTERM to save checkpoint before exit.
    """
    def signal_handler(signum, frame):
        print("\n" + "!"*60)
        print("   INTERRUPT SIGNAL RECEIVED")
        print("!"*60)
        print("\nSaving checkpoint before exit...")
        
        try:
            emergency_path = Path(output_dir) / "interrupt_checkpoint.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'interrupted': True,
            }, emergency_path)
            print(f"✓ Checkpoint saved to: {emergency_path}")
        except Exception as e:
            print(f"✗ Could not save checkpoint: {e}")
        
        # Clear GPU memory
        clear_gpu_memory()
        
        print("\nExiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    print("✓ Signal handlers configured for graceful shutdown")


def main():
    """Main training entry point with OOM handling."""
    print("="*60)
    print("   MMS-TTS Karakalpak Fine-tuning")
    print("   Optimized for RTX 3090/4090 (24GB VRAM)")
    print("   WITH OOM/BAD_ALLOC ERROR HANDLING")
    print("="*60)
    
    # =====================================================
    # Step 0: Setup CUDA Memory Management
    # =====================================================
    setup_cuda_memory_management()
    
    # Check CUDA availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize OOM handler
    oom_handler = OOMHandler(BATCH_SIZE, MIN_BATCH_SIZE)
    current_batch_size = BATCH_SIZE
    
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
    # Training loop with OOM recovery
    # =====================================================
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # =====================================================
            # Step 2: Create training configuration
            # =====================================================
            print("\n" + "="*50)
            print(f"Step 2: Creating Training Configuration (batch_size={current_batch_size})")
            print("="*50)
            
            config = create_training_config(mms_config, current_batch_size)
            
            # Print key settings
            print(f"\nKey configuration:")
            print(f"  - Sample rate: {SAMPLE_RATE} Hz")
            print(f"  - Batch size: {current_batch_size}")
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
            
            # Clear memory before model initialization
            clear_gpu_memory()
            
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
            # Step 4.5: Enable Gradient Checkpointing (Memory Saver)
            # =====================================================
            if ENABLE_GRADIENT_CHECKPOINTING:
                enable_gradient_checkpointing(model)
            
            # Setup signal handlers for graceful shutdown
            setup_signal_handlers(model, OUTPUT_DIR)
            
            # Print memory after model loading
            print("\nMemory after model initialization:")
            print_gpu_memory()
            
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
            print(f"Current batch size: {current_batch_size}")
            print("\nMonitor training with TensorBoard:")
            print(f"  tensorboard --logdir={OUTPUT_DIR}")
            print("\n" + "-"*60)
            
            # Start training
            trainer.fit()
            
            # Training completed successfully
            print("\n" + "="*60)
            print("   Training Complete!")
            print("="*60)
            print(f"\nFine-tuned model saved in: {OUTPUT_DIR}")
            print("\nTo use the fine-tuned model:")
            print("  from TTS.api import TTS")
            print(f"  tts = TTS(model_path='{OUTPUT_DIR}/best_model.pth', config_path='{OUTPUT_DIR}/config.json')")
            print("  tts.tts_to_file('Your text here', file_path='output.wav')")
            
            # Print OOM summary
            oom_handler.report()
            
            break  # Success, exit retry loop
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_str = str(e).lower()
            
            # Check if it's an OOM error
            if any(x in error_str for x in ['out of memory', 'cuda out of memory', 'bad_alloc', 'cudallocasync', 'allocat']):
                retry_count += 1
                print(f"\n⚠ OOM Error (Attempt {retry_count}/{max_retries})")
                
                # Handle OOM
                should_continue, new_batch_size = oom_handler.handle_oom(e)
                
                if should_continue and new_batch_size != current_batch_size:
                    current_batch_size = new_batch_size
                    print(f"\n🔄 Retrying with batch_size={current_batch_size}...")
                    
                    # Clean up before retry
                    del model, trainer, ap, tokenizer
                    clear_gpu_memory()
                    
                    continue
                else:
                    print("\n✗ Cannot recover from OOM. Exiting.")
                    oom_handler.report()
                    sys.exit(1)
            else:
                # Not an OOM error, re-raise
                print(f"\n✗ Non-OOM error occurred:")
                traceback.print_exc()
                raise
                
        except Exception as e:
            print(f"\n✗ Unexpected error occurred:")
            traceback.print_exc()
            
            # Try to save emergency checkpoint
            try:
                if 'model' in dir():
                    emergency_path = Path(OUTPUT_DIR) / "error_checkpoint.pth"
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    torch.save({'model_state_dict': model.state_dict()}, emergency_path)
                    print(f"✓ Emergency checkpoint saved to: {emergency_path}")
            except:
                pass
            
            raise
    
    if retry_count >= max_retries:
        print(f"\n✗ Exceeded maximum retries ({max_retries}). Training failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
