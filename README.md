# MMS-TTS Karakalpak Fine-tuning

Fine-tune Facebook's MMS-TTS (Karakalpak - kaa) model using Coqui TTS library on a single NVIDIA RTX 3090/4090.

## 🎯 Key Features

- **Frozen Text Encoder**: Prevents catastrophic forgetting on small datasets
- **Low Learning Rate**: Uses 1e-5 to ensure stable fine-tuning
- **16kHz Resampling**: Automatic audio conversion to MMS native sample rate
- **HuggingFace Integration**: Download private datasets with authentication

## 📁 Repository Structure

```
tts-finetuning/
├── requirements.txt      # Python dependencies
├── setup.sh             # System setup & MMS checkpoint download
├── prepare_dataset.py   # Dataset download & preprocessing
├── train.py             # Main training script
└── README.md            # This file
```

## 🚀 Quick Start

### 1. System Setup (on Vast.ai)

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install system dependencies (libsndfile1, espeak-ng, ffmpeg)
- Install Python requirements
- Download MMS Karakalpak checkpoint to `./mms_kaa/`

### 2. Prepare Dataset

Edit `prepare_dataset.py` and set:
```python
HF_TOKEN = "hf_xxxxxxxxxxxxx"  # Your HuggingFace token
HF_DATASET_REPO = "username/your-dataset"  # Your private repo
```

Then run:
```bash
python prepare_dataset.py
```

This will:
- Download your private dataset from HuggingFace
- Convert metadata.csv to LJSpeech format (train.txt)
- Resample all audio to 16kHz

### 3. Start Training

```bash
python train.py
```

Monitor with TensorBoard:
```bash
tensorboard --logdir=./output_kaa_finetuned
```

## 📊 Dataset Format

### Input (metadata.csv)
```csv
"file_name","text","speaker_name"
"wavs/audio_001.wav","Sálem, men Qaraqalpaq tilinde sóylemin.","speaker1"
"wavs/audio_002.wav","Búgin háwa júda jaqsı.","speaker1"
```

### Output (train.txt - LJSpeech format)
```
audio_001|Sálem, men Qaraqalpaq tilinde sóylemin.
audio_002|Búgin háwa júda jaqsı.
```

## ⚙️ Configuration

### Training Parameters (in train.py)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `BATCH_SIZE` | 32 | For RTX 3090 24GB |
| `SAMPLE_RATE` | 16000 | MMS native rate (DO NOT CHANGE) |
| `LEARNING_RATE` | 1e-5 | Low to prevent forgetting |
| `SAVE_STEP` | 1000 | Checkpoint interval |
| `EVAL_SPLIT_SIZE` | 10 | Validation samples |

### Freezing Strategy

The text encoder (`enc_p`) is **frozen** to preserve learned representations:

```python
for param in model.enc_p.parameters():
    param.requires_grad = False
```

**Why freeze?**
- Your dataset (~2 hours) is small compared to MMS pre-training data
- Unfreezing would cause catastrophic forgetting
- Only decoder/vocoder components need adaptation

## 🔧 Troubleshooting

### Dimension Mismatch Errors
- Ensure `use_phonemes=False` (character-based like MMS)
- Verify audio is resampled to 16kHz

### Out of Memory
- Reduce `BATCH_SIZE` to 16 or 8
- Enable gradient checkpointing

### Checkpoint Not Loading
- Use `strict=False` when loading weights
- Check checkpoint path in `mms_kaa/`

## 📦 Output

After training, find your model in `./output_kaa_finetuned/`:

```
output_kaa_finetuned/
├── best_model.pth      # Best checkpoint
├── checkpoint_*.pth    # Regular checkpoints
├── config.json         # Model configuration
└── events.*            # TensorBoard logs
```

### Using the Fine-tuned Model

```python
from TTS.api import TTS

tts = TTS(
    model_path="./output_kaa_finetuned/best_model.pth",
    config_path="./output_kaa_finetuned/config.json"
)
tts.tts_to_file("Sálem, men Qaraqalpaq tilinde sóylemin.", file_path="output.wav")
```

## 📝 License

This project uses:
- [Facebook MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) - MIT License
- [Coqui TTS](https://github.com/coqui-ai/TTS) - MPL-2.0 License
