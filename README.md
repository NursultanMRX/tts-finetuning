# 🎙️ MMS-TTS Karakalpak Fine-tuning

Fine-tune Facebook's **MMS-TTS (Massively Multilingual Speech)** model for Karakalpak language using Coqui TTS library.

> 📌 **Target Hardware**: NVIDIA RTX 3090/4090 or A40 (24GB+ VRAM)

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Repository Structure](#-repository-structure)
4. [Prerequisites](#-prerequisites)
5. [Vast.ai Setup](#️-vastai-setup)
6. [Quick Start Guide](#-quick-start-guide)
7. [Detailed Setup](#-detailed-setup)
8. [Dataset Format](#-dataset-format)
9. [Training Configuration](#️-training-configuration)
10. [Troubleshooting](#-troubleshooting)
11. [Using the Fine-tuned Model](#-using-the-fine-tuned-model)
12. [License](#-license)

---

## 🎯 Overview

This repository provides a complete pipeline for fine-tuning Facebook's MMS-TTS model on custom Karakalpak speech data. The MMS model supports 1,100+ languages, and we leverage its pre-trained Karakalpak checkpoint (`facebook/mms-tts-kaa`) as our starting point.

### Why Fine-tune?
- **Improve voice quality** for your specific speaker/domain
- **Adapt pronunciation** to your dataset's characteristics
- **Add custom vocabulary** (if extending the model)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧊 **Frozen Text Encoder** | Prevents catastrophic forgetting on small datasets |
| 📉 **Low Learning Rate** | Uses 1e-5 for stable fine-tuning |
| 🎵 **16kHz Resampling** | Automatic audio conversion to MMS native sample rate |
| 🤗 **HuggingFace Integration** | Download model & datasets directly from HuggingFace |
| 🔤 **Vocabulary Cleaning** | Auto-removes punctuation not in MMS vocabulary |
| 💾 **Checkpoint Saving** | Regular saves every 1000 steps |

---

## 📁 Repository Structure

```
tts-finetuning/
├── requirements.txt          # Python dependencies
├── setup.sh                  # System setup & environment
├── download_hf_model.py      # Download MMS model from HuggingFace
├── prepare_dataset.py        # Dataset download & preprocessing
├── quick_fix_vocab.py        # Fix vocabulary issues in train.txt
├── clean_text_for_vocab.py   # Detailed text cleaning utility
├── train.py                  # Main training script
├── debug_samples.py          # Debug dataset samples
├── mms_kaa_hf/               # Downloaded MMS model (created by download_hf_model.py)
│   ├── pytorch_model.bin     # Model weights
│   ├── config.json           # Model configuration
│   └── vocab.json            # 47-character vocabulary
├── my_dataset/               # Your prepared dataset (created by prepare_dataset.py)
│   ├── wavs/                 # Audio files (16kHz WAV)
│   └── train.txt             # LJSpeech format metadata
└── output_finetune/          # Training outputs & checkpoints
```

---

## � Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 3090, 4090, A40, etc.)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ free space

### Software Requirements
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.10+
- CUDA 11.8+ with cuDNN
- Git

### Accounts Needed
- **HuggingFace Account**: For downloading models and private datasets
- **HuggingFace Token**: Create at https://huggingface.co/settings/tokens

---

## �️ Vast.ai Setup

### ⚠️ IMPORTANT: Choose the Correct Template!

**❌ DO NOT use old templates like:**
```
nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04  # TOO OLD!
pytorch 1.0                                  # TOO OLD!
```

**✅ USE this template:**

| Setting | Recommended Value |
|---------|-------------------|
| **Docker Image** | `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel` |
| **CUDA** | 11.8+ |
| **PyTorch** | 2.0+ |
| **Python** | 3.10+ |
| **GPU** | RTX 3090, 4090, A40, A100 (24GB+ VRAM) |
| **Disk** | 50GB+ |

### How to Select Template on Vast.ai:

1. Go to [Vast.ai Console](https://cloud.vast.ai/)
2. Click **"Create Instance"**
3. In **Template** section, select **"PyTorch"**
4. Choose version: **`pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`**
5. Or use **Custom Docker Image**: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`

### Alternative Docker Images:

```bash
# Option 1: Official PyTorch (Recommended)
pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Option 2: NVIDIA NGC PyTorch
nvcr.io/nvidia/pytorch:23.10-py3

# Option 3: Latest PyTorch
pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
```

### After Starting Instance:

```bash
# Verify CUDA version (should be 11.8+)
nvcc --version

# Verify PyTorch version (should be 2.0+)
python -c "import torch; print(torch.__version__)"

# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available())"
```

---

## �🚀 Quick Start Guide

For experienced users, here's the TL;DR:

```bash
# 1. Clone repository
git clone https://github.com/NursultanMRX/tts-finetuning.git
cd tts-finetuning

# 2. Setup environment
bash setup.sh

# 3. Activate virtual environment
source venv_tts/bin/activate

# 4. Download MMS model from HuggingFace
python download_hf_model.py

# 5. Edit prepare_dataset.py with your HF token and dataset repo
nano prepare_dataset.py

# 6. Prepare dataset
python prepare_dataset.py

# 7. Fix vocabulary (remove punctuation not in MMS vocab)
python quick_fix_vocab.py

# 8. Start training!
python train.py
```

---

## 📖 Detailed Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/NursultanMRX/tts-finetuning.git
cd tts-finetuning
```

### Step 2: Run Setup Script

The setup script will:
- Install system dependencies (libsndfile, ffmpeg)
- Create Python virtual environment
- Install Python packages (TTS, torch, etc.)

```bash
bash setup.sh
```

### Step 3: Activate Virtual Environment

```bash
source venv_tts/bin/activate
```

### Step 4: Download MMS Model

This downloads the pre-trained MMS-TTS Karakalpak model from HuggingFace:

```bash
python download_hf_model.py
```

**What it downloads:**
- `mms_kaa_hf/pytorch_model.bin` - Pre-trained weights (~300MB)
- `mms_kaa_hf/config.json` - Model configuration
- `mms_kaa_hf/vocab.json` - 47-character vocabulary

### Step 5: Configure Dataset Download

Edit `prepare_dataset.py` to set your HuggingFace credentials:

```python
# --- SETTINGS ---
REPO_ID = "your-username/your-dataset"  # Your HF dataset repo
HF_TOKEN = "hf_xxxxxxxxxxxxx"            # Your HF token
LOCAL_DIR = "./my_dataset"
TARGET_SR = 16000  # MMS requires 16kHz
```

### Step 6: Prepare Dataset

```bash
python prepare_dataset.py
```

**What it does:**
1. Downloads your dataset from HuggingFace
2. Resamples all audio to 16kHz
3. Cleans text (removes out-of-vocabulary punctuation)
4. Creates `train.txt` in LJSpeech format
5. Skips audio files longer than 25 seconds

### Step 7: Fix Vocabulary (CRITICAL!)

The MMS vocabulary contains only **47 characters** (Cyrillic letters, space, dash). It does **NOT** include punctuation like `,` `.` `?` `!`.

Run this to clean your train.txt:

```bash
python quick_fix_vocab.py
```

**⚠️ If you skip this step, you will get CUDA errors:**
```
RuntimeError: index out of bounds
```

### Step 8: Start Training

```bash
python train.py
```

**Monitor training:**
```bash
# In another terminal
tensorboard --logdir=output_finetune
```

---

## 📊 Dataset Format

### Input: Your HuggingFace Dataset

Your dataset should have:
- `metadata.csv` with columns: `file_name`, `text`
- `wavs/` folder with audio files

```csv
file_name,text
wavs/audio001.wav,бул мениң биринши сөзим
wavs/audio002.wav,қарақалпақ тили гөззал тил
```

### Output: LJSpeech Format (train.txt)

After running `prepare_dataset.py`:

```
audio001|speaker1|бул мениң биринши сөзим
audio002|speaker1|қарақалпақ тили гөззал тил
```

**Format:** `filename|speaker|text` (no file extension, no path prefix)

---

## ⚙️ Training Configuration

### Key Parameters (in train.py)

```python
# Training settings
batch_size = 16          # Reduce if OOM (try 8 or 4)
epochs = 1000            # Total training epochs
save_step = 1000         # Save checkpoint every N steps
eval_batch_size = 4      # Evaluation batch size

# Audio settings
sample_rate = 16000      # Must be 16kHz for MMS
```

### Freezing Strategy

The **text encoder** (`enc_p`) is frozen to preserve learned representations:

```python
# In train.py
for param in model.enc_p.parameters():
    param.requires_grad = False
```

**Why freeze the text encoder?**
| Reason | Explanation |
|--------|-------------|
| Small dataset | Your ~2 hours is tiny vs MMS pre-training data |
| Prevent forgetting | Unfreezing would cause catastrophic forgetting |
| Faster training | Fewer parameters to update |
| Focus adaptation | Only decoder/vocoder need fine-tuning |

---

## 🔧 Troubleshooting

### ❌ "Character not found in vocabulary"

**Symptom:**
```
[!] Character ',' not found in the vocabulary. Discarding it.
[!] Character '.' not found in the vocabulary. Discarding it.
```

**Solution:**
```bash
python quick_fix_vocab.py
```

### ❌ CUDA Index Out of Bounds

**Symptom:**
```
RuntimeError: index out of bounds
max value is tensor(47., device='cuda:0')
```

**Cause:** Text contains characters not in the 47-character MMS vocabulary.

**Solution:**
```bash
python quick_fix_vocab.py
python train.py
```

### ❌ Out of Memory (OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce batch size in `train.py`:
```python
batch_size = 8  # or 4
eval_batch_size = 2
```

### ❌ Dimension Mismatch

**Symptom:**
```
RuntimeError: size mismatch for text_encoder.emb.weight
Expected [47, 192], got [48, 192]
```

**Cause:** Vocabulary size mismatch between model and config.

**Solution:**
- Re-run `python download_hf_model.py`
- Ensure `vocab.json` has exactly 47 entries
- Check `add_blank` setting (should match HF model)

### ❌ Checkpoint Not Loading

**Symptom:**
```
KeyError: 'model' not in checkpoint
```

**Solution:** The code handles both TTS and HuggingFace formats automatically. If issues persist:
```python
# train.py already handles this:
if "model" in checkpoint:
    state_dict = checkpoint["model"]  # TTS format
else:
    state_dict = checkpoint  # HuggingFace format
```

---

## 🎤 Using the Fine-tuned Model

After training, your checkpoints are saved in `output_finetune/`.

### Load and Generate Speech

```python
from TTS.api import TTS

# Load your fine-tuned model
tts = TTS(model_path="output_finetune/mms_kaa_finetune-xxx/best_model.pth",
          config_path="output_finetune/mms_kaa_finetune-xxx/config.json")

# Generate speech
tts.tts_to_file(text="Сәлем дүнья", file_path="output.wav")
```

### Export for Production

```python
import torch

# Load checkpoint
checkpoint = torch.load("output_finetune/.../best_model.pth")

# Save just the model weights
torch.save(checkpoint['model'], "my_karakalpak_tts.pth")
```

---

## 📝 MMS Vocabulary Reference

The MMS Karakalpak model uses exactly **47 characters**:

```
| а б в г д е ж з и й к л м н о п р с т у ф х ц ш ы ь э ю я
| ғ қ ң ү ҳ ә ө ў (Karakalpak-specific)
| (space) - – | μ ѓ є њ ќ ѳ
```

**NOT included:** `, . ? ! ; : " ' ( ) [ ] { }`

Punctuation is automatically removed during preprocessing.

---

## 📝 License

This project is licensed under the MIT License.

### Acknowledgments

- [Facebook MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) - Massively Multilingual Speech
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Text-to-Speech library
- [HuggingFace](https://huggingface.co/) - Model and dataset hosting

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📬 Contact

For questions or issues, please open an issue on GitHub.
