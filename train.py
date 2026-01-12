import os
import torch
# Trainer importi to'g'irlandi (Coqui 0.22.0 standarti)
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.datasets import load_tts_samples

# Xotira xavfsizligi
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# --- SOZLAMALAR ---
OUTPUT_PATH = "output_finetune"
DATASET_PATH = "./my_dataset"
# Hugging Face modelidan foydalanish (to'g'ri vocabulary bilan)
MMS_CKPT = "mms_kaa_hf/pytorch_model.bin"
MMS_CONFIG = "mms_kaa_hf/config.json"
VOCAB_FILE = "mms_kaa_hf/vocab.txt"

def train():
    # 1. Dataset konfiguratsiyasi
    # train.txt fayli DATASET_PATH da, wavs papkasi esa uning ichida
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="train.txt",
        path=DATASET_PATH  # "./my_dataset" - train.txt shu yerda
    )

    # 2. Audio konfiguratsiyasi (16kHz MMS standarti)
    from TTS.tts.configs.vits_config import VitsAudioConfig
    audio_config = VitsAudioConfig(
        sample_rate=16000,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    # 3. MMS Config'dan vocabulary'ni yuklash
    import json
    print(f"MMS Config yuklanmoqda: {MMS_CONFIG}")
    with open(MMS_CONFIG, 'r', encoding='utf-8') as f:
        mms_config = json.load(f)

    # MMS modelining o'z vocabulary'sini olish
    if 'vocab' in mms_config:
        vocab_dict = mms_config['vocab']
        # Index bo'yicha sort qilish
        vocab_sorted = sorted(vocab_dict.items(), key=lambda x: x[1])
        vocab_chars = [char for char, _ in vocab_sorted]
        vocab_string = ''.join(vocab_chars)
        print(f"  -> {len(vocab_chars)} ta harf MMS modelidan yuklandi")

        # Karakalpak harflarini tekshirish
        karakalpak_test = ['ғ', 'қ', 'ң', 'ү', 'ҳ', 'ә', 'ө', 'ў']
        found_chars = [c for c in karakalpak_test if c in vocab_string]
        print(f"  -> Karakalpak harflari: {len(found_chars)}/{len(karakalpak_test)} topildi")
        if len(found_chars) < len(karakalpak_test):
            missing = [c for c in karakalpak_test if c not in vocab_string]
            print(f"  -> YO'Q: {missing}")
    else:
        print("  -> OGOHLANTIRISH: Config'da 'vocab' topilmadi, default vocabulary ishlatiladi")
        vocab_string = None

    # 4. ASOSIY CONFIG (checkpoint bilan mos vocabulary)
    config_kwargs = {
        'audio': audio_config,
        'run_name': "mms_kaa_finetune",
        'batch_size': 16,  # 25s audio uchun 3090 da 16 xavfsiz
        'eval_batch_size': 4,
        'batch_group_size': 0,
        'num_loader_workers': 2,
        'num_eval_loader_workers': 2,
        'run_eval': True,
        'epochs': 1000,
        'text_cleaner': "basic_cleaners",
        'use_phonemes': False,
        'phoneme_language': None,
        'compute_input_seq_cache': False,  # Cache'ni o'chirish
        'mixed_precision': True,
        'output_path': OUTPUT_PATH,
        'datasets': [dataset_config],
        'save_step': 1000,
    }

    # Agar vocabulary mavjud bo'lsa, qo'shamiz
    if vocab_string:
        config_kwargs['characters'] = vocab_string

    config = VitsConfig(**config_kwargs)

    # 4. Modelni yaratish
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # 5. MMS og'irliklarini yuklash (HuggingFace format)
    print("MMS Checkpoint yuklanmoqda...")
    checkpoint = torch.load(MMS_CKPT, map_location="cpu")

    # HuggingFace checkpoint format: to'g'ridan-to'g'ri state_dict
    # TTS library format: {'model': state_dict}
    if "model" in checkpoint:
        # TTS format
        model.load_state_dict(checkpoint["model"], strict=False)
        print("  -> TTS format checkpoint yuklandi")
    else:
        # HuggingFace format - to'g'ridan-to'g'ri state_dict
        model.load_state_dict(checkpoint, strict=False)
        print("  -> HuggingFace format checkpoint yuklandi")

    # 6. TEXT ENCODERNI MUZLATISH (Harflarni unutmasligi uchun)
    print("Freeze Text Encoder: ACTIVE")

    # Text encoder turli nomlar bilan bo'lishi mumkin
    text_encoder_names = ['enc_p', 'text_encoder', 'encoder', 'encoder_text']
    frozen = False

    for encoder_name in text_encoder_names:
        if hasattr(model, encoder_name):
            encoder = getattr(model, encoder_name)
            for param in encoder.parameters():
                param.requires_grad = False
            print(f"  -> {encoder_name} muzlatildi")
            frozen = True
            break

    if not frozen:
        print("  -> OGOHLANTIRISH: Text encoder topilmadi!")
        print("  -> Mavjud modullar:")
        for name, _ in model.named_children():
            print(f"       - {name}")

    # 7. Trainer ishga tushadi
    trainer = Trainer(
        TrainerArgs(),
        config,
        OUTPUT_PATH,
        model=model,
        train_samples=load_tts_samples(dataset_config, eval_split=True)[0],
        eval_samples=load_tts_samples(dataset_config, eval_split=True)[1],
    )

    trainer.fit()

if __name__ == "__main__":
    train()
