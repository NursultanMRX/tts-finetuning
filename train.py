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

    # 3. MMS vocabulary'ni yuklash (vocab.json dan)
    import json
    import os

    vocab_json_path = os.path.join(os.path.dirname(MMS_CONFIG), "vocab.json")
    print(f"MMS Vocabulary yuklanmoqda: {vocab_json_path}")

    vocab_string = None
    if os.path.exists(vocab_json_path):
        with open(vocab_json_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)

        # Index bo'yicha sort qilish - MUHIM: Checkpoint bilan mos kelishi kerak!
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

        # CharactersConfig yaratish - MUHIM: Qo'shimcha tokenlar QO'SHMASLIK!
        # Checkpoint 47 ta token kutadi (0-46), shuning uchun 47 ta token yaratish kerak
        # Pad tokeni allaqachon vocab'da mavjud (odatda index 0 dagi '_')

        # Pad tokenni topish (odatda '_' yoki birinchi belgi)
        pad_char = "_" if "_" in vocab_string else vocab_chars[0]

        print(f"  -> Pad token: '{pad_char}' (index: {vocab_dict.get(pad_char, 'unknown')})")

        characters_config = CharactersConfig(
            characters_class=None,  # Default class
            characters=vocab_string,  # 47 ta belgi
            punctuations="",  # Bo'sh - barcha tinish belgilari allaqachon characters'da
            pad=pad_char,  # PAD BELGISI (index emas!)
            eos=None,  # Qo'shimcha EOS qo'shmaslik
            bos=None,  # Qo'shimcha BOS qo'shmaslik
            blank=None,  # Qo'shimcha BLANK qo'shmaslik
            is_unique=True,  # Takrorlanishni tekshirish
            is_sorted=False   # Tartibni o'zgartirmaslik
        )
        print(f"  -> CharactersConfig yaratildi: {len(vocab_string)} belgi")
    else:
        print(f"  -> XATO: {vocab_json_path} topilmadi!")
        print("  -> Iltimos, avval 'python download_hf_model.py' ishga tushiring")
        exit(1)
        characters_config = None

    # 4. ASOSIY CONFIG (checkpoint bilan mos vocabulary)
    config = VitsConfig(
        audio=audio_config,
        run_name="mms_kaa_finetune",
        batch_size=16,  # 25s audio uchun 3090 da 16 xavfsiz
        eval_batch_size=4,
        batch_group_size=0,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        run_eval=True,
        epochs=1000,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        phoneme_language=None,
        characters=characters_config,  # CharactersConfig obyekti
        compute_input_seq_cache=False,  # Cache'ni o'chirish
        mixed_precision=True,
        output_path=OUTPUT_PATH,
        datasets=[dataset_config],
        save_step=1000,
    )

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
        state_dict = checkpoint["model"]
        print("  -> TTS format checkpoint")
    else:
        # HuggingFace format - to'g'ridan-to'g'ri state_dict
        state_dict = checkpoint
        print("  -> HuggingFace format checkpoint")

    # Embedding layer hajmini tekshirish
    for key in state_dict.keys():
        if 'emb' in key.lower() and 'weight' in key:
            emb_shape = state_dict[key].shape
            print(f"  -> Checkpoint embedding: {key} = {emb_shape}")
            if len(emb_shape) == 2:
                vocab_size_in_ckpt = emb_shape[0]
                print(f"  -> Checkpoint vocabulary hajmi: {vocab_size_in_ckpt}")

                # Tokenizer vocabulary hajmi bilan taqqoslash
                # tokenizer.characters - Graphemes obyekti
                if hasattr(tokenizer.characters, 'vocab'):
                    tokenizer_vocab_size = len(tokenizer.characters.vocab)
                elif hasattr(tokenizer.characters, 'num_chars'):
                    tokenizer_vocab_size = tokenizer.characters.num_chars
                else:
                    tokenizer_vocab_size = len(str(tokenizer.characters))

                print(f"  -> Tokenizer vocabulary hajmi: {tokenizer_vocab_size}")
                print(f"  -> Tokenizer characters turi: {type(tokenizer.characters)}")

                if tokenizer_vocab_size != vocab_size_in_ckpt:
                    print(f"  -> XATO: Vocabulary o'lchami mos kelmaydi!")
                    print(f"  ->   Checkpoint: {vocab_size_in_ckpt}")
                    print(f"  ->   Tokenizer:  {tokenizer_vocab_size}")
                    print(f"  -> Config parametrlar:")
                    print(f"  ->   add_blank={config.add_blank if hasattr(config, 'add_blank') else 'N/A'}")
                    if hasattr(tokenizer.characters, 'blank'):
                        print(f"  ->   blank token: '{tokenizer.characters.blank}'")
                    if hasattr(tokenizer.characters, 'pad'):
                        print(f"  ->   pad token: '{tokenizer.characters.pad}'")
                    exit(1)
            break

    # Checkpoint yuklanishi
    model.load_state_dict(state_dict, strict=False)
    print("  -> Checkpoint yuklandi")

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
