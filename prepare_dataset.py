import os
import re
import json
import pandas as pd
import librosa
import soundfile as sf
from huggingface_hub import snapshot_download
from tqdm import tqdm

# --- SOZLAMALAR ---
REPO_ID = "nickoo004/karakalpak-tts-speaker1"
HF_TOKEN = "SIZNING_TOKENINGIZNI_SHU_YERGA_YOZING"
LOCAL_DIR = "./my_dataset"
TARGET_SR = 16000  # MMS uchun majburiy

# MMS vocabulary path
VOCAB_JSON_PATH = "mms_kaa_hf/vocab.json"


def load_mms_vocabulary():
    """Load MMS vocabulary from vocab.json"""
    if os.path.exists(VOCAB_JSON_PATH):
        with open(VOCAB_JSON_PATH, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        return set(vocab_dict.keys())
    else:
        print(f"⚠️  Warning: {VOCAB_JSON_PATH} not found!")
        print("   Run 'python download_hf_model.py' first for vocabulary-based cleaning.")
        return None


def clean_text_for_mms(text, vocab_set):
    """
    Clean text to only contain characters that exist in MMS vocabulary.
    Replaces out-of-vocabulary characters (like punctuation) with spaces.
    
    Args:
        text: Original text
        vocab_set: Set of valid characters from MMS vocabulary
    
    Returns:
        Cleaned text compatible with MMS model
    """
    if vocab_set is None:
        # If no vocabulary loaded, just do basic cleaning
        # Remove common punctuation that's unlikely to be in MMS vocab
        text = re.sub(r'[.,?!;:()[\]{}"\'\-–—…«»""'']', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    cleaned = []
    for char in text:
        if char in vocab_set:
            cleaned.append(char)
        else:
            cleaned.append(' ')  # Replace OOV with space (natural pause)
    
    # Clean up multiple spaces
    result = ''.join(cleaned)
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def main():
    # 0. Load MMS vocabulary for text cleaning
    print("MMS vocabulary yuklanmoqda...")
    vocab_set = load_mms_vocabulary()
    if vocab_set:
        print(f"  ✓ {len(vocab_set)} ta belgi yuklandi")
    else:
        print("  ⚠️  Asosiy tozalash rejimi ishlatiladi")

    # 1. Yuklab olish
    print("\nDataset yuklanmoqda...")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=LOCAL_DIR, token=HF_TOKEN)

    # 2. CSV ni o'qish va formatlash
    csv_path = os.path.join(LOCAL_DIR, "metadata.csv")
    df = pd.read_csv(csv_path)
    wavs_dir = os.path.join(LOCAL_DIR, "wavs")

    formatted_lines = []
    texts_cleaned = 0
    print("Audiolarni 16kHz ga o'tkazish va matnlarni tozalash...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Fayl nomini tozalash - barcha prefixlarni olib tashlash
        file_name = row['file_name']

        # wavs/ prefixini olib tashlash (agar bor bo'lsa)
        file_name = row['file_name'].replace("wavs/", "")

        # Agar .wav.wav bo'lsa, bittasini olib tashlash
        if file_name.endswith(".wav.wav"):
            file_name = file_name[:-4]  # Oxirgi .wav ni olib tashlash

        # .wav kengaytmasini ta'minlash (faylni o'qish uchun)
        if not file_name.endswith(".wav"):
            file_name += ".wav"

        # To'liq yo'l faylni o'qish uchun
        full_wav_path = os.path.join(wavs_dir, file_name)

        # train.txt uchun fayl nomi (.wav SIZ - ljspeech formatter o'zi qo'shadi!)
        file_name_for_train = file_name[:-4]  # .wav ni olib tashlash

        if os.path.exists(full_wav_path):
            # Audio davomiyligini tekshirish (Memory safety)
            audio, sr = librosa.load(full_wav_path, sr=None)
            duration = len(audio) / sr

            if duration > 25.0:
                print(f"Skipping {file_name}: Too long ({duration:.2f}s)")
                continue

            # 16kHz ga o'tkazish
            if sr != TARGET_SR:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
                sf.write(full_wav_path, audio_16k, TARGET_SR)

            # MATN TOZALASH - MMS vocabulary bilan mos qilish
            original_text = row['text']
            cleaned_text = clean_text_for_mms(original_text, vocab_set)
            
            if cleaned_text != original_text:
                texts_cleaned += 1
            
            # Bo'sh matn bo'lib qolsa, skip qilish
            if not cleaned_text.strip():
                print(f"Skipping {file_name}: Empty text after cleaning")
                continue

            # MUHIM: wavs/ va .wav qo'shmaslik - ljspeech formatter o'zi qo'shadi!
            formatted_lines.append(f"{file_name_for_train}|speaker1|{cleaned_text}")

    # 3. train.txt yaratish
    with open(os.path.join(LOCAL_DIR, "train.txt"), "w", encoding="utf-8") as f:
        for line in formatted_lines:
            f.write(line + "\n")

    print(f"\n✓ Tayyor! {len(formatted_lines)} ta audio o'qitishga tayyorlandi.")
    print(f"✓ {texts_cleaned} ta matn tozalandi (tinish belgilari olib tashlandi)")
    print(f"\n⚠️  Eslatma: Tinish belgilari (. , ? !) olib tashlandi.")
    print("   TTS modeli tabiiy pauzalarni kontekstdan o'rganadi.")


if __name__ == "__main__":
    main()
