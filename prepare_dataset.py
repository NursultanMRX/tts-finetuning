import os
import pandas as pd
import librosa
import soundfile as sf
from huggingface_hub import snapshot_download
from tqdm import tqdm

# --- SOZLAMALAR ---
REPO_ID = "nickoo004/karakalpak-tts-speaker1"
HF_TOKEN = "SIZNING_TOKENINGIZNI_SHU_YERGA_YOZING"
LOCAL_DIR = "./my_dataset"
TARGET_SR = 16000 # MMS uchun majburiy

def main():
    # 1. Yuklab olish
    print("Dataset yuklanmoqda...")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=LOCAL_DIR, token=HF_TOKEN)

    # 2. CSV ni o'qish va formatlash
    csv_path = os.path.join(LOCAL_DIR, "metadata.csv")
    df = pd.read_csv(csv_path)
    wavs_dir = os.path.join(LOCAL_DIR, "wavs")

    formatted_lines = []
    print("Audiolarni 16kHz ga o'tkazish va tekshirish...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Fayl nomini tozalash - barcha prefixlarni olib tashlash
        file_name = row['file_name']

        # wavs/ prefixini olib tashlash (agar bor bo'lsa)
        file_name = file_name.replace("wavs/", "")

        # Agar .wav.wav bo'lsa, bittasini olib tashlash
        if file_name.endswith(".wav.wav"):
            file_name = file_name[:-4]  # Oxirgi .wav ni olib tashlash

        # .wav kengaytmasini tekshirish
        if not file_name.endswith(".wav"):
            file_name += ".wav"

        # To'liq yo'l faylni o'qish uchun
        full_wav_path = os.path.join(wavs_dir, file_name)

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

            formatted_lines.append(f"wavs/{file_name}|speaker1|{row['text']}")

    # 3. train.txt yaratish
    with open(os.path.join(LOCAL_DIR, "train.txt"), "w", encoding="utf-8") as f:
        for line in formatted_lines:
            f.write(line + "\n")

    print(f"Tayyor! {len(formatted_lines)} ta audio o'qitishga tayyorlandi.")

if __name__ == "__main__":
    main()
