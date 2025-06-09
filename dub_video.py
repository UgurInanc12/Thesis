#!/usr/bin/env python
import os
import sys
import subprocess
import requests
import whisper
import time
import re
import uuid
from dotenv import load_dotenv
from TTS.api import TTS
from pydub import AudioSegment
import imageio_ffmpeg

# --- Yardımcı fonksiyonlar ---
def get_ffprobe_path():
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
    if not os.path.exists(ffprobe_path):
        raise RuntimeError("ffprobe binary bulunamadı! Lütfen ffprobe'un ffmpeg ile aynı klasörde olduğundan emin ol.")
    return ffprobe_path

def get_media_duration(file_path):
    ffprobe_path = get_ffprobe_path()
    cmd = [
        ffprobe_path, '-i', file_path,
        '-show_entries', 'format=duration',
        '-v', 'quiet',
        '-of', 'csv=p=0'
    ]
    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
        return float(output.strip())
    except Exception as e:
        print(f"Süre okunamadı: {e}")
        return 0.0

def split_sentences(text, max_chars=240):
    sentences = re.split(r'([.!?])', text)
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < max_chars:
            current += sent + " "
        else:
            chunks.append(current.strip())
            current = sent + " "
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c.strip()]

def merge_wavs(wav_paths, out_path):
    combined = AudioSegment.empty()
    for w in wav_paths:
        combined += AudioSegment.from_wav(w)
    combined.export(out_path, format='wav')

def adjust_audio_length(audio_path, video_duration, ffmpeg_path, base_dir, tag):
    audio_duration = get_media_duration(audio_path)
    if audio_duration == 0:
        return audio_path
    rate = video_duration / audio_duration
    if abs(rate - 1.0) < 0.01:
        return audio_path  # Neredeyse aynıysa gerek yok
    # FFmpeg'in atempo filtresi 0.5-2.0 arası çalışır, gerekirse zincirle
    filters = []
    orig_rate = rate
    temp_audio = os.path.join(base_dir, f"adjusted_{tag}.wav")
    while rate < 0.5:
        filters.append('atempo=0.5')
        rate /= 0.5
    while rate > 2.0:
        filters.append('atempo=2.0')
        rate /= 2.0
    filters.append(f'atempo={rate:.4f}')
    filter_str = ','.join(filters)
    cmd = [
        ffmpeg_path, '-y', '-i', audio_path,
        '-filter:a', filter_str,
        temp_audio
    ]
    subprocess.run(cmd, check=True)
    return temp_audio


def dub_video_main(input_video_path, target_lang):
    """
    input_video_path: str (video dosyasının yolu)
    target_lang: str (örn: 'en', 'tr', 'zh')
    """
    # --- Unique klasör oluştur ---
    session_id = str(uuid.uuid4())
    base_dir = os.path.join('outputs', session_id)
    os.makedirs(base_dir, exist_ok=True)
    # Video ana dizindeyse, bu klasöre kopyala
    video_base = os.path.basename(input_video_path)
    local_video_path = os.path.join(base_dir, video_base)
    if os.path.abspath(input_video_path) != os.path.abspath(local_video_path):
        import shutil
        shutil.copy(input_video_path, local_video_path)
    input_video_path = local_video_path
    # Dosya adları (hep unique path)
    base = os.path.splitext(video_base)[0]
    input_audio = os.path.join(base_dir, f"{base}_audio.wav")
    output_audio = os.path.join(base_dir, f"{base}_dub_{target_lang}.wav")
    output_video = os.path.join(base_dir, f"{base}_dubbed_{target_lang}.mp4")

    # .env oku
    load_dotenv()
    API_KEY = os.getenv('NVIDIA_API_KEY')
    if not API_KEY:
        raise RuntimeError('NVIDIA_API_KEY tanımlı değil!')

    def translate_with_retry(llm_url, headers, payload, timeout=30, max_attempts=5, wait=0):
        attempt = 1
        while attempt <= max_attempts:
            try:
                resp = requests.post(llm_url, headers=headers, json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp
            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
                print(f"⏳ Timeout! LLM isteği tekrar deneniyor... ({attempt}/{max_attempts})")
            except requests.exceptions.ConnectionError:
                print(f"🌐 Bağlantı hatası! LLM isteği tekrar deneniyor... ({attempt}/{max_attempts})")
            except Exception as e:
                print(f"❌ Diğer hata ({attempt}/{max_attempts}): {e}")
            attempt += 1
            if wait:
                time.sleep(wait)
        raise RuntimeError("❗ Maksimum deneme sayısı aşıldı, LLM isteği başarısız oldu.")

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    if not os.path.exists(ffmpeg_path):
        raise RuntimeError(f"❗ ffmpeg binary bulunamadı: {ffmpeg_path}")

    # --- 1. Videodan ses çıkar ---
    print(f"🎬 Videodan ses çıkarılıyor: {input_video_path} → {input_audio}")
    cmd = [ffmpeg_path, '-y', '-i', input_video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', input_audio]
    subprocess.run(cmd, check=True)

    # --- 2. Whisper ile metne dök ---
    print(f"🎙️ Ses dosyası ({input_audio}) metne dönüştürülüyor (Whisper Local)...")
    model = whisper.load_model('large-v3', device='cuda')
    result = model.transcribe(input_audio, language=None)
    detected_lang = result.get('language', 'bilinmiyor')
    original_text = result['text'].strip()
    print("Algılanan dil:", detected_lang)
    print("→ Orijinal metin (algılandı):", original_text)

    # --- 3. LLM ile hedef dile çevir ---
    print(f"\n📚 Metin {target_lang} diline çevriliyor...")
    llm_url = 'https://integrate.api.nvidia.com/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    system_prompt = (
        f"You are a professional translator specializing in localization for sports commentary.\n"
        f"Your sole task is to translate the provided text into {target_lang}, maintaining the original context and tone.\n"
        f"Do not address the user directly.\n"
        f"Do not add any explanations, comments, or conversational phrases to your output, other than standard punctuation and capitalization if needed.\n"
        f"Be accurate and natural in your translation, especially regarding sports-related terms.\n"
        f"Do not introduce any slang or non-sports commentary elements.\n"
        f"Your output should be ONLY the translation of the input text, without any additional text or formatting."
    )
    payload = {
        'model': 'meta/llama-4-maverick-17b-128e-instruct',
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': original_text}
        ],
        'temperature': 0.2
    }
    resp = translate_with_retry(llm_url, headers, payload, timeout=30, max_attempts=5, wait=0)
    translated_text = resp.json()['choices'][0]['message']['content'].strip()
    print(f"→ {target_lang} çeviri:", translated_text)

    # --- 4. TTS ile seslendir ---
    print(f"\n🎧 {target_lang.upper()} ses oluşturuluyor (Coqui TTS)...")
    MODEL_NAME = 'tts_models/multilingual/multi-dataset/xtts_v2'
    tts = TTS(MODEL_NAME)
    tts.to('cuda')
    chunks = split_sentences(translated_text, max_chars=240)
    wav_files = []
    for idx, chunk in enumerate(chunks):
        wav_name = os.path.join(base_dir, f'{base}_tts_part_{idx}_{target_lang}.wav')
        print(f"  > Parça {idx+1}/{len(chunks)}: {chunk[:60]}...")
        tts.tts_to_file(
            text=chunk,
            file_path=wav_name,
            speaker_wav=input_audio,
            language=target_lang
        )
        wav_files.append(wav_name)
    merge_wavs(wav_files, out_path=output_audio)
    for w in wav_files:
        os.remove(w)
    print(f"✅ Ses dosyası oluşturuldu → {output_audio}")

    # --- 5. Ses-video senkronizasyonu ---
    video_duration = get_media_duration(input_video_path)
    adjusted_audio = adjust_audio_length(output_audio, video_duration, ffmpeg_path, base_dir, f"{base}_{target_lang}")
    if adjusted_audio != output_audio:
        output_audio = adjusted_audio
        print(f"🔄 Dublaj sesi video süresine uyarlandı: {output_audio}")

    # --- 6. Yeni sesi videoya göm ---
    print(f"🎥 Yeni sesi videoya gömülüyor: {input_video_path} + {output_audio} → {output_video}")
    cmd = [
        ffmpeg_path, '-y',
        '-i', input_video_path,
        '-i', output_audio,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest', output_video
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ Video dublaj ile tamamlandı: {output_video}")

    # Sonuç video dosyasının yolunu döndür (Gradio için)
    return output_video

# Eğer bu dosya terminalden direkt çalıştırılırsa (manuel test için)
if __name__ == "__main__":
    test_video = 'input_video.mp4'
    test_target_lang = 'en'
    output = dub_video_main(test_video, test_target_lang)
    print("Dublajlı video oluşturuldu:", output)
