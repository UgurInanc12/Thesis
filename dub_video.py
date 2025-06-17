#!/usr/bin/env python
import os
import sys
import subprocess
import requests
import whisper
import time
import re
import uuid
import argparse
import torch
import shutil
from datetime import datetime
from dotenv import load_dotenv
from TTS.api import TTS
from pydub import AudioSegment

# FFmpeg ve ffprobe sistem PATH'inden direkt çağrılacak şekilde sabit tanımlar
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"
LLM_TRANSLATE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
WHISPER_MODEL = "large-v3"

# Yardımcı fonksiyonlar
def setup_logging(session_dir):
    """Log dosyası ve konsol çıktısı için setup"""
    log_file = os.path.join(session_dir, 'log.txt')
    
    class Logger:
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_file, 'a', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger()
    sys.stderr = sys.stdout
    
    print(f"### Oturum Başlatıldı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ###")
    print(f"### Oturum Dizini: {session_dir} ###")

def log_step(step_name, details=""):
    """Zaman damgalı log kaydı"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] [{step_name}] {details}")

def get_media_duration(file_path):
    cmd = [
        FFPROBE_BIN, '-i', file_path,
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

def extract_audio_segment(video_path, start_sec, end_sec, output_wav):
    """Videodan belirli zaman dilimini keser (tek kanallı)"""
    subprocess.run([
        FFMPEG_BIN, '-y',
        '-i', video_path,
        '-ss', str(start_sec),
        '-to', str(end_sec),
        '-ac', '1',  # Mono ses
        '-ar', '22050',  # Standart örnekleme oranı
        '-q:a', '0',
        '-map', '0:a:0',  # Sadece ilk ses kanalını al
        output_wav
    ], check=True)

def extract_audio_for_whisper(input_video_path, output_wav):
    """Whisper için tek kanallı ses çıkarır"""
    subprocess.run([
        FFMPEG_BIN, '-y',
        '-i', input_video_path,
        '-ac', '1',  # Tek kanal (mono)
        '-ar', '16000',  # 16kHz örnekleme oranı
        '-q:a', '0',
        '-map', '0:a:0',  # Sadece ilk ses kanalını al
        output_wav
    ], check=True)

def time_stretch_audio(input_wav, output_wav, target_duration):
    """Sesi hedef süreye göre hız ayarlar"""
    orig_duration = get_media_duration(input_wav)
    
    if abs(orig_duration - target_duration) < 0.1:
        # Süreler neredeyse aynıysa kopyala
        shutil.copy(input_wav, output_wav)
        return

    speed_factor = orig_duration / target_duration
    
    filters = []
    while speed_factor < 0.5:
        filters.append('atempo=0.5')
        speed_factor /= 0.5
    while speed_factor > 2.0:
        filters.append('atempo=2.0')
        speed_factor /= 2.0
    filters.append(f'atempo={speed_factor:.4f}')
    filter_str = ','.join(filters)
    
    cmd = [
        FFMPEG_BIN, '-y', 
        '-i', input_wav,
        '-filter:a', filter_str,
        output_wav
    ]
    subprocess.run(cmd, check=True)

def translate_with_retry(llm_url, headers, payload, timeout=30, max_attempts=5, wait=0):
    """Çeviri için LLM'ye istek gönderir"""
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
        time.sleep(wait * attempt)  # Artan bekleme süresi
    
    raise RuntimeError("❗ Maksimum deneme sayısı aşıldı, LLM isteği başarısız oldu.")

def dub_video_main(input_video_path, target_lang, output_video="dublajli_cikti.mp4"):
    """
    input_video_path: str (video dosyasının yolu)
    target_lang: str (örn: 'en', 'tr', 'zh')
    output_video: str (çıkış video dosyası)
    """
    # --- Oturum oluştur ---
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    session_dir = os.path.join('sessions', session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Loglama sistemini başlat
    setup_logging(session_dir)
    
    # Orijinal videoyu oturum klasörüne kopyala
    video_name = os.path.basename(input_video_path)
    archived_video = os.path.join(session_dir, video_name)
    shutil.copy(input_video_path, archived_video)
    log_step("OTURUM", f"ID: {session_id} | Video: {video_name} | Hedef Dil: {target_lang}")

    # Kök dosya adı (uzantı olmadan)
    base = os.path.splitext(video_name)[0]
    
    # .env oku
    load_dotenv()
    API_KEY = os.getenv('NVIDIA_API_KEY')
    if not API_KEY:
        raise RuntimeError('NVIDIA_API_KEY tanımlı değil!')

    # --- 1. STT için sesi çıkar ---
    log_step("SES_CIKAR", f"Whisper için ses çıkarılıyor...")
    input_audio = os.path.join(session_dir, f"{base}_audio.wav")
    extract_audio_for_whisper(archived_video, input_audio)
    log_step("SES_CIKAR", f"Ses dosyası oluşturuldu: {input_audio}")

    # --- 2. Whisper ile metne dök ve segmentleri al ---
    log_step("WHISPER", f"Başlıyor...")
    model = whisper.load_model(WHISPER_MODEL, device='cuda')
    result = model.transcribe(input_audio, word_timestamps=True)
    segments = result['segments']
    log_step("WHISPER", f"{len(segments)} segment bulundu")
    
    # Segment detaylarını logla
    for i, seg in enumerate(segments):
        log_step(f"SEGMENT_{i+1}", f"[{seg['start']:.1f}s-{seg['end']:.1f}s] Süre: {seg['end']-seg['start']:.1f}s")
        print(f"  Orijinal Metin: {seg['text']}")
        
        # Segment metnini dosyaya kaydet
        with open(os.path.join(session_dir, f'segment_{i+1}_original.txt'), 'w', encoding='utf-8') as f:
            f.write(seg['text'])

    # --- 3. Segmentleri çevir ---
    log_step("CEVIRI", f"{target_lang} diline çeviri başlıyor...")
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    for i, seg in enumerate(segments):
        # Spor içeriği için özel prompt
        system_prompt = (
            f"You are an expert sports commentator. Provide an accurate, natural-sounding translation "
            f"of the following commentary to {target_lang}. Maintain all sports terminology and excitement.\n"
            f"Output ONLY the translation without any additional text."
        )
        
        payload = {
            'model': 'meta/llama-4-maverick-17b-128e-instruct',
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': seg['text'].strip()}
            ],
            'temperature': 0.2
        }
        
        try:
            resp = translate_with_retry(
                LLM_TRANSLATE_URL, 
                headers, 
                payload,
                timeout=30,
                max_attempts=5,
                wait=1
            )
            translated = resp.json()['choices'][0]['message']['content'].strip()
            segments[i]['translated'] = translated
            
            log_step(f"CEVRI_{i+1}", f"{translated}")
            
            # Çevrilmiş metni dosyaya kaydet
            with open(os.path.join(session_dir, f'segment_{i+1}_translated.txt'), 'w', encoding='utf-8') as f:
                f.write(translated)
                
        except Exception as e:
            print(f"❌ Segment {i+1} çevirisi başarısız: {str(e)}")
            segments[i]['translated'] = "[ÇEVİRİ HATASI] " + seg['text']

    # --- 4. TTS için hazırlık ---
    log_step("TTS", f"{target_lang.upper()} ses sentezi başlatılıyor (Coqui TTS)...")
    MODEL_NAME = 'tts_models/multilingual/multi-dataset/xtts_v2'
    tts = TTS(MODEL_NAME)
    tts.to('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 5. Segment bazlı ses oluşturma ---
    video_duration = get_media_duration(archived_video)
    final_audio = AudioSegment.silent(duration=int(video_duration * 1000))
    
    for idx, seg in enumerate(segments):
        seg_start = seg['start']
        seg_end = seg['end']
        seg_duration = seg_end - seg_start
        
        # Çok kısa segmentleri (0.3s'den kısa) atla
        if seg_duration < 0.3:
            log_step(f"SEGMENT_{idx+1}", f"Atlandı (çok kısa süre: {seg_duration:.2f}s)")
            continue
            
        log_step(f"SEGMENT_{idx+1}", f"İşleniyor [{seg_start:.1f}s - {seg_end:.1f}s]")
        
        try:
            # Referans sesi kes
            ref_audio = os.path.join(session_dir, f'ref_{idx}.wav')
            extract_audio_segment(archived_video, seg_start, seg_end, ref_audio)
            log_step(f"SEGMENT_{idx+1}", f"Referans ses oluşturuldu: {ref_audio}")
            
            # TTS ile ses sentezi
            tts_audio = os.path.join(session_dir, f'tts_{idx}.wav')
            log_step(f"TTS_{idx+1}", f"Metin: {seg['translated'][:200]}{'...' if len(seg['translated'])>200 else ''}")
            
            # TTS metnini dosyaya kaydet
            with open(os.path.join(session_dir, f'tts_{idx}_text.txt'), 'w', encoding='utf-8') as f:
                f.write(seg['translated'])
            
            tts.tts_to_file(
                text=seg['translated'],
                file_path=tts_audio,
                speaker_wav=ref_audio,
                language=target_lang
            )
            log_step(f"TTS_{idx+1}", f"Ses dosyası oluşturuldu: {tts_audio}")
            
            # Süre ayarlama
            target_duration = seg_end - seg_start
            adjusted_audio = os.path.join(session_dir, f'adj_{idx}.wav')
            time_stretch_audio(tts_audio, adjusted_audio, target_duration)
            log_step(f"SEGMENT_{idx+1}", f"Süre ayarlandı: {adjusted_audio}")
            
            # Final sesine yerleştir
            seg_audio = AudioSegment.from_wav(adjusted_audio)
            final_audio = final_audio.overlay(seg_audio, position=int(seg_start*1000))
            
        except Exception as e:
            log_step("HATA", f"Segment #{idx+1} işlenirken hata: {str(e)}")
    
    # 6. Final sesini kaydet
    final_audio_path = os.path.join(session_dir, f'{base}_final_audio.wav')
    final_audio.export(final_audio_path, format='wav')
    log_step("SES_BIRLESTIRME", f"Final ses dosyası oluşturuldu: {final_audio_path}")

    # --- 7. Yeni sesi videoya göm ---
    log_step("VIDEO_OLUSTURMA", f"Yeni sesi videoya gömülüyor: {archived_video} + {final_audio_path} → {output_video}")
    
    # Final videoyu oturum dizinine kaydet
    session_output = os.path.join(session_dir, os.path.basename(output_video))
    
    cmd = [
        FFMPEG_BIN, '-y',
        '-i', archived_video,
        '-i', final_audio_path,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest', 
        session_output
    ]
    try:
        subprocess.run(cmd, check=True)
        log_step("TAMAMLANDI", f"Final video oluşturuldu: {session_output}")
        
        # Final videoyu orijinal çıktı konumuna kopyala
        shutil.copy(session_output, output_video)
        log_step("TAMAMLANDI", f"Video kopyalandı: {output_video}")
        
    except subprocess.CalledProcessError as e:
        log_step("HATA", f"FFmpeg hatası: {e}")
        raise

    # Oturum dizinini logla
    log_step("OTURUM", "Tüm işlemler tamamlandı. Ara dosyalar oturum dizininde saklandı.")
    print("\n### TÜM ARA DOSYALAR OTURUM KLASÖRÜNDE SAKLANDI ###")
    print(f"### Yol: {os.path.abspath(session_dir)} ###")
    
    return session_output

# CLI için ana fonksiyon
def main_cli():
    parser = argparse.ArgumentParser(description='Video dublaj aracı')
    parser.add_argument('input_video', help='Giriş video dosyası')
    parser.add_argument('target_lang', help='Hedef dil kodu (tr, en, de, ...)')
    parser.add_argument('output_video', nargs='?', default='dublajli_cikti.mp4', 
                        help='Çıkış video dosyası (varsayılan: dublajli_cikti.mp4)')
    args = parser.parse_args()
    
    dub_video_main(args.input_video, args.target_lang, args.output_video)

if __name__ == "__main__":
    main_cli()
