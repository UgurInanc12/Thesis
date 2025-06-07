#!/usr/bin/env python
import os, sys, subprocess, pathlib, requests, textwrap
from dotenv import load_dotenv
load_dotenv()
# -------------------- ÇEVİRİ (LLM) --------------------
API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    sys.exit("Hata: NVIDIA_API_KEY tanımlı değil!")

turkish_text = input("Türkçe metni girin: ")

llm_url = "https://integrate.api.nvidia.com/v1/chat/completions"
payload = {
    "model": "meta/llama-4-maverick-17b-128e-instruct",
    "messages": [
        {"role": "system",
         "content": "Translate the following Turkish text to English."},
        {"role": "user", "content": turkish_text}
    ],
    "temperature": 0.2
}
headers = {"Authorization": f"Bearer {API_KEY}",
           "Content-Type": "application/json"}

resp = requests.post(llm_url, headers=headers, json=payload)
resp.raise_for_status()
english_text = resp.json()["choices"][0]["message"]["content"].strip()
print("→ English:", english_text)

# -------------------- TTS (Magpie) --------------------
# python-clients deposunun yolu (proje klasörüne klonladığınızı varsayar)
talk_py = pathlib.Path(__file__).parent / "python-clients" / "scripts" / "tts" / "talk.py"
if not talk_py.exists():
    sys.exit("Hata: python-clients repo'sundaki talk.py bulunamadı!")

cmd = [
    sys.executable, str(talk_py),
    "--server", "grpc.nvcf.nvidia.com:443",  # bulut Riva
    "--use-ssl",
    "--metadata", "function-id", "877104f7-e885-42b9-8de8-f6e4c6303969",
    "--metadata", "authorization", f"Bearer {API_KEY}",
    "--language-code", "en-US",
    "--voice", "Magpie-Multilingual.EN-US.Sofia",
    "--text", english_text,
    "--output", "dub.wav"
]

print("⏩  TTS çağrısı yapılıyor…")
subprocess.run(cmd, check=True)
print("✅  Ses dosyası oluşturuldu → dub.wav")
