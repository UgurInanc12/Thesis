import gradio as gr
import os
import tempfile
import uuid
from dub_video import dub_video_main

def gradio_wrapper(input_video, target_lang):
    # Geçici bir çıktı dosyası oluştur
    output_filename = f"dublajli_{uuid.uuid4().hex}.mp4"
    output_video_path = os.path.join(tempfile.gettempdir(), output_filename)
    
    # Ana fonksiyonu çağır
    return dub_video_main(input_video, target_lang, output_video_path)

lang_choices = [
    "en", "tr", "pl", "ar", "bg", "ca", "cs", "da", "de", "el", 
    "es", "et", "fi", "fr", "he", "hi", "hr", "hu", "id", "it", 
    "ja", "ko", "lv", "nl", "no", "pt", "ro", "ru", "sk", "sl", 
    "sv", "sw", "th", "uk", "vi"
]

iface = gr.Interface(
    fn=gradio_wrapper,
    inputs=[
        gr.Video(label="Video Yükle", sources=["upload"], format="mp4"),
        gr.Dropdown(
            choices=lang_choices, 
            label="Hedef Dil",
            value="en",  # Varsayılan olarak Türkçe
            info="Dublaj için hedef dili seçin"
        )
    ],
    outputs=gr.Video(label="Dublajlı Video"),
    title="AI Video Dublaj Sistemi",
    description=(
        "<h3>Uzun ve kısa videolar için profesyonel dublaj çözümü</h3>"
        "Video yükleyin, hedef dili seçin. Sistem otomatik olarak:<br>"
        "1. Konuşmaları algılayacak<br>"
        "2. Metinleri çevirecek<br>"
        "3. Özgün ses tonuyla dublajlayacak<br>"
        "4. Videoyu senkronize edecek"
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(
        server_port=3080,
        server_name="127.0.0.1",
        share=False
    )
