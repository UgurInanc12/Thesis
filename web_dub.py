import gradio as gr
from dub_video import dub_video_main  # Ana fonksiyonunu böyle import edebilirsin

def gradio_wrapper(input_video, target_lang):
    # input_video: Dosya yolunu içerir (Gradio otomatik olarak dosyayı temp klasöre atar)
    # target_lang: Kullanıcı seçimi ("en", "tr", "pl" vb.)
    output = dub_video_main(input_video, target_lang)
    return output

iface = gr.Interface(
    fn=gradio_wrapper,
    inputs=[
        gr.Video(label="Video Yükle"),
        gr.Dropdown(choices=["en", "tr", "pl"], label="Hedef Dil")
    ],
    outputs=gr.Video(label="Dublajlı Video"),
    title="AI Video Dublaj Web Arayüzü",
    description="Videonu yükle, dublaj dilini seç, sonuç videoyu indir."
)

iface.launch(server_port=3080)
