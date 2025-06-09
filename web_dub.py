import gradio as gr
from dub_video import dub_video_main  # Az önce eklediğin fonksiyonu import ediyoruz

def gradio_wrapper(input_video, target_lang):
    return dub_video_main(input_video, target_lang)

lang_choices = [
    ("en"),
    ("tr"),
    ("pl"),
    ("ar"),
    ("bg"),
    ("ca"),
    ("cs"),
    ("da"),
    ("de"),
    ("el"),
    ("es"),
    ("et"),
    ("fi"),
    ("fr"),
    ("he"),
    ("hi"),
    ("hr"),
    ("hu"),
    ("id"),
    ("it"),
    ("ja"),
    ("ko"),
    ("lv"),
    ("nl"),
    ("no"),
    ("pt"),
    ("ro"),
    ("ru"),
    ("sk"),
    ("sl"),
    ("sv"),
    ("sw"),
    ("th"),
    ("uk"),
    ("vi"),
]

iface = gr.Interface(
    fn=gradio_wrapper,
    inputs=[
        gr.Video(label="Video Yükle"),
        gr.Dropdown(choices=lang_choices, label="Hedef Dil")
    ],
    outputs=gr.Video(label="Dublajlı Video"),
    title="AI Video Dublaj Web Arayüzü",
    description="Videonu yükle, dublaj dilini seç, sonuç videoyu indir."
)

iface.launch(server_port=3080)
