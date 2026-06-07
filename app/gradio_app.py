import sys
import os

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT_DIR)

from pathlib import Path
import gradio as gr
from src.detector import Detector

WEIGHTS_PATH = Path(ROOT_DIR) / "models" / "weights" / "efficientnet_b4.pth"
SAMPLES_DIR  = Path(ROOT_DIR) / "data" / "samples"
_detector    = None

WEIGHTS_URL = "https://huggingface.co/Oveea/deepsentinel-weights/resolve/main/efficientnet_b4.pth"

def _download_weights():
    print(f"Weights path: {WEIGHTS_PATH}")
    if WEIGHTS_PATH.exists():
        print("Weights already present.")
        return
    import requests
    print(f"Downloading weights from {WEIGHTS_URL} ...")
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(WEIGHTS_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(WEIGHTS_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Done: {WEIGHTS_PATH} ({WEIGHTS_PATH.stat().st_size / 1e6:.1f} MB)")


_download_weights()


def get_detector():
    global _detector
    if _detector is None:
        _detector = Detector(weights_path=str(WEIGHTS_PATH))
    return _detector


def predict(image):
    if image is None:
        return {}
    try:
        result = get_detector().predict(image)
    except Exception as e:
        raise gr.Error(str(e))
    fake_prob = result["fake_prob"]
    real_prob = round(1 - fake_prob, 4)
    return {"FAKE": fake_prob, "REAL": real_prob}


def build_examples():
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if SAMPLES_DIR.exists():
        return [str(p) for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in exts]
    return []


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a face image", sources=["upload", "webcam", "clipboard"]),
    outputs=gr.Label(num_top_classes=2, label="Detection Result"),
    title="DeepSentinel — Deepfake Image Detector",
    description=(
        "Upload a face photo to detect whether it is **REAL** or a **DEEPFAKE**. "
        "Powered by EfficientNet-B4 trained on Celeb-DF V2."
    ),
    examples=build_examples() or None,
    flagging_mode="never",
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
