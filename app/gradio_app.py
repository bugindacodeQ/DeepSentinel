import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import gradio as gr
from src.detector import Detector

WEIGHTS_PATH = Path("models/weights/efficientnet_b4.pth")
SAMPLES_DIR  = Path("data/samples")
_detector    = None


def get_detector():
    global _detector
    if _detector is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"Model weights not found at {WEIGHTS_PATH}.\n"
                "Train the model first (notebooks/03_training.ipynb on Kaggle),\n"
                "or download weights: python scripts/download_weights.py --repo YOUR_HF_REPO"
            )
        _detector = Detector(weights_path=str(WEIGHTS_PATH))
    return _detector


def predict(image):
    if image is None:
        return {}
    try:
        result = get_detector().predict(image)
    except FileNotFoundError as e:
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
    demo.launch(share=False)
