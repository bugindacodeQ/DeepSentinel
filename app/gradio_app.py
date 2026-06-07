import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import gradio as gr
from src.detector import Detector

WEIGHTS_PATH = Path("models/weights/efficientnet_b4.pth")
SAMPLES_DIR  = Path("data/samples")
_detector    = None


def _download_weights_if_needed():
    if WEIGHTS_PATH.exists():
        return
    hf_repo  = os.getenv("HF_REPO_ID")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_repo:
        raise FileNotFoundError(
            f"Weights not found at {WEIGHTS_PATH} and HF_REPO_ID env var is not set.\n"
            "Set HF_REPO_ID=your_username/deepsentinel-weights in Render environment variables."
        )
    print(f"Downloading weights from HuggingFace: {hf_repo} ...")
    from huggingface_hub import hf_hub_download
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=hf_repo,
        filename="efficientnet_b4.pth",
        local_dir=str(WEIGHTS_PATH.parent),
        token=hf_token,
    )
    print("Weights downloaded successfully.")


def get_detector():
    global _detector
    if _detector is None:
        _download_weights_if_needed()
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
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
