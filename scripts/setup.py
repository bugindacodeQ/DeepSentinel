"""
One-command project setup for DeepSentinel.
Run: python scripts/setup.py
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRS = [
    "data/raw/real",
    "data/raw/fake",
    "data/processed/train/real",
    "data/processed/train/fake",
    "data/processed/val/real",
    "data/processed/val/fake",
    "data/processed/test/real",
    "data/processed/test/fake",
    "data/samples",
    "models/weights",
    "models/exports",
]


def create_dirs():
    print("\n[1/3] Creating directory structure...")
    for d in DIRS:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch(exist_ok=True)
    print("      Done.")


def install_deps():
    print("\n[2/3] Installing dependencies...")
    req = ROOT / "requirements.txt"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
    print("      Done.")


def install_package():
    print("\n[3/3] Installing DeepSentinel as editable package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(ROOT)])
    print("      Done.")


def main():
    print("=" * 50)
    print("  DeepSentinel — Project Setup")
    print("=" * 50)
    create_dirs()
    install_deps()
    install_package()
    print("\n✓ Setup complete!")
    print("\nNext steps:")
    print("  1. Generate demo weights (for testing without training):")
    print("       python scripts/create_demo_weights.py")
    print("  2. Run the Gradio web app:")
    print("       python app/gradio_app.py")
    print("  3. Run the FastAPI server:")
    print("       uvicorn app.api:app --reload")
    print("\n  Or to use real deepfake-trained weights:")
    print("  1. Download a dataset (see README.md)")
    print("  2. Preprocess: run notebooks/02_preprocessing.ipynb on Colab")
    print("  3. Train: run notebooks/03_training.ipynb on Colab")
    print("  4. Copy resulting .pth file to models/weights/efficientnet_b4.pth")


if __name__ == "__main__":
    main()
