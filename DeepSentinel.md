# DeepSentinel — AI-Powered Deepfake Image Detection

A deepfake image detection system built on pretrained convolutional and transformer-based
models (EfficientNet-B4, Xception). Trained on Colab, deployable as a Gradio demo on
Hugging Face Spaces with a FastAPI backend for REST inference.

---

## Project Structure

```
DeepSentinel/
├── data/
│   ├── raw/                        # Raw dataset downloads (FaceForensics++, DFDC)
│   │   ├── real/                   # Authentic face images
│   │   └── fake/                   # Deepfake/manipulated images
│   ├── processed/                  # Cleaned, cropped & resized images
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── samples/                    # Small demo samples for quick testing
│
├── models/
│   ├── weights/                    # Downloaded pretrained weights
│   │   ├── efficientnet_b4.pth
│   │   └── xception.pth
│   └── exports/                    # Exported models (ONNX / TorchScript)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Dataset stats, class balance, sample viz
│   ├── 02_preprocessing.ipynb      # Face detection & augmentation pipeline
│   ├── 03_training.ipynb           # Colab training notebook (GPU)
│   └── 04_evaluation.ipynb         # Metrics: AUC, F1, confusion matrix
│
├── src/
│   ├── __init__.py
│   ├── model.py                    # Model definitions (EfficientNet-B4, Xception)
│   ├── detector.py                 # Core inference & prediction logic
│   ├── preprocess.py               # Face detection (MTCNN) & image transforms
│   ├── train.py                    # Training loop & checkpointing
│   ├── evaluate.py                 # Evaluation script & metric computation
│   └── utils.py                    # Shared helpers (logging, config loading)
│
├── app/
│   ├── gradio_app.py               # Hugging Face Spaces demo (Gradio UI)
│   └── api.py                      # FastAPI REST endpoint for inference
│
├── tests/
│   ├── test_detector.py
│   ├── test_preprocess.py
│   └── test_api.py
│
├── .gitignore
├── requirements.txt                # Core dependencies
├── requirements-colab.txt          # Colab-specific installs
└── README.md
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Model backbone | EfficientNet-B4 / Xception (PyTorch) |
| Face detection | MTCNN (facenet-pytorch) |
| Training | Google Colab (free GPU) |
| Model registry | Hugging Face Hub |
| Demo UI | Gradio → Hugging Face Spaces |
| API | FastAPI |
| Versioning | GitHub |

---

## Datasets

- [FaceForensics++](https://github.com/ondyari/FaceForensics) — manipulated video frames
- [DFDC (Deepfake Detection Challenge)](https://ai.facebook.com/datasets/dfdc/) — Meta's large-scale dataset
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) — high-quality celebrity deepfakes
