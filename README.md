# DeepSentinel — AI-Powered Deepfake Image Detection

Detects deepfake images using pretrained EfficientNet-B4 and Xception models.
Training runs on Google Colab (free GPU). Deployed as a Gradio demo on Hugging Face Spaces.

## Quickstart

```bash
pip install -r requirements.txt
python app/gradio_app.py
```

## Project Structure

See [DeepSentinel.md](DeepSentinel.md) for the full file tree and tech stack.

## Datasets

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DFDC](https://ai.facebook.com/datasets/dfdc/)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)

## Training on Colab

Open `notebooks/03_training.ipynb` in Google Colab, enable GPU runtime, and run all cells.
Trained weights are saved to `models/weights/` and pushed to Hugging Face Hub.
