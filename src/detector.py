import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from src.model import load_model
from src.preprocess import INFER_TRANSFORMS

_face_cascade = None

def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def _clahe_normalize(img_rgb: np.ndarray) -> np.ndarray:
    """Normalize lighting via CLAHE on the L channel — helps dark/unevenly-lit photos."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _crop_face(pil_image: Image.Image) -> Image.Image:
    img = np.array(pil_image.convert("RGB"))

    # Normalize lighting before face detection — improves detection in dark/dim photos
    img_norm = _clahe_normalize(img)
    gray = cv2.cvtColor(img_norm, cv2.COLOR_RGB2GRAY)
    cascade = _get_cascade()

    faces = []
    for scale, neighbors, minsize in [
        (1.1, 5, 80),
        (1.05, 3, 50),
        (1.3, 2, 30),
        (1.05, 2, 20),  # extra permissive pass for dark/challenging photos
    ]:
        faces = cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(minsize, minsize)
        )
        if len(faces) > 0:
            break

    if len(faces) == 0:
        h, w = img.shape[:2]
        side = min(h, w)
        y1 = (h - side) // 2
        x1 = (w - side) // 2
        crop = img_norm[y1:y1+side, x1:x1+side]
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.40 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        crop = img_norm[y1:y2, x1:x2]

    return Image.fromarray(crop)


def _tta_variants(face: Image.Image):
    """10 augmented variants that mirror the training distribution."""
    variants = [face]
    variants.append(face.transpose(Image.FLIP_LEFT_RIGHT))
    for angle in [-10, -5, 5, 10]:
        variants.append(face.rotate(angle, resample=Image.BILINEAR, expand=False))
    for factor in [0.85, 1.15]:
        variants.append(ImageEnhance.Brightness(face).enhance(factor))
    for factor in [0.85, 1.15]:
        variants.append(ImageEnhance.Contrast(face).enhance(factor))
    return variants  # 10 total


class Detector:
    def __init__(self, weights_path: str, model_name: str = "efficientnet_b4", device: str = "cpu"):
        self.device = device
        self.model = load_model(weights_path, model_name=model_name, device=device)

    def predict(self, image: Image.Image) -> dict:
        face = _crop_face(image)

        probs = []
        for variant in _tta_variants(face):
            tensor = INFER_TRANSFORMS(variant).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logit = self.model(tensor)
                probs.append(torch.sigmoid(logit).item())

        fake_prob = sum(probs) / len(probs)  # sigmoid = P(fake): Celeb-real=class 0, Celeb-synthesis=class 1
        real_prob = 1.0 - fake_prob

        label = "FAKE" if fake_prob >= 0.65 else "REAL"
        confidence = fake_prob if label == "FAKE" else real_prob
        return {"label": label, "confidence": round(confidence, 4), "fake_prob": round(fake_prob, 4)}
