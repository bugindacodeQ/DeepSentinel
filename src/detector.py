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

def _crop_face(pil_image: Image.Image) -> Image.Image:
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascade = _get_cascade()

    faces = []
    for scale, neighbors, minsize in [
        (1.1, 5, 80),
        (1.05, 3, 50),
        (1.3, 2, 30),
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
        return Image.fromarray(img[y1:y1+side, x1:x1+side])

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.40 * max(w, h))  # slightly more context than before
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    return Image.fromarray(img[y1:y2, x1:x2])


def _tta_variants(face: Image.Image):
    """10 augmented variants that mirror the training distribution."""
    variants = [face]

    # horizontal flip (same as training RandomHorizontalFlip)
    variants.append(face.transpose(Image.FLIP_LEFT_RIGHT))

    # small rotations — faces are never perfectly straight in real photos
    for angle in [-10, -5, 5, 10]:
        variants.append(face.rotate(angle, resample=Image.BILINEAR, expand=False))

    # brightness shifts matching training ColorJitter(brightness=0.2)
    for factor in [0.85, 1.15]:
        variants.append(ImageEnhance.Brightness(face).enhance(factor))

    # contrast shifts matching training ColorJitter(contrast=0.2)
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

        raw_avg = sum(probs) / len(probs)
        print(f"[DEBUG] raw sigmoid avg={raw_avg:.4f}  min={min(probs):.4f}  max={max(probs):.4f}")

        # ImageFolder alphabetical: fake=0, real=1 → sigmoid = P(real)
        real_prob = raw_avg
        fake_prob = 1.0 - real_prob

        label = "FAKE" if fake_prob >= 0.65 else "REAL"
        confidence = fake_prob if label == "FAKE" else real_prob
        return {"label": label, "confidence": round(confidence, 4), "fake_prob": round(fake_prob, 4)}
