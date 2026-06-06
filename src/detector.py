import torch
import cv2
import numpy as np
from PIL import Image
from src.model import load_model
from src.preprocess import preprocess_pil, INFER_TRANSFORMS

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

    # Try progressively more permissive detection
    faces = []
    for scale, neighbors, minsize in [
        (1.1, 5, 80),
        (1.05, 3, 50),
        (1.3, 2, 30),
    ]:
        faces = cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(minsize, minsize))
        if len(faces) > 0:
            break

    if len(faces) == 0:
        # No face found — center-crop the image (better than full resize)
        h, w = img.shape[:2]
        side = min(h, w)
        y1 = (h - side) // 2
        x1 = (w - side) // 2
        return Image.fromarray(img[y1:y1+side, x1:x1+side])

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.30 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    return Image.fromarray(img[y1:y2, x1:x2])


class Detector:
    def __init__(self, weights_path: str, model_name: str = "efficientnet_b4", device: str = "cpu"):
        self.device = device
        self.model = load_model(weights_path, model_name=model_name, device=device)

    def predict(self, image: Image.Image) -> dict:
        face = _crop_face(image)
        face_flipped = face.transpose(Image.FLIP_LEFT_RIGHT)

        probs = []
        for img in [face, face_flipped]:
            tensor = preprocess_pil(img, INFER_TRANSFORMS).to(self.device)
            with torch.no_grad():
                logit = self.model(tensor)
                probs.append(torch.sigmoid(logit).item())

        real_prob = sum(probs) / len(probs)  # model outputs P(real) — ImageFolder: fake=0, real=1
        fake_prob = 1.0 - real_prob
        label = "FAKE" if fake_prob >= 0.50 else "REAL"
        confidence = fake_prob if label == "FAKE" else real_prob
        return {"label": label, "confidence": round(confidence, 4), "fake_prob": round(fake_prob, 4)}
