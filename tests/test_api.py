import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@patch("app.api.Detector")
def test_health(mock_detector):
    from app.api import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.api.detector")
def test_predict_endpoint(mock_detector):
    mock_detector.predict.return_value = {"label": "REAL", "confidence": 0.92, "fake_prob": 0.08}
    from app.api import app
    client = TestClient(app)

    img = Image.new("RGB", (224, 224))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["label"] in ("REAL", "FAKE")
