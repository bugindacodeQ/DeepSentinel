import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import torch


@patch("src.detector.load_model")
def test_predict_returns_label(mock_load):
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.8]])
    mock_load.return_value = mock_model

    from src.detector import Detector
    detector = Detector.__new__(Detector)
    detector.device = "cpu"
    detector.model = mock_model

    image = Image.new("RGB", (224, 224))
    result = detector.predict(image)

    assert result["label"] in ("REAL", "FAKE")
    assert 0.0 <= result["fake_prob"] <= 1.0
