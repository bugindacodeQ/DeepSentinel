import torch
from PIL import Image
from src.preprocess import preprocess_pil, INFER_TRANSFORMS


def test_preprocess_output_shape():
    image = Image.new("RGB", (300, 300))
    tensor = preprocess_pil(image, INFER_TRANSFORMS)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_tensor_type():
    image = Image.new("RGB", (224, 224))
    tensor = preprocess_pil(image)
    assert isinstance(tensor, torch.Tensor)
