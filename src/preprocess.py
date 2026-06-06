from PIL import Image
import numpy as np
import torchvision.transforms as T

IMAGE_SIZE = 224

TRAIN_TRANSFORMS = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

INFER_TRANSFORMS = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(image_path: str, transform=None) -> "torch.Tensor":
    img = Image.open(image_path).convert("RGB")
    transform = transform or INFER_TRANSFORMS
    return transform(img).unsqueeze(0)


def preprocess_pil(image: Image.Image, transform=None) -> "torch.Tensor":
    transform = transform or INFER_TRANSFORMS
    return transform(image).unsqueeze(0)
