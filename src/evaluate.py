import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from src.model import load_model
from src.preprocess import INFER_TRANSFORMS
from src.utils import get_logger

logger = get_logger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(weights_path: str, test_dir: str = "data/processed/test", batch_size: int = 32):
    dataset = ImageFolder(test_dir, transform=INFER_TRANSFORMS)
    nw = 0 if os.name == "nt" else 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    model = load_model(weights_path, device=DEVICE)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            probs = torch.sigmoid(model(images)).squeeze(1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    preds = (np.array(all_probs) >= 0.5).astype(int)
    auc = roc_auc_score(all_labels, all_probs)
    logger.info(f"AUC: {auc:.4f}")
    logger.info("\n" + classification_report(all_labels, preds, target_names=["REAL", "FAKE"]))
    return auc


if __name__ == "__main__":
    evaluate("models/weights/efficientnet_b4.pth")
