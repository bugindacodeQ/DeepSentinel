import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.model import DeepfakeDetector
from src.preprocess import TRAIN_TRANSFORMS, INFER_TRANSFORMS
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    data_dir: str = "data/processed",
    weights_out: str = "models/weights/efficientnet_b4.pth",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
):
    ensure_dir("models/weights")

    train_ds = ImageFolder(f"{data_dir}/train", transform=TRAIN_TRANSFORMS)
    val_ds = ImageFolder(f"{data_dir}/val", transform=INFER_TRANSFORMS)
    # num_workers=0 on Windows avoids multiprocessing spawn issues
    nw = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw)

    model = DeepfakeDetector().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = _validate(model, val_loader, criterion)
        scheduler.step()
        logger.info(f"Epoch {epoch}/{epochs} | train_loss={total_loss/len(train_loader):.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_out)
            logger.info(f"Saved best model -> {weights_out}")


def _validate(model, loader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
            total += criterion(model(images), labels).item()
    return total / len(loader)


if __name__ == "__main__":
    train()
