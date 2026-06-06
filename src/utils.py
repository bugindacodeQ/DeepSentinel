import logging
import os


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(name)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
