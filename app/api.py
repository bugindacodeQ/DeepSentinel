import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from pathlib import Path
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from src.detector import Detector

WEIGHTS_PATH = Path(os.getenv("WEIGHTS_PATH", "models/weights/efficientnet_b4.pth"))

_detector: Detector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector
    if not WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"Model weights not found at {WEIGHTS_PATH}. "
            "Train the model first (notebooks/03_training.ipynb) or run: "
            "python scripts/download_weights.py --repo YOUR_HF_REPO"
        )
    _detector = Detector(weights_path=str(WEIGHTS_PATH))
    yield


app = FastAPI(title="DeepSentinel API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _detector is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = _detector.predict(image)
    return JSONResponse(content=result)
