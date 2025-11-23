import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import io
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import Lambda as _KerasLambda

def _lambda_passthrough_compute_output_shape(self, input_shape):
    return input_shape

# Monkeypatch only once
if not hasattr(_KerasLambda, "_cascade_patched"):
    _KerasLambda.compute_output_shape = _lambda_passthrough_compute_output_shape
    _KerasLambda._cascade_patched = True

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "lung_cancer_model_best.keras"
LABELS_PATH = BASE_DIR / "model" / "labels.json"

IMG_SIZE = 300

app = FastAPI()

# Enable Lambda deserialization (EfficientNet models sometimes need this)
tf.keras.config.enable_unsafe_deserialization()

# --- Load model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# --- Load label map ---
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)


@app.get("/")
def root():
    return {"message": "Lung Cancer Prediction API running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


def preprocess_pil(image: Image.Image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data))
        x = preprocess_pil(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        preds = model.predict(x, verbose=0)
        probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    top_idx = int(np.argmax(probs))
    top_label = labels.get(str(top_idx), f"class_{top_idx}")

    return {
        "class": top_label,
        "confidence": float(probs[top_idx]),
        "all_probabilities": {
            labels.get(str(i), f"class_{i}"): float(p)
            for i, p in enumerate(probs)
        }
    }
