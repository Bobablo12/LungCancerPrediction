import io
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import logging

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "lung_cancer_model_best2.keras"
LABELS_PATH = BASE_DIR / "model" / "labels.json"
IMG_SIZE = 300

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- LOAD MODEL ----------------
try:
    # No Lambda inside, preprocess will happen outside
    keras.config.enable_unsafe_deserialization()  # keep for safety
    model = keras.models.load_model(str(MODEL_PATH), compile=False)
    
    # Ensure weights are float32
    for w in model.weights:
        if w.dtype != tf.float32:
            w.assign(tf.cast(w, tf.float32))
    
    # Dummy test
    test_input = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    _ = model.predict(test_input, verbose=0)

    logger.info(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.exception(f"❌ Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}") from e

# Load labels
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
logger.info(f"✅ Labels loaded from {LABELS_PATH}")

# ---------------- INIT APP ----------------
app = FastAPI(title="Lung Cancer Prediction API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- HELPERS ----------------
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_pil(image: Image.Image) -> np.ndarray:
    """Preprocess PIL image for EfficientNet (moved outside model)."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32)
    arr = preprocess_input(arr)  # preprocess here
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)
    return arr

def run_inference(x: np.ndarray) -> dict:
    """Run inference and return class + probabilities."""
    preds = model.predict(x, verbose=0)
    probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
    top_idx = int(np.argmax(probs))
    top_label = labels.get(str(top_idx), f"class_{top_idx}")
    
    return {
        "class": top_label,
        "confidence": float(probs[top_idx]),
        "all_probabilities": {
            labels.get(str(i), f"class_{i}"): float(p) for i, p in enumerate(probs)
        }
    }

# ---------------- ROUTES ----------------
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        image = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    
    try:
        x = preprocess_pil(image)
        result = run_inference(x)
        return result
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")
