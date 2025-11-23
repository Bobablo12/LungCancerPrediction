import io
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from tensorflow import keras

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "lung_cancer_model_best.keras"
LABELS_PATH = BASE_DIR / "model" / "labels.json"
IMG_SIZE = 300

# FastAPI app
app = FastAPI()

# Enable Lambda deserialization
keras.config.enable_unsafe_deserialization()

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}\n{e}")

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# Root endpoint


# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Preprocess function
def preprocess_pil(image: Image.Image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32)
    x = tf.convert_to_tensor(arr, dtype=tf.float32)
    x = tf.expand_dims(x, 0)
    return x

# Predict endpoint
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        image = Image.open(io.BytesIO(data))
        x = preprocess_pil(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        preds = model.predict(x, verbose=0)
        probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

# Run locally
if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
