#!/usr/bin/env python3
"""
app.py
FastAPI server that rebuilds the same architecture and loads weights via model.load_weights(...)
No SavedModel required.
"""

import os
import io
import json
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---- config (edit if needed) ----
BASE_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS = BASE_DIR / "model" / "lung_cancer_model_best.keras"
LABELS_PATH = BASE_DIR / "model" / "labels.json"

BACKBONE = "b3"
IMG_SIZE = 300
NUM_CLASSES = 3
HEAD_UNITS = 512
HEAD_DROPOUT = 0.35

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# logger
logger = logging.getLogger("uvicorn.error")

def build_effnet_backbone(backbone='b3', img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                          head_units=HEAD_UNITS, head_dropout=HEAD_DROPOUT):
    inp = keras.Input(shape=(img_size, img_size, 3), name='input_image', dtype=tf.float32)
    x = layers.Rescaling(1.0/127.5, offset=-1.0, name="effnet_preprocess")(inp)

    if backbone.lower() == 'b1':
        Base = tf.keras.applications.EfficientNetB1
        weights_file = "efficientnetb1_notop.h5"
        weights_url = "https://storage.googleapis.com/keras-applications/efficientnetb1_notop.h5"
    else:
        Base = tf.keras.applications.EfficientNetB3
        weights_file = "efficientnetb3_notop.h5"
        weights_url = "https://storage.googleapis.com/keras-applications/efficientnetb3_notop.h5"

    base = Base(include_top=False, weights=None, input_shape=(img_size, img_size, 3), pooling='avg')
    try:
        weights_path = tf.keras.utils.get_file(weights_file, origin=weights_url, cache_subdir="models")
        base.load_weights(weights_path)
    except Exception:
        pass

    base.trainable = False
    feats = base(x, training=False)

    h = layers.BatchNormalization()(feats)
    h = layers.Dense(head_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-5))(h)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(head_dropout)(h)
    out = layers.Dense(num_classes, dtype='float32', name="logits")(h)

    model = keras.Model(inputs=inp, outputs=out, name=f"EffNet_{backbone}_head")
    return model

# build model and load weights
logger.info("Building model architecture...")
model = build_effnet_backbone(BACKBONE, IMG_SIZE)
if not MODEL_WEIGHTS.exists():
    raise RuntimeError(f"Model weights not found: {MODEL_WEIGHTS}")

logger.info("Attempting to load weights into architecture from %s", MODEL_WEIGHTS)
try:
    model.load_weights(str(MODEL_WEIGHTS))
    logger.info("Weights loaded via model.load_weights()")
except Exception as e:
    logger.warning("model.load_weights failed: %s. Trying fallback full-model load...", e)
    try:
        full = tf.keras.models.load_model(str(MODEL_WEIGHTS), compile=False, safe_mode=False)
        model.set_weights(full.get_weights())
        logger.info("Weights loaded from full-model fallback")
    except Exception as e2:
        logger.exception("Failed to load weights by any method: %s", e2)
        raise RuntimeError("Could not load model weights") from e2

# ensure weights float32
for w in model.weights:
    if w.dtype != tf.float32:
        try:
            w.assign(tf.cast(w, tf.float32))
        except Exception:
            tf.keras.backend.set_value(w, w.numpy().astype(np.float32))

# load labels
if Path(LABELS_PATH).exists():
    with open(LABELS_PATH, "r") as fh:
        labels = json.load(fh)
else:
    labels = {}

# create app
app = FastAPI(title="Lung Cancer Prediction API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bc4afe70-2db6-4641-bc8d-53bfbae6d369.lovableproject.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_pil(image: Image.Image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32)  # 0..255
    return np.expand_dims(arr, axis=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        image = Image.open(io.BytesIO(data))
        x = preprocess_pil(image)  # 0..255
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        preds = model.predict(x, verbose=0)[0]  # logits
        probs = tf.nn.softmax(preds).numpy()
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    top_idx = int(np.argmax(probs))
    top_label = labels.get(str(top_idx), f"class_{top_idx}")

    return {
        "class": top_label,
        "confidence": float(probs[top_idx]),
        "all_probabilities": { labels.get(str(i), f"class_{i}"): float(p) for i,p in enumerate(probs) }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
