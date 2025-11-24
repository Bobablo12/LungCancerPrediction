#!/usr/bin/env python3
"""
Minimal CLI inference for Lung Cancer model.
Usage:
    python inference.py /path/to/image.jpg
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---- Config ----
MODEL_WEIGHTS = "model/lung_cancer_model_best.keras"
LABELS_PATH   = "model/labels.json"
BACKBONE = "b3"
IMG_SIZE = 300
NUM_CLASSES = 3
HEAD_UNITS = 512
HEAD_DROPOUT = 0.35

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---- Helper Functions ----
def build_model():
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    x = layers.Rescaling(1.0/127.5, offset=-1.0)(inp)  # -1..1

    if BACKBONE.lower() == 'b1':
        Base = keras.applications.EfficientNetB1
    else:
        Base = keras.applications.EfficientNetB3

    base = Base(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
    base.trainable = False
    feats = base(x, training=False)

    h = layers.BatchNormalization()(feats)
    h = layers.Dense(HEAD_UNITS, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-5))(h)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(HEAD_DROPOUT)(h)
    out = layers.Dense(NUM_CLASSES, dtype='float32')(h)

    model = keras.Model(inputs=inp, outputs=out)
    return model

def load_labels(path=LABELS_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {str(i): f"class_{i}" for i in range(NUM_CLASSES)}

def preprocess_image(img_path):
    with Image.open(img_path) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

# ---- Main CLI ----
def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py /path/to/image.jpg")
        return 1

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print("ERROR: image not found:", img_path)
        return 2

    labels = load_labels()
    model = build_model()

    if not os.path.exists(MODEL_WEIGHTS):
        print("ERROR: weights not found:", MODEL_WEIGHTS)
        return 3

    model.load_weights(MODEL_WEIGHTS)

    x = preprocess_image(img_path)
    preds = model.predict(x, verbose=0)[0]
    probs = tf.nn.softmax(preds).numpy()
    top_idx = int(np.argmax(probs))
    top_label = labels.get(str(top_idx), f"class_{top_idx}")

    out = {
        "class": top_label,
        "confidence": float(probs[top_idx]),
        "all_probabilities": {labels.get(str(i), f"class_{i}"): float(p) for i, p in enumerate(probs)}
    }

    print(json.dumps(out, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
