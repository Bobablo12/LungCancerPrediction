#!/usr/bin/env python3
"""
inference.py
CLI inference using the reconstructed architecture + weights.
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

# ---- config (edit if you changed training) ----
MODEL_WEIGHTS = "model/lung_cancer_model_best.keras"   # weights file produced by your training script
LABELS_PATH   = "model/labels.json"
BACKBONE = "b3"
IMG_SIZE = 300
NUM_CLASSES = 3
HEAD_UNITS = 512
HEAD_DROPOUT = 0.35

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

def build_effnet_backbone(backbone='b3', img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                          head_units=HEAD_UNITS, head_dropout=HEAD_DROPOUT):
    inp = keras.Input(shape=(img_size, img_size, 3), name='input_image', dtype=tf.float32)
    # IMPORTANT: this matches training: rescaling to [-1,1] inside model
    x = layers.Rescaling(1.0/127.5, offset=-1.0, name="effnet_preprocess")(inp)

    if backbone.lower() == 'b1':
        Base = tf.keras.applications.EfficientNetB1
        weights_file = "efficientnetb1_notop.h5"
        weights_url = "https://storage.googleapis.com/keras-applications/efficientnetb1_notop.h5"
    elif backbone.lower() == 'b3':
        Base = tf.keras.applications.EfficientNetB3
        weights_file = "efficientnetb3_notop.h5"
        weights_url = "https://storage.googleapis.com/keras-applications/efficientnetb3_notop.h5"
    else:
        raise ValueError("backbone must be 'b1' or 'b3'")

    base = Base(include_top=False, weights=None, input_shape=(img_size, img_size, 3), pooling='avg')
    # optional: auto-download notop weights for base (safe)
    try:
        weights_path = tf.keras.utils.get_file(weights_file, origin=weights_url, cache_subdir="models")
        base.load_weights(weights_path)
    except Exception:
        # ignore if download/load fails (not required)
        pass

    base.trainable = False
    feats = base(x, training=False)

    h = layers.BatchNormalization()(feats)
    h = layers.Dense(head_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-5))(h)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(head_dropout)(h)
    out = layers.Dense(num_classes, dtype='float32', name="logits")(h)  # logits

    model = keras.Model(inputs=inp, outputs=out, name=f"EffNet_{backbone}_head")
    return model

def load_labels(path=LABELS_PATH):
    import json
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def preprocess_path(img_path):
    # match training preprocessing up to model's Rescaling: provide 0..255 float32
    with Image.open(img_path) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)  # 0..255
    return np.expand_dims(arr, axis=0)

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py /path/to/image.jpg", file=sys.stderr)
        sys.exit(2)
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print("ERROR: image not found:", img_path, file=sys.stderr)
        sys.exit(3)

    labels = load_labels()

    # build architecture and load weights
    model = build_effnet_backbone(BACKBONE, IMG_SIZE)
    if not os.path.exists(MODEL_WEIGHTS):
        print("ERROR: weights not found:", MODEL_WEIGHTS, file=sys.stderr)
        sys.exit(4)

    # load weights into architecture (weights-only load avoids deserializing lambdas)
    try:
        model.load_weights(MODEL_WEIGHTS)
    except Exception as e:
        print("Failed to load weights via model.load_weights():", e, file=sys.stderr)
        # fallback attempt: full model load (may segfault on some systems)
        try:
            full = tf.keras.models.load_model(MODEL_WEIGHTS, compile=False, safe_mode=False)
            model.set_weights(full.get_weights())
        except Exception as e2:
            print("Fallback full-model load failed too:", e2, file=sys.stderr)
            raise

    # ensure weights float32
    for w in model.weights:
        if w.dtype != tf.float32:
            try:
                w.assign(tf.cast(w, tf.float32))
            except Exception:
                tf.keras.backend.set_value(w, w.numpy().astype(np.float32))

    # predict
    x = preprocess_path(img_path)
    preds = model.predict(x, verbose=0)[0]  # logits
    probs = tf.nn.softmax(preds).numpy()
    top_idx = int(tf.argmax(probs).numpy())
    top_label = labels.get(str(top_idx), f"class_{top_idx}")

    out = {
        "class": top_label,
        "confidence": float(probs[top_idx]),
        "all_probabilities": { labels.get(str(i), f"class_{i}"): float(p) for i,p in enumerate(probs) }
    }

    print(json.dumps(out, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
