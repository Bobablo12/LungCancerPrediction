import os
import json
import numpy as np
import tensorflow as tf
import builtins as _builtins
from keras.layers import Lambda as _KerasLambda
from PIL import Image

# Make `tf` visible to deserialized Lambda functions
_builtins.tf = tf

def _lambda_passthrough_compute_output_shape(self, input_shape):
    return input_shape

if not hasattr(_KerasLambda, "_cascade_patched"):
    _KerasLambda.compute_output_shape = _lambda_passthrough_compute_output_shape
    _KerasLambda._cascade_patched = True

tf.keras.config.enable_unsafe_deserialization()

MODEL_PATH = "model/lung_cancer_model_best.keras"
LABELS_PATH = "model/labels.json"
IMG_SIZE = 300

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

# Load labels
with open(LABELS_PATH) as f:
    labels = json.load(f)


def preprocess(img_path):
    with Image.open(img_path) as img:
        img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(img_path):
    x = preprocess(img_path)
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 inference.py <image>")
        exit()
    print(json.dumps(predict(sys.argv[1]), indent=2))
