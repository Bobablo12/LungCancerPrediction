import os
import json
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

# Enable Lambda deserialization (required for your model architecture)
keras.config.enable_unsafe_deserialization()

MODEL_PATH = "model/lung_cancer_model_best.keras"
LABELS_PATH = "model/labels.json"
IMG_SIZE = 300

# Load model safely
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    raise RuntimeError(f"\n❌ Failed to load model at {MODEL_PATH}\n{e}")

# Load label map
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)


def preprocess(img_path: str) -> tf.Tensor:
    """Read and preprocess an image for EfficientNetB1 model inference."""
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"❌ Image not found: {img_path}")
    # Decode with PIL to avoid native decoder segfaults
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(im, dtype=np.float32)

    img = tf.convert_to_tensor(arr, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)
    return img


def predict(img_path: str) -> dict:
    """Run inference and return top class + full probabilities."""
    img = preprocess(img_path)

    # This keeps everything in eager mode → avoids segfaults
    preds = model.predict(img, verbose=0)

    # Softmax even if model already outputs softmax (safe no-op)
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 inference.py <image_path>")
        exit(1)

    img_path = sys.argv[1]
    result = predict(img_path)
    print(json.dumps(result, indent=2))
