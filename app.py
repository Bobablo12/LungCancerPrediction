# ---------------- LOAD MODEL ----------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "lung_cancer_model_best2.keras"
LABELS_PATH = BASE_DIR / "model" / "labels.json"
IMG_SIZE = 300

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable unsafe deserialization BEFORE loading
keras.config.enable_unsafe_deserialization()

# Define custom objects for Lambda and activations
def get_custom_objects():
    def swish_activation(x):
        return x * tf.sigmoid(x)

    def identity(x):
        return tf.identity(x)  # for Lambda layers

    return {
        'swish': swish_activation,
        'swish_activation': swish_activation,
        'identity': identity,
        'tf': tf,
        'Lambda': layers.Lambda,  # allow Lambda deserialization
    }

try:
    custom_objects = get_custom_objects()
    
    # Load the model safely with Lambda allowed
    model = keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
        custom_objects=custom_objects
    )
    
    # Ensure all weights are float32 (avoid precision issues)
    for w in model.weights:
        if w.dtype != tf.float32:
            w.assign(tf.cast(w, tf.float32))
    
    # Test model with dummy input
    test_input = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    _ = model.predict(test_input, verbose=0)
    
    logger.info(f"✅ Model loaded from {MODEL_PATH} successfully")
except Exception as e:
    logger.exception(f"❌ Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}") from e

# Load labels
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
logger.info(f"✅ Labels loaded from {LABELS_PATH}")
