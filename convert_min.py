#!/usr/bin/env python3
"""
convert_min.py
- Loads model/lung_cancer_model_best.keras (unsafe if needed)
- Casts any float16 weights -> float32
- Saves both:
    - model/lung_cancer_model_best_fp32.keras (Keras HDF5/.keras copy in float32)
    - model/saved_model_fp32/         (SavedModel format for serving)
Prints clear logs and exits nonzero on fatal errors.
"""

import os
import sys
import traceback
import tensorflow as tf
import keras

keras.config.enable_unsafe_deserialization()

MODEL_IN = "model/lung_cancer_model_best.keras"
KERAS_OUT = "model/lung_cancer_model_best_fp32.keras"
SAVED_OUT = "model/saved_model_fp32"

# Make sure we run CPU and quieter logs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

def main():
    if not os.path.exists(MODEL_IN):
        print(f"ERROR: input model not found: {MODEL_IN}", file=sys.stderr)
        sys.exit(2)

    print("Loading .keras model (unsafe deserialization enabled)...")
    try:
        model = tf.keras.models.load_model(MODEL_IN, compile=False, safe_mode=False)
    except Exception as e:
        print("Failed to load .keras model with safe_mode=False:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)

    print("Casting any non-float32 weights -> float32 (in-place)...")
    try:
        for w in model.weights:
            if w.dtype != tf.float32:
                try:
                    w.assign(tf.cast(w, tf.float32))
                except Exception:
                    # fallback assign via numpy
                    tf.keras.backend.set_value(w, w.numpy().astype("float32"))
    except Exception as e:
        print("Warning: weight casting encountered an error:", e, file=sys.stderr)
        traceback.print_exc()

    # Save .keras fp32 copy
    try:
        print(f"Saving fp32 .keras -> {KERAS_OUT} ...")
        model.save(KERAS_OUT, include_optimizer=False)
        print("Saved:", KERAS_OUT)
    except Exception as e:
        print("Warning: saving .keras fp32 failed:", e, file=sys.stderr)
        traceback.print_exc()

    # Save SavedModel (recommended for serving)
    try:
        print(f"Saving SavedModel -> {SAVED_OUT} ... (this avoids reserializing lambdas)")
        tf.saved_model.save(model, SAVED_OUT)
        print("Saved SavedModel:", SAVED_OUT)
    except Exception as e:
        print("ERROR: Saving SavedModel failed:", e, file=sys.stderr)
        traceback.print_exc()
        # don't exit here — we may still serve the .keras fp32 copy
        # but signal failure
        sys.exit(4)

    print("Conversion completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
