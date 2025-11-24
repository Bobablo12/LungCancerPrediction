#!/usr/bin/env python3
"""
Rebuild architecture, load weights from model/lung_cancer_model_best.keras (weights-only),
cast weights to float32, and export SavedModel to model/saved_model_fp32.

This avoids deserializing Lambda layers from the .keras file.
"""

import os, sys, traceback
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---- config ----
MODEL_WEIGHTS_FILE = "model/lung_cancer_model_best.keras"
OUT_SAVEDMODEL = "model/saved_model_fp32"
IMG_SIZE = 300
NUM_CLASSES = 3
BACKBONE = "b3"   # match training BACKBONE
HEAD_UNITS = 512
HEAD_DROPOUT = 0.35

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print("TF version:", tf.__version__)
print("Weights file exists:", os.path.exists(MODEL_WEIGHTS_FILE))
if not os.path.exists(MODEL_WEIGHTS_FILE):
    print("ERROR: weights file not found:", MODEL_WEIGHTS_FILE)
    sys.exit(2)

# ---- model builder (same as your training builder) ----
def build_effnet_backbone(backbone='b3', img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                          head_units=HEAD_UNITS, head_dropout=HEAD_DROPOUT):
    inp = keras.Input(shape=(img_size, img_size, 3), name='input_image', dtype=tf.float32)
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
    # download weights for backbone notop if missing (this is just for architecture)
    try:
        weights_path = tf.keras.utils.get_file(weights_file, origin=weights_url, cache_subdir="models")
        base.load_weights(weights_path)
    except Exception as e:
        print("Warning: failed to auto-download base weights:", e)

    base.trainable = False
    feats = base(x, training=False)

    h = layers.BatchNormalization()(feats)
    h = layers.Dense(head_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-5))(h)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(head_dropout)(h)
    out = layers.Dense(num_classes, dtype='float32', name="logits")(h)

    model = keras.Model(inputs=inp, outputs=out, name=f"EffNet_{backbone}_head")
    return model, base

# ---- build model ----
print("Building model architecture...")
model, base = build_effnet_backbone(backbone=BACKBONE, img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                                   head_units=HEAD_UNITS, head_dropout=HEAD_DROPOUT)

# ---- attempt to load weights in several ways ----
loaded = False
errors = []

# 1) try load_weights (this works if .keras contains HDF5-style weights or tf.Checkpoint)
try:
    print("Attempting model.load_weights(...) from", MODEL_WEIGHTS_FILE)
    model.load_weights(MODEL_WEIGHTS_FILE)
    loaded = True
    print("✅ model.load_weights succeeded")
except Exception as e:
    errors.append(("load_weights", str(e)))
    print("model.load_weights failed:", e)

# 2) If that failed, try loading via keras.models.load_model but into a separate process
#    to avoid segfaulting main interpreter (use subprocess to isolate)
if not loaded:
    print("Attempting isolated load (subprocess) to inspect file without crashing main process...")
    import subprocess, json, tempfile
    probe_script = r'''
import sys, json, traceback
import tensorflow as tf
try:
    m = tf.keras.models.load_model(sys.argv[1], compile=False, safe_mode=False)
    print("OK_LOAD")
except Exception as e:
    traceback.print_exc()
    print("ERR_LOAD", type(e).__name__, str(e))
    sys.exit(2)
'''
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as fh:
        fh.write(probe_script)
        probe_path = fh.name
    try:
        proc = subprocess.run([sys.executable, probe_path, MODEL_WEIGHTS_FILE],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120, text=True)
        out = proc.stdout
        print("Subprocess output:\n", out)
        if "OK_LOAD" in out:
            print("Subprocess was able to load the full model (unexpected). Trying to load into main process...")
            try:
                m2 = tf.keras.models.load_model(MODEL_WEIGHTS_FILE, compile=False, safe_mode=False)
                model.set_weights(m2.get_weights())
                loaded = True
                print("✅ Loaded weights via full-model route")
            except Exception as e:
                errors.append(("subproc_main_load", str(e)))
                print("Failed to transfer weights from subprocess-loaded model:", e)
        else:
            errors.append(("subproc_probe", out))
    except subprocess.TimeoutExpired:
        errors.append(("subproc_timeout", "timeout"))
    except Exception as e:
        errors.append(("subproc_exc", str(e)))
    finally:
        try:
            os.unlink(probe_path)
        except Exception:
            pass

# 3) If still not loaded, try h5py extraction if file is HDF5
if not loaded:
    try:
        import h5py
        print("Trying h5py to inspect file...")
        with h5py.File(MODEL_WEIGHTS_FILE, "r") as f:
            print("Top groups:", list(f.keys())[:10])
            # attempt to load weights by name if groups exist
            # This is best-effort; not always possible
        errors.append(("h5py_inspect", "inspected"))
    except Exception as e:
        errors.append(("h5py", str(e)))
        print("h5py inspect failed:", e)

# Final failure if not loaded
if not loaded:
    print("ERROR: Could not load weights into architecture. Errors / probes:")
    for tag, msg in errors:
        print(" -", tag, ":", msg)
    print("\nIf you see segmentation faults earlier, try running conversion inside a Linux container or use tensorflow-macos==2.15 + tensorflow-metal==1.1.0")
    sys.exit(10)

# ---- cast weights to float32 ----
print("Casting weights to float32 (in-place) if needed...")
for w in model.weights:
    if w.dtype != tf.float32:
        try:
            w.assign(tf.cast(w, tf.float32))
        except Exception:
            tf.keras.backend.set_value(w, w.numpy().astype(np.float32))
print("Casting done.")

# ---- save SavedModel ----
print("Saving SavedModel to", OUT_SAVEDMODEL)
try:
    tf.saved_model.save(model, OUT_SAVEDMODEL)
    print("Saved SavedModel ->", OUT_SAVEDMODEL)
except Exception as e:
    print("ERROR saving SavedModel:", e)
    traceback.print_exc()
    sys.exit(20)

print("All done.")
