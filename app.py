import os
import io
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = 300
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "lung_cancer_model_best2.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

# Disable GPU to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initialize FastAPI app
app = FastAPI(title="Lung Cancer Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom layer definitions
class Swish(keras.layers.Layer):
    def call(self, inputs):
        return inputs * tf.sigmoid(inputs)

# Define custom objects
custom_objects = {
    'Swish': Swish,
    'swish': Swish(),
    'tf': tf,
}

def load_model_with_retry(model_path: Path, max_retries: int = 3) -> keras.Model:
    """Attempt to load model with retries and different strategies."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1} to load model...")
            
            # Try different loading strategies
            if attempt == 0:
                # First try with safe_mode=False
                model = keras.models.load_model(
                    str(model_path),
                    compile=False,
                    custom_objects=custom_objects,
                    safe_mode=False
                )
            elif attempt == 1:
                # Try with a fresh session
                tf.keras.backend.clear_session()
                model = keras.models.load_model(
                    str(model_path),
                    compile=False,
                    custom_objects=custom_objects
                )
            else:
                # Final attempt with tensorflow.compat.v1
                import tensorflow.compat.v1 as tf1
                tf1.disable_v2_behavior()
                model = tf1.keras.models.load_model(
                    str(model_path),
                    compile=False,
                    custom_objects=custom_objects
                )
            
            # Test the model
            test_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            _ = model.predict(test_input, verbose=0)
            return model
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise

def load_labels() -> Dict[str, str]:
    """Load class labels from JSON file."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
    with open(LABELS_PATH, "r") as f:
        return {str(k): str(v) for k, v in json.load(f).items()}

# Load the model and labels
try:
    logger.info("Loading model...")
    model = load_model_with_retry(MODEL_PATH)
    logger.info("✅ Model loaded successfully!")
    
    # Wrap with preprocessing if needed
    inputs = keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x = inputs
    if not isinstance(model.layers[0], keras.layers.InputLayer):
        x = keras.applications.efficientnet.preprocess_input(inputs)
    outputs = model(x)
    model = keras.Model(inputs, outputs)
    
    # Test the final model
    test_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    _ = model.predict(test_input, verbose=0)
    logger.info("✅ Model test prediction successful!")

    # Load labels
    labels = load_labels()
    logger.info(f"✅ Loaded {len(labels)} class labels")

except Exception as e:
    logger.error(f"❌ Failed to initialize model: {str(e)}")
    logger.error("Please check the model file and try again")
    raise

def preprocess_image(image: Image.Image) -> tf.Tensor:
    """Preprocess image for model inference."""
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize and convert to array
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image, dtype=np.float32)
    
    # Add batch dimension
    return tf.expand_dims(image_array, axis=0)

def run_inference(image_tensor: tf.Tensor) -> Dict[str, Any]:
    """Run model inference on a single image."""
    # Get predictions
    preds = model.predict(image_tensor, verbose=0)
    probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
    
    # Get top prediction
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

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Lung Cancer Detection API is running",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of an uploaded image."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Read and validate image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess and run inference
        image_tensor = preprocess_image(image)
        result = run_inference(image_tensor)
        
        return result
        
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
