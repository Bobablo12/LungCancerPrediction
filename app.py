import os
import io
import json
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

# Custom Lambda layer class that always provides output_shape
class SafeLambda(keras.layers.Layer):
    """Lambda layer that always provides output_shape to avoid deserialization errors."""
    def __init__(self, function, output_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.function = function
        if output_shape is None:
            # Default to image shape for preprocessing
            self._output_shape = [IMG_SIZE, IMG_SIZE, 3]
        else:
            self._output_shape = output_shape
    
    def call(self, inputs):
        if callable(self.function):
            return self.function(inputs)
        return inputs
    
    def compute_output_shape(self, input_shape):
        if self._output_shape:
            batch_dim = input_shape[0] if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 else None
            return (batch_dim,) + tuple(self._output_shape)
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        if callable(self.function):
            if hasattr(self.function, '__name__'):
                config['function'] = self.function.__name__
            else:
                config['function'] = str(self.function)
        else:
            config['function'] = self.function
        config['output_shape'] = self._output_shape
        return config
    
    @classmethod
    def from_config(cls, config):
        function_name = config.get('function', 'preprocess_input')
        output_shape = config.get('output_shape', [IMG_SIZE, IMG_SIZE, 3])
        
        # Resolve function
        if function_name == 'preprocess_input' or 'preprocess' in str(function_name).lower():
            from tensorflow.keras.applications.efficientnet import preprocess_input
            function = preprocess_input
        else:
            function = lambda x: x
        
        return cls(function=function, output_shape=output_shape, **{k: v for k, v in config.items() 
                                                                     if k not in ['function', 'output_shape', 'name']})

# Monkey-patch Lambda layer to always provide output_shape
_original_lambda_from_config = keras.layers.Lambda.from_config.__func__
_original_lambda_compute_output_shape = keras.layers.Lambda.compute_output_shape

def _patched_lambda_from_config(cls, config):
    """Patched from_config that ensures output_shape is set."""
    # If output_shape is missing, infer it from input_shape or set default
    if 'output_shape' not in config or config.get('output_shape') is None:
        # Try to get input_shape from the config
        input_shape = config.get('batch_input_shape') or config.get('input_shape')
        if input_shape:
            # Remove batch dimension for output_shape
            if isinstance(input_shape, list) and len(input_shape) > 0:
                if input_shape[0] is None and len(input_shape) > 1:
                    config['output_shape'] = list(input_shape[1:])
                else:
                    config['output_shape'] = list(input_shape)
            else:
                config['output_shape'] = input_shape
        else:
            # Default for image preprocessing: [300, 300, 3]
            config['output_shape'] = [IMG_SIZE, IMG_SIZE, 3]
    
    # Call original from_config with cls
    return _original_lambda_from_config(cls, config)

def _patched_lambda_compute_output_shape(self, input_shape):
    """Patched compute_output_shape that always returns a shape."""
    try:
        # Try original method first
        return _original_lambda_compute_output_shape(self, input_shape)
    except (NotImplementedError, Exception):
        # If it fails, return input shape (common for preprocessing layers)
        return input_shape

# Apply monkey patches
keras.layers.Lambda.from_config = classmethod(_patched_lambda_from_config)
keras.layers.Lambda.compute_output_shape = _patched_lambda_compute_output_shape

# Define custom objects with common activation functions
# Import EfficientNet preprocessing function
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

custom_objects = {
    'Swish': Swish,
    'swish': Swish(),
    'tf': tf,
    'keras': keras,
    'relu': tf.keras.activations.relu,
    'sigmoid': tf.keras.activations.sigmoid,
    'softmax': tf.keras.activations.softmax,
    'input': keras.layers.Input,
    'Input': keras.layers.Input,
    'Lambda': SafeLambda,  # Use SafeLambda as fallback for Lambda layers
    'preprocess_input': effnet_preprocess,  # Add the preprocessing function
    'lambda': lambda x: x,  # Default lambda function
}

def patch_model_config_for_lambda(model_path: Path) -> Path:
    """Patch model config to add output_shape to Lambda layers, preserving .keras zip format."""
    import json
    import zipfile
    
    # Create a temporary patched model file
    patched_path = str(model_path) + ".patched"
    
    try:
        # Read the original zip and create a new one with patched config
        with zipfile.ZipFile(str(model_path), 'r') as zip_read:
            # Read the config
            config_data = zip_read.read('config.json')
            config = json.loads(config_data.decode('utf-8'))
            
            # Recursively patch Lambda layers to add output_shape
            def patch_lambda_layers(obj, input_shape=None):
                if isinstance(obj, dict):
                    class_name = obj.get('class_name', '')
                    
                    # Track input shapes as we traverse
                    if 'config' in obj:
                        config_dict = obj['config']
                        layer_input_shape = config_dict.get('batch_input_shape') or config_dict.get('input_shape') or input_shape
                        
                        if class_name == 'Lambda':
                            # Add output_shape for Lambda layers (same as input for preprocessing)
                            if 'output_shape' not in config_dict or config_dict.get('output_shape') is None:
                                if layer_input_shape:
                                    # Remove batch dimension for output_shape
                                    if isinstance(layer_input_shape, list) and len(layer_input_shape) > 0:
                                        # For shape like [None, 300, 300, 3], output_shape should be [300, 300, 3]
                                        if layer_input_shape[0] is None and len(layer_input_shape) > 1:
                                            output_shape = list(layer_input_shape[1:])
                                        else:
                                            output_shape = list(layer_input_shape)
                                    else:
                                        output_shape = layer_input_shape
                                    config_dict['output_shape'] = output_shape
                                else:
                                    # Default to IMG_SIZE x IMG_SIZE x 3 for image preprocessing
                                    config_dict['output_shape'] = [IMG_SIZE, IMG_SIZE, 3]
                        
                        # Recursively process nested structures
                        for key, value in config_dict.items():
                            if isinstance(value, (dict, list)):
                                patch_lambda_layers(value, layer_input_shape)
                    
                    # Also check for 'layers' key (for Sequential/Functional models)
                    if 'layers' in obj:
                        patch_lambda_layers(obj['layers'], input_shape)
                        
                elif isinstance(obj, list):
                    for item in obj:
                        patch_lambda_layers(item, input_shape)
            
            patch_lambda_layers(config)
            
            # Create new zip file with all original files, but patched config.json
            with zipfile.ZipFile(patched_path, 'w', zipfile.ZIP_DEFLATED) as zip_write:
                # Copy all files from original zip
                for item in zip_read.infolist():
                    if item.filename == 'config.json':
                        # Write patched config
                        zip_write.writestr(item.filename, json.dumps(config).encode('utf-8'))
                    else:
                        # Copy original file
                        zip_write.writestr(item, zip_read.read(item.filename))
        
        return Path(patched_path)
    except Exception as e:
        # Clean up on error
        if os.path.exists(patched_path):
            os.remove(patched_path)
        raise e

def load_model_safely(model_path: Path) -> keras.Model:
    """Load model with multiple fallback strategies."""
    # Enable unsafe deserialization at the beginning
    tf.keras.config.enable_unsafe_deserialization()
    
    # Clear any existing session
    tf.keras.backend.clear_session()
    
    # Strategy 1: Try loading with monkey-patched Lambda layer
    try:
        model = keras.models.load_model(
            str(model_path),
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False
        )
        logger.info("✅ Model loaded successfully with patched Lambda layer")
        return model
    except Exception as e:
        logger.warning(f"First attempt failed: {str(e)}")
    
    # Strategy 2: Try with compile=True
    try:
        tf.keras.backend.clear_session()
        model = keras.models.load_model(
            str(model_path),
            compile=True,
            custom_objects=custom_objects,
            safe_mode=False
        )
        logger.info("✅ Model loaded successfully with compile=True")
        return model
    except Exception as e:
        logger.warning(f"Second attempt failed: {str(e)}")
    
    # Strategy 3: Patch model config to add output_shape to Lambda layers, then load
    try:
        tf.keras.backend.clear_session()
        patched_path = patch_model_config_for_lambda(model_path)
        try:
            model = keras.models.load_model(
                str(patched_path),
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False
            )
            logger.info("✅ Model loaded successfully after patching model file")
            # Clean up patched file
            if os.path.exists(str(patched_path)):
                os.remove(str(patched_path))
            return model
        except Exception as load_error:
            # Clean up on error
            if os.path.exists(str(patched_path)):
                os.remove(str(patched_path))
            raise load_error
    except Exception as e:
        logger.warning(f"Third attempt (patching) failed: {str(e)}")
    
    # All strategies failed
    logger.error("❌ All loading strategies failed")
    raise RuntimeError("Failed to load model. Please check if the model file is corrupted or incompatible.")

def fix_lambda_layers(model: keras.Model) -> keras.Model:
    """Fix Lambda layers in the model by patching their call method to handle tf NameError."""
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
    
    # Recursively fix Lambda layers
    def fix_layer(layer):
        if isinstance(layer, keras.layers.Lambda):
            layer_name = getattr(layer, 'name', 'unknown')
            
            # Store original call method
            original_call = layer.call
            
            # Create a patched call method that catches NameError for tf
            def patched_call(inputs, mask=None, training=None):
                try:
                    return original_call(inputs, mask=mask, training=training)
                except NameError as name_err:
                    if 'tf' in str(name_err) or 'tensorflow' in str(name_err).lower():
                        # Use EfficientNet preprocessing directly
                        logger.info(f"Lambda layer {layer_name} had tf NameError, using EfficientNet preprocessing")
                        return effnet_preprocess(inputs)
                    raise
            
            # Replace the call method
            layer.call = patched_call
            logger.info(f"Patched Lambda layer: {layer_name}")
        
        # Recursively process nested layers
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                fix_layer(sublayer)
    
    # Fix all Lambda layers in the model
    for layer in model.layers:
        fix_layer(layer)
    
    return model

def load_labels() -> Dict[str, str]:
    """Load class labels from JSON file."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
    with open(LABELS_PATH, "r") as f:
        return {str(k): str(v) for k, v in json.load(f).items()}

# Load the model and labels
try:
    logger.info("Loading model...")
    model = load_model_safely(MODEL_PATH)
    logger.info("✅ Model loaded successfully!")
    
    # Fix Lambda layers that have tf reference issues
    try:
        model = fix_lambda_layers(model)
        logger.info("✅ Fixed Lambda layers")
    except Exception as e:
        logger.warning(f"Could not fix Lambda layers: {e}")
    
    # Test the model before wrapping
    test_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    try:
        _ = model.predict(test_input, verbose=0)
        logger.info("✅ Model test prediction successful!")
    except Exception as e:
        logger.warning(f"Model test failed, trying to wrap with preprocessing: {e}")
        # If test fails, try wrapping with preprocessing
        inputs = keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
        x = inputs
        if not isinstance(model.layers[0], keras.layers.InputLayer):
            x = keras.applications.efficientnet.preprocess_input(inputs)
        outputs = model(x)
        model = keras.Model(inputs, outputs)
        
        # Test again
        _ = model.predict(test_input, verbose=0)
        logger.info("✅ Model test prediction successful after wrapping!")

    # Load labels
    labels = load_labels()
    logger.info(f"✅ Loaded {len(labels)} class labels")

except Exception as e:
    logger.error(f"❌ Failed to initialize model: {str(e)}")
    logger.error("Please check if the model file is compatible with your TensorFlow version.")
    logger.error("Consider retraining the model or exporting it in a different format.")
    raise

def preprocess_image(image: Image.Image) -> tf.Tensor:
    """Preprocess image for model inference."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image, dtype=np.float32)
    return tf.expand_dims(image_array, axis=0)

def run_inference(image_tensor: tf.Tensor) -> Dict[str, Any]:
    """Run model inference on a single image."""
    preds = model.predict(image_tensor, verbose=0)
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
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        image = Image.open(io.BytesIO(contents))
        image_tensor = preprocess_image(image)
        result = run_inference(image_tensor)
        return result
        
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
