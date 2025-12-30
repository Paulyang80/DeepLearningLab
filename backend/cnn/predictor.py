"""
CNNPredictor: Handles image preprocessing and model inference for MNIST
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any

from .model import SimpleCNN


class CNNPredictor:
    """
    Predictor class for MNIST digit recognition.
    
    Handles:
    - Model loading and caching
    - Image preprocessing (base64 → tensor)
    - Inference with feature map extraction
    """
    
    # MNIST normalization constants (computed from training set)
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    
    def __init__(self, model_dir: Path):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory containing model weights (.pth files)
        """
        self.model_dir = model_dir
        self.best_model_path = model_dir / "mnist_cnn_best.pth"
        self.model_path = model_dir / "mnist_cnn.pth"
        self._model: Optional[SimpleCNN] = None
    
    def load_model(self) -> SimpleCNN:
        """
        Load CNN model from disk (with caching).
        
        Tries to load best model first, falls back to regular model.
        Uses untrained model if neither exists.
        
        Returns:
            Loaded SimpleCNN model in eval mode
        """
        if self._model is not None:
            return self._model
        
        model = SimpleCNN(dropout_rate=0.25)
        
        # Try to load best model first, fallback to regular model
        model_to_load = self.best_model_path if self.best_model_path.exists() else self.model_path
        
        if model_to_load.exists():
            try:
                state = torch.load(model_to_load, map_location="cpu")
                model.load_state_dict(state)
                print(f"✓ Loaded CNN model from {model_to_load}")
            except Exception as e:
                print(f"✗ Failed to load model: {e}. Using untrained model.")
        else:
            print(f"⚠ Model not found. Using untrained model. Run train_cnn.py first.")
        
        model.eval()
        self._model = model
        return model
    
    @classmethod
    def preprocess_image_base64(cls, image_str: str) -> torch.Tensor:
        """
        Convert base64/data-URL image to preprocessed tensor.
        
        Applies the same preprocessing as training:
        1. Decode base64 → PIL Image
        2. Convert to grayscale, resize to 28×28
        3. Normalize to [0, 1]
        4. Apply MNIST normalization (mean=0.1307, std=0.3081)
        
          Note: MNIST images are typically black background (0) with white digits (high values).
              Many drawing canvases produce white background with black ink; in that case, we
              invert automatically so inference matches MNIST polarity.
        
        Args:
            image_str: Base64-encoded image or data URL (e.g., "data:image/png;base64,...")
            
        Returns:
            Preprocessed tensor of shape (1, 1, 28, 28)
        """
        # Handle data URL format
        if image_str.startswith("data:"):
            header, data = image_str.split(",", 1)
            image_bytes = base64.b64decode(data)
        else:
            image_bytes = base64.b64decode(image_str)
        
        # Load image; if it has transparency, composite onto white first.
        img = Image.open(BytesIO(image_bytes))
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img).convert("L")
        else:
            img = img.convert("L")

        # Resize image
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        arr = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        arr = arr / 255.0

        # Heuristic: invert if the background is mostly white (common for canvas drawings).
        # MNIST is mostly black background with bright strokes.
        if float(np.mean(arr)) > 0.5:
            arr = 1.0 - arr
        
        # Apply MNIST normalization (same as training)
        arr = (arr - cls.MNIST_MEAN) / cls.MNIST_STD
        
        # Convert to tensor (1, 1, 28, 28)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return tensor
    
    def predict(
        self,
        image_str: str,
        return_feature_maps: bool = False
    ) -> Dict[str, Any]:
        """
        Predict digit from base64 image.
        
        Args:
            image_str: Base64-encoded image or data URL
            return_feature_maps: Whether to return conv layer feature maps
            
        Returns:
            Dictionary containing:
            - predicted_digit: int (0-9)
            - probabilities: List[float] (10 values summing to 1)
            - logits: List[float] (10 raw output values)
            - conv1_maps: Optional[List] (16 feature maps if requested)
            - conv2_maps: Optional[List] (32 feature maps if requested)
        """
        model = self.load_model()
        
        # Preprocess image
        try:
            x = self.preprocess_image_base64(image_str)
        except Exception as e:
            raise ValueError(f"Invalid image format: {e}")
        
        # Run inference
        with torch.inference_mode():
            logits = model(x)  # (1, 10)
        
        # Compute probabilities
        probs = F.softmax(logits, dim=1).squeeze(0)  # (10,)
        predicted = int(torch.argmax(probs).item())
        
        result = {
            "predicted_digit": predicted,
            "probabilities": [float(p) for p in probs.tolist()],
            "logits": [float(l) for l in logits.squeeze(0).tolist()],
        }
        
        # Optionally include feature maps
        if return_feature_maps:
            if model.conv1_output is not None:
                c1 = model.conv1_output.squeeze(0).cpu().numpy()  # (16, H, W)
                result["conv1_maps"] = [
                    [[float(v) for v in row] for row in channel]
                    for channel in c1
                ]
            
            if model.conv2_output is not None:
                c2 = model.conv2_output.squeeze(0).cpu().numpy()  # (32, H, W)
                result["conv2_maps"] = [
                    [[float(v) for v in row] for row in channel]
                    for channel in c2
                ]
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary with architecture info, paths, and status
        """
        return {
            "architecture": "SimpleCNN: Conv(16)->ReLU->Pool->Dropout->Conv(32)->ReLU->Pool->Dropout->FC(128)->Dropout->FC(10)",
            "input_shape": [1, 28, 28],
            "output_classes": 10,
            "model_path": str(self.model_path),
            "best_model_path": str(self.best_model_path),
            "loaded": self._model is not None,
            "exists": self.model_path.exists() or self.best_model_path.exists(),
            "parameters": self._model.get_num_parameters() if self._model else None,
        }
