#!/usr/bin/env python3
"""
Quick test to verify the refactored CNN module works correctly.
Run from backend/cnn/ directory:
    cd backend/cnn
    python3 test_cnn_module.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_model_import():
    """Test that model can be imported."""
    print("Testing model import...")
    from backend.cnn import SimpleCNN
    model = SimpleCNN(dropout_rate=0.25)
    print(f"✓ SimpleCNN imported successfully")
    print(f"  Parameters: {model.get_num_parameters():,}")
    return True

def test_predictor_init():
    """Test that predictor can be initialized."""
    print("\nTesting predictor initialization...")
    from backend.cnn import CNNPredictor
    from pathlib import Path
    
    model_dir = Path(__file__).parent.parent / "models"
    predictor = CNNPredictor(model_dir)
    print(f"✓ CNNPredictor initialized")
    return True

def test_predictor_load():
    """Test that model can be loaded."""
    print("\nTesting model loading...")
    from backend.cnn import CNNPredictor
    from pathlib import Path
    
    model_dir = Path(__file__).parent.parent / "models"
    predictor = CNNPredictor(model_dir)
    model = predictor.load_model()
    print(f"✓ Model loaded successfully")
    return True

def test_model_info():
    """Test that model info can be retrieved."""
    print("\nTesting model info...")
    from backend.cnn import CNNPredictor
    from pathlib import Path
    
    model_dir = Path(__file__).parent.parent / "models"
    predictor = CNNPredictor(model_dir)
    info = predictor.get_model_info()
    print(f"✓ Model info retrieved")
    print(f"  Architecture: {info['architecture']}")
    print(f"  Loaded: {info['loaded']}")
    print(f"  Exists: {info['exists']}")
    return True

def test_preprocessing():
    """Test image preprocessing."""
    print("\nTesting preprocessing...")
    from backend.cnn import CNNPredictor
    import torch
    import base64
    from io import BytesIO
    from PIL import Image
    import numpy as np
    
    # Create a simple test image (white background + black center)
    img = Image.new('L', (28, 28), color=255)
    pixels = img.load()
    for i in range(10, 18):
        for j in range(10, 18):
            pixels[i, j] = 0
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    data_url = f"data:image/png;base64,{img_base64}"
    
    # Preprocess
    tensor = CNNPredictor.preprocess_image_base64(data_url)
    print(f"✓ Preprocessing successful")
    print(f"  Tensor shape: {tensor.shape}")
    print(f"  Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("CNN Module Tests")
    print("="*60)
    
    tests = [
        test_model_import,
        test_predictor_init,
        test_predictor_load,
        test_model_info,
        test_preprocessing,
    ]
    
    results = []
    for test in tests:
        try:
            success = test()
            results.append(success)
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n✓ All tests passed! Module refactoring successful.")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
