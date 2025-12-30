#!/usr/bin/env python3
"""
Diagnose why web predictions might be returning 0.
Tests the complete prediction pipeline.
"""

import sys
from pathlib import Path
import requests
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_test_digit(digit: int) -> str:
    """Create a simple hand-drawn digit image and return base64."""
    img = Image.new('L', (280, 280), color=255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Draw a simple representation of the digit
    if digit == 0:
        draw.ellipse([80, 80, 200, 200], outline=0, width=20)
    elif digit == 1:
        draw.line([140, 80, 140, 200], fill=0, width=20)
    elif digit == 7:
        draw.line([80, 90, 200, 90], fill=0, width=20)
        draw.line([200, 90, 140, 200], fill=0, width=20)
    elif digit == 8:
        draw.ellipse([100, 80, 180, 140], outline=0, width=20)
        draw.ellipse([100, 140, 180, 200], outline=0, width=20)
    
    # Convert to base64 data URL
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    data_url = f"data:image/png;base64,{img_base64}"
    
    return data_url

def test_api_prediction():
    """Test API prediction with various digits."""
    print("="*60)
    print("Testing API Predictions")
    print("="*60)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/model/cnn/info", timeout=2)
        if response.status_code != 200:
            print("✗ Backend not responding correctly")
            return False
        
        info = response.json()
        print(f"✓ Backend is running")
        print(f"  Model loaded: {info['loaded']}")
        print(f"  Model exists: {info['exists']}")
        print(f"  Parameters: {info.get('parameters', 'N/A')}")
        
        if not info['exists']:
            print("\n⚠ WARNING: Model file doesn't exist!")
            print("  Run: python3 backend/train_cnn.py")
            return False
        
    except requests.exceptions.ConnectionError:
        print("✗ Backend is not running!")
        print("  Start it with: uvicorn backend.main:app --reload --port 8000")
        return False
    except requests.exceptions.Timeout:
        print("✗ Backend timeout!")
        return False
    
    print()
    
    # Test predictions for different digits
    test_digits = [0, 1, 7, 8]
    results = []
    
    for digit in test_digits:
        print(f"Testing digit {digit}...")
        data_url = create_test_digit(digit)
        
        try:
            response = requests.post(
                "http://localhost:8000/predict/mnist",
                json={"image": data_url, "return_feature_maps": False},
                timeout=5
            )
            
            if response.status_code != 200:
                print(f"  ✗ API error: {response.status_code}")
                print(f"    {response.text}")
                results.append(False)
                continue
            
            result = response.json()
            predicted = result['predicted_digit']
            probs = result['probabilities']
            confidence = max(probs) * 100
            
            match = predicted == digit
            symbol = "✓" if match else "✗"
            
            print(f"  {symbol} Predicted: {predicted} (confidence: {confidence:.1f}%)")
            
            if not match:
                # Show top 3 predictions
                sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
                print(f"     Top 3: ", end="")
                for d, p in sorted_probs[:3]:
                    print(f"{d}({p*100:.1f}%) ", end="")
                print()
            
            results.append(match)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append(False)
    
    print()
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} predictions correct")
    
    if passed == 0:
        print("\n⚠ ALL PREDICTIONS FAILED!")
        print("  This matches the user's issue.")
        print("\n  Possible causes:")
        print("  1. Model wasn't trained correctly")
        print("  2. Model file is corrupted")
        print("  3. Preprocessing mismatch")
        print("\n  Solutions:")
        print("  1. Retrain the model: python3 backend/train_cnn.py")
        print("  2. Check backend logs for errors")
        print("  3. Verify preprocessing in predictor.py")
    elif passed < total:
        print(f"\n⚠ Some predictions failed ({passed}/{total})")
        print("  This is normal for simple hand-drawn shapes.")
    else:
        print("\n✓ All predictions correct!")
    
    print("="*60)
    
    return passed > 0

def test_direct_module():
    """Test the CNN module directly without API."""
    print("\n" + "="*60)
    print("Testing Direct Module")
    print("="*60)
    
    try:
        from backend.cnn import CNNPredictor
        from pathlib import Path
        
        model_dir = Path("backend/models")
        predictor = CNNPredictor(model_dir)
        
        print("✓ Module imported successfully")
        
        # Test with a simple image
        data_url = create_test_digit(7)
        result = predictor.predict(data_url)
        
        print(f"✓ Direct prediction successful")
        print(f"  Predicted: {result['predicted_digit']}")
        print(f"  Max probability: {max(result['probabilities'])*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Direct module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CNN Prediction Diagnosis")
    print("="*60)
    print()
    
    # Test direct module first
    module_ok = test_direct_module()
    
    # Test API
    api_ok = test_api_prediction()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Direct Module: {'✓ OK' if module_ok else '✗ FAILED'}")
    print(f"API: {'✓ OK' if api_ok else '✗ FAILED'}")
    print("="*60)
    
    if not module_ok and not api_ok:
        print("\n⚠ Both tests failed. Model needs retraining.")
        print("  Run: python3 backend/train_cnn.py")
    elif module_ok and not api_ok:
        print("\n⚠ Module OK but API failed. Check backend logs.")
    elif not module_ok and api_ok:
        print("\n⚠ Unexpected: API works but module failed.")
    else:
        print("\n✓ Everything working correctly!")
