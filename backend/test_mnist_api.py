"""
Quick test script to verify MNIST prediction works correctly.
Creates a simple test image and sends it to the API.
"""

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import requests
import json

def test_with_real_mnist():
    """Test with a real MNIST image to verify preprocessing."""
    
    # Load a real MNIST test image
    data_root = Path(__file__).parent.parent / "notebooks" / "data"
    test_dataset = datasets.MNIST(root=data_root, train=False, download=False)
    
    # Get first image (should be a 7)
    img, label = test_dataset[0]
    print(f"Testing with MNIST test set image 0, true label: {label}")
    
    # Convert PIL image to base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    data_url = f"data:image/png;base64,{img_base64}"
    
    # Send to API
    response = requests.post(
        "http://localhost:8000/predict/mnist",
        json={"image": data_url, "return_feature_maps": False}
    )
    
    if response.status_code == 200:
        result = response.json()
        predicted = result["predicted_digit"]
        probs = result["probabilities"]
        
        print(f"\n✓ Prediction successful!")
        print(f"  Predicted digit: {predicted}")
        print(f"  True label: {label}")
        print(f"  Match: {'✓' if predicted == label else '✗'}")
        print(f"\n  Top 3 probabilities:")
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        for digit, prob in sorted_probs[:3]:
            print(f"    {digit}: {prob*100:.2f}%")
        
        return predicted == label
    else:
        print(f"✗ API error: {response.status_code}")
        print(response.text)
        return False

def test_multiple_images(n=10):
    """Test with multiple MNIST images."""
    data_root = Path(__file__).parent.parent / "notebooks" / "data"
    test_dataset = datasets.MNIST(root=data_root, train=False, download=False)
    
    correct = 0
    print(f"\nTesting {n} images from MNIST test set...\n")
    
    for i in range(n):
        img, label = test_dataset[i]
        
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        data_url = f"data:image/png;base64,{img_base64}"
        
        response = requests.post(
            "http://localhost:8000/predict/mnist",
            json={"image": data_url, "return_feature_maps": False}
        )
        
        if response.status_code == 200:
            result = response.json()
            predicted = result["predicted_digit"]
            match = predicted == label
            if match:
                correct += 1
            
            symbol = "✓" if match else "✗"
            print(f"  Image {i}: True={label}, Pred={predicted} {symbol}")
        else:
            print(f"  Image {i}: API Error")
    
    accuracy = correct / n * 100
    print(f"\n{'='*50}")
    print(f"Accuracy: {correct}/{n} = {accuracy:.1f}%")
    print(f"{'='*50}")
    
    return accuracy


if __name__ == "__main__":
    print("="*60)
    print("MNIST Prediction Test")
    print("="*60)
    
    # Test single image
    success = test_with_real_mnist()
    
    if success:
        # Test multiple images
        test_multiple_images(20)
    else:
        print("\n⚠ Single image test failed. Fix issues before batch testing.")
