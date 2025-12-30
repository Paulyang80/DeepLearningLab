"""
Diagnostic script to check what the preprocessing actually produces.
Run from backend/cnn/ directory:
    cd backend/cnn
    python3 diagnose_preprocess.py
"""

import sys
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load MNIST
data_root = Path(__file__).parent.parent.parent / "notebooks" / "data"

# Original dataset
orig_dataset = datasets.MNIST(root=data_root, train=False, download=False)
img_orig, label = orig_dataset[0]

print(f"True label: {label}")
print(f"Original image type: {type(img_orig)}")
print(f"Original image mode: {img_orig.mode}")
print(f"Original image size: {img_orig.size}")

# Convert to array to check values
arr_orig = np.array(img_orig)
print(f"\nOriginal pixel value range: [{arr_orig.min()}, {arr_orig.max()}]")
print(f"Original mean: {arr_orig.mean():.2f}")
print(f"Sample pixels (top-left 5x5):")
print(arr_orig[:5, :5])

# Training transform
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

tensor_train = train_transform(img_orig)
print(f"\nAfter training transform:")
print(f"  Shape: {tensor_train.shape}")
print(f"  Value range: [{tensor_train.min():.3f}, {tensor_train.max():.3f}]")
print(f"  Mean: {tensor_train.mean():.3f}")

# Check what our preprocess does
print("\n" + "="*60)
print("Testing our preprocessing function:")
print("="*60)

# Simulate what happens in API
import base64
from io import BytesIO

buffer = BytesIO()
img_orig.save(buffer, format="PNG")
img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Decode it back
image_bytes = base64.b64decode(img_base64)
img = Image.open(BytesIO(image_bytes)).convert("L")
img = img.resize((28, 28), Image.Resampling.LANCZOS)

arr = np.array(img, dtype=np.float32)
print(f"\nAfter loading from base64:")
print(f"  Pixel range: [{arr.min()}, {arr.max()}]")
print(f"  Mean: {arr.mean():.2f}")

# Without inversion
arr_no_inv = arr / 255.0
arr_no_inv = (arr_no_inv - 0.1307) / 0.3081
print(f"\nWithout inversion (current fix):")
print(f"  Range: [{arr_no_inv.min():.3f}, {arr_no_inv.max():.3f}]")
print(f"  Mean: {arr_no_inv.mean():.3f}")

# With inversion
arr_inv = 255.0 - arr
arr_inv = arr_inv / 255.0
arr_inv = (arr_inv - 0.1307) / 0.3081
print(f"\nWith inversion:")
print(f"  Range: [{arr_inv.min():.3f}, {arr_inv.max():.3f}]")
print(f"  Mean: {arr_inv.mean():.3f}")

print(f"\nExpected (training transform):")
print(f"  Range: [{tensor_train.min():.3f}, {tensor_train.max():.3f}]")
print(f"  Mean: {tensor_train.mean():.3f}")

# Compare
diff_no_inv = torch.abs(torch.from_numpy(arr_no_inv) - tensor_train.squeeze()).mean()
diff_inv = torch.abs(torch.from_numpy(arr_inv) - tensor_train.squeeze()).mean()

print(f"\n" + "="*60)
print(f"Mean absolute difference:")
print(f"  Without inversion: {diff_no_inv:.6f}")
print(f"  With inversion: {diff_inv:.6f}")
print(f"\n✓ Better preprocessing: {'WITHOUT inversion' if diff_no_inv < diff_inv else 'WITH inversion'}")
print("="*60)
