"""
Train a simple CNN on MNIST and save the model for the web visualizer.
Run this script from backend/cnn/ directory:
    cd backend/cnn
    python3 train_cnn.py

This is a convenience wrapper around the trainer module.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.cnn.trainer import train_mnist_cnn


if __name__ == "__main__":
    # Train with Mac GPU acceleration
    # 10 epochs takes ~2-3 minutes on Mac GPU
    train_mnist_cnn(epochs=10, batch_size=64, lr=0.001)
