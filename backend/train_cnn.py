"""
Train a simple CNN on MNIST and save the model for the web visualizer.
Run this script once to create backend/models/mnist_cnn.pth

This is a convenience wrapper around the cnn.trainer module.
"""

from backend.cnn.trainer import train_mnist_cnn


if __name__ == "__main__":
    # Train with Mac GPU acceleration
    # 10 epochs takes ~2-3 minutes on Mac GPU
    train_mnist_cnn(epochs=10, batch_size=64, lr=0.001)
