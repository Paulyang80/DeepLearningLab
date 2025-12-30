"""
CNN Module for MNIST Digit Recognition

This package contains:
- model.py: SimpleCNN architecture definition
- predictor.py: Preprocessing and prediction logic
- trainer.py: Training loop and utilities
"""

from .model import SimpleCNN
from .predictor import CNNPredictor

__all__ = ["SimpleCNN", "CNNPredictor"]
