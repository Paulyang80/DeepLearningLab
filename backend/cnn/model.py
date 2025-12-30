"""
SimpleCNN: Lightweight CNN architecture for MNIST digit recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for MNIST handwritten digit recognition.
    
    Architecture:
        Input (1, 28, 28)
        ↓
        Conv1: 1→16 (3×3, pad=1) + ReLU + MaxPool(2×2) + Dropout(0.25)
        ↓ (16, 14, 14)
        Conv2: 16→32 (3×3, pad=1) + ReLU + MaxPool(2×2) + Dropout(0.25)
        ↓ (32, 7, 7)
        Flatten → FC1: 1568→128 + ReLU + Dropout(0.25)
        ↓
        FC2: 128→10 (logits)
        ↓
        Softmax → Probability distribution
    
    Features:
        - Dropout regularization (default 0.25) to prevent overfitting
        - Feature map capturing for visualization
        - ~207k parameters
        - Expected test accuracy: 99%+
    """

    def __init__(self, dropout_rate: float = 0.25):
        """
        Initialize SimpleCNN.
        
        Args:
            dropout_rate: Dropout probability (default 0.25)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Store intermediate feature maps for visualization
        self.conv1_output: Optional[torch.Tensor] = None
        self.conv2_output: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Logits tensor of shape (batch_size, 10)
        """
        # x: (B, 1, 28, 28)
        x = self.conv1(x)  # (B, 16, 28, 28)
        self.conv1_output = x.detach()
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # (B, 16, 14, 14)
        x = self.dropout1(x)

        x = self.conv2(x)  # (B, 32, 14, 14)
        self.conv2_output = x.detach()
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # (B, 32, 7, 7)
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)  # (B, 32*7*7)
        x = F.relu(self.fc1(x))  # (B, 128)
        x = self.dropout2(x)
        x = self.fc2(x)  # (B, 10) logits
        return x
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
