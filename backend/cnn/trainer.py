"""
CNN Trainer: Training loop and utilities for MNIST CNN
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Optional, Tuple

from .model import SimpleCNN


class CNNTrainer:
    """
    Trainer for SimpleCNN on MNIST dataset.
    
    Features:
    - Automatic device selection (MPS/CUDA/CPU)
    - Data augmentation (rotation, translation)
    - Dropout regularization
    - Weight decay (L2 regularization)
    - Learning rate scheduling
    - Best model checkpointing
    """
    
    def __init__(
        self,
        data_root: Path,
        model_dir: Path,
        batch_size: int = 64,
        lr: float = 0.001,
        dropout_rate: float = 0.25,
        weight_decay: float = 1e-4,
    ):
        """
        Initialize trainer.
        
        Args:
            data_root: Root directory containing MNIST data
            model_dir: Directory to save trained models
            batch_size: Training batch size (default 64)
            lr: Initial learning rate (default 0.001)
            dropout_rate: Dropout probability (default 0.25)
            weight_decay: L2 regularization strength (default 1e-4)
        """
        self.data_root = data_root
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
        # Device selection: MPS (Mac GPU) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        if self.device.type == "mps":
            print("✓ Mac GPU (Metal) enabled! Training will be much faster.")
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and test dataloaders with augmentation.
        
        Returns:
            (train_loader, test_loader)
        """
        # Training transform with data augmentation
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # Random rotation ±10°
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation ±10%
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
        
        # Test transform (no augmentation)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root=self.data_root,
            train=True,
            download=False,
            transform=train_transform
        )
        test_dataset = datasets.MNIST(
            root=self.data_root,
            train=False,
            download=False,
            transform=test_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def train(self, epochs: int = 10, verbose: bool = True) -> Tuple[SimpleCNN, float]:
        """
        Train the CNN model.
        
        Args:
            epochs: Number of training epochs (default 10)
            verbose: Whether to print progress (default True)
            
        Returns:
            (trained_model, best_test_accuracy)
        """
        train_loader, test_loader = self.get_dataloaders()
        
        # Initialize model
        model = SimpleCNN(dropout_rate=self.dropout_rate).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2
        )
        
        if verbose:
            print(f"\nTraining for {epochs} epochs...")
            print(f"Model parameters: {model.get_num_parameters():,}")
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if verbose and (batch_idx + 1) % 200 == 0:
                    print(
                        f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                        f"Loss: {train_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%"
                    )
            
            # Evaluation phase
            test_acc = self.evaluate(model, test_loader)
            train_acc = 100.0 * correct / total
            
            # Update learning rate
            scheduler.step(test_acc)
            
            if verbose:
                print(f"✓ Epoch {epoch+1}/{epochs} complete")
                print(f"  Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
                print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_model_path = self.model_dir / "mnist_cnn_best.pth"
                torch.save(model.state_dict(), best_model_path)
                if verbose:
                    print(f"  ★ New best! Saved to mnist_cnn_best.pth")
            
            if verbose:
                print()
        
        # Save final model
        final_model_path = self.model_dir / "mnist_cnn.pth"
        torch.save(model.state_dict(), final_model_path)
        
        if verbose:
            print(f"\n" + "="*60)
            print(f"✓ Final model saved to {final_model_path}")
            print(f"  Final test accuracy: {test_acc:.2f}%")
            print(f"  Best test accuracy: {best_acc:.2f}%")
            print(f"  Best model: mnist_cnn_best.pth")
            print("="*60)
        
        return model, best_acc
    
    def evaluate(self, model: SimpleCNN, test_loader: DataLoader) -> float:
        """
        Evaluate model on test set.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            
        Returns:
            Test accuracy (%)
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100.0 * correct / total


def train_mnist_cnn(
    data_root: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
) -> None:
    """
    Convenience function to train MNIST CNN with default settings.
    
    Args:
        data_root: Root directory containing MNIST data (default: ../notebooks/data)
        model_dir: Directory to save models (default: ./models)
        epochs: Number of training epochs (default 10)
        batch_size: Training batch size (default 64)
        lr: Initial learning rate (default 0.001)
    """
    if data_root is None:
        data_root = Path(__file__).parent.parent.parent / "notebooks" / "data"
    
    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "models"
    
    trainer = CNNTrainer(
        data_root=data_root,
        model_dir=model_dir,
        batch_size=batch_size,
        lr=lr,
    )
    
    trainer.train(epochs=epochs, verbose=True)
