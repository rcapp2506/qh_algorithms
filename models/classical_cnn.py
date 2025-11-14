"""
Classical CNN baseline models for EuroSAT classification.

This module provides pure classical CNN architectures for comparison
with hybrid quantum-classical models.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple


def size_conv_layer(s: int, kernel_size: int, padding: int, stride: int) -> int:
    """Calculate output size after convolution layer."""
    size = int(((s - kernel_size + 2 * padding) / stride) + 1)
    return size


class ClassicalLeNet5(nn.Module):
    """
    Classical LeNet-5 architecture adapted for EuroSAT.

    This serves as a baseline to compare against hybrid quantum models.

    Args:
        in_shape: Input shape as (channels, width, height)
        num_classes: Number of output classes
    """

    def __init__(self, in_shape: Tuple[int, int, int], num_classes: int):
        super().__init__()

        if len(in_shape) != 3:
            raise ValueError(
                f"in_shape must be (channels, width, height), got {in_shape}"
            )
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        c, w, h = in_shape

        # Convolutional Layer 1
        c1 = 6
        self.conv_1 = nn.Conv2d(
            in_channels=c, out_channels=c1, kernel_size=5, padding=2, stride=1
        )
        w1 = size_conv_layer(w, kernel_size=5, padding=2, stride=1)
        h1 = size_conv_layer(h, kernel_size=5, padding=2, stride=1)

        # Max Pooling 1
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        w2 = size_conv_layer(w1, kernel_size=2, padding=0, stride=2)
        h2 = size_conv_layer(h1, kernel_size=2, padding=0, stride=2)

        # Convolutional Layer 2
        c2 = 16
        self.conv_2 = nn.Conv2d(
            in_channels=c1, out_channels=c2, kernel_size=5, stride=1
        )
        w3 = size_conv_layer(w2, kernel_size=5, padding=0, stride=1)
        h3 = size_conv_layer(h2, kernel_size=5, padding=0, stride=1)

        # Max Pooling 2
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        w4 = size_conv_layer(w3, kernel_size=2, padding=0, stride=2)
        h4 = size_conv_layer(h3, kernel_size=2, padding=0, stride=2)

        # Calculate flatten size
        self.flatten_size = c2 * w4 * h4

        # Fully Connected Layers
        fc_1_size = 120
        fc_2_size = 84

        self.fc_1 = nn.Linear(self.flatten_size, fc_1_size)
        self.fc_2 = nn.Linear(fc_1_size, fc_2_size)
        self.fc_3 = nn.Linear(fc_2_size, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Conv + Pool blocks
        x = self.max_pool1(self.relu(self.conv_1(x)))
        x = self.max_pool2(self.relu(self.conv_2(x)))

        # Flatten
        x = x.view(-1, self.flatten_size)

        # Fully connected layers
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.fc_3(x)

        return x


class ImprovedCNN(nn.Module):
    """
    Improved CNN with batch normalization and dropout.

    This is a more modern baseline with regularization techniques.

    Args:
        in_shape: Input shape as (channels, width, height)
        num_classes: Number of output classes
        dropout_rate: Dropout probability (default: 0.5)
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        num_classes: int,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        if len(in_shape) != 3:
            raise ValueError(
                f"in_shape must be (channels, width, height), got {in_shape}"
            )

        c, w, h = in_shape

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *in_shape)
            dummy_output = self.features(dummy_input)
            self.flatten_size = dummy_output.view(1, -1).size(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ClassicalCNN(pl.LightningModule):
    """
    PyTorch Lightning wrapper for classical CNN models.

    Args:
        model_type: Type of model ('lenet5' or 'improved')
        in_shape: Input shape as (channels, width, height)
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
        dropout_rate: Dropout rate (only for 'improved' model)
    """

    def __init__(
        self,
        model_type: str = 'lenet5',
        in_shape: Tuple[int, int, int] = (3, 64, 64),
        num_classes: int = 2,
        learning_rate: float = 0.0001,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()

        # Select model architecture
        if model_type == 'lenet5':
            self.network = ClassicalLeNet5(in_shape, num_classes)
        elif model_type == 'improved':
            self.network = ImprovedCNN(in_shape, num_classes, dropout_rate)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        inputs, labels = batch
        outputs = self(inputs)

        loss = self.loss(outputs, labels)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()

        # Logging
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        inputs, labels = batch
        outputs = self(inputs)

        loss = self.loss(outputs, labels)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()

        # Logging
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
