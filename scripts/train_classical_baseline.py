"""
Training script for classical CNN baseline.

This script demonstrates best practices for training a classical CNN
on the EuroSAT dataset with proper reproducibility settings.

Usage:
    python scripts/train_classical_baseline.py
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from dataio.loader_refactored import EuroSATDataModule
from models.classical_cnn import ClassicalCNN
from utils.reproducibility import set_seed, get_device
from utils.training import create_trainer


def main():
    """Main training function."""
    # ========================================
    # Configuration
    # ========================================
    SEED = 42
    EXPERIMENT_NAME = "classical_lenet5_baseline"

    # Data configuration
    TRAIN_DIR = "dataset/training"
    VAL_DIR = "dataset/validation"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    REDUCE_BY_FACTOR = 1.0  # Use full dataset
    IMAGE_SIZE = 64

    # Model configuration
    MODEL_TYPE = "lenet5"  # Options: 'lenet5', 'improved'
    INPUT_SHAPE = (3, 64, 64)
    NUM_CLASSES = 2  # Will be set based on selected classes
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.5

    # Training configuration
    MAX_EPOCHS = 30
    ACCELERATOR = "auto"  # 'auto', 'gpu', 'cpu'
    DEVICES = 1
    PRECISION = "32"  # '32', '16-mixed', 'bf16-mixed'

    # Class selection (None = all classes, or specify list)
    # For binary classification, use two semantically different classes
    SELECTED_CLASSES = ["AnnualCrop", "Forest"]  # Example: meaningful binary task
    # SELECTED_CLASSES = None  # Use all 10 classes

    # ========================================
    # Setup
    # ========================================
    print("\n" + "="*60)
    print("Classical CNN Baseline Training")
    print("="*60 + "\n")

    # Set random seed for reproducibility
    set_seed(SEED)

    # Get device info
    device = get_device(prefer_gpu=True)

    # ========================================
    # Data Loading
    # ========================================
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60 + "\n")

    data_module = EuroSATDataModule(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        reduce_by_factor=REDUCE_BY_FACTOR,
        selected_classes=SELECTED_CLASSES,
        image_size=IMAGE_SIZE,
        seed=SEED
    )

    # Setup to get number of classes
    data_module.setup()
    NUM_CLASSES = data_module.get_num_classes()

    # ========================================
    # Model Creation
    # ========================================
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60 + "\n")

    model = ClassicalCNN(
        model_type=MODEL_TYPE,
        in_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE
    )

    print(f"Model: {MODEL_TYPE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ========================================
    # Training
    # ========================================
    print("\n" + "="*60)
    print("Training")
    print("="*60 + "\n")

    trainer = create_trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        log_dir="lightning_logs",
        model_dir="saved_models",
        experiment_name=EXPERIMENT_NAME,
        early_stopping=True,
        early_stopping_patience=10,
        precision=PRECISION
    )

    # Train the model
    trainer.fit(model, data_module)

    # ========================================
    # Results
    # ========================================
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60 + "\n")

    # Get best metrics
    if trainer.checkpoint_callback:
        print(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")

    # Test on validation set
    print("\nRunning final validation...")
    results = trainer.validate(model, data_module)
    print(f"\nFinal validation metrics:")
    for key, value in results[0].items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "="*60)
    print("To view training logs, run:")
    print(f"  tensorboard --logdir=lightning_logs/{EXPERIMENT_NAME}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
