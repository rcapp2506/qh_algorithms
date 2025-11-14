"""
Training utilities and helper functions.
"""
import os
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor


def create_trainer(
    max_epochs: int = 30,
    accelerator: str = "auto",
    devices: int = 1,
    log_dir: str = "lightning_logs",
    model_dir: str = "saved_models",
    experiment_name: str = "experiment",
    early_stopping: bool = True,
    early_stopping_patience: int = 10,
    precision: str = "32",
    **kwargs
) -> pl.Trainer:
    """
    Create a PyTorch Lightning Trainer with sensible defaults.

    Args:
        max_epochs: Maximum number of training epochs
        accelerator: Accelerator type ('auto', 'gpu', 'cpu', 'mps')
        devices: Number of devices to use
        log_dir: Directory for logs
        model_dir: Directory for saved models
        experiment_name: Name of the experiment
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        precision: Training precision ('32', '16-mixed', 'bf16-mixed')
        **kwargs: Additional arguments passed to Trainer

    Returns:
        pl.Trainer: Configured trainer instance
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Setup callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(model_dir, experiment_name),
        filename='{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Setup logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=tb_logger,
        precision=precision,
        deterministic=True,  # For reproducibility
        enable_progress_bar=True,
        log_every_n_steps=10,
        **kwargs
    )

    print(f"\n{'='*60}")
    print(f"Trainer Configuration:")
    print(f"{'='*60}")
    print(f"Max epochs: {max_epochs}")
    print(f"Accelerator: {accelerator}")
    print(f"Devices: {devices}")
    print(f"Precision: {precision}")
    print(f"Log directory: {log_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Experiment name: {experiment_name}")
    print(f"Early stopping: {early_stopping} (patience={early_stopping_patience})")
    print(f"{'='*60}\n")

    return trainer


def size_conv_layer(s: int, kernel_size: int, padding: int, stride: int) -> int:
    """
    Calculate output size after a convolution layer.

    Args:
        s: Input size
        kernel_size: Kernel size
        padding: Padding
        stride: Stride

    Returns:
        int: Output size
    """
    return int(((s - kernel_size + 2 * padding) / stride) + 1)
