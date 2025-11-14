"""
Script to compare classical and hybrid quantum models.

This script trains both classical and hybrid models with identical settings
and compares their performance.

Usage:
    python scripts/compare_models.py
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
from dataio.loader_refactored import EuroSATDataModule
from models.classical_cnn import ClassicalCNN
from utils.reproducibility import set_seed
from utils.training import create_trainer


def train_model(model, data_module, experiment_name, max_epochs=30):
    """Train a model and return results."""
    trainer = create_trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        log_dir="lightning_logs_comparison",
        model_dir="saved_models_comparison",
        experiment_name=experiment_name,
        early_stopping=True,
        early_stopping_patience=10,
        precision="32"
    )

    trainer.fit(model, data_module)

    # Get validation results
    results = trainer.validate(model, data_module)[0]

    return {
        'experiment': experiment_name,
        'val_loss': results['val_loss'],
        'val_accuracy': results['val_accuracy'],
        'best_model_path': trainer.checkpoint_callback.best_model_path,
        'parameters': sum(p.numel() for p in model.parameters())
    }


def main():
    """Main comparison function."""
    print("\n" + "="*60)
    print("Model Comparison: Classical vs Hybrid Quantum")
    print("="*60 + "\n")

    # Configuration
    SEED = 42
    SELECTED_CLASSES = ["AnnualCrop", "Forest"]
    MAX_EPOCHS = 30
    BATCH_SIZE = 32

    # Set seed
    set_seed(SEED)

    # Create data module (shared across all models)
    data_module = EuroSATDataModule(
        train_dir="dataset/training",
        val_dir="dataset/validation",
        batch_size=BATCH_SIZE,
        num_workers=4,
        reduce_by_factor=1.0,
        selected_classes=SELECTED_CLASSES,
        image_size=64,
        seed=SEED
    )
    data_module.setup()
    num_classes = data_module.get_num_classes()

    results = []

    # ========================================
    # Train Classical LeNet5
    # ========================================
    print("\n" + "="*60)
    print("Training Classical LeNet5")
    print("="*60 + "\n")

    model_classical_lenet = ClassicalCNN(
        model_type='lenet5',
        in_shape=(3, 64, 64),
        num_classes=num_classes,
        learning_rate=0.001
    )

    result = train_model(
        model_classical_lenet,
        data_module,
        "classical_lenet5",
        MAX_EPOCHS
    )
    results.append(result)

    # ========================================
    # Train Improved CNN
    # ========================================
    print("\n" + "="*60)
    print("Training Improved CNN")
    print("="*60 + "\n")

    model_improved = ClassicalCNN(
        model_type='improved',
        in_shape=(3, 64, 64),
        num_classes=num_classes,
        learning_rate=0.001,
        dropout_rate=0.5
    )

    result = train_model(
        model_improved,
        data_module,
        "improved_cnn",
        MAX_EPOCHS
    )
    results.append(result)

    # ========================================
    # Train Hybrid Quantum Model (if available)
    # ========================================
    # Note: Uncomment when hybrid models are set up
    # try:
    #     from models.hybrid_cnn import HybridCNN
    #     print("\n" + "="*60)
    #     print("Training Hybrid Quantum CNN")
    #     print("="*60 + "\n")
    #
    #     model_hybrid = HybridCNN(
    #         in_shape=(3, 64, 64),
    #         num_classes=num_classes,
    #         learning_rate=0.001
    #     )
    #
    #     result = train_model(
    #         model_hybrid,
    #         data_module,
    #         "hybrid_quantum_cnn",
    #         MAX_EPOCHS
    #     )
    #     results.append(result)
    # except ImportError:
    #     print("\nHybrid quantum model not available. Skipping.")

    # ========================================
    # Results Comparison
    # ========================================
    print("\n" + "="*60)
    print("Results Comparison")
    print("="*60 + "\n")

    df = pd.DataFrame(results)
    df = df.sort_values('val_accuracy', ascending=False)

    print(df.to_string(index=False))

    # Save results
    results_path = "comparison_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Print winner
    print("\n" + "="*60)
    best_model = df.iloc[0]
    print(f"Best Model: {best_model['experiment']}")
    print(f"Validation Accuracy: {best_model['val_accuracy']:.4f}")
    print(f"Validation Loss: {best_model['val_loss']:.4f}")
    print(f"Parameters: {best_model['parameters']:,}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
