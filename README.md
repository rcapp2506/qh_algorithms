# Hybrid Quantum Convolutional Neural Networks for EuroSAT Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive implementation of **Hybrid Quantum-Classical Convolutional Neural Networks (HQCNN)** for satellite image classification using the EuroSAT dataset. This repository explores the integration of quantum computing with classical deep learning for Earth observation tasks.

![HQCNN Architecture](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/4609443/9656571/9647979/sebas9-3134785-large.gif)

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Best Practices](#best-practices)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This project implements and compares different approaches to satellite image classification:

1. **Classical CNN Baselines** - Pure classical deep learning models
2. **Hybrid Quantum-Classical Models** - Integration of quantum circuits with CNNs using:
   - **PennyLane** framework
   - **Qiskit** framework

The goal is to evaluate the potential advantages of quantum computing in remote sensing image classification tasks.

## ✨ Features

### Refactored Codebase (New!)

- ✅ **Reproducibility**: Fixed random seeds across all libraries
- ✅ **Configurable Data Loading**: Flexible class selection and dataset reduction
- ✅ **Classical Baselines**: LeNet5 and improved CNN architectures
- ✅ **Best Practices**: Proper logging, checkpointing, and early stopping
- ✅ **Clean Code**: Type hints, documentation, and modular design
- ✅ **Comparison Tools**: Scripts to compare classical vs quantum models
- ✅ **Device Management**: Proper CPU/GPU handling with mixed precision support

### Original Implementation

- 🔬 Hybrid quantum-classical neural networks
- 🛰️ EuroSAT satellite image classification
- 📊 PyTorch Lightning training infrastructure
- 🎯 TensorBoard logging and visualization
- 🔄 Multiple quantum framework support (PennyLane, Qiskit)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/qh_algorithms.git
cd qh_algorithms

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy
matplotlib
torch>=2.0.0
torchvision
pytorch-lightning>=2.0.0
qiskit>=0.43.0
pennylane>=0.30.0
tqdm
opencv-python
torchinfo
pandas
tensorboard
```

## 📊 Dataset

### EuroSAT Dataset

The **EuroSAT** dataset is a land use and land cover classification dataset based on Sentinel-2 satellite images.

- **Classes**: 10 (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake)
- **Total Images**: 27,000 labeled images
- **Image Size**: 64×64 pixels
- **Spectral Bands**: RGB (13 spectral bands available in full dataset)

### Download Dataset

```bash
# The dataset will be downloaded automatically when running the notebooks
# Or download manually from: https://github.com/phelber/EuroSAT

# Expected structure:
# dataset/
# ├── training/
# │   ├── AnnualCrop/
# │   ├── Forest/
# │   └── ...
# └── validation/
#     ├── AnnualCrop/
#     ├── Forest/
#     └── ...
```

### Split Dataset

```bash
# Split training data into train/validation (80/20)
python scripts/split_dataset_fixed.py --root dataset/training --split 0.2 --seed 42

# With custom settings
python scripts/split_dataset_fixed.py --root dataset/training --split 0.3 --seed 123 --copy
```

## 📁 Project Structure

```
qh_algorithms/
├── dataio/
│   ├── loader.py                    # Original data loader
│   └── loader_refactored.py         # Improved data loader with best practices
├── models/
│   ├── classical_cnn.py             # Classical baseline models
│   └── hybrid_cnn.py                # Hybrid quantum-classical models (WIP)
├── scripts/
│   ├── split_dataset.py             # Original dataset splitter
│   ├── split_dataset_fixed.py       # Fixed dataset splitter with seeds
│   ├── train_classical_baseline.py  # Training script for classical models
│   └── compare_models.py            # Model comparison script
├── utils/
│   ├── reproducibility.py           # Seed setting and device management
│   └── training.py                  # Training utilities and helpers
├── HQCNN.ipynb                      # PennyLane hybrid model notebook
├── esa_modello.ipynb                # Alternative implementation notebook
├── qiskit_one_q.ipynb               # Qiskit quantum convolution notebook
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🏃 Quick Start

### 1. Train Classical Baseline

```bash
python scripts/train_classical_baseline.py
```

This will train a LeNet5 model on the EuroSAT dataset with optimal settings.

### 2. Compare Models

```bash
python scripts/compare_models.py
```

This trains multiple models and generates a comparison report.

### 3. View Training Progress

```bash
tensorboard --logdir=lightning_logs
```

Then open http://localhost:6006 in your browser.

## 💻 Usage

### Custom Training

```python
from dataio.loader_refactored import EuroSATDataModule
from models.classical_cnn import ClassicalCNN
from utils.reproducibility import set_seed
from utils.training import create_trainer

# Set seed for reproducibility
set_seed(42)

# Create data module
data_module = EuroSATDataModule(
    train_dir="dataset/training",
    val_dir="dataset/validation",
    batch_size=32,
    selected_classes=["AnnualCrop", "Forest"],  # Binary classification
    image_size=64,
    seed=42
)

# Create model
model = ClassicalCNN(
    model_type='improved',
    in_shape=(3, 64, 64),
    num_classes=2,
    learning_rate=0.001
)

# Create trainer
trainer = create_trainer(
    max_epochs=30,
    experiment_name="my_experiment"
)

# Train
trainer.fit(model, data_module)
```

### Multi-Class Classification

```python
# Use all 10 classes
data_module = EuroSATDataModule(
    train_dir="dataset/training",
    val_dir="dataset/validation",
    selected_classes=None,  # None = all classes
    seed=42
)

data_module.setup()

model = ClassicalCNN(
    model_type='improved',
    num_classes=data_module.get_num_classes(),  # Automatically get 10
    learning_rate=0.001
)
```

### Reduced Dataset (for quick experiments)

```python
# Use only 10% of data
data_module = EuroSATDataModule(
    train_dir="dataset/training",
    val_dir="dataset/validation",
    reduce_by_factor=0.1,  # Use 10% of data
    seed=42
)
```

## 🏗️ Model Architectures

### Classical LeNet5

A classic CNN architecture adapted for EuroSAT:

```
Input (3×64×64)
    ↓
Conv2d(3→6, 5×5) + ReLU + MaxPool
    ↓
Conv2d(6→16, 5×5) + ReLU + MaxPool
    ↓
Flatten
    ↓
FC(→120) + ReLU
    ↓
FC(→84) + ReLU
    ↓
FC(→num_classes)
```

**Parameters**: ~61,000

### Improved CNN

Modern CNN with batch normalization and dropout:

```
Input (3×64×64)
    ↓
[Conv(3→32) + BN + ReLU + Conv(32→32) + BN + ReLU + MaxPool] ×1
    ↓
[Conv(32→64) + BN + ReLU + Conv(64→64) + BN + ReLU + MaxPool] ×1
    ↓
[Conv(64→128) + BN + ReLU + Conv(128→128) + BN + ReLU + MaxPool] ×1
    ↓
Flatten + Dropout(0.5)
    ↓
FC(→256) + ReLU + Dropout(0.5)
    ↓
FC(→num_classes)
```

**Parameters**: ~2.4M

### Hybrid Quantum CNN (PennyLane)

Classical preprocessing + quantum layer:

```
Input (3×64×64)
    ↓
Classical Conv Layers
    ↓
Quantum Layer (4 qubits, 2 layers)
    ├── BasicEntangledCircuit
    └── Measurement
    ↓
Classical FC Layers
    ↓
Output
```

### Hybrid Quantum CNN (Qiskit)

Quantum convolution approach:

```
Input (3×64×64)
    ↓
Classical Conv + MaxPool
    ↓
Quantum Convolution Layer
    ├── Feature Map: PauliFeatureMap(4)
    ├── Conv Layer (4 qubits)
    ├── Pool Layer (4→2 qubits)
    ├── Conv Layer (2 qubits)
    ├── Pool Layer (2→1 qubit)
    └── Z Measurement
    ↓
Classical FC Layers
    ↓
Output
```

## 📈 Results

### Expected Performance

| Model | Classes | Val Accuracy | Val Loss | Parameters | Training Time* |
|-------|---------|--------------|----------|------------|----------------|
| LeNet5 | 2 | ~95% | ~0.15 | 61K | ~5 min |
| Improved CNN | 2 | ~97% | ~0.10 | 2.4M | ~15 min |
| LeNet5 | 10 | ~88% | ~0.35 | 61K | ~10 min |
| Improved CNN | 10 | ~94% | ~0.20 | 2.4M | ~30 min |

*Training time on single GPU (RTX 3090)

### Binary Classification Results (from notebooks)

From `qiskit_one_q.ipynb`:
- **Training Accuracy**: 89.75%
- **Validation Accuracy**: 89.33%
- **Training Loss**: 0.660
- **Validation Loss**: 0.643

## ✅ Best Practices Implemented

### Reproducibility

- ✅ Fixed random seeds (Python, NumPy, PyTorch, CUDA)
- ✅ Deterministic CUDNN operations
- ✅ Reproducible data shuffling
- ✅ Saved hyperparameters in checkpoints

### Training

- ✅ PyTorch Lightning for clean training loops
- ✅ Early stopping to prevent overfitting
- ✅ Learning rate scheduling
- ✅ Model checkpointing (save top-k models)
- ✅ TensorBoard logging
- ✅ Mixed precision training support
- ✅ Proper train/val/test splits

### Data

- ✅ Data normalization (ImageNet statistics)
- ✅ Efficient data loading (num_workers, pin_memory)
- ✅ Configurable class selection
- ✅ Dataset statistics reporting
- ✅ Reproducible data splits

### Code Quality

- ✅ Type hints
- ✅ Comprehensive documentation
- ✅ Modular design
- ✅ Error handling
- ✅ Clean separation of concerns

## 📚 References

### Papers

1. **Sebastianelli, A., et al.** (2021). "On circuit-based hybrid quantum neural networks for remote sensing imagery classification." *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 15, 565-580.

2. **Zaidenberg, D. A., et al.** (2021). "Advantages and bottlenecks of quantum machine learning for remote sensing." *2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS*, pp. 5680-5683.

3. **Helber, P., et al.** (2019). "Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification." *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*.

### Resources

- **EuroSAT Dataset**: https://github.com/phelber/EuroSAT
- **Qiskit Tutorials**: https://qiskit.org/documentation/machine-learning/tutorials/
- **PennyLane QML**: https://pennylane.ai/qml/demos_qml.html
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/stable/

## 🐛 Known Issues & Future Work

### Current Limitations

1. **Binary Classification**: Original implementation uses only 2 classes
   - ✅ **Fixed**: Refactored loader supports all classes

2. **Hardcoded Paths**: Absolute paths in original notebooks
   - ✅ **Fixed**: Relative paths in refactored scripts

3. **No Reproducibility**: Missing random seeds
   - ✅ **Fixed**: Comprehensive seed setting

4. **Limited Baselines**: No classical comparison
   - ✅ **Fixed**: Added classical baselines

### Future Improvements

- [ ] Implement quantum models with refactored framework
- [ ] Add more quantum circuit architectures
- [ ] Benchmark on real quantum hardware
- [ ] Multi-GPU distributed training
- [ ] Hyperparameter optimization (Optuna)
- [ ] Class activation maps (Grad-CAM)
- [ ] Model interpretability tools
- [ ] Docker containerization
- [ ] Comprehensive unit tests
- [ ] CI/CD pipeline

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- EuroSAT dataset creators
- Qiskit and PennyLane development teams
- PyTorch Lightning community
- Original HQCNN paper authors

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: This repository contains both the original implementation (notebooks) and a refactored version (scripts and modules) with best practices. New users are encouraged to start with the refactored code.
