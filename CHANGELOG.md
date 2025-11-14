# Changelog

All notable changes to this project will be documented in this file.

## [Refactored Version] - 2025-11-14

### Added

#### New Modules
- `utils/reproducibility.py` - Seed setting and device management utilities
- `utils/training.py` - Training helpers and trainer configuration
- `models/classical_cnn.py` - Classical baseline CNN implementations
  - ClassicalLeNet5 - Traditional LeNet5 architecture
  - ImprovedCNN - Modern CNN with BatchNorm and Dropout
  - ClassicalCNN - PyTorch Lightning wrapper

#### New Scripts
- `scripts/train_classical_baseline.py` - Standalone training script with best practices
- `scripts/compare_models.py` - Automated model comparison tool
- `scripts/split_dataset_fixed.py` - Reproducible dataset splitting with fixed seeds

#### Data Loading
- `dataio/loader_refactored.py` - Improved EuroSAT data loader
  - Configurable class selection
  - Reproducible shuffling with seeds
  - Better error handling
  - Dataset statistics reporting
  - ImageNet normalization
  - Flexible dataset reduction

#### Documentation
- Comprehensive README.md with:
  - Installation instructions
  - Usage examples
  - Architecture descriptions
  - Best practices guide
  - References and citations
- CHANGELOG.md (this file)
- Inline code documentation and type hints

### Fixed

#### Reproducibility Issues
- Added fixed random seeds across all libraries (Python, NumPy, PyTorch, CUDA)
- Deterministic CUDNN operations
- Reproducible data shuffling
- Seeded dataset splitting

#### Data Handling
- Fixed arbitrary class selection (was first and last alphabetically)
- Now supports configurable class selection
- Added validation for selected classes
- Proper directory existence checking

#### Code Quality
- Removed hardcoded paths
- Added type hints throughout
- Comprehensive error messages
- Clean separation of concerns
- Removed large blocks of commented code

#### Training Infrastructure
- Proper mixed precision support (fixed deprecated autocast syntax)
- Correct device management (no GPU transfer in `__getitem__`)
- Early stopping callback
- Learning rate monitoring
- Model checkpointing (top-k models)
- TensorBoard logging with proper organization

### Improved

#### Configuration
- Centralized configuration in scripts
- Environment variable support
- Command-line argument parsing for scripts

#### Performance
- Efficient data loading (pin_memory, persistent_workers)
- Mixed precision training support
- Proper batch normalization usage
- Dropout regularization

#### Monitoring
- TensorBoard integration
- Progress bars with tqdm
- Dataset statistics printing
- Training metrics logging
- Model parameter counting

## [Original Version] - Previous

### Features

- Hybrid Quantum-Classical CNN implementation
- PennyLane integration (HQCNN.ipynb)
- Qiskit integration (qiskit_one_q.ipynb, esa_modello.ipynb)
- Basic EuroSAT data loader
- PyTorch Lightning training
- Dataset splitting utility

### Issues (Fixed in Refactored Version)

- No reproducibility guarantees (no fixed seeds)
- Hardcoded paths for specific environments
- Binary classification on arbitrary classes
- No classical baseline for comparison
- Limited configuration options
- Mixed Italian/English comments
- Inefficient GPU data transfer
- Deprecated PyTorch syntax

---

## Migration Guide

### From Original to Refactored Code

#### Data Loading

**Before:**
```python
from dataio.loader import EuroSATDataModule

data_module = EuroSATDataModule(
    num_workers=16,
    batch_size=8
)
```

**After:**
```python
from dataio.loader_refactored import EuroSATDataModule
from utils.reproducibility import set_seed

set_seed(42)

data_module = EuroSATDataModule(
    train_dir="dataset/training",
    val_dir="dataset/validation",
    batch_size=32,
    num_workers=4,
    selected_classes=["AnnualCrop", "Forest"],
    seed=42
)
```

#### Training

**Before:**
```python
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu"
)
```

**After:**
```python
from utils.training import create_trainer

trainer = create_trainer(
    max_epochs=30,
    accelerator="auto",
    experiment_name="my_experiment",
    early_stopping=True
)
```

#### Reproducibility

**New in Refactored:**
```python
from utils.reproducibility import set_seed, get_device

set_seed(42)  # For reproducibility
device = get_device(prefer_gpu=True)  # Automatic device selection
```

---

## Future Plans

### Upcoming Features
- [ ] Refactored hybrid quantum models
- [ ] Real quantum hardware integration
- [ ] Hyperparameter optimization
- [ ] Model interpretability (Grad-CAM)
- [ ] Multi-GPU training
- [ ] Docker support
- [ ] Unit tests
- [ ] CI/CD pipeline

### Under Consideration
- [ ] Transfer learning from ImageNet
- [ ] Data augmentation strategies
- [ ] Cross-validation support
- [ ] Ensemble methods
- [ ] Active learning
- [ ] Few-shot learning experiments
