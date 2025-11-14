# Quick Start Guide

Get started with HQCNN in 5 minutes!

## 🚀 Setup (2 minutes)

```bash
# Clone and enter directory
git clone <repository-url>
cd qh_algorithms

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Get the Dataset (1 minute)

### Option 1: Download EuroSAT

```bash
# Download from official source
wget http://madm.dfki.de/files/sentinel/EuroSAT.zip
unzip EuroSAT.zip -d dataset/

# Organize into train/val
python scripts/split_dataset_fixed.py
```

### Option 2: Use Existing Dataset

If you already have the dataset:
```bash
python scripts/split_dataset_fixed.py --root /path/to/your/dataset/training
```

## 🏃 Train Your First Model (2 minutes)

### Classical Baseline

```bash
# Train LeNet5 on binary classification
python scripts/train_classical_baseline.py
```

This will:
- ✅ Set random seed for reproducibility
- ✅ Load EuroSAT dataset (AnnualCrop vs Forest)
- ✅ Train LeNet5 for 30 epochs
- ✅ Save best models to `saved_models/`
- ✅ Log to TensorBoard

### View Training Progress

```bash
# In another terminal
tensorboard --logdir=lightning_logs
```

Open http://localhost:6006

## 🎯 Common Tasks

### Change Classes

Edit `scripts/train_classical_baseline.py`:

```python
# Binary classification
SELECTED_CLASSES = ["Forest", "Highway"]

# Or use all 10 classes
SELECTED_CLASSES = None
```

### Use Different Model

```python
# In train_classical_baseline.py
MODEL_TYPE = "improved"  # Instead of "lenet5"
```

### Quick Experiment (Small Dataset)

```python
# In train_classical_baseline.py
REDUCE_BY_FACTOR = 0.1  # Use only 10% of data
MAX_EPOCHS = 5
```

### Compare Models

```bash
python scripts/compare_models.py
```

Trains multiple models and shows comparison table.

## 📈 Results

After training, find:

- **Models**: `saved_models/classical_lenet5_baseline/`
- **Logs**: `lightning_logs/classical_lenet5_baseline/`
- **Best checkpoint**: Printed at end of training

## 🔧 Custom Training

```python
from dataio.loader_refactored import EuroSATDataModule
from models.classical_cnn import ClassicalCNN
from utils.reproducibility import set_seed
from utils.training import create_trainer

# Set seed
set_seed(42)

# Load data
data = EuroSATDataModule(
    selected_classes=["Forest", "Residential"],
    batch_size=64,
    seed=42
)

# Create model
model = ClassicalCNN(
    model_type='improved',
    num_classes=2,
    learning_rate=0.001
)

# Train
trainer = create_trainer(max_epochs=20)
trainer.fit(model, data)
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
BATCH_SIZE = 16  # Or 8
```

### Slow Training on CPU

```python
# Use GPU if available
ACCELERATOR = "gpu"

# Or reduce dataset
REDUCE_BY_FACTOR = 0.5
```

### Import Errors

```bash
# Make sure you're in the right directory
cd qh_algorithms

# And virtual environment is activated
source venv/bin/activate
```

## 📚 Next Steps

1. **Read the full README**: See `README.md` for detailed documentation
2. **Explore notebooks**: Check out `HQCNN.ipynb` for quantum models
3. **Experiment**: Try different architectures and hyperparameters
4. **Compare**: Run `compare_models.py` to benchmark
5. **Contribute**: Found a bug or improvement? Open an issue!

## 💡 Tips

- Always set `seed=42` for reproducible results
- Start with small experiments (`REDUCE_BY_FACTOR=0.1`)
- Use TensorBoard to monitor training
- Save your best models
- Compare against classical baselines

## 🎓 Learning Resources

- **EuroSAT Paper**: https://arxiv.org/abs/1709.00029
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/
- **Quantum ML**: https://pennylane.ai/qml/

---

**Need Help?** Open an issue on GitHub!
