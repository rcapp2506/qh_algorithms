"""
Reproducibility utilities for ensuring deterministic training.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✓ Random seed set to {seed} for reproducibility")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device for training.

    Args:
        prefer_gpu: Whether to prefer GPU if available (default: True)

    Returns:
        torch.device: Selected device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")

    return device
