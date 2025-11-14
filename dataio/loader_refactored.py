"""
Refactored EuroSAT dataset loader with improved configurability and best practices.
"""
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pytorch_lightning as pl
from PIL import Image
import os
import random
from typing import Optional, List, Tuple


class EuroSATDataset(Dataset):
    """
    EuroSAT dataset with configurable class selection.

    Args:
        root_dir: Root directory containing class folders
        transform: Optional transform to apply to images
        reduce_by_factor: Fraction of data to keep (1.0 = all data, 0.5 = half)
        selected_classes: List of class names to include (None = all classes)
        seed: Random seed for reproducible shuffling
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        reduce_by_factor: float = 1.0,
        selected_classes: Optional[List[str]] = None,
        seed: int = 42
    ):
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        # Get all available classes
        all_classes = sorted([d for d in os.listdir(root_dir)
                            if os.path.isdir(os.path.join(root_dir, d))])

        if len(all_classes) == 0:
            raise ValueError(f"No class directories found in {root_dir}")

        # Filter classes if specified
        if selected_classes is not None:
            invalid_classes = set(selected_classes) - set(all_classes)
            if invalid_classes:
                raise ValueError(f"Invalid classes: {invalid_classes}. "
                               f"Available classes: {all_classes}")
            self.classes = selected_classes
        else:
            self.classes = all_classes

        print(f"Using classes: {self.classes}")

        # Load data
        self.data = []
        for class_label in self.classes:
            class_path = os.path.join(root_dir, class_label)
            img_files = sorted(os.listdir(class_path))

            # Reduce dataset if specified
            if reduce_by_factor < 1.0:
                random.seed(seed)  # Use seed for reproducible reduction
                random.shuffle(img_files)
                keep_count = max(1, int(len(img_files) * reduce_by_factor))
                img_files = img_files[:keep_count]

            for img_file in img_files:
                img_path = os.path.join(class_path, img_file)
                self.data.append((img_path, self.classes.index(class_label)))

        # Shuffle data with seed for reproducibility
        random.seed(seed)
        random.shuffle(self.data)

        print(f"Loaded {len(self.data)} images from {len(self.classes)} classes")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def count_images_per_class(self) -> dict:
        """Count number of images in each class."""
        class_counts = {class_name: 0 for class_name in self.classes}
        for _, label in self.data:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        return class_counts

    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.classes


class EuroSATDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for EuroSAT dataset.

    Args:
        train_dir: Training data directory (default: dataset/training)
        val_dir: Validation data directory (default: dataset/validation)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        reduce_by_factor: Fraction of data to keep (1.0 = all data)
        selected_classes: List of class names to include (None = all)
        image_size: Resize images to this size (None = keep original)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        train_dir: str = "dataset/training",
        val_dir: str = "dataset/validation",
        batch_size: int = 64,
        num_workers: int = 4,
        reduce_by_factor: float = 1.0,
        selected_classes: Optional[List[str]] = None,
        image_size: Optional[int] = None,
        seed: int = 42
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.reduce_by_factor = reduce_by_factor
        self.selected_classes = selected_classes
        self.image_size = image_size
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation."""
        # Build transforms
        transform_list = []
        if self.image_size is not None:
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        transform_list.append(transforms.ToTensor())
        # Normalize with ImageNet stats (common practice for satellite imagery)
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
        transform = transforms.Compose(transform_list)

        # Create datasets
        self.train_dataset = EuroSATDataset(
            self.train_dir,
            transform=transform,
            reduce_by_factor=self.reduce_by_factor,
            selected_classes=self.selected_classes,
            seed=self.seed
        )

        self.valid_dataset = EuroSATDataset(
            self.val_dir,
            transform=transform,
            reduce_by_factor=self.reduce_by_factor,
            selected_classes=self.selected_classes,
            seed=self.seed
        )

        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.valid_dataset)}")
        print(f"Training class distribution: {self.train_dataset.count_images_per_class()}")
        print(f"Validation class distribution: {self.valid_dataset.count_images_per_class()}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def get_num_classes(self) -> int:
        """Get number of classes in the dataset."""
        return len(self.train_dataset.classes)
