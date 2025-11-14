"""
Dataset splitting utility with reproducibility.

Splits the EuroSAT dataset into training and validation sets with
a fixed random seed for reproducibility.

Usage:
    python scripts/split_dataset_fixed.py
"""
import shutil
import random
import glob
import os
import argparse


def split_dataset(
    root: str = 'dataset/training',
    split_factor: float = 0.2,
    seed: int = 42,
    copy_mode: bool = False
):
    """
    Split dataset into training and validation sets.

    Args:
        root: Path to the training data directory
        split_factor: Fraction of data to use for validation (0.0-1.0)
        seed: Random seed for reproducibility
        copy_mode: If True, copy files instead of moving them
    """
    # Set random seed for reproducibility
    random.seed(seed)

    print(f"Splitting dataset from: {root}")
    print(f"Validation split: {split_factor*100:.1f}%")
    print(f"Random seed: {seed}")
    print(f"Mode: {'copy' if copy_mode else 'move'}")

    total_moved = 0
    class_stats = {}

    for cl in glob.glob(os.path.join(root, '*')):
        if not os.path.isdir(cl):
            continue

        class_name = os.path.basename(cl)
        val_dir = cl.replace('training', 'validation')

        # Create validation directory
        os.makedirs(val_dir, exist_ok=True)

        # Get all images in this class
        imgs_in_cl = glob.glob(os.path.join(cl, '*'))
        imgs_in_cl = sorted(imgs_in_cl)  # Sort for reproducibility

        # Shuffle with seed
        random.shuffle(imgs_in_cl)

        # Calculate split point
        val_count = int(len(imgs_in_cl) * split_factor)
        validation_imgs = imgs_in_cl[:val_count]

        # Move or copy images
        operation = shutil.copy2 if copy_mode else shutil.move

        for img in validation_imgs:
            dest = img.replace('training', 'validation')
            operation(img, dest)

        total_moved += len(validation_imgs)
        class_stats[class_name] = {
            'total': len(imgs_in_cl),
            'validation': len(validation_imgs),
            'training': len(imgs_in_cl) - len(validation_imgs)
        }

        print(f"  {class_name}: {len(validation_imgs)}/{len(imgs_in_cl)} images to validation")

    print(f"\nTotal images processed: {total_moved}")
    print("\nClass distribution:")
    print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Total':<10}")
    print("-" * 55)

    for class_name, stats in sorted(class_stats.items()):
        print(f"{class_name:<20} {stats['training']:<10} {stats['validation']:<10} {stats['total']:<10}")

    print(f"\nDataset split complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split EuroSAT dataset')
    parser.add_argument(
        '--root',
        type=str,
        default='dataset/training',
        help='Path to training data directory'
    )
    parser.add_argument(
        '--split',
        type=float,
        default=0.2,
        help='Validation split fraction (0.0-1.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them'
    )

    args = parser.parse_args()

    split_dataset(
        root=args.root,
        split_factor=args.split,
        seed=args.seed,
        copy_mode=args.copy
    )
