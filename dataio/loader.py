import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pytorch_lightning as pl
from PIL import Image
import os
import random

class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None, reduce_by_half=False):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # get the list of all files and directories in the specified directory
        self.classes = [self.classes[0], self.classes[-1]]  # include only the first and last class

        self.data = []
        for class_label in self.classes:
            class_path = os.path.join(root_dir, class_label)
            img_files = os.listdir(class_path)
            
            if reduce_by_half:
                img_files = img_files[:len(img_files) // 2]  # keep only half of the images
            
            for img_file in img_files:
                img_path = os.path.join(class_path, img_file)
                self.data.append((img_path, self.classes.index(class_label)))  # add an index to the image path
        
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label 
    
    def count_images_per_class(self):
        class_counts = {class_name: 0 for class_name in self.classes}
        for _, label in self.data:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        return class_counts


class EuroSATDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size=64, num_workers=9, reduce_by_half=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.reduce_by_half = reduce_by_half  # store this option to apply in setup

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Apply reduce_by_half option to each dataset
        self.train_dataset = EuroSATDataset(self.train_dir, transform=transform, reduce_by_half=self.reduce_by_half)
        self.valid_dataset = EuroSATDataset(self.val_dir, transform=transform, reduce_by_half=self.reduce_by_half)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
'''
#forzo il sisitema ad andare su gpu
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pytorch_lightning as pl
from PIL import Image
import os
import random

class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None, reduce_by_half=False, device='cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device  # Specifica il dispositivo (CPU o GPU)
        self.classes = sorted(os.listdir(root_dir))
        self.classes = [self.classes[0], self.classes[-1]]  # Includi solo la prima e l'ultima classe

        self.data = []
        for class_label in self.classes:
            class_path = os.path.join(root_dir, class_label)
            img_files = os.listdir(class_path)

            if reduce_by_half:
                img_files = img_files[:len(img_files) // 2]  # Tieni solo metà delle immagini

            for img_file in img_files:
                img_path = os.path.join(class_path, img_file)
                self.data.append((img_path, self.classes.index(class_label)))

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Sposta l'immagine e l'etichetta sulla GPU
        image = image.to(self.device)
        label = torch.tensor(label, device=self.device)

        return image, label

    def count_images_per_class(self):
        class_counts = {class_name: 0 for class_name in self.classes}
        for _, label in self.data:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        return class_counts


class EuroSATDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size=64, num_workers=9, reduce_by_half=False, device='cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.reduce_by_half = reduce_by_half
        self.device = device  # Aggiungi il dispositivo

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Applica reduce_by_half e device a ciascun dataset
        self.train_dataset = EuroSATDataset(self.train_dir, transform=transform, reduce_by_half=self.reduce_by_half, device=self.device)
        self.valid_dataset = EuroSATDataset(self.val_dir, transform=transform, reduce_by_half=self.reduce_by_half, device=self.device)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
'''


