import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CIFAR100Dataset:
    def __init__(self, batch_size=128, num_workers=4, seed=42):
        """Initialize CIFAR-100 dataset with transforms"""
        self.batch_size = batch_size
        self.num_workers = num_workers
        set_seed(seed)  # Set random seed
        
        print("Setting up data transforms...")
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(224),  # EfficientNet expects 224x224
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def get_data_loaders(self):
        """Get train and test data loaders"""
        print("Downloading CIFAR-100 dataset...")
        
        # Load training data
        print("Loading training data...")
        train_dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # Load test data
        print("Loading test data...")
        test_dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        print(f"Dataset loaded successfully:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of workers: {self.num_workers}")
        
        print("Creating data loaders...")
        # Create data loaders with GPU optimizations
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            generator=torch.Generator().manual_seed(42)  # Set seed for shuffling
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,  # Use same batch size for consistency
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # Store class names
        self.classes = train_dataset.classes
        
        print("Data loaders created successfully!")
        return train_loader, test_loader 