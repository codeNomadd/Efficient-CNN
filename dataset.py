import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np
import torchvision
from torch.utils.data.sampler import WeightedRandomSampler

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
    def __init__(self, batch_size=128, num_workers=4):
        """Initialize CIFAR-100 dataset with native resolution (32x32)
        
        Note: We use native CIFAR-100 resolution instead of resizing to 224x224
        because:
        1. EfficientNet can work with any input size through compound scaling
        2. Native resolution preserves original image information
        3. Avoids artifacts from upscaling small images
        4. Better matches CIFAR-100's object scales
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transformations with improved augmentation for 32x32 images
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
    def get_data_loaders(self):
        """Get train and test data loaders with class balancing"""
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # Calculate class weights for balancing
        train_labels = [label for _, label in train_dataset]
        class_counts = torch.bincount(torch.tensor(train_labels))
        class_weights = 1. / class_counts
        sample_weights = class_weights[train_labels]
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader 