import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import random
import numpy as np
import os

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """Get the best available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device

class EfficientNetModel:
    def __init__(self, num_classes=100):
        """Initialize EfficientNet-B0 model with CIFAR-100 classes"""
        self.device = get_device()
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Modify the classifier for CIFAR-100
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        print(f"Model is on device: {next(self.model.parameters()).device}")
        
    def get_model(self):
        """Return the model"""
        return self.model

class CIFAR100Dataset:
    def __init__(self, batch_size=128, num_workers=4):
        """Initialize CIFAR-100 dataset with appropriate transformations"""
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transformations for training
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])
        
        # Define transformations for testing
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])
        
    def get_data_loaders(self):
        """Get train and test data loaders"""
        # Load CIFAR-100 dataset
        train_dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        test_dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader 