import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import numpy as np
import gc
from torch.cuda.amp import autocast, GradScaler
from model import EfficientNetModel
from dataset import CIFAR100Dataset, set_seed
from pathlib import Path

def clear_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class Trainer:
    def __init__(self, model, train_loader, test_loader, learning_rate=0.001, gradient_accumulation_steps=4):
        """Initialize trainer"""
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = model.device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.get_model().parameters(), lr=learning_rate)
        
        # Learning rate scheduler with cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # Number of epochs for the first restart
            T_mult=2,  # Factor to increase T_0 after a restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.get_model().train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model.get_model()(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with gradient accumulation
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss/len(self.train_loader), 100.*correct/total

    def test(self):
        """Test the model"""
        self.model.get_model().eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast():
                    outputs = self.model.get_model()(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.3f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        return total_loss/len(self.test_loader), 100.*correct/total

    def train(self, num_epochs):
        """Train the model"""
        best_acc = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch: {epoch+1}/{num_epochs}')
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Test
            test_loss, test_acc = self.test()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update best accuracy
            if test_acc > best_acc:
                best_acc = test_acc
                print(f'New best test accuracy: {best_acc:.2f}%')
                # Save best model
                torch.save(self.model.get_model().state_dict(), 'best_model.pth')
        
        return best_acc

def main():
    # Clear memory at start
    clear_memory()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize model
    print("\nInitializing EfficientNet-B0 model...")
    model = EfficientNetModel()
    print(f"Model initialized on device: {model.device}")

    # Initialize dataset
    print("\nLoading CIFAR-100 dataset...")
    dataset = CIFAR100Dataset(batch_size=128, num_workers=4)
    train_loader, test_loader = dataset.get_data_loaders()

    # Initialize trainer
    print("\nSetting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=model.device
    )

    # Train the model
    print("\nStarting training...")
    print(f"Training on device: {model.device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    best_accuracy = trainer.train(num_epochs=100)
    print(f"\nTraining completed! Best accuracy: {best_accuracy:.2f}%")
    
    # Final memory cleanup
    clear_memory()

if __name__ == "__main__":
    main() 