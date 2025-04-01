import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import gc
from torch.cuda.amp import autocast, GradScaler
from model import EfficientNetModel
from dataset import CIFAR100Dataset, set_seed
from monitor import SystemMonitor

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
        
        # Tensorboard writer
        self.writer = SummaryWriter('runs/efficientnet_cifar100')
        
        # Training metrics
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
        # System monitor
        self.monitor = SystemMonitor()
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.get_model().train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer_steps = 0
        
        for i, (inputs, labels) in enumerate(tqdm(self.train_loader, desc='Training')):
            # Move data to GPU with non_blocking=True
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Mixed precision training
            with autocast():
                outputs = self.model.get_model()(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps  # Normalize loss
            
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                optimizer_steps += 1
            
            running_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Clear memory after each batch
            del outputs, loss
            clear_memory()
            
            # Update system metrics
            self.monitor.update_metrics()
        
        epoch_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return epoch_loss, accuracy
    
    def evaluate(self):
        """Evaluate on test set"""
        self.model.get_model().eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Evaluating'):
                # Move data to GPU with non_blocking=True
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model.get_model()(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Clear memory after each batch
                del outputs, loss
                clear_memory()
                
                # Update system metrics
                self.monitor.update_metrics()
        
        epoch_loss = running_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        return epoch_loss, accuracy
    
    def train(self, num_epochs=30, save_best=True):
        """Train the model"""
        best_accuracy = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Train and evaluate
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/test', test_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/test', test_acc, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Save best model
            if save_best and test_acc > best_accuracy:
                best_accuracy = test_acc
                self.model.save_checkpoint(
                    'checkpoints/best_model.pth',
                    epoch,
                    self.optimizer,
                    test_loss,
                    test_acc
                )
            
            # Plot current metrics
            self.plot_metrics()
            
            # Print system metrics
            self.monitor.print_metrics()
            
            # Clear memory after each epoch
            clear_memory()
        
        # Print training time
        training_time = time.time() - start_time
        print(f'\nTraining completed in {training_time:.2f} seconds')
        print(f'Best test accuracy: {best_accuracy:.2f}%')
        
        # Save final system metrics
        self.monitor.save_metrics()
        
    def plot_metrics(self):
        """Plot training metrics"""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
        
        # Clear matplotlib memory
        plt.clf()
        plt.close('all')
        clear_memory()

def main():
    # Set random seeds for reproducibility
    set_seed(42)
    
    # Clear memory at start
    clear_memory()
    
    # Create necessary directories
    print("Creating project directories...")
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Initialize model
    print("\nInitializing EfficientNet-B0 model...")
    model = EfficientNetModel()
    print(f"Model initialized on device: {model.device}")

    # Initialize dataset with optimized parameters
    print("\nLoading CIFAR-100 dataset...")
    dataset = CIFAR100Dataset(batch_size=192, num_workers=6)  # Balanced batch size and workers
    train_loader, test_loader = dataset.get_data_loaders()

    # Initialize trainer with gradient accumulation
    print("\nSetting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=0.001,
        gradient_accumulation_steps=3  # Balanced for effective batch size
    )

    # Train the model
    print("\nStarting training...")
    print(f"Training on device: {model.device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Effective batch size: {192 * 3}")  # Actual batch size * gradient accumulation steps
    trainer.train(num_epochs=30, save_best=True)  # Increased epochs to 30

    print("\nTraining completed! Check the following:")
    print("- Training metrics plot: training_metrics.png")
    print("- Best model checkpoint: checkpoints/best_model.pth")
    print("- Tensorboard logs: runs/efficientnet_cifar100/")
    print("- System metrics log: logs/system_metrics.txt")
    
    # Final memory cleanup
    clear_memory()

if __name__ == "__main__":
    main() 