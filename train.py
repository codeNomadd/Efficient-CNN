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

class DynamicLRScheduler:
    def __init__(self, optimizer, initial_lr=0.001, min_lr=1e-6, patience=3, window_size=3):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        self.window_size = window_size
        self.loss_history = []
        self.lr_history = []
        
    def step(self, val_loss):
        self.loss_history.append(val_loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
        
        if len(self.loss_history) < self.window_size:
            return
        
        # Calculate loss change rate
        loss_changes = [self.loss_history[i] - self.loss_history[i-1] for i in range(1, len(self.loss_history))]
        avg_loss_change = sum(loss_changes) / len(loss_changes)
        
        # Calculate loss volatility
        loss_std = np.std(self.loss_history)
        
        # Dynamic learning rate adjustment
        current_lr = self.optimizer.param_groups[0]['lr']
        
        if avg_loss_change > 0:  # Loss is increasing
            # More aggressive reduction when loss is increasing and volatile
            reduction_factor = 1 - (0.1 * (1 + loss_std))
        else:  # Loss is decreasing
            # Smaller reduction when loss is decreasing
            reduction_factor = 1 - (0.05 * (1 - loss_std))
        
        # Ensure reduction factor is between 0.5 and 0.95
        reduction_factor = max(0.5, min(0.95, reduction_factor))
        
        # Apply learning rate update
        new_lr = current_lr * reduction_factor
        new_lr = max(self.min_lr, new_lr)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.lr_history.append(new_lr)
        
        # Print adjustment info
        print(f"\nLearning Rate Adjustment:")
        print(f"Average Loss Change: {avg_loss_change:.4f}")
        print(f"Loss Volatility: {loss_std:.4f}")
        print(f"Reduction Factor: {reduction_factor:.4f}")
        print(f"Old LR: {current_lr:.6f}")
        print(f"New LR: {new_lr:.6f}")
    
    def state_dict(self):
        """Return scheduler state as a dictionary"""
        return {
            'initial_lr': self.initial_lr,
            'min_lr': self.min_lr,
            'patience': self.patience,
            'window_size': self.window_size,
            'loss_history': self.loss_history,
            'lr_history': self.lr_history
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from a dictionary"""
        self.initial_lr = state_dict['initial_lr']
        self.min_lr = state_dict['min_lr']
        self.patience = state_dict['patience']
        self.window_size = state_dict['window_size']
        self.loss_history = state_dict['loss_history']
        self.lr_history = state_dict['lr_history']

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer, checkpoint_dir):
    """Train the model with dynamic learning rate scheduling based on loss changes"""
    best_val_loss = float('inf')
    best_model_path = checkpoint_dir / 'best_model.pth'
    patience = 3
    patience_counter = 0
    gradient_accumulation_steps = 4  # Accumulate gradients over 4 steps
    
    for epoch in range(num_epochs):
        # Training phase
        model.get_model().train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training...")
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model.get_model()(images)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps  # Normalize loss
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * gradient_accumulation_steps
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Clear memory after each batch
            del outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Handle remaining gradients
        if (i + 1) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.get_model().eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        print("Validating...")
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model.get_model()(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Clear memory after each batch
                del outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', val_acc, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.get_model().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.get_model().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

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

    # Initialize dataset with smaller batch size
    print("\nLoading CIFAR-100 dataset...")
    dataset = CIFAR100Dataset(batch_size=64, num_workers=4)  # Reduced batch size and workers
    train_loader, test_loader = dataset.get_data_loaders()

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.get_model().parameters(), lr=0.001)
    scheduler = DynamicLRScheduler(
        optimizer,
        initial_lr=0.001,
        min_lr=1e-6,
        patience=3,
        window_size=3
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/efficientnet_cifar100')
    
    # Train the model
    print("\nStarting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=30,
        device=model.device,
        writer=writer,
        checkpoint_dir=Path('checkpoints')
    )
    
    writer.close()
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 