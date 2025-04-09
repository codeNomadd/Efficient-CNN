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
import random
import json
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from model import EfficientNetModel
from dataset import CIFAR100Dataset
from monitor import SystemMonitor
import platform
import pandas as pd

def clear_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class SeedManager:
    def __init__(self, seed=None, save_dir='checkpoints'):
        """Initialize seed manager"""
        self.save_dir = save_dir
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.seed_info = {
            'seed': self.seed,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
    def set_seed(self):
        """Set all random seeds"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
    def save_seed_info(self):
        """Save seed information to file"""
        os.makedirs(self.save_dir, exist_ok=True)
        seed_file = os.path.join(self.save_dir, 'seed_info.json')
        
        # Load existing seed info if it exists
        if os.path.exists(seed_file):
            with open(seed_file, 'r') as f:
                existing_info = json.load(f)
                if not isinstance(existing_info, list):
                    existing_info = [existing_info]
        else:
            existing_info = []
        
        # Add new seed info
        existing_info.append(self.seed_info)
        
        # Save updated seed info
        with open(seed_file, 'w') as f:
            json.dump(existing_info, f, indent=4)
        
        print(f"\nSeed information saved to {seed_file}")
        print(f"Using seed: {self.seed}")
        
    def load_seed_info(self, index=-1):
        """Load seed information from file"""
        seed_file = os.path.join(self.save_dir, 'seed_info.json')
        if os.path.exists(seed_file):
            with open(seed_file, 'r') as f:
                seed_info_list = json.load(f)
                if not isinstance(seed_info_list, list):
                    seed_info_list = [seed_info_list]
                
                # Get the specified seed info (default to latest)
                seed_info = seed_info_list[index]
                self.seed = seed_info['seed']
                self.seed_info = seed_info
                print(f"\nLoaded seed information from {seed_file}")
                print(f"Using seed: {self.seed}")
                return True
        return False

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

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, run_dir):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.run_dir = run_dir
        
        # Initialize metrics
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
        self.learning_rates = []
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=run_dir)
        
        # Initialize model - ensure it's on the correct device
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(device)
        else:
            self.model = self.model.get_model().to(device)
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # Initialize learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.001,
            epochs=100,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # 30% of training for warmup
            div_factor=10.0,  # Initial learning rate = max_lr/10
            final_div_factor=1000.0,  # Min learning rate = max_lr/1000
            anneal_strategy='cos'
        )
        
        # Initialize loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Initialize mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=15,
            min_delta=0.001,
            verbose=True
        )
        
        # Initialize checkpointing
        self.checkpoint = Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            path=os.path.join(run_dir, 'checkpoints')
        )
        
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total,
                'lr': self.scheduler.get_last_lr()[0]
            })
        
        # Log metrics
        epoch_loss = running_loss/len(self.train_loader)
        epoch_acc = 100.*correct/total
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        self.learning_rates.append(self.scheduler.get_last_lr()[0])
        
        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], epoch)
        
        return epoch_loss, epoch_acc
    
    def test_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Mixed precision inference
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Log metrics
        epoch_loss = running_loss/len(self.test_loader)
        epoch_acc = 100.*correct/total
        self.test_losses.append(epoch_loss)
        self.test_accs.append(epoch_acc)
        
        self.writer.add_scalar('Loss/test', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/test', epoch_acc, epoch)
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs=100):
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            # Train and test
            train_loss, train_acc = self.train_epoch(epoch)
            test_loss, test_acc = self.test_epoch(epoch)
            
            # Print epoch results
            print(f'Epoch {epoch}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            
            # Save checkpoint
            self.checkpoint.save(epoch, test_acc)
            
            # Early stopping
            if self.early_stopping(test_loss):
                print('Early stopping triggered')
                break
            
            # Update best accuracy
            if test_acc > best_acc:
                best_acc = test_acc
                print(f'New best test accuracy: {best_acc:.2f}%')
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.run_dir, 'final_model.pth'))
        
        # Close TensorBoard writer
        self.writer.close()
        
        return best_acc

def main():
    # Clear memory at start
    clear_memory()
    
    # Create necessary directories
    print("Creating project directories...")
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Initialize seed manager
    print("\nInitializing seed manager...")
    seed_manager = SeedManager()
    seed_manager.set_seed()
    seed_manager.save_seed_info()

    # Initialize model
    print("\nInitializing EfficientNet-B0 model...")
    model = EfficientNetModel()
    print(f"Model initialized on device: {model.device}")

    # Initialize dataset with optimized parameters
    print("\nLoading CIFAR-100 dataset...")
    dataset = CIFAR100Dataset(batch_size=128, num_workers=4)
    train_loader, test_loader = dataset.get_data_loaders()

    # Initialize trainer
    print("\nSetting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=model.device,
        run_dir='runs/efficientnet_cifar100/'
    )

    # Train the model
    print("\nStarting training...")
    print(f"Training on device: {model.device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    best_accuracy = trainer.train(num_epochs=100)

    print("\nTraining completed! Check the following:")
    print("- Training metrics plot: training_metrics.png")
    print("- Best model checkpoint: checkpoints/best_model.pth")
    print("- Tensorboard logs: runs/efficientnet_cifar100/")
    print("- System metrics log: logs/system_metrics.txt")
    print("- Seed information: checkpoints/seed_info.json")
    
    # Final memory cleanup
    clear_memory()

if __name__ == "__main__":
    main() 