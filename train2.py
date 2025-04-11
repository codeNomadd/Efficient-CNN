import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import numpy as np
import gc
import json
import platform
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
from model import EfficientNetModel, CIFAR100Dataset, set_seed
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import signal
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR

class ModelEMA:
    """Model Exponential Moving Average"""
    def __init__(self, model, decay=0.999):
        self.ema = {}
        self.decay = decay
        self.model = model
        self._register_model()
    
    def _register_model(self):
        """Register model parameters for EMA"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema[name] = self.ema[name] * self.decay + param.data * (1 - self.decay)
    
    def apply(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.ema[name])
    
    def state_dict(self):
        """Return EMA state dict"""
        return self.ema
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict"""
        self.ema = state_dict

def mixup_data(x, y, alpha=0.1):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class Trainer:
    def __init__(self, model, train_loader, test_loader, learning_rate=0.01, gradient_accumulation_steps=4, use_mixup=True, checkpoint=None):
        """Initialize trainer for phase 2"""
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = model.device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixup = use_mixup
        self.best_acc = 0
        self.current_epoch = 0
        self.interrupted = False
        self.patience = 5
        self.no_improve_epochs = 0
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # SGD with momentum and weight decay
        self.optimizer = optim.SGD(
            model.get_model().parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )
        
        # Cosine annealing learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=20,  # 20 epochs for phase 2
            eta_min=1e-6
        )
        
        # Load optimizer and scheduler states if checkpoint exists
        if checkpoint is not None:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Optimizer state loaded from checkpoint")
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Scheduler state loaded from checkpoint")
                print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Initialize EMA
        self.ema = ModelEMA(model.get_model(), decay=0.999)
        if checkpoint is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            print(f"EMA state loaded from checkpoint")
        
        # Create directories
        os.makedirs('metrics', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        
        # Initialize metrics file
        with open('metrics/phase2_training_history.csv', 'w') as f:
            f.write('epoch,train_loss,test_loss,train_accuracy,test_accuracy,learning_rate,epoch_time\n')
        
        # Initialize metrics lists
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.learning_rates = []

        # Set up signal handler
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signal"""
        print("\nTraining interrupted! Saving checkpoints...")
        self.interrupted = True
        self.save_checkpoint(self.current_epoch, self.test_losses[-1], self.test_accuracies[-1], is_best=(self.test_accuracies[-1] > self.best_acc))
        print("Checkpoints saved. Exiting...")
        sys.exit(0)

    def save_checkpoint(self, epoch, test_loss, test_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.get_model().state_dict(),
            'ema_state_dict': self.ema.state_dict(),  # Use proper state_dict method
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': test_loss,
            'accuracy': test_acc,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'learning_rates': self.learning_rates
        }
        
        if is_best:
            torch.save(checkpoint, 'checkpoints/phase2_best_model.pth')
            self.best_acc = test_acc
            print(f'New best model saved with accuracy: {test_acc:.2f}%')
        else:
            torch.save(checkpoint, f'checkpoints/phase2_epoch_{epoch+1}.pth')
            print(f'Checkpoint saved for epoch {epoch+1}')

    def train_epoch(self):
        """Train for one epoch with optional Mixup"""
        self.model.get_model().train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply Mixup if enabled
            if self.use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.1)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model.get_model()(inputs)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with gradient accumulation
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.get_model().parameters(), max_norm=1.0)
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update EMA
                self.ema.update()
            
            # Update progress bar with original accuracy
            with torch.no_grad():
                original_outputs = self.model.get_model()(inputs)
                _, predicted = original_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_time = time.time() - epoch_start
        return total_loss/len(self.train_loader), 100.*correct/total, epoch_time

    def test(self):
        """Test the model using EMA weights"""
        # Store original model state
        original_model = self.model.get_model()
        original_mode = original_model.training
        
        # Apply EMA weights and set to eval mode
        self.ema.apply()
        self.model.get_model().eval()
        
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
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
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.3f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        # Restore original model state
        self.model.get_model().train(original_mode)
        
        return total_loss/len(self.test_loader), 100.*correct/total, all_preds, all_targets

    def train(self, num_epochs, start_epoch=30):
        """Train the model with early stopping"""
        try:
            for epoch in range(start_epoch, start_epoch + num_epochs):
                if self.interrupted:
                    break
                    
                self.current_epoch = epoch
                print(f'\nEpoch: {epoch+1}/{start_epoch + num_epochs}')
                
                # Train
                train_loss, train_acc, epoch_time = self.train_epoch()
                
                # Test
                test_loss, test_acc, preds, targets = self.test()
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update learning rate
                self.scheduler.step()
                
                # Save metrics to CSV
                with open('metrics/phase2_training_history.csv', 'a') as f:
                    f.write(f'{epoch+1},{train_loss:.4f},{test_loss:.4f},{train_acc:.2f},{test_acc:.2f},{current_lr:.6f},{epoch_time:.2f}\n')
                
                # Update metrics lists
                self.train_losses.append(train_loss)
                self.test_losses.append(test_loss)
                self.train_accuracies.append(train_acc)
                self.test_accuracies.append(test_acc)
                self.learning_rates.append(current_lr)
                
                # Update best accuracy and save checkpoint
                if test_acc > self.best_acc:
                    self.best_acc = test_acc
                    self.no_improve_epochs = 0
                    print(f'New best test accuracy: {self.best_acc:.2f}%')
                    self.save_checkpoint(epoch, test_loss, test_acc, is_best=True)
                else:
                    self.no_improve_epochs += 1
                    if self.no_improve_epochs >= self.patience:
                        print(f'\nEarly stopping triggered after {self.patience} epochs without improvement')
                        break
                
                # Save checkpoint every epoch in phase 2
                self.save_checkpoint(epoch, test_loss, test_acc)
                
                # Plot metrics
                plot_metrics(
                    self.train_losses,
                    self.test_losses,
                    self.train_accuracies,
                    self.test_accuracies,
                    self.learning_rates
                )
            
            return self.best_acc
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            print("Attempting to save checkpoints before exiting...")
            self.save_checkpoint(self.current_epoch, self.test_losses[-1], self.test_accuracies[-1], is_best=(self.test_accuracies[-1] > self.best_acc))
            raise e

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.get_model().load_state_dict(checkpoint['model_state_dict'])
    model.get_model().train()  # Set model to training mode
    epoch = checkpoint['epoch']
    
    # Get best accuracy from test accuracies if available, otherwise use saved accuracy
    best_acc = checkpoint.get('accuracy', -1)
    if 'test_accuracies' in checkpoint:
        best_acc = max(checkpoint['test_accuracies'])  # get best recorded test acc
    
    # Print detailed checkpoint information
    print(f"\nCheckpoint Information:")
    print(f"Epoch: {epoch}")
    print(f"Saved accuracy: {checkpoint.get('accuracy', -1):.2f}%")
    if 'test_accuracies' in checkpoint:
        print(f"Best test accuracy: {best_acc:.2f}%")
        print(f"Test accuracy history: {[f'{acc:.2f}%' for acc in checkpoint['test_accuracies']]}")
    print(f"Model state loaded successfully")
    
    return epoch, best_acc, checkpoint

def clear_memory():
    """Clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def save_seed_info(seed):
    """Save seed information for reproducibility"""
    seed_info = {
        'seed': seed,
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/seed_info.json', 'w') as f:
        json.dump(seed_info, f, indent=4)

def save_model_summary(model, batch_size):
    """Save model summary to file"""
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/model_summary.txt', 'w') as f:
        f.write(str(summary(model, input_size=(batch_size, 3, 224, 224))))

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, learning_rates):
    """Plot training metrics"""
    os.makedirs('plots', exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', marker='o', markersize=3)
    ax1.plot(test_losses, label='Test Loss', marker='o', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy', marker='o', markersize=3)
    ax2.plot(test_accuracies, label='Test Accuracy', marker='o', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Plot learning rate
    ax3.plot(learning_rates, label='Learning Rate', marker='o', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('plots/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Clear memory at start
    clear_memory()
    
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    save_seed_info(seed)
    
    # Initialize model
    print("\nInitializing EfficientNet-B0 model...")
    model = EfficientNetModel()
    print(f"Model initialized on device: {model.device}")

    # Load checkpoint from phase 1
    print("\nLoading checkpoint from phase 1...")
    start_epoch, start_acc, checkpoint = load_checkpoint(model, 'checkpoints/model_epoch_30.pth')
    print(f"\nStarting phase 2 training from epoch {start_epoch} with best accuracy {start_acc:.2f}%")

    # Initialize dataset
    print("\nLoading CIFAR-100 dataset...")
    dataset = CIFAR100Dataset(batch_size=128, num_workers=4)
    train_loader, test_loader = dataset.get_data_loaders()

    # Save model summary
    save_model_summary(model.get_model(), batch_size=128)

    # Initialize trainer for phase 2
    print("\nSetting up trainer for phase 2...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=0.01,  # Lower learning rate for fine-tuning
        gradient_accumulation_steps=4,
        use_mixup=True,
        checkpoint=checkpoint  # Pass checkpoint to restore states
    )

    # Print training configuration
    print("\nPhase 2 Training Configuration:")
    print(f"Optimizer: SGD with momentum=0.9, weight_decay=5e-4, nesterov=True")
    print(f"Learning Rate: 0.01 with CosineAnnealingLR (T_max=20, eta_min=1e-6)")
    print(f"Current Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
    print(f"Mixup: Enabled (alpha=0.1)")
    print(f"EMA: Enabled (decay=0.999)")
    print(f"Gradient Accumulation: 4 steps")
    print(f"Mixed Precision: Enabled")
    print(f"Early Stopping: 5 epochs patience")
    print(f"Checkpoint Saving: Every epoch")
    print(f"Metrics Logging: CSV and plots")

    # Train the model
    print("\nStarting phase 2 training...")
    print(f"Training on device: {model.device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    best_accuracy = trainer.train(num_epochs=20, start_epoch=start_epoch)
    print(f"\nPhase 2 training completed! Best accuracy: {best_accuracy:.2f}%")
    
    # Final memory cleanup
    clear_memory()

if __name__ == "__main__":
    main() 