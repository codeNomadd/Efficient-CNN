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
import matplotlib.pyplot as plt
import signal
import sys
from model import EfficientNetModel, CIFAR100Dataset, set_seed

# === CONFIGURATION ===
SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 4
PHASE1_EPOCHS = 30
PHASE2_EPOCHS = 20
PHASE2_LR = 0.01
GRAD_ACCUM_STEPS = 4
EARLY_STOPPING_PATIENCE = 5
EXP_DIR = 'experiments/combined_experiment'

# === UTILITY FUNCTIONS ===
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def save_seed_info(seed):
    info = {
        'seed': seed,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    os.makedirs(f'{EXP_DIR}/checkpoints', exist_ok=True)
    with open(f'{EXP_DIR}/checkpoints/seed_info.json', 'w') as f:
        json.dump(info, f, indent=4)

def save_model_summary(model):
    os.makedirs(f'{EXP_DIR}/logs', exist_ok=True)
    with open(f'{EXP_DIR}/logs/model_summary.txt', 'w') as f:
        f.write(str(summary(model, input_size=(BATCH_SIZE, 3, 224, 224), verbose=0)))

def plot_metrics(train_losses, test_losses, train_accs, test_accs, lrs):
    os.makedirs(f'{EXP_DIR}/plots', exist_ok=True)
    epochs = list(range(1, len(train_losses)+1))

    plt.figure(figsize=(15, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{EXP_DIR}/plots/loss_curve.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, test_accs, label='Test Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f'{EXP_DIR}/plots/accuracy_curve.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.plot(epochs, lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.savefig(f'{EXP_DIR}/plots/lr_schedule.png')
    plt.close()

class Trainer:
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.device = model.device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.phase = 1
        self.current_epoch = 0
        self.interrupted = False

        self.optimizer = optim.AdamW(model.get_model().parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scaler = GradScaler()

        self.train_losses, self.test_losses = [], []
        self.train_accs, self.test_accs, self.lrs = [], [], []
        self.best_acc = 0
        self.early_stop_counter = 0

        os.makedirs(f'{EXP_DIR}/checkpoints', exist_ok=True)
        with open(f'{EXP_DIR}/metrics/training_history.csv', 'w') as f:
            f.write('epoch,train_loss,test_loss,train_acc,test_acc,lr,epoch_time\n')

        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        print('\nTraining interrupted. Saving...')
        self.interrupted = True

    def adjust_for_phase2(self):
        self.phase = 2
        self.optimizer = optim.SGD(self.model.get_model().parameters(), lr=PHASE2_LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)
        print("\n--- Switched to Phase 2 (SGD) ---\n")

    def save_checkpoint(self, epoch, test_loss, test_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.get_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': test_loss,
            'accuracy': test_acc
        }
        torch.save(checkpoint, f'{EXP_DIR}/checkpoints/epoch_{epoch+1}.pth')
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            torch.save(self.model.get_model().state_dict(), f'{EXP_DIR}/checkpoints/best_model.pth')
            print(f'New best accuracy: {test_acc:.2f}%')
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

    def train_epoch(self):
        self.model.get_model().train()
        correct = total = total_loss = 0
        pbar = tqdm(self.train_loader)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with autocast():
                outputs = self.model.get_model()(inputs)
                loss = self.criterion(outputs, targets) / GRAD_ACCUM_STEPS

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.get_model().parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item() * GRAD_ACCUM_STEPS

            pbar.set_postfix(loss=total_loss/(total/targets.size(0)), acc=100.*correct/total)

        return total_loss/len(self.train_loader), 100.*correct/total

    def test(self):
        self.model.get_model().eval()
        total_loss = correct = total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with autocast():
                    outputs = self.model.get_model()(inputs)
                    loss = self.criterion(outputs, targets)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                total_loss += loss.item()

        return total_loss/len(self.test_loader), 100.*correct/total

    def train(self):
        total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS
        for epoch in range(total_epochs):
            if self.interrupted:
                break

            if epoch == PHASE1_EPOCHS:
                self.adjust_for_phase2()

            print(f"\nEpoch {epoch+1}/{total_epochs}")
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.test()
            lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            self.lrs.append(lr)

            with open(f'{EXP_DIR}/metrics/training_history.csv', 'a') as f:
                f.write(f'{epoch+1},{train_loss:.4f},{test_loss:.4f},{train_acc:.2f},{test_acc:.2f},{lr:.6f},{time.time()-start_time:.2f}\n')

            self.save_checkpoint(epoch, test_loss, test_acc)
            plot_metrics(self.train_losses, self.test_losses, self.train_accs, self.test_accs, self.lrs)

            if self.phase == 2 and self.early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print("\nEarly stopping triggered.")
                break

        print(f"\nTraining finished. Best Accuracy: {self.best_acc:.2f}%")

# === MAIN ENTRY ===
def main():
    clear_memory()
    set_seed(SEED)
    save_seed_info(SEED)

    model = EfficientNetModel()
    dataset = CIFAR100Dataset(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_loader, test_loader = dataset.get_data_loaders()

    save_model_summary(model.get_model())

    trainer = Trainer(model, train_loader, test_loader)
    trainer.train()
    clear_memory()

if __name__ == '__main__':
    main()
