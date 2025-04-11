# phase2.py

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
from torch.optim.lr_scheduler import CosineAnnealingLR
import signal
import sys

def clear_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
        self.decay = decay
        self.model = model

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.ema[name])

    def state_dict(self):
        return self.ema

    def load_state_dict(self, state_dict):
        self.ema = state_dict

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, learning_rates):
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(15, 12))
    epochs = list(range(1, len(train_losses)+1))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs, learning_rates, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate Schedule')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/phase2_training_metrics.png')
    plt.close()

class Phase2Trainer:
    def __init__(self, model, train_loader, test_loader, start_epoch=30, total_epochs=20):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = model.device
        self.start_epoch = start_epoch
        self.total_epochs = total_epochs
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.SGD(model.get_model().parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_epochs, eta_min=1e-6)
        self.scaler = GradScaler()
        self.ema = ModelEMA(model.get_model())
        self.train_losses, self.test_losses = [], []
        self.train_accuracies, self.test_accuracies = [], []
        self.learning_rates = []
        self.best_acc = 0
        self.interrupted = False

        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/phase2_training_history.csv', 'w') as f:
            f.write('epoch,train_loss,test_loss,train_acc,test_acc,lr,epoch_time\n')

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        print("\nInterrupted. Saving...")
        self.interrupted = True

    def train_one_epoch(self):
        self.model.get_model().train()
        correct, total, total_loss = 0, 0, 0
        pbar = tqdm(self.train_loader, desc='Training')

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with autocast():
                outputs = self.model.get_model()(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.ema.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix(loss=total_loss/(total/targets.size(0)), acc=100.*correct/total)

        return total_loss/len(self.train_loader), 100.*correct/total

    def evaluate(self):
        self.ema.apply()
        self.model.get_model().eval()
        correct, total, total_loss = 0, 0, 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with autocast():
                    outputs = self.model.get_model()(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                pbar.set_postfix(loss=total_loss/(total/targets.size(0)), acc=100.*correct/total)

        return total_loss/len(self.test_loader), 100.*correct/total

    def save_checkpoint(self, epoch, test_acc, test_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.get_model().state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': test_acc,
            'loss': test_loss
        }, f'checkpoints/phase2_epoch_{epoch+1}.pth')

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.total_epochs):
            if self.interrupted:
                break

            print(f'\nEpoch {epoch+1}/{self.start_epoch + self.total_epochs}')
            start_time = time.time()

            train_loss, train_acc = self.train_one_epoch()
            test_loss, test_acc = self.evaluate()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            self.learning_rates.append(current_lr)

            with open('metrics/phase2_training_history.csv', 'a') as f:
                f.write(f'{epoch+1},{train_loss:.4f},{test_loss:.4f},{train_acc:.2f},{test_acc:.2f},{current_lr:.6f},{time.time() - start_time:.2f}\n')

            self.scheduler.step()
            self.save_checkpoint(epoch, test_acc, test_loss)
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                torch.save(self.model.get_model().state_dict(), 'checkpoints/phase2_best_model.pth')
                print(f'New best accuracy: {test_acc:.2f}%')

            plot_metrics(self.train_losses, self.test_losses, self.train_accuracies, self.test_accuracies, self.learning_rates)

        print(f'\nTraining complete. Best accuracy: {self.best_acc:.2f}%')

def main():
    clear_memory()
    set_seed(42)
    model = EfficientNetModel()
    dataset = CIFAR100Dataset(batch_size=128, num_workers=4)
    train_loader, test_loader = dataset.get_data_loaders()
    summary(model.get_model(), input_size=(128, 3, 224, 224))

    checkpoint_path = 'checkpoints/model_epoch_30.pth'
    ckpt = torch.load(checkpoint_path)
    model.get_model().load_state_dict(ckpt['model_state_dict'])
    print(f"Resuming from epoch {ckpt['epoch']} with test accuracy {ckpt['accuracy']:.2f}%")

    trainer = Phase2Trainer(model, train_loader, test_loader, start_epoch=ckpt['epoch']+1)
    trainer.train()
    clear_memory()

if __name__ == '__main__':
    main()
