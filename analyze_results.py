import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import torchvision
from model import EfficientNetModel
from dataset import CIFAR100Dataset
from tqdm import tqdm
import datetime
import json
import shutil
from thop import profile
import os

class RunManager:
    def __init__(self, base_dir='analysis_results'):
        """Initialize the run manager"""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.runs_file = self.base_dir / 'runs_history.json'
        self.load_runs_history()
    
    def load_runs_history(self):
        """Load the history of all runs"""
        if self.runs_file.exists():
            with open(self.runs_file, 'r') as f:
                self.runs_history = json.load(f)
        else:
            self.runs_history = {}
    
    def save_runs_history(self):
        """Save the history of all runs"""
        with open(self.runs_file, 'w') as f:
            json.dump(self.runs_history, f, indent=4)
    
    def create_new_run(self, run_name=None):
        """Create a new run directory with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name is None:
            run_name = f"run_{timestamp}"
        
        run_dir = self.base_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        # Initialize run info
        run_info = {
            'timestamp': timestamp,
            'name': run_name,
            'path': str(run_dir),
            'metrics': {}
        }
        
        # Update history
        self.runs_history[run_name] = run_info
        self.save_runs_history()
        
        return run_dir, run_info
    
    def update_run_metrics(self, run_name, metrics):
        """Update metrics for a specific run"""
        if run_name in self.runs_history:
            self.runs_history[run_name]['metrics'].update(metrics)
            self.save_runs_history()
    
    def get_all_runs(self):
        """Get list of all runs"""
        return list(self.runs_history.keys())
    
    def plot_comparative_metrics(self, metric_name):
        """Plot a specific metric across all runs"""
        runs = []
        values = []
        
        for run_name, run_info in self.runs_history.items():
            if metric_name in run_info['metrics']:
                runs.append(run_name)
                values.append(run_info['metrics'][metric_name])
        
        if runs:
            plt.figure(figsize=(12, 6))
            
            # Plot line chart
            plt.plot(runs, values, marker='o', linewidth=2, markersize=8)
            
            # Customize the plot
            plt.title(f'Comparison of Top-1 Accuracy across runs', fontsize=14, pad=20)
            plt.xlabel('Training Run', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            
            # Set y-axis limits and ticks
            plt.ylim(70, 100)  # Set minimum to 70%
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels on points
            for i, v in enumerate(values):
                plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add horizontal grid lines at every 5%
            plt.yticks(np.arange(70, 101, 5))
            
            plt.tight_layout()
            
            save_path = self.base_dir / f'comparative_{metric_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Comparative plot saved to: {save_path}")

class ModelAnalyzer:
    def __init__(self, model_path, dataset, device='cuda'):
        """Initialize model analyzer"""
        self.device = device
        self.dataset = dataset
        
        # Load model
        if isinstance(model_path, str):
            self.model = EfficientNetModel()
            self.model.load_checkpoint(model_path)
            self.model = self.model.get_model().to(device)
        else:
            self.model = model_path.to(device)
        
        # Get data loaders
        self.train_loader, self.test_loader = dataset.get_data_loaders()
        
        # Initialize metrics
        self.metrics = {}
        
    def plot_training_history(self, run_dir):
        """Plot training history from TensorBoard logs"""
        try:
            # Load metrics from CSV
            metrics_file = os.path.join(run_dir, 'metrics', 'training_history.csv')
            if not os.path.exists(metrics_file):
                print(f"Metrics file not found: {metrics_file}")
                return
            
            metrics_df = pd.read_csv(metrics_file)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Training History', fontsize=16)
            
            # Plot accuracy
            ax = axes[0, 0]
            ax.plot(metrics_df['Epoch'], metrics_df['Train Accuracy'], label='Train', marker='o', markersize=3)
            ax.plot(metrics_df['Epoch'], metrics_df['Test Accuracy'], label='Test', marker='o', markersize=3)
            ax.set_title('Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True)
            ax.legend()
            
            # Add value labels at the end of lines
            last_epoch = metrics_df['Epoch'].iloc[-1]
            ax.text(last_epoch, metrics_df['Train Accuracy'].iloc[-1], 
                   f'{metrics_df["Train Accuracy"].iloc[-1]:.2f}%', ha='left', va='center')
            ax.text(last_epoch, metrics_df['Test Accuracy'].iloc[-1], 
                   f'{metrics_df["Test Accuracy"].iloc[-1]:.2f}%', ha='left', va='center')
            
            # Plot loss
            ax = axes[0, 1]
            ax.plot(metrics_df['Epoch'], metrics_df['Train Loss'], label='Train', marker='o', markersize=3)
            ax.plot(metrics_df['Epoch'], metrics_df['Test Loss'], label='Test', marker='o', markersize=3)
            ax.set_title('Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True)
            ax.legend()
            
            # Add value labels at the end of lines
            ax.text(last_epoch, metrics_df['Train Loss'].iloc[-1], 
                   f'{metrics_df["Train Loss"].iloc[-1]:.4f}', ha='left', va='center')
            ax.text(last_epoch, metrics_df['Test Loss'].iloc[-1], 
                   f'{metrics_df["Test Loss"].iloc[-1]:.4f}', ha='left', va='center')
            
            # Plot learning rate
            ax = axes[1, 0]
            ax.plot(metrics_df['Epoch'], metrics_df['Learning Rate'], marker='o', markersize=3)
            ax.set_title('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.grid(True)
            
            # Add value label at the end of line
            ax.text(last_epoch, metrics_df['Learning Rate'].iloc[-1], 
                   f'{metrics_df["Learning Rate"].iloc[-1]:.6f}', ha='left', va='center')
            
            # Add summary statistics
            ax = axes[1, 1]
            ax.axis('off')
            stats_text = (
                f'Total Epochs: {len(metrics_df)}\n'
                f'Best Test Accuracy: {metrics_df["Test Accuracy"].max():.2f}%\n'
                f'Final Test Accuracy: {metrics_df["Test Accuracy"].iloc[-1]:.2f}%\n'
                f'Best Test Loss: {metrics_df["Test Loss"].min():.4f}\n'
                f'Final Test Loss: {metrics_df["Test Loss"].iloc[-1]:.4f}\n'
                f'Initial LR: {metrics_df["Learning Rate"].iloc[0]:.6f}\n'
                f'Final LR: {metrics_df["Learning Rate"].iloc[-1]:.6f}'
            )
            ax.text(0.1, 0.5, stats_text, fontsize=12, va='center')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Training history plot saved to: {os.path.join(run_dir, 'training_history.png')}")
            
        except Exception as e:
            print(f"Error plotting training history: {str(e)}")
    
    def compute_confusion_matrix(self, run_dir):
        """Compute and plot confusion matrix"""
        try:
            self.model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in tqdm(self.test_loader, desc='Computing confusion matrix'):
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            # Compute confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            
            # Normalize confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.dataset.classes,
                       yticklabels=self.dataset.classes)
            plt.title('Normalized Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save confusion matrix metrics
            metrics = {
                'class': self.dataset.classes,
                'precision': precision_score(all_labels, all_preds, average=None),
                'recall': recall_score(all_labels, all_preds, average=None),
                'f1': f1_score(all_labels, all_preds, average=None)
            }
            pd.DataFrame(metrics).to_csv(os.path.join(run_dir, 'confusion_matrix_metrics.csv'), index=False)
            
            print(f"Confusion matrix plot saved to: {os.path.join(run_dir, 'confusion_matrix.png')}")
            print(f"Confusion matrix metrics saved to: {os.path.join(run_dir, 'confusion_matrix_metrics.csv')}")
            
        except Exception as e:
            print(f"Error computing confusion matrix: {str(e)}")
    
    def compute_accuracies(self, run_dir):
        """Compute overall and per-class accuracies"""
        try:
            self.model.eval()
            correct = 0
            total = 0
            class_correct = [0] * len(self.dataset.classes)
            class_total = [0] * len(self.dataset.classes)
            
            with torch.no_grad():
                for inputs, labels in tqdm(self.test_loader, desc='Computing accuracies'):
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(self.device)).sum().item()
                    
                    # Compute per-class accuracy
                    for i in range(len(labels)):
                        label = labels[i]
                        class_correct[label] += (predicted[i] == label).item()
                        class_total[label] += 1
            
            # Save overall accuracy
            overall_acc = 100 * correct / total
            pd.DataFrame({
                'metric': ['Overall Accuracy'],
                'value': [overall_acc]
            }).to_csv(os.path.join(run_dir, 'accuracies.csv'), index=False)
            
            # Save per-class accuracy
            class_acc = [100 * class_correct[i] / class_total[i] for i in range(len(self.dataset.classes))]
            pd.DataFrame({
                'class': self.dataset.classes,
                'accuracy': class_acc
            }).to_csv(os.path.join(run_dir, 'class_accuracies.csv'), index=False)
            
            # Plot per-class accuracy
            plt.figure(figsize=(15, 8))
            plt.bar(range(len(self.dataset.classes)), class_acc)
            plt.xticks(range(len(self.dataset.classes)), self.dataset.classes, rotation=45, ha='right')
            plt.title('Per-Class Accuracy')
            plt.xlabel('Class')
            plt.ylabel('Accuracy (%)')
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'class_accuracies.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Overall accuracy: {overall_acc:.2f}%")
            print(f"Accuracies saved to: {os.path.join(run_dir, 'accuracies.csv')}")
            print(f"Per-class accuracies saved to: {os.path.join(run_dir, 'class_accuracies.csv')}")
            print(f"Per-class accuracy plot saved to: {os.path.join(run_dir, 'class_accuracies.png')}")
            
        except Exception as e:
            print(f"Error computing accuracies: {str(e)}")
    
    def generate_model_summary(self, run_dir):
        """Generate model summary with parameters and FLOPs"""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Calculate FLOPs
            input_tensor = torch.randn(1, 3, 32, 32).to(self.device)
            flops, params = thop.profile(self.model, inputs=(input_tensor,))
            
            # Save summary
            summary = pd.DataFrame({
                'metric': ['Total Parameters', 'Trainable Parameters', 'FLOPs'],
                'value': [total_params, trainable_params, flops]
            })
            summary.to_csv(os.path.join(run_dir, 'model_summary.csv'), index=False)
            
            print(f"Model summary saved to: {os.path.join(run_dir, 'model_summary.csv')}")
            
        except Exception as e:
            print(f"Error generating model summary: {str(e)}")
    
    def analyze(self, run_dir):
        """Run all analysis steps"""
        try:
            os.makedirs(run_dir, exist_ok=True)
            
            print("\nAnalyzing model...")
            self.plot_training_history(run_dir)
            self.compute_confusion_matrix(run_dir)
            self.compute_accuracies(run_dir)
            self.generate_model_summary(run_dir)
            
            print("\nAnalysis completed successfully!")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")

def main():
    """Main function to run the analysis"""
    try:
        # Initialize the analyzer
        analyzer = ModelAnalyzer()
        
        # Load model and data
        if not analyzer.load_model_and_data():
            print("Failed to load model and data. Exiting.")
            return
            
        # Generate model summary with FLOPs
        analyzer.generate_model_summary()
        
        # Plot training history
        analyzer.plot_training_history()
        
        # Compute accuracies
        analyzer.compute_accuracies()
        
        # Compute class accuracies
        analyzer.compute_class_accuracies()
        
        # Compute confusion matrix
        analyzer.compute_confusion_matrix()
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 