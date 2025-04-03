import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import torchvision
from model import EfficientNetModel
from dataset import CIFAR100Dataset
from tqdm import tqdm
import datetime
import json
import shutil
from thop import profile

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
    def __init__(self, model_path='checkpoints/best_model.pth', logs_dir='runs/efficientnet_cifar100', run_name=None):
        """Initialize the analyzer with model and data"""
        print("\nInitializing ModelAnalyzer...")
        print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        
        print("Loading CIFAR-100 dataset...")
        self.dataset = CIFAR100Dataset(batch_size=256, num_workers=4)
        _, self.test_loader = self.dataset.get_data_loaders()
        self.class_names = self.dataset.classes
        
        # Initialize run manager
        self.run_manager = RunManager()
        self.run_dir, self.run_info = self.run_manager.create_new_run(run_name)
        print(f"Results will be saved in: {self.run_dir}/")
        
    def _load_model(self, model_path):
        """Load the trained model"""
        model = EfficientNetModel()
        checkpoint = torch.load(model_path, map_location=self.device)
        model.get_model().load_state_dict(checkpoint['model_state_dict'])
        model.get_model().eval()
        return model
    
    def plot_training_history(self):
        """Plot training vs validation accuracy/loss from TensorBoard logs"""
        print("\nPlotting training history...")
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            event_acc = EventAccumulator(str(self.run_dir))
            event_acc.Reload()
            
            # Extract scalars
            print("Extracting metrics from TensorBoard logs...")
            train_acc = pd.DataFrame(event_acc.Scalars('Accuracy/train'))
            test_acc = pd.DataFrame(event_acc.Scalars('Accuracy/test'))
            train_loss = pd.DataFrame(event_acc.Scalars('Loss/train'))
            test_loss = pd.DataFrame(event_acc.Scalars('Loss/test'))
            lr = pd.DataFrame(event_acc.Scalars('Learning_rate'))
            
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
            
            # Plot accuracy
            ax1.plot(train_acc.step, train_acc.value, label='Train', linewidth=2, marker='o')
            ax1.plot(test_acc.step, test_acc.value, label='Validation', linewidth=2, marker='o')
            ax1.set_title('Training vs Validation Accuracy', fontsize=14, pad=20)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Accuracy (%)', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_ylim(50, 100)  # Set y-axis limits for accuracy
            ax1.set_yticks(np.arange(50, 101, 5))  # Set y-axis ticks every 5%
            
            # Plot loss
            ax2.plot(train_loss.step, train_loss.value, label='Train', linewidth=2, marker='o')
            ax2.plot(test_loss.step, test_loss.value, label='Validation', linewidth=2, marker='o')
            ax2.set_title('Training vs Validation Loss', fontsize=14, pad=20)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Plot learning rate
            ax3.plot(lr.step, lr.value, label='Learning Rate', linewidth=2, marker='o', color='green')
            ax3.set_title('Learning Rate over Epochs', fontsize=14, pad=20)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Learning Rate', fontsize=12)
            ax3.legend(fontsize=10)
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.set_yscale('log')  # Use log scale for learning rate
            
            # Add value labels on points for all plots
            for ax, train_data, test_data in [(ax1, train_acc, test_acc), (ax2, train_loss, test_loss)]:
                for i, (train_val, test_val) in enumerate(zip(train_data.value, test_data.value)):
                    ax.text(i, train_val, f'{train_val:.2f}', ha='center', va='bottom')
                    ax.text(i, test_val, f'{test_val:.2f}', ha='center', va='top')
            
            # Add value labels for learning rate
            for i, lr_val in enumerate(lr.value):
                ax3.text(i, lr_val, f'{lr_val:.6f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save the figure
            save_path = self.run_dir / 'training_history.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training history plot saved to: {save_path}")
            
        except Exception as e:
            print(f"Error plotting training history: {e}")
    
    def compute_confusion_matrix(self):
        """Compute and visualize confusion matrix"""
        print("\nComputing confusion matrix...")
        try:
            # Initialize confusion matrix
            cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
            
            # Compute confusion matrix
            self.model.eval()
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        cm[t.long(), p.long()] += 1
            
            # Compute binary classification metrics for each class
            metrics = []
            for i in range(self.num_classes):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - (tp + fp + fn)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics.append({
                    'Class': i,
                    'True Positives': tp,
                    'False Positives': fp,
                    'False Negatives': fn,
                    'True Negatives': tn,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                })
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame(metrics)
            save_path = self.run_dir / 'confusion_matrix_metrics.csv'
            metrics_df.to_csv(save_path, index=False)
            print(f"Confusion matrix metrics saved to: {save_path}")
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            # Save the figure
            save_path = self.run_dir / 'confusion_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix plot saved to: {save_path}")
            
            return cm
            
        except Exception as e:
            print(f"Error computing confusion matrix: {e}")
            return None
    
    def compute_class_accuracies(self):
        """Compute per-class accuracies"""
        print("\nComputing class-wise accuracies...")
        cm, _ = self.compute_confusion_matrix()
        class_correct = np.diag(cm)
        class_total = np.sum(cm, axis=1)
        class_accuracies = class_correct / class_total
        
        # Create DataFrame with class accuracies
        df = pd.DataFrame({
            'Class': self.class_names,
            'Accuracy': class_accuracies * 100,
            'Correct': class_correct,
            'Total': class_total
        })
        df = df.sort_values('Accuracy', ascending=False)
        
        # Save to CSV
        save_path = self.run_dir / 'class_accuracies.csv'
        df.to_csv(save_path, index=False)
        print(f"Class accuracies saved to: {save_path}")
        
        # Plot top and bottom 10 classes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top 10
        sns.barplot(data=df.head(10), x='Class', y='Accuracy', ax=ax1)
        ax1.set_title('Top 10 Classes by Accuracy')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Bottom 10
        sns.barplot(data=df.tail(10), x='Class', y='Accuracy', ax=ax2)
        ax2.set_title('Bottom 10 Classes by Accuracy')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = self.run_dir / 'class_accuracies.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Class accuracies plot saved to: {save_path}")
        
        return df
    
    def find_most_confused_pairs(self):
        """Find and visualize top-10 most confused class pairs"""
        print("\nFinding most confused class pairs...")
        cm, _ = self.compute_confusion_matrix()
        np.fill_diagonal(cm, 0)  # Remove diagonal elements
        
        # Find top confused pairs
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j:
                    confused_pairs.append((i, j, cm[i,j]))
        
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        top_10_pairs = confused_pairs[:10]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        pair_names = [f"{self.class_names[i]} → {self.class_names[j]}" for i,j,_ in top_10_pairs]
        counts = [count for _,_,count in top_10_pairs]
        
        sns.barplot(x=counts, y=pair_names)
        plt.title('Top 10 Most Confused Class Pairs')
        plt.xlabel('Number of Confusions')
        
        plt.tight_layout()
        save_path = self.run_dir / 'confused_pairs.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confused pairs plot saved to: {save_path}")
        
        # Save to CSV
        df = pd.DataFrame({
            'True Class': [self.class_names[i] for i,_,_ in top_10_pairs],
            'Predicted Class': [self.class_names[j] for _,j,_ in top_10_pairs],
            'Count': counts
        })
        save_path = self.run_dir / 'confused_pairs.csv'
        df.to_csv(save_path, index=False)
        print(f"Confused pairs data saved to: {save_path}")
        
        return df
    
    def visualize_misclassified(self, num_samples=10):
        """Visualize sample misclassified images"""
        print("\nFinding and visualizing misclassified samples...")
        misclassified_images = []
        misclassified_labels = []
        misclassified_preds = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.get_model()(images)
                _, predicted = outputs.max(1)
                
                # Find misclassified samples
                mask = (predicted != labels)
                misclassified_images.extend(images[mask].cpu())
                misclassified_labels.extend(labels[mask].cpu())
                misclassified_preds.extend(predicted[mask].cpu())
                
                if len(misclassified_images) >= num_samples:
                    break
        
        print(f"Found {len(misclassified_images)} misclassified samples")
        
        # Plot samples
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()
        
        for idx in range(min(num_samples, len(misclassified_images))):
            img = misclassified_images[idx]
            true_label = self.class_names[misclassified_labels[idx]]
            pred_label = self.class_names[misclassified_preds[idx]]
            
            # Denormalize image
            img = img / 2 + 0.5  # reverse normalization
            img = img.numpy().transpose(1, 2, 0)
            
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=8)
        
        plt.tight_layout()
        save_path = self.run_dir / 'misclassified_samples.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Misclassified samples visualization saved to: {save_path}")
    
    def generate_model_summary(self):
        """Generate a summary of the model architecture and parameters"""
        print("\nGenerating model summary...")
        try:
            # Get model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Calculate FLOPs
            input_tensor = torch.randn(1, 3, 224, 224).to(self.device)
            flops, _ = profile(self.model, inputs=(input_tensor,))
            
            # Create summary dictionary
            summary = {
                'Model': self.model.__class__.__name__,
                'Total Parameters': f"{total_params:,}",
                'Trainable Parameters': f"{trainable_params:,}",
                'FLOPs': f"{flops:,}",
                'Input Shape': '(1, 3, 224, 224)',
                'Output Shape': f'(1, {self.num_classes})',
                'Device': str(self.device)
            }
            
            # Save to CSV
            df = pd.DataFrame([summary])
            save_path = self.run_dir / 'model_summary.csv'
            df.to_csv(save_path, index=False)
            print(f"Model summary saved to: {save_path}")
            
            # Print summary
            print("\nModel Summary:")
            for key, value in summary.items():
                print(f"{key}: {value}")
            
        except Exception as e:
            print(f"Error generating model summary: {e}")
    
    def compute_accuracies(self):
        """Compute top-1 and top-5 accuracies on test set"""
        print("\nComputing top-1 and top-5 accuracies...")
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Computing accuracies"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.get_model()(images)
                
                # Top-1 accuracy
                _, predicted = outputs.max(1)
                correct_top1 += predicted.eq(labels).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, True, True)
                correct_top5 += top5_pred.eq(labels.view(-1, 1)).sum().item()
                
                total += labels.size(0)
        
        top1_accuracy = 100. * correct_top1 / total
        top5_accuracy = 100. * correct_top5 / total
        
        print(f"\nTest Set Performance:")
        print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
        
        # Save to CSV
        df = pd.DataFrame({
            'Metric': ['Top-1 Accuracy', 'Top-5 Accuracy'],
            'Value': [f"{top1_accuracy:.2f}%", f"{top5_accuracy:.2f}%"]
        })
        save_path = self.run_dir / 'accuracies.csv'
        df.to_csv(save_path, index=False)
        print(f"Accuracies saved to: {save_path}")
        
        # Update run metrics
        self.run_manager.update_run_metrics(self.run_info['name'], {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy
        })
        
        return top1_accuracy, top5_accuracy
    
    def visualize_model_architecture(self):
        """Create a visualization of the model architecture"""
        print("\nGenerating model architecture visualization...")
        
        # Get model structure
        model = self.model.get_model()
        
        # Create architecture summary
        architecture_data = []
        total_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.Dropout)):
                params = sum(p.numel() for p in module.parameters())
                total_params += params
                
                # Format layer name
                layer_name = name.replace('_', ' ').title()
                
                # Get layer details
                if isinstance(module, nn.Conv2d):
                    details = f"Conv2d({module.in_channels}, {module.out_channels}, kernel={module.kernel_size})"
                elif isinstance(module, nn.Linear):
                    details = f"Linear({module.in_features}, {module.out_features})"
                elif isinstance(module, nn.BatchNorm2d):
                    details = f"BatchNorm2d({module.num_features})"
                elif isinstance(module, nn.Dropout):
                    details = f"Dropout(p={module.p})"
                
                architecture_data.append({
                    'Layer': layer_name,
                    'Type': module.__class__.__name__,
                    'Details': details,
                    'Parameters': f"{params:,}"
                })
        
        # Create DataFrame
        df = pd.DataFrame(architecture_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, len(architecture_data) * 0.5))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='left',
            loc='center',
            colWidths=[0.2, 0.2, 0.4, 0.2]
        )
        
        # Adjust table style
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title('EfficientNet-B0 Architecture', pad=20, fontsize=14)
        
        # Add total parameters
        plt.figtext(0.5, 0.02, f'Total Parameters: {total_params:,}', 
                   ha='center', fontsize=10)
        
        # Save the figure
        save_path = self.run_dir / 'model_architecture.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model architecture visualization saved to: {save_path}")
        
        return df

    def compute_confusion_matrix_metrics(self):
        """Compute and visualize confusion matrix metrics"""
        print("\nComputing confusion matrix metrics...")
        cm, _ = self.compute_confusion_matrix()
        
        # Calculate metrics
        total = np.sum(cm)
        true_positives = np.diag(cm)
        false_positives = np.sum(cm, axis=0) - true_positives
        false_negatives = np.sum(cm, axis=1) - true_positives
        true_negatives = total - true_positives - false_positives - false_negatives
        
        # Calculate accuracies
        accuracy = (true_positives + true_negatives) / total
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Create metrics DataFrame
        metrics_data = []
        for i in range(len(self.class_names)):
            metrics_data.append({
                'Class': self.class_names[i],
                'True Positives': true_positives[i],
                'False Positives': false_positives[i],
                'True Negatives': true_negatives[i],
                'False Negatives': false_negatives[i],
                'Accuracy': f"{accuracy[i]*100:.2f}%",
                'Precision': f"{precision[i]*100:.2f}%",
                'Recall': f"{recall[i]*100:.2f}%",
                'F1 Score': f"{f1_score[i]*100:.2f}%"
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, len(metrics_data) * 0.5))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='left',
            loc='center',
            colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        
        # Adjust table style
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title('Confusion Matrix Metrics by Class', pad=20, fontsize=14)
        
        # Save the figure
        save_path = self.run_dir / 'confusion_matrix_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix metrics visualization saved to: {save_path}")
        
        # Save to CSV
        csv_path = self.run_dir / 'confusion_matrix_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"Confusion matrix metrics saved to: {csv_path}")
        
        return df

def main():
    """Main function to analyze model results"""
    print("Starting model analysis...")
    
    # Initialize analyzer
    analyzer = ModelAnalyzer()
    
    print("\n1. Loading model and data...")
    analyzer.load_model_and_data()
    
    print("\n2. Computing accuracies...")
    analyzer.compute_accuracies()
    
    print("\n3. Computing confusion matrix...")
    analyzer.compute_confusion_matrix()
    
    print("\n4. Computing class accuracies...")
    analyzer.compute_class_accuracies()
    
    print("\n5. Plotting training history...")
    analyzer.plot_training_history()
    
    print("\n6. Generating model summary...")
    analyzer.generate_model_summary()
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main() 