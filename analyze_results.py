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

class ModelAnalyzer:
    def __init__(self, model_path='checkpoints/best_model.pth', logs_dir='runs/efficientnet_cifar100'):
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
        
        print(f"Setting up directories...")
        self.logs_dir = Path(logs_dir)
        self.output_dir = Path('analysis_results')
        self.output_dir.mkdir(exist_ok=True)
        print(f"Results will be saved in: {self.output_dir}/")
        
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
            event_acc = EventAccumulator(str(self.logs_dir))
            event_acc.Reload()
            
            # Extract scalars
            print("Extracting metrics from TensorBoard logs...")
            train_acc = pd.DataFrame(event_acc.Scalars('Accuracy/train'))
            test_acc = pd.DataFrame(event_acc.Scalars('Accuracy/test'))
            train_loss = pd.DataFrame(event_acc.Scalars('Loss/train'))
            test_loss = pd.DataFrame(event_acc.Scalars('Loss/test'))
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            ax1.plot(train_acc.step, train_acc.value, label='Train')
            ax1.plot(test_acc.step, test_acc.value, label='Validation')
            ax1.set_title('Training vs Validation Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot loss
            ax2.plot(train_loss.step, train_loss.value, label='Train')
            ax2.plot(test_loss.step, test_loss.value, label='Validation')
            ax2.set_title('Training vs Validation Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            save_path = self.output_dir / 'training_history.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training history plot saved to: {save_path}")
            
        except Exception as e:
            print(f"Error plotting training history: {e}")
    
    def compute_confusion_matrix(self):
        """Compute and plot confusion matrix"""
        print("\nComputing confusion matrix...")
        all_preds = []
        all_labels = []
        
        print("Running inference on test set...")
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Processing test set"):
                images = images.to(self.device)
                outputs = self.model.get_model()(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Compute confusion matrix
        print("Generating confusion matrices...")
        cm = confusion_matrix(all_labels, all_preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot raw confusion matrix
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.title('Raw Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        save_path = self.output_dir / 'confusion_matrix_raw.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Raw confusion matrix saved to: {save_path}")
        
        # Plot normalized confusion matrix
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        save_path = self.output_dir / 'confusion_matrix_normalized.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Normalized confusion matrix saved to: {save_path}")
        
        return cm, cm_norm
    
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
        save_path = self.output_dir / 'class_accuracies.csv'
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
        save_path = self.output_dir / 'class_accuracies.png'
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
        save_path = self.output_dir / 'confused_pairs.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confused pairs plot saved to: {save_path}")
        
        # Save to CSV
        df = pd.DataFrame({
            'True Class': [self.class_names[i] for i,_,_ in top_10_pairs],
            'Predicted Class': [self.class_names[j] for _,j,_ in top_10_pairs],
            'Count': counts
        })
        save_path = self.output_dir / 'confused_pairs.csv'
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
        save_path = self.output_dir / 'misclassified_samples.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Misclassified samples visualization saved to: {save_path}")
    
    def summarize_model(self):
        """Create summary table with model details"""
        print("\nGenerating model summary...")
        # Count parameters
        total_params = sum(p.numel() for p in self.model.get_model().parameters())
        trainable_params = sum(p.numel() for p in self.model.get_model().parameters() if p.requires_grad)
        
        # Compute accuracy on test set
        correct = 0
        total = 0
        print("Computing test set accuracy...")
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.get_model()(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100 * correct / total
        
        # Create summary DataFrame
        summary = pd.DataFrame({
            'Metric': ['Model Name', 'Total Parameters', 'Trainable Parameters', 'Test Accuracy'],
            'Value': ['EfficientNet-B0', f'{total_params:,}', f'{trainable_params:,}', f'{accuracy:.2f}%']
        })
        
        # Save to CSV
        save_path = self.output_dir / 'model_summary.csv'
        summary.to_csv(save_path, index=False)
        print(f"Model summary saved to: {save_path}")
        
        return summary
    
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
        save_path = self.output_dir / 'accuracies.csv'
        df.to_csv(save_path, index=False)
        print(f"Accuracies saved to: {save_path}")
        
        return top1_accuracy, top5_accuracy

def main():
    """Run all analyses"""
    print("Starting model analysis...")
    analyzer = ModelAnalyzer()
    
    print("\nComputing accuracies...")
    top1_acc, top5_acc = analyzer.compute_accuracies()
    
    print("\n1. Plotting training history...")
    analyzer.plot_training_history()
    
    print("\n2. Computing confusion matrices...")
    analyzer.compute_confusion_matrix()
    
    print("\n3. Computing class-wise accuracies...")
    class_accuracies = analyzer.compute_class_accuracies()
    print("\nTop 5 classes:")
    print(class_accuracies.head())
    print("\nBottom 5 classes:")
    print(class_accuracies.tail())
    
    print("\n4. Finding most confused class pairs...")
    confused_pairs = analyzer.find_most_confused_pairs()
    print("\nTop 5 confused pairs:")
    print(confused_pairs.head())
    
    print("\n5. Visualizing misclassified samples...")
    analyzer.visualize_misclassified()
    
    print("\n6. Generating model summary...")
    summary = analyzer.summarize_model()
    print("\nModel Summary:")
    print(summary.to_string(index=False))
    
    print("\nAnalysis complete! Results saved in 'analysis_results' directory.")

if __name__ == '__main__':
    main() 