import os
from model import EfficientNetModel
from dataset import CIFAR100Dataset
from train import Trainer

def main():
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Initialize model
    print("Initializing EfficientNet-B0 model...")
    model = EfficientNetModel()
    
    # Initialize dataset
    print("Loading CIFAR-100 dataset...")
    dataset = CIFAR100Dataset(batch_size=32, num_workers=4)
    train_loader, test_loader = dataset.get_data_loaders()
    
    # Initialize trainer
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=0.001
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train(num_epochs=10, save_best=True)
    
    # Plot training metrics
    print("\nGenerating training plots...")
    trainer.plot_metrics()
    
    print("\nTraining completed! Check the following:")
    print("- Training metrics plot: training_metrics.png")
    print("- Best model checkpoint: checkpoints/best_model.pth")
    print("- Tensorboard logs: runs/efficientnet_cifar100/")
    print("\nTo view Tensorboard logs, run:")
    print("tensorboard --logdir=runs/efficientnet_cifar100")

if __name__ == "__main__":
    main() 