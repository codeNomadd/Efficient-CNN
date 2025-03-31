import torch
import torch.nn as nn
import torchvision.models as models
import os

def get_device():
    """Get the best available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device

class EfficientNetModel:
    def __init__(self, num_classes=100):
        """Initialize EfficientNet-B0 model with CIFAR-100 classes"""
        self.device = get_device()
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Modify the classifier for CIFAR-100
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        print(f"Model is on device: {next(self.model.parameters()).device}")
        
    def get_model(self):
        """Return the model"""
        return self.model
    
    def save_checkpoint(self, path, epoch, optimizer, loss, accuracy):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
        }, path)
        
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint 