# EfficientNet-B0 on CIFAR-100 with Apple Silicon MPS

This project implements fine-tuning of EfficientNet-B0 on the CIFAR-100 dataset using PyTorch with Apple Silicon MPS (Metal Performance Shaders) acceleration. The implementation includes a complete training pipeline with monitoring, visualization, and checkpointing capabilities.

## Features

- EfficientNet-B0 model pretrained on ImageNet
- Fine-tuning on CIFAR-100 dataset
- Apple Silicon MPS (GPU) acceleration support
- Training metrics visualization
- Tensorboard integration
- Model checkpointing
- Progress bars and detailed logging

## Project Structure

```
efficientnet-cifar100/
├── main.py          # Main training pipeline
├── model.py         # EfficientNet model wrapper
├── dataset.py       # CIFAR-100 dataset loader
├── train.py         # Training loop implementation
├── README.md        # This file
├── requirements.txt # Project dependencies
└── checkpoints/     # Saved model checkpoints
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- macOS 13.0+ with Apple Silicon (M1/M2)
- Other dependencies listed in requirements.txt

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the training pipeline:
```bash
python main.py
```

2. Monitor training progress:
- Watch the console output for epoch-wise progress
- View Tensorboard logs:
```bash
tensorboard --logdir=runs/efficientnet_cifar100
```

3. Check results:
- Training metrics plot: `training_metrics.png`
- Best model checkpoint: `checkpoints/best_model.pth`
- Tensorboard logs: `runs/efficientnet_cifar100/`

## Model Details

### EfficientNet-B0
- Architecture: EfficientNet-B0 (pretrained on ImageNet)
- Input size: 224x224
- Number of classes: 100 (CIFAR-100)
- Optimizer: Adam (learning rate: 0.001)
- Loss function: CrossEntropyLoss

### Data Augmentation
- Random horizontal flip
- Random rotation (±10 degrees)
- Color jittering
- Normalization (ImageNet stats)

## Performance

### Expected Results
- Training time: ~2-3x faster on MPS vs CPU
- Target accuracy: 60%+ top-1 accuracy on CIFAR-100
- Memory usage: ~1-2GB GPU memory

### MPS Acceleration
- Automatically detects and uses Apple Silicon GPU
- Falls back to CPU if MPS is not available
- Provides significant speedup for training and inference

## Monitoring and Visualization

1. Real-time metrics:
   - Training loss
   - Validation loss
   - Training accuracy
   - Validation accuracy

2. Visualization tools:
   - Matplotlib plots for loss and accuracy
   - Tensorboard for detailed metrics
   - Progress bars for training and evaluation

## Checkpointing

- Saves best model based on validation accuracy
- Stores training state (epoch, optimizer, loss, accuracy)
- Checkpoints saved in `checkpoints/` directory

## Troubleshooting

1. MPS not available:
   - Ensure macOS 13.0+ is installed
   - Check PyTorch version (2.0+ required)
   - Verify Apple Silicon GPU is present

2. Memory issues:
   - Reduce batch size in dataset.py
   - Close other GPU-intensive applications
   - Monitor system memory usage

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 