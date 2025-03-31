# EfficientNet CIFAR-100 Classification

This project implements EfficientNet-B0 for CIFAR-100 image classification using PyTorch. It includes a complete training pipeline with comprehensive analysis tools, monitoring, and visualization capabilities.

## Features

- EfficientNet-B0 model (pretrained on ImageNet)
- CIFAR-100 dataset training pipeline
- Automatic hardware acceleration (CUDA/CPU)
- Mixed precision training
- Comprehensive analysis tools:
  - Training/validation metrics visualization
  - Confusion matrices
  - Per-class accuracy analysis
  - Most confused class pairs
  - Misclassified sample visualization
  - Model performance summary
- TensorBoard integration
- Checkpointing system
- System resource monitoring

## Project Structure

```
.
├── main.py              # Training pipeline entry point
├── model.py             # EfficientNet model implementation
├── dataset.py           # CIFAR-100 dataset handling
├── train.py             # Training loop and metrics
├── analyze_results.py   # Results analysis and visualization
├── monitor.py           # System resource monitoring
├── requirements.txt     # Project dependencies
└── README.md           # Documentation
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Required packages listed in `requirements.txt`

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
venv\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Start training:
```bash
python main.py
```

2. Monitor progress:
- Console output shows epoch-wise metrics
- View live training curves:
```bash
tensorboard --logdir=runs/efficientnet_cifar100
```

### Analysis

After training, analyze results:
```bash
python analyze_results.py
```

This generates comprehensive visualizations and metrics in the `analysis_results` directory:
- Training/validation curves (`training_history.png`)
- Confusion matrices (`confusion_matrix_raw.png`, `confusion_matrix_normalized.png`)
- Class-wise performance (`class_accuracies.csv`, `class_accuracies.png`)
- Most confused class pairs (`confused_pairs.csv`, `confused_pairs.png`)
- Misclassified examples (`misclassified_samples.png`)
- Model summary (`model_summary.csv`)

## Model Architecture

### EfficientNet-B0
- Base architecture: EfficientNet-B0
- Input size: 224x224
- Output classes: 100 (CIFAR-100)
- Parameters: ~4.1M
- Target accuracy: 84.4% (top-1)

### Training Configuration
- Optimizer: Adam
- Learning rate: 0.001
- Scheduler: ReduceLROnPlateau
- Loss function: CrossEntropyLoss
- Batch size: 128
- Mixed precision training enabled

### Data Augmentation
- Random horizontal flip
- Random rotation
- Color jittering
- Normalization

## Performance Monitoring

The training pipeline includes comprehensive monitoring:
- Training/validation metrics
- System resource usage (CPU, RAM, GPU if available)
- Training time per epoch
- Learning rate scheduling
- Memory management

## Checkpointing

- Best model saved based on validation accuracy
- Checkpoint contains:
  - Model state
  - Optimizer state
  - Training metrics
  - Current epoch

## Troubleshooting

1. Memory issues:
   - Reduce batch size in `dataset.py`
   - Enable gradient accumulation
   - Monitor system resources with `monitor.py`

2. Training issues:
   - Check learning rate scheduling
   - Verify data augmentation pipeline
   - Monitor loss curves for instability

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 