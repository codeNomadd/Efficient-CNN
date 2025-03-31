import psutil
import torch
import time
from datetime import datetime
import os

class SystemMonitor:
    def __init__(self):
        """Initialize system monitor"""
        self.metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'gpu_memory_used': [],
            'gpu_memory_total': [],
            'timestamp': []
        }
        
    def update_metrics(self):
        """Update system metrics"""
        # CPU and RAM usage
        self.metrics['cpu_percent'].append(psutil.cpu_percent())
        self.metrics['memory_percent'].append(psutil.virtual_memory().percent)
        
        # GPU metrics if available
        if torch.cuda.is_available():
            self.metrics['gpu_memory_used'].append(torch.cuda.memory_allocated() / 1024**3)  # GB
            self.metrics['gpu_memory_total'].append(torch.cuda.get_device_properties(0).total_memory / 1024**3)  # GB
        else:
            self.metrics['gpu_memory_used'].append(0)
            self.metrics['gpu_memory_total'].append(0)
            
        # Timestamp
        self.metrics['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
    def print_metrics(self):
        """Print current system metrics"""
        print("\nSystem Metrics:")
        print(f"CPU Usage: {self.metrics['cpu_percent'][-1]:.1f}%")
        print(f"Memory Usage: {self.metrics['memory_percent'][-1]:.1f}%")
        
        if torch.cuda.is_available():
            print(f"GPU Memory Used: {self.metrics['gpu_memory_used'][-1]:.2f} GB")
            print(f"GPU Memory Total: {self.metrics['gpu_memory_total'][-1]:.2f} GB")
            print(f"GPU Memory Usage: {(self.metrics['gpu_memory_used'][-1] / self.metrics['gpu_memory_total'][-1] * 100):.1f}%")
        
    def save_metrics(self):
        """Save metrics to file"""
        os.makedirs('logs', exist_ok=True)
        with open('logs/system_metrics.txt', 'w') as f:
            f.write("System Metrics Log\n")
            f.write("=================\n\n")
            
            for i in range(len(self.metrics['timestamp'])):
                f.write(f"Time: {self.metrics['timestamp'][i]}\n")
                f.write(f"CPU Usage: {self.metrics['cpu_percent'][i]:.1f}%\n")
                f.write(f"Memory Usage: {self.metrics['memory_percent'][i]:.1f}%\n")
                if torch.cuda.is_available():
                    f.write(f"GPU Memory Used: {self.metrics['gpu_memory_used'][i]:.2f} GB\n")
                    f.write(f"GPU Memory Total: {self.metrics['gpu_memory_total'][i]:.2f} GB\n")
                    f.write(f"GPU Memory Usage: {(self.metrics['gpu_memory_used'][i] / self.metrics['gpu_memory_total'][i] * 100):.1f}%\n")
                f.write("\n") 