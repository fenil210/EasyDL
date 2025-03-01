import torch
import torch.nn as nn
from torch import Tensor

class CNN(nn.Module):
    """Modular CNN implementation with configurable layers."""
    
    def __init__(self, in_channels: int, num_classes: int, layer_config: list):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes
            layer_config: List of dictionaries specifying layer configurations
        """
        super(CNN, self).__init__()
        self.layers = nn.Sequential()
        
        # Build convolutional layers
        for i, config in enumerate(layer_config):
            self.layers.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=config['out_channels'],
                    kernel_size=config['kernel_size'],
                    stride=config['stride'],
                    padding=config['padding'],
                )
            )
            in_channels = config['out_channels']  # Update in_channels for next layer
            
            # Add activation
            if config['activation'] == 'relu':
                self.layers.add_module(f"relu_{i}", nn.ReLU())
            elif config['activation'] == 'leaky_relu':
                self.layers.add_module(f"leaky_relu_{i}", nn.LeakyReLU(0.1))
                
            # Add pooling if specified
            if 'pool' in config:
                self.layers.add_module(
                    f"pool_{i}",
                    nn.MaxPool2d(
                        kernel_size=config['pool']['kernel_size'],
                        stride=config['pool']['stride']
                    )
                )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layer
        self.fc = nn.Linear(in_channels * 4 * 4, num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.adaptive_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x