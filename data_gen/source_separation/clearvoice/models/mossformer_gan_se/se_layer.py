from torch import nn
import torch

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) Layer.

    This layer implements the Squeeze-and-Excitation mechanism, which adaptively
    recalibrates channel-wise feature responses by explicitly modeling 
    interdependencies between channels. It enhances the representational power
    of a neural network by emphasizing informative features while suppressing
    less useful ones.

    Args:
        channel (int): The number of input channels.
        reduction (int, optional): Reduction ratio for the dimensionality
            of the intermediate representations. Default is 16.
    """
    
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_layer = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # First linear layer
            nn.ReLU(inplace=True),                    # Activation layer
            nn.Linear(channel // reduction, channel),  # Second linear layer
            nn.Sigmoid()                              # Sigmoid activation for scaling
        )
        
        # Adaptive max pooling to reduce spatial dimensions to 1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool_layer = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # First linear layer
            nn.ReLU(inplace=True),                    # Activation layer
            nn.Linear(channel // reduction, channel),  # Second linear layer
            nn.Sigmoid()                              # Sigmoid activation for scaling
        )

    def forward(self, x):
        """
        Forward pass for the SE Layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) where:
                B = batch size,
                C = number of channels,
                H = height,
                W = width.

        Returns:
            Tensor: Output tensor of the same shape as input `x` after
            applying the squeeze-and-excitation mechanism.
        """
        
        b, c, _, _ = x.size()  # Unpack input dimensions
        x_avg = self.avg_pool(x).view(b, c)  # Squeeze: apply average pooling
        x_avg = self.avg_pool_layer(x_avg).view(b, c, 1, 1)  # Excitation: pass through layers

        x_max = self.max_pool(x).view(b, c)  # Squeeze: apply max pooling
        x_max = self.max_pool_layer(x_max).view(b, c, 1, 1)  # Excitation: pass through layers
        
        # Scale the input features by the computed channel weights
        y = (x_avg + x_max) * x  
        return y  # Return the recalibrated output
