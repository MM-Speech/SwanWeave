from torch import nn
import torch

## Referencing the paper: https://arxiv.org/pdf/1709.01507
class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer (SELayer) for enhancing channel-wise feature responses.

    The SELayer implements the Squeeze-and-Excitation block as proposed in the paper,
    which adaptively recalibrates channel-wise feature responses by modeling the interdependencies
    between channels.

    Args:
        channel (int): The number of input channels.
        reduction (int): The reduction ratio for the number of channels in the bottleneck.
                         Default is 16.
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        # Adaptive average pooling to generate a global descriptor
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers for the real part
        self.fc_r = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # Reduce channels
            nn.ReLU(inplace=True),                     # Activation function
            nn.Linear(channel // reduction, channel),  # Restore channels
            nn.Sigmoid()                               # Sigmoid activation to scale outputs
        )

        # Fully connected layers for the imaginary part
        self.fc_i = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # Reduce channels
            nn.ReLU(inplace=True),                     # Activation function
            nn.Linear(channel // reduction, channel),  # Restore channels
            nn.Sigmoid()                               # Sigmoid activation to scale outputs
        )

    def forward(self, x):
        """
        Forward pass for the SELayer.

        The forward method applies the squeeze-and-excitation operation on the input tensor `x`.
        It computes the channel-wise attention weights for both the real and imaginary parts 
        of the input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W), where:
                              B - batch size,
                              C - number of channels,
                              D - depth,
                              H - height,
                              W - width.

        Returns:
            torch.Tensor: Output tensor after applying channel-wise attention, 
                          same shape as input `x`.
        """
        # Extract the batch size and number of channels
        b, c, _, _, _ = x.size()

        # Compute the squeeze operation for the real part
        x_r = self.avg_pool(x[:, :, :, :, 0]).view(b, c)  # Global average pooling
        # Compute the squeeze operation for the imaginary part
        x_i = self.avg_pool(x[:, :, :, :, 1]).view(b, c)  # Global average pooling

        # Calculate channel-wise attention for the real part
        y_r = self.fc_r(x_r).view(b, c, 1, 1, 1) - self.fc_i(x_i).view(b, c, 1, 1, 1)
        # Calculate channel-wise attention for the imaginary part
        y_i = self.fc_r(x_i).view(b, c, 1, 1, 1) + self.fc_i(x_r).view(b, c, 1, 1, 1)

        # Concatenate real and imaginary attention weights along the channel dimension
        y = torch.cat([y_r, y_i], 4)

        # Scale the input features by the attention weights
        return x * y
