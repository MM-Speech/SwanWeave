import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from .conformer import ConformerBlock

class LearnableSigmoid(nn.Module):
    """A learnable sigmoid activation function that scales the output 
    based on the input features.

    Args:
        in_features (int): The number of input features for the sigmoid function.
        beta (float, optional): A scaling factor for the sigmoid output. Default is 1.
    
    Attributes:
        beta (float): The scaling factor for the sigmoid function.
        slope (Parameter): Learnable parameter that adjusts the slope of the sigmoid.
    """
    
    def __init__(self, in_features, beta=1):
        """Initializes the LearnableSigmoid module.

        Args:
            in_features (int): Number of input features.
            beta (float, optional): Scaling factor for the sigmoid output.
        """
        super().__init__()
        self.beta = beta  # Scaling factor for the sigmoid
        self.slope = nn.Parameter(torch.ones(in_features))  # Learnable slope parameter
        self.slope.requiresGrad = True  # Ensure gradient updates

    def forward(self, x):
        """Forward pass of the learnable sigmoid function.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, in_features].

        Returns:
            torch.Tensor: The scaled sigmoid output tensor.
        """
        return self.beta * torch.sigmoid(self.slope * x)


#%% Spectrograms
def segment_specs(y, seg_length=15, seg_hop=4, max_length=None):
    """Segments a spectrogram into smaller segments for input to a CNN. 
    Each segment includes neighboring frequency bins to preserve 
    contextual information.

    Args:
        y (torch.Tensor): Input spectrogram tensor of shape [B, H, W], 
                          where B is batch size, H is number of mel bands, 
                          and W is the length of the spectrogram.
        seg_length (int): Length of each segment (must be odd). Default is 15.
        seg_hop (int): Hop length for segmenting the spectrogram. Default is 4.
        max_length (int, optional): Maximum number of windows allowed. If the number of 
                                     windows exceeds this, a ValueError is raised.

    Returns:
        torch.Tensor: Segmented tensor with shape [B*n, C, H, seg_length], where n is the 
                      number of segments, C is the number of channels (always 1).
    
    Raises:
        ValueError: If seg_length is even or if the number of windows exceeds max_length.
    """
    # Ensure segment length is odd
    if seg_length % 2 == 0:
        raise ValueError('seg_length must be odd! (seg_length={})'.format(seg_length))
    
    # Convert input to tensor if it's not already
    if not torch.is_tensor(y):
        y = torch.tensor(y)

    B, _, _ = y.size()  # Extract batch size and dimensions
    for b in range(B):
        x = y[b, :, :]  # Extract the current batch's spectrogram
        n_wins = x.shape[1] - (seg_length - 1)  # Calculate number of windows
        
        # Segment the mel-spectrogram
        idx1 = torch.arange(seg_length)  # Indices for segment length
        idx2 = torch.arange(n_wins)  # Indices for number of windows
        idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)  # Create indices for segments
        x = x.transpose(1, 0)[idx3, :].unsqueeze(1).transpose(3, 2)  # Rearrange dimensions for CNN input

        # Adjust segments based on hop length
        if seg_hop > 1:
            x = x[::seg_hop, :]  # Downsample segments
            n_wins = int(np.ceil(n_wins / seg_hop))  # Update number of windows

        # Pad the segments if max_length is specified
        if max_length is not None:
            if max_length < n_wins:
                raise ValueError('n_wins {} > max_length {}. Increase max window length max_segments!'.format(n_wins, max_length))
            x_padded = torch.zeros((max_length, x.shape[1], x.shape[2], x.shape[3]))  # Create a padded tensor
            x_padded[:n_wins, :] = x  # Fill the padded tensor with the segments
            x = x_padded  # Update x to the padded tensor

        # Concatenate segments from each batch
        if b == 0:
            z = x.unsqueeze(0)  # Initialize z for the first batch
        else:
            z = torch.cat((z, x.unsqueeze(0)), axis=0)  # Concatenate to z

    # Reshape the final tensor for output
    B, n, c, f, t = z.size()
    z = z.view(B * n, c, f, t)  # Combine batch and segment dimensions
    return z  # Return the segmented spectrogram tensor

class AdaptCNN(nn.Module):
    """
    AdaptCNN: A convolutional neural network (CNN) with adaptive max pooling that 
    can be used as a framewise model. This architecture is more flexible than a 
    standard CNN, which requires a fixed input dimension. The model consists of six 
    convolutional layers, with adaptive pooling at each layer to handle varying input sizes.

    Args:
        input_channels (int): Number of input channels (default is 2).
        c_out_1 (int): Number of output channels for the first convolutional layer (default is 16).
        c_out_2 (int): Number of output channels for the second convolutional layer (default is 32).
        c_out_3 (int): Number of output channels for the third and subsequent convolutional layers (default is 64).
        kernel_size (list or int): Size of the convolutional kernels (default is [3, 3]).
        dropout (float): Dropout rate for regularization (default is 0.2).
        pool_1 (list): Pooling parameters for the first adaptive pooling layer (default is [101, 7]).
        pool_2 (list): Pooling parameters for the second adaptive pooling layer (default is [50, 7]).
        pool_3 (list): Pooling parameters for the third adaptive pooling layer (default is [25, 5]).
        pool_4 (list): Pooling parameters for the fourth adaptive pooling layer (default is [12, 5]).
        pool_5 (list): Pooling parameters for the fifth adaptive pooling layer (default is [6, 3]).
        fc_out_h (int, optional): Number of output units for the final fully connected layer. If None, the output size is determined from previous layers.

    Attributes:
        name (str): Name of the model.
        dropout (Dropout): Dropout layer for regularization.
        conv1, conv2, conv3, conv4, conv5, conv6 (Conv2d): Convolutional layers.
        bn1, bn2, bn3, bn4, bn5, bn6 (BatchNorm2d): Batch normalization layers.
        fc (Linear, optional): Fully connected layer.
        fan_out (int): Output dimension of the final layer.
    """
    
    def __init__(self, 
                 input_channels=2,
                 c_out_1=16, 
                 c_out_2=32,
                 c_out_3=64,
                 kernel_size=[3, 3], 
                 dropout=0.2,
                 pool_1=[101, 7],
                 pool_2=[50, 7],
                 pool_3=[25, 5],
                 pool_4=[12, 5],
                 pool_5=[6, 3],
                 fc_out_h=None):
        """Initializes the AdaptCNN model with the specified parameters."""
        super().__init__()
        self.name = 'CNN_adapt'

        # Model parameters
        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        self.pool_3 = pool_3
        self.pool_4 = pool_4
        self.pool_5 = pool_5
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        # Dropout layer for regularization
        self.dropout = nn.Dropout2d(p=self.dropout_rate)
        
        # Ensure kernel_size is a tuple
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
            
        # Set kernel size for the last convolutional layer
        self.kernel_size_last = (self.kernel_size[0], self.pool_5[1])
            
        # Determine padding for convolutional layers based on kernel size
        if self.kernel_size[1] == 1:
            self.cnn_pad = (1, 0)  # No padding needed for 1D convolution
        else:
            self.cnn_pad = (1, 1)   # Padding for 2D convolution
            
        # Define convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(self.input_channels, self.c_out_1, self.kernel_size, padding=self.cnn_pad)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = nn.Conv2d(self.conv1.out_channels, self.c_out_2, self.kernel_size, padding=self.cnn_pad)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.c_out_3, self.kernel_size, padding=self.cnn_pad)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)

        self.conv4 = nn.Conv2d(self.conv3.out_channels, self.c_out_3, self.kernel_size, padding=self.cnn_pad)
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)

        self.conv5 = nn.Conv2d(self.conv4.out_channels, self.c_out_3, self.kernel_size, padding=self.cnn_pad)
        self.bn5 = nn.BatchNorm2d(self.conv5.out_channels)

        self.conv6 = nn.Conv2d(self.conv5.out_channels, self.c_out_3, self.kernel_size_last, padding=(1, 0))
        self.bn6 = nn.BatchNorm2d(self.conv6.out_channels)
        
        # Define fully connected layer if output size is specified
        if self.fc_out_h:
            self.fc = nn.Linear(self.conv6.out_channels * self.pool_3[0], self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.pool_3[0])

    def forward(self, x):
        """Defines the forward pass of the AdaptCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_channels, height, width].

        Returns:
            torch.Tensor: Output tensor after passing through the CNN layers.
        """
        # Forward pass through each layer with ReLU activation and adaptive pooling
        x = F.relu(self.bn1(self.conv1(x)))  # First convolutional layer
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_1))  # Adaptive pooling after conv1

        x = F.relu(self.bn2(self.conv2(x)))  # Second convolutional layer
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_2))  # Adaptive pooling after conv2
        
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.bn3(self.conv3(x)))  # Third convolutional layer
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_3))  # Adaptive pooling after conv3

        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.bn4(self.conv4(x)))  # Fourth convolutional layer
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_4))  # Adaptive pooling after conv4

        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.bn5(self.conv5(x)))  # Fifth convolutional layer
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_5))  # Adaptive pooling after conv5

        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.bn6(self.conv6(x)))  # Last convolutional layer
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, self.conv6.out_channels * self.pool_5[0])  
        
        # Apply fully connected layer if defined
        if self.fc_out_h:
            x = self.fc(x)  # Fully connected output
        
        return x  # Return the output tensor

class PoolAttFF(nn.Module):
    """
    PoolAttFF: An attention pooling module with an additional feed-forward network.
    
    This module performs attention-based pooling on input features followed by a 
    feed-forward neural network. The attention mechanism helps in focusing on the 
    important parts of the input while pooling.

    Args:
        d_input (int): The dimensionality of the input features (default is 384).
        output_size (int): The size of the output after the feed-forward network (default is 1).
        h (int): The size of the hidden layer in the feed-forward network (default is 128).
        dropout (float): The dropout rate for regularization (default is 0.1).

    Attributes:
        linear1 (Linear): First linear layer transforming input features to hidden size.
        linear2 (Linear): Second linear layer producing attention scores.
        linear3 (Linear): Final linear layer producing the output.
        activation (function): Activation function used in the network (ReLU).
        dropout (Dropout): Dropout layer for regularization.
    """
    
    def __init__(self, d_input=384, output_size=1, h=128, dropout=0.1):
        """Initializes the PoolAttFF module with the specified parameters."""
        super().__init__()

        # Define the feed-forward layers
        self.linear1 = nn.Linear(d_input, h)  # First linear layer
        self.linear2 = nn.Linear(h, 1)         # Second linear layer for attention scores

        self.linear3 = nn.Linear(d_input, output_size)  # Final output layer

        self.activation = F.relu  # Activation function
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization

    def forward(self, x):
        """Defines the forward pass of the PoolAttFF module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_input].

        Returns:
            torch.Tensor: Output tensor after attention pooling and feed-forward network.
        """
        # Compute attention scores
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2, 1)  # Transpose for softmax

        # Apply softmax to get attention weights
        att = F.softmax(att, dim=2)  # Softmax along the sequence length

        # Perform attention pooling
        x = torch.bmm(att, x)  # Batch matrix multiplication
        x = x.squeeze(1)  # Remove unnecessary dimension
        x = self.linear3(x)  # Final output layer

        return x  # Return the output tensor


class Discriminator(nn.Module):
    """
    Discriminator: A neural network that predicts a normalized PESQ value 
    between a predicted waveform (x) and a ground truth waveform (y).

    The model concatenates the two input waveforms, processes them through 
    a convolutional network (CNN), applies self-attention, and outputs a 
    value between 0 and 1 using a sigmoid activation function.

    Args:
        ndf (int): Number of filters in the convolutional layers (not directly used in this implementation).
        in_channel (int): Number of input channels (default is 2).

    Attributes:
        dim (int): Dimensionality of the feature representation (default is 384).
        cnn (AdaptCNN): CNN model for feature extraction.
        att (Sequential): Sequential stack of Conformer blocks for attention processing.
        pool (PoolAttFF): Attention pooling module.
        sigmoid (LearnableSigmoid): Sigmoid layer for final output.
    """
    
    def __init__(self, ndf, in_channel=2):
        """Initializes the Discriminator with specified parameters."""
        super().__init__()
        self.dim = 384  # Dimensionality of the feature representation
        self.cnn = AdaptCNN()  # CNN model for feature extraction

        # Define attention layers using Conformer blocks
        self.att = nn.Sequential(
            ConformerBlock(dim=self.dim, dim_head=self.dim // 4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2),
            ConformerBlock(dim=self.dim, dim_head=self.dim // 4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        )

        # Define attention pooling module
        self.pool = PoolAttFF()
        self.sigmoid = LearnableSigmoid(1)  # Sigmoid layer for output normalization

    def forward(self, x, y):
        """Defines the forward pass of the Discriminator.

        Args:
            x (torch.Tensor): Predicted waveform tensor of shape [batch_size, 1, height, width].
            y (torch.Tensor): Ground truth waveform tensor of shape [batch_size, 1, height, width].

        Returns:
            torch.Tensor: Output tensor representing the predicted PESQ value.
        """
        B, _, _, _ = x.size()  # Get the batch size from input x
        x = segment_specs(x.squeeze(1))  # Segment and process predicted waveform
        y = segment_specs(y.squeeze(1))  # Segment and process ground truth waveform

        # Concatenate the processed waveforms
        xy = torch.cat([x, y], dim=1)  # Concatenate along the channel dimension
        cnn_out = self.cnn(xy)  # Extract features using CNN

        _, d = cnn_out.size()  # Get dimensions of CNN output
        cnn_out = cnn_out.view(B, -1, d)  # Reshape for attention processing
        att_out = self.att(cnn_out)  # Apply self-attention layers
        pool_out = self.pool(att_out)  # Apply attention pooling module
        
        out = self.sigmoid(pool_out)  # Normalize output using sigmoid function
        return out  # Return the predicted PESQ value
