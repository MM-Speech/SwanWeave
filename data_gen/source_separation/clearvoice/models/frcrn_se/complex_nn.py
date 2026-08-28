import torch
import torch.nn as nn
import torch.nn.functional as F

class UniDeepFsmn(nn.Module):
    """
    A single layer Deep Feedforward Sequential Memory Network (FSMN) for unidirectional processing.
    This model uses a combination of linear layers and convolutional layers to process input features.

    Attributes:
    - input_dim (int): Number of input features.
    - output_dim (int): Number of output features.
    - lorder (int): Order of the linear filter.
    - hidden_size (int): Number of hidden units in the linear layer.
    """
    
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return  # If no filter order is provided, skip initialization

        self.lorder = lorder
        self.hidden_size = hidden_size

        # Linear layer to project input to hidden space
        self.linear = nn.Linear(input_dim, hidden_size)
        # Projection layer to output space
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        # Depthwise convolution layer for filtering
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder, 1], [1, 1], groups=output_dim, bias=False)

    def forward(self, input):
        """
        Forward pass through the UniDeepFsmn model.
        
        Parameters:
        - input (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        
        Returns:
        - Tensor: Output tensor after processing, with the same shape as input.
        """
        # Apply linear transformation and ReLU activation
        f1 = F.relu(self.linear(input))
        # Project to output dimension
        p1 = self.project(f1)

        # Reshape and pad the tensor for convolution
        x = torch.unsqueeze(p1, 1)  # Shape: (b, c, T, h)
        x_per = x.permute(0, 3, 2, 1)  # Permute to shape (b, h, T, c)
        y = F.pad(x_per, [0, 0, self.lorder - 1, 0])  # Pad the tensor

        # Add convolutional output to original input
        out = x_per + self.conv1(y)

        out1 = out.permute(0, 3, 2, 1)  # Restore original shape
        return input + out1.squeeze()  # Return the combined output


class ComplexUniDeepFsmn(nn.Module):
    """
    A complex variant of the UniDeepFsmn that processes complex-valued input.
    This model has separate layers for the real and imaginary components.

    Attributes:
    - nIn (int): Number of input features.
    - nHidden (int): Number of hidden units in the FSMN layers.
    - nOut (int): Number of output features.
    """

    def __init__(self, nIn, nHidden=128, nOut=128):
        super(ComplexUniDeepFsmn, self).__init__()

        # Initialize FSMN layers for real and imaginary parts
        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_re_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)
        self.fsmn_im_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)

    def forward(self, x):
        """
        Forward pass through the ComplexUniDeepFsmn model.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, channels, height, time, 2) where 
                      the last dimension represents the real and imaginary parts.

        Returns:
        - Tensor: Output tensor after processing.
        """
        # Reshape input to [b, c*h, T, 2]
        b, c, h, T, d = x.size()
        x = torch.reshape(x, (b, c * h, T, d))  # Flatten channel and height
        x = torch.transpose(x, 1, 2)  # Permute to [b, T, c*h, 2]

        # Process the real and imaginary parts
        real_L1 = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary_L1 = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])
        
        # Process the second layer
        real = self.fsmn_re_L2(real_L1) - self.fsmn_im_L2(imaginary_L1)
        imaginary = self.fsmn_re_L2(imaginary_L1) + self.fsmn_im_L2(real_L1)

        # Combine real and imaginary parts into output tensor
        output = torch.stack((real, imaginary), dim=-1)  # Shape: [b, T, h, 2]
        output = torch.transpose(output, 1, 2)  # Shape: [b, h, T, 2]
        output = torch.reshape(output, (b, c, h, T, d))  # Restore original shape

        return output


class ComplexUniDeepFsmn_L1(nn.Module):
    """
    A complex variant of UniDeepFsmn for the first layer.
    This model processes complex-valued input and has two FSMN layers for the real and imaginary parts.

    Attributes:
    - nIn (int): Number of input features.
    - nHidden (int): Number of hidden units in the FSMN layers.
    - nOut (int): Number of output features.
    """

    def __init__(self, nIn, nHidden=128, nOut=128):
        super(ComplexUniDeepFsmn_L1, self).__init__()

        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)

    def forward(self, x):
        """
        Forward pass through the ComplexUniDeepFsmn_L1 model.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, channels, height, time, 2).

        Returns:
        - Tensor: Output tensor after processing.
        """
        b, c, h, T, d = x.size()
        x = torch.transpose(x, 1, 3)  # Shape: [b, T, h, c, 2]
        x = torch.reshape(x, (b * T, h, c, d))  # Reshape to process

        # Process the real and imaginary parts
        real = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])

        # Combine results and reshape back to original dimensions
        output = torch.stack((real, imaginary), dim=-1)  # Shape: [b*T, h, c, 2]
        output = torch.reshape(output, (b, T, h, c, d))  # Restore shape to [b, T, h, c, 2]
        output = torch.transpose(output, 1, 3)  # Shape: [b, c, h, T, 2]

        return output


class BidirectionalLSTM_L1(nn.Module):
    """
    A unidirectional LSTM model for processing sequences.

    Attributes:
    - nIn (int): Number of input features.
    - nHidden (int): Number of hidden units in the LSTM.
    - nOut (int): Number of output features.
    """
    
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM_L1, self).__init__()

        self.rnn = nn.GRU(nIn, nHidden, bidirectional=False)  # Using GRU instead of LSTM

    def forward(self, input):
        """
        Forward pass through the BidirectionalLSTM_L1 model.

        Parameters:
        - input (Tensor): Input tensor of shape (sequence_length, batch_size, input_dim).

        Returns:
        - Tensor: Output tensor after processing.
        """
        output, _ = self.rnn(input)  # Forward pass through GRU
        return output

class BidirectionalLSTM_L2(nn.Module):
    """
    A unidirectional Long Short-Term Memory (LSTM) network that processes input sequences 
    and produces an output using a linear embedding layer.

    Attributes:
        rnn (nn.GRU): The GRU layer for processing the input sequences.
        embedding (nn.Linear): A linear layer that transforms the output of the GRU to the desired output dimension.
    """

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM_L2, self).__init__()

        # Initialize the GRU layer
        self.rnn = nn.GRU(nIn, nHidden, bidirectional=False)
        # Initialize the linear embedding layer
        self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        """
        Forward pass through the Bidirectional LSTM network.

        Args:
            input (torch.Tensor): Input tensor of shape (T, b, nIn), where T is the sequence length, 
                                  b is the batch size, and nIn is the input feature size.

        Returns:
            torch.Tensor: Output tensor of shape (T, b, nOut), where nOut is the output feature size.
        """
        recurrent, _ = self.rnn(input)  # Process the input through the GRU layer
        # Get the shape of the recurrent output
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)  # Flatten the output for the embedding layer
        output = self.embedding(t_rec)  # Apply the linear embedding layer
        output = output.view(T, b, -1)  # Reshape back to (T, b, nOut)

        return output


class ComplexBidirectionalLSTM(nn.Module):
    """
    A complex-valued bidirectional LSTM that processes input sequences containing real 
    and imaginary components, producing a complex-valued output.

    Attributes:
        lstm_re_L1 (BidirectionalLSTM_L1): First layer for the real part of the input.
        lstm_im_L1 (BidirectionalLSTM_L1): First layer for the imaginary part of the input.
        lstm_re_L2 (BidirectionalLSTM_L2): Second layer for the real part of the input.
        lstm_im_L2 (BidirectionalLSTM_L2): Second layer for the imaginary part of the input.
    """

    def __init__(self, nIn, nHidden=128, nOut=1024):
        super(ComplexBidirectionalLSTM, self).__init__()

        # Initialize the first and second LSTM layers for real and imaginary components
        self.lstm_re_L1 = BidirectionalLSTM_L1(nIn, nHidden, nOut)
        self.lstm_im_L1 = BidirectionalLSTM_L1(nIn, nHidden, nOut)
        self.lstm_re_L2 = BidirectionalLSTM_L2(nHidden, nHidden, nOut)
        self.lstm_im_L2 = BidirectionalLSTM_L2(nHidden, nHidden, nOut)

    def forward(self, x):
        """
        Forward pass through the complex-valued bidirectional LSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (b, c, h, T, 2) where:
                - b is the batch size,
                - c is the number of channels,
                - h is the number of hidden units,
                - T is the sequence length,
                - 2 represents the real and imaginary parts.

        Returns:
            torch.Tensor: Output tensor of shape (b, c, h, T, 2).
        """
        # Get the shape of the input tensor
        b, c, h, T, d = x.size()
        # Reshape the input for processing
        x = torch.reshape(x, (b, c*h, T, d))
        # Transpose to prepare for LSTM processing
        x = torch.transpose(x, 0, 2)  # Shape: (T, c*h, d)
        x = torch.transpose(x, 1, 2)  # Shape: (T, d, c*h)

        # Process the real and imaginary parts through LSTM layers
        real_L1 = self.lstm_re_L1(x[..., 0]) - self.lstm_im_L1(x[..., 1])
        imaginary_L1 = self.lstm_re_L1(x[..., 1]) + self.lstm_im_L1(x[..., 0])
        real = self.lstm_re_L2(real_L1) - self.lstm_im_L2(imaginary_L1)
        imaginary = self.lstm_re_L2(imaginary_L1) + self.lstm_im_L2(real_L1)

        # Stack the real and imaginary parts to create the output tensor
        output = torch.stack((real, imaginary), dim=-1)  # Shape: (T, b, h, 2)
        output = torch.transpose(output, 1, 2)  # Shape: (T, h, b, 2)
        output = torch.transpose(output, 0, 2)  # Shape: (b, h, T, 2)
        output = torch.reshape(output, (b, c, h, T, d))  # Shape: (b, c, h, T, 2)

        return output


class ComplexConv2d(nn.Module):
    """
    A complex-valued 2D convolutional layer that processes input tensors with real 
    and imaginary parts, returning a complex output.

    Attributes:
        conv_re (nn.Conv2d): Convolutional layer for the real part.
        conv_im (nn.Conv2d): Convolutional layer for the imaginary part.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        # Initialize convolutional layers for real and imaginary components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):
        """
        Forward pass through the complex convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, axis1, axis2, 2) 
                             representing the real and imaginary parts.

        Returns:
            torch.Tensor: Output tensor containing the convolved real and imaginary parts.
        """
        # Apply convolution to the real and imaginary parts
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)  # Stack real and imaginary components

        return output


class ComplexConvTranspose2d(nn.Module):
    """
    A complex-valued 2D transposed convolutional layer that processes input tensors 
    with real and imaginary parts, returning a complex output.

    Attributes:
        tconv_re (nn.ConvTranspose2d): Transposed convolutional layer for the real part.
        tconv_im (nn.ConvTranspose2d): Transposed convolutional layer for the imaginary part.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        # Initialize transposed convolutional layers for real and imaginary components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):
        """
        Forward pass through the complex transposed convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, axis1, axis2, 2) 
                             representing the real and imaginary parts.

        Returns:
            torch.Tensor: Output tensor containing the transposed convoluted real and imaginary parts.
        """
        # Apply transposed convolution to the real and imaginary parts
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)  # Stack real and imaginary components

        return output

class ComplexBatchNorm2d(nn.Module):
    """
    A complex-valued batch normalization layer that normalizes input tensors with 
    separate real and imaginary components.

    This layer applies batch normalization independently to the real and imaginary parts of the input,
    ensuring that each part is normalized appropriately. It is particularly useful in complex-valued networks,
    where inputs are represented as pairs of real and imaginary components.

    Attributes:
        bn_re (nn.BatchNorm2d): Batch normalization layer for the real part of the input.
        bn_im (nn.BatchNorm2d): Batch normalization layer for the imaginary part of the input.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        """
        Initializes the ComplexBatchNorm2d layer.

        Args:
            num_features (int): Number of features (channels) for the input.
            eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-5.
            momentum (float, optional): Momentum for the running mean and variance. Default is 0.1.
            affine (bool, optional): If True, this layer has learnable parameters. Default is True.
            track_running_stats (bool, optional): If True, track the running mean and variance. Default is True.
        """
        super().__init__()
        # Initialize batch normalization layers for real and imaginary parts
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        """
        Forward pass through the complex batch normalization layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, height, width, 2),
                              where the last dimension represents the real and imaginary parts.

        Returns:
            torch.Tensor: Output tensor containing the normalized real and imaginary components,
                          with the same shape as the input tensor.
        """
        # Apply batch normalization to the real part
        real = self.bn_re(x[..., 0])
        # Apply batch normalization to the imaginary part
        imag = self.bn_im(x[..., 1])
        # Stack the normalized real and imaginary parts back together
        output = torch.stack((real, imag), dim=-1)

        return output
