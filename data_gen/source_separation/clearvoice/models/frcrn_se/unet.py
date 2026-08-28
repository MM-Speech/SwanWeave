import torch
import torch.nn as nn
import torch.nn.functional as F
from . import complex_nn
from .se_layer import SELayer

class Encoder(nn.Module):
    """
    Encoder module for a neural network, responsible for downsampling input features.

    This module consists of a convolutional layer followed by batch normalization and a Leaky ReLU activation.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple): Size of the convolutional kernel.
        stride (tuple): Stride of the convolution.
        padding (tuple, optional): Padding for the convolution. If None, 'SAME' padding is applied.
        complex (bool, optional): If True, use complex convolution layers. Default is False.
        padding_mode (str, optional): Padding mode for convolution. Default is "zeros".
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, complex=False, padding_mode="zeros"):
        super().__init__()
        
        # Determine padding for 'SAME' padding if not provided
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]

        # Select convolution and batch normalization layers based on complex flag
        if complex:
            conv = complex_nn.ComplexConv2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        # Define convolutional layer, batch normalization, and activation function
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where B is batch size, 
                             C is the number of channels, H is height, and W is width.

        Returns:
            torch.Tensor: Output tensor after applying convolution, batch normalization, and activation.
        """
        x = self.conv(x)   # Apply convolution
        x = self.bn(x)     # Apply batch normalization
        x = self.relu(x)   # Apply Leaky ReLU activation
        return x


class Decoder(nn.Module):
    """
    Decoder module for a neural network, responsible for upsampling input features.

    This module consists of a transposed convolutional layer followed by batch normalization 
    and a Leaky ReLU activation.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple): Size of the transposed convolutional kernel.
        stride (tuple): Stride of the transposed convolution.
        padding (tuple, optional): Padding for the transposed convolution. Default is (0, 0).
        complex (bool, optional): If True, use complex transposed convolution layers. Default is False.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), complex=False):
        super().__init__()

        # Select transposed convolution and batch normalization layers based on complex flag
        if complex:
            tconv = complex_nn.ComplexConvTranspose2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
        
        # Define transposed convolutional layer, batch normalization, and activation function
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where B is batch size, 
                             C is the number of channels, H is height, and W is width.

        Returns:
            torch.Tensor: Output tensor after applying transposed convolution, batch normalization, and activation.
        """
        x = self.transconv(x)  # Apply transposed convolution
        x = self.bn(x)         # Apply batch normalization
        x = self.relu(x)       # Apply Leaky ReLU activation
        return x


class UNet(nn.Module):
    """
    U-Net architecture for  handling both real and complex inputs.

    This model uses an encoder-decoder structure with skip connections between corresponding encoder 
    and decoder layers. Squeeze-and-Excitation (SE) layers are integrated into the network for channel 
    attention enhancement.

    Args:
        input_channels (int, optional): Number of input channels. Default is 1.
        complex (bool, optional): If True, use complex layers. Default is False.
        model_complexity (int, optional): Determines the number of channels in the model. Default is 45.
        model_depth (int, optional): Depth of the U-Net model (number of encoder/decoder pairs). Default is 20.
        padding_mode (str, optional): Padding mode for convolutions. Default is "zeros".
    """
    
    def __init__(self, input_channels=1,
                 complex=False,
                 model_complexity=45,
                 model_depth=20,
                 padding_mode="zeros"):
        super().__init__()

        # Adjust model complexity for complex models
        if complex:
            model_complexity = int(model_complexity // 1.414)

        # Initialize model parameters based on specified complexity and depth
        self.set_size(model_complexity=model_complexity, input_channels=input_channels, model_depth=model_depth)
        self.encoders = []
        self.model_length = model_depth // 2
        self.fsmn = complex_nn.ComplexUniDeepFsmn(128, 128, 128)
        self.se_layers_enc = []
        self.fsmn_enc = []

        # Build the encoder structure
        for i in range(self.model_length):
            fsmn_enc = complex_nn.ComplexUniDeepFsmn_L1(128, 128, 128)
            self.add_module("fsmn_enc{}".format(i), fsmn_enc)
            self.fsmn_enc.append(fsmn_enc)
            module = Encoder(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i], complex=complex, padding_mode=padding_mode)
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)
            se_layer_enc = SELayer(self.enc_channels[i + 1], 8)
            self.add_module("se_layer_enc{}".format(i), se_layer_enc)
            self.se_layers_enc.append(se_layer_enc)

        # Build the decoder structure
        self.decoders = []
        self.fsmn_dec = []
        self.se_layers_dec = [] 

        for i in range(self.model_length):
            fsmn_dec = complex_nn.ComplexUniDeepFsmn_L1(128, 128, 128)
            self.add_module("fsmn_dec{}".format(i), fsmn_dec)
            self.fsmn_dec.append(fsmn_dec)
            module = Decoder(self.dec_channels[i] * 2, self.dec_channels[i + 1], kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i], complex=complex)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)
            if i < self.model_length - 1:
                se_layer_dec = SELayer(self.dec_channels[i + 1], 8)
                self.add_module("se_layer_dec{}".format(i), se_layer_dec)
                self.se_layers_dec.append(se_layer_dec)

        # Define final linear layer based on complex flag
        if complex:
            conv = complex_nn.ComplexConv2d
        else:
            conv = nn.Conv2d

        linear = conv(self.dec_channels[-1], 1, 1)  # Final layer to output desired channels

        self.add_module("linear", linear)
        self.complex = complex
        self.padding_mode = padding_mode

        # Convert lists to ModuleLists for proper parameter registration
        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)
        self.se_layers_enc = nn.ModuleList(self.se_layers_enc)
        self.se_layers_dec = nn.ModuleList(self.se_layers_dec)
        self.fsmn_enc = nn.ModuleList(self.fsmn_enc)
        self.fsmn_dec = nn.ModuleList(self.fsmn_dec)

    def forward(self, inputs):
        """
        Forward pass for the UNet model.

        This method processes the input tensor through the encoder-decoder architecture,
        applying convolutional layers, FSMNs, and SE layers. Skip connections are used
        to merge features from the encoder to the decoder.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after processing, representing the computed features.
        """
        x = inputs  # Initialize input tensor
        xs = []  # List to store input tensors for skip connections
        xs_se = []  # List to store outputs after applying SE layers
        xs_se.append(x)  # Add the initial input to the SE outputs list

        # Forward pass through the encoder layers
        for i, encoder in enumerate(self.encoders):
            xs.append(x)  # Store the current input for skip connections
            if i > 0:
                x = self.fsmn_enc[i](x)  # Apply FSMN if not the first encoder
            x = encoder(x)  # Apply the encoder layer
            xs_se.append(self.se_layers_enc[i](x))  # Apply SE layer and store the result

        x = self.fsmn(x)  # Apply the final FSMN after all encoders
        p = x  # Initialize output tensor for decoders

        # Forward pass through the decoder layers
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)  # Apply the decoder layer
            if i < self.model_length - 1:
                p = self.fsmn_dec[i](p)  # Apply FSMN if not the last decoder
            if i == self.model_length - 1:
                break  # Stop processing at the last decoder layer
            if i < self.model_length - 2:
                p = self.se_layers_dec[i](p)  # Apply SE layer for intermediate decoders
            p = torch.cat([p, xs_se[self.model_length - 1 - i]], dim=1)  # Concatenate skip connection

        # Final output processing
        # cmp_spec: [batch, 1, 513, 64, 2]
        cmp_spec = self.linear(p)  # Apply linear transformation to produce final output
        return cmp_spec  # Return the computed output tensor

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        """
        Set the architecture parameters for the UNet model based on specified complexity and depth.

        This method configures the encoder and decoder layers of the UNet by setting the number of channels, 
        kernel sizes, strides, and paddings for each layer according to the provided model complexity 
        and depth. 

        Args:
            model_complexity (int): Base number of channels for the model.
            model_depth (int, optional): Depth of the UNet model, determining the number of encoder/decoder pairs.
                                          Default is 20.
            input_channels (int, optional): Number of input channels to the model. Default is 1.
        
        Raises:
            ValueError: If an unknown model depth is provided.
        """

        # Configuration for model depth of 14
        if model_depth == 14:
            # Set encoder channels for model depth of 14
            self.enc_channels = [input_channels,
                                 128,
                                 128,
                                 128,
                                 128,
                                 128,
                                 128,
                                 128]
            
            # Define kernel sizes for encoder layers
            self.enc_kernel_sizes = [(5, 2),
                                     (5, 2),
                                     (5, 2),
                                     (5, 2),
                                     (5, 2),
                                     (5, 2),
                                     (2, 2)]
            
            # Define strides for encoder layers
            self.enc_strides = [(2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1)]
            
            # Define paddings for encoder layers
            self.enc_paddings = [(0, 1),
                                 (0, 1),
                                 (0, 1),
                                 (0, 1),
                                 (0, 1),
                                 (0, 1),
                                 (0, 1)]

            # Set decoder channels for model depth of 14
            self.dec_channels = [64,
                                 128,
                                 128,
                                 128,
                                 128,
                                 128,
                                 128,
                                 1]
            
            # Define kernel sizes for decoder layers
            self.dec_kernel_sizes = [(2, 2),
                                     (5, 2),
                                     (5, 2),
                                     (5, 2),
                                     (6, 2),
                                     (5, 2),
                                     (5, 2)]

            # Define strides for decoder layers
            self.dec_strides = [(2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1)]

            # Define paddings for decoder layers
            self.dec_paddings = [(0, 1),
                                 (0, 1),
                                 (0, 1),
                                 (0, 1),
                                 (0, 1),
                                 (0, 1),
                                 (0, 1)]

        # Configuration for model depth of 20
        elif model_depth == 20:
            # Set encoder channels for model depth of 20
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]

            # Define kernel sizes for encoder layers
            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (6, 4),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]

            # Define strides for encoder layers
            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1)]

            # Define paddings for encoder layers
            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 None,  # None padding for certain layers
                                 None,
                                 None,  # Adjusted padding based on layer requirements
                                 None,
                                 None,
                                 None,
                                 None,
                                 None]

            # Set decoder channels for model depth of 20
            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2]

            # Define kernel sizes for decoder layers
            self.dec_kernel_sizes = [(4, 3),
                                     (4, 2),
                                     (4, 3),
                                     (4, 2),
                                     (4, 3),
                                     (4, 2),
                                     (6, 3),
                                     (7, 4),
                                     (1, 7),
                                     (7, 1)]

            # Define strides for decoder layers
            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (1, 1),
                                (1, 1)]

            # Define paddings for decoder layers
            self.dec_paddings = [(1, 1),
                                 (1, 0),
                                 (1, 1),
                                 (1, 0),
                                 (1, 1),
                                 (1, 0),
                                 (2, 1),
                                 (2, 1),
                                 (0, 3),
                                 (3, 0)]
        else:
            # Raise an error if an unknown model depth is specified
            raise ValueError("Unknown model depth : {}".format(model_depth))
