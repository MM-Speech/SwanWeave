"""
Implementation of the MossFormer2 block
This source code is rewritten by Shengkui Zhao based on https://github.com/lucidrains/FLASH-pytorch
"""

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torchinfo import summary
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from .conv_module import ConvModule, GLU, FFConvM_Dilated
from .fsmn import UniDeepFsmn, UniDeepFsmn_dilated
from .layer_norm import CLayerNorm, GLayerNorm, GlobLayerNorm, ILayerNorm

# Functions

def identity(t, *args, **kwargs):
    """Identity function, returns the input tensor unchanged."""
    return t

def append_dims(x, num_dims):
    """Appends extra dimensions to the input tensor `x`."""
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

def exists(val):
    """Checks if a value exists (is not None)."""
    return val is not None

def default(val, d):
    """Returns the value if it exists, otherwise returns the default `d`."""
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    """Returns the padding required to make `n` a multiple of `mult`."""
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# ScaleNorm Layer

class ScaleNorm(nn.Module):
    """
    ScaleNorm Layer: A variant of LayerNorm that scales the input tensor
    by a factor proportional to the inverse square root of the dimension.
    
    Args:
        dim (int): Dimensionality of the input.
        eps (float): A small value to avoid division by zero.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5  # Scaling factor
        self.eps = eps  # Epsilon for numerical stability
        self.g = nn.Parameter(torch.ones(1))  # Trainable scaling parameter

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale  # Compute norm
        return x / norm.clamp(min=self.eps) * self.g  # Scale input by norm

# Absolute Positional Encodings

class ScaledSinuEmbedding(nn.Module):
    """
    Scaled Sinusoidal Embedding: Generates sinusoidal positional encodings
    that are scaled by a learnable parameter.
    
    Args:
        dim (int): Dimensionality of the embedding.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))  # Learnable scaling parameter
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # Inverse frequency for sinusoidal embeddings
        self.register_buffer('inv_freq', inv_freq)  # Save as a non-trainable buffer

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device=device).type_as(self.inv_freq)  # Time indices
        sinu = einsum('i, j -> i j', t, self.inv_freq)  # Sinusoidal function
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)  # Concatenate sine and cosine embeddings
        return emb * self.scale  # Scale the embeddings

# Offset-Scale Layer

class OffsetScale(nn.Module):
    """
    OffsetScale: Applies an element-wise affine transformation (scaling and offset)
    to the input tensor.
    
    Args:
        dim (int): Dimensionality of the input.
        heads (int): Number of heads for multi-head operations.
    """
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))  # Learnable scaling parameter
        self.beta = nn.Parameter(torch.zeros(heads, dim))  # Learnable bias parameter
        nn.init.normal_(self.gamma, std=0.02)  # Initialize gamma with a small standard deviation

    def forward(self, x):
        # Apply scaling and offset, then split along the head dimension
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)  # Unbind (split) the tensor along the head dimension

# Feedforward Convolution Module

class FFConvM(nn.Module):
    """
    Feedforward Convolution Module: A feedforward network with normalization,
    linear projection, and convolution for processing sequential data.
    
    Args:
        dim_in (int): Input dimensionality.
        dim_out (int): Output dimensionality.
        norm_klass (class): Normalization layer class (e.g., LayerNorm).
        dropout (float): Dropout rate.
    """
    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),  # Apply normalization
            nn.Linear(dim_in, dim_out),  # Linear transformation
            nn.SiLU(),  # SiLU activation function
            ConvModule(dim_out),  # Apply convolution module
            nn.Dropout(dropout)  # Apply dropout for regularization
        )

    def forward(self, x):
        output = self.mdl(x)  # Forward pass through the module
        return output

class FFM(nn.Module):
    """
    Feedforward Module (FFM): A basic feedforward network that consists of 
    normalization, linear projection, activation, and dropout for regularization.
    
    Args:
        dim_in (int): Input dimensionality.
        dim_out (int): Output dimensionality.
        norm_klass (class): Normalization layer class, default is LayerNorm.
        dropout (float): Dropout rate for regularization, default is 0.1.
    """
    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        super().__init__()
        # Define a sequential feedforward network with normalization, linear projection, and activation
        self.mdl = nn.Sequential(
            norm_klass(dim_in),           # Apply normalization to stabilize learning
            nn.Linear(dim_in, dim_out),   # Linear transformation to project input to output dimensionality
            nn.SiLU(),                    # SiLU activation function for non-linearity
            nn.Dropout(dropout)           # Dropout for regularization to prevent overfitting
        )

    def forward(self, x):
        """Forward pass through the feedforward network."""
        output = self.mdl(x)  # Apply the feedforward module to the input
        return output

class FLASH_ShareA_FFConvM(nn.Module):
    """
    FLASH_ShareA_FFConvM: A block that combines feedforward convolutional modules (FFConvM) 
    with a specialized attention mechanism to process sequences in groups and 
    perform efficient attention calculations. 

    This module includes both quadratic and linear attention mechanisms, 
    with optional token shifting for better performance in causal settings. 
    It also supports rotary positional embeddings and flexible normalization.

    Args:
        dim (int): The input and output dimensionality of the model.
        group_size (int): The size of groups used for attention calculations. Default is 256.
        query_key_dim (int): Dimensionality of the query and key vectors. Default is 128.
        expansion_factor (float): Factor to expand the dimensionality in the hidden layer. Default is 1.0.
        causal (bool): Whether to use causal attention (for autoregressive tasks). Default is False.
        dropout (float): Dropout rate for regularization. Default is 0.1.
        rotary_pos_emb (RotaryEmbedding, optional): Positional embedding using rotary encoding. Default is None.
        norm_klass (class): Normalization class, defaults to LayerNorm.
        shift_tokens (bool): Whether to shift tokens before attention for performance boost. Default is True.
    """
    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=1.,
        causal=False,
        dropout=0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)        
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # Rotary positional embeddings
        self.rotary_pos_emb = rotary_pos_emb

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Input projections
        self.to_hidden = FFConvM(  # FFConvM for value and gating
            dim_in=dim,
            dim_out=hidden_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.to_qk = FFConvM(  # FFConvM for query and key
            dim_in=dim,
            dim_out=query_key_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )

        # Scaling and offset for attention
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)

        # Output projection
        self.to_out = FFConvM(  # FFConvM to combine and produce final output
            dim_in=dim*2,
            dim_out=dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        
        # Sigmoid gate activation
        self.gateActivate = nn.Sigmoid()

    def forward(self, x, *, mask=None):
        """
        Forward pass for the block.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence length, dim).
            mask (Tensor, optional): Attention mask. Default is None.

        Returns:
            Tensor: Output tensor after attention and feedforward operations.
        """

        # Save input as residual for skip connection
        residual = x

        # Optional token shifting
        if self.shift_tokens:
            x_shift, x_pass = x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)  # Shift tokens
            x = torch.cat((x_shift, x_pass), dim=-1)

        # Projections for value and gating
        v, u = self.to_hidden(x).chunk(2, dim=-1)  # Split into two branches: v and u
        qk = self.to_qk(x)  # Query and key projections

        # Offset and scale for queries and keys
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)

        # Calculate attention output
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, mask)

        # Gated interaction between attention outputs and gating mechanism
        out = (att_u * v) * self.gateActivate(att_v * u)

        # Residual connection and output projection
        x = residual + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        """
        Computes attention using quadratic and linear mechanisms.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence length, dim).
            quad_q (Tensor): Quadratic query.
            lin_q (Tensor): Linear query.
            quad_k (Tensor): Quadratic key.
            lin_k (Tensor): Linear key.
            v (Tensor): Value tensor.
            u (Tensor): Gating tensor.
            mask (Tensor, optional): Attention mask. Default is None.

        Returns:
            Tuple[Tensor]: Attention outputs for value and gating.
        """
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        # Mask for linear attention (if provided)
        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # Rotary embeddings for queries and keys (if provided)
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # Padding to match group size
        padding = padding_to_multiple_of(n, g)
        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: F.pad(t, (0, 0, 0, padding), value=0.), (quad_q, quad_k, lin_q, lin_k, v, u))

            mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
            mask = F.pad(mask, (0, padding), value=False)

        # Group inputs along sequence
        quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n=self.group_size), (quad_q, quad_k, lin_q, lin_k, v, u))

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j=g)

        # Quadratic attention
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g
        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        # Linear attention (with cumulative sum for causal mode)
        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g
            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / g
            lin_ku = lin_ku.cumsum(dim=1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        # Fold groups back into full sequence
        return map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out_v + lin_out_v, quad_out_u + lin_out_u))

class Gated_FSMN(nn.Module):
    """
    Gated_FSMN: A gated feedforward sequential memory network (FSMN) block that combines
    the outputs of two feedforward convolutional modules (FFConvM) to enhance sequence modeling.
    This module applies gated interactions between the outputs of FSMN and a second FFConvM block.

    The FSMN is useful for capturing long-term dependencies in sequential data while 
    the gating mechanism regulates the influence of FSMN outputs.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        lorder (int): Filter length or order for FSMN.
        hidden_size (int): Size of the hidden layers used within the FSMN and FFConvM.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        lorder,
        hidden_size
    ):
        super().__init__()

        # FFConvM block for 'u' branch
        self.to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )

        # FFConvM block for 'v' branch
        self.to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )

        # Unidirectional FSMN (UniDeepFsmn) for processing 'u' branch
        self.fsmn = UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)

    def forward(self, x):
        """
        Forward pass for the Gated_FSMN block.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence length, in_channels).

        Returns:
            Tensor: Output tensor after applying gated FSMN and feedforward operations.
        """
        input = x  # Save original input for skip connection

        # Process input through FFConvM for both 'u' and 'v' branches
        x_u = self.to_u(x)
        x_v = self.to_v(x)

        # Apply FSMN to the 'u' branch
        x_u = self.fsmn(x_u)

        # Gated interaction between 'u' and 'v' branches, followed by skip connection
        x = x_v * x_u + input
        return x

class Gated_FSMN_Block(nn.Module):
    """
    Gated-FSMN Block: A sequential block that combines a Gated Feedforward Sequential Memory Network (FSMN)
    with normalization and convolutional layers for enhanced feature learning. This block applies gating 
    mechanisms on sequential data to capture long-term dependencies, while maintaining efficient processing.

    Args:
        dim (int): Number of input channels.
        inner_channels (int, optional): Number of channels used in the inner layers. Defaults to 256.
        group_size (int, optional): Size of the groups in sequential processing. Defaults to 256.
        norm_type (str, optional): Type of normalization to use ('scalenorm' or 'layernorm'). Defaults to 'scalenorm'.
    """
    def __init__(self,
                 dim,
                 inner_channels=256,
                 group_size=256, 
                 norm_type='scalenorm'):
        super(Gated_FSMN_Block, self).__init__()

        # Select the normalization method based on 'norm_type'
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # First 1D convolution layer to project input to 'inner_channels' dimension
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),  # Pointwise convolution
            nn.PReLU(),  # Parametric ReLU activation
        )

        # First layer normalization (using CLayerNorm for channel-wise normalization)
        self.norm1 = CLayerNorm(inner_channels)

        # Gated FSMN for long-term sequential modeling with gating mechanism
        self.gated_fsmn = Gated_FSMN(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)

        # Second layer normalization (channel-wise) after FSMN
        self.norm2 = CLayerNorm(inner_channels)

        # Second 1D convolution layer to project output back to 'dim' dimension
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        """
        Forward pass through the Gated-FSMN Block.

        Args:
            input (Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        
        # Apply first 1D convolution and activation (transpose to match Conv1d format)
        conv1 = self.conv1(input.transpose(2, 1))
        
        # Normalize the output of the first convolution
        norm1 = self.norm1(conv1)
        
        # Apply the Gated FSMN block to the normalized output (transpose to match FSMN format)
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))
        
        # Normalize the output of FSMN (transpose back to match Conv1d format)
        norm2 = self.norm2(seq_out.transpose(2, 1))
        
        # Apply second 1D convolution
        conv2 = self.conv2(norm2)
        
        # Add the input (skip connection) and return the result (transpose back to original format)
        return conv2.transpose(2, 1) + input

import torch.nn as nn

class Gated_FSMN_dilated(nn.Module):
    """
    Gated FSMN (Finite State Machine Network) with dilated convolutions.
    
    This module implements a gated mechanism using two parallel feedforward 
    convolutions to generate the input for a dilated FSMN. The gated outputs 
    are combined to enhance the input features, allowing for better speech 
    enhancement performance.

    Attributes:
        to_u (FFConvM): Feedforward convolution module for input transformation 
                         to the u-gate.
        to_v (FFConvM): Feedforward convolution module for input transformation 
                         to the v-gate.
        fsmn (UniDeepFsmn_dilated): The dilated FSMN for processing the u-gate 
                                     output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lorder: int,
        hidden_size: int
    ):
        """
        Initializes the Gated_FSMN_dilated module.
        
        Args:
            in_channels (int): Number of input channels (features).
            out_channels (int): Number of output channels (features).
            lorder (int): Order of the FSMN.
            hidden_size (int): Number of hidden units in the feedforward layers.
        """
        super().__init__()
        
        # Feedforward convolution for the u-gate
        self.to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        
        # Feedforward convolution for the v-gate
        self.to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        
        # Initialize the dilated FSMN
        self.fsmn = UniDeepFsmn_dilated(in_channels, out_channels, lorder, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Gated FSMN module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length).
        
        Returns:
            torch.Tensor: Output tensor after processing through the gated FSMN.
        """
        input = x  # Store the original input for later use
        x_u = self.to_u(x)  # Process input through u-gate
        x_v = self.to_v(x)  # Process input through v-gate
        
        x_u = self.fsmn(x_u)  # Apply FSMN to u-gate output
        
        # Combine the outputs from u-gate and v-gate with the original input
        x = x_v * x_u + input  # Gated output with residual connection
        
        return x  # Return the final output tensor

import torch.nn as nn

class Gated_FSMN_Block_Dilated(nn.Module):
    """
    Gated FSMN (Finite State Machine Network) block with dilated convolutions.

    This module implements a Gated FSMN block that utilizes dilated convolutions 
    for feature extraction and gating mechanisms to enhance speech processing. 
    The architecture consists of convolutional layers followed by normalization 
    and a gated FSMN for robust feature extraction.

    Attributes:
        group_size (int): Size of the groups for normalization.
        conv1 (nn.Sequential): Initial 1D convolutional layer followed by 
                               PReLU activation.
        norm1 (CLayerNorm): First normalization layer.
        gated_fsmn (Gated_FSMN_dilated): Gated FSMN module for processing.
        norm2 (CLayerNorm): Second normalization layer.
        conv2 (nn.Conv1d): Final 1D convolutional layer to map features back 
                           to the original dimension.
    """

    def __init__(self,
                 dim: int,
                 inner_channels: int = 256,
                 group_size: int = 256, 
                 norm_type: str = 'scalenorm',
                 ):
        """
        Initializes the Gated_FSMN_Block_Dilated module.
        
        Args:
            dim (int): The number of input channels (features).
            inner_channels (int): The number of channels in the inner layers.
            group_size (int): Size of the groups for normalization.
            norm_type (str): Type of normalization to use ('scalenorm' or 'layernorm').
        """
        super(Gated_FSMN_Block_Dilated, self).__init__()

        # Set normalization class based on the specified type
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm  # Use ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm  # Use LayerNorm

        self.group_size = group_size

        # Initial convolution layer with PReLU activation
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        
        self.norm1 = CLayerNorm(inner_channels)  # First normalization layer
        
        # Gated FSMN block with dilated convolutions
        self.gated_fsmn = Gated_FSMN_dilated(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        
        self.norm2 = CLayerNorm(inner_channels)  # Second normalization layer
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)  # Output convolution layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Gated FSMN block.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_length, dim).
        
        Returns:
            torch.Tensor: Output tensor after processing through the Gated FSMN block.
        """
        # Apply the first convolution and transpose for correct dimensions
        conv1 = self.conv1(input.transpose(2, 1))
        norm1 = self.norm1(conv1)  # Apply normalization after first convolution
        
        # Process through the gated FSMN and transpose back to original dimensions
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))
        norm2 = self.norm2(seq_out.transpose(2, 1))  # Apply normalization after gated FSMN
        
        # Apply the second convolution and return the residual connection
        conv2 = self.conv2(norm2)  # Final convolution
        return conv2.transpose(2, 1) + input  # Return output with residual connection


class MossformerBlock_GFSMN(nn.Module):
    """
    Mossformer2 Block with Gated FSMN: A module that integrates single-head gated attention mechanisms 
    with Gated Feedforward Sequential Memory Networks (FSMNs) to enhance feature representation 
    in sequential data. This block employs multiple layers of attention and gated mechanisms 
    for improved learning capabilities.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of layers to stack in the block.
        group_size (int, optional): Size of the groups for sequential processing. Defaults to 256.
        query_key_dim (int, optional): Dimension for query and key projections in attention. Defaults to 128.
        expansion_factor (float, optional): Factor to expand dimensions in the feedforward layers. Defaults to 4.0.
        causal (bool, optional): Whether to apply causal masking in attention. Defaults to False.
        attn_dropout (float, optional): Dropout rate for attention layers. Defaults to 0.1.
        norm_type (str, optional): Type of normalization to use ('scalenorm' or 'layernorm'). Defaults to 'scalenorm'.
        shift_tokens (bool, optional): Whether to apply token shifting. Defaults to True.
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        causal=False,
        attn_dropout=0.1,
        norm_type='scalenorm',
        shift_tokens=True
    ):
        super().__init__()

        # Assert valid normalization type
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        # Choose normalization class based on the specified type
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # Initialize rotary positional embeddings with a maximum dimension of 32
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))

        # Create a list of Gated FSMN blocks for each layer
        self.fsmn = nn.ModuleList([Gated_FSMN_Block_Dilated(dim) for _ in range(depth)])

        # Create a list of FLASH attention layers
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens
            ) for _ in range(depth)
        ])

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        """
        Build a sequential block of UniDeep FSMNs.

        Args:
            in_channels (int): Number of input channels for the FSMN.
            out_channels (int): Number of output channels for the FSMN.
            lorder (int): Order for the FSMN.
            hidden_size (int): Hidden size for the FSMN.
            repeats (int, optional): Number of repetitions of the FSMN block. Defaults to 1.

        Returns:
            nn.Sequential: A sequential module containing the specified number of UniDeep FSMNs.
        """
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(
        self,
        x,
        *,
        mask=None
    ):
        """
        Forward pass through the Mossformer Block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim).
            mask (Tensor, optional): Attention mask to apply. Defaults to None.

        Returns:
            Tensor: Output tensor after passing through all layers, of shape (batch_size, seq_len, dim).
        """
        ii = 0
        # Iterate through all FLASH attention layers and Gated FSMN blocks
        for flash in self.layers:
            x = flash(x, mask=mask)  # Apply FLASH attention layer
            x = self.fsmn[ii](x)     # Apply corresponding Gated FSMN block
            ii += 1  # Increment index for the Gated FSMN block

        return x  # Return the final output after all layers

class MossformerBlock(nn.Module):
    """
    Mossformer Block: A module that employs a series of signle-head gated attention layers to process 
    sequential data. This block is designed for flexibility in feature dimension, depth, 
    and normalization techniques, making it suitable for various tasks in deep learning.

    Args:
        dim (int): Number of input channels (features).
        depth (int): Number of layers in the block.
        group_size (int, optional): Size of the groups for processing. Defaults to 256.
        query_key_dim (int, optional): Dimension for query and key projections in attention. Defaults to 128.
        expansion_factor (float, optional): Factor to expand the dimensionality in feedforward layers. Defaults to 4.0.
        causal (bool, optional): Whether to apply causal masking in attention. Defaults to False.
        attn_dropout (float, optional): Dropout rate applied to attention layers. Defaults to 0.1.
        norm_type (str, optional): Type of normalization to apply ('scalenorm' or 'layernorm'). Defaults to 'scalenorm'.
        shift_tokens (bool, optional): Whether to apply token shifting. Defaults to True.
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        causal=False,
        attn_dropout=0.1,
        norm_type='scalenorm',
        shift_tokens=True
    ):
        super().__init__()

        # Assert valid normalization type
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        # Select normalization class based on the specified type
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # Initialize rotary positional embeddings, limiting max dimension to 32
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))
        
        # Create a list of FLASH attention layers for the specified depth
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens
            ) for _ in range(depth)
        ])

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        """
        Build a sequential block of UniDeep FSMNs.

        Args:
            in_channels (int): Number of input channels for the FSMN.
            out_channels (int): Number of output channels for the FSMN.
            lorder (int): Order for the FSMN.
            hidden_size (int): Hidden size for the FSMN.
            repeats (int, optional): Number of repetitions of the FSMN block. Defaults to 1.

        Returns:
            nn.Sequential: A sequential module containing the specified number of UniDeep FSMNs.
        """
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(
        self,
        x,
        *,
        mask=None
    ):
        """
        Forward pass through the Mossformer Block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim).
            mask (Tensor, optional): Attention mask to apply. Defaults to None.

        Returns:
            Tensor: Output tensor after passing through all layers, of shape (batch_size, seq_len, dim).
        """
        ii = 0
        # Iterate through all FLASH attention layers and apply them to the input
        for flash in self.layers:
            x = flash(x, mask=mask)  # Apply FLASH attention layer
            ii += 1  # Increment layer index

        return x  # Return the final output after processing through all layers

