import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from .conv_module import ConvModule

# Helper functions

def exists(val):
    """
    Check if a value is not None.
    
    Args:
        val: The value to check.
        
    Returns:
        bool: True if the value exists (is not None), False otherwise.
    """
    return val is not None

def default(val, d):
    """
    Return the value if it exists, otherwise return a default value.
    
    Args:
        val: The value to check.
        d: The default value to return if val is None.
        
    Returns:
        The original value or the default value.
    """
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    """
    Calculate padding to make a number a multiple of another number.
    
    Args:
        n (int): The number to pad.
        mult (int): The multiple to pad to.
    
    Returns:
        int: The padding value.
    """
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# ScaleNorm
class ScaleNorm(nn.Module):
    """
    Normalization layer that scales inputs based on the dimensionality of the input.
    
    Args:
        dim (int): The input dimension.
        eps (float): A small value to prevent division by zero (default: 1e-5).
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5  # Scale factor based on input dimension
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))  # Learnable scale parameter

    def forward(self, x):
        # Normalize the input along the last dimension and apply scaling
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g

# Absolute positional encodings
class ScaledSinuEmbedding(nn.Module):
    """
    Sine-cosine absolute positional embeddings with scaling.
    
    Args:
        dim (int): The dimension of the positional embedding.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)  # Store frequency values for sine and cosine

    def forward(self, x):
        # Generate sine and cosine positional encodings
        n, device = x.shape[1], x.device
        t = torch.arange(n, device=device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        return emb * self.scale  # Apply scaling to the positional embeddings

# T5 relative positional bias
class T5RelativePositionBias(nn.Module):
    """
    Relative positional bias based on T5 model design.
    
    Args:
        scale (float): Scaling factor for the bias.
        causal (bool): Whether to apply a causal mask (default: False).
        num_buckets (int): Number of relative position buckets (default: 32).
        max_distance (int): Maximum distance for relative positions (default: 128).
    """
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128):
        super().__init__()
        self.eps = 1e-5
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)  # Bias embedding for relative positions

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        """
        Bucket relative positions into discrete ranges for bias calculation.
        
        Args:
            relative_position (Tensor): The relative position tensor.
            causal (bool): Whether to consider causality.
            num_buckets (int): Number of relative position buckets.
            max_distance (int): Maximum distance for the position.

        Returns:
            Tensor: Bucketed relative positions.
        """
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        # Calculate relative position bias for attention
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)  # Get bias values
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale  # Apply scaling to the bias

# Relative Position Embeddings
class RelativePosition(nn.Module):
    """
    Relative positional embeddings with configurable number of units and max position.
    
    Args:
        num_units (int): The number of embedding units (default: 32).
        max_relative_position (int): The maximum relative position (default: 128).
    """
    def __init__(self, num_units=32, max_relative_position=128):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)  # Initialize embedding weights

    def forward(self, x):
        # Generate relative position embeddings
        length_q, length_k, device = *x.shape[-2:], x.device
        range_vec_q = torch.arange(length_q, dtype=torch.long, device=device)
        range_vec_k = torch.arange(length_k, dtype=torch.long, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]  # Compute relative distances
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = (distance_mat_clipped + self.max_relative_position)
        embeddings = self.embeddings_table[final_mat]  # Get embeddings based on distances

        return embeddings

# Offset and Scale module
class OffsetScale(nn.Module):
    """
    Offset and scale operation applied across heads and dimensions.
    
    Args:
        dim (int): Input dimensionality.
        heads (int): Number of attention heads (default: 1).
    """
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))  # Learnable scaling parameter
        self.beta = nn.Parameter(torch.zeros(heads, dim))  # Learnable offset parameter
        nn.init.normal_(self.gamma, std=0.02)  # Initialize gamma with small random values

    def forward(self, x):
        # Apply offset and scale across heads
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)  # Return the result unbound along the last head dimension

class FFConvM(nn.Module):
    """
    FFConvM is a feedforward convolutional module that applies a series of transformations
    to an input tensor. The transformations include normalization, linear projection,
    activation, convolution, and dropout. It combines feedforward layers with a convolutional
    module to enhance the feature extraction process.

    Args:
        dim_in: Input feature dimension.
        dim_out: Output feature dimension.
        norm_klass: Normalization class to apply (default is LayerNorm).
        dropout: Dropout probability to prevent overfitting (default is 0.1).
    """
    def __init__(
        self,
        dim_in,    # Input feature dimension
        dim_out,   # Output feature dimension
        norm_klass=nn.LayerNorm,  # Normalization class (default: LayerNorm)
        dropout=0.1  # Dropout probability
    ):
        super().__init__()

        # Sequentially apply normalization, linear transformation, activation, convolution, and dropout
        self.mdl = nn.Sequential(
            norm_klass(dim_in),  # Apply normalization (LayerNorm by default)
            nn.Linear(dim_in, dim_out),  # Linear projection from dim_in to dim_out
            nn.SiLU(),  # Activation function (SiLU - Sigmoid Linear Unit)
            ConvModule(dim_out),  # Apply convolution using ConvModule
            nn.Dropout(dropout)  # Apply dropout for regularization
        )

    def forward(self, x):
        """
        Forward pass through the module.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, dim_in)
        
        Returns:
            output: Transformed output tensor of shape (batch_size, seq_length, dim_out)
        """
        output = self.mdl(x)  # Pass the input through the sequential model
        return output  # Return the processed output

class MossFormer(nn.Module):
    """
    The MossFormer class implements a transformer-based model designed for handling
    triple-attention mechanisms with both quadratic and linear attention components. 
    The model processes inputs through token shifts, multi-head attention, and gated 
    feedforward layers, while optionally supporting causal operations.
    
    Args:
        dim (int): Dimensionality of input features.
        group_size (int): Size of the group dimension for attention.
        query_key_dim (int): Dimensionality of the query and key vectors for attention.
        expansion_factor (float): Expansion factor for the hidden dimensions.
        causal (bool): Whether to apply causal masking for autoregressive tasks.
        dropout (float): Dropout rate for regularization.
        norm_klass (nn.Module): Normalization layer to be applied.
        shift_tokens (bool): Whether to apply token shifting as a preprocessing step.
    """
    def __init__(
        self,
        dim,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 4.,
        causal = False,
        dropout = 0.1,
        norm_klass = nn.LayerNorm,
        shift_tokens = True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)        
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # Positional embeddings for attention.
        self.rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # Dropout layer for regularization.
        self.dropout = nn.Dropout(dropout)
        
        # Projection layers for input features to hidden dimensions.
        self.to_hidden = FFConvM(
            dim_in = dim,
            dim_out = hidden_dim,
            norm_klass = norm_klass,
            dropout = dropout,
        )
        self.to_qk = FFConvM(
            dim_in = dim,
            dim_out = query_key_dim,
            norm_klass = norm_klass,
            dropout = dropout,
        )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)

        # Output projection layer to return to original feature dimensions.
        self.to_out = FFConvM(
            dim_in = dim * int(expansion_factor // 2),
            dim_out = dim,
            norm_klass = norm_klass,
            dropout = dropout,
        )
        
        self.gateActivate = nn.Sigmoid() 

    def forward(
        self,
        x,
        *,
        mask = None
    ):
        """
        Forward pass for the MossFormer module.

        Args:
            x (Tensor): Input tensor of shape (B, T, Q, C) where:
                B = batch size,
                T = total sequence length,
                Q = number of query features,
                C = feature dimension.
            mask (Tensor, optional): Attention mask for padding.

        Returns:
            Tensor: Output tensor of shape (B, T, C).
        """
        
        # Unpack input dimensions
        B, T, Q, C = x.size()
        x = x.view(B*T, Q, C)  # Reshape input for processing

        # Prenormalization step
        normed_x = x 

        # Optionally shift tokens for better performance
        residual = x  # Store residual for skip connection

        if self.shift_tokens:
            # Split and shift tokens for enhanced information flow
            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)  # Pad to maintain shape
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

        # Initial projections to hidden space
        v, u = self.to_hidden(normed_x).chunk(2, dim = -1)  # Split into two tensors
        qk = self.to_qk(normed_x)  # Project to query/key dimensions

        # Offset and scale for attention
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, B)

        # Gate the outputs and apply skip connection
        out = (att_u * v) * self.gateActivate(att_v * u)       
        x = x + self.to_out(out)  # Combine with residual
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, B, mask = None):
        """
        Calculates both quadratic and linear attention outputs.

        Args:
            x (Tensor): Input tensor of shape (B, n, d).
            quad_q (Tensor): Quadratic queries tensor.
            lin_q (Tensor): Linear queries tensor.
            quad_k (Tensor): Quadratic keys tensor.
            lin_k (Tensor): Linear keys tensor.
            v (Tensor): Value tensor for attention.
            u (Tensor): Auxiliary tensor for attention.
            B (int): Batch size.
            mask (Tensor, optional): Attention mask for padding.

        Returns:
            Tuple[Tensor, Tensor]: Quadratic and linear attention outputs.
        """
        
        b, n, device, g = x.shape[0], x.shape[-2],  x.device, self.group_size

        if exists(mask):
            # Apply mask to linear keys if provided
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # Rotate queries and keys using positional embeddings
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # Padding to handle groups
        padding = padding_to_multiple_of(n, n)

        if padding > 0:
            # Pad tensors to accommodate group sizes
            quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v, u))

            mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
            mask = F.pad(mask, (0, padding), value = False)

        # Reshape for grouped attention
        quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n = n), (quad_q, quad_k, lin_q, lin_k, v, u))

        BT, K, Q, C = quad_q.size()
        quad_q_c = quad_q.view(B, -1, Q, C).transpose(2, 1)  # Prepare for computation
        quad_k_c = quad_k.view(B, -1, Q, C).transpose(2, 1)  
        v_c = v.view(B, -1, Q, C).transpose(2, 1)  
        u_c = u.view(B, -1, Q, C).transpose(2, 1)  

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j = n)  # Adjust mask dimensions

        # Calculate quadratic attention output
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / n
        sim_c = einsum('... i d, ... j d -> ... i j', quad_q_c, quad_k_c) / quad_q_c.shape[-2]
        
        # Avoid introducing infinite loss probability
        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)  # Apply dropout for regularization
        
        attn_c = F.relu(sim_c) ** 2
        attn_c = self.dropout(attn_c)  # Apply dropout for the computed attention
        mask_c = torch.eye(quad_q_c.shape[-2], dtype = torch.bool, device = device)
        attn_c = attn_c.masked_fill(mask_c, 0.)  # Mask diagonal for attention

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)  # Apply the mask to the main attention

        if self.causal:
            # Create a causal mask for the attention
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)  # Apply causal mask

        # Calculate the output for quadratic attention
        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        # Calculate output for the causal quadratic attention
        quad_out_v_c = einsum('... i j, ... j d -> ... i d', attn_c, v_c)
        quad_out_u_c = einsum('... i j, ... j d -> ... i d', attn_c, u_c)
        quad_out_v_c = quad_out_v_c.transpose(2, 1).contiguous().view(BT, K, Q, C)
        quad_out_u_c = quad_out_u_c.transpose(2, 1).contiguous().view(BT, K, Q, C)

        # Combine the outputs from quadratic attention
        quad_out_v = quad_out_v + quad_out_v_c
        quad_out_u = quad_out_u + quad_out_u_c

        # Calculate linear attention output
        if self.causal:
            # Handle causal linear attention
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / n
            lin_kv = lin_kv.cumsum(dim = 1)  # Exclusive cumulative sum
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / n
            lin_ku = lin_ku.cumsum(dim = 1)  # Exclusive cumulative sum
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value = 0.)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
            # Handle non-causal linear attention
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        # Reshape and excise out padding
        quad_attn_out_v, lin_attn_out_v = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out_v, lin_out_v))
        quad_attn_out_u, lin_attn_out_u = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out_u, lin_out_u))

        return quad_attn_out_v + lin_attn_out_v, quad_attn_out_u + lin_attn_out_u     
