from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.8,
        commitment: float = 1.0,
        eps: float = 1e-5,
        n_embed: int | None = None,
    ):
        super().__init__()
        n_embed = codebook_size if n_embed is None else n_embed
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    @property
    def codebook(self):
        return self.embed.transpose(0, 1)

    def _ema_inplace(self, moving_avg: torch.Tensor, new: torch.Tensor) -> None:
        moving_avg.data.mul_(self.decay).add_(new, alpha=(1 - self.decay))

    def _laplace_smoothing(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.eps) / (x.sum() + self.n_embed * self.eps)

    def forward(self, input_tensor: torch.Tensor):
        dtype = input_tensor.dtype
        flatten = input_tensor.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input_tensor.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:
            self._ema_inplace(self.cluster_size, embed_onehot.sum(0))
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self._ema_inplace(self.embed_avg, embed_sum)
            cluster_size = self._laplace_smoothing(self.cluster_size) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        loss = F.mse_loss(quantize.detach(), input_tensor) * self.commitment
        quantize = input_tensor + (quantize - input_tensor).detach()

        avg_probs = torch.mean(embed_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantize, loss, perplexity

    def forward_index(self, input_tensor: torch.Tensor):
        flatten = input_tensor.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*input_tensor.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))
        quantize = input_tensor + (quantize - input_tensor).detach()
        return quantize, embed_ind


class ResidualVQ(nn.Module):
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])

    def forward(self, x: torch.Tensor):
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_perplexities = []
        for layer in self.layers:
            quantized, loss, perplexity = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_losses.append(loss)
            all_perplexities.append(perplexity)
        return quantized_out, torch.stack(all_losses), torch.stack(all_perplexities)

    def forward_index(self, x: torch.Tensor, flatten_idx: bool = False):
        quantized_out = 0.0
        residual = x
        all_indices = []
        for index, layer in enumerate(self.layers):
            quantized, indices = layer.forward_index(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            if flatten_idx:
                indices += self.codebook_size * index
            all_indices.append(indices)
        return quantized_out, torch.stack(all_indices)

    def initial(self) -> None:
        self.codebook = []
        for layer in self.layers:
            self.codebook.append(layer.codebook)
        self.codebook_size = self.codebook[0].size(0)
        self.codebook = torch.stack(self.codebook)
        self.codebook = self.codebook.reshape(-1, self.codebook.size(-1))

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        quantized_out = F.embedding(indices, self.codebook)
        return torch.sum(quantized_out, dim=0, keepdim=True)


class Quantizer(nn.Module):
    def __init__(self, code_dim: int, codebook_num: int, codebook_size: int):
        super().__init__()
        self.codebook = ResidualVQ(
            dim=code_dim,
            num_quantizers=codebook_num,
            codebook_size=codebook_size,
        )

    def initial(self) -> None:
        self.codebook.initial()

    def forward(self, z: torch.Tensor):
        zq, vqloss, perplexity = self.codebook(z.transpose(2, 1))
        return zq.transpose(2, 1), vqloss, perplexity

    def inference(self, z: torch.Tensor):
        zq, indices = self.codebook.forward_index(z.transpose(2, 1))
        return zq.transpose(2, 1), indices

    def encode(self, z: torch.Tensor):
        zq, indices = self.codebook.forward_index(z.transpose(2, 1), flatten_idx=True)
        return zq, indices

    def decode(self, indices: torch.Tensor):
        return self.codebook.lookup(indices)


class Conv1d1x1(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size=1, bias=bias)


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = -1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = -1,
        output_padding: int = -1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if padding < 0:
            padding = (stride + 1) // 2
        if output_padding < 0:
            output_padding = 1 if stride % 2 else 0
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = False,
        nonlinear_activation: str = "ELU",
        nonlinear_activation_params=None,
    ):
        super().__init__()
        nonlinear_activation_params = nonlinear_activation_params or {}
        self.activation = getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
        )
        self.conv2 = Conv1d1x1(out_channels, out_channels, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.activation(x))
        y = self.conv2(self.activation(y))
        return x + y


class Projector(nn.Module):
    def __init__(
        self,
        input_channels: int,
        code_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.project = Conv1d(
            input_channels,
            code_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations=(1, 1),
        unit_kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.res_units = nn.ModuleList(
            [
                ResidualUnit(
                    in_channels,
                    in_channels,
                    kernel_size=unit_kernel_size,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3 if stride == 1 else (2 * stride),
            stride=stride,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res_unit in self.res_units:
            x = res_unit(x)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        encode_channels: int,
        channel_ratios=(1, 1),
        strides=(1, 1),
        kernel_size: int = 3,
        bias: bool = True,
        block_dilations=(1, 1),
        unit_kernel_size: int = 3,
    ):
        super().__init__()
        if len(channel_ratios) != len(strides):
            raise ValueError("channel_ratios and strides must have the same length")

        self.conv = Conv1d(
            in_channels=input_channels,
            out_channels=encode_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=False,
        )
        self.conv_blocks = nn.ModuleList()
        current_channels = encode_channels
        for index, stride in enumerate(strides):
            out_channels = int(encode_channels * channel_ratios[index])
            self.conv_blocks.append(
                EncoderBlock(
                    current_channels,
                    out_channels,
                    stride,
                    dilations=block_dilations,
                    unit_kernel_size=unit_kernel_size,
                    bias=bias,
                )
            )
            current_channels = out_channels
        self.out_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        for block in self.conv_blocks:
            x = block(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations=(1, 1),
        unit_kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()

        if stride == 1:
            self.conv = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                bias=bias,
            )
        else:
            self.conv = ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(2 * stride),
                stride=stride,
                bias=bias,
            )
        self.res_units = nn.ModuleList(
            [
                ResidualUnit(
                    out_channels,
                    out_channels,
                    kernel_size=unit_kernel_size,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        for res_unit in self.res_units:
            x = res_unit(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        code_dim: int,
        output_channels: int,
        decode_channels: int,
        channel_ratios=(1, 1),
        strides=(1, 1),
        kernel_size: int = 3,
        bias: bool = True,
        block_dilations=(1, 1),
        unit_kernel_size: int = 3,
    ):
        super().__init__()
        if len(channel_ratios) != len(strides):
            raise ValueError("channel_ratios and strides must have the same length")

        self.conv1 = Conv1d(
            in_channels=code_dim,
            out_channels=int(decode_channels * channel_ratios[0]),
            kernel_size=kernel_size,
            stride=1,
            bias=False,
        )
        self.conv_blocks = nn.ModuleList()
        in_channels = int(decode_channels * channel_ratios[0])
        for index, stride in enumerate(strides):
            if index + 1 < len(channel_ratios):
                out_channels = int(decode_channels * channel_ratios[index + 1])
            else:
                out_channels = decode_channels
            self.conv_blocks.append(
                DecoderBlock(
                    in_channels,
                    out_channels,
                    stride,
                    dilations=block_dilations,
                    unit_kernel_size=unit_kernel_size,
                    bias=bias,
                )
            )
            in_channels = out_channels
        self.conv2 = Conv1d(
            in_channels=decode_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for block in self.conv_blocks:
            x = block(x)
        return self.conv2(x)


class VevoRepCodec(nn.Module):
    def __init__(
        self,
        input_channels: int = 768,
        output_channels: int = 768,
        encode_channels: int = 768,
        decode_channels: int = 768,
        code_dim: int = 768,
        codebook_num: int = 1,
        codebook_size: int = 1024,
        bias: bool = True,
        enc_ratios=(1, 1),
        dec_ratios=(1, 1),
        enc_strides=(1, 1),
        dec_strides=(1, 1),
        enc_kernel_size: int = 3,
        dec_kernel_size: int = 3,
        enc_block_dilations=(1, 1),
        enc_block_kernel_size: int = 3,
        dec_block_dilations=(1, 1),
        dec_block_kernel_size: int = 3,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_channels=input_channels,
            encode_channels=encode_channels,
            channel_ratios=enc_ratios,
            strides=enc_strides,
            kernel_size=enc_kernel_size,
            bias=bias,
            block_dilations=enc_block_dilations,
            unit_kernel_size=enc_block_kernel_size,
        )
        self.decoder = Decoder(
            code_dim=code_dim,
            output_channels=output_channels,
            decode_channels=decode_channels,
            channel_ratios=dec_ratios,
            strides=dec_strides,
            kernel_size=dec_kernel_size,
            bias=bias,
            block_dilations=dec_block_dilations,
            unit_kernel_size=dec_block_kernel_size,
        )
        self.projector = Projector(
            input_channels=self.encoder.out_channels,
            code_dim=code_dim,
            kernel_size=3,
            stride=1,
            bias=False,
        )
        self.quantizer = Quantizer(
            code_dim=code_dim,
            codebook_num=codebook_num,
            codebook_size=codebook_size,
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        z = self.projector(x)
        zq, vqloss, perplexity = self.quantizer(z)
        y = self.decoder(zq)
        return y, zq, z, vqloss, perplexity
