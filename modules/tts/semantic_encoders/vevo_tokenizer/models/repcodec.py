from __future__ import annotations

import torch
import torch.nn as nn

from .residual_vq import ResidualVQ
from .vocos_backbone import VocosBackbone


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv1d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        nn.init.constant_(module.bias, 0)


class RepCodec(nn.Module):
    def __init__(
        self,
        codebook_size: int = 8192,
        hidden_size: int = 1024,
        codebook_dim: int = 8,
        vocos_dim: int = 384,
        vocos_intermediate_dim: int = 2048,
        vocos_num_layers: int = 12,
        num_quantizers: int = 1,
        downsample_scale: int = 1,
        cfg=None,
    ):
        super().__init__()
        codebook_size = getattr(cfg, "codebook_size", codebook_size)
        codebook_dim = getattr(cfg, "codebook_dim", codebook_dim)
        hidden_size = getattr(cfg, "hidden_size", hidden_size)
        vocos_dim = getattr(cfg, "vocos_dim", vocos_dim)
        vocos_intermediate_dim = getattr(
            cfg,
            "vocos_intermediate_dim",
            vocos_intermediate_dim,
        )
        vocos_num_layers = getattr(cfg, "vocos_num_layers", vocos_num_layers)
        num_quantizers = getattr(cfg, "num_quantizers", num_quantizers)
        downsample_scale = getattr(cfg, "downsample_scale", downsample_scale)

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.num_quantizers = num_quantizers
        self.downsample_scale = downsample_scale

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )

        self.quantizer = ResidualVQ(
            input_dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_type="fvq",
            quantizer_dropout=0.0,
            commitment=0.15,
            codebook_loss_weight=1.0,
            use_l2_normlize=True,
        )

        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            _,
        ) = self.quantizer(x)
        x_rec = self.decoder(quantized_out)
        codebook_loss = (all_codebook_losses + all_commit_losses).mean()
        return x_rec, codebook_loss, all_indices

    def quantize(self, x: torch.Tensor):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        (
            quantized_out,
            all_indices,
            _all_commit_losses,
            _all_codebook_losses,
            _,
        ) = self.quantizer(x)
        if all_indices.shape[0] == 1:
            return all_indices.squeeze(0), quantized_out.transpose(1, 2)
        return all_indices, quantized_out.transpose(1, 2)

    def reset_parameters(self) -> None:
        self.apply(_init_weights)
