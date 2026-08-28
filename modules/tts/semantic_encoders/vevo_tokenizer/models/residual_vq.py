from __future__ import annotations

import torch
import torch.nn as nn

from .factorized_vector_quantize import FactorizedVectorQuantize


class ResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        quantizer_type: str = "fvq",
        quantizer_dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        if quantizer_type != "fvq":
            raise ValueError(f"unsupported quantizer type for minimal runtime: {quantizer_type}")

        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.quantizers = nn.ModuleList(
            [
                FactorizedVectorQuantize(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z: torch.Tensor, n_quantizers: int | None = None):
        quantized_out = 0.0
        residual = z
        all_commit_losses = []
        all_codebook_losses = []
        all_indices = []
        all_quantized = []

        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        for index, quantizer in enumerate(self.quantizers):
            if not self.training and index >= n_quantizers:
                break

            z_q_i, commit_loss_i, codebook_loss_i, indices_i, _ = quantizer(residual)
            quantized_out = quantized_out + z_q_i
            residual = residual - z_q_i
            all_commit_losses.append(commit_loss_i.mean())
            all_codebook_losses.append(codebook_loss_i.mean())
            all_indices.append(indices_i)
            all_quantized.append(z_q_i)

        all_commit_losses, all_codebook_losses, all_indices, all_quantized = map(
            torch.stack,
            (all_commit_losses, all_codebook_losses, all_indices, all_quantized),
        )

        return (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            all_quantized,
        )
