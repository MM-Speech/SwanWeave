from dataclasses import dataclass, field
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from modules.commons.hf.transformer import TransformerDecoderModel
from modules.commons.hf.transformer_config import TransformerConfig

from utils.nn.seq_utils import sequence_mask
from utils.commons.tensor_utils import all_gather_varlen_tensor_stack, all_gather_varlen_tensor

def build_nepa_model(hparams, attn_implementation='flash_attention_2'):
    config = ModelArgs(
        sample_rate=hparams['audio_sample_rate'],
        input_channels=hparams.get('audio_input_channels', 1),
        strides=hparams.get('nepa_strides', (10, 8, 8)),
        encoder_dims=hparams.get('nepa_encoder_dims', (64, 256, 1024)),
        hidden_size=hparams.get('nepa_hidden_size', 1024),
        num_hidden_layers=hparams.get('nepa_num_hidden_layers', 24),
        num_attention_heads=hparams.get('nepa_num_attention_heads', 16),
        num_key_value_heads=hparams.get('nepa_num_key_value_heads', 8),

        encoder_mode=hparams.get('nepa_encoder_mode', 'nonoverlap'),
        encoder_kernels=hparams.get('nepa_encoder_kernels', None),
        encoder_norm=hparams.get('nepa_encoder_norm', 'rmsnorm'),
        encoder_norm_eps=hparams.get('nepa_encoder_norm_eps', 1e-6),
        encoder_pad_mode=hparams.get('nepa_encoder_pad_mode', 'replicate'),
        pred_steps=tuple(hparams.get('nepa_pred_steps', (1,))),
        pred_step_reduction=hparams.get('nepa_pred_step_reduction', 'mean'),

        attn_implementation=attn_implementation,
        gradient_checkpointing=hparams.get('gradient_checkpointing', False),

        loss_type=hparams.get('nepa_loss_type', 'cos'),
        infonce_neighbor_window=hparams.get('nepa_infonce_neighbor_window', 0),
    )
    return NepaModel(config)


@dataclass
class ModelArgs:
    # audio
    sample_rate: int = 16000
    input_channels: int = 1

    # encoder
    patch_size: int = field(init=False)
    hop_size: int = field(init=False)
    strides: tuple = (10, 8, 8)
    encoder_dims: tuple = (64, 256, 1024)

    # encoder behavior
    encoder_mode: str = "nonoverlap"   # "nonoverlap" | "overlap_causal"
    encoder_kernels: Union[tuple, None] = None   # e.g. (20,16,16) when overlap_causal
    encoder_norm: str = "rmsnorm"      # "rmsnorm" | "layernorm" | "groupnorm1"(legacy)
    encoder_norm_eps: float = 1e-6
    encoder_pad_mode: str = "replicate"  # "replicate" | "constant" | "learned"

    # decoder
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8

    # training
    loss_type: str = 'cos'  # 'cos' | 'infonce'
    infonce_neighbor_window: int = 0
    attn_implementation: str = 'flash_attention_2'
    gradient_checkpointing: bool = False

    # NEW: multi-step prediction
    pred_steps: tuple = (1,)           # e.g. (1,2,4,8)
    pred_step_reduction: str = "mean"  # "mean" | "sum"

    def __post_init__(self):
        self.patch_size = np.prod(self.strides)
        self.hop_size = self.patch_size


class ChannelRMSNorm(nn.Module):
    """RMSNorm on last dim (channel), per time step: [B,T,C]."""
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        if self.weight is not None:
            x = x * self.weight
        return x


class ChannelNorm1d(nn.Module):
    """Apply LN/RMSNorm per time step on conv output [B,C,T]."""
    def __init__(self, dim: int, norm_type: str = "rmsnorm", eps: float = 1e-6):
        super().__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        if norm_type == "rmsnorm":
            self.norm = ChannelRMSNorm(dim, eps=eps, affine=True)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        elif norm_type == "groupnorm1":
            # legacy: not recommended for strict causal consistency
            self.norm = nn.GroupNorm(1, dim, eps=eps, affine=True)
        else:
            raise ValueError(f"Unknown encoder_norm: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        if self.norm_type == "groupnorm1":
            return self.norm(x)
        x = x.transpose(1, 2)     # [B,T,C]
        x = self.norm(x)
        x = x.transpose(1, 2)     # [B,C,T]
        return x


class CausalStridedConv1d(nn.Module):
    """
    Causal strided conv: left pad only, no future leakage.
    Works for k>s (overlap) or k==s.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
        pad_mode: str = "replicate",   # "replicate" | "constant" | "learned"
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=self.kernel_size, stride=self.stride, padding=0, bias=bias)

        pad_left = self.kernel_size - 1
        if pad_mode == "learned":
            # learned "start context", no future leakage, reduces boundary artifact vs zeros
            self.left_pad = nn.Parameter(torch.zeros(1, in_ch, pad_left))
        else:
            self.left_pad = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        pad_left = self.kernel_size - 1
        if pad_left > 0:
            if self.pad_mode == "learned":
                pad = self.left_pad.expand(x.size(0), -1, -1)
                x = torch.cat([pad, x], dim=2)
            elif self.pad_mode == "replicate":
                x = F.pad(x, (pad_left, 0), mode="replicate")
            else:
                x = F.pad(x, (pad_left, 0), mode="constant", value=0.0)
        return self.conv(x)


class NepaEncoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        layers = []
        kernels = config.encoder_kernels if config.encoder_kernels is not None else config.strides
        assert len(kernels) == len(config.strides), "encoder_kernels must match strides length"
        assert len(config.encoder_dims) == len(config.strides), "encoder_dims must match strides length"

        in_ch = config.input_channels
        for i, (stride, ksz, out_ch) in enumerate(zip(config.strides, kernels, config.encoder_dims)):
            if config.encoder_mode == "overlap_causal":
                conv = CausalStridedConv1d(
                    in_ch, out_ch,
                    kernel_size=int(ksz),
                    stride=int(stride),
                    bias=True,
                    pad_mode=config.encoder_pad_mode,
                )
            else:
                # legacy non-overlap hard patchify (your current behavior)
                # NOTE: still "causal" in the sense that output frame i can be emitted
                # only after observing its whole block [i*s, i*s+s-1].
                conv = nn.Conv1d(in_ch, out_ch, kernel_size=int(stride), stride=int(stride), padding=0, bias=True)

            layers.append(conv)
            layers.append(ChannelNorm1d(out_ch, norm_type=config.encoder_norm, eps=config.encoder_norm_eps))
            layers.append(nn.SiLU())
            in_ch = out_ch

        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(config.encoder_dims[-1], config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, T, C]
        x = x.transpose(1, 2)              # [B, C, T]
        x = self.layers(x)                 # [B, C, T']
        x = x.transpose(1, 2)              # [B, T', C]
        x = self.out(x)                    # [B, T', hidden]
        return x
    

class NepaModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.encoder = NepaEncoder(config)
        self.decoder = TransformerDecoderModel(TransformerConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * 4,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_act="silu",
            use_gated_attention=True,
            use_cache=False,
            attn_implementation=config.attn_implementation,
        ))

        if self.config.loss_type == 'infonce':
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.config.gradient_checkpointing and self.training:
            self.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict = None):
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    @staticmethod
    def _downsample_lens(wav_lens: torch.Tensor, strides: tuple, mode: str) -> torch.Tensor:
        lens = wav_lens
        for s in strides:
            if mode == "overlap_causal":
                # ceil-div
                lens = (lens + s - 1) // s
            else:
                # floor-div (your current non-overlap conv)
                lens = lens // s
        return lens

    def forward(self, wavs, wav_lens=None, training=False):
        # wavs [B, T] or [B, T, C]
        if len(wavs.shape) == 2:
            wavs = wavs[..., None]

        if wav_lens is None:
            wav_lens = torch.full((wavs.shape[0],), wavs.shape[1], device=wavs.device, dtype=torch.long)
        else:
            wav_lens = wav_lens.to(device=wavs.device, dtype=torch.long)

        x = self.encoder(wavs)  # [B, T', C]

        feat_lens = self._downsample_lens(wav_lens, self.config.strides, self.config.encoder_mode)
        attention_mask = sequence_mask(feat_lens, maxlen=x.shape[1])  # [B, T'] (bool)

        model_outputs = self.decoder.forward(
            inputs_embeds=x, attention_mask=attention_mask
        ).last_hidden_state  # [B, T', C]

        if not training:
            return model_outputs, attention_mask

        target_all = x.detach()  # [B, T', C]
        steps = tuple(sorted(set(int(s) for s in self.config.pred_steps if int(s) > 0)))
        if len(steps) == 0:
            steps = (1,)

        # ---------------- COS LOSS (unchanged) ----------------
        if self.config.loss_type == "cos":
            losses = []
            for k in steps:
                if model_outputs.size(1) <= k:
                    continue
                pred = model_outputs[:, :-k]         # [B, T'-k, C]
                target = target_all[:, k:]           # [B, T'-k, C]
                loss_mask = attention_mask[:, k:]    # [B, T'-k]
                if loss_mask.any():
                    cos = F.cosine_similarity(pred, target, dim=-1)  # [B, T'-k]
                    m = loss_mask.float()
                    loss_k = (cos * m).sum() / (m.sum() + 1e-8)
                    losses.append(1.0 - loss_k)

            if len(losses) == 0:
                loss = model_outputs.new_tensor(0.0)
            else:
                loss = sum(losses) / len(losses) if self.config.pred_step_reduction == "mean" else sum(losses)

            return {
                "hidden_states": model_outputs,
                "ntokens": attention_mask.sum(),
                "loss": loss,
            }

        # ---------------- INFONCE LOSS (gather once) ----------------
        elif self.config.loss_type == "infonce":
            infonce_neighbor_window = int(getattr(self.config, "infonce_neighbor_window", 0))
            logit_scale = self.logit_scale.clamp(max=np.log(100.0)).exp()

            B, Tp, C = target_all.shape
            device = target_all.device

            # ---- build shared local key bank once: all valid tokens by attention_mask ----
            key_b, key_t = torch.where(attention_mask)  # [N_key_local]

            key_local = target_all[key_b, key_t]                 # [N_key_local, C]
            key_local = F.normalize(key_local, dim=-1)           # [N_key_local, C]
            key_time_local = key_t                               # [N_key_local] (time index)

            # map (b,t) -> local key index, -1 if invalid
            key_index_map = torch.full((B, Tp), -1, device=device, dtype=torch.long)
            key_index_map[key_b, key_t] = torch.arange(key_b.numel(), device=device, dtype=torch.long)

            if dist.is_available() and dist.is_initialized():
                # gather keys and key_time ONCE
                all_keys = all_gather_varlen_tensor(key_local, dim=0)          # [N_total, C]
                all_key_time = all_gather_varlen_tensor(key_time_local, dim=0) # [N_total]

                world_size = dist.get_world_size()
                rank = dist.get_rank()
                local_len = torch.tensor([key_local.size(0)], device=device, dtype=torch.long)
                lens_list = [torch.zeros_like(local_len) for _ in range(world_size)]
                dist.all_gather(lens_list, local_len)
                lens = torch.cat(lens_list).to(device)
                offset = int(lens[:rank].sum().item())
            else:
                all_keys = key_local
                all_key_time = key_time_local
                offset = 0

            losses = []

            # precompute local slice for neighbor masking
            local_cols = slice(offset, offset + key_local.size(0))
            all_keys_t = all_keys.t()

            if infonce_neighbor_window > 0:
                w = infonce_neighbor_window
                neighbor_offsets = torch.arange(-w, w + 1, device=device)
                neighbor_offsets = neighbor_offsets[neighbor_offsets != 0]  # [2w]
            else:
                neighbor_offsets = None

            for k in steps:
                if model_outputs.size(1) <= k:
                    continue

                # queries are at time t (0..Tp-k-1) where both t and t+k are valid
                pred_mask = attention_mask[:, :-k]   # [B, Tp-k]
                pos_mask = attention_mask[:, k:]     # [B, Tp-k]
                valid = pred_mask & pos_mask

                qb, qt = torch.where(valid)
                if qb.numel() == 0:
                    continue

                # queries: h_t
                q = model_outputs[qb, qt]            # [Nq, C]
                q = F.normalize(q, dim=-1)

                # positive key is at (b, t+k) in the shared key bank
                pos_local = key_index_map[qb, qt + k]  # [Nq]
                ok = pos_local >= 0
                if not ok.all():
                    qb = qb[ok]
                    qt = qt[ok]
                    q = q[ok]
                    pos_local = pos_local[ok]
                    if q.numel() == 0:
                        continue

                labels = pos_local + offset          # [Nq]

                logits = logit_scale * (q @ all_keys_t)  # [Nq, N_total]

                # mimic old per-k key bank (mask keys with time < k)
                if all_key_time is not None:
                    bad_cols = all_key_time < k
                    logits[:, bad_cols] = float("-inf")

                # neighbor window masking on LOCAL columns only
                if neighbor_offsets is not None:
                    tt = qt + k  # [Nq]
                    bb = qb      # [Nq]

                    neigh_t = tt.unsqueeze(1) + neighbor_offsets.unsqueeze(0)  # [Nq, 2w]
                    valid_t = (neigh_t >= 0) & (neigh_t < Tp)
                    neigh_t_clamped = neigh_t.clamp(0, Tp - 1)

                    bb_exp = bb.unsqueeze(1).expand_as(neigh_t_clamped)
                    neigh_key_idx = key_index_map[bb_exp, neigh_t_clamped]   # [Nq, 2w]

                    valid_nb = valid_t & (neigh_key_idx >= 0)
                    if valid_nb.any():
                        logits_local = logits[:, local_cols]  # [Nq, N_key_local]

                        rows = torch.arange(q.size(0), device=device).unsqueeze(1).expand_as(neigh_key_idx)
                        rows = rows[valid_nb]                 # [N_edges]
                        cols = neigh_key_idx[valid_nb]        # [N_edges] local key indices

                        logits_local[rows, cols] = float("-inf")
                        logits[:, local_cols] = logits_local

                loss_k = F.cross_entropy(logits, labels)
                losses.append(loss_k)

            if len(losses) == 0:
                loss = model_outputs.new_tensor(0.0)
            else:
                loss = sum(losses) / len(losses) if self.config.pred_step_reduction == "mean" else sum(losses)

            return {
                "hidden_states": model_outputs,
                "ntokens": attention_mask.sum(),
                "loss": loss,
            }

        else:
            raise ValueError(f"Unknown loss_type: {self.config.loss_type}")


if __name__ == '__main__':
    import librosa
    import os
    from utils.commons.hparams import set_hparams
    from utils.commons.ckpt_utils import load_ckpt, get_all_ckpt_steps
    import matplotlib.pyplot as plt

    # ckpt = 'checkpoints/260122_nepa'
    # ckpt = 'checkpoints/260124_nepa'
    ckpt = 'checkpoints/260213_nepa_base'

    hparams = set_hparams(f"{ckpt}/config.yaml", global_hparams=False)
    model = build_nepa_model(hparams).cuda()
    model.eval()
    load_ckpt(model, ckpt, 'model')

    wav, _ = librosa.load('user/prompts/dzq_enhanced.wav', sr=16000)
    wav = torch.from_numpy(wav)[None, :].to('cuda').to(torch.float16)
    wav = wav[:, :wav.shape[1] // model.config.hop_size * model.config.hop_size]

    with torch.no_grad(), torch.autocast('cuda', torch.float16):
        model_outputs = model(wav)

    print(f"{model_outputs = }")

    feat = model_outputs[0].detach().cpu().numpy()  # [T, C]
    # 可选：为了可视化更清晰，可以做一下简单标准化（按特征维维度）
    # feat = (feat - feat.mean(axis=0, keepdims=True)) / (feat.std(axis=0, keepdims=True) + 1e-6)
    plt.figure(figsize=(10, 4))
    # imshow 的输入是 [H, W]，这里希望竖直方向是 feature_dim，所以转置一下
    plt.imshow(feat.T, aspect='auto', origin='lower', interpolation='nearest')
    plt.colorbar(label="Feature value")
    plt.xlabel("Time steps")
    plt.ylabel("Feature dimension")
    plt.title("Nepa semantic features (hidden states)")
    plt.tight_layout()
    # save_path = 'user/temp/nepa_feat.png'
    save_path = f'infer_out/asr/nepa/{os.path.basename(ckpt)}/step{get_all_ckpt_steps(ckpt)[-1]}'
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    plt.savefig(save_path, dpi=150)
    plt.close()
