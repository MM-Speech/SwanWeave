import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import attrdictionary
import torchdiffeq

from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.commons.hf.transformer import TransformerEncoderModel
from modules.commons.hf.transformer_config import TransformerConfig
from modules.commons.hf.transformer_dit import TransformerDiTModel
from modules.commons.hf.transformer_dit_config import TransformerDiTConfig

from utils.nn.seq_utils import sequence_mask
from utils.commons.io import print_once

"""
Pure Mega3.2
"""

@dataclass
class ModelArgs:
    # llama
    encoder_dim: int = 1536
    encoder_n_layers: int = 36
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = 8

    in_channels: int = 32
    out_channels: int = 32
    
    attn_implementation: str = 'flash_attention_2'

    # trainging
    do_checkpoint: bool = False

class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp

        self.encoder = TransformerDiTModel(TransformerDiTConfig(
            hidden_size=hp.encoder_dim, intermediate_size=hp.encoder_dim * 4,
            num_hidden_layers=hp.encoder_n_layers, num_attention_heads=hp.encoder_n_heads,
            num_key_value_heads=hp.encoder_n_kv_heads, use_gated_attention=True,
            attn_implementation=hp.attn_implementation
        ))
        self.prenet = nn.Linear(self.hp.in_channels + 1, self.hp.encoder_dim)
        self.lat_proj = nn.Linear(self.hp.encoder_dim * 2, self.hp.encoder_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

        self.ph_encoder = TransformerEncoderModel(TransformerConfig(
            hidden_size=hp.encoder_dim, intermediate_size=hp.encoder_dim * 4,
            num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=8, 
            attn_implementation=hp.attn_implementation
        ))
        self.tone_embed = nn.Embedding(32, hp.encoder_dim, padding_idx=0)
        self.ph_embed = nn.Embedding(302, hp.encoder_dim)
        self.ling_pre_net = torch.nn.Sequential(*[
            torch.nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate([2, 2])
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        # Linear and Embedding layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.hp.encoder_n_layers))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.hp.encoder_n_layers)
                )
        # Time embedding MLP
        nn.init.normal_(self.f5_time_embed.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.f5_time_embed.time_mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks
        for block in self.encoder.layers:
            nn.init.zeros_(block.input_layernorm.linear.weight)
            nn.init.zeros_(block.input_layernorm.linear.bias)

        # Zero-out output layers
        nn.init.zeros_(self.encoder.norm.linear.weight)
        nn.init.zeros_(self.encoder.norm.linear.bias)
        

    def forward(self, inputs, sigmas=None, x_noisy=None):
        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask
        bsz, device = ctx_feature.size(0), ctx_feature.device

        x = inputs['lat']
        x_mask = sequence_mask(inputs['lat_lens'])
    
        # Here, x is x1 in CFM
        x0 = torch.randn_like(x)
        t = self.flow_matcher.time_sampler.sample([x0.shape[0]], x0.device).type_as(x0)
        xt = t[:, None, None] * x + (1 - t[:, None, None]) * x0
        ut = x - x0
        
        # define noisy_input and target
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            t = self.f5_time_embed(t)
        x_noisy = (xt * (1 - ctx_mask)).bfloat16() + ctx_feature # prefix ref wav
        target = ut

        x_txt = self.forward_ph_encoder(inputs)
            
        x_noisy = self.lat_proj(torch.cat([self.prenet(torch.cat([x_noisy, ctx_mask], -1)), x_txt], -1))
        
        self.encoder.gradient_checkpointing = self.hp.do_checkpoint
        encoder_out = self.encoder(
            inputs_embeds=x_noisy,
            time_step=t,
            attention_mask=x_mask,
        ).last_hidden_state
        
        pred = self.postnet(encoder_out)

        return pred, target

    def _forward_ph_encoder(self, ph_tokens, tone_tokens):
        ph_mask = ph_tokens > 0
        ph_embed = self.ph_embed(ph_tokens)
        tone_embed = self.tone_embed(tone_tokens)
        x_ph = ph_embed + tone_embed
        x_ph = self.ph_encoder(inputs_embeds=x_ph, attention_mask=ph_mask).last_hidden_state
        return x_ph

    def forward_ph_encoder(self, inputs):
        x_ph = self._forward_ph_encoder(inputs["phone"], inputs["tone"])
        x_ph = self.ling_pre_net(expand_states(x_ph, inputs['mel2ph']).transpose(1, 2)).transpose(1, 2)
        return x_ph

    def _forward(self, x, cond, timesteps, seq_cfg_w=(1.4, 3.0), timestep_annealing_w=(1.0, 0.0, 1.0)):
        """ When we use torchdiffeq, we need to include the CFG process inside _forward() """
        ctx = cond['ctx']
        ctx_mask = cond['ctx_mask']
        attn_mask = cond['attn_mask']
        x_txt = cond['x_txt']
        x = x * (1 - ctx_mask) + ctx
        x = self.lat_proj(torch.cat([self.prenet(torch.cat([x, ctx_mask], -1)), x_txt], -1))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_embed = self.f5_time_embed(timesteps)
        pred_v = self.encoder(inputs_embeds=x, time_step=t_embed, attention_mask=attn_mask).last_hidden_state
        pred = self.postnet(pred_v)

        a, b, p = timestep_annealing_w
        gamma_t = a + b * torch.pow(1 - timesteps, p)
        seq_cfg_w = [gamma_t * w for w in seq_cfg_w]
        
        cond_all, cond_txt, uncond = pred.chunk(3)
        pred = (
            uncond + 
            seq_cfg_w[0] * (cond_txt - uncond) + 
            seq_cfg_w[1] * (cond_all - cond_txt)
        )

        return pred

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, seq_cfg_w=[1.4, 3.0], timestep_annealing_w=(1.0, 0.0, 1.0), use_amo_sampler=False, **kwargs):
        x_ph = self.forward_ph_encoder(inputs)

        x_mask = torch.ones_like(x_ph[:, :, 0])

        (bsz, tgt_len, _), device = x_ph.shape, x_ph.device

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        x_txt = x_ph

        cond = {
            'ctx': ctx_feature,
            'ctx_mask': ctx_mask,
            'attn_mask': x_mask,
            'x_txt': x_txt,
        }
        
        ''' Euler ODE solver '''
        t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
        
        sway_sampling_coef = -1.0
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)
        else:
            t_schedule = 0.5 * (1 - torch.cos(torch.pi * t_schedule))
            
        if use_amo_sampler:

            def amo_sampling(sample, sigma, sigma_next, pred_v):
                # Upcast to avoid precision issues when computing prev_sample
                t = sigma
                s = sigma_next
                x_t = sample
                c = 3  # 2
                o = min(s + c * (s - t), 1)
                pred_x_o = x_t + (o - t) * pred_v
                a = s / o
                b = (torch.clamp_min((1 - s) ** 2 - (a * (1 - o)) ** 2, 0)) ** 0.5
                noises = torch.randn(size=x_t.shape, device=x_t.device)
                prev_sample = a * pred_x_o + b * noises
                prev_sample = prev_sample.to(pred_v.dtype)
                return prev_sample

            x = torch.randn([1, tgt_len, self.hp.out_channels], device=device)                
            for step_index in range(timesteps):
                sigma = t_schedule[step_index].to(x_txt.dtype)
                sigma_next = t_schedule[step_index + 1]
                model_out = self._forward(torch.cat([x] * bsz), cond, timesteps=sigma.unsqueeze(0), seq_cfg_w=seq_cfg_w, timestep_annealing_w=timestep_annealing_w)
                x = amo_sampling(x, sigma, sigma_next, model_out)
        
        else:

            traj = torchdiffeq.odeint(
                lambda t, x: self._forward(
                    torch.cat([x] * bsz), cond, timesteps=t.unsqueeze(0), seq_cfg_w=seq_cfg_w, timestep_annealing_w=timestep_annealing_w),
                torch.randn([1, tgt_len, self.hp.out_channels], device=device),
                t_schedule,
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
            x = traj[-1]
        
        return x