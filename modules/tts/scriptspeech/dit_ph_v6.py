import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Callable, Optional
import json

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import attrdictionary
import torchdiffeq

from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.tts.llama_dit.llama_ca import LLaMa
from modules.commons.pos_encoding import SinusoidalPositionalEmbedding

from utils.nn.seq_utils import sequence_mask, remove_prefix, remove_suffix, add_prefix_nd
from utils.nn.generation_utils import stochastic_round
from utils.commons.io import print_once

@dataclass
class ModelArgs:
    # text
    vocab_size: int = None
    text_dim: int = 1024

    # llama
    encoder_dim: int = 1024
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    mlp_extend: float = None
    max_seq_len: int = 16384
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 4
    use_causal_attn: bool = False
    use_qk_norm: bool = False
    use_dynamic_cross_gate: bool = False

    in_channels: int = 16
    out_channels: int = 16

    # diffusion
    cfg_mask_text_token: int = None
    text_fill_token: int = None
    ph_fill_token: int = None

    # trainging
    do_checkpoint: bool = False


class TotalDurationPredictor(nn.Module):
    def __init__(self, hp: ModelArgs, init_pretrained=False, audio_encoder_ckpt='checkpoints/wavlm/WavLM-Large.pt'):
        super().__init__()
        self.hp = hp

        from modules.asr.llama.llama import LLaMa as LLaMaSmall, ModelArgs as ModelArgsSmall
        from modules.tts.ar_dur.dur_lm import ContinuousTimeEncoderV2
        from modules.asr.wavlm.WavLM import WavLM, WavLMConfig
        if init_pretrained:
            checkpoint = torch.load(audio_encoder_ckpt)
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            model.load_state_dict(checkpoint['model'])
        else:
            cfg = json.load(open('checkpoints/wavlm/WavLM-Large-Config.json'))
            cfg = WavLMConfig(cfg)
            model = WavLM(cfg)
        self.audio_encoder_dim = audio_encoder_dim = cfg.encoder_embed_dim
        self.audio_encoder = model
        self.audio_encoder_hopsize = 320
        self.audio_encoder_sample_rate = 16000
        self.audio_proj = nn.Linear(audio_encoder_dim, hp.encoder_dim, bias=False)

        self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.text_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=4, n_heads=16,
            use_causal_attn=False, 
            crossattn_n_layers=4,
            use_dynamic_cross_gate=True,
            use_qk_norm=False,
        ))
        self.tone_embed = nn.Embedding(32, hp.encoder_dim, padding_idx=0)
        self.ph_embed = nn.Embedding(302, hp.encoder_dim)
        self.ph_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=4, n_heads=16,
            use_causal_attn=False,
            use_qk_norm=False,
        ))

        self.decoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=8, n_heads=16,
            use_causal_attn=False, 
            crossattn_n_layers=8,
            use_dynamic_cross_gate=True,
            use_qk_norm=False,
        ))

        self.head = nn.Linear(hp.encoder_dim, 2)
        # torch.nn.init.constant_(self.head.bias, 20)

    def forward(self, inputs, do_checkpoint=False):

        ph_tokens = inputs["phone"]     # [B, T]
        tone_tokens = inputs["tone"]
        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        ph_mask = inputs.get('ph_mask', ph_tokens > 0)
        bsz = ph_tokens.shape[0]
        
        wavs = inputs['wavs']
        wav_mask = inputs['wav_mask']
        audio_feat, audio_mask = self.encode(wavs, wav_mask, do_checkpoint=True)

        x = self.decode(txt_tokens, txt_mask, ph_tokens, tone_tokens, ph_mask, audio_feat, audio_mask, do_checkpoint)

        return x

    def encode(self, wavs, wav_mask, do_checkpoint=False):
        audio_feat, audio_mask = self.audio_encoder.extract_features(wavs, padding_mask=~(wav_mask.bool()), do_checkpoint=do_checkpoint)    # [B, T, C]
        audio_mask = (~(audio_mask.bool())).to(audio_feat)
        audio_feat = self.audio_proj(audio_feat)
        return audio_feat, audio_mask

    def decode(self, txt_tokens, txt_mask, ph_tokens, tone_tokens, ph_mask, audio_feat, audio_mask, do_checkpoint=False):
        dtype = torch.bfloat16

        ph_embeds = self.ph_embed(ph_tokens)
        tone_embeds = self.tone_embed(tone_tokens)
        ph_embeds = ph_embeds + tone_embeds
        ph_embeds = self.ph_encoder(ph_embeds, ph_mask, do_checkpoint=do_checkpoint)

        txt_embeds = self.text_embedder(txt_tokens).to(dtype)
        txt_embeds = self.text_encoder(
            txt_embeds, txt_mask, context=ph_embeds, context_lens=ph_mask.sum(1), do_checkpoint=do_checkpoint
        )

        x = self.decoder(
            txt_embeds, txt_mask, context=audio_feat, context_lens=audio_mask.sum(1), do_checkpoint=do_checkpoint
        )
        x = self.head(x)
        x = x[..., 0] * torch.sigmoid(x[..., 1])    # [B, T]
        x = torch.exp(x)  # [B, T]
        x = (x * txt_mask).sum(1)   # [B]

        return x


class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp

        self.encoder = LLaMa(hp)
        
        self.lat_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.ctx_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.ctx_mask_proj = nn.Linear(1, self.hp.encoder_dim)
        self.prenet = nn.Linear(self.hp.encoder_dim * 2 , self.hp.encoder_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)
        
        if hp.use_caption_encoder:
            self.caption_proj = nn.Linear(self.hp.caption_dim, self.hp.encoder_dim)
            if hp.use_caption_text_mark:
                self.caption_text_mark_embed = nn.Embedding(2, self.hp.encoder_dim)

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

        from modules.asr.llama.llama import LLaMa as LLaMaSmall, ModelArgs as ModelArgsSmall
        from modules.tts.ar_dur.dur_lm import ContinuousTimeEncoderV2
                
        self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.text_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=4, n_heads=16,
            use_causal_attn=False, 
            crossattn_n_layers=4,
            use_dynamic_cross_gate=True
        ))
        self.tone_embed = nn.Embedding(32, hp.encoder_dim, padding_idx=0)
        self.ph_embed = nn.Embedding(302, hp.encoder_dim)
        self.ph_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=2, n_heads=16,
            use_causal_attn=False,
        ))
        self.ling_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=4, n_heads=16,
            use_causal_attn=False, 
            crossattn_n_layers=8,
            use_dynamic_cross_gate=True
        ))
        self.ling_audio_prenet = nn.Linear(hp.in_channels, hp.encoder_dim)
        self.ling_audio_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=2, n_heads=16,
            use_causal_attn=False
        ))

        # # init all weights
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
            nn.init.zeros_(block.attention_norm.linear.weight)
            nn.init.zeros_(block.attention_norm.linear.bias)

        # Zero-out output layers
        nn.init.zeros_(self.encoder.norm.linear.weight)
        nn.init.zeros_(self.encoder.norm.linear.bias)
        nn.init.zeros_(self.encoder.out_proj.weight)
        nn.init.zeros_(self.encoder.out_proj.bias)

    def forward_ling_encoder(self, inputs, x_mask):
        do_checkpoint = self.hp.do_checkpoint
        dtype = torch.bfloat16
        # if torch.is_autocast_enabled():
        #     dtype = torch.get_autocast_gpu_dtype()

        ph_tokens = inputs["phone"]     # [B, T]
        tone_tokens = inputs["tone"]
        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        bsz = ph_tokens.shape[0]
        
        ph_mask = inputs.get('ph_mask', ph_tokens > 0)
        ph_cfg_mask = (ph_tokens[:, 0] == 301)[:, None]

        ph_embeds = self.ph_embed(ph_tokens)
        tone_embeds = self.tone_embed(tone_tokens)
        ph_embeds = ph_embeds + tone_embeds
        ph_embeds = self.ph_encoder(ph_embeds, ph_mask, do_checkpoint=do_checkpoint)

        txt_embeds = self.text_embedder(txt_tokens).to(dtype)
        txt_embeds = self.text_encoder(
            txt_embeds, txt_mask, context=ph_embeds, context_lens=ph_mask.sum(1), do_checkpoint=do_checkpoint
        )

        tgt_len = x_mask.shape[1]

        x_txt = torch.full((bsz, tgt_len), self.hp.text_fill_token).to(txt_tokens)
        x_txt = self.text_embedder(x_txt).to(dtype)
        x_txt_mask = sequence_mask(txt_mask.long().sum(1), tgt_len)
        x_txt[x_txt_mask] = txt_embeds[txt_mask]

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask
        ctx = self.ling_audio_encoder(self.ling_audio_prenet(ctx_feature), ctx_mask.squeeze(), do_checkpoint=do_checkpoint)
        x_txt = self.ling_encoder(
            x_txt, x_mask, context=ctx, context_lens=ctx_mask.sum(1).squeeze(), do_checkpoint=do_checkpoint
        )
        
        return x_txt

    def forward(self, inputs):
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
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = self.f5_time_embed(t)
        x_noisy = (xt * (1 - ctx_mask)).bfloat16()
        target = ut

        x_txt = self.forward_ling_encoder(inputs, x_mask)

        if self.hp.use_caption_encoder and 'caption_emb' in inputs and inputs['caption_emb'] is not None:
            caption_embs = self.caption_proj(inputs['caption_emb'])
            if self.hp.use_caption_text_mark:
                caption_text_mark_embed = self.caption_text_mark_embed(inputs['caption_text_mark'].long())
                caption_embs = caption_embs + caption_text_mark_embed
            caption_lens = inputs['caption_lens']
        else:
            caption_embs, caption_lens = None, None
            
        x_noisy = self.lat_proj(x_noisy) + self.ctx_proj(ctx_feature) + self.ctx_mask_proj(ctx_mask)
        x_noisy = self.prenet(torch.cat([x_noisy, x_txt], dim=-1))

        encoder_out = self.encoder(x_noisy, t, attn_mask=x_mask, do_checkpoint=self.hp.do_checkpoint, 
                                   context=caption_embs, context_lens=caption_lens)
        
        pred = self.postnet(encoder_out)

        return pred, target

    def _forward(self, x, cond, timesteps, seq_cfg_w=(1.0, 1.5, 1.5, 1.5), timestep_annealing_w=(1.0, 0.0, 1.0)):
        """ When we use torchdiffeq, we need to include the CFG process inside _forward() """
        ctx = cond['ctx']
        ctx_mask = cond['ctx_mask']
        attn_mask = cond['attn_mask']
        x_txt = cond['x_txt']
        
        x = x * (1 - ctx_mask)
        
        x = self.lat_proj(x) + self.ctx_proj(ctx) + self.ctx_mask_proj(ctx_mask)
        x = self.prenet(torch.cat([x, x_txt], dim=-1))
        
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_embed = self.f5_time_embed(timesteps)
        pred_v = self.encoder(x, t_embed, attn_mask=attn_mask, do_checkpoint=self.hp.do_checkpoint)
        pred = self.postnet(pred_v)

        if isinstance(timesteps, torch.Tensor) and timesteps.ndim > 0 and timesteps.shape[0] > pred.shape[0] // 3:
            timesteps, _, _ = timesteps.chunk(3)
            if timesteps.ndim == 1:
                timesteps = timesteps[:, None, None]
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
    def inference(self, inputs, timesteps=20, seq_cfg_w=[1.5, 3.0], timestep_annealing_w=(0.6, 0.6, 1.0), return_timesteps=False, **kwargs):
        tgt_len = inputs['tgt_len']     # reference + target
        x_mask = sequence_mask(tgt_len)
        
        x_txt = self.forward_ling_encoder(inputs, x_mask)

        (bsz, tgt_len, _), device = x_txt.shape, x_txt.device
        bsz = bsz // 3

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        cond = {
            'ctx': ctx_feature,
            'ctx_mask': ctx_mask,
            'attn_mask': x_mask,
            'x_txt': x_txt,
        }
        if inputs.get('vad_mask') is not None:
            cond['vad_mask'] = inputs['vad_mask']

        ''' Euler ODE solver '''
        t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
        
        sway_sampling_coef = -1.0
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)
        else:
            t_schedule = 0.5 * (1 - torch.cos(torch.pi * t_schedule))
        
        traj = torchdiffeq.odeint(
            lambda t, x: self._forward(
                torch.cat([x] * 3), cond, timesteps=t.unsqueeze(0), seq_cfg_w=seq_cfg_w, timestep_annealing_w=timestep_annealing_w),
            torch.randn([bsz, tgt_len, self.hp.out_channels], device=device),
            t_schedule,
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        x = traj[-1]

        if return_timesteps:
            return x, t_schedule
        
        return x

