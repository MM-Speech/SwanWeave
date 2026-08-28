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
from einops import rearrange, repeat
import torchdiffeq

from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.tts.llama_dit.llama_ca import LLaMa as LLaMaDiT, ModelArgs as LLaMaDiTModelArgs
from modules.asr.llama.llama import LLaMa, ModelArgs as LLaMaModelArgs

from utils.nn.seq_utils import sequence_mask, add_prefix, add_prefix_nd, remove_prefix, last_token_mask, build_last_k_soft_labels
from utils.nn.generation_utils import amo_sampling
from utils.losses.focal_loss import sigmoid_focal_loss
from utils.commons.io import print_once
from utils.nn.seq_utils import get_incremental_state, set_incremental_state, softmax, make_positions

@dataclass
class ModelArgs:    
    # text
    text_vocab_size: int = None
    text_dim: int = 1024

    # audio
    patch_size: int = 4
    ctx_n_patches: int = 2
    add_vad_mask: bool = False

    # caption
    caption_dim: int = 1024
    use_caption_encoder: bool = True
    crossattn_n_layers: int = 24
    
    # encoder
    encoder_dim: int = 1024
    encoder_n_layers: int = 6
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    
    # lm
    lm_dim: int = 1024
    lm_enc_n_layers: int = 2
    lm_dec_n_layers: int = 36
    lm_n_heads: int = 16
    
    # stop clf
    focal_loss_alpha: float = 0.99
    focal_loss_gamma: float = 1.5
    
    # decoder
    decoder_dim: int = 1024
    decoder_n_layers: int = 6
    decoder_n_heads: int = 16
    decoder_n_kv_heads: int = None
    training_patch_keep_ratio: float = -1.0
    
    # latent
    in_channels: int = 16
    out_channels: int = 16

    # trainging
    do_checkpoint: bool = False
    warm_up_lm: bool = False
    

class StopClf(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(input_channels, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear3 = nn.Linear(hidden_size, output_channels, bias=False)
    
    def forward(self, x):
        x = F.silu(self.linear1(x))
        x = F.silu(self.linear2(x))
        x = self.linear3(x)
        return x


class DiTARModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = self.hp = config
        
        # linguistic
        self.text_embedder = nn.Embedding(config.text_vocab_size, config.text_dim)
        self.text_encoder = LLaMa(LLaMaModelArgs(dim=config.text_dim, n_layers=4, n_heads=16, use_causal_attn=False))
        
        self.ling_postnet = nn.Linear(config.text_dim, config.lm_dim, bias=False)
        
        # semantic
        if config.use_caption_encoder:
            self.caption_lm_proj = nn.Linear(config.caption_dim, config.lm_dim)
            self.caption_decoder_proj = nn.Linear(config.caption_dim, config.decoder_dim)
            self.caption_encoder = LLaMa(LLaMaModelArgs(dim=config.lm_dim, n_layers=config.lm_enc_n_layers, use_causal_attn=False))

        # encoder
        self.encoder_prenet = nn.Linear(config.in_channels, config.encoder_dim)
        self.encoder = LLaMa(LLaMaModelArgs(
            dim=config.encoder_dim, n_layers=config.encoder_n_layers, 
            n_heads=config.encoder_n_heads, use_qk_norm=True, use_causal_attn=False
        ))
        self.encoder_cls_token = nn.Parameter(torch.randn((1, 1, config.encoder_dim)))
        self.encoder_postnet = nn.Linear(config.encoder_dim, config.lm_dim, bias=False)
        
        # lm
        self.lm = LLaMa(LLaMaModelArgs(
            dim=config.lm_dim,
            n_heads=config.lm_n_heads,
            n_layers=config.lm_dec_n_layers,
            use_qk_norm=True,
            use_gated_attention=True,
            crossattn_n_layers=config.lm_dec_n_layers if config.use_caption_encoder else 0
        ))
        self.speech_start_token = nn.Parameter(torch.randn((1, 1, config.lm_dim)))
        self.lm_postnet = nn.Linear(config.lm_dim, config.decoder_dim, bias=False)
        self.stop_clf = StopClf(config.lm_dim, 1)
        if config.warm_up_lm:
            self.lm_input = nn.Linear(config.in_channels, config.lm_dim, bias=False)
            self.lm_output = nn.Linear(config.lm_dim, config.out_channels, bias=False)
        
        # decoder
        self.decoder = LLaMaDiT(LLaMaDiTModelArgs(
            encoder_dim=config.decoder_dim,
            encoder_n_layers=config.decoder_n_layers,
            encoder_n_heads=config.decoder_n_heads,
            crossattn_n_layers=config.decoder_n_layers if config.use_caption_encoder else 0,
            use_caption_pool_in_adaln=True,
            use_caption_pool_gate_in_adaln=True,
            ffn_dim_multiplier=4,
            use_qk_norm=True
        ))
        
        self.decoder_prenet = nn.Linear(config.in_channels, config.decoder_dim)
        self.ctx_mask_embed = nn.Embedding(2, config.decoder_dim)
        self.decoder_postnet = nn.Linear(config.decoder_dim, config.out_channels)
        
        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(config.decoder_dim)
        
    def forward(self, inputs):
        
        if self.config.warm_up_lm:
            return self.forward_warmup_lm(inputs)
        
        x: torch.Tensor = inputs['lat']
        bsz, device = x.size(0), x.device
        float_type = autocast_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
        x_lens = inputs['lat_lens']
        x_mask = sequence_mask(inputs['lat_lens'], maxlen=x.shape[1]).long()
        
        x_ling, x_ling_mask = self.forward_ling_encoder(inputs)
        h, h_mask = self.forward_encoder(x, x_mask)
        
        ######
        # lm #
        ######
        x_ling = add_prefix_nd(
            x_ling, x_ling_mask.sum(1), self.speech_start_token.to(x_ling).repeat(x_ling.shape[0], 1, 1), 
            torch.ones(x_ling.shape[0], dtype=torch.long, device=device)
        )
        x_ling_mask = sequence_mask(x_ling_mask.sum(1) + 1, maxlen=x_ling.shape[1])
        h = add_prefix_nd(x_ling, x_ling_mask.sum(1), h, h_mask.sum(1))
        h_mask = sequence_mask(x_ling_mask.sum(1) + h_mask.sum(1), maxlen=h.shape[1])
        
        caption_emb, context_lens = None, None
        if self.config.use_caption_encoder and inputs.get('caption_emb') is not None:
            caption_emb = self.caption_lm_proj(inputs['caption_emb'])
            caption_mask = sequence_mask(inputs['caption_lens'], maxlen=inputs['caption_emb'].shape[1])
            caption_emb = self.caption_encoder(caption_emb, caption_mask)
            context_lens=caption_mask.sum(1)
        
        h = self.lm(
            x=h, attn_mask=h_mask,
            context=caption_emb,
            context_lens=context_lens,
            do_checkpoint=self.config.do_checkpoint
        )
        h = remove_prefix(h, prefix_lens=x_ling_mask.sum(1) - 1, output_lens=h_mask.sum(1) - x_ling_mask.sum(1))
        h_mask = sequence_mask(h_mask.sum(1) - x_ling_mask.sum(1), maxlen=h.shape[1])
        
        stop_logits = self.stop_clf(h)[..., 0]
        stop_labels = build_last_k_soft_labels(h_mask, K=4, gamma=2).to(stop_logits.dtype)
        stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_labels, reduction='none')
        stop_loss = (stop_loss * h_mask).sum() / (h_mask).sum()
        
        h = self.lm_postnet(h)  # [B, T, C]
        
        #######
        # dit #
        #######
        h = rearrange(h, 'b t c -> (b t) c')[:, None, :]    # [BxT, 1, C]
        ctx_n_patches, patch_size = self.config.ctx_n_patches, self.config.patch_size
        if ctx_n_patches > 0:
            z = torch.cat([x.new_full((x.shape[0], ctx_n_patches * patch_size, x.shape[2]), 0.0), x], dim=1).unfold(
                dimension=1, 
                size=(ctx_n_patches + 1) * patch_size,
                step=patch_size
            )   # [B, T, C, P]
            z_mask = torch.cat([x_mask.new_full((x_mask.shape[0], ctx_n_patches * patch_size), 1.0), x_mask], dim=1).unfold(
                dimension=1, 
                size=(ctx_n_patches + 1) * patch_size,
                step=patch_size
            )   # [B, T, P]
        else:
            z = x.unfold(dimension=1, size=patch_size, step=patch_size)   # [B, T, C, P]
            z_mask = x_mask.unfold(dimension=1, size=patch_size, step=patch_size)   # [B, T, P]
        z = rearrange(z, 'b t c p -> (b t) p c')    # [BxT, P, C]
        z_mask = rearrange(z_mask, 'b t p -> (b t) p')  # [BxT, P]
        
        ctx_mask = torch.zeros_like(z_mask)
        ctx_mask[:, :ctx_n_patches * patch_size] = 1
        
        # CFG
        ctx_cfg_mask = torch.rand_like(ctx_mask[:, 0].to(float_type))[:, None, None]  # [BxT, 1, 1]
        ctx_cfg_mask = (ctx_cfg_mask < 0.15).to(float_type)
        z = z * ctx_mask[..., None] * (1 - ctx_cfg_mask) + z * (1 - ctx_mask[..., None])
        h_cfg_mask = torch.rand_like(ctx_mask[:, 0].to(float_type))[:, None, None]    # [BxT, 1, 1]
        h_cfg_mask = (h_cfg_mask < 0.15).to(float_type)
        h = h * (1 - h_cfg_mask)
        
        z0 = torch.randn_like(z)
        t = self.flow_matcher.time_sampler.sample([z0.shape[0]], z0.device).type_as(z0)
        zt = t[:, None, None] * z + (1 - t[:, None, None]) * z0
        ut = z - z0
        
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = self.f5_time_embed(t)
        z_noisy = zt * (1 - ctx_mask[..., None]) + z * ctx_mask[..., None]
        target = ut
        
        z_noisy = self.decoder_prenet(z_noisy)
        
        z_noisy = torch.cat([h.to(z_noisy.dtype), z_noisy], dim=1)    # [BxT, 1+P, C]
        z_mask = torch.cat([torch.ones_like(z_mask)[:, 0:1], z_mask], dim=1)
        ctx_mask = torch.cat([torch.ones_like(ctx_mask)[:, 0:1], ctx_mask], dim=1)
        
        z_noisy = z_noisy + self.ctx_mask_embed(ctx_mask.long()).to(z_noisy.dtype)
        
        if 0 < self.config.training_patch_keep_ratio < 1:
            n_patches_keep = max(1, int(z.shape[0] * self.config.training_patch_keep_ratio))
            indices = torch.randperm(z.shape[0], device=z.device)[:n_patches_keep]
            z_noisy = z_noisy[indices]
            t = t[indices]
            z_mask = z_mask[indices]
            ctx_mask = ctx_mask[indices]
            target = target[indices]
        
        pred = self.decoder(z_noisy, t.to(z_noisy.dtype), z_mask, do_checkpoint=self.config.do_checkpoint)
        
        pred = self.decoder_postnet(pred)
        pred = pred[:, 1:]
        
        loss_mask = z_mask * (1 - ctx_mask)
        loss_mask = loss_mask[:, 1:, None]
        diff_loss = F.mse_loss(pred.float(), target.float(), reduction='none')
        diff_loss = (diff_loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]
        
        ret = {
            'stop_loss': stop_loss,
            'diff_loss': diff_loss,
            'pred': pred,
            'target': target,
            'loss_mask': loss_mask,
            'ctx_mask': ctx_mask,
            'stop_labels': stop_labels
        }
        
        return ret
    
    def forward_warmup_lm(self, inputs):
        audio_feat = inputs['audio_feat']   # [B, T, C]
        audio_feat_mask = inputs['audio_feat_mask']
        audio_feat = F.normalize(audio_feat, dim=-1)
        bsz, device = audio_feat.size(0), audio_feat.device
        
        x_ling, x_ling_mask = self.forward_ling_encoder(inputs)
        h = self.lm_input(audio_feat)
        h_mask = audio_feat_mask
        
        x_ling = add_prefix_nd(
            x_ling, x_ling_mask.sum(1), self.speech_start_token.to(x_ling).repeat(x_ling.shape[0], 1, 1), 
            torch.ones(x_ling.shape[0], dtype=torch.long, device=device)
        )
        x_ling_mask = sequence_mask(x_ling_mask.sum(1) + 1, maxlen=x_ling.shape[1])
        h = add_prefix_nd(x_ling, x_ling_mask.sum(1), h, h_mask.sum(1))
        h_mask = sequence_mask(x_ling_mask.sum(1) + h_mask.sum(1), maxlen=h.shape[1])
        
        h = self.lm(x=h, attn_mask=h_mask, do_checkpoint=self.config.do_checkpoint)
        h = remove_prefix(h, prefix_lens=x_ling_mask.sum(1) - 1, output_lens=h_mask.sum(1) - x_ling_mask.sum(1))
        h_mask = sequence_mask(h_mask.sum(1) - x_ling_mask.sum(1), maxlen=h.shape[1])
        
        stop_logits = self.stop_clf(h)[..., 0]
        stop_labels = build_last_k_soft_labels(h_mask, K=4, gamma=2).to(stop_logits.dtype)
        stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_labels, reduction='none')
        stop_loss = (stop_loss * h_mask).sum() / (h_mask).sum()
        
        h = self.lm_output(h)
        
        cos = F.cosine_similarity(h, audio_feat, dim=-1)    # [B, T]
        cos = (cos * h_mask).sum() / h_mask.sum()
        loss = 1.0 - cos
        
        return {
            'stop_loss': stop_loss,
            'cos': loss,
            'ntokens': h_mask.sum()
        }
        
    def forward_ling_encoder(self, inputs):
        txt_tokens = inputs["txt_tokens"]
        txt_embed = self.text_embedder(txt_tokens)
        txt_mask = inputs["txt_mask"]
        x_txt = self.text_encoder(txt_embed, attn_mask=txt_mask)
        x_txt = self.ling_postnet(x_txt)       
        return x_txt, txt_mask

    def forward_encoder(self, x, x_mask):        
        x = self.encoder_prenet(x)
        bsz = x.shape[0]

        x = rearrange(x, "b (t p) c -> (b t) p c", p=self.config.patch_size)
        x_mask = rearrange(x_mask, "b (t p) -> (b t) p", p=self.config.patch_size)
        
        x = torch.cat([self.encoder_cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x_mask = torch.cat([torch.ones((x.shape[0], 1)).to(x_mask), x_mask], dim=1)
        
        x = self.encoder(x, x_mask)
        
        x = x[:, 0:1, :]
        x_mask = x_mask[:, 0:1]
        
        x = rearrange(x, "(b t) p c -> b (t p) c", b=bsz)
        x_mask = rearrange(x_mask, "(b t) p -> b (t p)", b=bsz)
        
        x = self.encoder_postnet(x)
        
        return x, x_mask
        
    def inference(self, inputs, timesteps=20, seq_cfg_w=(1.4, 3), timestep_annealing_w=(1.0, 0.0, 1.0), 
                  start_pos=0, use_amo_sampler=True, use_tqdm=False, max_new_patches=1024):
        ref_x = inputs['ref_lat']
        bsz, device = ref_x.size(0), ref_x.device
        float_type = autocast_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else ref_x.dtype
        ref_x_lens = inputs['ref_lat_lens']
        ref_x_mask = sequence_mask(ref_x_lens, maxlen=ref_x.shape[1]).long()
        
        # ref_txt_tokens = inputs['ref_txt_tokens']
        # ref_txt_mask = inputs['ref_txt_mask']
        txt_tokens = inputs['txt_tokens']
        txt_mask = inputs['txt_mask']
        
        # txt_tokens = add_prefix(ref_txt_tokens, ref_txt_mask.sum(1), txt_tokens, txt_mask.sum(1))
        # txt_mask = sequence_mask(ref_txt_mask.sum(1) + txt_mask.sum(1), maxlen=txt_tokens.shape[1])
        
        x_ling, x_ling_mask = self.forward_ling_encoder({"txt_tokens": txt_tokens, "txt_mask": txt_mask})
        ref_h, ref_h_mask = self.forward_encoder(ref_x, ref_x_mask)
        
        caption_emb, caption_mask, caption_lens = None, None, None
        if self.config.use_caption_encoder and inputs.get('caption_emb') is not None:
            caption_emb = inputs['caption_emb']
            caption_lens = inputs['caption_lens']
            caption_mask = sequence_mask(caption_lens, maxlen=caption_emb.shape[1])
            caption_emb = self.caption_encoder(caption_emb, caption_mask)
            
        step_ref = ref_h.shape[1] - 1
        x_ling = add_prefix_nd(
            x_ling, x_ling_mask.sum(1), self.speech_start_token.to(x_ling).repeat(x_ling.shape[0], 1, 1), 
            torch.ones(x_ling.shape[0], dtype=torch.long, device=device)
        )
        x_ling_mask = sequence_mask(x_ling_mask.sum(1) + 1, maxlen=x_ling.shape[1])
        ref_h = add_prefix_nd(x_ling, x_ling_mask.sum(1), ref_h, ref_h_mask.sum(1))
        ref_h_mask = sequence_mask(x_ling_mask.sum(1) + ref_h_mask.sum(1), maxlen=ref_h.shape[1])
        
        # prefill
        _ = self.lm(
            x=ref_h[:, :-1], attn_mask=ref_h_mask[:, :-1],
            context=caption_emb, context_lens=caption_lens, 
            start_pos=start_pos, use_cache=True
        )
        start_pos = start_pos + ref_h.shape[1] - 1
        lm_input = ref_h[:, -1:]
        
        # dit ctx
        ctx_n_patches, patch_size = self.config.ctx_n_patches, self.config.patch_size
        if ctx_n_patches > 0:
            z_ctx = ref_x[:, -ctx_n_patches * patch_size:]
        
        def forward_step(lm_input, step, start_pos, z_ctx=None):
            lm_output = self.lm.forward(
                x=lm_input, attn_mask=torch.ones_like(lm_input[..., 0]).bool(),
                context=caption_emb, context_lens=caption_lens,
                start_pos=start_pos, use_cache=True
            )
            stop_logits = self.stop_clf(lm_output)[..., 0]  # [B, 1]
            stop_prob = torch.sigmoid(stop_logits)
            stop = stop_prob > 0.1
            
            lm_output = self.lm_postnet(lm_output)  # [B, 1, C]
            
            h = torch.cat([
                lm_output,
                lm_output,
                torch.zeros_like(lm_output)
            ], dim=0).to(float_type)
            
            if z_ctx is not None:
                z_ctx = torch.cat([
                    z_ctx,
                    torch.zeros_like(z_ctx),
                    torch.zeros_like(z_ctx),
                ], dim=0)
            
            t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
            t_schedule = 0.5 * (1 - torch.cos(torch.pi * t_schedule))
            tgt_len = patch_size
            
            cond = {
                'z_ctx': z_ctx,
                'h': h,
            }
        
            traj = torchdiffeq.odeint(
                lambda t, x: forward_dit_step(
                    torch.cat([x] * 3), cond, t=t.unsqueeze(0), 
                    seq_cfg_w=seq_cfg_w, timestep_annealing_w=timestep_annealing_w
                ),
                torch.randn([bsz, tgt_len, self.hp.out_channels], device=device),
                t_schedule,
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
            x = traj[-1]
            
            return x, stop, stop_prob
            
        def forward_dit_step(x, cond, t, seq_cfg_w=(1.4, 3), timestep_annealing_w=(1.0, 0.0, 1.0)):
            z_ctx = cond['z_ctx']
            h = cond['h']

            if z_ctx is not None:
                x = torch.cat([z_ctx, x], dim=1)
            x = self.decoder_prenet(x)
            x = torch.cat([h, x], dim=1)
            x_mask = torch.ones_like(x[..., 0]).long()
            ctx_mask = torch.zeros_like(x_mask)
            if z_ctx is not None:
                ctx_mask[:, :1 + z_ctx.shape[1]] = 1
            else:
                ctx_mask[:, :1] = 1
            
            x = x + self.ctx_mask_embed(ctx_mask.long()).to(x.dtype)
            
            with torch.amp.autocast('cuda', dtype=torch.float32):
                t_embed = self.f5_time_embed(t).expand(x.shape[0], -1)
                
            pred = self.decoder(x, t_embed.to(x.dtype), x_mask)
                
            if z_ctx is not None:
                pred = pred[:, 1 + z_ctx.shape[1]:]
            else:
                pred = pred[:, 1:]
            pred = self.decoder_postnet(pred)
            
            a, b, p = timestep_annealing_w
            gamma_t = a + b * torch.pow(1 - t, p)
            seq_cfg_w = [gamma_t * w for w in seq_cfg_w]
            
            cond_all, cond_txt, uncond = pred.chunk(3)
            pred = (
                uncond + 
                seq_cfg_w[0] * (cond_txt - uncond) + 
                seq_cfg_w[1] * (cond_all - cond_txt)
            )
                        
            return pred
        
        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(range(max_new_patches), desc='| Generating Speech')
        else:
            it = range(max_new_patches)
            
        stop_logits_lst = []
        res_patches = []
        for step in it:
            x_patch, stop, stop_logits = forward_step(
                lm_input, step + step_ref, start_pos, 
                z_ctx[:, -ctx_n_patches * patch_size:] if ctx_n_patches > 0 else None
            )   # [B, T, C]
            h_patch, h_patch_mask = self.forward_encoder(x_patch, torch.ones_like(x_patch[..., 0]).long())
            lm_input = h_patch
            start_pos = start_pos + 1
            stop_logits_lst.append(stop_logits.item())
            
            if ctx_n_patches > 0:
                z_ctx = torch.cat([z_ctx, x_patch], dim=1)
            res_patches.append(x_patch)
            
            if stop.all():
                break
            
        res_lat = torch.cat(res_patches, dim=1)
        
        return res_lat, stop_logits_lst
            
        