import logging
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import torch
from torch import nn
import torchdiffeq
from modules.tts.llama_dit.llama_prompt import LLaMa
from modules.commons.engram import NGramEngram
from utils.nn.seq_utils import sequence_mask

logger = logging.getLogger(__name__)

@dataclass
class ModelArgs:
    # text
    vocab_size: int = None
    text_dim: int = 1024

    # audio
    audio_vocab_size: int = None
    audio_tokenizer: str = 'glm4v'
    
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

    caption_dim: int = 3584 + 1 # dim of seedance text encoder + content mask

    in_channels: int = 16
    out_channels: int = 16

    # trainging
    do_checkpoint: bool = False
    use_qk_norm: bool = False

    cfg_mask_text_token: int = None
    text_fill_token: int = None
    use_caption_pool_in_adaln: bool = False
    use_caption_text_mark: bool = False
    use_spk_mask: bool = False 
    use_bgm_flag: bool = False
    use_quality_flag: bool = False

    use_gated_attention: bool = False
    use_dynamic_cross_gate: bool = False

    use_engram: bool = False
    engram_target: Literal["caption", "text", "both"] = "caption"
    engram_orders: Tuple[int, ...] = (1, 2, 3)
    engram_num_heads: int = 4
    engram_table_size: int = 262144
    engram_head_dim: int = 64
    engram_dropout: float = 0.0
    engram_gate_bias: float = -4.0
    engram_conv_kernel: int = 3
    engram_special_replace_id: int = 1

    use_moe_ffn: bool = False
    moe_p: float = 0.7                # Top-P routing threshold
    moe_num_routed: int = 8           # routed experts count
    moe_num_shared: int = 2           # shared experts count (always-on)
    moe_num_null: int = 4             # null experts count (no compute)
    moe_aux_loss_weight: float = 0.01 # training-time weight (you can anneal in loop)
    moe_use_gumbel: bool = False
    moe_gumbel_tau_start: float = 1.0       # 初始温度（训练早期）
    moe_gumbel_tau_end: float = 0.3         # 最终温度（训练后期）
    moe_gumbel_tau_anneal_steps: int = 200_000  # 退火步数
    moe_expert_dropout: float = 0.0   # 每个 step 随机屏蔽部分专家（0~1）
    moe_max_experts_per_token: int = 4  # 每个 token 最多用几个 routed experts
    moe_use_bias_balance: bool = True
    moe_bias_update_rate: float = 0.05
    moe_bias_momentum: float = 0.9
    moe_bias_clamp: float = 5.0
    moe_capacity_factor_min: float = 1.0
    moe_capacity_factor_max: float = 2.0
    moe_overflow_drop: bool = True
    moe_use_t_budget: bool = True
    moe_p_min: float = 0.4
    moe_p_max: float = 0.95
    moe_null_logit_bias_min: float = -2.0
    moe_null_logit_bias_max: float = 0.0
    decoder_sparse_step: int = 1
    mlp_only_layers: Tuple[int, ...] = ()

class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp

        self.encoder = LLaMa(hp)
        self.prenet = nn.Linear(self.hp.encoder_dim * 2 , self.hp.encoder_dim)

        self.lat_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.ctx_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.ctx_mask_proj = nn.Linear(1, self.hp.encoder_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)
        self.caption_proj = nn.Linear(self.hp.caption_dim, self.hp.encoder_dim)
        if hp.use_caption_text_mark:
            self.caption_text_mark_embed = nn.Embedding(4, self.hp.encoder_dim)

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

        from modules.asr.llama.llama import LLaMa as LLaMaSmall, ModelArgs as ModelArgsSmall

        self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.text_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=8, n_heads=16,
            use_causal_attn=False, 
        ))

        if hp.use_spk_mask:
            self.spk_mask_embedder = nn.Embedding(5, hp.encoder_dim)  # 0=none, 1..4=S1..S4

        self.use_engram = bool(getattr(hp, "use_engram", False))
        self.engram_target = getattr(hp, "engram_target", "caption")
        if self.use_engram and (self.engram_target in ("caption", "both")):
            self.caption_engram = NGramEngram(
                model_dim=self.hp.encoder_dim,
                orders=tuple(getattr(hp, "engram_orders", (1, 2, 3))),
                num_heads=int(getattr(hp, "engram_num_heads", 4)),
                table_size=int(getattr(hp, "engram_table_size", 262144)),
                head_dim=int(getattr(hp, "engram_head_dim", 64)),
                dropout=float(getattr(hp, "engram_dropout", 0.0)),
                gate_bias=float(getattr(hp, "engram_gate_bias", -4.0)),
                conv_kernel=int(getattr(hp, "engram_conv_kernel", 3)),
            )
        else:
            self.caption_engram = None

        if self.use_engram and (self.engram_target in ("text", "both")):
            self.text_engram = NGramEngram(
                model_dim=self.hp.encoder_dim,
                orders=tuple(getattr(hp, "engram_orders", (2, 3))),
                num_heads=int(getattr(hp, "engram_num_heads", 4)),
                table_size=int(getattr(hp, "engram_table_size", 262144)),
                head_dim=int(getattr(hp, "engram_head_dim", 64)),
                dropout=float(getattr(hp, "engram_dropout", 0.0)),
                gate_bias=float(getattr(hp, "engram_gate_bias", -4.0)),
                conv_kernel=int(getattr(hp, "engram_conv_kernel", 3)),
            )
        else:
            self.text_engram = None
            
        if hp.use_bgm_flag:
            self.bgm_flag_embed = nn.Embedding(3, hp.encoder_dim)
        if hp.use_quality_flag:
            self.quality_flag_embed = nn.Embedding(4, hp.encoder_dim)

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

    def forward_text_encoder(self, inputs, x_mask):
        tgt_len = x_mask.shape[1]
        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        bsz = txt_tokens.shape[0]

        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        # ===== 1) build filled text ids: [B, tgt_len] =====
        x_txt_ids = torch.full((bsz, tgt_len), self.hp.text_fill_token, device=txt_tokens.device, dtype=txt_tokens.dtype)
        fill_pos = sequence_mask(txt_mask.long().sum(1), tgt_len)  # [B, tgt_len]
        x_txt_ids[fill_pos] = txt_tokens[txt_mask]

        token_emb = self.text_embedder(x_txt_ids.long())  # [B, tgt_len, C]

        # ===== 2) add spk_mask embedding if enabled =====
        if self.hp.use_spk_mask and ('spk_mask' in inputs) and (inputs['spk_mask'] is not None):
            spk_mask = inputs['spk_mask']  # expected shape [B, T_txt] with same txt_mask
            # 对齐到和 x_txt_ids 一样的“前缀填充”布局
            spk_ids = torch.zeros((bsz, tgt_len), device=txt_tokens.device, dtype=torch.long)
            spk_ids[fill_pos] = spk_mask[txt_mask].long().clamp_(0, 4)
            token_emb = token_emb + self.spk_mask_embedder(spk_ids)  # [B, tgt_len, C]

        if self.text_engram is not None:
            special_ids = [
                int(getattr(self.hp, "text_fill_token", -1)),
                int(getattr(self.hp, "cfg_mask_text_token", -1)),
            ]
            token_emb = token_emb + self.text_engram(
                x_txt_ids,
                x_mask,
                token_emb,
                special_ids=[s for s in special_ids if s >= 0],
                special_replace_id=int(getattr(self.hp, "engram_special_replace_id", 1)),
            )

        x_txt = self.text_encoder(token_emb, x_mask)  # [B, tgt_len, C]
        return x_txt

    def _encode_caption_context(self, caption_emb, caption_lens, caption_ids=None, caption_text_mark=None):
        if (caption_emb is None) or (caption_lens is None):
            return None, None

        caption_ctx = self.caption_proj(caption_emb)
        max_ctx_len = int(caption_ctx.shape[1])

        cap_lens = caption_lens.to(device=caption_ctx.device, dtype=torch.long)
        if cap_lens.numel() == 0:
            return None, None
        eff_len = int(cap_lens.max().item())
        eff_len = min(eff_len, max_ctx_len)
        if eff_len <= 0:
            return None, None

        caption_ctx = caption_ctx[:, :eff_len]
        cap_lens = cap_lens.clamp(min=0, max=eff_len)
        caption_mask = sequence_mask(cap_lens, maxlen=eff_len)

        if (self.caption_engram is not None) and (caption_ids is not None):
            cap_ids = caption_ids.to(device=caption_ctx.device)
            if int(cap_ids.shape[1]) < eff_len:
                cap_ids = torch.nn.functional.pad(cap_ids, (0, eff_len - int(cap_ids.shape[1])), value=0)
            cap_ids = cap_ids[:, :eff_len]
            caption_ctx = caption_ctx + self.caption_engram(
                cap_ids.to(torch.long),
                caption_mask,
                caption_ctx,
                special_replace_id=int(getattr(self.hp, "engram_special_replace_id", 1)),
            )

        if self.hp.use_caption_text_mark and (caption_text_mark is not None):
            mark = self.caption_text_mark_embed(caption_text_mark.to(device=caption_ctx.device, dtype=torch.long))
            caption_ctx = caption_ctx + mark[:, :eff_len]

        return caption_ctx, cap_lens

    def forward(self, inputs, sigmas=None, x0=None, t=None):
        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask
        x = inputs['lat']
        x_mask = sequence_mask(inputs['lat_lens'], maxlen=x.shape[1])

        if x0 is None:
            x0 = torch.randn_like(x)
        if t is None:
            t = self.flow_matcher.time_sampler.sample([x0.shape[0]], x0.device).type_as(x0)

        # CFM: x is x1
        xt = t[:, None, None] * x + (1 - t[:, None, None]) * x0
        ut = x - x0

        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_emb = self.f5_time_embed(t)
        x_noisy = (xt * (1 - ctx_mask)).bfloat16()
        target = ut

        bgm_flag = inputs.get('bgm_flag', None)
        if bgm_flag is not None and hasattr(self, 'bgm_flag_embed'):
            t_emb = t_emb + self.bgm_flag_embed(bgm_flag)
        quality_flag = inputs.get('quality_flag', None)
        if quality_flag is not None and hasattr(self, 'quality_flag_embed'):
            t_emb = t_emb + self.quality_flag_embed(quality_flag)

        x_txt = self.forward_text_encoder(inputs, x_mask)

        caption_embs, caption_lens = self._encode_caption_context(
            inputs.get('caption_emb', None),
            inputs.get('caption_lens', None),
            inputs.get('caption_ids', None),
            inputs.get('caption_text_mark', None),
        )

        x_noisy = self.lat_proj(x_noisy) + self.ctx_proj(ctx_feature) + self.ctx_mask_proj(ctx_mask)
        x_noisy = self.prenet(torch.cat([x_noisy, x_txt], dim=-1))

        use_moe = bool(getattr(self.hp, "use_moe_ffn", False))

        if use_moe:
            encoder_out, moe_aux = self.encoder(
                x_noisy, t_emb, attn_mask=x_mask,
                do_checkpoint=self.hp.do_checkpoint,
                context=caption_embs,
                context_lens=caption_lens,
            )
        else:
            encoder_out = self.encoder(
                x_noisy, t_emb, attn_mask=x_mask,
                do_checkpoint=self.hp.do_checkpoint,
                context=caption_embs,
                context_lens=caption_lens,
            )
            moe_aux = None

        pred = self.postnet(encoder_out)

        if use_moe:
            return pred, target, moe_aux
        return pred, target

    def _forward(self, x, cond, timesteps, seq_cfg_w=[1.5, 3.0], timestep_annealing_w=(0.6, 0.6, 1.0)):
        """When we use torchdiffeq, we need to include the CFG process inside _forward()."""
        ctx = cond['ctx']
        ctx_mask = cond['ctx_mask']
        attn_mask = cond['attn_mask']
        x_txt = cond['x_txt']

        caption_embs = cond.get('_caption_ctx', None)
        caption_lens = cond.get('_caption_ctx_lens', None)
        if caption_embs is None:
            # backward compat: allow precomputed non-underscored keys
            caption_embs = cond.get('caption_ctx', None)
            caption_lens = cond.get('caption_ctx_lens', None)

        if caption_embs is None:
            caption_embs, caption_lens = self._encode_caption_context(
                cond.get('caption_emb', None),
                cond.get('caption_lens', None),
                cond.get('caption_ids', None),
                cond.get('caption_text_mark', None),
            )
            # cache to avoid recompute per ODE step
            cond['_caption_ctx'] = caption_embs
            cond['_caption_ctx_lens'] = caption_lens

        x = x * (1 - ctx_mask)
        x = self.lat_proj(x) + self.ctx_proj(ctx) + self.ctx_mask_proj(ctx_mask)
        x = self.prenet(torch.cat([x, x_txt], dim=-1))

        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_emb = self.f5_time_embed(timesteps)

        if (t_emb is not None) and (t_emb.ndim == 2) and (t_emb.shape[0] != x.shape[0]):
            if t_emb.shape[0] == 1:
                t_emb = t_emb.expand(x.shape[0], -1)
            elif x.shape[0] % t_emb.shape[0] == 0:
                t_emb = t_emb.repeat(x.shape[0] // t_emb.shape[0], 1)
            else:
                raise ValueError(f"t_emb batch mismatch: got {t_emb.shape[0]}, expected {x.shape[0]}")

        bgm_flag = cond.get('bgm_flag', None)
        if bgm_flag is not None and hasattr(self, 'bgm_flag_embed'):
            t_emb = t_emb + self.bgm_flag_embed(bgm_flag)
        quality_flag = cond.get('quality_flag', None)
        if quality_flag is not None and hasattr(self, 'quality_flag_embed'):
            t_emb = t_emb + self.quality_flag_embed(quality_flag)

        use_moe = bool(getattr(self.hp, "use_moe_ffn", False))
        pred_v = self.encoder(
            x, t_emb, attn_mask=attn_mask,
            context=caption_embs,
            context_lens=caption_lens,
            do_checkpoint=self.hp.do_checkpoint
        )
        if use_moe:
            pred_v = pred_v[0]

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
    def inference(
        self, inputs, timesteps=20, seq_cfg_w=[1.5, 3.0], timestep_annealing_w=(0.6, 0.6, 1.0), 
        use_amo_sampler=False, use_sway=True, return_timesteps=False, **kwargs
    ):
        tgt_len = inputs['tgt_len']     # reference + target
        x_mask = sequence_mask(tgt_len, maxlen=inputs['ctx_mask'].shape[1])
        
        x_txt = self.forward_text_encoder(inputs, x_mask)

        (bsz, tgt_len, _), device = x_txt.shape, x_txt.device
        bsz = bsz // 3

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        caption_ctx, caption_ctx_lens = self._encode_caption_context(
            inputs.get('caption_emb', None),
            inputs.get('caption_lens', None),
            inputs.get('caption_ids', None),
            inputs.get('caption_text_mark', None),
        )

        cond = {
            'ctx': ctx_feature,
            'ctx_mask': ctx_mask,
            'attn_mask': x_mask,
            'x_txt': x_txt,
            'txt_lens': inputs.get('txt_lens', inputs['txt_mask'].sum(1)),
            '_caption_ctx': caption_ctx,
            '_caption_ctx_lens': caption_ctx_lens,
            'bgm_flag': inputs.get('bgm_flag', None),
            'quality_flag': inputs.get('quality_flag', None),
        }

        ''' Euler ODE solver '''
        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
        if use_sway:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

        if use_amo_sampler:

            def amo_sampling(sample, sigma, sigma_next, pred_v):
                # sample: [1,T,C] ; pred_v: [1,T,C]（CFG 后）
                t = sigma
                s = sigma_next
                x_t = sample

                c = 3.0
                o = torch.clamp(s + c * (s - t), max=1.0)

                pred_x_o = x_t + (o - t) * pred_v
                a = s / o
                b = torch.sqrt(torch.clamp_min((1 - s) ** 2 - (a * (1 - o)) ** 2, 0.0))

                noises = torch.randn_like(x_t)
                prev_sample = a * pred_x_o + b * noises
                prev_sample = prev_sample.to(pred_v.dtype)
                return prev_sample

            x = torch.randn([bsz, tgt_len, self.hp.out_channels], device=device)
            for step_index in range(timesteps):
                sigma = t_schedule[step_index].to(x_txt.dtype)
                sigma_next = t_schedule[step_index + 1].to(x_txt.dtype)

                model_out = self._forward(
                    torch.cat([x] * 3),
                    cond,
                    timesteps=sigma.unsqueeze(0),
                    seq_cfg_w=seq_cfg_w,
                    timestep_annealing_w=timestep_annealing_w,
                )
                x = amo_sampling(x, sigma, sigma_next, model_out)

            return x

        else:
            traj = torchdiffeq.odeint(
                lambda t, x: self._forward(
                    torch.cat([x] * 3),
                    cond,
                    timesteps=t.unsqueeze(0),
                    seq_cfg_w=seq_cfg_w,
                    timestep_annealing_w=timestep_annealing_w,
                ),
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


class PostTrainNFTWrapper(Diffusion):
    def forward(
        self,
        inputs,
        sigmas=None,
        x_noisy=None,
        seq_cfg_w=None,
        timestep_annealing_w=(0.6, 0.6, 1.0),
    ):
        # RL / custom-step path: externally provided (x_noisy, sigmas)
        # - This path is FSDP-safe because it runs inside `forward()`.
        if (sigmas is not None) and (x_noisy is not None):
            tgt_len = inputs.get('tgt_len', None)
            if tgt_len is None:
                raise ValueError("inputs['tgt_len'] is required when (sigmas, x_noisy) are provided")

            x_mask = sequence_mask(tgt_len, maxlen=x_noisy.shape[1])
            x_txt = self.forward_text_encoder(inputs, x_mask)

            ctx_mask = inputs['ctx_mask']
            ctx_feature = inputs['lat_ctx'] * ctx_mask

            caption_ctx, caption_ctx_lens = self._encode_caption_context(
                inputs.get('caption_emb', None),
                inputs.get('caption_lens', None),
                inputs.get('caption_ids', None),
                inputs.get('caption_text_mark', None),
            )

            bgm_flag = inputs.get('bgm_flag', None)
            if (bgm_flag is not None) and hasattr(self, 'bgm_flag_embed'):
                if not torch.is_tensor(bgm_flag):
                    bgm_flag = torch.as_tensor(bgm_flag, device=x_txt.device)
                bgm_flag = bgm_flag.to(device=x_txt.device, dtype=torch.long).view(-1)
                bgm_flag = bgm_flag.clamp(0, 2)
                if (x_noisy.shape[0] % 3 == 0) and (bgm_flag.numel() * 3 == x_noisy.shape[0]):
                    bgm_flag = torch.cat([
                        bgm_flag,
                        bgm_flag,
                        torch.full_like(bgm_flag, 2),
                    ], dim=0)
                elif bgm_flag.numel() == 1 and x_noisy.shape[0] != 1:
                    bgm_flag = bgm_flag.expand(x_noisy.shape[0])
            else:
                bgm_flag = None

            cond = {
                'ctx': ctx_feature,
                'ctx_mask': ctx_mask,
                'attn_mask': x_mask,
                'x_txt': x_txt,
                '_caption_ctx': caption_ctx,
                '_caption_ctx_lens': caption_ctx_lens,
                'bgm_flag': bgm_flag,
            }

            if seq_cfg_w is None:
                seq_cfg_w = [1.5, 3.0]

            pred = self._forward(
                x_noisy,
                cond,
                sigmas,
                seq_cfg_w=seq_cfg_w,
                timestep_annealing_w=timestep_annealing_w,
            )
            return pred
        
        return super().forward(inputs)
