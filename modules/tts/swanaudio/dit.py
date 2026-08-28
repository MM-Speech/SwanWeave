import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union
import torch
from torch import nn
import torchdiffeq
# from modules.commons.hf.transformer_dit_moe import TransformerMoeDiTModel
from modules.commons.hf.transformer_dit_config import TransformerDiTConfig
from modules.commons.hf.transformer import RMSNorm, TransformerEncoderModel
from modules.commons.hf.transformer_config import TransformerConfig
from modules.commons.hf.transformer_dit import AdaLayerNormZero, AdaLayerNormZero_Final
from modules.commons.engram import NGramEngram

from utils.nn.seq_utils import sequence_mask

logger = logging.getLogger(__name__)

ForwardTask = Literal["training", "encode_cond", "denoise_step"]


@dataclass
class ModelArgs:
    # text
    vocab_size: int = None
    text_dim: int = 1024
    text_encoder_n_layers: int = 8
    text_encoder_n_heads: int = 16

    # transformer
    encoder_dim: int = 1024
    encoder_ffn_mult: float = 4.0
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = 8

    caption_dim: int = 5120

    in_channels: int = 96
    out_channels: int = 96

    # training
    do_checkpoint: bool = False
    attn_implementation: str = "flash_attention_2"
    torch_compile_enabled: bool = False

    cfg_mask_text_token: int = None
    text_fill_token: int = None
    use_caption_pool_in_adaln: bool = False
    use_caption_text_mark: bool = False
    use_spk_mask: bool = False
    use_bgm_flag: bool = False
    use_quality_flag: bool = False

    use_gated_attention: bool = False
    use_dynamic_cross_gate: bool = False

    use_moe_ffn: bool = False
    moe_p: float = 0.7
    moe_num_routed: int = 8
    moe_num_shared: int = 2
    moe_num_null: int = 4
    moe_use_gumbel: bool = False
    moe_gumbel_tau_start: float = 1.0
    moe_gumbel_tau_end: float = 0.3
    moe_gumbel_tau_anneal_steps: int = 200_000
    moe_expert_dropout: float = 0.0
    moe_capacity_factor_min: float = 1.0
    moe_capacity_factor_max: float = 2.0
    moe_overflow_drop: bool = True
    moe_use_t_budget: bool = True
    moe_p_min: float = 0.4
    moe_p_max: float = 0.95
    moe_null_logit_bias_min: float = -2.0
    moe_null_logit_bias_max: float = 0.0
    moe_load_balance_loss_coef: float = 1e-2
    moe_router_z_loss_coef: float = 1e-3
    moe_use_ec: bool = False
    decoder_sparse_step: int = 1
    ec_early_dense_moe_layers: int = 2
    ec_early_capacity_boost_moe_layers: int = 2
    ec_early_capacity_ratio: float = None
    ec_default_step_ratio: float = 0.5

    # ===== Engram switch =====
    use_engram: bool = False
    engram_target: Literal["caption", "text", "both"] = "caption"

    # hashed N-gram memory (simplified Engram)
    engram_orders: Tuple[int, ...] = (1, 2, 3)      # paper常用{2,3}:contentReference[oaicite:2]{index=2}
    engram_num_heads: int = 4                   # 多头hash思想:contentReference[oaicite:3]{index=3}
    engram_table_size: int = 262144             # 2^18，先别太大（显存）
    engram_head_dim: int = 64                   # 每个 (order,head) 的embedding维度
    engram_dropout: float = 0.0
    engram_gate_bias: float = -4.0              # sigmoid(-4)≈0.018，冷启动更稳
    engram_conv_kernel: int = 3                 # 可选轻量conv（论文里也提到）:contentReference[oaicite:4]{index=4}

    # special token handling
    engram_special_replace_id: int = 1          # 把各种 special token 统一替换成一个ID（避免n-gram爆炸）


class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp
        self._in_forward: bool = False

        config_dict = dict(
            hidden_size=hp.encoder_dim,
            intermediate_size=int(hp.encoder_dim * hp.encoder_ffn_mult),
            num_hidden_layers=hp.encoder_n_layers,
            num_cross_attention_layers=hp.encoder_n_layers,
            num_attention_heads=hp.encoder_n_heads,
            num_key_value_heads=hp.encoder_n_kv_heads,
            use_dynamic_cross_gate=hp.use_dynamic_cross_gate,
            use_gated_attention=hp.use_gated_attention,
            use_caption_pool_in_adaln=hp.use_caption_pool_in_adaln,
            attn_implementation=hp.attn_implementation,
        )

        if hp.moe_use_ec:
            from modules.tts.swanaudio.swan_dit_ecmoe import TransformerEcMoeDiTModel
            config_dict.update(dict(
                use_moe_ffn=hp.use_moe_ffn,
                decoder_sparse_step=hp.decoder_sparse_step,
                moe_num_routed=hp.moe_num_routed,
                moe_num_shared=hp.moe_num_shared,
                moe_capacity_factor_min=hp.moe_capacity_factor_min,
                moe_capacity_factor_max=hp.moe_capacity_factor_max,
                ec_early_dense_moe_layers=hp.ec_early_dense_moe_layers,
                ec_early_capacity_boost_moe_layers=hp.ec_early_capacity_boost_moe_layers,
                ec_early_capacity_ratio=hp.ec_early_capacity_ratio,
                ec_default_step_ratio=hp.ec_default_step_ratio,
            ))
            self.encoder = TransformerEcMoeDiTModel(TransformerDiTConfig(**config_dict))
        else:
            from modules.tts.swanaudio.swan_dit_moe import TransformerMoeDiTModel
            config_dict.update(dict(
                use_moe_ffn=hp.use_moe_ffn,
                moe_p=hp.moe_p,
                moe_num_routed=hp.moe_num_routed,
                moe_num_shared=hp.moe_num_shared,
                moe_num_null=hp.moe_num_null,
                moe_use_gumbel=hp.moe_use_gumbel,
                moe_gumbel_tau_start=hp.moe_gumbel_tau_start,
                moe_gumbel_tau_end=hp.moe_gumbel_tau_end,
                moe_gumbel_tau_anneal_steps=hp.moe_gumbel_tau_anneal_steps,
                moe_expert_dropout=hp.moe_expert_dropout,
                moe_capacity_factor_min=hp.moe_capacity_factor_min,
                moe_capacity_factor_max=hp.moe_capacity_factor_max,
                moe_overflow_drop=hp.moe_overflow_drop,
                moe_use_t_budget=hp.moe_use_t_budget,
                moe_p_min=hp.moe_p_min,
                moe_p_max=hp.moe_p_max,
                moe_null_logit_bias_min=hp.moe_null_logit_bias_min,
                moe_null_logit_bias_max=hp.moe_null_logit_bias_max,
                moe_load_balance_loss_coef=hp.moe_load_balance_loss_coef,
                moe_router_z_loss_coef=hp.moe_router_z_loss_coef,
                output_router_logits=True, 
            ))
            self.encoder = TransformerMoeDiTModel(TransformerDiTConfig(**config_dict))

        if hp.do_checkpoint:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hp.torch_compile_enabled:
            self.encoder = torch.compile(self.encoder, fullgraph=False, dynamic=False, mode="default")

        self.prenet = nn.Linear(self.hp.encoder_dim * 2, self.hp.encoder_dim)

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

        self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.text_encoder = TransformerEncoderModel(
            TransformerConfig(
                hidden_size=hp.encoder_dim,
                intermediate_size=hp.encoder_dim * 3,
                num_hidden_layers=hp.text_encoder_n_layers,
                num_attention_heads=hp.text_encoder_n_heads,
                num_key_value_heads=hp.text_encoder_n_heads // 2,
                attn_implementation=hp.attn_implementation,
                use_cache=False,
            )
        )
        if hp.do_checkpoint:
            self.text_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hp.torch_compile_enabled:
            self.text_encoder = torch.compile(self.text_encoder, fullgraph=False, dynamic=False, mode="default")

        if hp.use_spk_mask:
            self.spk_mask_embedder = nn.Embedding(5, hp.encoder_dim)

        # ===== Engram (optional) =====
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

    def _require_in_forward(self, fn_name: str) -> None:
        if not self._in_forward:
            raise RuntimeError(
                f"{fn_name} contains parameterized ops and must be called via Diffusion.forward(...). "
                f"For FSDP safety, use forward(task='encode_cond'/'denoise_step'/'training') "
                f"or the wrappers encode_cond()/denoise_step()/training_forward()."
            )

    def _init_weights(self) -> None:
        init_std = 0.02 / math.sqrt(2 * self.hp.encoder_n_layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif hasattr(module, "reset_packed_weights"):
                module.reset_packed_weights(init_std)

        nn.init.normal_(self.f5_time_embed.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.f5_time_embed.time_mlp[2].weight, std=0.02)

        for block in self.encoder.layers:
            nn.init.zeros_(block.input_layernorm.linear.weight)
            nn.init.zeros_(block.input_layernorm.linear.bias)

        nn.init.zeros_(self.encoder.norm.linear.weight)
        nn.init.zeros_(self.encoder.norm.linear.bias)

    def _get_safe_bf16_cast_policy(self) -> Dict[str, Any]:
        if bool(getattr(self.hp, "use_moe_ffn", False)) and not bool(getattr(self.hp, "moe_use_ec", False)):
            raise NotImplementedError("cast_safe_params_to_bf16 currently supports dense and ECMoE models only")

        keep_fp32_module_types = (RMSNorm, AdaLayerNormZero, AdaLayerNormZero_Final)
        keep_fp32_module_suffixes = {"f5_time_embed", "cross_gating_proj", "cap_proj", "cap_gate"}
        keep_fp32_param_suffixes = {"cross_gate"}

        if bool(getattr(self.hp, "moe_use_ec", False)):
            keep_fp32_module_suffixes.add("mlp.gate")

        return {
            "keep_fp32_module_types": keep_fp32_module_types,
            "keep_fp32_module_suffixes": keep_fp32_module_suffixes,
            "keep_fp32_param_suffixes": keep_fp32_param_suffixes,
        }

    def cast_safe_params_to_bf16(self) -> Dict[str, int]:
        policy = self._get_safe_bf16_cast_policy()
        keep_fp32_names = set()

        for module_name, module in self.named_modules():
            keep_by_type = isinstance(module, policy["keep_fp32_module_types"])
            keep_by_name = any(module_name.endswith(s) for s in policy["keep_fp32_module_suffixes"])
            if not (keep_by_type or keep_by_name):
                continue
            for param_name, _ in module.named_parameters(recurse=True):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                keep_fp32_names.add(full_name)

        for param_name, _ in self.named_parameters():
            if any(param_name.endswith(s) for s in policy["keep_fp32_param_suffixes"]):
                keep_fp32_names.add(param_name)

        num_bf16 = 0
        num_fp32 = 0
        for param_name, param in self.named_parameters():
            if not param.is_floating_point():
                continue
            target_dtype = torch.float32 if param_name in keep_fp32_names else torch.bfloat16
            if param.dtype != target_dtype:
                param.data = param.data.to(dtype=target_dtype)
            if param.grad is not None and param.grad.is_floating_point() and param.grad.dtype != target_dtype:
                param.grad.data = param.grad.data.to(dtype=target_dtype)
            if target_dtype == torch.bfloat16:
                num_bf16 += param.numel()
            else:
                num_fp32 += param.numel()

        return {"bf16_params": num_bf16, "fp32_params": num_fp32}

    def _time_embed(self, t: torch.Tensor, *, batch: int) -> torch.Tensor:
        self._require_in_forward("_time_embed")

        if not isinstance(t, torch.Tensor):
            raise TypeError("t must be a torch.Tensor")

        if t.ndim == 0:
            t = t.view(1)
        if t.shape[0] == 1 and batch != 1:
            t = t.expand(batch)
        elif t.shape[0] != batch:
            raise ValueError(f"timesteps batch mismatch: got {t.shape[0]}, expected {batch}")

        device_type = t.device.type
        enabled = device_type == "cuda"
        with torch.amp.autocast(device_type=device_type, dtype=torch.float32, enabled=enabled):
            t_emb = self.f5_time_embed(t)
        return t_emb

    def _encode_caption(
        self, inputs: Dict[str, Any], *, max_len: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        self._require_in_forward("_encode_caption")

        caption_emb = inputs.get("caption_emb", None)
        if caption_emb is None:
            return None, None

        caption_lens = inputs.get("caption_lens", None)
        if caption_lens is None:
            raise KeyError("inputs['caption_lens'] is required when inputs['caption_emb'] is provided")

        caption_ctx = self.caption_proj(caption_emb)
        if max_len is None:
            max_len = int(caption_ctx.shape[1])
        caption_mask = sequence_mask(caption_lens, maxlen=max_len)

        # ===== Engram on caption (after proj, before cross-attn) =====
        if (self.caption_engram is not None) and ("caption_ids" in inputs):
            cap_ids = inputs["caption_ids"]  # [B, Tc]
            cap_ids = cap_ids[:, :max_len]

            # special token 统一替换：你在外面传入 special_ids 列表即可
            caption_ctx = caption_ctx + self.caption_engram(
                cap_ids,
                caption_mask,
                caption_ctx,
                special_replace_id=int(getattr(self.hp, "engram_special_replace_id", 1)),
            )

        if self.hp.use_caption_text_mark:
            caption_text_mark = inputs.get("caption_text_mark", None)
            if caption_text_mark is None:
                raise KeyError("inputs['caption_text_mark'] is required when hp.use_caption_text_mark=True")
            caption_ctx = caption_ctx + self.caption_text_mark_embed(caption_text_mark.long())

        return caption_ctx, caption_mask

    def _encode_text(self, inputs: Dict[str, Any], x_mask: torch.Tensor) -> torch.Tensor:
        self._require_in_forward("_encode_text")

        tgt_len = int(x_mask.shape[1])
        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        bsz = int(txt_tokens.shape[0])

        x_txt_ids = torch.full(
            (bsz, tgt_len),
            self.hp.text_fill_token,
            device=txt_tokens.device,
            dtype=txt_tokens.dtype,
        )
        fill_pos = sequence_mask(txt_mask.long().sum(1), tgt_len)
        x_txt_ids[fill_pos] = txt_tokens[txt_mask]

        token_emb = self.text_embedder(x_txt_ids.long())

        # ===== Engram on text tokens (before text_encoder) =====
        if self.text_engram is not None:
            # 把 fill_token / cfg_mask_token 当 special 统一替换，避免 n-gram 被这些token污染
            special_ids = [
                int(getattr(self.hp, "text_fill_token", -1)),
                int(getattr(self.hp, "cfg_mask_text_token", -1)),
            ]
            token_emb = token_emb + self.text_engram(
                x_txt_ids,          # [B, tgt_len]
                x_mask,             # [B, tgt_len] bool
                token_emb,          # gate based on current embedding
                special_ids=[s for s in special_ids if s >= 0],
                special_replace_id=int(getattr(self.hp, "engram_special_replace_id", 1)),
            )

        if self.hp.use_spk_mask and ("spk_mask" in inputs) and (inputs["spk_mask"] is not None):
            spk_mask = inputs["spk_mask"]
            spk_ids = torch.zeros((bsz, tgt_len), device=txt_tokens.device, dtype=torch.long)
            spk_ids[fill_pos] = spk_mask[txt_mask].long().clamp_(0, 4)
            token_emb = token_emb + self.spk_mask_embedder(spk_ids)

        x_txt = self.text_encoder(inputs_embeds=token_emb, attention_mask=x_mask).last_hidden_state
        return x_txt
    
    def _encode_cond(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._require_in_forward("_encode_cond_impl")

        tgt_len = inputs["tgt_len"]
        attn_mask = sequence_mask(tgt_len)

        x_txt = self._encode_text(inputs, attn_mask)

        ctx_mask = inputs["ctx_mask"]
        ctx = inputs["lat_ctx"] * ctx_mask

        caption_ctx, caption_mask = self._encode_caption(inputs)

        return {
            "attn_mask": attn_mask,
            "x_txt": x_txt,
            "ctx_mask": ctx_mask,
            "ctx": ctx,
            "caption_ctx": caption_ctx,
            "caption_mask": caption_mask,
            "bgm_flag": inputs.get("bgm_flag", None),
            "quality_flag": inputs.get("quality_flag", None),
        }

    def _core_dit(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        x_txt: torch.Tensor,
        attn_mask: torch.Tensor,
        caption_ctx: Optional[torch.Tensor] = None,
        caption_mask: Optional[torch.Tensor] = None,
        bgm_flag: Optional[torch.Tensor] = None,
        quality_flag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        self._require_in_forward("_core_dit")

        x = x * (1 - ctx_mask)
        x = self.lat_proj(x) + self.ctx_proj(ctx) + self.ctx_mask_proj(ctx_mask)
        x = self.prenet(torch.cat([x, x_txt], dim=-1))

        t_emb = self._time_embed(t, batch=int(x.shape[0]))

        if bgm_flag is not None and hasattr(self, 'bgm_flag_embed'):
            t_emb = t_emb + self.bgm_flag_embed(bgm_flag)
        if quality_flag is not None and hasattr(self, 'quality_flag_embed'):
            t_emb = t_emb + self.quality_flag_embed(quality_flag)

        out = self.encoder(
            inputs_embeds=x,
            attention_mask=attn_mask,
            time_step=t_emb,
            encoder_hidden_states=caption_ctx,
            encoder_attention_mask=caption_mask,
        )
        pred = self.postnet(out.last_hidden_state)

        moe_aux = None
        if bool(getattr(self.hp, "use_moe_ffn", False)):
            moe_aux = getattr(out, "router_logits", None)

        return pred, moe_aux

    def _apply_cfg(
        self,
        pred_3b: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        seq_cfg_w: Tuple[float, float],
        timestep_annealing_w: Tuple[float, float, float],
    ) -> torch.Tensor:
        if pred_3b.shape[0] % 3 != 0:
            return pred_3b

        bsz = pred_3b.shape[0] // 3
        cond_all, cond_txt, uncond = pred_3b.chunk(3, dim=0)

        t = timesteps
        if t.ndim == 0:
            t = t.view(1)
        if t.shape[0] == pred_3b.shape[0]:
            t = t[:bsz]
        elif t.shape[0] == 1:
            t = t.expand(bsz)
        elif t.shape[0] != bsz:
            raise ValueError(f"timesteps shape {tuple(t.shape)} incompatible with CFG batch {pred_3b.shape[0]}")

        if t.ndim == 1:
            t = t[:, None, None]

        a, b, p = timestep_annealing_w
        gamma_t = a + b * torch.pow(1 - t, p)

        w_txt = gamma_t * float(seq_cfg_w[0])
        w_all = gamma_t * float(seq_cfg_w[1])

        return uncond + w_txt * (cond_txt - uncond) + w_all * (cond_all - cond_txt)

    def forward(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        x_noisy: Optional[torch.Tensor] = None,
        *,
        task: ForwardTask = "training",
        x: Optional[torch.Tensor] = None,
        cond: Optional[Dict[str, Any]] = None,
        timesteps: Optional[Union[torch.Tensor, float]] = None,
        seq_cfg_w: Tuple[float, float] = (1.5, 3.0),
        timestep_annealing_w: Tuple[float, float, float] = (0.6, 0.6, 1.0),
    ):
        prev = self._in_forward
        self._in_forward = True
        try:
            if task == "encode_cond":
                if inputs is None:
                    raise ValueError("forward(task='encode_cond') requires inputs")
                return self._encode_cond(inputs)

            if task == "training":
                if inputs is None:
                    raise ValueError("forward(task='training') requires inputs")

                ctx_mask = inputs["ctx_mask"]
                ctx = inputs["lat_ctx"] * ctx_mask
                x_gt = inputs["lat"]

                x_mask = sequence_mask(inputs["lat_lens"], maxlen=int(x_gt.shape[1]))

                x0 = torch.randn_like(x_gt)
                t = self.flow_matcher.time_sampler.sample([x0.shape[0]], x0.device).type_as(x0)
                xt = t[:, None, None] * x_gt + (1 - t[:, None, None]) * x0
                target = x_gt - x0

                if x_noisy is None:
                    x_in = xt * (1 - ctx_mask)
                else:
                    x_in = x_noisy

                x_in = x_in.to(dtype=self.lat_proj.weight.dtype)

                x_txt = self._encode_text(inputs, x_mask)
                caption_ctx, caption_mask = self._encode_caption(inputs)

                pred, moe_aux = self._core_dit(
                    x=x_in,
                    t=t,
                    ctx=ctx,
                    ctx_mask=ctx_mask,
                    x_txt=x_txt,
                    attn_mask=x_mask,
                    caption_ctx=caption_ctx,
                    caption_mask=caption_mask,
                    bgm_flag=inputs.get("bgm_flag", None),
                    quality_flag=inputs.get("quality_flag", None),
                )

                if bool(getattr(self.hp, "use_moe_ffn", False)):
                    return pred, target, moe_aux
                return pred, target

            if task == "denoise_step":
                if x is None:
                    raise ValueError("forward(task='denoise_step') requires x")
                if timesteps is None:
                    raise ValueError("forward(task='denoise_step') requires timesteps")

                if cond is None:
                    if inputs is None:
                        raise ValueError("forward(task='denoise_step') requires cond or inputs")
                    cond = self._encode_cond(inputs)

                cond_bsz = int(cond["ctx_mask"].shape[0])
                x_bsz = int(x.shape[0])

                if cond_bsz % 3 == 0:
                    base_bsz = cond_bsz // 3
                    if x_bsz == base_bsz:
                        x_model = torch.cat([x, x, x], dim=0)
                    elif x_bsz == cond_bsz:
                        x_model = x
                    else:
                        raise ValueError(f"x batch {x_bsz} is incompatible with cond batch {cond_bsz}")
                else:
                    base_bsz = cond_bsz
                    if x_bsz != base_bsz:
                        raise ValueError(f"x batch {x_bsz} is incompatible with cond batch {cond_bsz}")
                    x_model = x

                if isinstance(timesteps, torch.Tensor):
                    t = timesteps.to(device=x_model.device, dtype=x_model.dtype)
                else:
                    t = torch.tensor(timesteps, device=x_model.device, dtype=x_model.dtype)

                pred_3b, _ = self._core_dit(
                    x=x_model,
                    t=t,
                    ctx=cond["ctx"],
                    ctx_mask=cond["ctx_mask"],
                    x_txt=cond["x_txt"],
                    attn_mask=cond["attn_mask"],
                    caption_ctx=cond.get("caption_ctx", None),
                    caption_mask=cond.get("caption_mask", None),
                    bgm_flag=cond.get("bgm_flag", None),
                    quality_flag=cond.get("quality_flag", None),
                )

                if cond_bsz % 3 == 0:
                    return self._apply_cfg(
                        pred_3b,
                        t if isinstance(t, torch.Tensor) else torch.tensor(t, device=pred_3b.device),
                        seq_cfg_w=seq_cfg_w,
                        timestep_annealing_w=timestep_annealing_w,
                    )
                return pred_3b

            raise ValueError(f"Unknown task={task}")

        finally:
            self._in_forward = prev

    def training_forward(self, inputs: Dict[str, Any]):
        return self.forward(inputs, task="training")

    def encode_cond(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.forward(inputs, task="encode_cond")

    def build_denoise_cond(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.forward(inputs, task="encode_cond")

    def denoise_step(
        self,
        x: torch.Tensor,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        cond: Optional[Dict[str, Any]] = None,
        timesteps: Union[torch.Tensor, float],
        seq_cfg_w: Tuple[float, float] = (1.5, 3.0),
        timestep_annealing_w: Tuple[float, float, float] = (0.6, 0.6, 1.0),
    ) -> torch.Tensor:
        return self.forward(
            inputs,
            task="denoise_step",
            x=x,
            cond=cond,
            timesteps=timesteps,
            seq_cfg_w=seq_cfg_w,
            timestep_annealing_w=timestep_annealing_w,
        )

    def inference(
        self,
        inputs: Dict[str, Any],
        timesteps: int = 20,
        seq_cfg_w: Tuple[float, float] = (1.5, 3.0),
        timestep_annealing_w: Tuple[float, float, float] = (0.6, 0.6, 1.0),
        use_amo_sampler: bool = False,
        use_sway: bool = True,
        return_timesteps: bool = False,
        **kwargs,
    ):
        cond = self.forward(inputs, task="encode_cond")

        total_bsz = int(cond["x_txt"].shape[0])
        tgt_len = int(cond["x_txt"].shape[1])
        device = cond["x_txt"].device

        if total_bsz % 3 == 0:
            bsz = total_bsz // 3
        else:
            bsz = total_bsz

        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1, device=device)
        if use_sway:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

        if use_amo_sampler:

            def amo_sampling(sample, sigma, sigma_next, pred_v):
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
                return prev_sample.to(pred_v.dtype)

            x = torch.randn([bsz, tgt_len, self.hp.out_channels], device=device)
            for step_index in range(timesteps):
                sigma = t_schedule[step_index].to(x.dtype)
                sigma_next = t_schedule[step_index + 1].to(x.dtype)

                pred_v = self.forward(
                    task="denoise_step",
                    x=x,
                    cond=cond,
                    timesteps=sigma,
                    seq_cfg_w=seq_cfg_w,
                    timestep_annealing_w=timestep_annealing_w,
                )
                x = amo_sampling(x, sigma, sigma_next, pred_v)

        else:

            def ode_fn(t, x_state):
                return self.forward(
                    task="denoise_step",
                    x=x_state,
                    cond=cond,
                    timesteps=t,
                    seq_cfg_w=seq_cfg_w,
                    timestep_annealing_w=timestep_annealing_w,
                )

            traj = torchdiffeq.odeint(
                ode_fn,
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
