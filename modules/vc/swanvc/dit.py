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
from modules.commons.hf.transformer_dit import AdaLayerNormZero, AdaLayerNormZero_Final, TransformerDiTModel
from modules.commons.engram import NGramEngram

from utils.nn.seq_utils import sequence_mask

logger = logging.getLogger(__name__)

ForwardTask = Literal["training", "encode_cond", "denoise_step"]


@dataclass
class ModelArgs:
    # semantic token
    vocab_size: int = None
    patch_size: int = 2

    in_channels: int = 48
    out_channels: int = 48

    # transformer
    encoder_dim: int = 1024
    encoder_ffn_mult: float = 4.0
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = 8
    use_gated_attention: bool = False

    # training
    do_checkpoint: bool = False
    attn_implementation: str = "flash_attention_2"
    torch_compile_enabled: bool = False

    cfg_mask_token: int = None


class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp
        self._in_forward: bool = False

        config_dict = dict(
            hidden_size=hp.encoder_dim,
            intermediate_size=int(hp.encoder_dim * hp.encoder_ffn_mult),
            num_hidden_layers=hp.encoder_n_layers,
            num_attention_heads=hp.encoder_n_heads,
            num_key_value_heads=hp.encoder_n_kv_heads,
            use_gated_attention=hp.use_gated_attention,
            attn_implementation=hp.attn_implementation,
        )
        self.encoder = TransformerDiTModel(TransformerDiTConfig(**config_dict))

        if hp.do_checkpoint:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hp.torch_compile_enabled:
            self.encoder = torch.compile(self.encoder, fullgraph=False, dynamic=False, mode="default")

        self.prenet = nn.Linear(self.hp.encoder_dim * 2, self.hp.encoder_dim)
        self.lat_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.ctx_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.ctx_mask_proj = nn.Linear(1, self.hp.encoder_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)

        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

        self.semantic_embed = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.semantic_patch_merger = nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=hp.patch_size, stride=hp.patch_size)

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

    def _encode_semantic(self, inputs: Dict[str, Any], x_mask: torch.Tensor) -> torch.Tensor:
        self._require_in_forward("_encode_semantic")

        tgt_len = int(x_mask.shape[1])
        semantic_tokens = inputs["semantic_tokens"]
        semantic_mask = inputs["semantic_mask"]
        bsz = int(semantic_tokens.shape[0])

        semantic_emb = self.semantic_embed(semantic_tokens)     # [B, T, C]
        semantic_emb = self.semantic_patch_merger(semantic_emb.transpose(1, 2)).transpose(1, 2)

        return semantic_emb
    
    def _encode_cond(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._require_in_forward("_encode_cond_impl")

        tgt_len = inputs["tgt_len"]
        attn_mask = sequence_mask(tgt_len)

        semantic_emb = self._encode_semantic(inputs, attn_mask)

        ctx_mask = inputs["ctx_mask"]
        ctx = inputs["lat_ctx"] * ctx_mask

        return {
            "attn_mask": attn_mask,
            "semantic_emb": semantic_emb,
            "ctx_mask": ctx_mask,
            "ctx": ctx,
        }

    def _core_dit(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        semantic_emb: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        self._require_in_forward("_core_dit")

        x = x * (1 - ctx_mask)
        x = self.lat_proj(x) + self.ctx_proj(ctx) + self.ctx_mask_proj(ctx_mask)
        x = self.prenet(torch.cat([x, semantic_emb], dim=-1))

        t_emb = self._time_embed(t, batch=int(x.shape[0]))

        out = self.encoder(
            inputs_embeds=x,
            attention_mask=attn_mask,
            time_step=t_emb,
        )
        pred = self.postnet(out.last_hidden_state)

        return pred

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

                semantic_emb = self._encode_semantic(inputs, x_mask)

                pred = self._core_dit(
                    x=x_in,
                    t=t,
                    ctx=ctx,
                    ctx_mask=ctx_mask,
                    semantic_emb=semantic_emb,
                    attn_mask=x_mask,
                )

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

                pred_3b = self._core_dit(
                    x=x_model,
                    t=t,
                    ctx=cond["ctx"],
                    ctx_mask=cond["ctx_mask"],
                    semantic_emb=cond["semantic_emb"],
                    attn_mask=cond["attn_mask"],
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

        total_bsz = int(cond["semantic_emb"].shape[0])
        tgt_len = int(cond["semantic_emb"].shape[1])
        device = cond["semantic_emb"].device

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
