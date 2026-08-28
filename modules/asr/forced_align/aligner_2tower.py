import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import math
import json
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from modules.commons.hf.transformer import TransformerDecoderModel, TransformerEncoderModel
from modules.commons.hf.transformer_config import TransformerConfig
from modules.asr.nepa.nepa import build_nepa_model

from utils.nn.seq_utils import sequence_mask
from utils.nn.tf32_utils import tf32_enable
from utils.commons.tensor_utils import all_gather_varlen_tensor_stack, all_gather_varlen_tensor
from utils.commons.hparams import set_hparams
from utils.commons.ckpt_utils import load_ckpt

def build_text_tokenizer(hparams):
    from utils.text.cosyvoice2_tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    return tokenizer

def build_aligner_model(hparams, attn_implementation='flash_attention_2', init_pretrained=True):
    tokenizer = build_text_tokenizer(hparams)

    config = ModelArgs(
        sample_rate=hparams['audio_sample_rate'],

        vocab_size=tokenizer.encoding.n_vocab,
        wbd_token_id=tokenizer.encode('<|wbd|>')[0],

        nepa_ckpt=hparams.get('nepa_ckpt', 'checkpoints/260205_nepa'),
        init_pretrained=init_pretrained,
        freeze_nepa=hparams.get('freeze_nepa', False),

        audio_hidden_size=hparams.get('audio_hidden_size', 512),
        audio_num_hidden_layers=hparams.get('audio_num_hidden_layers', 4),
        audio_num_attention_heads=hparams.get('audio_num_attention_heads', 8),
        audio_num_key_value_heads=hparams.get('audio_num_key_value_heads', 8),

        text_hidden_size=hparams.get('text_hidden_size', 512),
        text_num_hidden_layers=hparams.get('text_num_hidden_layers', 16),
        text_num_attention_heads=hparams.get('text_num_attention_heads', 8),
        text_num_key_value_heads=hparams.get('text_num_key_value_heads', 8),

        hidden_size=hparams.get('hidden_size', 512),
        sim_type=hparams.get('sim_type', 'cos'),

        conf_min=hparams.get('conf_min', 0.2),
        use_frame_weight=hparams.get('use_frame_weight', True),

        pause_hidden_size=hparams.get('pause_hidden_size', 256),
        pause_threshold=hparams.get('pause_threshold', 0.5),
        pause_hard_gating=hparams.get('pause_hard_gating', True),

        blank_mlp_hidden_size=hparams.get('blank_mlp_hidden_size', 256),
        blank_dur_weight=hparams.get('blank_dur_weight', 0.5),

        # ===== NEW: zero-duration policy =====
        # 'drop' | 'keep_one_frame' | 'conf_split'
        zero_dur_policy=hparams.get('zero_dur_policy', 'drop'),
        zero_dur_conf_keep_thresh=hparams.get('zero_dur_conf_keep_thresh', 0.6),
        # 当没有 word_conf 时，conf_split 的回退策略
        zero_dur_no_conf_fallback=hparams.get('zero_dur_no_conf_fallback', 'drop'),

        # ===== NEW: monitor =====
        enable_monitor_stats=hparams.get('enable_monitor_stats', True),

        # ===== V5: transition potential + CRF =====
        v5_transition_hidden_size=hparams.get('v5_transition_hidden_size', 256),
        v5_transition_scale=hparams.get('v5_transition_scale', 1.0),
        v5_use_crf_loss=hparams.get('v5_use_crf_loss', True),
        v5_crf_loss_weight=hparams.get('v5_crf_loss_weight', 1.0),
        v5_enable_skip_pair_head=hparams.get('v5_enable_skip_pair_head', True),

        attn_implementation=attn_implementation,
        gradient_checkpointing=hparams.get('gradient_checkpointing', False),
    )

    model_version = str(hparams.get('model_version', 'aligner_2tower'))
    if model_version == "aligner_2tower_v6":
        model = ForcedAlignerV6(config)
    elif model_version == "aligner_2tower_v5":
        model = ForcedAlignerV5(config)
    elif model_version == "aligner_2tower_v4":
        model = ForcedAlignerV4(config)
    elif model_version == "aligner_2tower_v3":
        model = ForcedAlignerV3(config)
    elif model_version == "aligner_2tower_v2":
        model = ForcedAlignerV2(config)
    else:
        model = ForcedAligner(config)
    return model, tokenizer


@dataclass
class ModelArgs:
    # audio
    sample_rate: int = 16000
    hop_size: int = None  # will be filled from nepa.config.hop_size

    # text
    vocab_size: int = None
    wbd_token_id: int = None

    # pretrained audio encoder
    nepa_ckpt: str = None
    init_pretrained: bool = True
    freeze_nepa: bool = True

    # audio encoder
    audio_hidden_size: int = 512
    audio_num_hidden_layers: int = 2
    audio_num_attention_heads: int = 8
    audio_num_key_value_heads: int = 8

    # text encoder
    text_hidden_size: int = 512
    text_num_hidden_layers: int = 16
    text_num_attention_heads: int = 8
    text_num_key_value_heads: int = 8

    # aligner
    hidden_size: int = 512
    sim_type: str = 'cos'  # 'cos' | 'dot'

    attn_implementation: str = 'flash_attention_2'
    gradient_checkpointing: bool = False

    # training knobs
    conf_min: float = 0.2
    use_frame_weight: bool = True

    # V3: pause / blank gating
    pause_hidden_size: int = 256
    pause_threshold: float = 0.5
    pause_hard_gating: bool = True

    # V4: gap-aware blank prototypes + blank duration loss
    blank_mlp_hidden_size: int = 256
    blank_dur_weight: float = 0.5

    # ===== NEW: zero-duration handling =====
    zero_dur_policy: str = "conf_split"           # 'drop' | 'keep_one_frame' | 'conf_split'
    zero_dur_conf_keep_thresh: float = 0.6
    zero_dur_no_conf_fallback: str = "keep_one_frame"

    # ===== NEW: monitor =====
    enable_monitor_stats: bool = True

    # ===== NEW: V5 transition + CRF =====
    v5_transition_hidden_size: int = 256
    v5_transition_scale: float = 1.0
    v5_use_crf_loss: bool = True
    v5_crf_loss_weight: float = 1.0
    v5_enable_skip_pair_head: bool = True

    # ===== V6: score canonicalization + dual gamma + posterior decode =====
    v6_enable_unary_canon: bool = True
    v6_enable_trans_canon: bool = True
    v6_canon_apply_in_train: bool = True     # 先设 True 做 V6；如果想先只推理验证，可设 False
    v6_canon_eps: float = 1e-5
    v6_canon_clip: float = 8.0               # z-score 后可选裁剪，抑制极端值
    v6_canon_use_mad: bool = False           # 先用 std，简单稳定

    # dual gamma
    v6_log_gamma_unary_init: float = math.log(10.0)  # 复用你 V5 的习惯
    v6_log_gamma_trans_init: float = math.log(1.0)
    v6_gamma_unary_max: float = 30.0
    v6_gamma_trans_max: float = 10.0

    # posterior / MBR-like decode
    v6_default_decode_mode: str = "posterior"   # "viterbi" | "posterior"
    v6_posterior_decode_use_topology: bool = True   # posterior + 拓扑约束（推荐）
    v6_posterior_entropy_floor: float = 0.0   # 可先不用


class ForcedAligner(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        # Nepa
        if config.nepa_ckpt.endswith('.ckpt'):
            nepa_hp = set_hparams(os.path.join(Path(config.nepa_ckpt).parent, 'config.yaml'), global_hparams=False)
        else:
            nepa_hp = set_hparams(os.path.join(config.nepa_ckpt, 'config.yaml'), global_hparams=False)
        self.nepa = build_nepa_model(nepa_hp, attn_implementation=config.attn_implementation)
        if config.init_pretrained:
            load_ckpt(self.nepa, config.nepa_ckpt)

        self.config.hop_size = int(self.nepa.config.hop_size)

        if config.freeze_nepa:
            self.nepa.eval()
            for p in self.nepa.parameters():
                p.requires_grad = False

        self.audio_proj = nn.Linear(self.nepa.config.hidden_size, config.audio_hidden_size, bias=False)
        self.audio_encoder = TransformerEncoderModel(
            config=TransformerConfig(
                hidden_size=config.audio_hidden_size,
                num_hidden_layers=config.audio_num_hidden_layers,
                num_attention_heads=config.audio_num_attention_heads,
                num_key_value_heads=config.audio_num_key_value_heads,
                intermediate_size=config.audio_hidden_size * 4,
                use_cache=False,
                attn_implementation=config.attn_implementation,
            )
        )
        self.audio_out = nn.Linear(config.audio_hidden_size, config.hidden_size, bias=False)

        self.text_embed = nn.Embedding(config.vocab_size, config.text_hidden_size)
        self.text_encoder = TransformerEncoderModel(
            config=TransformerConfig(
                hidden_size=config.text_hidden_size,
                num_hidden_layers=config.text_num_hidden_layers,
                num_attention_heads=config.text_num_attention_heads,
                num_key_value_heads=config.text_num_key_value_heads,
                intermediate_size=config.text_hidden_size * 4,
                use_cache=False,
                attn_implementation=config.attn_implementation,
            )
        )
        self.text_out = nn.Linear(config.text_hidden_size, config.hidden_size, bias=False)

        self.blank_probe = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        self.blank_bias = nn.Parameter(torch.zeros([]))

        self.log_gamma = nn.Parameter(torch.zeros([]))

        if config.gradient_checkpointing and self.training:
            self.nepa.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.audio_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.text_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def _frame_sec(self) -> float:
        return float(self.config.hop_size) / float(self.config.sample_rate)

    def _encode_audio(self, wavs, wav_lens=None):
        if self.config.freeze_nepa:
            with torch.no_grad():
                feat, audio_mask = self.nepa(wavs, wav_lens)  # eval-mode => Tensor[B, T', C]
        else:
            feat, audio_mask = self.nepa(wavs, wav_lens)
            
        x = self.audio_proj(feat)
        x = self.audio_encoder(inputs_embeds=x, attention_mask=audio_mask).last_hidden_state
        x = self.audio_out(x)
        return x, audio_mask

    def _encode_text(self, txt_tokens, txt_lens=None):
        if txt_lens is None:
            txt_lens = torch.full((txt_tokens.shape[0],), txt_tokens.shape[1], device=txt_tokens.device, dtype=torch.long)
        else:
            txt_lens = txt_lens.long()

        txt_mask = sequence_mask(txt_lens, maxlen=txt_tokens.shape[1])  # bool [B, N]
        x = self.text_embed(txt_tokens.clamp_min(0).clamp_max(self.config.vocab_size - 1))
        x = self.text_encoder(inputs_embeds=x, attention_mask=txt_mask).last_hidden_state
        x = self.text_out(x)
        return x, txt_mask

    def encode(self, wavs, txt_tokens, wav_lens=None, txt_lens=None):
        audio_feat, audio_mask = self._encode_audio(wavs, wav_lens)
        text_feat, text_mask = self._encode_text(txt_tokens, txt_lens)
        return audio_feat, audio_mask, text_feat, text_mask

    def _gather_wbd_embeddings(
        self,
        text_feat: torch.Tensor,
        txt_tokens: torch.Tensor,
        txt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = text_feat.shape
        wbd_id = int(self.config.wbd_token_id)

        wbd_mask_raw = (txt_tokens == wbd_id) & txt_mask  # [B, N] bool
        wbd_lens = wbd_mask_raw.long().sum(dim=1)         # [B]
        Wmax = int(wbd_lens.max().detach().cpu().item()) if B > 0 else 0

        if Wmax == 0:
            wbd_feat = text_feat.new_zeros((B, 0, C))
            wbd_mask = torch.zeros((B, 0), device=text_feat.device, dtype=torch.bool)
            return wbd_feat, wbd_mask

        wbd_rank = wbd_mask_raw.long().cumsum(dim=1) - 1  # only valid where wbd_mask_raw is True

        b_idx, t_idx = torch.nonzero(wbd_mask_raw, as_tuple=True)  # [K], [K]
        r_idx = wbd_rank[b_idx, t_idx]                             # [K] in [0, Wmax)

        wbd_feat = text_feat.new_zeros((B, Wmax, C))
        wbd_feat[b_idx, r_idx] = text_feat[b_idx, t_idx]

        wbd_mask = (torch.arange(Wmax, device=text_feat.device)[None, :] < wbd_lens[:, None])
        return wbd_feat, wbd_mask

    def _prepare_word_time_supervision(
        self,
        audio_mask: torch.Tensor,                 # [B,T]
        wbd_mask: torch.Tensor,                   # [B,W]
        word_start_times: Optional[torch.Tensor],
        word_end_times: Optional[torch.Tensor],
        word_conf: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        统一处理 word 时间标注（含 zero-duration 策略），供 V1/V2/V3/V4 target builder 复用。

        返回 dict（常用字段）:
        - has_times: bool
        - st, ed: [B,W] float (padded/truncated)
        - valid_words: [B,W] bool   # 最终参与 word 监督的词
        - zero_eq: [B,W] bool       # end == start
        - zero_keep: [B,W] bool     # zero-duration 但保留为1帧监督
        - zero_drop: [B,W] bool     # zero-duration 且被视为缺失/忽略
        - start_idx, end_idx: [B,W] long  (valid_words 里 end_idx 已强制 >= start_idx+1)
        - conf_weight: [B,W] float or None  # 已按 conf_min clamp，且无效词置0
        - monitor: Dict[str, Tensor]
        """
        device = audio_mask.device
        B, T = audio_mask.shape
        W = int(wbd_mask.shape[1])

        audio_lens = audio_mask.long().sum(dim=1)
        frame_sec = float(self._frame_sec())

        # 默认输出（无时间监督）
        out = {
            "has_times": False,
            "audio_lens": audio_lens,
            "frame_sec": torch.tensor(frame_sec, device=device, dtype=torch.float32),
            "st": None,
            "ed": None,
            "valid_words": torch.zeros((B, W), device=device, dtype=torch.bool),
            "zero_eq": torch.zeros((B, W), device=device, dtype=torch.bool),
            "zero_keep": torch.zeros((B, W), device=device, dtype=torch.bool),
            "zero_drop": torch.zeros((B, W), device=device, dtype=torch.bool),
            "start_idx": torch.zeros((B, W), device=device, dtype=torch.long),
            "end_idx": torch.zeros((B, W), device=device, dtype=torch.long),
            "conf_weight": None,
            "monitor": {
                "mon_wbd_count": wbd_mask.float().sum().detach(),
                "mon_time_has_supervision": torch.tensor(
                    1.0 if (W > 0 and word_start_times is not None and word_end_times is not None) else 0.0,
                    device=device
                ),
                "mon_word_time_nonneg_count": torch.tensor(0.0, device=device),
                "mon_word_time_pos_count": torch.tensor(0.0, device=device),
                "mon_word_time_zeroeq_count": torch.tensor(0.0, device=device),
                "mon_word_time_zeroeq_keep_count": torch.tensor(0.0, device=device),
                "mon_word_time_zeroeq_drop_count": torch.tensor(0.0, device=device),
                "mon_word_valid_count": torch.tensor(0.0, device=device),
            },
        }

        if W == 0 or (word_start_times is None) or (word_end_times is None):
            return out

        # ---- pad / truncate st, ed ----
        st = word_start_times.to(device=device, dtype=torch.float32)
        ed = word_end_times.to(device=device, dtype=torch.float32)

        if st.ndim == 1:
            st = st[None, :]
        if ed.ndim == 1:
            ed = ed[None, :]

        if st.size(1) < W:
            st = F.pad(st, (0, W - st.size(1)), value=-1.0)
        else:
            st = st[:, :W]

        if ed.size(1) < W:
            ed = F.pad(ed, (0, W - ed.size(1)), value=-1.0)
        else:
            ed = ed[:, :W]

        # ---- optional conf pad ----
        conf_raw = None
        if word_conf is not None:
            conf_raw = word_conf.to(device=device, dtype=torch.float32)
            if conf_raw.ndim == 1:
                conf_raw = conf_raw[None, :]
            if conf_raw.size(1) < W:
                conf_raw = F.pad(conf_raw, (0, W - conf_raw.size(1)), value=1.0)
            else:
                conf_raw = conf_raw[:, :W]

        # ---- validity split ----
        base_nonneg = wbd_mask & (st >= 0.0) & (ed >= st)   # 含 ed == st
        pos_mask = wbd_mask & (st >= 0.0) & (ed > st)

        # zero-duration（考虑 float 噪声）
        zero_eq = base_nonneg & torch.isclose(ed, st, rtol=0.0, atol=1e-6)

        zero_keep = torch.zeros_like(zero_eq)
        zero_drop = torch.zeros_like(zero_eq)

        policy = str(getattr(self.config, "zero_dur_policy", "conf_split"))

        if policy == "drop":
            zero_drop = zero_eq
        elif policy == "keep_one_frame":
            zero_keep = zero_eq
        elif policy == "conf_split":
            if conf_raw is None:
                fallback = str(getattr(self.config, "zero_dur_no_conf_fallback", "keep_one_frame"))
                if fallback == "drop":
                    zero_drop = zero_eq
                else:
                    zero_keep = zero_eq
            else:
                thr = float(getattr(self.config, "zero_dur_conf_keep_thresh", 0.6))
                zero_keep = zero_eq & (conf_raw >= thr)
                zero_drop = zero_eq & (~zero_keep)
        else:
            raise ValueError(f"Unsupported zero_dur_policy={policy}")

        valid_words = pos_mask | zero_keep

        # ---- frame index ----
        start_idx = torch.round(st / frame_sec).long().clamp_min(0)
        end_idx = torch.round(ed / frame_sec).long().clamp_min(0)

        start_idx = torch.minimum(start_idx, audio_lens[:, None])
        end_idx = torch.minimum(end_idx, audio_lens[:, None])

        # 关键：如果 start 已经到 audio_lens，说明已经没有任何可分配帧了，必须 drop
        valid_words = valid_words & (start_idx < audio_lens[:, None])

        # 对最终 valid_words 强制至少 1 帧（包括 zero_keep）
        end_idx = torch.where(valid_words, torch.maximum(end_idx, start_idx + 1), end_idx)

        # （可选防御）再 clamp 一次，虽然上面 valid_words 限制后通常不需要
        end_idx = torch.minimum(end_idx, audio_lens[:, None])

        # ---- conf weight (for frame weighting only) ----
        conf_weight = None
        if conf_raw is not None:
            conf_weight = conf_raw.clamp(min=float(self.config.conf_min), max=1.0)
            conf_weight = torch.where(valid_words, conf_weight, torch.zeros_like(conf_weight))

        out.update({
            "has_times": True,
            "st": st,
            "ed": ed,
            "valid_words": valid_words,
            "zero_eq": zero_eq,
            "zero_keep": zero_keep,
            "zero_drop": zero_drop,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "conf_weight": conf_weight,
            "monitor": {
                "mon_wbd_count": wbd_mask.float().sum().detach(),
                "mon_time_has_supervision": torch.tensor(1.0, device=device),
                "mon_word_time_nonneg_count": base_nonneg.float().sum().detach(),
                "mon_word_time_pos_count": pos_mask.float().sum().detach(),
                "mon_word_time_zeroeq_count": zero_eq.float().sum().detach(),
                "mon_word_time_zeroeq_keep_count": zero_keep.float().sum().detach(),
                "mon_word_time_zeroeq_drop_count": zero_drop.float().sum().detach(),
                "mon_word_valid_count": valid_words.float().sum().detach(),
            },
        })
        return out

    def _build_frame_targets_from_times(
        self,
        audio_mask: torch.Tensor,
        word_start_times: Optional[torch.Tensor],
        word_end_times: Optional[torch.Tensor],
        word_conf: Optional[torch.Tensor],
        wbd_mask: torch.Tensor,
        *,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, T = audio_mask.shape
        device = audio_mask.device

        W = int(wbd_mask.shape[1])
        dur_tgt = torch.zeros((B, W), device=device, dtype=torch.float32)

        target = torch.full((B, T), fill_value=ignore_index, device=device, dtype=torch.long)
        target = torch.where(audio_mask, torch.zeros_like(target), target)  # valid frames => blank(0)

        weight = torch.where(
            audio_mask,
            torch.ones((B, T), device=device, dtype=torch.float32),
            torch.zeros((B, T), device=device, dtype=torch.float32)
        )

        prep = self._prepare_word_time_supervision(
            audio_mask=audio_mask,
            wbd_mask=wbd_mask,
            word_start_times=word_start_times,
            word_end_times=word_end_times,
            word_conf=word_conf,
        )

        monitor = dict(prep["monitor"])
        monitor["mon_audio_valid_frame_count"] = audio_mask.float().sum().detach()

        # 没有可用时间监督：保持全 blank target
        if W == 0 or (not prep["has_times"]):
            monitor["mon_target_word_frame_count"] = torch.tensor(0.0, device=device)
            monitor["mon_target_blank_frame_count"] = audio_mask.float().sum().detach()
            monitor["mon_target_word_frame_ratio"] = torch.tensor(0.0, device=device)
            monitor["mon_frame_weight_word_mean"] = torch.tensor(1.0, device=device)
            return target, weight, dur_tgt, monitor

        valid_words = prep["valid_words"]        # [B,W]
        start_idx = prep["start_idx"]            # [B,W]
        end_idx = prep["end_idx"]                # [B,W]
        conf_weight = prep["conf_weight"]        # [B,W] or None

        dur_tgt = ((end_idx - start_idx).clamp_min(0).float() * valid_words.float())
        dur_mask_tgt = valid_words.float()

        t_idx = torch.arange(T, device=device)[None, :, None]  # [1, T, 1]
        in_span = (
            valid_words[:, None, :]
            & (t_idx >= start_idx[:, None, :])
            & (t_idx < end_idx[:, None, :])
            & audio_mask[:, :, None]
        )  # [B, T, W] bool

        word_ids = torch.arange(1, W + 1, device=device, dtype=torch.long)[None, None, :]  # [1,1,W]
        word_target = torch.where(in_span, word_ids, 0).amax(dim=-1)  # [B,T]
        target = torch.where(audio_mask, word_target, target)

        if conf_weight is not None:
            conf_frame = torch.where(in_span, conf_weight[:, None, :], 0.0).amax(dim=-1)  # [B,T]
            weight = torch.where(word_target > 0, conf_frame, weight)

        # ---- monitor ----
        is_word_frame = (word_target > 0) & audio_mask
        is_blank_frame = (word_target == 0) & audio_mask

        monitor["mon_target_word_frame_count"] = is_word_frame.float().sum().detach()
        monitor["mon_target_blank_frame_count"] = is_blank_frame.float().sum().detach()
        monitor["mon_target_word_frame_ratio"] = (
            is_word_frame.float().sum() / audio_mask.float().sum().clamp_min(1.0)
        ).detach()

        if conf_weight is not None:
            monitor["mon_frame_weight_word_mean"] = (
                (weight * is_word_frame.float()).sum() / is_word_frame.float().sum().clamp_min(1.0)
            ).detach()
        else:
            monitor["mon_frame_weight_word_mean"] = torch.tensor(1.0, device=device)

        return target, weight, dur_tgt, dur_mask_tgt, monitor

    def _compute_alignment_logits(
        self,
        audio_feat: torch.Tensor,
        audio_mask: torch.Tensor,
        wbd_feat: torch.Tensor,
        wbd_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: [B, T, 1+W]，第 0 类是 blank，其余是 word(wbd)。
        """
        min_val = torch.finfo(audio_feat.dtype).min
        B, T, C = audio_feat.shape
        W = wbd_feat.shape[1]

        if self.config.sim_type == 'cos':
            a = F.normalize(audio_feat, dim=-1)
            w = F.normalize(wbd_feat, dim=-1)
        else:
            a, w = audio_feat, wbd_feat

        gamma = torch.exp(self.log_gamma).clamp(0.7, 3.0).to(audio_feat.dtype)

        blank_logit = (a * self.blank_probe.to(a.dtype)).sum(dim=-1, keepdim=True) + self.blank_bias.to(a.dtype)
        word_logits = torch.bmm(a, w.transpose(1, 2))  # [B, T, W]

        logits = torch.cat([blank_logit, word_logits], dim=-1) * gamma

        if W > 0:
            logits_words = logits[:, :, 1:]
            logits_words = logits_words.masked_fill(~wbd_mask[:, None, :], min_val)
            logits[:, :, 1:] = logits_words

        logits = logits.masked_fill(~audio_mask[:, :, None], min_val)
        return logits

    def forward(
        self,
        wavs: torch.Tensor,
        texts: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        text_lens: Optional[torch.Tensor] = None,
        *,
        word_start_times: Optional[torch.Tensor] = None,
        word_end_times: Optional[torch.Tensor] = None,
        word_conf: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练 forward：
          - wavs: [B, Twav]
          - texts: [B, Ttxt]
          - word_start_times/word_end_times/word_conf: [B, W] (seconds)
        """
        audio_feat, audio_mask, text_feat, text_mask = self.encode(wavs, texts, wav_lens, text_lens)

        wbd_feat, wbd_mask = self._gather_wbd_embeddings(text_feat, texts, text_mask)
        logits = self._compute_alignment_logits(audio_feat, audio_mask, wbd_feat, wbd_mask)  # [B,T,1+W]

        use_frame_weight = bool(getattr(self.config, "use_frame_weight", True))
        word_conf_for_targets = word_conf

        target, frame_weight, dur_tgt, dur_mask_tgt, target_monitor = self._build_frame_targets_from_times(
            audio_mask,
            word_start_times=word_start_times,
            word_end_times=word_end_times,
            word_conf=word_conf_for_targets,
            wbd_mask=wbd_mask,
        )

        if not use_frame_weight:
            frame_weight = torch.ones_like(frame_weight)

        B, T, C = logits.shape
        ignore_index = -100

        ce_per = F.cross_entropy(
            logits.float().reshape(-1, C),
            target.reshape(-1),
            reduction='none',
            ignore_index=ignore_index,
        ).view(B, T)

        valid = (target != ignore_index) & audio_mask
        denom = valid.float().sum().clamp_min(1.0)
        align_loss = (ce_per * frame_weight * valid.float()).sum() / denom

        probs = torch.softmax(logits.float(), dim=-1)  # [B,T,1+W]
        dur_loss = logits.new_tensor(0.0)
        mono_loss = logits.new_tensor(0.0)

        W = wbd_feat.shape[1]
        if W > 0:
            # duration loss（允许空隙：blank 不参与）
            dur_pred = (probs[:, :, 1:] * audio_mask[:, :, None].float()).sum(dim=1)  # [B,W]
            # dur_mask = wbd_mask.float()
            dur_mask = dur_mask_tgt

            l1 = (dur_pred - dur_tgt).abs()
            l1 = (l1 * dur_mask).sum() / dur_mask.sum().clamp_min(1.0)

            log_l2 = (torch.log1p(dur_pred.clamp_min(0.0)) - torch.log1p(dur_tgt.clamp_min(0.0))).pow(2)
            log_l2 = (log_l2 * dur_mask).sum() / dur_mask.sum().clamp_min(1.0)

            dur_loss = l1 + log_l2

            # monotonic loss（只看 word 概率，忽略 blank；允许 blank 插空）
            p_words = probs[:, :, 1:]  # [B,T,W]
            p_sum = p_words.sum(dim=-1).clamp_min(1e-6)  # [B,T]
            idx = torch.arange(1, W + 1, device=logits.device, dtype=torch.float32)[None, None, :]
            expected = (p_words * idx).sum(dim=-1) / p_sum  # [B,T]

            dec = F.relu(expected[:, :-1] - expected[:, 1:])  # [B,T-1]
            mono_mask = audio_mask[:, 1:] & audio_mask[:, :-1]
            mono_loss = (dec * mono_mask.float()).sum() / mono_mask.float().sum().clamp_min(1.0)

        # ---- monitor stats (debug/training logging) ----
        monitor_stats = {}
        if bool(getattr(self.config, "enable_monitor_stats", True)):
            valid = (target != ignore_index) & audio_mask

            # p(target)
            target_safe = target.clamp_min(0)
            p_target = probs.gather(dim=-1, index=target_safe.unsqueeze(-1)).squeeze(-1)
            valid_f = valid.float()
            monitor_stats["mon_p_target_mean"] = (
                (p_target * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            ).detach()

            # entropy over all classes
            ent = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)  # [B,T]
            monitor_stats["mon_entropy_valid_mean"] = (
                (ent * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            ).detach()

            # word mass（对 base/V1/V2/V3：类 1..W）
            word_mass = probs[:, :, 1:].sum(dim=-1)
            monitor_stats["mon_word_mass_valid_mean"] = (
                (word_mass * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            ).detach()

            # attach target-builder monitors
            for k, v in target_monitor.items():
                monitor_stats[k] = v.detach() if torch.is_tensor(v) else v

        out = {
            'align_loss': align_loss,
            'dur_loss': dur_loss,
            'mono_loss': mono_loss,
            'ntokens': audio_mask.sum(),
            'gamma': torch.exp(self.log_gamma).clamp(0.7, 3.0).detach(),
        }
        out.update(monitor_stats)
        return out

    @torch.inference_mode()
    def inference(
        self,
        wavs: torch.Tensor,
        texts: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        text_lens: Optional[torch.Tensor] = None,
        *,
        return_debug: bool = False,
    ):
        """
        输出每个 wbd(word) 的 (start,end,conf)，支持空隙/停顿。
        返回 list，长度 B；每个元素是 dict：
          - 'word_start_times': Tensor[W]
          - 'word_end_times': Tensor[W]
          - 'word_conf': Tensor[W]
        """
        audio_feat, audio_mask, text_feat, text_mask = self.encode(wavs, texts, wav_lens, text_lens)
        wbd_feat, wbd_mask = self._gather_wbd_embeddings(text_feat, texts, text_mask)

        logits = self._compute_alignment_logits(audio_feat, audio_mask, wbd_feat, wbd_mask).float()
        probs = torch.softmax(logits, dim=-1)  # [B, T, 1+Wmax]

        B, T, _ = probs.shape
        device = probs.device
        frame_sec = float(self._frame_sec())

        audio_lens = audio_mask.long().sum(dim=1)   # [B]
        word_lens_raw = wbd_mask.long().sum(dim=1)  # [B]

        # 防御性处理：若 word 数 > 帧数，DP 无法保证每个 word 至少 1 帧
        word_lens = torch.minimum(word_lens_raw, audio_lens)

        Tmax = int(audio_lens.max().detach().cpu().item()) if B > 0 else 0
        Wmax = int(word_lens.max().detach().cpu().item()) if B > 0 else 0

        results: List[Dict[str, torch.Tensor]] = []
        if Tmax <= 0 or Wmax <= 0:
            for _ in range(B):
                out = {
                    'word_start_times': torch.zeros((0,), device=wavs.device),
                    'word_end_times': torch.zeros((0,), device=wavs.device),
                    'word_conf': torch.zeros((0,), device=wavs.device),
                }
                if return_debug:
                    out['debug'] = {
                        'frame_sec': frame_sec,
                        'audio_len_frames': 0,
                        'word_len': 0,
                        'dp': torch.zeros((0, 1), dtype=torch.float16, device='cpu'),
                        'states': torch.zeros((0,), dtype=torch.long, device='cpu'),
                        'probs_words': torch.zeros((0, 0), dtype=torch.float16, device='cpu'),
                        'probs_blank': torch.zeros((0,), dtype=torch.float16, device='cpu'),
                    }
                results.append(out)
            return results

        probs = probs[:, :Tmax, :1 + Wmax]  # [B, Tmax, 1+Wmax]
        valid_time = (torch.arange(Tmax, device=device)[None, :] < audio_lens[:, None])  # [B, Tmax]

        # log-prob（避免 -inf / nan）
        p_blank = probs[:, :, 0].clamp_min(1e-12).log()       # [B, Tmax]
        p_words = probs[:, :, 1:].clamp_min(1e-12).log()      # [B, Tmax, Wmax]

        # 对不存在的 word（超过 word_lens）置 -inf，让 DP 不会走到那些 word state
        neg_inf = p_blank.new_tensor(-1e9)
        word_exist = (torch.arange(Wmax, device=device)[None, None, :] < word_lens[:, None, None])  # [B,1,Wmax]
        p_words = p_words.masked_fill(~word_exist, neg_inf)

        # DP states: blank0, word1, blank1, ..., wordW, blankW  => S=2W+1
        Smax = 2 * Wmax + 1
        s_ids = torch.arange(Smax, device=device, dtype=torch.long)[None, :]  # [1, Smax]

        # 每个样本有效 state 上界：2*Wb（blankWb 的 index）
        state_valid = (s_ids <= (2 * word_lens)[:, None])  # [B, Smax] bool

        # init dp at t=0
        dp_prev = p_blank.new_full((B, Smax), fill_value=neg_inf)
        dp_prev[:, 0] = p_blank[:, 0]
        if Wmax > 0:
            dp_prev[:, 1] = p_words[:, 0, 0]  # word1
        dp_prev = dp_prev.masked_fill(~state_valid, neg_inf)

        # debug: 保存 dp 历史（很耗显存，只在 debug 下启用）
        dp_hist = None
        if return_debug:
            dp_hist = torch.empty((B, Tmax, Smax), device=device, dtype=torch.float16)
            dp_hist[:, 0, :] = dp_prev.to(torch.float16)

        # backpointer：0=stay, 1=from s-1, 2=from s-2 (CTC skip blank)
        bp_step = torch.zeros((B, Tmax, Smax), device=device, dtype=torch.uint8)

        # 记录每个样本结束帧位置的 dp 值，用于决定 end_s
        t_end = (audio_lens - 1).clamp_min(0)  # [B]
        dp_end = torch.where((t_end == 0)[:, None], dp_prev, dp_prev.new_full((B, Smax), neg_inf))

        # adv2 只对 word state（奇数 s）开放
        is_word_state = (torch.arange(Smax, device=device) % 2 == 1)[None, :]  # [1, Smax]

        for t in range(1, Tmax):
            # emit_t: [B, Smax]
            emit_t = p_blank[:, t][:, None].expand(B, Smax).clone()
            emit_t[:, 1::2] = p_words[:, t, :]  # odd states = word_j

            prev = dp_prev  # [B, Smax]
            adv1 = torch.cat([neg_inf.expand(B, 1), prev[:, :-1]], dim=1)  # s-1 -> s

            if Smax >= 3:
                adv2 = torch.cat([neg_inf.expand(B, 2), prev[:, :-2]], dim=1)  # s-2 -> s
                adv2 = torch.where(is_word_state, adv2, neg_inf)               # only for word states
            else:
                adv2 = neg_inf.expand(B, Smax)

            cand = torch.stack([prev, adv1, adv2], dim=0)  # [3, B, Smax]
            best_prev, step = cand.max(dim=0)              # best_prev: [B,Smax], step: [B,Smax] in {0,1,2}

            dp_t = best_prev + emit_t

            # 对无效 time step（超过音频长度）保持 dp 不变
            vt = valid_time[:, t][:, None]  # [B,1]
            dp_t = torch.where(vt, dp_t, prev)
            step = torch.where(vt, step.to(torch.uint8), torch.zeros_like(step, dtype=torch.uint8))

            # 对无效 state 置 -inf，避免污染
            dp_t = dp_t.masked_fill(~state_valid, neg_inf)
            step = torch.where(state_valid, step, torch.zeros_like(step, dtype=torch.uint8))

            dp_prev = dp_t
            bp_step[:, t, :] = step

            if dp_hist is not None:
                dp_hist[:, t, :] = dp_prev.to(torch.float16)

            # 若某些样本在 t 结束，保存 dp_end
            end_mask = (t_end == t)
            if end_mask.any():
                dp_end[end_mask] = dp_prev[end_mask]

        # 选择 end state：在 blankW 与 wordW 二选一（对应 s=2W, s=2W-1）
        end_blank_s = (2 * word_lens).clamp_min(0).clamp_max(Smax - 1)   # [B]
        end_word_s = (end_blank_s - 1).clamp_min(0).clamp_max(Smax - 1)  # [B]

        score_blank = dp_end.gather(1, end_blank_s[:, None]).squeeze(1)  # [B]
        score_word = dp_end.gather(1, end_word_s[:, None]).squeeze(1)    # [B]

        choose_blank = score_blank > score_word
        end_s = torch.where(
            word_lens == 0,
            torch.zeros_like(end_blank_s),
            torch.where(choose_blank, end_blank_s, end_word_s),
        )  # [B]

        # backtrace states[b,t]（只对 t<=t_end 的部分有效）
        states = torch.zeros((B, Tmax), device=device, dtype=torch.long)
        s = end_s.clone()

        for t in range(Tmax - 1, -1, -1):
            active = (t <= t_end)  # [B]
            if active.any():
                states[active, t] = s[active]
            if t > 0:
                step_t = bp_step[:, t, :].gather(1, s[:, None]).squeeze(1).long()  # [B] in {0,1,2}
                step_t = step_t * active.long()
                s = s - step_t

        # ========= 提取每个 word 的 start/end =========
        w_ids = torch.arange(Wmax, device=device, dtype=torch.long)  # [Wmax]
        word_states = 2 * w_ids + 1                                  # [Wmax]
        valid_word = (w_ids[None, :] < word_lens[:, None])           # [B, Wmax]

        t_ids = torch.arange(Tmax, device=device, dtype=torch.long)[None, :, None]  # [1, Tmax, 1]

        mask_word_frame = (
            valid_time[:, :, None]
            & valid_word[:, None, :]
            & (states[:, :, None] == word_states[None, None, :])
        )  # [B, Tmax, Wmax]

        start_idx = torch.where(mask_word_frame, t_ids, torch.full_like(t_ids, Tmax)).amin(dim=1)  # [B, Wmax]
        end_idx = torch.where(mask_word_frame, t_ids, torch.full_like(t_ids, -1)).amax(dim=1) + 1  # [B, Wmax]
        has_span = (end_idx > 0) & valid_word  # [B, Wmax]

        start_idx = torch.where(has_span, start_idx, torch.zeros_like(start_idx))
        end_idx = torch.where(has_span, end_idx, torch.zeros_like(end_idx))

        word_start_times_all = start_idx.float() * frame_sec  # [B, Wmax]
        word_end_times_all = end_idx.float() * frame_sec      # [B, Wmax]

        # ========= 置信度：对齐段内 mean p(word_j)/(p(word_j)+p(blank)) =========
        probs_blank = probs[:, :, 0]   # [B, Tmax]
        probs_words = probs[:, :, 1:]  # [B, Tmax, Wmax]
        eps = 1e-6

        conf_frame = probs_words / (probs_blank[:, :, None] + probs_words + eps)  # [B, Tmax, Wmax]
        conf_frame = conf_frame * mask_word_frame.float()

        sum_conf = conf_frame.sum(dim=1)  # [B, Wmax]
        len_seg = mask_word_frame.float().sum(dim=1).clamp_min(1.0)  # [B, Wmax]

        word_conf_all = sum_conf / len_seg
        word_conf_all = torch.where(has_span, word_conf_all, torch.zeros_like(word_conf_all))

        # ========= 打包成 list（只切片，不做 DP）=========
        for b in range(B):
            Wb = int(word_lens[b].item())
            out = {
                'word_start_times': word_start_times_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
                'word_end_times': word_end_times_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
                'word_conf': word_conf_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
            }

            if return_debug:
                Tb = int(audio_lens[b].item())
                Sb = int(2 * Wb + 1) if Wb > 0 else 1

                dbg_dp = torch.zeros((0, 1), dtype=torch.float16, device='cpu')
                if dp_hist is not None and Tb > 0:
                    dbg_dp = dp_hist[b, :Tb, :Sb].detach().to('cpu')

                dbg_states = states[b, :Tb].detach().to('cpu') if Tb > 0 else torch.zeros((0,), dtype=torch.long)
                dbg_probs_blank = probs_blank[b, :Tb].detach().to('cpu').to(torch.float16) if Tb > 0 else torch.zeros((0,), dtype=torch.float16)

                if Wb > 0 and Tb > 0:
                    dbg_probs_words = probs_words[b, :Tb, :Wb].detach().to('cpu').to(torch.float16)
                else:
                    dbg_probs_words = torch.zeros((Tb, 0), dtype=torch.float16, device='cpu')

                out['debug'] = {
                    'frame_sec': frame_sec,
                    'audio_len_frames': Tb,
                    'word_len': Wb,
                    'dp': dbg_dp,
                    'states': dbg_states,
                    'probs_words': dbg_probs_words,
                    'probs_blank': dbg_probs_blank,
                }

            results.append(out)

        return results


class ForcedAlignerV2(ForcedAligner):
    def __init__(self, config: ModelArgs):
        nn.Module.__init__(self)
        self.config = config

        # Nepa
        if config.nepa_ckpt.endswith('.ckpt'):
            nepa_hp = set_hparams(os.path.join(Path(config.nepa_ckpt).parent, 'config.yaml'), global_hparams=False)
        else:
            nepa_hp = set_hparams(os.path.join(config.nepa_ckpt, 'config.yaml'), global_hparams=False)
        self.nepa = build_nepa_model(nepa_hp, attn_implementation=config.attn_implementation)
        if config.init_pretrained:
            load_ckpt(self.nepa, config.nepa_ckpt)

        self.config.hop_size = int(self.nepa.config.hop_size)

        if config.freeze_nepa:
            self.nepa.eval()
            for p in self.nepa.parameters():
                p.requires_grad = False

        self.audio_proj = nn.Linear(self.nepa.config.hidden_size, config.audio_hidden_size, bias=False)
        self.audio_encoder = TransformerEncoderModel(
            config=TransformerConfig(
                hidden_size=config.audio_hidden_size,
                num_hidden_layers=config.audio_num_hidden_layers,
                num_cross_attention_layers=config.audio_num_hidden_layers,
                num_attention_heads=config.audio_num_attention_heads,
                num_key_value_heads=config.audio_num_key_value_heads,
                intermediate_size=config.audio_hidden_size * 4,
                use_dynamic_cross_gate=True,
                use_cache=False,
                attn_implementation=config.attn_implementation,
            )
        )
        self.audio_out = nn.Linear(config.audio_hidden_size, config.hidden_size, bias=False)

        self.text_embed = nn.Embedding(config.vocab_size, config.text_hidden_size)
        self.text_prenet = TransformerEncoderModel(
            config=TransformerConfig(
                hidden_size=config.text_hidden_size,
                num_hidden_layers=max(2, config.text_num_hidden_layers // 4),
                num_attention_heads=config.text_num_attention_heads,
                num_key_value_heads=config.text_num_key_value_heads,
                intermediate_size=config.text_hidden_size * 4,
                use_cache=False,
                attn_implementation=config.attn_implementation,
            )
        )
        self.text_encoder = TransformerEncoderModel(
            config=TransformerConfig(
                hidden_size=config.text_hidden_size,
                num_hidden_layers=config.text_num_hidden_layers,
                num_cross_attention_layers=config.text_num_hidden_layers,
                num_attention_heads=config.text_num_attention_heads,
                num_key_value_heads=config.text_num_key_value_heads,
                intermediate_size=config.text_hidden_size * 4,
                use_dynamic_cross_gate=True,
                use_cache=False,
                attn_implementation=config.attn_implementation,
            )
        )
        self.text_out = nn.Linear(config.text_hidden_size, config.hidden_size, bias=False)

        self.blank_probe = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        self.blank_bias = nn.Parameter(torch.zeros([]))

        self.log_gamma = nn.Parameter(torch.zeros([]))

        if config.gradient_checkpointing and self.training:
            self.nepa.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.audio_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.text_prenet.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.text_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def encode(self, wavs, txt_tokens, wav_lens=None, txt_lens=None):
        if self.config.freeze_nepa:
            with torch.no_grad():
                audio_feat, audio_mask = self.nepa(wavs, wav_lens)  # eval-mode => Tensor[B, T', C]
        else:
            audio_feat, audio_mask = self.nepa(wavs, wav_lens)
        audio_feat = self.audio_proj(audio_feat)

        if txt_lens is None:
            txt_lens = torch.full((txt_tokens.shape[0],), txt_tokens.shape[1], device=txt_tokens.device, dtype=torch.long)
        else:
            txt_lens = txt_lens.long()
        txt_mask = sequence_mask(txt_lens, maxlen=txt_tokens.shape[1])  # bool [B, N]
        txt_feat = self.text_embed(txt_tokens.clamp_min(0).clamp_max(self.config.vocab_size - 1))
        txt_feat = self.text_prenet(inputs_embeds=txt_feat, attention_mask=txt_mask).last_hidden_state
        
        audio_out = self.audio_encoder.forward(
            inputs_embeds=audio_feat,
            attention_mask=audio_mask,
            encoder_hidden_states=txt_feat,
            encoder_attention_mask=txt_mask,
        ).last_hidden_state
        audio_out = self.audio_out(audio_out)

        txt_out = self.text_encoder.forward(
            inputs_embeds=txt_feat,
            attention_mask=txt_mask,
            encoder_hidden_states=audio_feat,
            encoder_attention_mask=audio_mask,
        ).last_hidden_state
        txt_out = self.text_out(txt_out)

        return audio_out, audio_mask, txt_out, txt_mask
    

class ForcedAlignerV3(ForcedAlignerV2):
    def __init__(self, config: ModelArgs):
        super().__init__(config)

        hs = int(getattr(config, 'hidden_size', 512))
        ph = int(getattr(config, 'pause_hidden_size', 256))
        self.pause_head = nn.Sequential(
            nn.Linear(2 * hs, ph),
            nn.SiLU(),
            nn.Linear(ph, 1),
        )

    def _compute_pause_logits(
        self,
        wbd_feat: torch.Tensor,
        wbd_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pause(after word k): logits shape [B, W-1]
        mask: [B, W-1] (need both k and k+1 valid)
        """
        B, W, C = wbd_feat.shape
        if W <= 1:
            logits = wbd_feat.new_zeros((B, 0))
            mask = torch.zeros((B, 0), dtype=torch.bool, device=wbd_feat.device)
            return logits, mask

        left = wbd_feat[:, :-1, :]
        right = wbd_feat[:, 1:, :]
        feat = torch.cat([left, right], dim=-1)  # [B, W-1, 2C]
        logits = self.pause_head(feat).squeeze(-1)  # [B, W-1]

        mask = (wbd_mask[:, :-1] & wbd_mask[:, 1:])
        return logits, mask

    def _build_pause_targets_from_times(
        self,
        audio_mask: torch.Tensor,
        word_start_times: Optional[torch.Tensor],
        word_end_times: Optional[torch.Tensor],
        wbd_mask: torch.Tensor,
        word_conf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        严格 pause 定义（帧域）：
        - pause_k = 1  <=> end_idx[k] != start_idx[k+1]
        其中 start/end_idx 都是按 frame_sec round 得到。

        Returns:
            pause_tgt: float32 [B, W-1] in {0,1}
            pause_mask: bool [B, W-1]
        """
        device = audio_mask.device
        B, T = audio_mask.shape
        W = int(wbd_mask.shape[1])

        if W <= 1:
            pause_tgt = torch.zeros((B, 0), dtype=torch.float32, device=device)
            pause_mask = torch.zeros((B, 0), dtype=torch.bool, device=device)
            return pause_tgt, pause_mask

        if word_start_times is None or word_end_times is None:
            pause_tgt = torch.zeros((B, W - 1), dtype=torch.float32, device=device)
            pause_mask = torch.zeros((B, W - 1), dtype=torch.bool, device=device)
            return pause_tgt, pause_mask

        prep = self._prepare_word_time_supervision(
            audio_mask=audio_mask,
            wbd_mask=wbd_mask,
            word_start_times=word_start_times,
            word_end_times=word_end_times,
            word_conf=word_conf,
        )
        if not prep["has_times"]:
            pause_tgt = torch.zeros((B, W - 1), dtype=torch.float32, device=device)
            pause_mask = torch.zeros((B, W - 1), dtype=torch.bool, device=device)
            return pause_tgt, pause_mask

        valid_words = prep["valid_words"]
        start_idx = prep["start_idx"]
        end_idx = prep["end_idx"]

        pause_mask = (valid_words[:, :-1] & valid_words[:, 1:])
        pause_tgt = (end_idx[:, :-1] != start_idx[:, 1:]).to(torch.float32)
        pause_tgt = torch.where(pause_mask, pause_tgt, torch.zeros_like(pause_tgt))
        return pause_tgt, pause_mask

    def forward(
        self,
        wavs: torch.Tensor,
        texts: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        text_lens: Optional[torch.Tensor] = None,
        *,
        word_start_times: Optional[torch.Tensor] = None,
        word_end_times: Optional[torch.Tensor] = None,
        word_conf: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        audio_feat, audio_mask, text_feat, text_mask = self.encode(wavs, texts, wav_lens, text_lens)

        wbd_feat, wbd_mask = self._gather_wbd_embeddings(text_feat, texts, text_mask)
        logits = self._compute_alignment_logits(audio_feat, audio_mask, wbd_feat, wbd_mask)  # [B,T,1+W]

        use_frame_weight = bool(getattr(self.config, "use_frame_weight", True))
        word_conf_for_targets = word_conf  # 保留给 zero-duration / pause target 的 conf_split

        target, frame_weight, dur_tgt, dur_mask_tgt, target_monitor = self._build_frame_targets_from_times(
            audio_mask,
            word_start_times=word_start_times,
            word_end_times=word_end_times,
            word_conf=word_conf_for_targets,
            wbd_mask=wbd_mask,
        )

        if not use_frame_weight:
            frame_weight = torch.ones_like(frame_weight)

        B, T, C = logits.shape
        ignore_index = -100

        ce_per = F.cross_entropy(
            logits.float().reshape(-1, C),
            target.reshape(-1),
            reduction='none',
            ignore_index=ignore_index,
        ).view(B, T)

        valid = (target != ignore_index) & audio_mask
        denom = valid.float().sum().clamp_min(1.0)
        align_loss = (ce_per * frame_weight * valid.float()).sum() / denom

        probs = torch.softmax(logits.float(), dim=-1)  # [B,T,1+W]
        dur_loss = logits.new_tensor(0.0)
        mono_loss = logits.new_tensor(0.0)

        W = wbd_feat.shape[1]
        if W > 0:
            dur_pred = (probs[:, :, 1:] * audio_mask[:, :, None].float()).sum(dim=1)  # [B,W]
            # dur_mask = wbd_mask.float()
            dur_mask = dur_mask_tgt

            l1 = (dur_pred - dur_tgt).abs()
            l1 = (l1 * dur_mask).sum() / dur_mask.sum().clamp_min(1.0)

            log_l2 = (torch.log1p(dur_pred.clamp_min(0.0)) - torch.log1p(dur_tgt.clamp_min(0.0))).pow(2)
            log_l2 = (log_l2 * dur_mask).sum() / dur_mask.sum().clamp_min(1.0)

            dur_loss = l1 + log_l2

            p_words = probs[:, :, 1:]  # [B,T,W]
            p_sum = p_words.sum(dim=-1).clamp_min(1e-6)  # [B,T]
            idx = torch.arange(1, W + 1, device=logits.device, dtype=torch.float32)[None, None, :]
            expected = (p_words * idx).sum(dim=-1) / p_sum  # [B,T]

            dec = F.relu(expected[:, :-1] - expected[:, 1:])  # [B,T-1]
            mono_mask = audio_mask[:, 1:] & audio_mask[:, :-1]
            mono_loss = (dec * mono_mask.float()).sum() / mono_mask.float().sum().clamp_min(1.0)

        pause_logits, pause_mask = self._compute_pause_logits(wbd_feat, wbd_mask)
        pause_tgt, pause_tgt_mask = self._build_pause_targets_from_times(
            audio_mask,
            word_start_times=word_start_times,
            word_end_times=word_end_times,
            wbd_mask=wbd_mask,
            word_conf=word_conf_for_targets,
        )

        pause_mask = pause_mask & pause_tgt_mask
        pause_loss = logits.new_tensor(0.0)
        pause_acc = logits.new_tensor(0.0)

        if pause_logits.numel() > 0 and pause_mask.any():
            bce = F.binary_cross_entropy_with_logits(
                pause_logits.float(),
                pause_tgt.float(),
                reduction='none',
            )
            pause_loss = (bce * pause_mask.float()).sum() / pause_mask.float().sum().clamp_min(1.0)

            pred = (torch.sigmoid(pause_logits.float()) >= float(self.config.pause_threshold))
            gt = (pause_tgt >= 0.5)
            pause_acc = ((pred == gt) & pause_mask).float().sum() / pause_mask.float().sum().clamp_min(1.0)

        monitor_stats = {}
        if bool(getattr(self.config, "enable_monitor_stats", True)):
            valid = (target != ignore_index) & audio_mask
            valid_f = valid.float()

            target_safe = target.clamp_min(0)
            p_target = probs.gather(dim=-1, index=target_safe.unsqueeze(-1)).squeeze(-1)
            monitor_stats["mon_p_target_mean"] = (
                (p_target * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            ).detach()

            ent = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)
            monitor_stats["mon_entropy_valid_mean"] = (
                (ent * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            ).detach()

            word_mass = probs[:, :, 1:].sum(dim=-1)
            monitor_stats["mon_word_mass_valid_mean"] = (
                (word_mass * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            ).detach()

            for k, v in target_monitor.items():
                monitor_stats[k] = v.detach() if torch.is_tensor(v) else v

        out = {
            'align_loss': align_loss,
            'dur_loss': dur_loss,
            'mono_loss': mono_loss,
            'pause_loss': pause_loss,
            'pause_acc': pause_acc.detach(),
            'ntokens': audio_mask.sum(),
            'gamma': torch.exp(self.log_gamma).clamp(0.7, 3.0).detach(),
        }
        out.update(monitor_stats)
        return out

    @torch.inference_mode()
    def inference(
        self,
        wavs: torch.Tensor,
        texts: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        text_lens: Optional[torch.Tensor] = None,
        *,
        return_debug: bool = False,
    ):
        audio_feat, audio_mask, text_feat, text_mask = self.encode(wavs, texts, wav_lens, text_lens)
        wbd_feat, wbd_mask = self._gather_wbd_embeddings(text_feat, texts, text_mask)

        logits = self._compute_alignment_logits(audio_feat, audio_mask, wbd_feat, wbd_mask).float()
        probs = torch.softmax(logits, dim=-1)  # [B, T, 1+Wmax]

        B, T, _ = probs.shape
        device = probs.device
        frame_sec = float(self._frame_sec())

        audio_lens = audio_mask.long().sum(dim=1)   # [B]
        word_lens_raw = wbd_mask.long().sum(dim=1)  # [B]
        word_lens = torch.minimum(word_lens_raw, audio_lens)

        Tmax = int(audio_lens.max().detach().cpu().item()) if B > 0 else 0
        Wmax = int(word_lens.max().detach().cpu().item()) if B > 0 else 0

        results: List[Dict[str, torch.Tensor]] = []
        if Tmax <= 0 or Wmax <= 0:
            for _ in range(B):
                out = {
                    'word_start_times': torch.zeros((0,), device=wavs.device),
                    'word_end_times': torch.zeros((0,), device=wavs.device),
                    'word_conf': torch.zeros((0,), device=wavs.device),
                }
                if return_debug:
                    out['debug'] = {
                        'frame_sec': frame_sec,
                        'audio_len_frames': 0,
                        'word_len': 0,
                        'dp': torch.zeros((0, 1), dtype=torch.float16, device='cpu'),
                        'states': torch.zeros((0,), dtype=torch.long, device='cpu'),
                        'probs_words': torch.zeros((0, 0), dtype=torch.float16, device='cpu'),
                        'probs_blank': torch.zeros((0,), dtype=torch.float16, device='cpu'),
                        'pause_prob_after': torch.zeros((0,), dtype=torch.float16, device='cpu'),
                    }
                results.append(out)
            return results

        probs = probs[:, :Tmax, :1 + Wmax]  # [B, Tmax, 1+Wmax]
        valid_time = (torch.arange(Tmax, device=device)[None, :] < audio_lens[:, None])  # [B, Tmax]

        p_blank = probs[:, :, 0].clamp_min(1e-12).log()       # [B, Tmax]
        p_words = probs[:, :, 1:].clamp_min(1e-12).log()      # [B, Tmax, Wmax]

        neg_inf = p_blank.new_tensor(-1e9)
        word_exist = (torch.arange(Wmax, device=device)[None, None, :] < word_lens[:, None, None])
        p_words = p_words.masked_fill(~word_exist, neg_inf)

        pause_logits, _ = self._compute_pause_logits(
            wbd_feat[:, :Wmax, :],
            (torch.arange(Wmax, device=device)[None, :] < word_lens[:, None]),
        )
        pause_prob = torch.sigmoid(pause_logits.float())  # [B, Wmax-1]

        allow_blank_k = torch.ones((B, Wmax + 1), device=device, dtype=torch.bool)  # blank0..blankW
        if Wmax >= 2:
            allow_between = (pause_prob >= float(self.config.pause_threshold))
            if bool(getattr(self.config, 'pause_hard_gating', True)):
                allow_blank_k[:, 1:Wmax] = allow_between
            else:
                allow_blank_k[:, 1:Wmax] = True

        Smax = 2 * Wmax + 1
        s_ids = torch.arange(Smax, device=device, dtype=torch.long)[None, :]
        state_valid = (s_ids <= (2 * word_lens)[:, None])

        blank_state_ids = torch.arange(0, Smax, 2, device=device, dtype=torch.long)  # 0,2,4,...
        blank_k = (blank_state_ids // 2).clamp_max(Wmax)  # [Wmax+1]
        allow_blank_state = allow_blank_k.gather(1, blank_k[None, :].expand(B, -1))  # [B, Wmax+1]

        dp_prev = p_blank.new_full((B, Smax), fill_value=neg_inf)
        dp_prev[:, 0] = p_blank[:, 0]
        if Wmax > 0:
            dp_prev[:, 1] = p_words[:, 0, 0]
        dp_prev = dp_prev.masked_fill(~state_valid, neg_inf)

        if bool(getattr(self.config, 'pause_hard_gating', True)):
            dp_prev[:, 0::2] = dp_prev[:, 0::2].masked_fill(~allow_blank_state, neg_inf)

        dp_hist = None
        if return_debug:
            dp_hist = torch.empty((B, Tmax, Smax), device=device, dtype=torch.float16)
            dp_hist[:, 0, :] = dp_prev.to(torch.float16)

        bp_step = torch.zeros((B, Tmax, Smax), device=device, dtype=torch.uint8)

        t_end = (audio_lens - 1).clamp_min(0)
        dp_end = torch.where((t_end == 0)[:, None], dp_prev, dp_prev.new_full((B, Smax), neg_inf))

        is_word_state = (torch.arange(Smax, device=device) % 2 == 1)[None, :]

        for t in range(1, Tmax):
            emit_t = p_blank[:, t][:, None].expand(B, Smax).clone()
            emit_t[:, 1::2] = p_words[:, t, :]

            if bool(getattr(self.config, 'pause_hard_gating', True)):
                emit_t[:, 0::2] = emit_t[:, 0::2].masked_fill(~allow_blank_state, neg_inf)
            else:
                if Wmax >= 2:
                    forbid_between = ~(pause_prob >= float(self.config.pause_threshold))
                    penalty = p_blank.new_tensor(2.0)
                    emit_t[:, 2:2 * Wmax:2] = emit_t[:, 2:2 * Wmax:2] - penalty * forbid_between.to(emit_t.dtype)

            prev = dp_prev
            adv1 = torch.cat([neg_inf.expand(B, 1), prev[:, :-1]], dim=1)

            if Smax >= 3:
                adv2 = torch.cat([neg_inf.expand(B, 2), prev[:, :-2]], dim=1)
                adv2 = torch.where(is_word_state, adv2, neg_inf)
            else:
                adv2 = neg_inf.expand(B, Smax)

            cand = torch.stack([prev, adv1, adv2], dim=0)
            best_prev, step = cand.max(dim=0)
            dp_t = best_prev + emit_t

            vt = valid_time[:, t][:, None]
            dp_t = torch.where(vt, dp_t, prev)
            step = torch.where(vt, step.to(torch.uint8), torch.zeros_like(step, dtype=torch.uint8))

            dp_t = dp_t.masked_fill(~state_valid, neg_inf)
            step = torch.where(state_valid, step, torch.zeros_like(step, dtype=torch.uint8))

            dp_prev = dp_t
            bp_step[:, t, :] = step

            if dp_hist is not None:
                dp_hist[:, t, :] = dp_prev.to(torch.float16)

            end_mask = (t_end == t)
            if end_mask.any():
                dp_end[end_mask] = dp_prev[end_mask]

        end_blank_s = (2 * word_lens).clamp_min(0).clamp_max(Smax - 1)
        end_word_s = (end_blank_s - 1).clamp_min(0).clamp_max(Smax - 1)

        score_blank = dp_end.gather(1, end_blank_s[:, None]).squeeze(1)
        score_word = dp_end.gather(1, end_word_s[:, None]).squeeze(1)

        choose_blank = score_blank > score_word
        end_s = torch.where(
            word_lens == 0,
            torch.zeros_like(end_blank_s),
            torch.where(choose_blank, end_blank_s, end_word_s),
        )

        states = torch.zeros((B, Tmax), device=device, dtype=torch.long)
        s = end_s.clone()

        for t in range(Tmax - 1, -1, -1):
            active = (t <= t_end)
            if active.any():
                states[active, t] = s[active]
            if t > 0:
                step_t = bp_step[:, t, :].gather(1, s[:, None]).squeeze(1).long()
                step_t = step_t * active.long()
                s = s - step_t

        w_ids = torch.arange(Wmax, device=device, dtype=torch.long)
        word_states = 2 * w_ids + 1
        valid_word = (w_ids[None, :] < word_lens[:, None])

        t_ids = torch.arange(Tmax, device=device, dtype=torch.long)[None, :, None]

        mask_word_frame = (
            valid_time[:, :, None]
            & valid_word[:, None, :]
            & (states[:, :, None] == word_states[None, None, :])
        )

        start_idx = torch.where(mask_word_frame, t_ids, torch.full_like(t_ids, Tmax)).amin(dim=1)
        end_idx = torch.where(mask_word_frame, t_ids, torch.full_like(t_ids, -1)).amax(dim=1) + 1
        has_span = (end_idx > 0) & valid_word

        start_idx = torch.where(has_span, start_idx, torch.zeros_like(start_idx))
        end_idx = torch.where(has_span, end_idx, torch.zeros_like(end_idx))

        word_start_times_all = start_idx.float() * frame_sec
        word_end_times_all = end_idx.float() * frame_sec

        probs_blank = probs[:, :, 0]
        probs_words = probs[:, :, 1:]
        eps = 1e-6

        conf_frame = probs_words / (probs_blank[:, :, None] + probs_words + eps)
        conf_frame = conf_frame * mask_word_frame.float()

        sum_conf = conf_frame.sum(dim=1)
        len_seg = mask_word_frame.float().sum(dim=1).clamp_min(1.0)

        word_conf_all = sum_conf / len_seg
        word_conf_all = torch.where(has_span, word_conf_all, torch.zeros_like(word_conf_all))

        for b in range(B):
            Wb = int(word_lens[b].item())
            out = {
                'word_start_times': word_start_times_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
                'word_end_times': word_end_times_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
                'word_conf': word_conf_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
            }

            if return_debug:
                Tb = int(audio_lens[b].item())
                Sb = int(2 * Wb + 1) if Wb > 0 else 1

                dbg_dp = torch.zeros((0, 1), dtype=torch.float16, device='cpu')
                if dp_hist is not None and Tb > 0:
                    dbg_dp = dp_hist[b, :Tb, :Sb].detach().to('cpu')

                dbg_states = states[b, :Tb].detach().to('cpu') if Tb > 0 else torch.zeros((0,), dtype=torch.long)
                dbg_probs_blank = probs_blank[b, :Tb].detach().to('cpu').to(torch.float16) if Tb > 0 else torch.zeros((0,), dtype=torch.float16)

                if Wb > 0 and Tb > 0:
                    dbg_probs_words = probs_words[b, :Tb, :Wb].detach().to('cpu').to(torch.float16)
                else:
                    dbg_probs_words = torch.zeros((Tb, 0), dtype=torch.float16, device='cpu')

                dbg_pause_prob = torch.sigmoid(pause_logits[b, :max(Wb - 1, 0)].float()).detach().to('cpu').to(torch.float16)

                out['debug'] = {
                    'frame_sec': frame_sec,
                    'audio_len_frames': Tb,
                    'word_len': Wb,
                    'dp': dbg_dp,
                    'states': dbg_states,
                    'probs_words': dbg_probs_words,
                    'probs_blank': dbg_probs_blank,
                    'pause_prob_after': dbg_pause_prob,
                }

            results.append(out)

        return results
    

class ForcedAlignerV4(ForcedAlignerV2):
    """
    V4: State space = blank0, word1, blank1, ..., wordW, blankW  (S=2W+1)
        - blank prototypes: blank0_param, base_blank + gap_blank_delta_mlp([w_k,w_{k+1}]), blankW_param
        - single bmm after interleaving prototypes
        - duration loss includes word + blank (blank weighted by config.blank_dur_weight)
        - mono_loss computed on word states only
        - inference output kept identical to previous (word start/end/conf)
        - debug heatmap: words only (no blank columns); probs_blank is global blank sum
    """
    def __init__(self, config: ModelArgs):
        super().__init__(config)

        hs = int(getattr(config, "hidden_size", 512))
        mh = int(getattr(config, "blank_mlp_hidden_size", 256))

        # Remove old global blank probe/bias from V1/V2 to avoid optimizer confusion
        self.blank_probe = None
        self.blank_bias = None

        # V4 blank prototypes
        self.base_blank = nn.Parameter(torch.randn(hs) * 0.02)
        self.blank0_param = nn.Parameter(torch.randn(hs) * 0.02)
        self.blankW_param = nn.Parameter(torch.randn(hs) * 0.02)

        # Gap blank delta MLP: Δ_k = MLP([word_k, word_{k+1}])
        self.gap_blank_delta_mlp = nn.Sequential(
            nn.Linear(2 * hs, mh),
            nn.SiLU(),
            nn.Linear(mh, hs),
        )
        # Small init for last layer -> Δ≈0 at start
        nn.init.zeros_(self.gap_blank_delta_mlp[-1].weight)
        nn.init.zeros_(self.gap_blank_delta_mlp[-1].bias)

    # ---------- helpers ----------
    def _build_blank_prototypes(
        self,
        wbd_feat: torch.Tensor,   # [B, Wmax, C]
        wbd_mask: torch.Tensor,   # [B, Wmax] bool
    ):
        """
        Returns:
            blank_feat: [B, Wmax+1, C]
            blank_mask: [B, Wmax+1] bool (blank0 always; internal blanks require adjacent words; blankW requires W>0)
            word_lens:  [B] long
        """
        B, Wmax, C = wbd_feat.shape
        device = wbd_feat.device

        word_lens = wbd_mask.long().sum(dim=1)          # [B]
        has_word = (word_lens > 0)

        # blank_feat default = base_blank everywhere
        blank_feat = self.base_blank.view(1, 1, C).expand(B, Wmax + 1, C).clone()

        # internal blanks: 1..Wmax-1 use base_blank + delta
        if Wmax >= 2:
            left = wbd_feat[:, :-1, :]
            right = wbd_feat[:, 1:, :]
            pair = torch.cat([left, right], dim=-1)     # [B, Wmax-1, 2C]
            delta = self.gap_blank_delta_mlp(pair)      # [B, Wmax-1, C]
            blank_feat[:, 1:Wmax, :] = self.base_blank.view(1, 1, C) + delta

        # blank0
        blank_feat[:, 0, :] = self.blank0_param

        # blankW at index = word_lens (only if has_word)
        end_mask = torch.zeros((B, Wmax + 1), dtype=torch.bool, device=device)
        # scatter True into position word_lens (clamp for safety)
        end_idx = word_lens.clamp(min=0, max=Wmax)
        end_mask.scatter_(1, end_idx[:, None], has_word[:, None])
        blank_feat = torch.where(end_mask[:, :, None], self.blankW_param.view(1, 1, C), blank_feat)

        # blank_mask: blank0 always valid; internal blanks require adjacent words; blankW requires has_word
        blank_mask = torch.zeros((B, Wmax + 1), dtype=torch.bool, device=device)
        blank_mask[:, 0] = True

        if Wmax >= 2:
            internal = (wbd_mask[:, :-1] & wbd_mask[:, 1:])  # [B, Wmax-1]
            blank_mask[:, 1:Wmax] = internal

        blank_mask |= end_mask  # end blank

        return blank_feat, blank_mask, word_lens

    def _compute_state_logits(
        self,
        audio_feat: torch.Tensor,   # [B, T, C]
        audio_mask: torch.Tensor,   # [B, T] bool
        wbd_feat: torch.Tensor,     # [B, Wmax, C]
        wbd_mask: torch.Tensor,     # [B, Wmax] bool
    ):
        """
        logits: [B, T, Smax], Smax=2*Wmax+1
        state order: blank0, word1, blank1, ..., wordW, blankW
        """
        B, T, C = audio_feat.shape
        Wmax = wbd_feat.shape[1]
        Smax = 2 * Wmax + 1
        neg_inf = audio_feat.new_tensor(-1e9)

        blank_feat, blank_mask, _ = self._build_blank_prototypes(wbd_feat, wbd_mask)

        # interleave prototypes
        proto = audio_feat.new_zeros((B, Smax, C))
        proto[:, 0::2, :] = blank_feat
        proto[:, 1::2, :] = wbd_feat

        state_mask = torch.zeros((B, Smax), dtype=torch.bool, device=audio_feat.device)
        state_mask[:, 0::2] = blank_mask
        state_mask[:, 1::2] = wbd_mask

        if self.config.sim_type == "cos":
            a = F.normalize(audio_feat, dim=-1)
            p = F.normalize(proto, dim=-1)
        else:
            a, p = audio_feat, proto

        gamma = torch.exp(self.log_gamma).clamp(0.7, 30.0).to(a.dtype)
        logits = torch.bmm(a, p.transpose(1, 2)) * gamma  # [B,T,Smax]

        # mask invalid states / frames
        logits = logits.masked_fill(~state_mask[:, None, :], neg_inf)
        logits = logits.masked_fill(~audio_mask[:, :, None], neg_inf)

        return logits, state_mask, blank_mask

    def _build_state_targets_from_times(
        self,
        audio_mask: torch.Tensor,                 # [B,T]
        wbd_mask: torch.Tensor,                   # [B,W]
        word_start_times: Optional[torch.Tensor] = None,
        word_end_times: Optional[torch.Tensor] = None,
        word_conf: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = -100,
    ):
        device = audio_mask.device
        B, T = audio_mask.shape
        W = int(wbd_mask.shape[1])

        audio_lens = audio_mask.long().sum(dim=1)  # [B]
        has_word = (wbd_mask.long().sum(dim=1) > 0)

        # default: all valid frames are blank0
        target = torch.full((B, T), fill_value=ignore_index, device=device, dtype=torch.long)
        target = torch.where(audio_mask, torch.zeros_like(target), target)  # blank0=0
        frame_weight = torch.where(audio_mask,
                                torch.ones((B, T), device=device, dtype=torch.float32),
                                torch.zeros((B, T), device=device, dtype=torch.float32))

        word_dur_tgt = torch.zeros((B, W), device=device, dtype=torch.float32)
        blank_dur_tgt = torch.zeros((B, W + 1), device=device, dtype=torch.float32)
        word_dur_mask = torch.zeros((B, W), device=device, dtype=torch.float32)
        blank_dur_mask = torch.zeros((B, W + 1), device=device, dtype=torch.float32)

        prep = self._prepare_word_time_supervision(
            audio_mask=audio_mask,
            wbd_mask=wbd_mask,
            word_start_times=word_start_times,
            word_end_times=word_end_times,
            word_conf=word_conf,
        )
        monitor = dict(prep["monitor"])
        monitor["mon_audio_valid_frame_count"] = audio_mask.float().sum().detach()

        # W=0 or no timestamps: keep blank0 targets only; supervise blank0 duration
        if W == 0 or (not prep["has_times"]):
            blank_dur_tgt[:, 0] = audio_lens.float()
            blank_dur_mask[:, 0] = 1.0

            monitor["mon_target_word_frame_count"] = torch.tensor(0.0, device=device)
            monitor["mon_target_blank_frame_count"] = audio_mask.float().sum().detach()
            monitor["mon_target_word_frame_ratio"] = torch.tensor(0.0, device=device)
            monitor["mon_frame_weight_word_mean"] = torch.tensor(1.0, device=device)

            return target, frame_weight, word_dur_tgt, blank_dur_tgt, word_dur_mask, blank_dur_mask, monitor
        
        frame_sec = float(self._frame_sec())

        valid_words = prep["valid_words"]          # [B,W]
        start_idx = prep["start_idx"]              # [B,W]
        end_idx = prep["end_idx"]                  # [B,W]
        conf_weight = prep["conf_weight"]          # [B,W] or None

        word_dur_mask = valid_words.float()

        # ---- build per-frame word mask ----
        t_idx = torch.arange(T, device=device)[None, :, None]  # [1,T,1]
        in_span = (
            valid_words[:, None, :]
            & (t_idx >= start_idx[:, None, :])
            & (t_idx < end_idx[:, None, :])
            & audio_mask[:, :, None]
        )  # [B,T,W]

        # word state ids: 1,3,5,...
        word_state_ids = (2 * torch.arange(W, device=device, dtype=torch.long) + 1)[None, None, :]  # [1,1,W]
        word_state = torch.where(in_span, word_state_ids, torch.zeros_like(word_state_ids)).amax(dim=-1)  # [B,T]
        is_word_frame = (word_state > 0) & audio_mask

        # ---- gap(blank_k) assignment by counting ended valid words ----
        ended = (valid_words[:, None, :] & (t_idx >= end_idx[:, None, :]) & audio_mask[:, :, None])  # [B,T,W]
        gap_idx = ended.long().sum(dim=-1)  # [B,T] in [0..W]
        blank_state = (2 * gap_idx).long()

        # frame target: word state if in word, else blank_k
        target = torch.where(audio_mask, torch.where(is_word_frame, word_state, blank_state), target)
        target = torch.where(audio_mask, target, torch.full_like(target, ignore_index))

        # ---- frame weight by word confidence (only on word frames) ----
        if conf_weight is not None:
            word_idx = ((word_state - 1) // 2).clamp(min=0, max=max(W - 1, 0))  # [B,T]
            conf_frame = conf_weight.gather(1, word_idx)
            frame_weight = torch.where(is_word_frame, conf_frame, frame_weight)

        # ---- duration targets: words ----
        word_dur_tgt = ((end_idx - start_idx).clamp_min(0).float() * valid_words.float())

        # ---- duration targets: blanks (W+1) ----
        # 更稳：blank0 用“第一个 valid word 的 start”；若没有 valid word，则整段都归 blank0
        if W > 0:
            ar = torch.arange(W, device=device)[None, :].expand(B, W)
            first_valid_pos = torch.where(valid_words, ar, torch.full_like(ar, W)).amin(dim=1)  # [B]
            has_any_valid = (first_valid_pos < W)

            first_valid_pos_safe = first_valid_pos.clamp(min=0, max=max(W - 1, 0))
            first_valid_start = start_idx.gather(1, first_valid_pos_safe[:, None]).squeeze(1).float()

            blank_dur_tgt[:, 0] = torch.where(has_any_valid, first_valid_start, audio_lens.float())
            blank_dur_mask[:, 0] = 1.0
        else:
            blank_dur_tgt[:, 0] = audio_lens.float()
            blank_dur_mask[:, 0] = 1.0

        # internal blanks k=1..W-1 require adjacent valid words
        if W >= 2:
            gap_dur = (start_idx[:, 1:] - end_idx[:, :-1]).clamp_min(0).float()  # [B,W-1]
            internal_mask = (valid_words[:, :-1] & valid_words[:, 1:]).float()
            blank_dur_tgt[:, 1:W] = gap_dur * internal_mask
            blank_dur_mask[:, 1:W] = internal_mask

        # end blank：用“最后一个 valid word 的 end”；如果没有 valid word，则不单独加 end blank（blank0 已覆盖整段）
        if W > 0:
            ar = torch.arange(W, device=device)[None, :].expand(B, W)
            last_valid_pos = torch.where(valid_words, ar, torch.full_like(ar, -1)).amax(dim=1)  # [B]
            has_any_valid = (last_valid_pos >= 0)

            last_valid_pos_safe = last_valid_pos.clamp(min=0, max=max(W - 1, 0))
            last_valid_end = end_idx.gather(1, last_valid_pos_safe[:, None]).squeeze(1)
            end_dur = (audio_lens - last_valid_end).clamp_min(0).float() * has_any_valid.float()

            # end blank 的 state index 仍然是 blankW（按文本长度）
            word_lens = wbd_mask.long().sum(dim=1)  # [B]
            end_slot = word_lens[:, None].clamp(min=0, max=W)

            add_end = torch.zeros_like(blank_dur_tgt)
            add_end_mask = torch.zeros_like(blank_dur_mask)
            add_end.scatter_(1, end_slot, end_dur[:, None])
            add_end_mask.scatter_(1, end_slot, has_any_valid.float()[:, None])

            blank_dur_tgt = blank_dur_tgt + add_end
            blank_dur_mask = torch.clamp(blank_dur_mask + add_end_mask, 0.0, 1.0)

        # ---- monitor ----
        is_blank_frame = (~is_word_frame) & audio_mask
        monitor["mon_target_word_frame_count"] = is_word_frame.float().sum().detach()
        monitor["mon_target_blank_frame_count"] = is_blank_frame.float().sum().detach()
        monitor["mon_target_word_frame_ratio"] = (
            is_word_frame.float().sum() / audio_mask.float().sum().clamp_min(1.0)
        ).detach()

        if conf_weight is not None:
            monitor["mon_frame_weight_word_mean"] = (
                (frame_weight * is_word_frame.float()).sum() / is_word_frame.float().sum().clamp_min(1.0)
            ).detach()
        else:
            monitor["mon_frame_weight_word_mean"] = torch.tensor(1.0, device=device)

        return target, frame_weight, word_dur_tgt, blank_dur_tgt, word_dur_mask, blank_dur_mask, monitor

    # ---------- forward ----------
    def forward(
        self,
        wavs: torch.Tensor,
        texts: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        text_lens: Optional[torch.Tensor] = None,
        *,
        word_start_times: Optional[torch.Tensor] = None,
        word_end_times: Optional[torch.Tensor] = None,
        word_conf: Optional[torch.Tensor] = None,
    ):
        audio_feat, audio_mask, text_feat, text_mask = self.encode(wavs, texts, wav_lens, text_lens)
        wbd_feat, wbd_mask = self._gather_wbd_embeddings(text_feat, texts, text_mask)  # [B,W,C], [B,W]

        logits, state_mask, blank_mask = self._compute_state_logits(audio_feat, audio_mask, wbd_feat, wbd_mask)  # [B,T,S]

        use_frame_weight = bool(getattr(self.config, "use_frame_weight", True))
        word_conf_for_targets = word_conf  # 保留给 zero-duration policy / conf_split 使用

        target, frame_weight, word_dur_tgt, blank_dur_tgt, word_dur_mask, blank_dur_mask, target_monitor = \
            self._build_state_targets_from_times(
                audio_mask=audio_mask,
                wbd_mask=wbd_mask,
                word_start_times=word_start_times,
                word_end_times=word_end_times,
                word_conf=word_conf_for_targets,
            )

        if not use_frame_weight:
            frame_weight = torch.ones_like(frame_weight)

        B, T, S = logits.shape
        ignore_index = -100

        ce_per = F.cross_entropy(
            logits.float().reshape(-1, S),
            target.reshape(-1),
            reduction="none",
            ignore_index=ignore_index,
        ).view(B, T)

        valid = (target != ignore_index) & audio_mask
        denom = valid.float().sum().clamp_min(1.0)
        align_loss = (ce_per * frame_weight * valid.float()).sum() / denom

        probs = torch.softmax(logits.float(), dim=-1)  # [B,T,S]
        probs = probs * audio_mask[:, :, None].float()

        # duration preds
        probs_blank = probs[:, :, 0::2]  # [B,T,W+1]
        probs_word = probs[:, :, 1::2]   # [B,T,W]

        dur_blank_pred = probs_blank.sum(dim=1)  # [B,W+1]
        dur_word_pred = probs_word.sum(dim=1)    # [B,W]

        # duration loss (word)
        dur_loss = logits.new_tensor(0.0)
        mono_loss = logits.new_tensor(0.0)

        if dur_word_pred.numel() > 0 and word_dur_mask.sum() > 0:
            wm = word_dur_mask
            l1 = (dur_word_pred - word_dur_tgt).abs()
            l1 = (l1 * wm).sum() / wm.sum().clamp_min(1.0)

            log_l2 = (torch.log1p(dur_word_pred.clamp_min(0.0)) - torch.log1p(word_dur_tgt.clamp_min(0.0))).pow(2)
            log_l2 = (log_l2 * wm).sum() / wm.sum().clamp_min(1.0)

            dur_word_loss = l1 + log_l2
        else:
            dur_word_loss = logits.new_tensor(0.0)

        # duration loss (blank)
        if dur_blank_pred.numel() > 0 and blank_dur_mask.sum() > 0:
            bm = blank_dur_mask
            l1b = (dur_blank_pred - blank_dur_tgt).abs()
            l1b = (l1b * bm).sum() / bm.sum().clamp_min(1.0)

            log_l2b = (torch.log1p(dur_blank_pred.clamp_min(0.0)) - torch.log1p(blank_dur_tgt.clamp_min(0.0))).pow(2)
            log_l2b = (log_l2b * bm).sum() / bm.sum().clamp_min(1.0)

            dur_blank_loss = l1b + log_l2b
        else:
            dur_blank_loss = logits.new_tensor(0.0)

        dur_loss = dur_word_loss + float(getattr(self.config, "blank_dur_weight", 0.5)) * dur_blank_loss

        # mono loss (words only)
        W = probs_word.shape[-1]
        if W > 0:
            p_words = probs_word  # already masked by audio_mask
            p_sum = p_words.sum(dim=-1).clamp_min(1e-6)  # [B,T]
            idx = torch.arange(1, W + 1, device=logits.device, dtype=torch.float32)[None, None, :]
            expected = (p_words * idx).sum(dim=-1) / p_sum  # [B,T]

            dec = F.relu(expected[:, :-1] - expected[:, 1:])
            mono_mask = audio_mask[:, 1:] & audio_mask[:, :-1]
            mono_loss = (dec * mono_mask.float()).sum() / mono_mask.float().sum().clamp_min(1.0)

        monitor_stats = {}
        if bool(getattr(self.config, "enable_monitor_stats", True)):
            valid = (target != ignore_index) & audio_mask
            valid_f = valid.float()

            # p(target)
            target_safe = target.clamp_min(0)
            p_target = probs.gather(dim=-1, index=target_safe.unsqueeze(-1)).squeeze(-1)
            monitor_stats["mon_p_target_mean"] = (
                (p_target * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            ).detach()

            # entropy
            ent = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)
            monitor_stats["mon_entropy_valid_mean"] = (
                (ent * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            ).detach()

            # V4 word mass = odd states
            word_mass = probs[:, :, 1::2].sum(dim=-1)
            monitor_stats["mon_word_mass_valid_mean"] = (
                (word_mass * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            ).detach()

            # word-frame 专用 p(target)（只看奇数state）
            is_word_target = valid & ((target % 2) == 1)
            is_word_target_f = is_word_target.float()
            monitor_stats["mon_p_target_word_only_mean"] = (
                (p_target * is_word_target_f).sum() / is_word_target_f.sum().clamp_min(1.0)
            ).detach()

            for k, v in target_monitor.items():
                monitor_stats[k] = v.detach() if torch.is_tensor(v) else v

        out = {
            "align_loss": align_loss,
            "dur_loss": dur_loss,
            "mono_loss": mono_loss,
            "ntokens": audio_mask.sum(),
            "gamma": torch.exp(self.log_gamma).clamp(0.7, 30.0).detach(),
        }
        out.update(monitor_stats)
        return out

    # ---------- inference ----------
    @torch.inference_mode()
    def inference(
        self,
        wavs: torch.Tensor,
        texts: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        text_lens: Optional[torch.Tensor] = None,
        *,
        return_debug: bool = False,
    ):
        audio_feat, audio_mask, text_feat, text_mask = self.encode(wavs, texts, wav_lens, text_lens)
        wbd_feat, wbd_mask = self._gather_wbd_embeddings(text_feat, texts, text_mask)  # [B,Wraw,C], [B,Wraw]

        logits, state_mask, _ = self._compute_state_logits(audio_feat, audio_mask, wbd_feat, wbd_mask)
        logits = logits.float()

        probs_state = torch.softmax(logits, dim=-1)  # [B,T,S]
        B, T, Smax = probs_state.shape
        device = probs_state.device
        frame_sec = float(self._frame_sec())

        audio_lens = audio_mask.long().sum(dim=1)             # [B]
        word_lens_raw = wbd_mask.long().sum(dim=1)            # [B]
        word_lens = torch.minimum(word_lens_raw, audio_lens)  # [B]

        Tmax = int(audio_lens.max().detach().cpu().item()) if B > 0 else 0
        Wmax = int(word_lens.max().detach().cpu().item()) if B > 0 else 0

        results = []
        if Tmax <= 0:
            for _ in range(B):
                out = {
                    "word_start_times": torch.zeros((0,), device=wavs.device),
                    "word_end_times": torch.zeros((0,), device=wavs.device),
                    "word_conf": torch.zeros((0,), device=wavs.device),
                }
                if return_debug:
                    out["debug"] = {
                        "frame_sec": frame_sec,
                        "audio_len_frames": 0,
                        "word_len": 0,
                        "dp": torch.zeros((0, 1), dtype=torch.float16, device="cpu"),
                        "states": torch.zeros((0,), dtype=torch.long, device="cpu"),
                        "probs_words": torch.zeros((0, 0), dtype=torch.float16, device="cpu"),
                        "probs_blank": torch.zeros((0,), dtype=torch.float16, device="cpu"),
                    }
                results.append(out)
            return results

        # If Wmax==0, only blank0 exists for those samples
        Suse = 2 * Wmax + 1
        probs_state = probs_state[:, :Tmax, :Suse]
        logits = logits[:, :Tmax, :Suse]

        valid_time = (torch.arange(Tmax, device=device)[None, :] < audio_lens[:, None])  # [B,Tmax]

        # logp for DP
        logp = torch.log(probs_state.clamp_min(1e-12))  # [B,T,Suse]
        neg_inf = logp.new_tensor(-1e9)

        # DP valid states: s <= 2*word_lens (blankW index)
        s_ids = torch.arange(Suse, device=device, dtype=torch.long)[None, :]  # [1,Suse]
        state_valid = (s_ids <= (2 * word_lens)[:, None])  # [B,Suse]

        # init
        dp_prev = logp.new_full((B, Suse), fill_value=neg_inf)
        dp_prev[:, 0] = logp[:, 0, 0]
        if Suse >= 2:
            dp_prev[:, 1] = logp[:, 0, 1]
        dp_prev = dp_prev.masked_fill(~state_valid, neg_inf)

        dp_hist = None
        if return_debug:
            dp_hist = torch.empty((B, Tmax, Suse), device=device, dtype=torch.float16)
            dp_hist[:, 0, :] = dp_prev.to(torch.float16)

        bp_step = torch.zeros((B, Tmax, Suse), device=device, dtype=torch.uint8)

        t_end = (audio_lens - 1).clamp_min(0)
        dp_end = torch.where((t_end == 0)[:, None], dp_prev, dp_prev.new_full((B, Suse), neg_inf))

        is_word_state = (torch.arange(Suse, device=device) % 2 == 1)[None, :]  # [1,Suse]

        for t in range(1, Tmax):
            emit_t = logp[:, t, :]  # [B,Suse]

            prev = dp_prev
            adv1 = torch.cat([neg_inf.expand(B, 1), prev[:, :-1]], dim=1)

            if Suse >= 3:
                adv2 = torch.cat([neg_inf.expand(B, 2), prev[:, :-2]], dim=1)
                adv2 = torch.where(is_word_state, adv2, neg_inf)
            else:
                adv2 = neg_inf.expand(B, Suse)

            cand = torch.stack([prev, adv1, adv2], dim=0)  # [3,B,S]
            best_prev, step = cand.max(dim=0)

            dp_t = best_prev + emit_t

            vt = valid_time[:, t][:, None]
            dp_t = torch.where(vt, dp_t, prev)
            step = torch.where(vt, step.to(torch.uint8), torch.zeros_like(step, dtype=torch.uint8))

            dp_t = dp_t.masked_fill(~state_valid, neg_inf)
            step = torch.where(state_valid, step, torch.zeros_like(step, dtype=torch.uint8))

            dp_prev = dp_t
            bp_step[:, t, :] = step

            if dp_hist is not None:
                dp_hist[:, t, :] = dp_prev.to(torch.float16)

            end_mask = (t_end == t)
            if end_mask.any():
                dp_end[end_mask] = dp_prev[end_mask]

        end_blank_s = (2 * word_lens).clamp_min(0).clamp_max(Suse - 1)
        end_word_s = (end_blank_s - 1).clamp_min(0).clamp_max(Suse - 1)

        score_blank = dp_end.gather(1, end_blank_s[:, None]).squeeze(1)
        score_word = dp_end.gather(1, end_word_s[:, None]).squeeze(1)

        choose_blank = score_blank > score_word
        end_s = torch.where(
            word_lens == 0,
            torch.zeros_like(end_blank_s),
            torch.where(choose_blank, end_blank_s, end_word_s),
        )

        # backtrace
        states = torch.zeros((B, Tmax), device=device, dtype=torch.long)
        s = end_s.clone()

        for t in range(Tmax - 1, -1, -1):
            active = (t <= t_end)
            if active.any():
                states[active, t] = s[active]
            if t > 0:
                step_t = bp_step[:, t, :].gather(1, s[:, None]).squeeze(1).long()
                step_t = step_t * active.long()
                s = s - step_t

        # extract word spans
        w_ids = torch.arange(Wmax, device=device, dtype=torch.long)
        word_states = 2 * w_ids + 1
        valid_word = (w_ids[None, :] < word_lens[:, None])

        t_ids = torch.arange(Tmax, device=device, dtype=torch.long)[None, :, None]
        mask_word_frame = (
            valid_time[:, :, None]
            & valid_word[:, None, :]
            & (states[:, :, None] == word_states[None, None, :])
        )

        start_idx = torch.where(mask_word_frame, t_ids, torch.full_like(t_ids, Tmax)).amin(dim=1)
        end_idx = torch.where(mask_word_frame, t_ids, torch.full_like(t_ids, -1)).amax(dim=1) + 1
        has_span = (end_idx > 0) & valid_word

        start_idx = torch.where(has_span, start_idx, torch.zeros_like(start_idx))
        end_idx = torch.where(has_span, end_idx, torch.zeros_like(end_idx))

        word_start_times_all = start_idx.float() * frame_sec
        word_end_times_all = end_idx.float() * frame_sec

        # conf A: mean p(word_k) within assigned frames
        probs_words = probs_state[:, :, 1::2]  # [B,T,Wmax]
        conf_frame = probs_words * mask_word_frame.float()
        sum_conf = conf_frame.sum(dim=1)
        len_seg = mask_word_frame.float().sum(dim=1).clamp_min(1.0)
        word_conf_all = sum_conf / len_seg
        word_conf_all = torch.where(has_span, word_conf_all, torch.zeros_like(word_conf_all))

        # global blank prob for debug/plot (no blank columns)
        probs_blank_global = probs_state[:, :, 0::2].sum(dim=-1)  # [B,T]

        for b in range(B):
            Wb = int(word_lens[b].item())
            out = {
                "word_start_times": word_start_times_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
                "word_end_times": word_end_times_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
                "word_conf": word_conf_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
            }

            if return_debug:
                Tb = int(audio_lens[b].item())
                Sb = int(2 * Wb + 1) if Wb > 0 else 1

                dbg_dp = torch.zeros((0, 1), dtype=torch.float16, device="cpu")
                if dp_hist is not None and Tb > 0:
                    dbg_dp = dp_hist[b, :Tb, :Sb].detach().to("cpu")

                dbg_states = states[b, :Tb].detach().to("cpu") if Tb > 0 else torch.zeros((0,), dtype=torch.long)
                dbg_probs_blank = probs_blank_global[b, :Tb].detach().to("cpu").to(torch.float16) if Tb > 0 else torch.zeros((0,), dtype=torch.float16)

                if Wb > 0 and Tb > 0:
                    dbg_probs_words = probs_words[b, :Tb, :Wb].detach().to("cpu").to(torch.float16)
                else:
                    dbg_probs_words = torch.zeros((Tb, 0), dtype=torch.float16, device="cpu")

                out["debug"] = {
                    "frame_sec": frame_sec,
                    "audio_len_frames": Tb,
                    "word_len": Wb,
                    "dp": dbg_dp,
                    "states": dbg_states,
                    "probs_words": dbg_probs_words,   # T×W, no blank columns
                    "probs_blank": dbg_probs_blank,   # global blank = sum over all blank_k
                }

            results.append(out)

        return results


class ForcedAlignerV5(ForcedAlignerV4):
    """
    V5 = V4 unary(state logits) + transition potential + CRF training + Viterbi inference.
    状态空间保持 V4:
      blank0, word1, blank1, ..., wordW, blankW
    允许转移:
      stay(s->s), adv1(s-1->s), adv2(s-2->s; only for word states)
    """

    def __init__(self, config: ModelArgs):
        super().__init__(config)

        self.log_gamma = nn.Parameter(torch.ones([]) * math.log(10))
        self.gamma_max = 30.0

        hs = int(getattr(config, "hidden_size", 512))
        th = int(getattr(config, "v5_transition_hidden_size", 256))

        # 目标状态特征 -> incoming transition score (stay/adv1)
        self.v5_trans_stay_head = nn.Sequential(
            nn.Linear(hs, th),
            nn.SiLU(),
            nn.Linear(th, 1),
        )
        self.v5_trans_adv1_head = nn.Sequential(
            nn.Linear(hs, th),
            nn.SiLU(),
            nn.Linear(th, 1),
        )

        # adv2 (skip blank) 对 word_j 的额外 pair score: f([word_{j-1}, word_j])
        self.v5_enable_skip_pair_head = bool(getattr(config, "v5_enable_skip_pair_head", True))
        if self.v5_enable_skip_pair_head:
            self.v5_trans_adv2_pair_head = nn.Sequential(
                nn.Linear(2 * hs, th),
                nn.SiLU(),
                nn.Linear(th, 1),
            )
            nn.init.zeros_(self.v5_trans_adv2_pair_head[-1].weight)
            nn.init.zeros_(self.v5_trans_adv2_pair_head[-1].bias)
        else:
            self.v5_trans_adv2_pair_head = None

        # 三种 incoming transition 的全局 bias
        self.v5_trans_bias = nn.Parameter(torch.zeros(3))  # [stay, adv1, adv2]

        # 小初始化，避免训练初期 transition 过强
        for mod in [self.v5_trans_stay_head, self.v5_trans_adv1_head]:
            nn.init.zeros_(mod[-1].weight)
            nn.init.zeros_(mod[-1].bias)

    # ------------------------------------------------------------------
    # helpers: small fast kernels
    # ------------------------------------------------------------------
    @staticmethod
    def _v5_logaddexp3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # 比 stack(...).logsumexp(dim=0) 少一次大临时张量分配
        return torch.logaddexp(torch.logaddexp(a, b), c)

    @staticmethod
    def _v5_max3_with_step(
        cand_stay: torch.Tensor,
        cand_adv1: torch.Tensor,
        cand_adv2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 max 和 argmax(step in {0,1,2})，tie-break 与 torch.stack([stay,adv1,adv2]).max(dim=0) 一致：
        stay 优先于 adv1，adv1 优先于 adv2（因为仅在更大时才替换）。
        """
        best = cand_stay
        step = torch.zeros_like(cand_stay, dtype=torch.uint8)

        take_adv1 = cand_adv1 > best
        best = torch.where(take_adv1, cand_adv1, best)
        step = torch.where(take_adv1, torch.ones_like(step), step)

        take_adv2 = cand_adv2 > best
        best = torch.where(take_adv2, cand_adv2, best)
        step = torch.where(take_adv2, torch.full_like(step, 2), step)

        return best, step

    @staticmethod
    def _v5_select_trans_by_state_and_step(
        trans_in: torch.Tensor,   # [B,S,3]
        curr_state: torch.Tensor, # [B,Tm]
        step_idx: torch.Tensor,   # [B,Tm]
    ) -> torch.Tensor:
        """
        选择 trans_in[b, curr_state, step_idx] -> [B,Tm]
        比先 gather 成 [B,Tm,3] 再 gather 一次更省一点中间张量。
        """
        B, Tm = curr_state.shape
        if Tm == 0:
            return trans_in.new_zeros((B, 0))
        b_idx = torch.arange(B, device=trans_in.device)[:, None].expand(B, Tm)
        return trans_in[b_idx, curr_state, step_idx]

    # ------------------------------------------------------------------
    # helpers: build proto/state mask + unary + transition
    # ------------------------------------------------------------------
    def _build_state_proto_and_mask(
        self,
        wbd_feat: torch.Tensor,   # [B,W,C]
        wbd_mask: torch.Tensor,   # [B,W]
    ):
        """
        复用 V4 blank prototype 逻辑，返回 interleaved state proto / mask。
        """
        B, Wmax, C = wbd_feat.shape
        blank_feat, blank_mask, word_lens = self._build_blank_prototypes(wbd_feat, wbd_mask)

        Smax = 2 * Wmax + 1
        proto = wbd_feat.new_zeros((B, Smax, C))
        proto[:, 0::2, :] = blank_feat
        proto[:, 1::2, :] = wbd_feat

        state_mask = torch.zeros((B, Smax), dtype=torch.bool, device=wbd_feat.device)
        state_mask[:, 0::2] = blank_mask
        state_mask[:, 1::2] = wbd_mask
        return proto, state_mask, blank_mask, word_lens

    def _compute_v5_unary_and_transition(
        self,
        audio_feat: torch.Tensor,   # [B,T,C]
        audio_mask: torch.Tensor,   # [B,T]
        wbd_feat: torch.Tensor,     # [B,W,C]
        wbd_mask: torch.Tensor,     # [B,W]
    ):
        """
        Returns:
            unary_logits: [B,T,S]
            trans_in:     [B,S,3]   # incoming transition score for {stay, adv1, adv2}
            state_mask:   [B,S]
            blank_mask:   [B,W+1]
            proto:        [B,S,C]
        """
        B, T, C = audio_feat.shape
        Wmax = wbd_feat.shape[1]
        Smax = 2 * Wmax + 1
        device = audio_feat.device

        proto, state_mask, blank_mask, word_lens = self._build_state_proto_and_mask(wbd_feat, wbd_mask)

        use_fp32 = (not torch.is_grad_enabled())

        if use_fp32:
            if audio_feat.is_cuda:
                with torch.autocast(device_type="cuda", enabled=False):
                    audio_f = audio_feat.float()
                    proto_f = proto.float()

                    if self.config.sim_type == "cos":
                        a = F.normalize(audio_f, dim=-1)
                        p = F.normalize(proto_f, dim=-1)
                    else:
                        a, p = audio_f, proto_f

                    gamma = torch.exp(self.log_gamma.float()).clamp(0.7, self.gamma_max)
                    unary_logits = torch.bmm(a, p.transpose(1, 2)) * gamma
            else:
                audio_f = audio_feat.float()
                proto_f = proto.float()

                if self.config.sim_type == "cos":
                    a = F.normalize(audio_f, dim=-1)
                    p = F.normalize(proto_f, dim=-1)
                else:
                    a, p = audio_f, proto_f

                gamma = torch.exp(self.log_gamma.float()).clamp(0.7, self.gamma_max)
                unary_logits = torch.bmm(a, p.transpose(1, 2)) * gamma
        else:
            if self.config.sim_type == "cos":
                a = F.normalize(audio_feat, dim=-1)
                p = F.normalize(proto, dim=-1)
            else:
                a, p = audio_feat, proto

            gamma = torch.exp(self.log_gamma).clamp(0.7, self.gamma_max).to(a.dtype)
            unary_logits = torch.bmm(a, p.transpose(1, 2)) * gamma  # [B,T,S]

        finfo_u = torch.finfo(unary_logits.dtype)
        neg_inf_u = -1e9 if finfo_u.min <= -1e9 else float(finfo_u.min)

        finfo_p = torch.finfo(proto.dtype)
        neg_inf_p = -1e9 if finfo_p.min <= -1e9 else float(finfo_p.min)

        unary_logits = unary_logits.masked_fill(~state_mask[:, None, :], neg_inf_u)
        unary_logits = unary_logits.masked_fill(~audio_mask[:, :, None], neg_inf_u)

        # transition potential (incoming)
        tscale = float(getattr(self.config, "v5_transition_scale", 1.0))

        stay_score = self.v5_trans_stay_head(proto).squeeze(-1) + self.v5_trans_bias[0]  # [B,S]
        adv1_score = self.v5_trans_adv1_head(proto).squeeze(-1) + self.v5_trans_bias[1]  # [B,S]

        trans_in = proto.new_full((B, Smax, 3), fill_value=neg_inf_p)  # [B,S,3]
        trans_in[:, :, 0] = stay_score
        trans_in[:, :, 1] = adv1_score

        # adv2: only for word states s>=3 (word2, word3, ...)
        adv2_score = proto.new_full((B, Smax), fill_value=neg_inf)

        if Wmax >= 2:
            # 基础 bias（仅对可用 word state）
            adv2_score[:, 3:2 * Wmax:2] = self.v5_trans_bias[2]

            # pair head: [word_{j-1}, word_j] -> adv2 to word_j
            if self.v5_enable_skip_pair_head and self.v5_trans_adv2_pair_head is not None:
                left = wbd_feat[:, :-1, :]   # word1..word_{W-1}
                right = wbd_feat[:, 1:, :]   # word2..word_W
                pair = torch.cat([left, right], dim=-1)  # [B,W-1,2C]
                pair_score = self.v5_trans_adv2_pair_head(pair).squeeze(-1)  # [B,W-1]
                adv2_score[:, 3:2 * Wmax:2] = adv2_score[:, 3:2 * Wmax:2] + pair_score

        trans_in[:, :, 2] = adv2_score

        # 结构合法性约束（按目标状态 s）
        s_ids = torch.arange(Smax, device=device, dtype=torch.long)[None, :]  # [1,S]
        is_word_state = (s_ids % 2 == 1)
        can_adv1 = (s_ids >= 1)
        can_adv2 = (s_ids >= 3) & is_word_state

        trans_in[:, :, 1] = trans_in[:, :, 1].masked_fill(~can_adv1, neg_inf_p)
        trans_in[:, :, 2] = trans_in[:, :, 2].masked_fill(~can_adv2, neg_inf_p)

        # 无效状态不应该作为目标状态
        trans_in = trans_in.masked_fill(~state_mask[:, :, None], neg_inf_p)

        if tscale != 1.0:
            # 对 finite 值缩放，避免 -1e9 * scale
            finite_mask = (trans_in > neg_inf_p / 2)
            trans_in = torch.where(finite_mask, trans_in * tscale, trans_in)

        return unary_logits, trans_in, state_mask, blank_mask, proto

    # ------------------------------------------------------------------
    # CRF training helpers
    # ------------------------------------------------------------------
    def _crf_log_partition(
        self,
        unary_logits: torch.Tensor,  # [B,T,S]
        trans_in: torch.Tensor,      # [B,S,3]
        audio_mask: torch.Tensor,    # [B,T]
        word_lens: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        """
        线性链（但状态图受限）CRF 的 logZ。使用与 Viterbi 相同的三种转移拓扑。
        """
        B, T, Smax = unary_logits.shape
        device = unary_logits.device
        neg_inf = unary_logits.new_tensor(-1e9)

        audio_lens = audio_mask.long().sum(dim=1)
        Tmax = int(audio_lens.max().detach().cpu().item()) if B > 0 else 0
        if Tmax <= 0:
            return unary_logits.new_zeros((B,))

        valid_time = (torch.arange(Tmax, device=device)[None, :] < audio_lens[:, None])  # [B,T]
        s_ids = torch.arange(Smax, device=device, dtype=torch.long)[None, :]              # [1,S]
        state_valid = (s_ids <= (2 * word_lens)[:, None])                                 # [B,S]

        unary = unary_logits[:, :Tmax, :]
        trans_stay, trans_adv1, trans_adv2 = trans_in.unbind(dim=-1)  # each [B,S]

        neg_col1 = neg_inf.expand(B, 1)
        neg_col2 = neg_inf.expand(B, 2)

        # init: t=0 仅允许 state 0 / 1（与 V4/V5 inference 一致）
        alpha_prev = unary.new_full((B, Smax), fill_value=neg_inf)
        alpha_prev[:, 0] = unary[:, 0, 0]
        if Smax >= 2:
            alpha_prev[:, 1] = unary[:, 0, 1]
        alpha_prev = alpha_prev.masked_fill(~state_valid, neg_inf)

        t_end = (audio_lens - 1).clamp_min(0)
        alpha_end = torch.where((t_end == 0)[:, None], alpha_prev, alpha_prev.new_full((B, Smax), neg_inf))

        for t in range(1, Tmax):
            emit_t = unary[:, t, :]  # [B,S]

            # incoming candidates to state s
            stay = alpha_prev + trans_stay  # s->s
            adv1 = torch.cat([neg_col1, alpha_prev[:, :-1]], dim=1) + trans_adv1
            if Smax >= 3:
                adv2 = torch.cat([neg_col2, alpha_prev[:, :-2]], dim=1) + trans_adv2
            else:
                adv2 = neg_inf.expand(B, Smax)

            alpha_t = self._v5_logaddexp3(stay, adv1, adv2) + emit_t

            vt = valid_time[:, t][:, None]
            alpha_t = torch.where(vt, alpha_t, alpha_prev)
            alpha_t = alpha_t.masked_fill(~state_valid, neg_inf)

            alpha_prev = alpha_t

            end_mask = (t_end == t)
            if end_mask.any():
                alpha_end[end_mask] = alpha_prev[end_mask]

        end_blank_s = (2 * word_lens).clamp_min(0).clamp_max(Smax - 1)
        end_word_s = (end_blank_s - 1).clamp_min(0).clamp_max(Smax - 1)

        score_blank = alpha_end.gather(1, end_blank_s[:, None]).squeeze(1)
        score_word = alpha_end.gather(1, end_word_s[:, None]).squeeze(1)

        logZ = torch.logaddexp(score_blank, score_word)
        logZ = torch.where(word_lens == 0, score_blank, logZ)  # W=0 只能结束在 state0
        return logZ

    def _crf_target_path_score_and_reachable(
        self,
        unary_logits: torch.Tensor,   # [B,T,S]
        trans_in: torch.Tensor,       # [B,S,3]
        target: torch.Tensor,         # [B,T]
        audio_mask: torch.Tensor,     # [B,T]
        word_lens: Optional[torch.Tensor] = None,   # NEW
        state_mask: Optional[torch.Tensor] = None,   # NEW, [B,S]
        ignore_index: int = -100,
    ):
        """
        用 target state 序列计算 path score，并判断 target 是否在状态图上可达。
        """
        B, T, S = unary_logits.shape
        device = unary_logits.device

        valid = (target != ignore_index) & audio_mask
        target_safe = target.clamp_min(0).clamp_max(S - 1)

        # unary score
        unary_sel = unary_logits.gather(dim=-1, index=target_safe.unsqueeze(-1)).squeeze(-1)  # [B,T]
        unary_score = (unary_sel * valid.float()).sum(dim=1)  # [B]

        # -------- NEW: frame-level state legality checks --------
        # 1) dynamic state valid: s <= 2*word_lens
        if word_lens is not None:
            dyn_state_ok = (target_safe <= (2 * word_lens[:, None]).clamp_min(0))
        else:
            dyn_state_ok = torch.ones_like(valid, dtype=torch.bool)

        # 2) static state mask valid (V4/V5 blank internal mask, etc.)
        if state_mask is not None:
            state_ok_sel = state_mask.gather(dim=1, index=target_safe)  # [B,T]
        else:
            state_ok_sel = torch.ones_like(valid, dtype=torch.bool)

        frame_state_ok = (~valid) | (dyn_state_ok & state_ok_sel)
        frame_state_ok_all = frame_state_ok.all(dim=1)

        # pair transitions
        if T <= 1:
            if T == 0:
                reachable = torch.ones((B,), dtype=torch.bool, device=device)
                start_ok = reachable
                end_ok = reachable
            else:
                # t=0 的起始状态合法：只能是 0 或 1（在 valid 时）
                start_ok = (~valid[:, 0]) | ((target_safe[:, 0] == 0) | (target_safe[:, 0] == 1))

                # NEW: end-state legality（T=1 时 end 就是 t=0）
                if word_lens is not None:
                    end_state = target_safe[:, 0]
                    end_blank_s = (2 * word_lens).clamp_min(0).clamp_max(S - 1)
                    end_word_s = (end_blank_s - 1).clamp_min(0).clamp_max(S - 1)
                    end_ok = (end_state == end_blank_s) | ((word_lens > 0) & (end_state == end_word_s))
                    end_ok = torch.where(valid[:, 0], end_ok, torch.ones_like(end_ok, dtype=torch.bool))
                else:
                    end_ok = torch.ones((B,), dtype=torch.bool, device=device)

                reachable = start_ok & frame_state_ok_all & end_ok

            return unary_score, reachable, {
                "mon_v5_target_pair_valid_count": torch.tensor(0.0, device=device),
                "mon_v5_target_pair_unreachable_count": torch.tensor(0.0, device=device),
                "mon_v5_target_reachable_sample_ratio": reachable.float().mean().detach() if B > 0 else torch.tensor(0.0, device=device),
                # NEW
                "mon_v5_target_state_illegal_frame_count": ((valid & (~(dyn_state_ok & state_ok_sel))).float().sum()).detach(),
                "mon_v5_target_state_ok_ratio": ((valid & (dyn_state_ok & state_ok_sel)).float().sum() / valid.float().sum().clamp_min(1.0)).detach(),
                "mon_v5_target_end_ok_ratio": end_ok.float().mean().detach() if B > 0 else torch.tensor(0.0, device=device),
            }

        prev_t = target_safe[:, :-1]
        curr_t = target_safe[:, 1:]
        valid_pair = valid[:, :-1] & valid[:, 1:]  # [B,T-1]

        delta = curr_t - prev_t
        is_stay = (delta == 0)
        is_adv1 = (delta == 1)
        is_adv2 = (delta == 2) & ((curr_t % 2) == 1)

        pair_ok = is_stay | is_adv1 | is_adv2

        # t=0 的起始状态也要合法：只能是 0 或 1（在 valid 时）
        start_ok = (~valid[:, 0]) | ((target_safe[:, 0] == 0) | (target_safe[:, 0] == 1))

        # -------- NEW: end-state legality check --------
        if word_lens is not None:
            audio_lens = audio_mask.long().sum(dim=1)                      # [B]
            t_end = (audio_lens - 1).clamp_min(0)                          # [B]
            end_state = target_safe.gather(1, t_end[:, None]).squeeze(1)   # [B]

            end_blank_s = (2 * word_lens).clamp_min(0).clamp_max(S - 1)
            end_word_s = (end_blank_s - 1).clamp_min(0).clamp_max(S - 1)

            end_ok = (end_state == end_blank_s) | ((word_lens > 0) & (end_state == end_word_s))
            has_valid_frame = valid.any(dim=1)
            end_ok = torch.where(has_valid_frame, end_ok, torch.ones_like(end_ok, dtype=torch.bool))
        else:
            end_ok = torch.ones((B,), dtype=torch.bool, device=device)

        reachable = start_ok & frame_state_ok_all & (~(valid_pair & (~pair_ok)).any(dim=1)) & end_ok

        # transition type idx
        step_idx = torch.zeros_like(curr_t)  # default stay=0
        step_idx = torch.where(is_adv1, torch.ones_like(step_idx), step_idx)
        step_idx = torch.where(is_adv2, torch.full_like(step_idx, 2), step_idx)

        # 取 trans_in[b, curr_state, step_idx]
        trans_sel = self._v5_select_trans_by_state_and_step(trans_in, curr_t, step_idx)  # [B,T-1]

        trans_score = (trans_sel * (valid_pair & pair_ok).float()).sum(dim=1)
        path_score = unary_score + trans_score

        mon = {
            "mon_v5_target_pair_valid_count": valid_pair.float().sum().detach(),
            "mon_v5_target_pair_unreachable_count": (valid_pair & (~pair_ok)).float().sum().detach(),
            "mon_v5_target_reachable_sample_ratio": reachable.float().mean().detach(),
            # NEW
            "mon_v5_target_state_illegal_frame_count": ((valid & (~(dyn_state_ok & state_ok_sel))).float().sum()).detach(),
            "mon_v5_target_state_ok_ratio": ((valid & (dyn_state_ok & state_ok_sel)).float().sum() / valid.float().sum().clamp_min(1.0)).detach(),
            "mon_v5_target_end_ok_ratio": end_ok.float().mean().detach(),
        }
        return path_score, reachable, mon

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        wavs: torch.Tensor,
        texts: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        text_lens: Optional[torch.Tensor] = None,
        *,
        word_start_times: Optional[torch.Tensor] = None,
        word_end_times: Optional[torch.Tensor] = None,
        word_conf: Optional[torch.Tensor] = None,
    ):
        # 1) 编码
        audio_feat, audio_mask, text_feat, text_mask = self.encode(wavs, texts, wav_lens, text_lens)
        wbd_feat, wbd_mask = self._gather_wbd_embeddings(text_feat, texts, text_mask)

        # 2) V5 unary + transition
        unary_logits, trans_in, state_mask, blank_mask, proto = self._compute_v5_unary_and_transition(
            audio_feat, audio_mask, wbd_feat, wbd_mask
        )  # unary_logits:[B,T,S]

        # 3) 构造 state-level target（复用 V4）
        use_frame_weight = bool(getattr(self.config, "use_frame_weight", True))
        word_conf_used = word_conf if use_frame_weight else None

        target, frame_weight, word_dur_tgt, blank_dur_tgt, word_dur_mask, blank_dur_mask, target_monitor = \
            self._build_state_targets_from_times(
                audio_mask=audio_mask,
                wbd_mask=wbd_mask,
                word_start_times=word_start_times,
                word_end_times=word_end_times,
                word_conf=word_conf_used,
            )

        if not use_frame_weight:
            frame_weight = torch.ones_like(frame_weight)

        B, T, S = unary_logits.shape
        ignore_index = -100

        # 统一 float32（避免重复 cast）
        unary_f = unary_logits.float()
        trans_f = trans_in.float()

        # 4) CE (aux)
        ce_per = F.cross_entropy(
            unary_f.reshape(-1, S),
            target.reshape(-1),
            reduction="none",
            ignore_index=ignore_index,
        ).view(B, T)

        valid = (target != ignore_index) & audio_mask
        valid_f = valid.float()
        ce_loss = (ce_per * frame_weight * valid_f).sum() / valid_f.sum().clamp_min(1.0)

        # 5) CRF NLL (main)
        word_lens = wbd_mask.long().sum(dim=1)

        # NEW: 保守判定“target 被压缩/丢词”的样本（包括 zero-drop / 时间非法导致的 drop）
        # V4 target builder 返回的 word_dur_mask 本质是 valid_words.float()
        valid_word_lens = word_dur_mask.sum(dim=1).long()   # [B]
        drop_like_samples = (valid_word_lens < word_lens)   # [B]

        crf_loss = unary_logits.new_tensor(0.0)

        if bool(getattr(self.config, "v5_use_crf_loss", True)):
            logZ = self._crf_log_partition(
                unary_logits=unary_f,
                trans_in=trans_f,
                audio_mask=audio_mask,
                word_lens=word_lens,
            )  # [B]

            path_score, reachable, crf_mon = self._crf_target_path_score_and_reachable(
                unary_logits=unary_f,
                trans_in=trans_f,
                target=target,
                audio_mask=audio_mask,
                word_lens=word_lens,       # NEW
                state_mask=state_mask,     # NEW
                ignore_index=ignore_index,
            )  # [B], [B]

            # NEW: 对 drop-like 样本跳过 CRF（保守稳定版）
            crf_active = reachable & (~drop_like_samples)

            if crf_active.any():
                valid_len = valid.float().sum(dim=1).clamp_min(1.0)   # [B]
                nll = (logZ - path_score) / valid_len
                crf_loss = (nll * crf_active.float()).sum() / crf_active.float().sum().clamp_min(1.0)
            else:
                crf_loss = unary_logits.new_tensor(0.0)

            # NEW: 额外 monitor
            crf_mon["mon_v5_crf_reachable_count"] = reachable.float().sum().detach()
            crf_mon["mon_v5_crf_active_sample_count"] = crf_active.float().sum().detach()
            crf_mon["mon_v5_crf_skip_droplike_sample_count"] = drop_like_samples.float().sum().detach()
            crf_mon["mon_v5_crf_active_sample_ratio"] = crf_active.float().mean().detach()

        else:
            crf_mon = {
                "mon_v5_target_pair_valid_count": unary_logits.new_tensor(0.0),
                "mon_v5_target_pair_unreachable_count": unary_logits.new_tensor(0.0),
                "mon_v5_target_reachable_sample_ratio": unary_logits.new_tensor(0.0),
                # NEW
                "mon_v5_target_state_illegal_frame_count": unary_logits.new_tensor(0.0),
                "mon_v5_target_state_ok_ratio": unary_logits.new_tensor(0.0),
                "mon_v5_target_end_ok_ratio": unary_logits.new_tensor(0.0),
                "mon_v5_crf_reachable_count": unary_logits.new_tensor(0.0),
                "mon_v5_crf_active_sample_count": unary_logits.new_tensor(0.0),
                "mon_v5_crf_skip_droplike_sample_count": unary_logits.new_tensor(0.0),
                "mon_v5_crf_active_sample_ratio": unary_logits.new_tensor(0.0),
            }

        # crf_w = float(getattr(self.config, "v5_crf_loss_weight", 1.0))
        # align_loss = ce_w * ce_loss + crf_w * crf_loss if bool(getattr(self.config, "v5_use_crf_loss", True)) else ce_loss
        align_loss = ce_loss

        # 6) probs（用于 duration / mono / monitor）
        # 只做一次 softmax，后面 monitor 直接复用 probs_nomask
        probs_nomask = torch.softmax(unary_f, dim=-1)                  # [B,T,S]
        probs = probs_nomask * audio_mask[:, :, None].float()          # [B,T,S]
        probs_blank = probs[:, :, 0::2]  # [B,T,W+1]
        probs_word = probs[:, :, 1::2]   # [B,T,W]

        # 7) duration / mono（沿用 V4）
        dur_blank_pred = probs_blank.sum(dim=1)  # [B,W+1]
        dur_word_pred = probs_word.sum(dim=1)    # [B,W]

        dur_loss = unary_logits.new_tensor(0.0)
        mono_loss = unary_logits.new_tensor(0.0)

        if dur_word_pred.numel() > 0 and word_dur_mask.sum() > 0:
            wm = word_dur_mask
            l1 = (dur_word_pred - word_dur_tgt).abs()
            l1 = (l1 * wm).sum() / wm.sum().clamp_min(1.0)

            log_l2 = (torch.log1p(dur_word_pred.clamp_min(0.0)) - torch.log1p(word_dur_tgt.clamp_min(0.0))).pow(2)
            log_l2 = (log_l2 * wm).sum() / wm.sum().clamp_min(1.0)

            dur_word_loss = l1 + log_l2
        else:
            dur_word_loss = unary_logits.new_tensor(0.0)

        if dur_blank_pred.numel() > 0 and blank_dur_mask.sum() > 0:
            bm = blank_dur_mask
            l1b = (dur_blank_pred - blank_dur_tgt).abs()
            l1b = (l1b * bm).sum() / bm.sum().clamp_min(1.0)

            log_l2b = (torch.log1p(dur_blank_pred.clamp_min(0.0)) - torch.log1p(blank_dur_tgt.clamp_min(0.0))).pow(2)
            log_l2b = (log_l2b * bm).sum() / bm.sum().clamp_min(1.0)

            dur_blank_loss = l1b + log_l2b
        else:
            dur_blank_loss = unary_logits.new_tensor(0.0)

        dur_loss = dur_word_loss + float(getattr(self.config, "blank_dur_weight", 0.5)) * dur_blank_loss

        W = probs_word.shape[-1]
        if W > 0:
            p_words = probs_word
            p_sum = p_words.sum(dim=-1).clamp_min(1e-6)
            idx = torch.arange(1, W + 1, device=unary_logits.device, dtype=torch.float32)[None, None, :]
            expected = (p_words * idx).sum(dim=-1) / p_sum

            dec = F.relu(expected[:, :-1] - expected[:, 1:])
            mono_mask = audio_mask[:, 1:] & audio_mask[:, :-1]
            mono_loss = (dec * mono_mask.float()).sum() / mono_mask.float().sum().clamp_min(1.0)

        # 8) monitor stats
        monitor_stats = {}
        if bool(getattr(self.config, "enable_monitor_stats", True)):
            target_safe = target.clamp_min(0).clamp_max(S - 1)

            p_target = probs_nomask.gather(dim=-1, index=target_safe.unsqueeze(-1)).squeeze(-1)
            monitor_stats["mon_p_target_mean"] = ((p_target * valid_f).sum() / valid_f.sum().clamp_min(1.0)).detach()

            ent = -(probs_nomask.clamp_min(1e-12).log() * probs_nomask).sum(dim=-1)
            monitor_stats["mon_entropy_valid_mean"] = ((ent * valid_f).sum() / valid_f.sum().clamp_min(1.0)).detach()

            word_mass = probs_nomask[:, :, 1::2].sum(dim=-1)
            monitor_stats["mon_word_mass_valid_mean"] = ((word_mass * valid_f).sum() / valid_f.sum().clamp_min(1.0)).detach()

            is_word_target = valid & ((target_safe % 2) == 1)
            is_word_target_f = is_word_target.float()
            monitor_stats["mon_p_target_word_only_mean"] = (
                (p_target * is_word_target_f).sum() / is_word_target_f.sum().clamp_min(1.0)
            ).detach()

            # transition 统计（看 target path 上的 transition）
            if T > 1:
                prev_t = target_safe[:, :-1]
                curr_t = target_safe[:, 1:]
                valid_pair = valid[:, :-1] & valid[:, 1:]
                delta = curr_t - prev_t
                is_stay = valid_pair & (delta == 0)
                is_adv1 = valid_pair & (delta == 1)
                is_adv2 = valid_pair & (delta == 2) & ((curr_t % 2) == 1)

                # 只在 monitor 时取 [B,T-1,3]
                trans_all_cur = trans_f.gather(
                    dim=1, index=curr_t.unsqueeze(-1).expand(-1, -1, 3)
                )  # [B,T-1,3]

                monitor_stats["mon_v5_trans_stay_mean"] = (
                    (trans_all_cur[:, :, 0] * is_stay.float()).sum() / is_stay.float().sum().clamp_min(1.0)
                ).detach()
                monitor_stats["mon_v5_trans_adv1_mean"] = (
                    (trans_all_cur[:, :, 1] * is_adv1.float()).sum() / is_adv1.float().sum().clamp_min(1.0)
                ).detach()
                monitor_stats["mon_v5_trans_adv2_mean"] = (
                    (trans_all_cur[:, :, 2] * is_adv2.float()).sum() / is_adv2.float().sum().clamp_min(1.0)
                ).detach()
            else:
                z = unary_logits.new_tensor(0.0)
                monitor_stats["mon_v5_trans_stay_mean"] = z
                monitor_stats["mon_v5_trans_adv1_mean"] = z
                monitor_stats["mon_v5_trans_adv2_mean"] = z

            # attach target builder monitor
            for k, v in target_monitor.items():
                monitor_stats[k] = v.detach() if torch.is_tensor(v) else v
            for k, v in crf_mon.items():
                monitor_stats[k] = v.detach() if torch.is_tensor(v) else v

            monitor_stats["mon_v5_ce_loss"] = ce_loss.detach()
            monitor_stats["mon_v5_crf_loss"] = crf_loss.detach()

        out = {
            "align_loss": align_loss,
            "crf_loss": crf_loss,
            # "ce_loss": ce_loss.detach(),     # 方便观察
            # "crf_loss": crf_loss.detach(),   # 方便观察
            "dur_loss": dur_loss,
            "mono_loss": mono_loss,
            "ntokens": audio_mask.sum(),
            "gamma": torch.exp(self.log_gamma).clamp(0.7, self.gamma_max).detach(),
        }
        out.update(monitor_stats)
        return out

    def _v5_decode_states_viterbi(
        self,
        unary: torch.Tensor,   # [B, Tmax, Suse]
        trans_in: torch.Tensor,       # [B, Suse, 3]
        audio_lens: torch.Tensor,     # [B]
        word_lens: torch.Tensor,      # [B]
        *,
        return_debug: bool = False,
        backtrace_on_cpu: bool = True,
        profiler=None,
    ):
        B, Tmax, Suse = unary.shape
        device = unary.device
        neg_inf = unary.new_tensor(-1e9)
        prof = profiler
        
        valid_time = (torch.arange(Tmax, device=device)[None, :] < audio_lens[:, None])  # [B,T]
        s_ids = torch.arange(Suse, device=device, dtype=torch.long)[None, :]  # [1,S]
        state_valid = (s_ids <= (2 * word_lens)[:, None])  # [B,S]

        trans_stay, trans_adv1, trans_adv2 = trans_in.unbind(dim=-1)
        # neg_col1 = neg_inf.expand(B, 1)
        # neg_col2 = neg_inf.expand(B, 2)

        # 用于构造 "shifted prev"（相当于原先 cat([neg, prev[:, :-1]]) / cat([neg,neg, prev[:, :-2]])）
        shift_adv1_buf = unary.new_empty((B, Suse))  # [B,S]
        shift_adv2_buf = unary.new_empty((B, Suse)) if Suse >= 3 else None  # [B,S] or None

        # 用于 step 的无效位置清零，避免循环里 zeros_like(step)
        zero_step_buf = torch.zeros((B, Suse), device=device, dtype=torch.uint8)

        with (prof.section("viterbi_dp_forward") if prof is not None else nullcontext()):
            # init
            dp_prev = unary.new_full((B, Suse), fill_value=neg_inf)
            dp_prev[:, 0] = unary[:, 0, 0]
            if Suse >= 2:
                dp_prev[:, 1] = unary[:, 0, 1]
            dp_prev = dp_prev.masked_fill(~state_valid, neg_inf)

            dp_hist = None
            if return_debug:
                dp_hist = torch.empty((B, Tmax, Suse), device=device, dtype=torch.float16)
                dp_hist[:, 0, :] = dp_prev.to(torch.float16)

            bp_step = torch.zeros((B, Tmax, Suse), device=device, dtype=torch.uint8)

            t_end = (audio_lens - 1).clamp_min(0)
            dp_end = torch.where((t_end == 0)[:, None], dp_prev, dp_prev.new_full((B, Suse), neg_inf))

            min_audio_len = int(audio_lens.min().detach().cpu().item()) if B > 0 else 0

            has_adv2 = (Suse >= 3)

            # -----------------------------
            # A) 前缀段：所有样本都 active（无需 vt/where）
            # t = 1 .. min_audio_len-1
            # -----------------------------
            for t in range(1, min_audio_len):
                emit_t = unary[:, t, :]  # [B,S]
                prev = dp_prev

                cand_stay = prev + trans_stay

                shift_adv1_buf[:, 0] = neg_inf
                shift_adv1_buf[:, 1:] = prev[:, :-1]
                cand_adv1 = shift_adv1_buf + trans_adv1

                if has_adv2:
                    shift_adv2_buf[:, :2] = neg_inf
                    shift_adv2_buf[:, 2:] = prev[:, :-2]
                    cand_adv2 = shift_adv2_buf + trans_adv2
                else:
                    cand_adv2 = shift_adv1_buf
                    cand_adv2.fill_(neg_inf)

                best_prev, step = self._v5_max3_with_step(cand_stay, cand_adv1, cand_adv2)
                dp_t = best_prev + emit_t

                # 这里所有样本都 active，不需要 vt = valid_time[:, t]
                # 仍保留 state_valid mask（低风险版）
                dp_t = dp_t.masked_fill(~state_valid, neg_inf)
                step = torch.where(state_valid, step.to(torch.uint8), zero_step_buf)

                dp_prev = dp_t
                bp_step[:, t, :] = step

                if dp_hist is not None:
                    dp_hist[:, t, :] = dp_prev.to(torch.float16)

                end_mask = (t_end == t)
                if end_mask.any():
                    dp_end[end_mask] = dp_prev[end_mask]

            # -----------------------------
            # B) 尾部段：部分样本已结束（需要 vt/where）
            # t = min_audio_len .. Tmax-1
            # -----------------------------
            for t in range(max(1, min_audio_len), Tmax):
                emit_t = unary[:, t, :]  # [B,S]
                prev = dp_prev

                cand_stay = prev + trans_stay

                shift_adv1_buf[:, 0] = neg_inf
                shift_adv1_buf[:, 1:] = prev[:, :-1]
                cand_adv1 = shift_adv1_buf + trans_adv1

                if has_adv2:
                    shift_adv2_buf[:, :2] = neg_inf
                    shift_adv2_buf[:, 2:] = prev[:, :-2]
                    cand_adv2 = shift_adv2_buf + trans_adv2
                else:
                    cand_adv2 = shift_adv1_buf
                    cand_adv2.fill_(neg_inf)

                best_prev, step = self._v5_max3_with_step(cand_stay, cand_adv1, cand_adv2)
                dp_t = best_prev + emit_t

                vt = valid_time[:, t][:, None]
                dp_t = torch.where(vt, dp_t, prev)
                step = torch.where(vt, step.to(torch.uint8), zero_step_buf)

                dp_t = dp_t.masked_fill(~state_valid, neg_inf)
                step = torch.where(state_valid, step, zero_step_buf)

                dp_prev = dp_t
                bp_step[:, t, :] = step

                if dp_hist is not None:
                    dp_hist[:, t, :] = dp_prev.to(torch.float16)

                end_mask = (t_end == t)
                if end_mask.any():
                    dp_end[end_mask] = dp_prev[end_mask]

        with (prof.section("viterbi_backtrace") if prof is not None else nullcontext()):
            end_blank_s = (2 * word_lens).clamp_min(0).clamp_max(Suse - 1)
            end_word_s = (end_blank_s - 1).clamp_min(0).clamp_max(Suse - 1)

            score_blank = dp_end.gather(1, end_blank_s[:, None]).squeeze(1)
            score_word = dp_end.gather(1, end_word_s[:, None]).squeeze(1)

            choose_blank = score_blank > score_word
            end_s = torch.where(
                word_lens == 0,
                torch.zeros_like(end_blank_s),
                torch.where(choose_blank, end_blank_s, end_word_s),
            )  # [B]

            use_cpu_backtrace = bool(backtrace_on_cpu and (B <= 100))

            if use_cpu_backtrace:
                # ---- CPU backtrace (通常对 B 小、T 大更友好) ----
                # 只搬运回溯必需数据
                bp_step_cpu = bp_step.detach().to("cpu", non_blocking=False)   # [B,T,S] uint8
                t_end_cpu = t_end.detach().to("cpu")
                end_s_cpu = end_s.detach().to("cpu")

                states_cpu = torch.zeros((B, Tmax), dtype=torch.long, device="cpu")

                # Python 循环在 CPU 上执行
                for b in range(B):
                    Tb_end = int(t_end_cpu[b].item())
                    s_b = int(end_s_cpu[b].item())
                    for t in range(Tb_end, -1, -1):
                        states_cpu[b, t] = s_b
                        if t > 0:
                            step_bt = int(bp_step_cpu[b, t, s_b].item())  # {0,1,2}
                            s_b -= step_bt

                states = states_cpu.to(device=device, non_blocking=False)
            else:
                # ---- 原 GPU backtrace ----
                states = torch.zeros((B, Tmax), device=device, dtype=torch.long)
                s = end_s.clone()

                for t in range(Tmax - 1, -1, -1):
                    active = (t <= t_end)
                    if active.any():
                        states[active, t] = s[active]
                    if t > 0:
                        step_t = bp_step[:, t, :].gather(1, s[:, None]).squeeze(1).long()
                        step_t = step_t * active.long()
                        s = s - step_t


        return states, {"dp": dp_hist}  # dp_hist可选

    # ------------------------------------------------------------------
    # inference: Viterbi with transition potential
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def inference(
        self,
        wavs: torch.Tensor,
        texts: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        text_lens: Optional[torch.Tensor] = None,
        emit_temp: float = 1.0,
        trans_temp: float = 1.0,
        *,
        backtrace_on_cpu: bool = True,
        return_debug: bool = False,
        profile: bool = False,
    ):

        from utils.profile.cuda_wall import CudaWallProfiler
        prof = CudaWallProfiler(enabled=profile, device=wavs.device)

        with prof.section("encode"):
            audio_feat, audio_mask, text_feat, text_mask = self.encode(wavs, texts, wav_lens, text_lens)
        with prof.section("gather_wbd"):
            wbd_feat, wbd_mask = self._gather_wbd_embeddings(text_feat, texts, text_mask)

        with prof.section("compute_state_logits"):
            with tf32_enable(enable=False):
                unary_logits, trans_in, state_mask, _, _ = self._compute_v5_unary_and_transition(
                    audio_feat, audio_mask, wbd_feat, wbd_mask
                )
            unary_logits = unary_logits.float()
            trans_in = trans_in.float()

        B, T, Smax_full = unary_logits.shape
        device = unary_logits.device
        frame_sec = float(self._frame_sec())

        audio_lens = audio_mask.long().sum(dim=1)             # [B]
        word_lens_raw = wbd_mask.long().sum(dim=1)            # [B]
        word_lens = torch.minimum(word_lens_raw, audio_lens)  # [B]

        Tmax = int(audio_lens.max().detach().cpu().item()) if B > 0 else 0
        Wmax = int(word_lens.max().detach().cpu().item()) if B > 0 else 0

        results = []
        if Tmax <= 0:
            for _ in range(B):
                out = {
                    "word_start_times": torch.zeros((0,), device=wavs.device),
                    "word_end_times": torch.zeros((0,), device=wavs.device),
                    "word_conf": torch.zeros((0,), device=wavs.device),
                }
                if return_debug:
                    out["debug"] = {
                        "frame_sec": frame_sec,
                        "audio_len_frames": 0,
                        "word_len": 0,
                        "dp": torch.zeros((0, 1), dtype=torch.float16, device="cpu"),
                        "states": torch.zeros((0,), dtype=torch.long, device="cpu"),
                        "probs_words": torch.zeros((0, 0), dtype=torch.float16, device="cpu"),
                        "probs_blank": torch.zeros((0,), dtype=torch.float16, device="cpu"),
                    }
                results.append(out)
            return results

        # 截断到 batch 内最大有效 W
        Suse = 2 * Wmax + 1
        unary = unary_logits[:, :Tmax, :Suse].float() / max(emit_temp, 1e-6)         # [B,T,S]
        trans_in = trans_in[:, :Suse, :].float() / max(trans_temp, 1e-6)              # [B,S,3]

        # debug/conf 用 emission-only prob（和 V4 一致风格）
        # 只在截断后 softmax，避免在 padding 区域浪费计算
        probs_state = torch.softmax(unary, dim=-1)

        valid_time = (torch.arange(Tmax, device=device)[None, :] < audio_lens[:, None])  # [B,Tmax]

        states, dec_aux = self._v5_decode_states_viterbi(
            unary=unary,
            trans_in=trans_in,
            audio_lens=audio_lens,
            word_lens=word_lens,
            return_debug=return_debug,
            backtrace_on_cpu=backtrace_on_cpu,
            profiler=prof,
        )
        dp_hist = dec_aux['dp']

        # 提取 word spans（沿用 V4）
        w_ids = torch.arange(Wmax, device=device, dtype=torch.long)
        word_states = 2 * w_ids + 1
        valid_word = (w_ids[None, :] < word_lens[:, None])

        t_ids = torch.arange(Tmax, device=device, dtype=torch.long)[None, :, None]
        mask_word_frame = (
            valid_time[:, :, None]
            & valid_word[:, None, :]
            & (states[:, :, None] == word_states[None, None, :])
        )

        start_idx = torch.where(mask_word_frame, t_ids, torch.full_like(t_ids, Tmax)).amin(dim=1)
        end_idx = torch.where(mask_word_frame, t_ids, torch.full_like(t_ids, -1)).amax(dim=1) + 1
        has_span = (end_idx > 0) & valid_word

        start_idx = torch.where(has_span, start_idx, torch.zeros_like(start_idx))
        end_idx = torch.where(has_span, end_idx, torch.zeros_like(end_idx))

        word_start_times_all = start_idx.float() * frame_sec
        word_end_times_all = end_idx.float() * frame_sec

        # conf（保持 V4 风格：段内 mean p(word_k)）
        probs_words = probs_state[:, :, 1::2]  # [B,T,W]
        conf_frame = probs_words * mask_word_frame.float()
        sum_conf = conf_frame.sum(dim=1)
        len_seg = mask_word_frame.float().sum(dim=1).clamp_min(1.0)
        word_conf_all = sum_conf / len_seg
        word_conf_all = torch.where(has_span, word_conf_all, torch.zeros_like(word_conf_all))

        # debug: global blank prob
        probs_blank_global = probs_state[:, :, 0::2].sum(dim=-1)  # [B,T]

        with prof.section("pack_results"):
            for b in range(B):
                Wb = int(word_lens[b].item())
                out = {
                    "word_start_times": word_start_times_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
                    "word_end_times": word_end_times_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
                    "word_conf": word_conf_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
                }

                if return_debug:
                    Tb = int(audio_lens[b].item())
                    Sb = int(2 * Wb + 1) if Wb > 0 else 1

                    dbg_dp = torch.zeros((0, 1), dtype=torch.float16, device="cpu")
                    if dp_hist is not None and Tb > 0:
                        dbg_dp = dp_hist[b, :Tb, :Sb].detach().to("cpu")

                    dbg_states = states[b, :Tb].detach().to("cpu") if Tb > 0 else torch.zeros((0,), dtype=torch.long)
                    dbg_probs_blank = probs_blank_global[b, :Tb].detach().to("cpu").to(torch.float16) if Tb > 0 else torch.zeros((0,), dtype=torch.float16)

                    if Wb > 0 and Tb > 0:
                        dbg_probs_words = probs_words[b, :Tb, :Wb].detach().to("cpu").to(torch.float16)
                    else:
                        dbg_probs_words = torch.zeros((Tb, 0), dtype=torch.float16, device="cpu")

                    out["debug"] = {
                        "frame_sec": frame_sec,
                        "audio_len_frames": Tb,
                        "word_len": Wb,
                        "dp": dbg_dp,
                        "states": dbg_states,
                        "probs_words": dbg_probs_words,   # T×W, no blank columns
                        "probs_blank": dbg_probs_blank,   # global blank = sum over all blank_k
                    }

                results.append(out)

        if profile:
            profile_report = prof.report(extra={
                "model": "ForcedAlignerV5",
                "B": int(B),
                "Tmax": int(Tmax),
                "Wmax": int(Wmax),
                "Suse": int(Suse),
                "return_debug": bool(return_debug),
            })
            del profile_report['phases_agg_sorted_by_wall']
            print(json.dumps(profile_report, indent=4, ensure_ascii=False))

        return results
    

class ForcedAlignerV6(ForcedAlignerV5):
    """
    V6:
      - unary canonicalization (per-frame over valid states)
      - transition canonicalization (per-sample over valid transition entries)
      - dual gamma: gamma_unary / gamma_trans
      - posterior decode mode (topology-constrained posterior decoding)
    """
    def __init__(self, config: ModelArgs):
        super().__init__(config)

        # ===== dual gamma =====
        # 保留 self.log_gamma 作为 unary gamma（兼容 V5 输出里的 gamma）
        self.log_gamma = nn.Parameter(torch.ones([]) * float(getattr(config, "v6_log_gamma_unary_init", math.log(10.0))))
        self.log_gamma_trans = nn.Parameter(torch.ones([]) * float(getattr(config, "v6_log_gamma_trans_init", math.log(1.0))))

        self.gamma_max = float(getattr(config, "v6_gamma_unary_max", 30.0))
        self.gamma_trans_max = float(getattr(config, "v6_gamma_trans_max", 10.0))

    # ---------------------------
    # masked normalization helpers
    # ---------------------------
    def _masked_zscore(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        dims,
        eps: float = 1e-5,
        clip: Optional[float] = None,
    ) -> torch.Tensor:
        """
        x: arbitrary float tensor
        mask: bool tensor, same shape as x
        dims: tuple of dims to normalize over
        """
        mask_f = mask.to(dtype=x.dtype)
        cnt = mask_f.sum(dim=dims, keepdim=True).clamp_min(1.0)

        mean = (x * mask_f).sum(dim=dims, keepdim=True) / cnt
        xc = x - mean

        var = ((xc * xc) * mask_f).sum(dim=dims, keepdim=True) / cnt
        std = torch.sqrt(var + eps)

        z = xc / std
        if clip is not None and clip > 0:
            z = z.clamp(min=-clip, max=clip)

        # 无效位置保留原值（后面会继续 masked_fill 成 neg_inf）
        z = torch.where(mask, z, x)
        return z

    def _canon_unary(
        self,
        unary_logits: torch.Tensor,   # [B,T,S], raw (尚未 masked_fill neg_inf 或已 mask 均可)
        state_mask: torch.Tensor,     # [B,S]
        audio_mask: torch.Tensor,     # [B,T]
    ) -> torch.Tensor:
        if not bool(getattr(self.config, "v6_enable_unary_canon", True)):
            return unary_logits
        if self.training and not bool(getattr(self.config, "v6_canon_apply_in_train", True)):
            return unary_logits

        valid = state_mask[:, None, :] & audio_mask[:, :, None]  # [B,T,S]
        eps = float(getattr(self.config, "v6_canon_eps", 1e-5))
        clip = float(getattr(self.config, "v6_canon_clip", 8.0))
        # 逐帧（每个 b,t）在 state 维做 z-score
        return self._masked_zscore(unary_logits, valid, dims=(-1,), eps=eps, clip=clip)

    def _canon_trans(
        self,
        trans_in: torch.Tensor,       # [B,S,3]
        trans_valid_mask: torch.Tensor,   # [B,S,3] bool
    ) -> torch.Tensor:
        if not bool(getattr(self.config, "v6_enable_trans_canon", True)):
            return trans_in
        if self.training and not bool(getattr(self.config, "v6_canon_apply_in_train", True)):
            return trans_in

        eps = float(getattr(self.config, "v6_canon_eps", 1e-5))
        clip = float(getattr(self.config, "v6_canon_clip", 8.0))
        # 每个样本在 (S,3) 上做 z-score（只对有效转移项）
        return self._masked_zscore(trans_in, trans_valid_mask, dims=(1, 2), eps=eps, clip=clip)

    # --------------------------------------------
    # V6 version of V5 score builder (same signature)
    # --------------------------------------------
    def _compute_v5_unary_and_transition(
        self,
        audio_feat: torch.Tensor,   # [B,T,C]
        audio_mask: torch.Tensor,   # [B,T]
        wbd_feat: torch.Tensor,     # [B,W,C]
        wbd_mask: torch.Tensor,     # [B,W]
    ):
        """
        Returns:
            unary_logits: [B,T,S]   (V6 canonicalized + dual-gamma scaled)
            trans_in:     [B,S,3]   (V6 canonicalized + dual-gamma scaled)
            state_mask:   [B,S]
            blank_mask:   [B,W+1]
            proto:        [B,S,C]
        """
        B, T, C = audio_feat.shape
        Wmax = wbd_feat.shape[1]
        Smax = 2 * Wmax + 1
        device = audio_feat.device

        proto, state_mask, blank_mask, word_lens = self._build_state_proto_and_mask(wbd_feat, wbd_mask)

        # ===== raw unary (no gamma yet) =====
        use_fp32 = (not torch.is_grad_enabled())

        if use_fp32:
            if audio_feat.is_cuda:
                with torch.autocast(device_type="cuda", enabled=False):
                    audio_f = audio_feat.float()
                    proto_f = proto.float()

                    if self.config.sim_type == "cos":
                        a = F.normalize(audio_f, dim=-1)
                        p = F.normalize(proto_f, dim=-1)
                    else:
                        a, p = audio_f, proto_f

                    unary_raw = torch.bmm(a, p.transpose(1, 2))  # [B,T,S]
            else:
                audio_f = audio_feat.float()
                proto_f = proto.float()

                if self.config.sim_type == "cos":
                    a = F.normalize(audio_f, dim=-1)
                    p = F.normalize(proto_f, dim=-1)
                else:
                    a, p = audio_f, proto_f

                unary_raw = torch.bmm(a, p.transpose(1, 2))
        else:
            if self.config.sim_type == "cos":
                a = F.normalize(audio_feat, dim=-1)
                p = F.normalize(proto, dim=-1)
            else:
                a, p = audio_feat, proto
            unary_raw = torch.bmm(a, p.transpose(1, 2))  # [B,T,S]

        # ===== raw transition (no gamma yet) =====
        tscale = float(getattr(self.config, "v5_transition_scale", 1.0))  # 先保留 V5 配置项语义

        stay_score = self.v5_trans_stay_head(proto).squeeze(-1) + self.v5_trans_bias[0]  # [B,S]
        adv1_score = self.v5_trans_adv1_head(proto).squeeze(-1) + self.v5_trans_bias[1]  # [B,S]

        finfo_u = torch.finfo(unary_raw.dtype)
        finfo_t = torch.finfo(proto.dtype)
        neg_inf_u = -1e9 if finfo_u.min <= -1e9 else float(finfo_u.min)
        neg_inf_t = -1e9 if finfo_t.min <= -1e9 else float(finfo_t.min)

        trans_raw = proto.new_full((B, Smax, 3), fill_value=neg_inf_t)  # [B,S,3]
        trans_raw[:, :, 0] = stay_score
        trans_raw[:, :, 1] = adv1_score

        adv2_score = proto.new_full((B, Smax), fill_value=neg_inf_t)
        if Wmax >= 2:
            adv2_score[:, 3:2 * Wmax:2] = self.v5_trans_bias[2]
            if self.v5_enable_skip_pair_head and self.v5_trans_adv2_pair_head is not None:
                left = wbd_feat[:, :-1, :]
                right = wbd_feat[:, 1:, :]
                pair = torch.cat([left, right], dim=-1)
                pair_score = self.v5_trans_adv2_pair_head(pair).squeeze(-1)
                adv2_score[:, 3:2 * Wmax:2] = adv2_score[:, 3:2 * Wmax:2] + pair_score

        trans_raw[:, :, 2] = adv2_score

        # 结构合法性约束（按目标状态 s）
        s_ids = torch.arange(Smax, device=device, dtype=torch.long)[None, :]
        is_word_state = (s_ids % 2 == 1)
        can_adv1 = (s_ids >= 1)
        can_adv2 = (s_ids >= 3) & is_word_state

        trans_raw[:, :, 1] = trans_raw[:, :, 1].masked_fill(~can_adv1, neg_inf_t)
        trans_raw[:, :, 2] = trans_raw[:, :, 2].masked_fill(~can_adv2, neg_inf_t)

        # 无效 state 不作为目标状态
        trans_raw = trans_raw.masked_fill(~state_mask[:, :, None], neg_inf_t)

        # ===== canonicalization BEFORE gamma =====
        unary_c = self._canon_unary(unary_raw, state_mask, audio_mask)

        trans_valid_mask = (trans_raw > neg_inf_t / 2)
        trans_c = self._canon_trans(trans_raw, trans_valid_mask)

        # ===== dual gamma =====
        gamma_unary = torch.exp(self.log_gamma).clamp(0.7, self.gamma_max).to(unary_c.dtype)
        gamma_trans = torch.exp(self.log_gamma_trans).clamp(0.1, self.gamma_trans_max).to(trans_c.dtype)

        unary_logits = unary_c * gamma_unary
        trans_in = trans_c * gamma_trans

        # 保留 V5 的 transition_scale 语义（放在 canonicalization+gamma 后）
        if tscale != 1.0:
            finite_mask = (trans_in > neg_inf_t / 2)
            trans_in = torch.where(finite_mask, trans_in * tscale, trans_in)

        # 最终 mask invalid states / frames
        unary_logits = unary_logits.masked_fill(~state_mask[:, None, :], neg_inf_u)
        unary_logits = unary_logits.masked_fill(~audio_mask[:, :, None], neg_inf_u)
        trans_in = trans_in.masked_fill(~trans_valid_mask, neg_inf_t)

        return unary_logits, trans_in, state_mask, blank_mask, proto

    def _v6_forward_backward_posteriors(
        self,
        unary_logits: torch.Tensor,  # [B,T,S]
        trans_in: torch.Tensor,      # [B,S,3]
        audio_mask: torch.Tensor,    # [B,T]
        word_lens: torch.Tensor,     # [B]
    ):
        """
        返回:
        log_post: [B,T,S]  (invalid位置会是很小值)
        logZ: [B]
        """
        B, T, Smax = unary_logits.shape
        device = unary_logits.device
        neg_inf = unary_logits.new_tensor(-1e9)

        audio_lens = audio_mask.long().sum(dim=1)
        Tmax = int(audio_lens.max().detach().cpu().item()) if B > 0 else 0
        if Tmax <= 0:
            return unary_logits.new_full((B, 0, Smax), neg_inf), unary_logits.new_zeros((B,))

        unary = unary_logits[:, :Tmax, :]
        valid_time = (torch.arange(Tmax, device=device)[None, :] < audio_lens[:, None])
        s_ids = torch.arange(Smax, device=device, dtype=torch.long)[None, :]
        state_valid = (s_ids <= (2 * word_lens)[:, None])

        trans_stay, trans_adv1, trans_adv2 = trans_in.unbind(dim=-1)  # [B,S]
        neg_col1 = neg_inf.expand(B, 1)
        neg_col2 = neg_inf.expand(B, 2)

        # ---------- forward alpha ----------
        alpha_prev = unary.new_full((B, Smax), fill_value=neg_inf)
        alpha_prev[:, 0] = unary[:, 0, 0]
        if Smax >= 2:
            alpha_prev[:, 1] = unary[:, 0, 1]
        alpha_prev = alpha_prev.masked_fill(~state_valid, neg_inf)

        alpha_hist = unary.new_full((B, Tmax, Smax), fill_value=neg_inf)
        alpha_hist[:, 0, :] = alpha_prev

        for t in range(1, Tmax):
            emit_t = unary[:, t, :]

            stay = alpha_prev + trans_stay
            adv1 = torch.cat([neg_col1, alpha_prev[:, :-1]], dim=1) + trans_adv1
            adv2 = torch.cat([neg_col2, alpha_prev[:, :-2]], dim=1) + trans_adv2 if Smax >= 3 else neg_inf.expand(B, Smax)

            alpha_t = torch.logaddexp(torch.logaddexp(stay, adv1), adv2) + emit_t

            vt = valid_time[:, t][:, None]
            alpha_t = torch.where(vt, alpha_t, alpha_prev)
            alpha_t = alpha_t.masked_fill(~state_valid, neg_inf)

            alpha_prev = alpha_t
            alpha_hist[:, t, :] = alpha_t

        t_end = (audio_lens - 1).clamp_min(0)
        alpha_end = alpha_hist.gather(1, t_end[:, None, None].expand(-1, 1, Smax)).squeeze(1)  # [B,S]

        end_blank_s = (2 * word_lens).clamp_min(0).clamp_max(Smax - 1)
        end_word_s = (end_blank_s - 1).clamp_min(0).clamp_max(Smax - 1)

        score_blank = alpha_end.gather(1, end_blank_s[:, None]).squeeze(1)
        score_word = alpha_end.gather(1, end_word_s[:, None]).squeeze(1)
        logZ = torch.logaddexp(score_blank, score_word)
        logZ = torch.where(word_lens == 0, score_blank, logZ)

        # ---------- backward beta ----------
        beta_next = unary.new_full((B, Smax), fill_value=neg_inf)
        # 在结束状态集合上初始化 beta=0（对应 alpha_end + beta_end -> logZ）
        beta_next.scatter_(1, end_blank_s[:, None], torch.zeros((B, 1), device=device, dtype=beta_next.dtype))
        has_word = (word_lens > 0)
        if has_word.any():
            # 不能直接 scatter 会覆盖 blank，所以用 where 累加式 logaddexp 更稳，但这里结束状态最多两个，简单处理
            idx = end_word_s[:, None]
            val = torch.zeros((B, 1), device=device, dtype=beta_next.dtype)
            cur = beta_next.gather(1, idx)
            beta_next.scatter_(1, idx, torch.logaddexp(cur, val))

        beta_hist = unary.new_full((B, Tmax, Smax), fill_value=neg_inf)
        beta_hist[:, Tmax - 1, :] = beta_next

        # 预先构造 next-state 索引（从当前状态出发可去到哪）
        # forward里是 incoming到curr；backward里等价看 outgoing from curr:
        # stay: s -> s
        # adv1: s -> s+1
        # adv2: s -> s+2 (only if next is word state)
        for t in range(Tmax - 2, -1, -1):
            # beta_t(s) = logsum_{s'} [ trans(s->s') + unary_{t+1}(s') + beta_{t+1}(s') ]
            emit_next = unary[:, t + 1, :]
            nxt = beta_next

            # stay to s
            cand_stay = trans_stay + emit_next + nxt  # [B,S], interpreted as target=s (same index)

            # adv1: curr=s -> next=s+1
            # 对 curr 索引对齐：需要 target 的 adv1 incoming score 在 next state 上
            adv1_emit_nxt = trans_adv1[:, 1:] + emit_next[:, 1:] + nxt[:, 1:]  # [B,S-1] for curr=0..S-2
            cand_adv1 = torch.cat([adv1_emit_nxt, neg_inf.expand(B, 1)], dim=1)

            # adv2: curr=s -> next=s+2 (next must be word state; trans_adv2 stored on next)
            if Smax >= 3:
                adv2_emit_nxt = trans_adv2[:, 2:] + emit_next[:, 2:] + nxt[:, 2:]  # [B,S-2]
                cand_adv2 = torch.cat([adv2_emit_nxt, neg_inf.expand(B, 2)], dim=1)
            else:
                cand_adv2 = neg_inf.expand(B, Smax)

            beta_t = torch.logaddexp(torch.logaddexp(cand_stay, cand_adv1), cand_adv2)

            vt = valid_time[:, t][:, None]
            beta_t = torch.where(vt, beta_t, beta_next)
            beta_t = beta_t.masked_fill(~state_valid, neg_inf)

            beta_next = beta_t
            beta_hist[:, t, :] = beta_t

        log_post = alpha_hist + beta_hist - logZ[:, None, None]
        log_post = log_post.masked_fill(~valid_time[:, :, None], neg_inf)
        log_post = log_post.masked_fill(~state_valid[:, None, :], neg_inf)

        return log_post, logZ

    # ---------------------------
    # Forward: keep V5 style, add gamma_trans
    # ---------------------------
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out["gamma_unary"] = torch.exp(self.log_gamma).clamp(0.7, self.gamma_max).detach()
        out["gamma_trans"] = torch.exp(self.log_gamma_trans).clamp(0.1, self.gamma_trans_max).detach()
        # backward compat: gamma 继续表示 unary gamma
        out["gamma"] = out["gamma_unary"]
        return out

    @torch.inference_mode()
    def inference(
        self,
        wavs: torch.Tensor,
        texts: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        text_lens: Optional[torch.Tensor] = None,
        emit_temp: float = 1.0,
        trans_temp: float = 1.0,
        *,
        decode_mode: Optional[str] = None,   # "viterbi" | "posterior"
        backtrace_on_cpu: bool = True,
        return_debug: bool = False,
        profile: bool = False,
    ):
        decode_mode = decode_mode or str(getattr(self.config, "v6_default_decode_mode", "viterbi"))
        assert decode_mode in ["viterbi", "posterior"]

        # ==== 与 V5 一样：encode + score ====
        audio_feat, audio_mask, text_feat, text_mask = self.encode(wavs, texts, wav_lens, text_lens)
        wbd_feat, wbd_mask = self._gather_wbd_embeddings(text_feat, texts, text_mask)

        with tf32_enable(enable=False):
            unary_logits, trans_in, state_mask, _, _ = self._compute_v5_unary_and_transition(
                audio_feat, audio_mask, wbd_feat, wbd_mask
            )

        unary_logits = unary_logits.float()
        trans_in = trans_in.float()

        B, T, Smax_full = unary_logits.shape
        device = unary_logits.device
        frame_sec = float(self._frame_sec())

        audio_lens = audio_mask.long().sum(dim=1)
        word_lens_raw = wbd_mask.long().sum(dim=1)
        word_lens = torch.minimum(word_lens_raw, audio_lens)

        Tmax = int(audio_lens.max().detach().cpu().item()) if B > 0 else 0
        Wmax = int(word_lens.max().detach().cpu().item()) if B > 0 else 0

        results = []
        if Tmax <= 0:
            for _ in range(B):
                out = {
                    "word_start_times": torch.zeros((0,), device=wavs.device),
                    "word_end_times": torch.zeros((0,), device=wavs.device),
                    "word_conf": torch.zeros((0,), device=wavs.device),
                }
                if return_debug:
                    out["debug"] = {
                        "frame_sec": frame_sec,
                        "audio_len_frames": 0,
                        "word_len": 0,
                        "dp": torch.zeros((0, 1), dtype=torch.float16, device="cpu"),
                        "states": torch.zeros((0,), dtype=torch.long, device="cpu"),
                        "probs_words": torch.zeros((0, 0), dtype=torch.float16, device="cpu"),
                        "probs_blank": torch.zeros((0,), dtype=torch.float16, device="cpu"),
                        "decode_mode": decode_mode,
                    }
                results.append(out)
            return results

        Suse = 2 * Wmax + 1
        unary = unary_logits[:, :Tmax, :Suse].float() / max(emit_temp, 1e-6)
        trans = trans_in[:, :Suse, :].float() / max(trans_temp, 1e-6)

        # emission-only prob（debug/conf风格沿用V5）
        probs_state_emit = torch.softmax(unary, dim=-1)  # [B,T,S]

        # ==== decode ====
        if decode_mode == "viterbi":
            # 直接复用你抽出来的 Viterbi helper（建议从 V5 inference 中提）
            states, dec_aux = self._v5_decode_states_viterbi(
                unary=unary,
                trans_in=trans,
                audio_lens=audio_lens,
                word_lens=word_lens,
                return_debug=return_debug,
                backtrace_on_cpu=backtrace_on_cpu,
            )
            log_post = None

        else:  # posterior
            # 1) forward-backward 得到 state posterior
            log_post, logZ = self._v6_forward_backward_posteriors(
                unary_logits=unary,
                trans_in=trans,
                audio_mask=audio_mask[:, :Tmax],
                word_lens=word_lens,
            )  # [B,T,S], [B]

            # 2) topology-constrained posterior decoding:
            #    maximize sum_t log p(s_t|x), 使用同样拓扑，但不再叠加 transition（避免重复计分）
            neg_inf = unary.new_tensor(-1e9)
            trans_zero = trans.new_full(trans.shape, fill_value=neg_inf)

            # stay / adv1 / adv2 设为0（仅拓扑约束）
            # valid transition位置按 trans 原本是否有限来继承合法性
            valid_tr = (trans > neg_inf / 2)
            trans_zero = torch.where(valid_tr, torch.zeros_like(trans_zero), trans_zero)

            states, dec_aux = self._v5_decode_states_viterbi(
                unary=log_post,   # posterior score
                trans_in=trans_zero,     # 只保留拓扑约束
                audio_lens=audio_lens,
                word_lens=word_lens,
                return_debug=return_debug,
                backtrace_on_cpu=backtrace_on_cpu,
            )

        # ==== 以下部分基本复用 V5：从 states 提 span + conf + debug ====
        valid_time = (torch.arange(Tmax, device=device)[None, :] < audio_lens[:, None])

        w_ids = torch.arange(Wmax, device=device, dtype=torch.long)
        word_states = 2 * w_ids + 1
        valid_word = (w_ids[None, :] < word_lens[:, None])

        t_ids = torch.arange(Tmax, device=device, dtype=torch.long)[None, :, None]
        mask_word_frame = (
            valid_time[:, :, None]
            & valid_word[:, None, :]
            & (states[:, :, None] == word_states[None, None, :])
        )

        start_idx = torch.where(mask_word_frame, t_ids, torch.full_like(t_ids, Tmax)).amin(dim=1)
        end_idx = torch.where(mask_word_frame, t_ids, torch.full_like(t_ids, -1)).amax(dim=1) + 1
        has_span = (end_idx > 0) & valid_word

        start_idx = torch.where(has_span, start_idx, torch.zeros_like(start_idx))
        end_idx = torch.where(has_span, end_idx, torch.zeros_like(end_idx))

        word_start_times_all = start_idx.float() * frame_sec
        word_end_times_all = end_idx.float() * frame_sec

        # conf（沿用V5风格：用 emission-only probs 的 word state mean）
        probs_words = probs_state_emit[:, :, 1::2]  # [B,T,W]
        conf_frame = probs_words * mask_word_frame.float()
        sum_conf = conf_frame.sum(dim=1)
        len_seg = mask_word_frame.float().sum(dim=1).clamp_min(1.0)
        word_conf_all = sum_conf / len_seg
        word_conf_all = torch.where(has_span, word_conf_all, torch.zeros_like(word_conf_all))

        probs_blank_global = probs_state_emit[:, :, 0::2].sum(dim=-1)

        # 可选：posterior entropy（更好的稳定性可视化）
        posterior_entropy = None
        if log_post is not None:
            post = torch.exp(log_post.clamp_max(0.0))
            posterior_entropy = -(post.clamp_min(1e-12).log() * post).sum(dim=-1)  # [B,T]

        for b in range(B):
            Wb = int(word_lens[b].item())
            Tb = int(audio_lens[b].item())

            out = {
                "word_start_times": word_start_times_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
                "word_end_times": word_end_times_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
                "word_conf": word_conf_all[b, :Wb].to(device=wavs.device, dtype=torch.float32),
            }

            if return_debug:
                dbg = {
                    "frame_sec": frame_sec,
                    "audio_len_frames": Tb,
                    "word_len": Wb,
                    "states": states[b, :Tb].detach().to("cpu"),
                    "probs_words": probs_words[b, :Tb, :Wb].detach().to("cpu").to(torch.float16) if (Tb > 0 and Wb > 0) else torch.zeros((Tb, 0), dtype=torch.float16),
                    "probs_blank": probs_blank_global[b, :Tb].detach().to("cpu").to(torch.float16) if Tb > 0 else torch.zeros((0,), dtype=torch.float16),
                    "decode_mode": decode_mode,
                }

                # 复用你的 dp debug
                if "dp" in dec_aux:
                    Sb = int(2 * Wb + 1) if Wb > 0 else 1
                    dbg["dp"] = dec_aux["dp"][b, :Tb, :Sb].detach().to("cpu") if Tb > 0 else torch.zeros((0, 1), dtype=torch.float16)
                else:
                    dbg["dp"] = torch.zeros((0, 1), dtype=torch.float16)

                if posterior_entropy is not None:
                    dbg["posterior_entropy"] = posterior_entropy[b, :Tb].detach().to("cpu").to(torch.float16)
                    dbg["posterior_word_mass"] = torch.exp(log_post[b, :Tb, 1::2]).sum(dim=-1).detach().to("cpu").to(torch.float16) if (Tb > 0 and Wb > 0) else torch.zeros((Tb,), dtype=torch.float16)

                out["debug"] = dbg

            results.append(out)

        return results

