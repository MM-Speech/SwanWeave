import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from modules.commons.hf.transformer import TransformerDecoderModel, TransformerEncoderModel
from modules.commons.hf.transformer_config import TransformerConfig
from modules.commons.hf.transformer_dit import TransformerDiTModel, TimestepEmbedding
from modules.commons.hf.transformer_dit_config import TransformerDiTConfig
from modules.asr.nepa.nepa import build_nepa_model
from modules.flow_matching.mask_cfm import MaskFlowMatching

from utils.nn.seq_utils import sequence_mask
from utils.commons.tensor_utils import all_gather_varlen_tensor_stack, all_gather_varlen_tensor
from utils.commons.hparams import set_hparams
from utils.commons.ckpt_utils import load_ckpt


def build_text_tokenizer(hparams):
    from utils.text.cosyvoice2_tokenizer import get_tokenizer

    timestamp_start = float(hparams.get('timestamp_start', 0.0))
    timestamp_end = float(hparams.get('timestamp_end', 300.0))
    timestamp_step = float(hparams.get('timestamp_step', 0.08))

    n_steps = int(round((timestamp_end - timestamp_start) / timestamp_step))
    timestamps = [timestamp_start + i * timestamp_step for i in range(n_steps + 1)]
    timestamp_tokens = [f"<|TS{t:.2f}|>" for t in timestamps]

    tokenizer = get_tokenizer(special_symbols=tuple(timestamp_tokens))

    tokenizer.timestamp_start = timestamp_start
    tokenizer.timestamp_end = timestamp_end
    tokenizer.timestamp_step = timestamp_step
    tokenizer.timestamp_start_id = tokenizer.encode(f"<|TS{timestamp_start:.2f}|>")[0]
    tokenizer.timestamp_end_id = tokenizer.encode(f"<|TS{timestamp_end:.2f}|>")[0]

    return tokenizer

def build_aligner_model(hparams, attn_implementation='flash_attention_2', init_pretrained=True):
    tokenizer = build_text_tokenizer(hparams)

    config = ModelArgs(
        sample_rate=hparams['audio_sample_rate'],

        vocab_size=tokenizer.encoding.n_vocab,
        mask_id=tokenizer.encode('<MASK>')[0],

        timestamp_start=float(getattr(tokenizer, 'timestamp_start', hparams.get('timestamp_start', 0.0))),
        timestamp_end=float(getattr(tokenizer, 'timestamp_end', hparams.get('timestamp_end', 300.0))),
        timestamp_step=float(getattr(tokenizer, 'timestamp_step', hparams.get('timestamp_step', 0.08))),
        timestamp_start_id=int(getattr(tokenizer, 'timestamp_start_id', tokenizer.encode('<|TS0.00|>')[0])),
        timestamp_end_id=int(getattr(tokenizer, 'timestamp_end_id', tokenizer.encode('<|TS300.00|>')[0])),

        mono_loss_weight=float(hparams.get('mono_loss_weight', 0.0)),
        mono_min_gap=float(hparams.get('mono_min_gap', 0.0)),
        mono_pos_weight_masked=float(hparams.get('mono_pos_weight_masked', 1.0)),
        mono_pos_weight_unmasked=float(hparams.get('mono_pos_weight_unmasked', 0.25)),

        self_conditioning=hparams.get('self_conditioning', False),
        self_conditioning_prob=float(hparams.get('self_conditioning_prob', 0.5)),
        self_conditioning_mode=hparams.get('self_conditioning_mode', 'replace_masked'),

        nepa_ckpt=hparams.get('nepa_ckpt', 'checkpoints/260205_nepa'),
        init_pretrained=init_pretrained,
        freeze_nepa=hparams.get('freeze_nepa', False),

        audio_hidden_size=hparams.get('audio_hidden_size', 512),
        audio_num_hidden_layers=hparams.get('audio_num_hidden_layers', 2),
        audio_num_attention_heads=hparams.get('audio_num_attention_heads', 8),
        audio_num_key_value_heads=hparams.get('audio_num_key_value_heads', 8),

        hidden_size=hparams.get('hidden_size', 512),
        num_hidden_layers=hparams.get('num_hidden_layers', 16),
        num_attention_heads=hparams.get('num_attention_heads', 8),
        num_key_value_heads=hparams.get('num_key_value_heads', 8),

        attn_implementation=attn_implementation,
        gradient_checkpointing=hparams.get('gradient_checkpointing', False),
    )
    model = ForcedAligner(config)
    return model, tokenizer


@dataclass
class ModelArgs:
    # audio
    sample_rate: int = 16000
    hop_size: int = None  # will be filled from nepa.config.hop_size

    # text
    vocab_size: int = None
    mask_id: int = None

    # timestamp token space (for monotonic regularization / decoding)
    timestamp_start: float = 0.0
    timestamp_end: float = 300.0
    timestamp_step: float = 0.08
    timestamp_start_id: int = None
    timestamp_end_id: int = None

    # monotonic regularization (training)
    mono_loss_weight: float = 0.0
    mono_min_gap: float = 0.0
    mono_pos_weight_masked: float = 1.0
    mono_pos_weight_unmasked: float = 0.25

    # self-conditioning (training)
    self_conditioning: bool = False
    self_conditioning_prob: float = 0.5
    self_conditioning_mode: str = 'replace_masked'

    # pretrained audio encoder
    audio_encoder_type: str = 'nepa'
    nepa_ckpt: str = None
    init_pretrained: bool = True
    freeze_nepa: bool = True

    # audio encoder
    audio_hidden_size: int = 512
    audio_num_hidden_layers: int = 2
    audio_num_attention_heads: int = 8
    audio_num_key_value_heads: int = 8

    # transformer
    hidden_size: int = 512
    num_hidden_layers: int = 16
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    is_causal: bool = False

    attn_implementation: str = 'flash_attention_2'
    gradient_checkpointing: bool = False


class TimeStampTokenizer:
    def __init__(self, timestamp_start=0.0, timestamp_end=300.0, timestamp_step=0.08):
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.timestamp_step = timestamp_step

        self.timestamp_size = int((timestamp_end - timestamp_start) / timestamp_step) + 1

    def __len__(self):
        return self.timestamp_size

    def encode(self, timestamps: torch.Tensor):
        return ((timestamps - self.timestamp_start) / self.timestamp_step).long()

    def decode(self, timestamps: torch.Tensor):
        return timestamps.float() * self.timestamp_step + self.timestamp_start


class ForcedAlignerBackbone(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        if config.nepa_ckpt.endswith('.ckpt'):
            nepa_hp = set_hparams(os.path.join(Path(config.nepa_ckpt).parent, 'config.yaml'), global_hparams=False)
        else:
            nepa_hp = set_hparams(os.path.join(config.nepa_ckpt, 'config.yaml'), global_hparams=False)
        self.nepa = build_nepa_model(nepa_hp, attn_implementation=config.attn_implementation)
        if config.init_pretrained:
            load_ckpt(self.nepa, config.nepa_ckpt, strict=False)
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
        
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer = TransformerDiTModel(
            config=TransformerDiTConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size * 4,
                num_hidden_layers=config.num_hidden_layers,
                num_cross_attention_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                attn_implementation=config.attn_implementation,
                use_dynamic_cross_gate=True
            )
        )
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.time_embed = TimestepEmbedding(config.hidden_size)

        if config.gradient_checkpointing and self.training:
            self.nepa.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.audio_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward_encoder(self, wavs, wav_lens):
        if self.config.freeze_nepa:
            with torch.no_grad():
                feat, audio_mask = self.nepa(wavs, wav_lens)  # eval-mode => Tensor[B, T', C]
        else:
            feat, audio_mask = self.nepa(wavs, wav_lens)
        x = self.audio_proj(feat)
        x = self.audio_encoder(inputs_embeds=x, attention_mask=audio_mask).last_hidden_state
        x = self.audio_out(x)
        return x, audio_mask

    def _time_embed(self, t: torch.Tensor, *, batch: int) -> torch.Tensor:
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
            t_emb = self.time_embed(t)
        return t_emb

    def forward(
        self,
        input_ids,
        timestep,
        cond,
        attn_mask
    ):
        audio_feat, audio_feat_mask = cond['audio_feat'], cond['audio_feat_mask']
        timestep = self._time_embed(timestep, batch=audio_feat.shape[0])
        x = self.embed(input_ids)
        x = self.transformer.forward(
            inputs_embeds=x,
            time_step=timestep,
            attention_mask=attn_mask,
            encoder_hidden_states=audio_feat,
            encoder_attention_mask=audio_feat_mask
        ).last_hidden_state
        x = self.head(x)
        return x


class ForcedAligner(MaskFlowMatching):
    def __init__(self, config: ModelArgs):
        backbone = ForcedAlignerBackbone(config)
        super().__init__(
            vocab_size=config.vocab_size,
            mask_id=config.mask_id,
            backbone=backbone,
            schedule="cosine",            # 'linear' | 'cosine' | 'quadratic'
            t_eps=0.05,
        )
        self.config = config
        
    # def _compute_mono_loss_from_logits(
    #     self,
    #     *,
    #     logits: torch.Tensor,          # [B,T,V]
    #     maskable: torch.Tensor,        # bool[B,T]
    #     pos_mask: Optional[torch.Tensor],  # bool[B,T] (mask & lossable). None => all weight=1
    # ) -> torch.Tensor:
    #     device = logits.device
    #     B, T, _ = logits.shape

    #     ts_start_id = getattr(self.config, 'timestamp_start_id', None)
    #     ts_end_id = getattr(self.config, 'timestamp_end_id', None)
    #     if ts_start_id is None or ts_end_id is None:
    #         return logits.new_zeros(())

    #     ts_start_id = int(ts_start_id)
    #     ts_end_id = int(ts_end_id)
    #     if ts_end_id < ts_start_id:
    #         return logits.new_zeros(())

    #     ts_ids = torch.arange(ts_start_id, ts_end_id + 1, device=device, dtype=torch.float32)
    #     ts_start = float(getattr(self.config, 'timestamp_start', 0.0))
    #     ts_step = float(getattr(self.config, 'timestamp_step', 0.08))
    #     mono_min_gap = float(getattr(self.config, 'mono_min_gap', 0.0) or 0.0)

    #     w_masked = float(getattr(self.config, 'mono_pos_weight_masked', 1.0) or 1.0)
    #     w_unmasked = float(getattr(self.config, 'mono_pos_weight_unmasked', 1.0) or 1.0)

    #     losses: List[torch.Tensor] = []
    #     for b in range(B):
    #         pos_all = torch.nonzero(maskable[b], as_tuple=False).squeeze(-1)
    #         if pos_all.numel() < 2:
    #             continue
    #         if (pos_all.numel() % 2) == 1:
    #             pos_all = pos_all[:-1]
    #         if pos_all.numel() < 2:
    #             continue

    #         logits_b = logits[b, pos_all, ts_start_id:ts_end_id + 1].float()
    #         probs_b = torch.softmax(logits_b, dim=-1)

    #         exp_id = (probs_b * ts_ids).sum(dim=-1)
    #         exp_time = ts_start + (exp_id - float(ts_start_id)) * ts_step

    #         start_t = exp_time[0::2]
    #         end_t = exp_time[1::2]
    #         if start_t.numel() == 0:
    #             continue

    #         if pos_mask is None:
    #             w_pos = exp_time.new_full(exp_time.shape, 1.0)
    #         else:
    #             pm = pos_mask[b].index_select(0, pos_all).to(device=device)
    #             w_pos = torch.where(pm, exp_time.new_full(pm.shape, w_masked), exp_time.new_full(pm.shape, w_unmasked))

    #         w_start = w_pos[0::2]
    #         w_end = w_pos[1::2]

    #         w_pair = torch.minimum(w_start, w_end)
    #         denom_self = w_pair.sum().clamp_min(1e-6)
    #         l_self = (torch.relu(start_t - end_t) * w_pair).sum() / denom_self

    #         if start_t.numel() > 1:
    #             w_adj = torch.minimum(w_end[:-1], w_start[1:])
    #             denom_next = w_adj.sum().clamp_min(1e-6)
    #             l_next = (torch.relu(end_t[:-1] - start_t[1:] + mono_min_gap) * w_adj).sum() / denom_next
    #         else:
    #             l_next = exp_time.new_zeros(())

    #         losses.append(l_self + l_next)

    #     if len(losses) == 0:
    #         return logits.new_zeros(())

    #     return torch.stack(losses, dim=0).mean()

    def _compute_mono_loss_from_logits(
        self,
        *,
        logits: torch.Tensor,          # [B,T,V]
        maskable: torch.Tensor,        # bool[B,T]
        pos_mask: Optional[torch.Tensor],  # bool[B,T] (mask & lossable). None => all weight=1
    ) -> torch.Tensor:
        device = logits.device
        B, T, _ = logits.shape

        ts_start_id = getattr(self.config, 'timestamp_start_id', None)
        ts_end_id = getattr(self.config, 'timestamp_end_id', None)
        if ts_start_id is None or ts_end_id is None:
            return logits.new_zeros(())

        ts_start_id = int(ts_start_id)
        ts_end_id = int(ts_end_id)
        if ts_end_id < ts_start_id:
            return logits.new_zeros(())

        ts_start = float(getattr(self.config, 'timestamp_start', 0.0))
        ts_step = float(getattr(self.config, 'timestamp_step', 0.08))
        mono_min_gap = float(getattr(self.config, 'mono_min_gap', 0.0) or 0.0)

        w_masked = float(getattr(self.config, 'mono_pos_weight_masked', 1.0) or 1.0)
        w_unmasked = float(getattr(self.config, 'mono_pos_weight_unmasked', 1.0) or 1.0)

        counts = maskable.sum(dim=1).to(torch.long)  # [B]
        Mmax = int(counts.max().item()) if counts.numel() > 0 else 0
        if Mmax < 2:
            return logits.new_zeros(())

        # 取每个样本里 maskable=True 的位置索引（按序），并 padding 到 [B, Mmax]
        pos_idx = torch.arange(T, device=device, dtype=torch.long).view(1, T).expand(B, T)  # [B,T]
        pos_idx = pos_idx.masked_fill(~maskable, T)  # 非 maskable 填 T（哨兵）
        idx_sorted, _ = torch.sort(pos_idx, dim=1)  # [B,T] 升序：maskable 位置在前，T 在后
        idx_pos = idx_sorted[:, :Mmax]  # [B,Mmax]

        valid_pos = (torch.arange(Mmax, device=device).view(1, Mmax) < counts.view(B, 1))  # bool[B,Mmax]
        idx_pos_clamped = idx_pos.clamp(0, T - 1)  # gather 需要合法索引；无效位置之后会被 valid_pos 权重清零

        # 取 timestamp token 的 logits 子空间
        logits_ts = logits[..., ts_start_id:ts_end_id + 1]  # [B,T,Mts]
        Mts = int(logits_ts.shape[-1])
        if Mts <= 0:
            return logits.new_zeros(())

        # gather 出所有 maskable 位置的 timestamp logits: [B,Mmax,Mts]
        gather_idx = idx_pos_clamped.unsqueeze(-1).expand(B, Mmax, Mts)
        logits_sel = logits_ts.gather(dim=1, index=gather_idx).float()

        # 期望 token_id -> 期望时间（秒）
        ts_ids = torch.arange(ts_start_id, ts_end_id + 1, device=device, dtype=torch.float32)  # [Mts]
        probs = torch.softmax(logits_sel, dim=-1)  # [B,Mmax,Mts]
        exp_id = (probs * ts_ids.view(1, 1, -1)).sum(dim=-1)  # [B,Mmax]
        exp_time = ts_start + (exp_id - float(ts_start_id)) * ts_step  # [B,Mmax]

        # 位置权重：masked 更大、unmasked 更小；无效 padding 位置权重=0
        if pos_mask is None:
            w_pos = exp_time.new_ones((B, Mmax))
        else:
            pm_sel = pos_mask.gather(dim=1, index=idx_pos_clamped).bool()  # [B,Mmax]
            w_pos = torch.where(
                pm_sel,
                exp_time.new_full(pm_sel.shape, w_masked),
                exp_time.new_full(pm_sel.shape, w_unmasked),
            )

        w_pos = w_pos * valid_pos.to(dtype=w_pos.dtype)  # padding 位置清零

        # 成对约束：pos_all 按序应为 [s0,e0,s1,e1,...]
        start_t = exp_time[:, 0::2]  # [B,Ws]
        end_t = exp_time[:, 1::2]    # [B,We]
        w_start = w_pos[:, 0::2]
        w_end = w_pos[:, 1::2]

        Wpair = min(start_t.shape[1], end_t.shape[1])
        if Wpair <= 0:
            return logits.new_zeros(())

        start_t = start_t[:, :Wpair]
        end_t = end_t[:, :Wpair]
        w_start = w_start[:, :Wpair]
        w_end = w_end[:, :Wpair]

        w_pair = torch.minimum(w_start, w_end)  # [B,Wpair]
        denom_self = w_pair.sum(dim=1)  # [B]
        l_self = (torch.relu(start_t - end_t) * w_pair).sum(dim=1) / denom_self.clamp_min(1e-6)
        l_self = torch.where(denom_self > 0, l_self, l_self.new_zeros(denom_self.shape))

        if Wpair > 1:
            w_adj = torch.minimum(w_end[:, :-1], w_start[:, 1:])  # [B,Wpair-1]
            denom_next = w_adj.sum(dim=1)
            l_next = (torch.relu(end_t[:, :-1] - start_t[:, 1:] + mono_min_gap) * w_adj).sum(dim=1) / denom_next.clamp_min(1e-6)
            l_next = torch.where(denom_next > 0, l_next, l_next.new_zeros(denom_next.shape))
        else:
            l_next = l_self.new_zeros((B,))

        loss_per_sample = l_self + l_next
        valid_sample = denom_self > 0
        if valid_sample.any():
            return loss_per_sample[valid_sample].mean()

        return logits.new_zeros(())

    def compute_loss(
        self,
        x0: torch.Tensor,                     # [B, T]
        t: Optional[torch.Tensor] = None,     # [B]
        padding_mask: Optional[torch.Tensor] = None,  # [B,T] True at pad
        cond: Optional[object] = None,        # forwarded to transformer
        loss_mask: Optional[torch.Tensor] = None,     # bool[B,T]; True contributes to loss
        maskable_mask: Optional[torch.Tensor] = None, # bool[B,T]; True can be masked
        never_mask_ids: Optional[object] = None,      # ids never masked
        ignore_loss_ids: Optional[object] = None,     # ids ignored in loss
    ) -> Dict[str, torch.Tensor]:
        mono_w = float(getattr(self.config, 'mono_loss_weight', 0.0) or 0.0)
        need_extra = mono_w > 0.0

        fm = super().compute_loss(
            x0=x0,
            t=t,
            padding_mask=padding_mask,
            cond=cond,
            loss_mask=loss_mask,
            maskable_mask=maskable_mask,
            never_mask_ids=never_mask_ids,
            ignore_loss_ids=ignore_loss_ids,
            self_conditioning=self.config.self_conditioning,
            self_conditioning_prob=self.config.self_conditioning_prob,
            self_conditioning_mode=self.config.self_conditioning_mode,
            return_details=need_extra,
            return_logits=need_extra,
        )

        mono_loss = fm["loss"].new_zeros(())
        if need_extra:
            logits = fm.pop("logits", None)
            maskable = fm.get("maskable", None)
            pos_mask = fm.get("pos", None)

            if torch.is_tensor(logits) and torch.is_tensor(maskable):
                mono_loss = self._compute_mono_loss_from_logits(
                    logits=logits,
                    maskable=maskable,
                    pos_mask=pos_mask if torch.is_tensor(pos_mask) else None,
                )

            fm["mono_loss"] = mono_loss

            for k in ("x_t", "mask", "valid", "lossable", "maskable", "attn_mask", "pos", "ce_sum", "w", "loss_per_sample"):
                fm.pop(k, None)
        else:
            fm["mono_loss"] = mono_loss

        return fm

    def forward(
        self,
        wavs: torch.Tensor,
        texts: torch.Tensor,
        wav_lens: torch.Tensor,
        text_lens: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        audio_feat, audio_feat_mask = self.backbone.forward_encoder(wavs, wav_lens)
        cond = {'audio_feat': audio_feat, 'audio_feat_mask': audio_feat_mask}
        attn_mask = sequence_mask(text_lens, maxlen=texts.shape[1])

        fm_result = self.compute_loss(
            x0=texts,
            t=None,
            padding_mask=~attn_mask,
            cond=cond,
            loss_mask=loss_mask,
            maskable_mask=None
        )

        fm_result['ntokens'] = audio_feat_mask.sum() + attn_mask.sum()

        return fm_result

    def inference(
        self,
        wavs: torch.Tensor,
        texts: torch.Tensor,
        wav_lens: torch.Tensor,
        text_lens: torch.Tensor,
        timesteps: int = 10,
        temperature: float = 0.7,
        token_topk: int = 5,
        *,
        return_scores: bool = False,
        return_debug: bool = False,
    ):
        audio_feat, audio_feat_mask = self.backbone.forward_encoder(wavs, wav_lens)
        cond = {'audio_feat': audio_feat, 'audio_feat_mask': audio_feat_mask}
        attn_mask = sequence_mask(text_lens, maxlen=texts.shape[1])

        infer_out = self.infer(
            x_init=texts,
            steps=timesteps,
            temperature=temperature,
            topk_per_step=None,
            token_topk=token_topk,
            cond=cond,
            padding_mask=~attn_mask,
            schedule_mode='uniform_alpha',
            inference_mode='remask_refine',
            preserve_input_tokens=True,
            ensure_no_mask=True,
            return_scores=return_scores,
        )

        if isinstance(infer_out, dict):
            pred = infer_out.get('tokens', None)
            token_conf = infer_out.get('token_conf', None)
            token_margin = infer_out.get('token_margin', None)
            token_score = infer_out.get('token_score', None)
        else:
            pred = infer_out
            token_conf = None
            token_margin = None
            token_score = None

        output = {
            'pred': pred,
            'token_conf': token_conf,
            'token_margin': token_margin,
            'token_score': token_score,
        }
        if return_debug:
            output['debug'] = None

        return output
        


