import os
import re
import soundfile as sf
import random
from attrdictionary import AttrDict
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from utils.commons.base_task_old import BaseTask
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint, load_ckpt_moe
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.hparams import hparams
from utils.commons.os_utils import kill_void
from utils.commons.io import print_once
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK
from utils.commons.tensor_utils import move_to_cpu, convert_to_np
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.model_utils import print_arch, num_params, unwrap_model, freeze_by_module_name
from utils.nn.ema import EMAModel, restore_ema
from utils.nn.seq_utils import sequence_mask, add_prefix

from modules.tts.scriptspeech.build_model_utils import build_vae, build_qwen3, shard_model_in_node, DiTBuildModelMixin
from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from tasks.tts.prompt_task import PromptBaseTask

class PromptDiTDpoTask(DiTBuildModelMixin, PromptBaseTask):
    def build_model(self):
        self._build_model()

        self.vae.to(self.trainer.device)
        self.vae = torch.compile(self.vae, mode='max-autotune')

        self.dit.to(self.trainer.device)

        # reference：冻结一个拷贝
        self.ref = deepcopy(self.dit)
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False
        self.ref.to(self.trainer.device)

        # 只训练 policy，ref 作为 frozen 模型
        return {"trainable": [self.dit], "others": [self.ref]}

    def load_model(self):
        ckpt_path = hparams.get("load_ckpt", "")
        ckpt_audio = hparams.get("load_ckpt_audio", "")

        if ckpt_path == "":
            return

        load_ckpt(self.dit, ckpt_path, "dit", strict=False, mmap=True)

        # ref 初始化为和 policy 相同的权重
        self.ref.load_state_dict(self.dit.state_dict(), strict=False)

    def forward_one(self, model, sample, which="pos", x0=None, t=None):
        """
        model: self.dit 或 self.ref
        which: "pos" 或 "neg"，用 sample['lat_good'] / sample['lat_bad']
        x0, t: 共用的噪声和时间步（可选）
        返回：per-sample loss [B]
        """
        device = self.trainer.device

        # 这里仍然从 sample['lat_good'] / sample['lat_bad'] 取 latent
        lat = sample["lat_good"] if which == "pos" else sample["lat_bad"]  # [B, T, C]
        lat = lat.to(device)
        lat_lens = sample["lat_lens"].to(device)   # 假设 good/bad 长度一样

        text = sample["text"]
        # text tokenize，和 PromptDiTTask.run_model 里的逻辑保持一致
        if 'txt_tokens' not in sample:
            text_inputs = self.dit_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
            txt_tokens = text_inputs['input_ids']   # [B, T]
            txt_mask = text_inputs['attention_mask'].bool()
            txt_tokens[~txt_mask] = self.cfg_mask_text_token
            txt_lens = txt_mask.int().sum(1)
        else:
            txt_tokens = sample['txt_tokens']
            txt_lens = sample['txt_lengths']
            txt_mask = sequence_mask(txt_lens, maxlen=txt_tokens.shape[1])
            txt_tokens[~txt_mask] = self.cfg_mask_text_token

        # ===== ctx zero-shot: 使用 dataset 提供的 ctx_mask / ctx_lat =====
        # ctx_mask: [B, T_lat, 1] (latent级)
        ctx_mask = sample.get("ctx_mask", None)

        if ctx_mask is None:
            # 没有ctx：全0
            ctx_mask = torch.zeros(lat.shape[0], lat.shape[1], 1, device=device, dtype=torch.float32)
        else:
            # dataset 可能给的是 [T_lat,1] 或 torch.float32
            if isinstance(ctx_mask, torch.Tensor):
                ctx_mask = ctx_mask.to(device=device, dtype=torch.float32)
            else:
                ctx_mask = torch.as_tensor(ctx_mask, device=device, dtype=torch.float32)

            # 兼容 [T,1] -> [B,T,1]
            if ctx_mask.dim() == 2:
                ctx_mask = ctx_mask.unsqueeze(0).expand(lat.shape[0], -1, -1).contiguous()

            # 兼容长度不一致：裁/补到 lat 的 T
            T_lat = lat.shape[1]
            if ctx_mask.shape[1] > T_lat:
                ctx_mask = ctx_mask[:, :T_lat, :]
            elif ctx_mask.shape[1] < T_lat:
                pad = T_lat - ctx_mask.shape[1]
                ctx_mask = F.pad(ctx_mask, (0, 0, 0, pad), value=0.0)

        # lat_ctx: 只在ctx区域填入 latent，其余为0
        lat_ctx = lat * ctx_mask


        # 构造 inputs
        inputs = {
            "txt_tokens": txt_tokens,
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,
            "ctx_mask": ctx_mask,
            "caption_emb": None,
            "caption_lens": None,
            "caption_text_mark": None,
            "vad_mask": None,
            "spk_mask": sample.get("spk_mask", None),
        }

        # loss_mask：和 PromptDiTTask.run_model 一样的逻辑
        loss_mask = sequence_mask(lat_lens, maxlen=lat.shape[1])[:, :, None] * (1 - ctx_mask)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            # 这里用你改过的 Diffusion.forward，支持传入 x0 / t
            if hparams.get('use_moe_ffn', False):
                pred, target, _ = model(inputs, x0=x0, t=t)
            else:
                pred, target = model(inputs, x0=x0, t=t)

        loss = F.mse_loss(pred.float(), target.float(), reduction="none")   # [B, T, C]
        loss = (loss * loss_mask).sum(dim=[1, 2]) / (loss_mask.sum(dim=[1, 2]) + 1e-8)  # [B]
        return loss

    def _training_step(self, sample, batch_idx, optimizer_idx):
        device = self.trainer.device

        # ========= 从 good / bad wav 编码 latent =========
        wavs_good = sample["wavs_good"].float().to(device)  # [B, Twav]
        wavs_bad = sample["wavs_bad"].float().to(device)
        wav_lengths = sample["wav_lengths"].to(device)      # [B]

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                lat_good = self.vae.encode_latent(wavs_good)  # [B, T, C]
                lat_bad = self.vae.encode_latent(wavs_bad)    # [B, T, C]

        # 和 PromptDiTTask.run_model 一致的长度计算，但做 clamp，避免 off-by-one
        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']  # [B]
        lat_lens = torch.clamp(lat_lens, min=1, max=lat_good.shape[1])


        # 构建一个局部 sample，把 latent 塞进去，后面 forward_one 沿用原逻辑
        sample = dict(sample)  # 浅拷贝，避免直接改 dataloader 返回对象
        sample["lat_good"] = lat_good
        sample["lat_bad"] = lat_bad
        sample["lat_lens"] = lat_lens
        # ========= wav -> latent 到此结束 =========

        # 采样 x0, t，给 good/bad 共用一套
        B, T, C = lat_good.shape
        x0 = torch.randn_like(lat_good)      # [B, T, C]
        t = torch.rand(B, device=device)     # [B]

        # policy: good / bad
        loss_pos = self.forward_one(self.dit, sample, "pos", x0=x0, t=t)  # [B]
        loss_neg = self.forward_one(self.dit, sample, "neg", x0=x0, t=t)

        # reference: no_grad
        with torch.no_grad():
            loss_pos_ref = self.forward_one(self.ref, sample, "pos", x0=x0, t=t)
            loss_neg_ref = self.forward_one(self.ref, sample, "neg", x0=x0, t=t)

        beta_target = hparams.get("beta_dpo", 50.0)
        warmup = hparams.get("beta_warmup_steps", 2000)
        beta = beta_target * min(1.0, global_step / warmup)

        margin = (loss_pos - loss_pos_ref) - (loss_neg - loss_neg_ref)
        dpo_loss = -F.logsigmoid(-beta * margin)  # [B]
        total_loss = dpo_loss.mean()

        # 记录一些监控项
        logs = {
            "rl_loss/pos": loss_pos.mean().detach(),
            "rl_loss/neg": loss_neg.mean().detach(),
            "rl_loss/pos_ref": loss_pos_ref.mean().detach(),
            "rl_loss/neg_ref": loss_neg_ref.mean().detach(),
            "rl_loss/pos_margin": (loss_pos - loss_pos_ref).mean().detach(),
            "rl_loss/neg_margin": (loss_neg - loss_neg_ref).mean().detach(),
            "rl_loss/pos_neg_margin": margin.mean().detach(),
        }

        return total_loss, logs
