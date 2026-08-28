import os
import re
import soundfile as sf
import random
from attrdictionary import AttrDict
from copy import deepcopy
import math
import json
import pickle
import sys
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

from modules.tts.swanaudio.build_model_utils import DiTBuildModelMixin
from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from tasks.tts.swan_task import SwanBaseTask


class SwanDiTTask(DiTBuildModelMixin, SwanBaseTask):

    def build_model(self):
        self._build_model(attn_implementation='flash_attention_2')
        self.vae.to(self.trainer.device, torch.bfloat16)
        
        if hparams.get('use_ema', False):
            print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
            self.ema_model = deepcopy(self.dit)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_model.to(self.trainer.device)
            
            return {'trainable': [self.dit, self.ema_model], 'others': []}

        return {'trainable': [self.dit], 'others': []}

    def _dump_first_batch_and_exit(self, sample, batch_idx):
        # 只让 rank0 做 dump，避免多进程并发写
        if getattr(self.trainer, "proc_rank", 0) != 0:
            os._exit(0)

        dump_dir = os.path.join(hparams["work_dir"], "debug", "first_batch")
        os.makedirs(dump_dir, exist_ok=True)

        # ---------- 0) 不保存 wav：先把 wav/ctx_wav 从 sample 里剥离，只记元信息 ----------
        audio_meta = {}
        sample_noaudio = {}

        for k, v in sample.items():
            if k in ("wavs", "ctx_wavs"):
                if isinstance(v, torch.Tensor):
                    audio_meta[k] = {
                        "removed": True,
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                        "device": str(v.device),
                    }
                else:
                    audio_meta[k] = {"removed": True, "type": str(type(v))}
                continue
            sample_noaudio[k] = v

        # 只把“非音频字段”搬到 CPU，避免把超大 wav 从 GPU->CPU 拷贝
        sample_cpu = move_to_cpu(sample_noaudio)

        # ---------- 1) 保存非音频 sample（pickle 最完整） ----------
        with open(os.path.join(dump_dir, "sample_noaudio.pkl"), "wb") as f:
            pickle.dump(sample_cpu, f)

        # ---------- 2) 保存可读 json（对大 tensor 不做 min/max/mean，避免扫全量数据） ----------
        def _to_jsonable(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()
                info = {"_type": "tensor", "shape": list(x.shape), "dtype": str(x.dtype)}
                # 只有小 tensor 才算统计，避免对大 tensor 计算 min/max/mean 太慢
                if x.numel() <= 200_000:
                    info.update({
                        "min": float(x.min().item()) if x.numel() > 0 else None,
                        "max": float(x.max().item()) if x.numel() > 0 else None,
                        "mean": float(x.float().mean().item()) if x.numel() > 0 else None,
                    })
                return info
            if isinstance(x, np.ndarray):
                return {"_type": "ndarray", "shape": list(x.shape), "dtype": str(x.dtype)}
            if isinstance(x, (str, int, float, bool)) or x is None:
                return x
            if isinstance(x, (list, tuple)):
                return [_to_jsonable(v) for v in x]
            if isinstance(x, dict):
                return {str(k): _to_jsonable(v) for k, v in x.items()}
            return {"_type": str(type(x))}

        try:
            with open(os.path.join(dump_dir, "sample_noaudio_readable.json"), "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(sample_cpu), f, ensure_ascii=False, indent=2)
        except Exception as e:
            with open(os.path.join(dump_dir, "json_error.txt"), "w", encoding="utf-8") as f:
                f.write(repr(e))

        # ---------- 3) 记录被移除的音频字段元信息 ----------
        with open(os.path.join(dump_dir, "audio_meta.json"), "w", encoding="utf-8") as f:
            json.dump(audio_meta, f, ensure_ascii=False, indent=2)

        # ---------- 4) 保存一些关键信息 ----------
        with open(os.path.join(dump_dir, "meta.txt"), "w", encoding="utf-8") as f:
            f.write(f"global_step={int(self.global_step)}\n")
            f.write(f"batch_idx={int(batch_idx)}\n")
            f.write(f"keys_saved={list(sample_cpu.keys())}\n")
            f.write(f"audio_meta_keys={list(audio_meta.keys())}\n")

        # 立即退出（最稳，不给 DDP 留清理机会，防 hang）
        os._exit(0)



    def load_model(self):
        ckpt_path = hparams.get("load_ckpt", "")
        ckpt_audio = hparams.get("load_ckpt_audio", "")

        if ckpt_path == "":
            return

        if hparams.get("use_moe_ffn", False):
            load_ckpt_moe(
                self.dit,
                ckpt_base_dir=ckpt_path,
                ckpt_audio_base_dir=ckpt_audio,
                model_name="dit",
                strict=False,
                mmap=True,
            )
            if hparams.get("use_ema", False):
                self.ema_model.load_state_dict(self.dit.state_dict(), strict=False)
        else:
            if ckpt_path != "":
                load_ckpt(self.dit, ckpt_path, "dit", strict=False, mmap=True)
                if hparams.get("use_ema", False):
                    load_ckpt(self.ema_model, ckpt_path, "dit", strict=False, mmap=True)

    def _training_step(self, sample, batch_idx, optimizer_idx):
        # dump 第一个 batch 然后退出
        if int(self.global_step) == 0 and int(batch_idx) == 0:
            self._dump_first_batch_and_exit(sample, batch_idx)

        if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
            kill_void()
        loss_output, model_out = self.run_model(sample)

        # === moe_aux_loss 的权重，线性从1到0 ===
        base_moe_w = hparams.get('moe_aux_loss_weight', 1.0)
        anneal_steps = hparams.get('moe_aux_anneal_steps', 200000)
        if anneal_steps > 0 and base_moe_w > 0:
            ratio = max(0.0, 1.0 - min(self.global_step, anneal_steps) / float(anneal_steps))
            moe_w = base_moe_w * ratio
        else:
            moe_w = base_moe_w

        loss_weights = {
            'diff_loss': 1.0,
            'moe_aux_loss': moe_w,
        }

        total_loss = sum([
            loss_weights.get(k, 1.0) * v
            for k, v in loss_output.items()
            if isinstance(v, torch.Tensor) and v.requires_grad
        ])

        if self.trainer.proc_rank == 0 and self.global_step < 10:
            save_dir = f'{hparams["work_dir"]}/sample_batches/step_{self.global_step}'
            os.makedirs(save_dir, exist_ok=True)
            sample = convert_to_np(move_to_cpu(sample))
            for i in range(sample['nsamples']):
                sf.write(f"{save_dir}/{i}.wav", sample['wavs'][i, :sample['wav_lengths'][i]], 24000, 'PCM_16')
                sf.write(f"{save_dir}/{i}_ctx.wav", sample['ctx_wavs'][i, :sample['wav_lengths'][i]], 24000, 'PCM_16')
                np.save(f"{save_dir}/{i}.npy", sample, allow_pickle=True)
            del sample['wavs']
            del sample['ctx_wavs']

        return total_loss, loss_output

    def run_goku_text_encoder(self, captions: list):
        inputs = self.goku_tokenizer(
            captions,
            padding=True,  # Dynamic / longest
            truncation=True,
            max_length=hparams['text_max_token_length'],
            return_tensors="pt",
        )

        input_ids, attention_masks = inputs.input_ids.cuda(), inputs.attention_mask.cuda()
        encoder_hidden_states = self.goku_text_encoder(
            input_ids,
            return_dict=False,
            attention_mask=attention_masks,
        )[0]  # [B, T, C]

        # 新的多类掩码（0/1/2）
        if hparams.get('use_caption_text_mark', False):
            caption_text_mark = build_audio_mask_from_ids(
                input_ids=input_ids,
                attention_mask=attention_masks,
                tokenizer=self.goku_tokenizer,
            )  
        else:
            caption_text_mark = None

        return encoder_hidden_states, caption_text_mark, attention_masks, input_ids

    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}

        # 推理时外面自己走 inference，这里直接返回空
        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        text = sample['text']

        captions = None
        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']
        device = wavs.device

        ctx_wavs = sample["ctx_wavs"]
        ctx_wav_lengths = sample["ctx_wav_lengths"]
        ctx_mask = sample["ctx_mask"]
        if len(ctx_mask.shape) == 2:
            ctx_mask = ctx_mask[:, :, None]

        use_ref = True
        if hparams.get('drop_ref_wav', 0.5) > random.random():
            use_ref = False
            ctx_mask = torch.zeros_like(ctx_mask)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                lat, _ = self.vae.encode_latent(wavs, wav_lengths, chunk_sec=20, max_batch_size=128)
                if use_ref:
                    lat_ctx, _ = self.vae.encode_latent(ctx_wavs, ctx_wav_lengths, chunk_sec=20, max_batch_size=128)
                    lat_ctx = torch.nn.functional.pad(lat_ctx, (0, 0, 0, lat.size(1) - lat_ctx.size(1)), mode='constant', value=0)
                else:
                    lat_ctx = torch.zeros_like(lat)

        if random.random() < 0.001:
            print('| text sample', text[0])

        # text tokenize
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

        loss_mask = sequence_mask(lat_lens)[:, :, None] * (1 - ctx_mask)

        # check text validation
        if txt_tokens.shape[1] > lat.shape[1]:
            print(f'|Warning: text lengths [{txt_tokens.shape[1]}] > wav latent [{lat.shape[1]}], clipping...')
            line = txt_tokens[torch.argmax(txt_mask.sum(1))]
            print(self.dit_text_tokenizer.decode(line.detach().cpu().numpy().tolist()))

            line_idxs = np.argsort(txt_lens.cpu().numpy()).tolist()
            for line_idx in reversed(line_idxs):
                if txt_lens[line_idx] > lat.shape[1]:
                    loss_mask[line_idx] = 0.0
                else:
                    break

            txt_tokens = txt_tokens[:, :lat.shape[1]]
            txt_mask = txt_mask[:, :lat.shape[1]]
            txt_lens = txt_mask.sum(1).int()

        # CFG Mask on latent context
        lat_cfg_mask = torch.rand_like(lat_ctx[:, :1, 0].float())
        lat_cfg_mask = (lat_cfg_mask < 0.15).long()
        lat_ctx = (lat_ctx * ctx_mask * (1 - lat_cfg_mask)[:, :, None])

        # CFG Mask on text tokens
        txt_cfg_mask = torch.rand_like(txt_tokens[:, :1].float())
        txt_cfg_mask = (txt_cfg_mask < 0.15).long()
        txt_tokens = txt_tokens * (1 - txt_cfg_mask) + self.cfg_mask_text_token * txt_cfg_mask

        if hparams.get('use_caption', False):
            if (hparams.get('use_global', False) and use_ref) or (sample['global'] == ''):
                captions = sample['local']       # task1: ctx + local
            else:
                captions = sample['caption']

        # ------------ encode caption ------------
        text_embs = None
        caption_text_mark = None
        caption_lens = None
        cap_input_ids = None

        if captions is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    text_embs, caption_text_mark, text_att_mask, cap_input_ids = self.run_goku_text_encoder(captions)
                    text_embs = text_embs * text_att_mask[..., None]
                    text_cfg_mask = torch.rand_like(text_embs[:, :1].float())
                    text_cfg_mask = (text_cfg_mask < 0.15).long()
                    text_embs = text_embs * (1 - text_cfg_mask)
                    caption_lens = text_att_mask.sum(-1)

        inputs = {
            "txt_tokens": txt_tokens.long(),
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            'spk_mask': sample['spk_mask'] if hparams['use_spk_mask'] and 'spk_mask' in sample else None,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,
            "ctx_mask": ctx_mask,
            "caption_emb": text_embs,
            "caption_lens": caption_lens,
            "caption_text_mark": caption_text_mark,
            "caption_ids": cap_input_ids,
        }

        if not infer:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                if hparams.get('use_moe_ffn', False):
                    model_out, target, moe_aux = self.dit(inputs)
                else:
                    model_out, target = self.dit(inputs)
                    moe_aux = None

            # 主 diffusion loss
            if loss_mask.sum() == 0:
                return {}, model_out

            loss = F.mse_loss(model_out.float(), target.float(), reduction='none')
            loss = (loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]
            losses_out['diff_loss'] = loss

            # MoE aux loss（未乘权重，权重在 _training_step 里给）
            if moe_aux is not None:
                losses_out['moe_aux_loss'] = moe_aux

            losses_out['bs'] = loss_mask.shape[0]
            losses_out['ntokens'] = sum(lat_lens)
            return losses_out, model_out
        else:
            return losses_out, model_out
