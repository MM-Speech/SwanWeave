import os
import random
import re
import math
import traceback
import json
import tempfile
import time
from datetime import datetime
import uuid
import contextlib
import sys

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributed.fsdp import FullyShardedDataParallel
import torch.distributed
import torch.distributed as dist
import torchaudio
import numpy as np
from copy import deepcopy
import soundfile as sf
from tqdm import tqdm

from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.hparams import hparams, set_hparams
from utils.commons.os_utils import kill_void, handle_exacption
from utils.commons.dataset_utils import data_loader, build_dataloader, collate_xd
from utils.commons.trainer import LOCAL_RANK
from utils.commons.io import print_once, json_dump, json_dumps
from utils.commons.tensor_utils import move_to_cpu, move_to_cuda, convert_to_np, tensors_to_scalars, \
    all_gather_varlen_tensor, tensor_mean_per_element, tensor_mean_per_seq, slice_batch_value, repeat_batch_value
from utils.commons.seq_utils import seq_match
from utils.text.text_encoder import TokenTextEncoder
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.seq_utils import sequence_mask, add_prefix, remove_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model, freeze_by_module_name

from modules.tts.wavvae.decoder.wavvae_v3 import vae_decode
from modules.tts.scriptspeech.build_model_utils import build_vae, build_qwen3, shard_model_in_node, \
    DiTBuildModelMixin, DiTBuildModelMixinV2, DiTBuildModelMixinV4, DiTBuildModelMixinV5, DiTBuildModelMixinV6
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask


DEBUG = True


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    elif decay_type == 3:
        flat = 0
        uprate = 0.1
        uphold = 0.999
    elif decay_type == 4:
        flat = 0
        uprate = 0.1
        uphold = 0.5
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def repeat_pad_wavs(wavs: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
    """
    wavs: [B, T], 任意 dtype（float）和 device
    wav_lens: [B], int/long，表示每条音频有效长度；要求 >0 且 <= T
    返回：将右侧 padding 替换为循环重复后的 [B, T]
    """
    assert wavs.dim() == 2, "wavs 应为 [B, T]"
    B, T = wavs.shape
    device = wavs.device
    lengths = wav_lens.to(device)
    if torch.any(lengths <= 0):
        raise ValueError("wav_lens 必须 > 0")
    # 构造每行的索引：0..T-1 对每个长度取模
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
    idx = (idx % lengths.unsqueeze(1)).long()  # [B, T], 每行 < L_i
    # 按行 gather；因为 idx < L_i，只会从有效区域取值，不会取到原始 padding
    out = torch.gather(wavs, dim=1, index=idx)
    return out


class PromptTTSPostTrainBaseTask(DiTBuildModelMixin, ScriptSpeechBaseTask):
    def __init__(self):
        super().__init__()

        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        if hparams['use_audio_dataset']:
            self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
            self.processer_fn = import_module_bystr(hparams['processer_fn'])
            self.build_fast_dataloader = import_module_bystr(hparams['build_fast_dataloader'])
            self.train_dataloader = TTSDatasetMixin.train_dataloader.__get__(self)
        else:
            self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
            self.test_dataloader = FastDatasetMixin.test_dataloader.__get__(self)
            self.val_dataloader = FastDatasetMixin.val_dataloader.__get__(self)
        self.hparams = hparams
        from attrdictionary import AttrDict
        self.config = AttrDict(hparams)
        
        self._debug_run_dirs = {}  # 缓存不同 tag 的 run_dir

        self._skip_zero_reward_enabled = hparams.get('skip_zero_reward_samples', True)
        self._zero_reward_eps = float(hparams.get('skip_zero_reward_eps', 1e-8))

        self._pt_backward_step = 0   # 每次 backward +1
        self._pt_opt_step = 0        # 每次 optimizer.step +1

        reward_eval_cfg = hparams.get('reward_eval', {}) or {}
        self._reward_eval_enabled = bool(reward_eval_cfg.get('enable', False))
        self._reward_eval_interval = max(1, int(reward_eval_cfg.get('interval', 5) or 5))
        self._reward_eval_num_batches = max(1, int(reward_eval_cfg.get('num_batches', 1) or 1))
        self._reward_eval_num_generation_per_prompt = max(
            1,
            int(
                reward_eval_cfg.get(
                    'num_generation_per_prompt',
                    hparams.get('sample', {}).get('num_generation_per_prompt', 1),
                ) or 1
            ),
        )
        self._reward_eval_rollout_micro_batch_size = int(
            reward_eval_cfg.get(
                'rollout_micro_batch_size',
                hparams.get('sample', {}).get('rollout_micro_batch_size', 0),
            ) or 0
        )
        self._reward_eval_seed = int(reward_eval_cfg.get('seed', 20260316))
        self._reward_eval_samples = []
        self._reward_eval_sample_keys = set()
        self._reward_eval_train_step_offset = (
            self._reward_eval_num_batches if self._reward_eval_enabled else 0
        )

    def _init_debug_run_dir(self, tag: str) -> str:
        """
        为某一类 debug 输出(例如 gemini3pro)初始化一个本次运行唯一的目录:
        <work_dir>/sample_batches/<tag>/<run_id>/
        多卡同步，且不会覆盖旧 run。
        """
        if tag in self._debug_run_dirs:
            return self._debug_run_dirs[tag]

        base_root = os.path.join(hparams["work_dir"], "sample_batches", tag)
        if self.trainer.proc_rank == 0:
            os.makedirs(base_root, exist_ok=True)

            # 使用 时间戳 + UUID 片段 保证唯一且可读
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_" + uuid.uuid4().hex[:8]
            run_dir = os.path.join(base_root, run_id)
            os.makedirs(run_dir, exist_ok=False)   # 不允许已有目录，避免误覆盖
        else:
            run_dir = None

        # 如果是分布式，广播给所有 rank
        if dist.is_available() and dist.is_initialized():
            obj_list = [run_dir]
            dist.broadcast_object_list(obj_list, src=0)
            run_dir = obj_list[0]
            # 确保目录在所有 rank 上都可见
            dist.barrier()

        self._debug_run_dirs[tag] = run_dir
        return run_dir

    def _should_run_reward_eval(self) -> bool:
        if not self._reward_eval_enabled:
            return False
        return int(self.posttrain_epoch) % self._reward_eval_interval == 0

    def _reward_eval_state_path(self) -> str:
        rank = int(getattr(self.trainer, 'proc_rank', 0))
        return os.path.join(hparams["work_dir"], f"reward_eval_state_rank{rank}.pt")

    def _save_reward_eval_state(self):
        if not self._reward_eval_enabled:
            return
        if len(self._reward_eval_samples) == 0:
            return

        state_path = self._reward_eval_state_path()
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        state_tmp = state_path + ".tmp"
        state = {
            "version": 1,
            "num_batches": int(self._reward_eval_num_batches),
            "train_step_offset": int(self._reward_eval_train_step_offset),
            "samples": self._reward_eval_samples,
            "sample_keys": list(self._reward_eval_sample_keys),
        }
        torch.save(state, state_tmp)
        os.replace(state_tmp, state_path)

    def _load_reward_eval_state(self):
        if not self._reward_eval_enabled:
            return

        state_path = self._reward_eval_state_path()
        if not os.path.isfile(state_path):
            if self.trainer.proc_rank == 0 and int(self.global_step) >= int(self._reward_eval_train_step_offset):
                print(f"| WARN: reward-eval state not found on resume: {state_path}")
            return

        try:
            state = torch.load(state_path, map_location='cpu')
        except Exception:
            if self.trainer.proc_rank == 0:
                traceback.print_exc()
                print(f"| WARN: failed to load reward-eval state: {state_path}")
            return

        saved_num_batches = int(state.get("num_batches", 0) or 0)
        saved_train_step_offset = int(state.get("train_step_offset", 0) or 0)
        if self.trainer.proc_rank == 0:
            if saved_num_batches not in [0, int(self._reward_eval_num_batches)]:
                print(
                    f"| WARN: reward_eval.num_batches changed after resume: "
                    f"saved={saved_num_batches}, current={int(self._reward_eval_num_batches)}"
                )
            if saved_train_step_offset not in [0, int(self._reward_eval_train_step_offset)]:
                print(
                    f"| WARN: reward-eval train_step_offset changed after resume: "
                    f"saved={saved_train_step_offset}, current={int(self._reward_eval_train_step_offset)}"
                )

        samples = state.get("samples", []) or []
        if self._reward_eval_num_batches > 0:
            samples = samples[:self._reward_eval_num_batches]

        sample_keys = state.get("sample_keys", []) or []
        sample_keys = set(sample_keys[:len(samples)])
        if len(sample_keys) != len(samples):
            sample_keys = set()
            for sample in samples:
                item_names = sample.get('item_name', [])
                if isinstance(item_names, torch.Tensor):
                    cache_key = tuple(item_names.detach().cpu().tolist())
                elif isinstance(item_names, (list, tuple)):
                    cache_key = tuple(item_names)
                else:
                    cache_key = (str(item_names),)
                sample_keys.add(cache_key)

        self._reward_eval_samples = samples
        self._reward_eval_sample_keys = sample_keys

        if self.trainer.proc_rank == 0:
            print(
                f"| Loaded reward-eval state: "
                f"{len(self._reward_eval_samples)}/{self._reward_eval_num_batches} cached batches"
            )

    def _logical_posttrain_epoch(self, step=None) -> int:
        step = int(self.global_step if step is None else step)
        offset = int(self._reward_eval_train_step_offset or 0)
        return max(step - offset, 0)

    def _cache_reward_eval_sample(self, sample):
        if sample is None:
            return False
        if len(self._reward_eval_samples) >= self._reward_eval_num_batches:
            return False

        item_names = sample.get('item_name', [])
        if isinstance(item_names, torch.Tensor):
            cache_key = tuple(item_names.detach().cpu().tolist())
        elif isinstance(item_names, (list, tuple)):
            cache_key = tuple(item_names)
        else:
            cache_key = (str(item_names),)

        if cache_key in self._reward_eval_sample_keys:
            return False

        sample_cpu = deepcopy(move_to_cpu(sample))
        self._reward_eval_samples.append(sample_cpu)
        self._reward_eval_sample_keys.add(cache_key)

        if self.trainer.proc_rank == 0:
            print(
                f"| Cached reward-eval batch "
                f"{len(self._reward_eval_samples)}/{self._reward_eval_num_batches} "
                f"from train-stream"
            )
        self._save_reward_eval_state()
        return True

    def _should_skip_training_for_reward_eval(self, sample) -> bool:
        if not self._reward_eval_enabled:
            return False

        if int(self.global_step) < int(self._reward_eval_train_step_offset):
            cached = self._cache_reward_eval_sample(sample)
            if self.trainer.proc_rank == 0:
                cache_cnt = len(self._reward_eval_samples)
                print(
                    f"| Reward-eval warmup step {int(self.global_step) + 1}/"
                    f"{int(self._reward_eval_train_step_offset)}: "
                    f"cached={cache_cnt}, training skipped"
                )
            return True

        return False

    def maybe_run_reward_eval(self):
        return

    def on_train_start(self):
        super().on_train_start()
        self._load_reward_eval_state()


    def build_model(self):

        trainable_modules = []
        other_modules = []

        self._build_model()
        self.dit.to(torch.bfloat16)
        trainable_modules.append(self.dit)

        self.vae.to(self.trainer.device, torch.bfloat16)
        
        if hparams.get('freeze_ling_encoder', False) is True:
            frozen_modules = freeze_by_module_name(
                unwrap_model(self.dit),
                freeze_modules=['text_embedder', 'text_encoder', 'spk_mask_embedder']
            )
            cr = '\n'
            print_once(f"| Freeze following modules:{cr}{cr.join([f'| - {frozen_module}' for frozen_module in frozen_modules])}")

        self.dit_old = deepcopy(self.dit)
        self.dit_old.eval()
        for param in self.dit_old.parameters():
            param.requires_grad = False
        self.dit_old.to(self.trainer.device, torch.bfloat16)
        other_modules.append(self.dit_old)

        self.dit_ref = deepcopy(self.dit)
        self.dit_ref.eval()
        for param in self.dit_ref.parameters():
            param.requires_grad = False
        self.dit_ref.to(self.trainer.device, torch.bfloat16)
        # trainable_modules.append(self.dit_ref)

        if hparams.get('use_ema', False):
            print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
            self.ema_model = deepcopy(self.dit)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_model.to(self.trainer.device)

            trainable_modules.append(self.ema_model)

        self.reward_model_pack = self.build_reward_model()
        self.resamplers = {}

        # load pretrained ref models
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.dit_ref, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
        else:
            print(f"| Warning: Post Training without Pretrained Models!!!")
            
        return {'trainable': trainable_modules, 'others': other_modules}

    def build_reward_model(self):
        raise NotImplementedError("Reward model is not implemented for post training.")

    def run_reward_model(self, wavs, wav_lens, **kwargs):
        raise NotImplementedError("Reward model is not implemented for post training.")

    @torch.no_grad()
    def ema_update(self, ema_model, model, decay):
        ema_params = dict(unwrap_model(ema_model).named_parameters())
        for n, p in unwrap_model(model).named_parameters():
            p_ema = ema_params[n]
            src = p.detach()
            if src.dtype != torch.float32:
                src = src.float()
            if p_ema.dtype != torch.float32:
                p_ema.data = p_ema.data.float()
            p_ema.mul_(decay).add_(src.to(p_ema.device), alpha=1.0 - decay)

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.dit, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
            load_ckpt(self.dit_old, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
            # load_ckpt(self.dit_ref, hparams['load_ckpt'], 'dit', strict=False, mmap=True) # ref must be loaded everytime, so I move it to build_model
            if hparams.get('use_ema', False):
                load_ckpt(self.ema_model, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
        else:
            print(f"| Warning: Post Training without Pretrained Models!!!")

    def build_optimizer(self):
        optimizer = AdamW(unwrap_model(self.dit).parameters(), **self.config.optimizer)
        return optimizer

    def build_scheduler(self, optimizer):
        return None

    def fsdp_optm2model(self):
        return [self.dit]

    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        import modules.flow_matching.llama
        import modules.asr.llama.llama
        import modules.asr.llama.llama_seq2seq
        import modules.tts.llama_dit.llama
        import modules.tts.llama_dit.llama_ca
        import modules.tts.llama_dit.llama_prompt

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                # Linear, Sequential, Conv1d, Conv2d, Embedding,
                modules.flow_matching.llama.TransformerBlock,
                modules.asr.llama.llama.TransformerBlock,
                modules.asr.llama.llama_seq2seq.DecoderBlock,
                modules.asr.llama.llama_seq2seq.EncoderBlock,
                modules.tts.llama_dit.llama.TransformerBlock,
                modules.tts.llama_dit.llama_ca.TransformerBlock,
                modules.tts.llama_dit.llama_prompt.TransformerBlock,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer"),
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    def prepare_cfg_input(self, inputs, device):
        inputs = deepcopy(inputs)

        inputs['txt_tokens'] = torch.cat([
            inputs['txt_tokens'], inputs['txt_tokens'],
            torch.full(inputs['txt_tokens'].size(), self.cfg_mask_text_token, device=device)
        ], dim=0)
        inputs['txt_mask'] = torch.cat([inputs['txt_mask']] * 3, dim=0)

        inputs['lat'] = torch.cat([
            inputs['lat'], torch.zeros_like(inputs['lat']), torch.zeros_like(inputs['lat'])
        ], dim=0)
        inputs['lat_ctx'] = torch.cat([
            inputs['lat_ctx'], torch.zeros_like(inputs['lat_ctx']), torch.zeros_like(inputs['lat_ctx'])
        ], dim=0)
        inputs['ctx_mask'] = torch.cat([inputs['ctx_mask']] * 3, dim=0)

        inputs['tgt_len'] = torch.cat([inputs['tgt_len']] * 3, dim=0)

        if inputs.get('spk_mask') is not None:
            inputs['spk_mask'] = torch.cat([inputs['spk_mask']] * 3, dim=0)

        return inputs
    
    def prepare_inputs(self, sample):
        item_names = sample['item_name']
        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        ctx_wavs = sample["ctx_wavs"]
        ctx_mask = sample["ctx_mask"]
        ctx_mask = ctx_mask.float()
        if len(ctx_mask.shape) == 2:
            ctx_mask = ctx_mask[:, :, None]
        text = sample['text']
        tgt_text = sample['tgt_text']
        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']
        ctx_lens = ctx_mask.sum(1).squeeze(-1).long()
        ctx_wav_lens = ctx_lens * hparams['hop_size'] * hparams['vae_stride']
        ref_wav_paths = sample['ref_wav_paths']
        bsz = wavs.size(0)

        device = wavs.device

        # audio encode
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            try:
                print(f"{wavs.shape = }, {ctx_wavs.shape = }")
                lat = self.vae.encode_latent(wavs)
                lat_ctx = self.vae.encode_latent(ctx_wavs)
                lat_ctx = torch.nn.functional.pad(lat_ctx, (0,0,0,lat.size(1)-lat_ctx.size(1)), mode='constant', value=0)
            except RuntimeError:
                traceback.print_exc()
                print(f"{wavs.shape = }, {ctx_wavs.shape = }")
                sys.exit(1)

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

        inputs = {
            'item_names': item_names,
            'wavs': wavs,
            'wav_lengths': wav_lengths,
            'lat': lat,
            'lat_ctx': lat_ctx,
            'tgt_len': lat_lens,
            'ctx_mask': ctx_mask,
            'ctx_lens': ctx_lens,
            'ctx_wav_lens': ctx_wav_lens,
            'txt_tokens': txt_tokens,
            'txt_mask': txt_mask,
            'spk_mask': sample['spk_mask'] if 'spk_mask' in sample else None,
            'text': text,
            'tgt_text': tgt_text,
            'ref_wav_paths': ref_wav_paths,
        }

        return inputs
    
    def sampling(self, inputs, sample, infer_batch_size: int = None):
        device = self.trainer.device

        bsz = int(inputs['wav_lengths'].shape[0])
        if infer_batch_size is None:
            infer_batch_size = int(hparams.get('sample', {}).get('rollout_micro_batch_size', 0) or 0)
        if infer_batch_size <= 0:
            infer_batch_size = bsz
        infer_batch_size = min(infer_batch_size, bsz)

        lat_pred_chunks = []
        rewards_chunks = []
        t_schedule = None

        for start in range(0, bsz, infer_batch_size):
            end = min(start + infer_batch_size, bsz)
            slc = slice(start, end)

            inputs_mb = slice_batch_value(inputs, slc)
            sample_mb = slice_batch_value(sample, slc)

            item_names = inputs_mb['item_names']
            ctx_mask = inputs_mb['ctx_mask']
            lat_ctx = inputs_mb['lat_ctx']
            wav_lengths = inputs_mb['wav_lengths']
            ctx_wav_lens = inputs_mb['ctx_wav_lens']

            infer_inputs = self.prepare_cfg_input(inputs_mb, device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True), torch.no_grad():
                lat_pred_mb, t_schedule_mb = self.dit_old.inference(
                    infer_inputs,
                    timesteps=hparams['sample']['num_steps'],
                    seq_cfg_w=hparams['sample']['guidance_scale'],
                    timestep_annealing_w=(0.6, 0.6, 1.0),
                    return_timesteps=True,
                )

                if t_schedule is None:
                    t_schedule = t_schedule_mb

                lat_pred_mb = lat_pred_mb * (1 - ctx_mask) + lat_ctx * ctx_mask
                try:
                    wav_pred = vae_decode(self.vae, lat_pred_mb)[:, 0]
                except:
                    if self.trainer.proc_rank_local == 0:
                        traceback.print_exc()
                        print(f"| ERROR: VAE decoding fail: lat_pred.shape={lat_pred_mb.shape}, skip")
                        print(f"{item_names = }")
                    return

                wav_pred_lens = wav_lengths - ctx_wav_lens
                wav_pred = remove_prefix(wav_pred[..., None], ctx_wav_lens, wav_pred_lens)[..., 0].float()

                with torch.cuda.amp.autocast(enabled=False):
                    rewards_mb = self.run_reward_model(wav_pred, wav_pred_lens, inputs=inputs_mb, sample=sample_mb)

            lat_pred_chunks.append(lat_pred_mb)
            rewards_chunks.append(rewards_mb)

        lat_pred = torch.cat(lat_pred_chunks, dim=0) if len(lat_pred_chunks) > 1 else lat_pred_chunks[0]
        if len(rewards_chunks) == 1:
            rewards = rewards_chunks[0]
        else:
            rewards = {k: torch.cat([rc[k] for rc in rewards_chunks], dim=0) for k in rewards_chunks[0]}

        out = dict(inputs)
        out.update({
            'timesteps': t_schedule,
            'lat_pred': lat_pred,
            'rewards': rewards
        })
        return out

    def training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
            kill_void()

        hop_size = hparams['hop_size']
        vae_stride = hparams['vae_stride']

        self.posttrain_epoch = self._logical_posttrain_epoch(self.global_step)
        if not hasattr(self, 'samples_data_list'):
            self.samples_data_list = []

        if self._should_skip_training_for_reward_eval(sample):
            return {'loss': None}

        world_size = self.trainer.num_total_gpus
        rank = self.trainer.proc_rank

        ############
        # SAMPLING #
        ############

        # all_item_names = [None for _ in range(world_size)]
        # torch.distributed.all_gather_object(all_item_names, sample['item_name'])
        # print(f"{all_item_names = }")

        inputs_base = self.prepare_inputs(sample)
        bsz = len(sample['item_name'])
        device = self.trainer.device

        num_generation_per_prompt = int(hparams.get('sample', {}).get('num_generation_per_prompt', 1) or 1)
        num_generation_per_prompt = max(num_generation_per_prompt, 1)

        rollout_micro_batch_size = int(hparams.get('sample', {}).get('rollout_micro_batch_size', 0) or 0)
        if rollout_micro_batch_size <= 0:
            rollout_micro_batch_size = bsz  # 保持旧行为：不合并 gen
        rollout_micro_batch_size = max(1, rollout_micro_batch_size)

        gen_per_call = max(1, rollout_micro_batch_size // max(1, bsz))

        if self._skip_zero_reward_enabled:
            if (
                (not hasattr(self, '_zero_reward_skip_stats'))
                or (self._zero_reward_skip_stats.get('epoch') != self.posttrain_epoch)
            ):
                self._zero_reward_skip_stats = {'epoch': self.posttrain_epoch, 'total': 0, 'skipped': 0}

        sampling_pbar = None
        if self.trainer.proc_rank_local == 0:
            sampling_pbar = tqdm(
                total=num_generation_per_prompt,
                desc=f"| PostTrain Epoch {self.posttrain_epoch}: sampling",
                position=0,
            )

        gen_start = 0
        while gen_start < num_generation_per_prompt:
            cur_rep = min(gen_per_call, num_generation_per_prompt - gen_start)

            inputs_rep = repeat_batch_value(inputs_base, cur_rep)
            sample_rep = repeat_batch_value(sample, cur_rep)

            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices, enabled=True):
                try:
                    rank = int(self.trainer.proc_rank)
                except Exception:
                    rank = 0
                seed = int(hparams.get('dataloader_seed', 1231)) + int(self.posttrain_epoch) * 1000 + rank * 100 + gen_start
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                inputs = self.sampling(inputs_rep, sample_rep, infer_batch_size=rollout_micro_batch_size)
                
            if inputs is None:
                gen_start += cur_rep
                if sampling_pbar is not None:
                    sampling_pbar.update(cur_rep)
                continue

            inputs = move_to_cpu(inputs)

            store_raw_wavs = bool(hparams.get('store_raw_wavs_in_rl_buffer', False))

            bsz_rep = int(inputs['wav_lengths'].shape[0])
            for b_i in range(bsz_rep):
                item = {
                    'item_names': inputs['item_names'][b_i],
                    'lat': inputs['lat'][b_i],
                    'lat_ctx': inputs['lat_ctx'][b_i],
                    'tgt_len': inputs['tgt_len'][b_i],
                    'ctx_mask': inputs['ctx_mask'][b_i],
                    'txt_tokens': inputs['txt_tokens'][b_i],
                    'txt_mask': inputs['txt_mask'][b_i],
                    'wavs': inputs['wavs'][b_i],
                    'wav_lengths': inputs['wav_lengths'][b_i],
                    'timesteps': inputs['timesteps'],
                    'lat_pred': inputs['lat_pred'][b_i],
                }
                if store_raw_wavs:
                    item['wavs'] = inputs['wavs'][b_i]
                    item['wav_lengths'] = inputs['wav_lengths'][b_i]
                if inputs.get('rewards'):
                    item['rewards'] = {k: inputs['rewards'][k][b_i] for k in inputs['rewards']}
                    if self._skip_zero_reward_enabled:
                        self._zero_reward_skip_stats['total'] += 1
                        if ('sum' in item['rewards']) and self._is_zero_reward(item['rewards']['sum']):
                            self._zero_reward_skip_stats['skipped'] += 1
                            continue
                if inputs.get('rewards_futures'):
                    item['rewards_futures'] = inputs['rewards_futures'][b_i]
                self.samples_data_list.append(item)

            gen_start += cur_rep
            if sampling_pbar is not None:
                sampling_pbar.update(cur_rep)

        ############
        # TRAINING #
        ############

        self.training_epoch()
            
        return {'loss': None}   # poison pill to skip trainer
    
    def model_forward_step(self, dit, xt, t, inputs, do_checkpoint=False):
        """Forward one DiT step used by NFT/RL training.

        Important: when using FSDP, calling sub-methods like `forward_text_encoder()` and `_forward()` directly
        can bypass FSDP's parameter unshard/reshard logic. Prefer routing through `dit(...)` (module forward)
        when possible.
        """
        seq_cfg_w = hparams['sample']['guidance_scale']
        timestep_annealing_w = (0.6, 0.6, 1.0)

        try:
            out = dit(
                inputs,
                sigmas=t,
                x_noisy=xt,
                seq_cfg_w=seq_cfg_w,
                timestep_annealing_w=timestep_annealing_w,
            )
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out
        except TypeError:
            tgt_len = inputs['tgt_len']     # reference + target
            x_mask = sequence_mask(tgt_len, maxlen=xt.shape[1])
            x_txt = dit.forward_text_encoder(inputs, x_mask)

            ctx_mask = inputs['ctx_mask']
            ctx_feature = inputs['lat_ctx'] * ctx_mask

            cond = {
                'ctx': ctx_feature,
                'ctx_mask': ctx_mask,
                'attn_mask': x_mask,
                'x_txt': x_txt,
            }

            pred = dit._forward(
                xt,
                cond,
                t,
                seq_cfg_w=seq_cfg_w,
                timestep_annealing_w=timestep_annealing_w,
            )
            return pred
    
    def compute_advantages(self, collated_samples):
        world_size = self.trainer.num_total_gpus
        rank = self.trainer.proc_rank
        
        rewards = collated_samples['rewards']['sum']
        item_names = collated_samples['item_names']

        if self._skip_zero_reward_enabled:
            keep_mask = torch.isfinite(rewards) & (rewards.abs() > self._zero_reward_eps)
            collated_samples, dropped = self._filter_collated_samples_by_keep_mask(collated_samples, keep_mask)
            if dropped > 0 and self.trainer.proc_rank == 0:
                print(f"| Skip zero-reward samples: dropped={dropped}/{int(keep_mask.numel())}, eps={self._zero_reward_eps}")
            rewards = collated_samples['rewards']['sum']
            item_names = collated_samples['item_names']

        if hparams['per_prompt_stat_tracking']:
            if world_size > 1:
                all_rewards = all_gather_varlen_tensor(rewards, dim=0)  # [B]
                all_item_names = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(all_item_names, item_names)
                all_item_names = [item_name for item_names in all_item_names for item_name in item_names]
                assert len(all_rewards) == len(all_item_names)
            else:
                all_rewards = rewards
                all_item_names = item_names
            item_names_unique = list(set(all_item_names))
            item_name2idxs = {item_name: [] for item_name in item_names_unique}
            for item_name_idx, item_name in enumerate(all_item_names):
                item_name2idxs[item_name].append(item_name_idx)
            item_name2avg = {}
            item_name2std = {}
            for item_name in item_names_unique:
                item_name_idxs = item_name2idxs[item_name]
                if len(item_name_idxs) == 0:
                    item_name2avg[item_name] = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
                    item_name2std[item_name] = torch.zeros_like(item_name2avg[item_name])
                else:
                    item_name2avg[item_name] = all_rewards[item_name_idxs].mean()
                    item_name2std[item_name] = (
                        all_rewards[item_name_idxs].std()
                        if len(item_name_idxs) > 1
                        else torch.zeros_like(item_name2avg[item_name])
                    )
            rewards_avg_list, rewards_std_list = [], []
            for item_name in item_names:
                rewards_avg_list.append(item_name2avg[item_name])
                rewards_std_list.append(item_name2std[item_name])
            if len(rewards_avg_list) == 0:
                rewards_avg = rewards.new_empty((0,))
                rewards_std = rewards.new_empty((0,))
            else:
                rewards_avg = torch.stack(rewards_avg_list)
                rewards_std = torch.stack(rewards_std_list)
        else:
            if world_size > 1:
                all_rewards = all_gather_varlen_tensor(rewards, dim=0)
            else:
                all_rewards = rewards
            rewards_avg = all_rewards.mean()
            rewards_std = all_rewards.std()

            if all_rewards.numel() == 0:
                rewards_avg = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
                rewards_std = torch.zeros_like(rewards_avg)
            else:
                rewards_avg = all_rewards.mean()
                rewards_std = all_rewards.std() if all_rewards.numel() > 1 else torch.zeros_like(rewards_avg)

        advantages = (rewards - rewards_avg) / (rewards_std + 1e-4)
        collated_samples['advantages'] = advantages
        
        if self.trainer.proc_rank == 0:
            if 'all_rewards' in locals() and all_rewards.numel() > 0:
                total_mean = all_rewards.mean().item()
                total_std = all_rewards.std().item() if all_rewards.numel() > 1 else 0.0
                print(f"| Total rewards: mean={total_mean}, std={total_std}")
            else:
                print("| Total rewards: empty after filtering")
        
        return collated_samples, all_rewards
    
    def training_epoch(self):
        world_size = self.trainer.num_total_gpus
        rank = self.trainer.proc_rank
        
        device = self.trainer.device
        
        hop_size = hparams['hop_size']
        vae_stride = hparams['vae_stride']
        # num_train_timesteps = int(hparams['sample']['num_steps'] * hparams['train']['timestep_fraction'])
        
        if (not hasattr(self, 'samples_data_list')) or (len(self.samples_data_list) == 0):
            if self.trainer.proc_rank == 0:
                print(f"| PostTrain Epoch {self.posttrain_epoch}: no samples collected, skip training_epoch")
            return

        if self._skip_zero_reward_enabled and hasattr(self, '_zero_reward_skip_stats'):
            if (
                self.trainer.proc_rank == 0
                and self._zero_reward_skip_stats.get('epoch') == self.posttrain_epoch
                and self._zero_reward_skip_stats.get('total', 0) > 0
            ):
                skipped = int(self._zero_reward_skip_stats.get('skipped', 0))
                total = int(self._zero_reward_skip_stats.get('total', 0))
                print(f"| Skip zero-reward (sampling stage): skipped={skipped}/{total} ({(skipped / max(total, 1)):.2%})")

        collated_samples = {}
        for k in self.samples_data_list[0].keys():
            if isinstance(self.samples_data_list[0][k], torch.Tensor):
                if self.samples_data_list[0][k].ndim == 0:
                    collated_samples[k] = torch.stack([s[k] for s in self.samples_data_list])
                else:
                    collated_samples[k] = collate_xd([s[k] for s in self.samples_data_list])
            elif k == 'rewards':
                collated_samples[k] = {rk: torch.stack([s[k][rk] for s in self.samples_data_list]) for rk in self.samples_data_list[0][k]}
            else:
                collated_samples[k] = [s[k] for s in self.samples_data_list]

        if not bool(hparams.get('store_raw_wavs_in_rl_buffer', False)):
            # Raw waveforms are not used in the NFT loss; keep them off CUDA to save VRAM.
            collated_samples.pop('wavs', None)
            collated_samples.pop('wav_lengths', None)

        collated_samples = move_to_cuda(collated_samples, device=device)
        self.samples_data_list.clear()
        
        collated_samples, all_rewards = self.compute_advantages(collated_samples)

        if collated_samples['rewards']['sum'].numel() == 0:
            if self.trainer.proc_rank == 0:
                print(f"| PostTrain Epoch {self.posttrain_epoch}: all samples filtered (zero reward), skip training")
            return
        
        # log epoch rewards
        if self.trainer.proc_rank == 0:
            self.trainer.logger.add_scalar('monitor/epoch_rewards', all_rewards.mean().item(), self.posttrain_epoch)
        
        filtered_samples = collated_samples
        del filtered_samples['item_names']

        # delete timestep endpoints
        filtered_samples["timesteps"] = filtered_samples["timesteps"][:, 1:-1]
        # filter timesteps
        timestep_range = hparams['train'].get('timestep_range', [0.0, 1.0])
        timestep_range = [int(timestep_range[0] * filtered_samples["timesteps"].shape[1]), int(timestep_range[1] * filtered_samples["timesteps"].shape[1])]
        filtered_samples["timesteps"] = filtered_samples["timesteps"][:, timestep_range[0]:timestep_range[1]]

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape
        num_train_timesteps = int(num_timesteps_filtered)

        if total_batch_size_filtered <= 0 or num_train_timesteps <= 0:
            if self.trainer.proc_rank == 0:
                print(
                    f"| PostTrain Epoch {self.posttrain_epoch}: empty samples after filtering timesteps "
                    f"(B={total_batch_size_filtered}, T={num_train_timesteps}), skip training"
                )
            return

        train_micro_bsz = int(hparams['train']['batch_size'])
        train_micro_bsz = max(train_micro_bsz, 1)

        grad_accum_steps = int(hparams['train'].get('gradient_accumulation_steps', 1) or 1)
        grad_accum_steps = max(grad_accum_steps, 1)

        effective_grad_accum_steps = grad_accum_steps * num_train_timesteps
        accumulated_steps_since_opt = 0

        self.trainer.optimizers[0].zero_grad()

        for inner_epoch in range(hparams['train']['num_inner_epochs']):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {}
            for k, v in filtered_samples.items():
                if k == 'rewards':
                    shuffled_filtered_samples[k] = {rk: v[rk][perm] for rk in v}
                else:
                    shuffled_filtered_samples[k] = v[perm]

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            shuffled_filtered_samples["timesteps"] = shuffled_filtered_samples["timesteps"][
                torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
            ]

            samples_batched_list = []
            for start in range(0, total_batch_size_filtered, train_micro_bsz):
                end = min(start + train_micro_bsz, total_batch_size_filtered)
                batch_dict = {}
                for key, val_tensor in shuffled_filtered_samples.items():
                    if key == 'rewards':
                        batch_dict[key] = {rk: val_tensor[rk][start:end] for rk in val_tensor}
                    else:
                        batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            num_batches = len(samples_batched_list)

            if self.trainer.proc_rank_local == 0:
                inner_training_pbar = tqdm(
                    list(enumerate(samples_batched_list)),
                    desc=f"| PostTrain Epoch {self.posttrain_epoch}/{inner_epoch}: training",
                    position=0,
                )
            else:
                inner_training_pbar = list(enumerate(samples_batched_list))
            for i, train_sample_batch in inner_training_pbar:
                # current_micro_batch_size = len(train_sample_batch["txt_tokens"])
                
                print(
                    f"| rewards={train_sample_batch['rewards']['sum'].mean().detach():.4f}, "
                    f"rewards_std={train_sample_batch['rewards']['sum'].std().detach():.4f}, "
                    f"advantages={train_sample_batch['advantages'].mean().detach():.4f}"
                )
                print(f"| each reward: ", end='')
                for rk in train_sample_batch['rewards']:
                    if rk != 'sum':
                        print(f"{rk}={train_sample_batch['rewards'][rk].mean().detach():.4f}, ", end='')
                # print()

                timestep_progress_bar_log = {}

                if self.trainer.proc_rank_local == 0:
                    inner_timestep_training_pbar = tqdm(
                        enumerate(range(num_train_timesteps)),
                        desc="| Training Timestep",
                        position=1,
                        leave=False,
                    )
                else:
                    inner_timestep_training_pbar = enumerate(range(num_train_timesteps))
                for j_idx, j_timestep_orig_idx in inner_timestep_training_pbar:
                    
                    assert j_idx == j_timestep_orig_idx

                    x1 = train_sample_batch['lat_pred']
                    t = train_sample_batch["timesteps"][:, j_idx]   # [B]
                    t_expanded = t[:, None, None]
                    x0 = torch.randn_like(x1)
                    xt = t_expanded * x1 + (1 - t_expanded) * x0

                    lat_lens = train_sample_batch['tgt_len']
                    ctx_mask = train_sample_batch['ctx_mask']
                    loss_mask = sequence_mask(lat_lens, maxlen=xt.shape[1])[:, :, None].to(xt) * (1-ctx_mask)
                    # loss_mask_sum = loss_mask.sum()
                    # loss_mask_scale = loss_mask.shape[0] * loss_mask.shape[1] / loss_mask_sum
                    loss_mask_batch_scale = loss_mask.shape[1] / loss_mask[..., 0].sum(dim=1)   # [B]

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                        infer_inputs = self.prepare_cfg_input(train_sample_batch, device)
                        xt_input = torch.cat([xt] * 3, dim=0)
                        t_3 = t.unsqueeze(1).repeat(1,3).reshape(-1)
                        
                        with torch.no_grad():
                            old_prediction = self.model_forward_step(unwrap_model(self.dit_old), xt_input, t_3, infer_inputs)
                            ref_forward_prediction = self.model_forward_step(unwrap_model(self.dit_ref), xt_input, t_3, infer_inputs)

                        forward_prediction = self.model_forward_step(unwrap_model(self.dit), xt_input, t_3, infer_inputs)

                    loss_terms = {}

                    # Policy Gradient Loss
                    adv_clip_max = hparams['train']['adv_clip_max']
                    nft_beta = hparams['nft_beta']
                    advantages_clip = torch.clamp(
                        train_sample_batch["advantages"], -adv_clip_max, adv_clip_max,
                    )
                    if 'adv_mode' in hparams['train']:
                        if hparams['train']["adv_mode"] == "positive_only":
                            advantages_clip = torch.clamp(advantages_clip, 0, adv_clip_max)
                        elif hparams['train']["adv_mode"] == "negative_only":
                            advantages_clip = torch.clamp(advantages_clip, -adv_clip_max, 0)
                        elif hparams['train']["adv_mode"] == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                            )
                        elif hparams['train']["adv_mode"] == "binary":
                            advantages_clip = torch.sign(advantages_clip)
                    loss_terms["monitor/advantage"] = train_sample_batch["advantages"].mean().detach()
                    loss_terms["monitor/rewards"] = train_sample_batch["rewards"]['sum'].mean().detach()
                    loss_terms["monitor/rewards_std"] = train_sample_batch["rewards"]['sum'].std().detach()
                    for rk in train_sample_batch['rewards']:
                        if rk != 'sum':
                            loss_terms[f"monitor/rewards_{rk}"] = train_sample_batch["rewards"][rk].mean().detach()
                            loss_terms[f"monitor/rewards_{rk}_std"] = train_sample_batch["rewards"][rk].std().detach()

                    # normalize advantage
                    normalized_advantages_clip = (advantages_clip / adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)
                    positive_prediction = nft_beta * forward_prediction + (1 - nft_beta) * old_prediction.detach()
                    implicit_negative_prediction = (1.0 + nft_beta) * old_prediction.detach() - nft_beta * forward_prediction
                    
                    loss_terms["monitor/x1_norm"] = tensor_mean_per_element(x1**2, loss_mask).detach()
                    loss_terms["monitor/x1_norm_max"] = ( (x1**2) * loss_mask ).max().detach()
                    loss_terms["monitor/old_deviate"] = tensor_mean_per_element((forward_prediction - old_prediction) ** 2, loss_mask).detach()
                    loss_terms["monitor/old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2 * loss_mask).detach()
                    loss_terms["monitor/advantages_clip"] = normalized_advantages_clip.mean().detach()

                    # adaptive weighting
                    x1_prediction = xt + (1 - t_expanded) * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs((x1_prediction.float() - x1.float()) * loss_mask)
                            .mean(dim=tuple(range(1, x1.ndim)), keepdim=True)
                            .clip(min=0.00001) * loss_mask_batch_scale[:, None, None]
                        )
                    positive_loss = tensor_mean_per_seq((x1_prediction - x1) ** 2 / weight_factor, loss_mask)
                    loss_terms["pos"] = positive_loss.mean().detach()

                    negative_x1_prediction = xt + (1 - t_expanded) * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs((negative_x1_prediction.float() - x1.float()) * loss_mask)
                            .mean(dim=tuple(range(1, x1.ndim)), keepdim=True)
                            .clip(min=0.00001) * loss_mask_batch_scale[:, None, None]
                        )
                    negative_loss = tensor_mean_per_seq((negative_x1_prediction - x1) ** 2 / negative_weight_factor, loss_mask)
                    loss_terms['neg'] = negative_loss.mean().detach()

                    ori_policy_loss = r * positive_loss / nft_beta + (1.0 - r) * negative_loss / nft_beta
                    policy_loss = ori_policy_loss * adv_clip_max
                    policy_loss = policy_loss.mean()

                    loss = policy_loss
                    loss_terms["policy"] = policy_loss.detach()

                    kl_div_loss = tensor_mean_per_element((forward_prediction - ref_forward_prediction) ** 2, loss_mask)
                    loss += hparams['train']['beta'] * kl_div_loss
                    loss_terms["kl"] = kl_div_loss.detach()
                    loss_terms["kl_old"] = tensor_mean_per_element((old_prediction - ref_forward_prediction) ** 2, loss_mask)

                    # loss_terms["total_loss"] = loss.detach()

                    scaled_loss = loss / effective_grad_accum_steps

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if self.trainer.amp and self.trainer.amp_scalar is not None:
                            self.trainer.amp_scalar.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                    
                    accumulated_steps_since_opt += 1
                    self._pt_backward_step += 1

                    has_nan_grad, nan_params_names = self._check_nan_grad()
                    if has_nan_grad:
                        print(
                            f"| WARN: found nan in grad! first nan params: {nan_params_names[0]}; "
                            f"last nan params: {nan_params_names[-1]}. reset accumulation."
                        )
                        self.trainer.optimizers[0].zero_grad()
                        accumulated_steps_since_opt = 0

                    did_opt_step = False
                    if (not has_nan_grad) and (accumulated_steps_since_opt > 0) and (accumulated_steps_since_opt % effective_grad_accum_steps == 0):
                        self._optimizer0_step(loss_terms=loss_terms, rescale_grads=1.0)
                        did_opt_step = True
                        accumulated_steps_since_opt = 0

                    backward_step = self._pt_backward_step
                    opt_step = self._pt_opt_step

                    loss_terms = tensors_to_scalars(loss_terms)
                    for params_group_i in range(len(self.trainer.optimizers[0].param_groups)):
                        loss_terms[f'lr/optimizer0_params_group{params_group_i}'] = self.trainer.optimizers[0].param_groups[params_group_i]['lr']
                    # add to progress bar
                    progress_bar_log = {}
                    for k, v in loss_terms.items():
                        if 'monitor/' in k:
                            continue
                        if '/' in k:
                            k_split = k.split("/")
                            assert len(k_split) == 2, "we only support one `/` in tag_name, i.e., `<tag>/<sub_tag>`"
                            k = k.replace("/", "_")
                        k = k.replace('optimizer', 'optm').replace('params_group', 'pg')
                        assert k not in progress_bar_log, f"we got duplicate tags in log_outputs, check this `{k}`"
                        progress_bar_log[k] = v
                    # progress_bar_log['gn'] = grad_norm
                    for k, v in progress_bar_log.items():
                        timestep_progress_bar_log[k] = timestep_progress_bar_log.get(k, 0.0) + v / num_train_timesteps
                    # add to tensorboard
                    tb_log = {}
                    for k, v in loss_terms.items():
                        if '/' in k:
                            tb_log[k] = v
                        else:
                            tb_log[f'tr/{k}'] = v
                    if self.trainer.proc_rank == 0:
                        for k, v in tb_log.items():
                            if isinstance(v, torch.Tensor):
                                v = v.item()
                            self.trainer.logger.add_scalar(k, v, backward_step)
                        # if j_idx % hparams['train']['pbar_log_interval'] == 0:
                        #     progress_bar_log = ', '.join([f"{k}={v:.5f}" for k, v in progress_bar_log.items()])
                        #     print(f"| training_step:{self.posttrain_epoch}/{inner_epoch}/{i}/{j_idx} | global step:{global_step} | {progress_bar_log}")
                        progress_bar_log = ', '.join([f"{k}={v:.5f}" for k, v in progress_bar_log.items()])
                        inner_timestep_training_pbar.set_postfix_str(
                            f"| training_step:{self.posttrain_epoch}/{inner_epoch}/{i}/{j_idx} | "
                            f"backward_step:{backward_step} | opt_step:{opt_step} | "
                            f"{progress_bar_log}"
                        )
                
                if loss_terms.get('monitor/grad_norm_optm0'):
                    timestep_progress_bar_log['gn'] = loss_terms['monitor/grad_norm_optm0']
                timestep_progress_bar_log = ', '.join([f"{k}={v:.5f}" for k, v in timestep_progress_bar_log.items()])
                print(f"\n| Training Timestep End | {timestep_progress_bar_log}")

            # ... after inner loops, before barrier ...
            # flush tail grads (drop_last=False semantics)
            if accumulated_steps_since_opt > 0:
                has_nan_grad, _ = self._check_nan_grad()
                if has_nan_grad:
                    print("| WARN: tail grads contain NaN, skip tail optimizer step and clear grads")
                    self.trainer.optimizers[0].zero_grad()
                    accumulated_steps_since_opt = 0
                else:
                    if self.trainer.proc_rank == 0:
                        print(
                            f"| Tail optimizer step: accumulated_steps_since_opt={accumulated_steps_since_opt}, "
                            f"effective_grad_accum_steps={effective_grad_accum_steps}"
                        )
                    rescale = float(effective_grad_accum_steps) / float(accumulated_steps_since_opt)
                    self._optimizer0_step(loss_terms=None, rescale_grads=rescale)
                    accumulated_steps_since_opt = 0

        if world_size > 1:
            torch.distributed.barrier()

        with torch.no_grad():
            update_steps = int(getattr(self, "_pt_opt_step", 0))
            backward_steps = int(getattr(self, "_pt_backward_step", 0))
            decay = return_decay(update_steps, hparams['decay_type'])
            print(
                f"| PostTrain Epoch:{self.posttrain_epoch} | backward_steps:{backward_steps} | "
                f"update_steps:{update_steps} | inner_epochs end, updating old model with decay {decay}"
            )
            def _norm_param_name(name: str) -> str:
                if name.startswith('module.'):
                    name = name[len('module.'):]
                if name.startswith('_fsdp_wrapped_module.'):
                    name = name[len('_fsdp_wrapped_module.'):]
                name = name.replace('_orig_mod.', '')
                return name

            if isinstance(self.dit, FullyShardedDataParallel):
                offload_to_cpu = bool(hparams.get('fsdp_update_old_offload_to_cpu', True))
                moved_old_to_cpu = False

                try:
                    if offload_to_cpu:
                        self.dit_old.to('cpu')
                        moved_old_to_cpu = True

                    with self.dit.summon_full_params(
                        self.dit,
                        rank0_only=False,
                        writeback=False,
                        with_grads=False,
                        offload_to_cpu=offload_to_cpu,
                    ):
                        online = {_norm_param_name(n): p for n, p in self.dit.named_parameters()}
                        target = {_norm_param_name(n): p for n, p in self.dit_old.named_parameters()}

                        for n, p_online in online.items():
                            p_tgt = target.get(n, None)
                            if p_tgt is None:
                                continue

                            p_src = p_online.detach()
                            if p_src.dtype != p_tgt.dtype:
                                p_src = p_src.to(dtype=p_tgt.dtype)

                            if p_src.device != p_tgt.device:
                                p_src = p_src.to(device=p_tgt.device)

                            p_tgt.data.copy_(p_tgt.data * decay + p_src * (1.0 - decay))
                finally:
                    if moved_old_to_cpu:
                        self.dit_old.to(self.trainer.device)
            else:
                online = {_norm_param_name(n): p for n, p in unwrap_model(self.dit).named_parameters()}
                target = {_norm_param_name(n): p for n, p in unwrap_model(self.dit_old).named_parameters()}

                for n, p_online in online.items():
                    p_tgt = target.get(n, None)
                    if p_tgt is None:
                        continue

                    p_src = p_online.detach()
                    if p_src.dtype != p_tgt.dtype:
                        p_src = p_src.to(dtype=p_tgt.dtype)

                    if p_src.device != p_tgt.device:
                        p_src = p_src.to(device=p_tgt.device)

                    p_tgt.data.copy_(p_tgt.data * decay + p_src * (1.0 - decay))

        if world_size > 1:
            torch.distributed.barrier()

        self.maybe_run_reward_eval()

        print()

    def _check_nan_grad(self):
        has_nan_grad = False
        nan_params_names = []
        for name, param in self.named_parameters():
            if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                has_nan_grad = True
                nan_params_names.append(name)
        return has_nan_grad, nan_params_names

    def _optimizer0_step(self, loss_terms: dict = None, rescale_grads: float = 1.0):
        opt = self.trainer.optimizers[0]

        if self.trainer.amp and self.trainer.amp_scalar is not None:
            self.trainer.amp_scalar.unscale_(opt)

        if rescale_grads != 1.0:
            scale = float(rescale_grads)
            for group in opt.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(scale)

        grad_norm = self.compute_grad_norm(opt, distributed=True, norm_type=2.0)
        if isinstance(loss_terms, dict):
            loss_terms['monitor/grad_norm_optm0'] = grad_norm

        if self.gradient_clip_norm > 0 or self.gradient_clip_val > 0:
            for n in self.trainer.training_module_names:
                m = getattr(self, n)
                if self.gradient_clip_norm > 0:
                    if isinstance(m, FullyShardedDataParallel):
                        _ = m.clip_grad_norm_(self.gradient_clip_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(m.parameters(), self.gradient_clip_norm)
                if self.gradient_clip_val > 0:
                    assert not isinstance(m, FullyShardedDataParallel)
                    torch.nn.utils.clip_grad_value_(m.parameters(), self.gradient_clip_val)

        if self.trainer.amp and self.trainer.amp_scalar is not None:
            self.trainer.amp_scalar.step(opt)
            self.trainer.amp_scalar.update()
        else:
            opt.step()
        opt.zero_grad()

        if hparams.get('use_ema', False):
            self.ema_update(self.ema_model, self.dit, self.config.ema_decay)

        self._pt_opt_step += 1
        return grad_norm

    def _is_zero_reward(self, reward_tensor) -> bool:
        if reward_tensor is None:
            return False
        if not torch.is_tensor(reward_tensor):
            return False
        try:
            return float(reward_tensor.float().abs().item()) <= self._zero_reward_eps
        except Exception:
            return False

    def _filter_collated_samples_by_keep_mask(self, collated_samples, keep_mask: torch.Tensor):
        if keep_mask is None:
            return collated_samples, 0
        if not torch.is_tensor(keep_mask):
            keep_mask = torch.tensor(keep_mask, device=self.trainer.device, dtype=torch.bool)
        keep_mask = keep_mask.bool().view(-1)

        total = int(keep_mask.numel())
        kept = int(keep_mask.long().sum().item()) if total > 0 else 0
        dropped = total - kept
        if dropped <= 0:
            return collated_samples, 0

        keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

        def _maybe_filter_val(val):
            if torch.is_tensor(val):
                if val.ndim >= 1 and val.shape[0] == total:
                    return val[keep_idx]
                return val
            if isinstance(val, dict):
                out = {}
                for k, vv in val.items():
                    if torch.is_tensor(vv) and vv.ndim >= 1 and vv.shape[0] == total:
                        out[k] = vv[keep_idx]
                    else:
                        out[k] = vv
                return out
            if isinstance(val, list):
                if len(val) == total:
                    keep_idx_cpu = keep_idx.detach().cpu().tolist()
                    return [val[i] for i in keep_idx_cpu]
                return val
            return val

        filtered = {k: _maybe_filter_val(v) for k, v in collated_samples.items()}
        return filtered, dropped


class PromptTTSPostTrainMultiRewardTask(PromptTTSPostTrainBaseTask):
    def build_reward_model(self):

        reward_model_pack = {}
        
        if hparams['reward'].get('gemini3pro'):
            from tasks.tts.posttrain.reward_models.gemini import GeminiRewardModel
            reward_model_pack['gemini3pro'] = GeminiRewardModel()

        if hparams['reward'].get('phone'):
            from inference.asr.nar_mfa_infer import MFAInfer
            mfa_ckpt = 'checkpoints/251104_nar_mfa_v6_long_base_robust/model_ckpt_steps_100000.ckpt'
            model = MFAInfer(self.trainer.device, mfa_ckpt, torch_compile=False, precision=torch.bfloat16)
            reward_model_pack['mfa'] = model

        if hparams['reward'].get('sim'):
            from modules.asr.speaker_verification.ecapa_tdnn import ECAPA_TDNN_SMALL
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
            state_dict = torch.load('checkpoints/wavlm/wavlm_large_finetune.pth')
            load_results = model.load_state_dict(state_dict['model'], strict=False)
            model.eval()
            model.to(self.trainer.device)
            reward_model_pack['sim'] = model

        if hparams['reward'].get('mos'):
            from torchaudio.pipelines import SQUIM_SUBJECTIVE
            subjective_model = SQUIM_SUBJECTIVE.get_model().to(self.trainer.device)
            subjective_model.eval()
            for param in subjective_model.parameters():
                param.requires_grad = False
                param.grad = None
            wav_nmr, sr_nmr = torchaudio.load(torchaudio.utils.download_asset("tutorial-assets/ctc-decoding/1688-142285-0007.wav"))
            wav_nmr = wav_nmr.to(self.trainer.device)
            if sr_nmr != 16000:
                wav_nmr = torchaudio.functional.resample(wav_nmr, sr_nmr, 16000)
            reward_model_pack['mos'] = {'model': subjective_model, 'wav_nmr': wav_nmr}
        
        if hparams['reward'].get('stoi') or hparams['reward'].get('pesq'):
            from torchaudio.pipelines import SQUIM_OBJECTIVE
            objective_model = SQUIM_OBJECTIVE.get_model().to(self.trainer.device)
            objective_model.eval()
            for param in objective_model.parameters():
                param.requires_grad = False
                param.grad = None
            reward_model_pack['stoi_pesq'] = objective_model
            
        return reward_model_pack
    
    def run_reward_gemini3pro_model(self, wavs, wav_lens, **kwargs):
        bsz = wavs.shape[0]
        inputs = kwargs['inputs']
        tgt_text = inputs['tgt_text']
        ref_wav_paths = inputs['ref_wav_paths']
        
        wavs_np = []
        for i in range(bsz):
            wavs_np.append(wavs[i, :wav_lens[i]].cpu().numpy())
            
        run_dir = self._init_debug_run_dir("gemini3pro")
        save_dir = os.path.join(run_dir, f"step_{self.posttrain_epoch:09d}")
        if self.trainer.proc_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        time_start = time.time()
            
        rewards = []
        gemini_results = []
        with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
            # save_dir = temp_dir
            for i in range(bsz):
                temp_path = os.path.join(save_dir, f"rank{self.trainer.proc_rank}_{i}.wav")
                sf.write(temp_path, wavs_np[i], hparams['audio_sample_rate'], 'PCM_16')
                
                result = self.reward_model_pack['gemini3pro'].process(temp_path, tgt_text[i])

                rewards.append(result['Final_Weighted_Score'] / 10)
                
                # saving #
                result['all_text'] = inputs['text'][i]
                result['text'] = tgt_text[i]
                result['wav_path'] = temp_path
                result['ref_wav_path'] = ref_wav_paths[i]
                
                gemini_results.append(result)
                
            json_dump(gemini_results, os.path.join(save_dir, f'gemini_results_rank{self.trainer.proc_rank}.json'))
            
        if self.trainer.proc_rank == 0:
            print(f"Gemini3pro processing time: {time.time() - time_start} seconds")

        return torch.Tensor(rewards).to(wavs.device)
        
    def run_reward_mfa_model(self, wavs, wav_lens, **kwargs):
        bsz = wavs.shape[0]

        mfa_model = self.reward_model_pack['mfa']
        
        wavs_np = []
        for i in range(bsz):
            wavs_np.append(wavs[i, :wav_lens[i]].cpu().numpy())
        
        mfa_outputs = mfa_model.forward_batch(wavs_np, sample_rates=24000, timesteps=12, max_batch_duration=200, max_batch_size=10, use_tqdm=False)

        inputs = kwargs['inputs']
        phone = inputs['phone'].cpu().numpy()
        tone = inputs['tone'].cpu().numpy()
        ph_mask = inputs['ph_mask'].cpu().numpy()
        ph_lens = ph_mask.sum(1)
        ctx_ph_mask = inputs['ctx_ph_mask'].cpu().numpy()
        ctx_ph_lens = ctx_ph_mask.sum(1)

        rewards = []

        def score_fn(x, y):
            if x[0] == y[0]:
                if x[1] == y[1]:
                    return 1.0
                else:
                    return 0.5
            else:
                return 0.0

        for i in range(bsz):
            if mfa_outputs[i] is None:
                rewards.append(0.0)
                continue

            ph_gt = phone[i, ctx_ph_lens[i]: ph_lens[i]]
            tone_gt = tone[i, ctx_ph_lens[i]: ph_lens[i]]
            ph_pred = mfa_model.ling_dict['phone'].encode(' '.join(mfa_outputs[i]['ph']))
            tone_pred = mfa_model.ling_dict['tone'].encode(' '.join(mfa_outputs[i]['tone']))

            ph_gt = [(p, t) for p, t in zip(ph_gt, tone_gt) if p != 145]
            ph_pred = [(p, t) for p, t in zip(ph_pred, tone_pred) if p != 145]

            try:
                reward, _ = seq_match(ph_gt, ph_pred, score_fn=score_fn, metric='mean')
            except IndexError:
                reward = 0.0

            rewards.append(reward)

            if DEBUG:
                if (
                    self.trainer.proc_rank == 0 and 
                    (self.posttrain_epoch < 10 or self.posttrain_epoch % 20 == 0)
                ):
                    save_dir = f'{hparams["work_dir"]}/sample_batches/step_{self.posttrain_epoch}'
                    os.makedirs(save_dir, exist_ok=True)
                    json_dump({
                        'ph_gt': mfa_model.ling_dict['phone'].decode([p[0] for p in ph_gt]).split(' '),
                        'tone_gt': mfa_model.ling_dict['tone'].decode([p[1] for p in ph_gt]).split(' '),
                        'ph_pred': mfa_model.ling_dict['phone'].decode([p[0] for p in ph_pred]).split(' '),
                        'tone_pred': mfa_model.ling_dict['tone'].decode([p[1] for p in ph_pred]).split(' '),
                        'reward': reward
                    }, f"{save_dir}/phone_tone_pred_{i}.json")

        return torch.Tensor(rewards).to(wavs.device)

    def run_reward_sim_model(self, wavs, wav_lens, **kwargs):
        sample = kwargs['sample']

        ctx_wavs = sample["ctx_wavs"]
        ctx_mask = sample["ctx_mask"]
        if len(ctx_mask.shape) == 3:
            ctx_mask = ctx_mask[:, :, 0]
        ctx_lens = ctx_mask.sum(dim=1) * hparams['hop_size'] * hparams['vae_stride']

        wavs = repeat_pad_wavs(wavs, wav_lens)
        ctx_wavs = repeat_pad_wavs(ctx_wavs, ctx_lens)

        if hparams['audio_sample_rate'] != 16000 and hparams['audio_sample_rate'] not in self.resamplers:
            self.resamplers[hparams['audio_sample_rate']] = torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=16000).to(self.trainer.device)

        wavs = self.resamplers[hparams['audio_sample_rate']](wavs)
        ctx_wavs = self.resamplers[hparams['audio_sample_rate']](ctx_wavs)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            embeds_pred = self.reward_model_pack['sim'](wavs)   # [B, 256]
            embeds_ctx = self.reward_model_pack['sim'](ctx_wavs)

        sim = F.cosine_similarity(embeds_pred, embeds_ctx)  # [B]

        return sim

    def run_reward_mos_model(self, wavs, wav_lens, **kwargs):     
        bsz = wavs.shape[0]  
        wavs = repeat_pad_wavs(wavs, wav_lens)

        if hparams['audio_sample_rate'] != 16000 and hparams['audio_sample_rate'] not in self.resamplers:
            self.resamplers[hparams['audio_sample_rate']] = torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=16000).to(self.trainer.device)

        wavs = self.resamplers[hparams['audio_sample_rate']](wavs)
        wav_nmr = torch.cat([self.reward_model_pack['mos']['wav_nmr']] * bsz, dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            mos = self.reward_model_pack['mos']['model'](wavs, wav_nmr)

        return mos / 5

    def run_reward_stoi_pesq_model(self, wavs, wav_lens, **kwargs):
        wavs = repeat_pad_wavs(wavs, wav_lens)

        if hparams['audio_sample_rate'] != 16000 and hparams['audio_sample_rate'] not in self.resamplers:
            self.resamplers[hparams['audio_sample_rate']] = torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=16000).to(self.trainer.device)

        wavs = self.resamplers[hparams['audio_sample_rate']](wavs)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            stoi, pesq, si_sdr = self.reward_model_pack['stoi_pesq'](wavs)

        return stoi, pesq / 5

    def run_reward_model(self, wavs, wav_lens, **kwargs):
        bsz = wavs.shape[0]

        rewards = {}
        
        if hparams['reward'].get('gemini3pro'):
            rewards['gemini3pro'] = self.run_reward_gemini3pro_model(wavs, wav_lens, **kwargs)

        if hparams['reward'].get('phone'):
            rewards['phone'] = self.run_reward_mfa_model(wavs, wav_lens, **kwargs)

        if hparams['reward'].get('sim'):
            rewards['sim'] = self.run_reward_sim_model(wavs, wav_lens, **kwargs)
        
        if hparams['reward'].get('mos'):
            rewards['mos'] = self.run_reward_mos_model(wavs, wav_lens, **kwargs)
        
        if hparams['reward'].get('stoi') or hparams['reward'].get('pesq'):
            stoi, pesq = self.run_reward_stoi_pesq_model(wavs, wav_lens, **kwargs)
            if hparams['reward']['stoi']:
                rewards['stoi'] = stoi
            if hparams['reward']['pesq']:
                rewards['pesq'] = pesq
        
        rewards_agg = torch.zeros(bsz, device=wavs.device)
        for k in rewards:
            rewards_agg += rewards[k]
        rewards['sum'] = rewards_agg / len(rewards)
        
        return rewards
    

class PromptTTSPostTrainMultiRewardUnsyncTask(PromptTTSPostTrainMultiRewardTask):
    def build_reward_model(self):
        return {}

    def sampling(
        self,
        inputs,
        sample,
        infer_batch_size: int = None,
        sampler_model=None,
        debug_tag: str = "rewards",
        save_prefix: str = "train",
        epoch_for_save: int = None,
    ):
        device = self.trainer.device

        bsz = int(inputs['wav_lengths'].shape[0])
        if infer_batch_size is None:
            infer_batch_size = int(hparams.get('sample', {}).get('rollout_micro_batch_size', 0) or 0)
        if infer_batch_size <= 0:
            infer_batch_size = bsz
        infer_batch_size = min(infer_batch_size, bsz)

        sampler_model = self.dit_old if sampler_model is None else sampler_model
        epoch_for_save = int(self.posttrain_epoch if epoch_for_save is None else epoch_for_save)

        run_dir = self._init_debug_run_dir(debug_tag)
        save_dir = os.path.join(run_dir, f"epoch_{epoch_for_save:09d}")
        if self.trainer.proc_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        base_offset = len(getattr(self, 'samples_data_list', []))

        lat_pred_chunks = []
        rewards_futures_all = []
        t_schedule = None

        for start in range(0, bsz, infer_batch_size):
            end = min(start + infer_batch_size, bsz)
            slc = slice(start, end)

            inputs_mb = slice_batch_value(inputs, slc)
            sample_mb = slice_batch_value(sample, slc)
            infer_inputs_mb = self.prepare_cfg_input(inputs_mb, device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True), torch.no_grad():
                lat_pred_mb, t_schedule_mb = sampler_model.inference(
                    infer_inputs_mb,
                    timesteps=hparams['sample']['num_steps'],
                    seq_cfg_w=hparams['sample']['guidance_scale'],
                    timestep_annealing_w=(0.6, 0.6, 1.0),
                    return_timesteps=True,
                )

                if t_schedule is None:
                    t_schedule = t_schedule_mb

                ctx_mask_mb = inputs_mb['ctx_mask']
                lat_ctx_mb = inputs_mb['lat_ctx']
                wav_lengths_mb = inputs_mb['wav_lengths']
                ctx_wav_lens_mb = inputs_mb['ctx_wav_lens']

                lat_pred_mb = lat_pred_mb * (1 - ctx_mask_mb) + lat_ctx_mb * ctx_mask_mb

                try:
                    wav_pred_mb = vae_decode(self.vae, lat_pred_mb)[:, 0]
                except:
                    if self.trainer.proc_rank_local == 0:
                        traceback.print_exc()
                        print(f"| ERROR: VAE decoding fail: lat_pred.shape={lat_pred_mb.shape}, skip")
                        print(f"item_names={inputs_mb.get('item_names')}")
                    return

                wav_pred_lens_mb = wav_lengths_mb - ctx_wav_lens_mb
                wav_pred_mb = remove_prefix(wav_pred_mb[..., None], ctx_wav_lens_mb, wav_pred_lens_mb)[..., 0].float()

                rewards_futures_mb = self.push_samples_future(
                    wav_pred_mb,
                    wav_pred_lens_mb,
                    inputs_mb,
                    sample=sample_mb,
                    offset=base_offset + start,
                    save_dir=save_dir,
                    save_prefix=save_prefix,
                    epoch_for_save=epoch_for_save,
                )

            lat_pred_chunks.append(lat_pred_mb)
            rewards_futures_all.extend(rewards_futures_mb)

        lat_pred = torch.cat(lat_pred_chunks, dim=0) if len(lat_pred_chunks) > 1 else lat_pred_chunks[0]

        out = dict(inputs)
        out.update({
            'timesteps': t_schedule,
            'lat_pred': lat_pred,
            'rewards_futures': rewards_futures_all,
        })
        return out
    
    def push_samples_future(
        self,
        wav_pred,
        wav_lens,
        inputs,
        sample=None,
        offset: int = 0,
        save_dir: str = None,
        save_prefix: str = "train",
        epoch_for_save: int = None,
    ):
        epoch_for_save = int(self.posttrain_epoch if epoch_for_save is None else epoch_for_save)
        
        if save_dir is None:
            run_dir = self._init_debug_run_dir("rewards")
            save_dir = os.path.join(run_dir, f"epoch_{epoch_for_save:09d}")
            if self.trainer.proc_rank == 0:
                os.makedirs(save_dir, exist_ok=True)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        bsz = len(wav_pred)
        wav_pred = [wav_pred[i, :wav_lens[i]].cpu().numpy() for i in range(bsz)]

        rewards_futures = []
        
        for i in range(bsz):
            rewards_future = {}
            inputs_to_reward = {
                'wav': wav_pred[i],
                'sr': hparams['audio_sample_rate'],
                'text': inputs['tgt_text'][i],
                'ref_wav_path': inputs['ref_wav_paths'][i],
            }

            if sample is not None and 'ctx_wavs' in sample:
                ref_wav = sample['ctx_wavs'][i]
                if isinstance(ref_wav, torch.Tensor):
                    ref_wav = ref_wav.detach().float()
                    if 'ctx_mask' in sample:
                        ctx_mask = sample['ctx_mask'][i]
                        if isinstance(ctx_mask, torch.Tensor):
                            if ctx_mask.ndim == 2:
                                ctx_mask = ctx_mask[:, 0]
                            ctx_wav_len = int(ctx_mask.float().sum().item()) * hparams['hop_size'] * hparams['vae_stride']
                            ctx_wav_len = max(1, min(ctx_wav_len, int(ref_wav.shape[-1])))
                            ref_wav = ref_wav[:ctx_wav_len]
                    inputs_to_reward['ref_wav'] = ref_wav.cpu().numpy()
                    inputs_to_reward['ref_sr'] = hparams['audio_sample_rate']

            inputs_path = os.path.join(hparams.get('reward_work_dir', 'user/reward_cache'), hparams['exp_name'], 'inputs')
            os.makedirs(inputs_path, exist_ok=True)
            save_name = f"{save_prefix}_rank{self.trainer.proc_rank}_epoch{epoch_for_save}_item{offset + i}.npy"
            inputs_path = os.path.join(inputs_path, save_name)
            with open(inputs_path + '.tmp', 'wb') as f:
                np.save(f, inputs_to_reward, allow_pickle=True)
            os.rename(inputs_path + '.tmp', inputs_path)
            outputs_path = os.path.join(hparams.get('reward_work_dir', 'user/reward_cache'), hparams['exp_name'], 'outputs')
            os.makedirs(outputs_path, exist_ok=True)
            outputs_path = os.path.join(outputs_path, save_name)
            rewards_future['result_path'] = outputs_path
            
            temp_path = os.path.join(
                save_dir,
                f"{save_prefix}_rank{self.trainer.proc_rank}_{offset + i}.wav",
            )
            sf.write(temp_path, wav_pred[i], hparams['audio_sample_rate'], 'PCM_16')
            rewards_future['wav_path'] = temp_path
            
            rewards_future['text'] = inputs['tgt_text'][i]
            rewards_future['all_text'] = inputs['text'][i]
            rewards_future['ref_wav_path'] = inputs['ref_wav_paths'][i]
            
            rewards_futures.append(rewards_future)
            
        return rewards_futures
    
    def fetch_rewards_results(
        self,
        rewards_futures,
        debug_tag: str = "rewards",
        epoch_for_save: int = None,
        debug_file_prefix: str = "",
    ):
        device = self.trainer.device
        epoch_for_save = int(self.posttrain_epoch if epoch_for_save is None else epoch_for_save)

        # 训练用的数值 reward
        rewards = {}
        # debug 用的统一 JSON：一条样本一条记录
        debug_items = []
        
        for rewards_future in rewards_futures:
            rewards_result_path = rewards_future['result_path']
            while not os.path.isfile(rewards_result_path):
                time.sleep(0.1)
            rewards_result = np.load(rewards_result_path, allow_pickle=True)
            if (
                isinstance(rewards_result, np.ndarray)
                and rewards_result.dtype == object
                and rewards_result.size == 1
            ):
                rewards_result = rewards_result.item()
            os.remove(rewards_result_path)

            # 这一条样本的公共信息
            debug_item = {
                'all_text': rewards_future['all_text'],
                'tgt_text': rewards_future['text'],
                'wav_path': rewards_future['wav_path'],
                'ref_wav_path': rewards_future['ref_wav_path'],
            }

            # rewards_result 形如 {"gemini3pro": {...}, "lai": {...}, ...}
            # 兜底：reward worker/模型出错时，可能写出 None/非法对象
            if rewards_result is None:
                rewards_result = {}
                debug_item['reward_error'] = 'rewards_result is None'
            elif not isinstance(rewards_result, dict):
                try:
                    rewards_result = dict(rewards_result)
                except Exception:
                    debug_item['reward_error'] = f"invalid rewards_result type: {type(rewards_result)}"
                    rewards_result = {}

            debug_item['reward_scalar'] = {}
            sample_scores = {}

            for reward_k, reward_val in rewards_result.items():
                reward_val = reward_val if reward_val is not None else {}

                try:
                    if reward_k == 'gemini3pro':
                        score = float(reward_val['Final_Weighted_Score'] / 10.0)
                    elif reward_k == 'lai':
                        score = float(reward_val.get('pause_reward', 0.0))
                    elif reward_k == 'qwen3aligner':
                        score = float(reward_val.get('pause_reward', 0.0))
                    elif reward_k == 'phone':
                        score = float(reward_val.get('score', 0.0))
                    elif reward_k == 'sim':
                        score = float(reward_val.get('score', 0.0))
                    else:
                        score = float(reward_val)
                except Exception:
                    score = 0.0

                sample_scores[reward_k] = score
                debug_item['reward_scalar'][reward_k] = score
                debug_item[reward_k] = reward_val

            # 保证 rewards[reward_k] 的长度对齐每条样本，避免后续聚合 shape mismatch
            cur_idx = len(debug_items)
            for reward_k in sample_scores:
                if reward_k not in rewards:
                    rewards[reward_k] = [0.0] * cur_idx
            for reward_k in rewards:
                rewards[reward_k].append(float(sample_scores.get(reward_k, 0.0)))

            debug_item['reward_scalar']['sum'] = (
                float(np.mean(list(sample_scores.values()))) if len(sample_scores) > 0 else 0.0
            )
            debug_items.append(debug_item)
        
        # 统一保存 debug JSON：每条样本一条记录，所有 reward_k 挂在一起
        run_dir = self._init_debug_run_dir(debug_tag)
        save_dir = os.path.join(run_dir, f"epoch_{epoch_for_save:09d}")
        os.makedirs(save_dir, exist_ok=True)
        json_dump(
            debug_items,
            os.path.join(save_dir, f'{debug_file_prefix}rewards_results_rank{self.trainer.proc_rank}.json')
        )
        
        # 聚合所有 reward_k，形成 rewards['sum']，用于训练
        rewards_agg = torch.zeros(len(rewards_futures), device=device)
        if len(rewards) == 0:
            rewards['sum'] = rewards_agg
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return rewards
        for reward_k in rewards:
            rewards[reward_k] = torch.tensor(rewards[reward_k], device=device)
            rewards_agg += rewards[reward_k]
        rewards['sum'] = rewards_agg / len(rewards)
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        return rewards

    def _run_reward_eval_batch(self, sample_cpu, eval_batch_idx: int):
        device = self.trainer.device
        sample = move_to_cuda(deepcopy(sample_cpu), device=device)
        inputs_base = self.prepare_inputs(sample)
        bsz = int(inputs_base['wav_lengths'].shape[0])

        num_generation_per_prompt = self._reward_eval_num_generation_per_prompt
        rollout_micro_batch_size = self._reward_eval_rollout_micro_batch_size
        if rollout_micro_batch_size <= 0:
            rollout_micro_batch_size = bsz
        rollout_micro_batch_size = max(1, rollout_micro_batch_size)
        gen_per_call = max(1, rollout_micro_batch_size // max(1, bsz))

        rewards_futures_all = []
        gen_start = 0
        while gen_start < num_generation_per_prompt:
            cur_rep = min(gen_per_call, num_generation_per_prompt - gen_start)

            inputs_rep = repeat_batch_value(inputs_base, cur_rep)
            sample_rep = repeat_batch_value(sample, cur_rep)

            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices, enabled=True):
                rank = int(getattr(self.trainer, 'proc_rank', 0))
                seed = (
                    self._reward_eval_seed
                    + int(self.posttrain_epoch) * 1000
                    + eval_batch_idx * 10000
                    + rank * 100
                    + gen_start
                )
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                sampled = self.sampling(
                    inputs_rep,
                    sample_rep,
                    infer_batch_size=rollout_micro_batch_size,
                    sampler_model=self.dit_old,
                    debug_tag="val_rewards",
                    save_prefix=f"valb{eval_batch_idx}",
                    epoch_for_save=self.posttrain_epoch,
                )

            if sampled is not None and sampled.get('rewards_futures'):
                rewards_futures_all.extend(sampled['rewards_futures'])
            gen_start += cur_rep

        if len(rewards_futures_all) == 0:
            return {}

        return self.fetch_rewards_results(
            rewards_futures_all,
            debug_tag="val_rewards",
            epoch_for_save=self.posttrain_epoch,
            debug_file_prefix=f"valb{eval_batch_idx}_",
        )

    def maybe_run_reward_eval(self):
        if not self._should_run_reward_eval():
            return

        if len(self._reward_eval_samples) == 0:
            if self.trainer.proc_rank == 0:
                print("| Reward eval skipped: no cached eval samples yet")
            return

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        per_reward_means = {}
        for eval_batch_idx, sample_cpu in enumerate(self._reward_eval_samples):
            rewards = self._run_reward_eval_batch(sample_cpu, eval_batch_idx)
            if len(rewards) == 0:
                continue
            for reward_k, reward_v in rewards.items():
                gathered = all_gather_varlen_tensor(reward_v, dim=0) if self.trainer.num_total_gpus > 1 else reward_v
                if gathered.numel() == 0:
                    continue
                per_reward_means.setdefault(reward_k, []).append(float(gathered.mean().item()))

        if self.trainer.proc_rank == 0:
            if len(per_reward_means) == 0:
                print(f"| Reward eval epoch {self.posttrain_epoch}: empty")
                return

            summary = []
            for reward_k in sorted(per_reward_means.keys()):
                mean_v = float(np.mean(per_reward_means[reward_k]))
                self.trainer.logger.add_scalar(
                    f"monitor/val_epoch_rewards_{reward_k}",
                    mean_v,
                    self.posttrain_epoch,
                )
                summary.append(f"{reward_k}={mean_v:.4f}")

            if 'sum' in per_reward_means:
                self.trainer.logger.add_scalar(
                    "monitor/val_epoch_rewards",
                    float(np.mean(per_reward_means['sum'])),
                    self.posttrain_epoch,
                )

            print(
                f"| Reward eval epoch {self.posttrain_epoch} | "
                + ", ".join(summary)
            )
    
    def compute_advantages(self, collated_samples):
        world_size = self.trainer.num_total_gpus
        rank = self.trainer.proc_rank
        device = self.trainer.device
        
        time_start = time.time()
        rewards = self.fetch_rewards_results(collated_samples['rewards_futures'])
        if self.trainer.proc_rank == 0:
            print(f"\nReward model processing time (retrival): {time.time() - time_start} seconds")
        del collated_samples['rewards_futures']
        
        collated_samples['rewards'] = rewards

        rewards = collated_samples['rewards']['sum']
        item_names = collated_samples['item_names']

        if self._skip_zero_reward_enabled:
            keep_mask = torch.isfinite(rewards) & (rewards.abs() > self._zero_reward_eps)
            collated_samples, dropped = self._filter_collated_samples_by_keep_mask(collated_samples, keep_mask)
            if dropped > 0 and self.trainer.proc_rank == 0:
                print(
                    f"| Skip zero-reward samples (reward retrieval stage): dropped={dropped}/{int(keep_mask.numel())}, "
                    f"eps={self._zero_reward_eps}"
                )
            rewards = collated_samples['rewards']['sum']
            item_names = collated_samples['item_names']

        if hparams['per_prompt_stat_tracking']:
            if world_size > 1:
                all_rewards = all_gather_varlen_tensor(rewards, dim=0)  # [B]
                all_item_names = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(all_item_names, item_names)
                all_item_names = [item_name for item_names in all_item_names for item_name in item_names]
                assert len(all_rewards) == len(all_item_names)
            else:
                all_rewards = rewards
                all_item_names = item_names
            item_names_unique = list(set(all_item_names))
            item_name2idxs = {item_name: [] for item_name in item_names_unique}
            for item_name_idx, item_name in enumerate(all_item_names):
                item_name2idxs[item_name].append(item_name_idx)
            item_name2avg = {}
            item_name2std = {}
            for item_name in item_names_unique:
                item_name_idxs = item_name2idxs[item_name]
                if len(item_name_idxs) == 0:
                    item_name2avg[item_name] = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
                    item_name2std[item_name] = torch.zeros_like(item_name2avg[item_name])
                else:
                    item_name2avg[item_name] = all_rewards[item_name_idxs].mean()
                    item_name2std[item_name] = (
                        all_rewards[item_name_idxs].std()
                        if len(item_name_idxs) > 1
                        else torch.zeros_like(item_name2avg[item_name])
                    )
            rewards_avg_list, rewards_std_list = [], []
            for item_name in item_names:
                rewards_avg_list.append(item_name2avg[item_name])
                rewards_std_list.append(item_name2std[item_name])
            if len(rewards_avg_list) == 0:
                rewards_avg = rewards.new_empty((0,))
                rewards_std = rewards.new_empty((0,))
            else:
                rewards_avg = torch.stack(rewards_avg_list)
                rewards_std = torch.stack(rewards_std_list)
        else:
            if world_size > 1:
                all_rewards = all_gather_varlen_tensor(rewards, dim=0)
            else:
                all_rewards = rewards
            if all_rewards.numel() == 0:
                rewards_avg = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
                rewards_std = torch.zeros_like(rewards_avg)
            else:
                rewards_avg = all_rewards.mean()
                rewards_std = all_rewards.std() if all_rewards.numel() > 1 else torch.zeros_like(rewards_avg)

        advantages = (rewards - rewards_avg) / (rewards_std + 1e-4)
        collated_samples['advantages'] = advantages

        if self.trainer.proc_rank == 0:
            if all_rewards.numel() > 0:
                total_mean = all_rewards.mean().item()
                total_std = all_rewards.std().item() if all_rewards.numel() > 1 else 0.0
                adv_mean = advantages.mean().item() if advantages.numel() > 0 else 0.0
                adv_std = advantages.std().item() if advantages.numel() > 1 else 0.0
                print(f"| Total rewards: mean={total_mean}, std={total_std}")
                print(f"| Total advantages: mean={adv_mean}, std={adv_std}")
            else:
                print("| Total rewards/advantages: empty after filtering")
                
        return collated_samples, all_rewards