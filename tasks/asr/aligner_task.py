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


class Aligner2TowerTask(FastDatasetMixin, ScriptSpeechBaseTask):
    def __init__(self):
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
        self.test_dataloader = FastDatasetMixin.test_dataloader.__get__(self)
        self.val_dataloader = FastDatasetMixin.val_dataloader.__get__(self)
        self.hparams = hparams
        self.config = AttrDict(hparams)

        super().__init__()

    def build_model(self):
        from modules.asr.forced_align.aligner_2tower import build_aligner_model
        self.model, self.text_tokenizer = build_aligner_model(hparams, init_pretrained=True)
        self.model.train()

        return {'trainable': [self.model], 'others': []}

    def fsdp_wrap_policy(self):
        import modules.commons.hf.transformer
        import modules.commons.hf.transformer_moe

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                modules.commons.hf.transformer.EncoderLayer,
                modules.commons.hf.transformer.DecoderLayer,
                modules.commons.hf.transformer_moe.EncoderLayer,
                modules.commons.hf.transformer_moe.DecoderLayer,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.model, hparams['load_ckpt'], 'model', strict=False, mmap=True)
            
    def build_optimizer(self):
        optimizer = AdamW(unwrap_model(self.model).parameters(), **self.config.optimizer)
        return optimizer

    def fsdp_optm2model(self):
        return [self.model]
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.00001:
            kill_void()
        loss_output, model_out = self.run_model(sample)

        loss_weights = {
            'loss': 1.0,
            'align_loss': hparams.get('align_loss_weight', 1.0),
            'dur_loss': hparams.get('dur_loss_weight', 1.0),
            'mono_loss': hparams.get('mono_loss_weight', 1.0),
            'pause_loss': hparams.get('pause_loss_weight', 1.0),
            'crf_loss': hparams.get('v5_crf_loss_weight', 1.0),
        }
        total_loss = sum(
            [loss_weights.get(k, 1) * v for k, v in loss_output.items()
             if isinstance(v, torch.Tensor) and v.requires_grad]
        )
        return total_loss, loss_output

    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}
        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        txt_tokens = sample['txt_tokens']
        txt_lens = sample['txt_lengths']

        # 你数据里每个 word 的 (start,end,conf)（单位：秒）
        word_start_times = sample.get('word_start_times', sample.get('word_start', None))
        word_end_times = sample.get('word_end_times', sample.get('word_end', None))
        word_conf = sample.get('word_conf', sample.get('word_confs', None))

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(wavs.is_cuda)):
            model_outputs = self.model(
                wavs,
                txt_tokens,
                wav_lengths,
                txt_lens,
                word_start_times=word_start_times,
                word_end_times=word_end_times,
                word_conf=word_conf,
            )

        if 'align_loss' in model_outputs:
            losses_out['align_loss'] = model_outputs['align_loss']
        if 'dur_loss' in model_outputs:
            losses_out['dur_loss'] = model_outputs['dur_loss']
        if 'mono_loss' in model_outputs:
            losses_out['mono_loss'] = model_outputs['mono_loss']
        if 'pause_loss' in model_outputs:
            losses_out['pause_loss'] = model_outputs['pause_loss']
        if 'crf_loss' in model_outputs:
            losses_out['crf_loss'] = model_outputs['crf_loss']

        losses_out['bs'] = wavs.shape[0]
        losses_out['ntokens'] = model_outputs.get('ntokens', wavs.new_tensor(0))

        if 'gamma' in model_outputs:
            losses_out['monitor/gamma'] = model_outputs['gamma']
        if 'pause_acc' in model_outputs:
            losses_out['monitor/pause_acc'] = model_outputs['pause_acc']
        if 'mon_p_target_mean' in model_outputs:
            losses_out['monitor/mon_p_target_mean'] = model_outputs['mon_p_target_mean']
        if 'mon_entropy_valid_mean' in model_outputs:
            losses_out['monitor/mon_entropy_valid_mean'] = model_outputs['mon_entropy_valid_mean']
        if 'mon_word_mass_valid_mean' in model_outputs:
            losses_out['monitor/mon_word_mass_valid_mean'] = model_outputs['mon_word_mass_valid_mean']
        if 'mon_p_target_word_only_mean' in model_outputs:
            losses_out['monitor/mon_p_target_word_only_mean'] = model_outputs['mon_p_target_word_only_mean']
        if 'mon_v5_trans_stay_mean' in model_outputs:
            losses_out['monitor/mon_v5_trans_stay_mean'] = model_outputs['mon_v5_trans_stay_mean']
        if 'mon_v5_trans_adv1_mean' in model_outputs:
            losses_out['monitor/mon_v5_trans_adv1_mean'] = model_outputs['mon_v5_trans_adv1_mean']
        if 'mon_v5_trans_adv2_mean' in model_outputs:
            losses_out['monitor/mon_v5_trans_adv2_mean'] = model_outputs['mon_v5_trans_adv2_mean']

        return losses_out, model_out
    

class Aligner2MDMTask(Aligner2TowerTask):
    def build_model(self):
        from modules.asr.forced_align.aligner_mdm import build_aligner_model
        self.model, self.text_tokenizer = build_aligner_model(hparams, init_pretrained=True)
        self.model.train()
        for n_, m_ in unwrap_model(self.model.backbone).named_children():
            num_params(m_, model_name=n_)

        return {'trainable': [self.model], 'others': []}

    def fsdp_wrap_policy(self):
        import modules.commons.hf.transformer
        import modules.commons.hf.transformer_dit
        import modules.commons.hf.transformer_moe

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                modules.commons.hf.transformer.EncoderLayer,
                modules.commons.hf.transformer.DecoderLayer,
                modules.commons.hf.transformer_moe.EncoderLayer,
                modules.commons.hf.transformer_moe.DecoderLayer,
                modules.commons.hf.transformer_dit.EncoderLayer,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.00001:
            kill_void()
        loss_output, model_out = self.run_model(sample)

        loss_weights = {
            'diff_loss': 1.0,
            'mono_loss': hparams.get('mono_loss_weight', 0.0),
        }
        total_loss = sum(
            [loss_weights.get(k, 1) * v for k, v in loss_output.items()
             if isinstance(v, torch.Tensor) and v.requires_grad]
        )

        if self.trainer.proc_rank == 0 and self.global_step < 20:
            save_dir = f'{hparams["work_dir"]}/sample_batches/step_{self.global_step}'
            os.makedirs(save_dir, exist_ok=True)
            sample = convert_to_np(move_to_cpu(sample))
            for i in range(sample['nsamples']):
                sf.write(f"{save_dir}/{i}.wav", sample['wavs'][i, :sample['wav_lengths'][i]], 24000, 'PCM_16')
                np.save(f"{save_dir}/{i}.npy", sample, allow_pickle=True)
            del sample['wavs']

        return total_loss, loss_output

    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}
        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        txt_tokens = sample['txt_tokens']
        txt_lens = sample['txt_lengths']
        timestamp_mask = sample['timestamp_mask']

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(wavs.is_cuda)):
            model_outputs = self.model(
                wavs,
                txt_tokens,
                wav_lengths,
                txt_lens,
                loss_mask=timestamp_mask
            )

        losses_out['diff_loss'] = model_outputs['loss']
        if 'mono_loss' in model_outputs:
            losses_out['mono_loss'] = model_outputs['mono_loss']

        losses_out['bs'] = wavs.shape[0]
        losses_out['ntokens'] = model_outputs.get('ntokens', wavs.new_tensor(0))

        return losses_out, model_out
    