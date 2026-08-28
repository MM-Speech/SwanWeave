import os
import random

from attrdictionary import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from utils.commons.ckpt_utils import load_ckpt
from utils.commons.import_utils import import_module_bystr
from utils.commons.hparams import hparams, set_hparams
from utils.nn.seq_utils import sequence_mask, add_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model
from utils.commons.os_utils import kill_void
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK

from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin
from modules.asr.scriptasr.causal_asr_v2 import build_asr_text_tokenizer, build_asr_model

class CausalASRTask(ScriptSpeechBaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
        self.test_dataloader = FastDatasetMixin.test_dataloader.__get__(self)
        self.val_dataloader = FastDatasetMixin.val_dataloader.__get__(self)
        self.hparams = hparams
        self.config = AttrDict(hparams)

    def build_model(self):
        self.asr_text_tokenizer, self.asr_vocab_size = build_asr_text_tokenizer()
        self.model = build_asr_model(hparams, self.asr_text_tokenizer, init_pretrained=True, vocab_size=self.asr_vocab_size)
        self.model.train()

        return {'trainable': [self.model], 'others': []}

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.model, hparams['load_ckpt'], 'model', strict=False)

    def build_optimizer(self):
        optimizer = AdamW(unwrap_model(self.model).parameters(), **self.config.optimizer)
        return optimizer

    def fsdp_optm2model(self):
        return [self.model]
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        if random.random() < 0.01:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'ce_loss': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['total_loss'] = total_loss.item()

        return total_loss, loss_output
    
    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}

        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        # resample
        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        if not hasattr(self, 'resampler'):
            self.resampler = torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=16000).to(wavs.device)
        wavs = self.resampler(wavs)
        wav_lengths = (wav_lengths * 16000 / hparams['audio_sample_rate']).int()

        txt_tokens = sample['txt_tokens']
        txt_lens = sample['txt_lengths']
        txt_mask = sequence_mask(txt_lens, txt_tokens.shape[1])
        device = wavs.device
    
        bot = self.asr_text_tokenizer.encode('<|startoftranscript|>')[0]
        eot = self.asr_text_tokenizer.encode('<|endoftext|>')[0]
        txt_tokens[~txt_mask] = eot
        txt_tokens_with_bot_eot = torch.cat([
            torch.full((txt_tokens.shape[0], 1), bot, dtype=torch.long, device=device),
            txt_tokens,
            torch.full((txt_tokens.shape[0], 1), eot, dtype=torch.long, device=device)
        ], dim=1)
        txt_lens_with_bot_eot = txt_lens + 2
        txt_mask_with_bot_eot = sequence_mask(txt_lens_with_bot_eot, txt_tokens_with_bot_eot.shape[1])

        seg_dur = sample['seg_dur']     # [B, T]
        seg_offsets = torch.cumsum(seg_dur, dim=1)
        seg_offsets = seg_offsets // 2  # 50hz for wavlm
        seg_dur_ = seg_offsets[:, 1:] - seg_offsets[:, :-1]
        seg_dur[:, 0] = seg_offsets[:, 0]
        seg_dur[:, 1:] = seg_dur_

        inputs = {
            'wavs': wavs,
            'wav_mask': sequence_mask(wav_lengths),
            'txt_tokens': txt_tokens,
            'txt_mask': txt_mask,
            'txt_tokens_with_bot_eot': txt_tokens_with_bot_eot,
            'txt_mask_with_bot_eot': txt_mask_with_bot_eot,
            'seg_dur': seg_dur,
            'seg_mask': sequence_mask(sample['seg_dur_len'], sample['seg_dur'].shape[1]),
            'token_seg_id': sample['token_seg_id'],
        }

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model_outputs = self.model(inputs)

        losses_out['ce_loss'] = model_outputs['ce_loss']
        losses_out['align_loss'] = model_outputs['align_loss']
        losses_out['bd_agg_loss'] = model_outputs['bd_agg_loss']
        losses_out['mono_loss'] = model_outputs['mono_loss']

        with torch.no_grad():
            losses_out['monitor/gamma'] = model_outputs['gamma']
            losses_out['bs'] = wavs.shape[0]
            losses_out['ntokens'] = model_outputs['ntokens']

        return losses_out, model_out

    