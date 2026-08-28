import math
import os
import soundfile as sf
import random
from attrdictionary import AttrDict
from copy import deepcopy
import time

import torch
import torchaudio
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

from utils.commons.ckpt_utils import load_ckpt, load_ckpt_moe
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.hparams import hparams
from utils.commons.os_utils import kill_void
from utils.commons.io import print_once
from utils.commons.tensor_utils import move_to_cpu, convert_to_np
from utils.commons.dataset_utils import collate_xd
from utils.nn.model_utils import unwrap_model
from utils.nn.seq_utils import sequence_mask, add_prefix, remove_prefix
from utils.audio.transform import batch_resample

from modules.vc.swanvc.build_model_utils import DiTBuildModelMixin
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from tasks.tts.swan_task import SwanDiTTask


class SwanVCDiTTask(DiTBuildModelMixin, SwanDiTTask):

    def build_model(self):
        self._build_model(attn_implementation='flash_attention_2')
        self.vae.to(self.trainer.device, torch.bfloat16)

        if not hparams.get('use_fsdp'):
            cast_result = self.dit.cast_safe_params_to_bf16()
            print_once(f"| DiT: Cast {cast_result['bf16_params'] / 1_000_000:.3f} params to bf16, remaining {cast_result['fp32_params'] / 1_000_000:.3f} params in fp32")

        if torch.__version__.split(".")[0] == '2' and hparams.get("torch_compile", False):
            self.torch_compile_enabled = True
            self.vae.encode_latent = torch.compile(
                self.vae.encode_latent,
                fullgraph=False, dynamic=False, mode="default"
            )
        
        if hparams.get('use_ema', False):
            print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
            self.ema_model = deepcopy(self.dit)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_model.to(self.trainer.device)
            
            return {'trainable': [self.dit, self.ema_model], 'others': []}

        return {'trainable': [self.dit], 'others': []}

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
            kill_void()
        loss_output, model_out = self.run_model(sample)

        loss_weights = {
            'diff_loss': 1.0,
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
    
    def encode_semantic(self, wavs_16k, wavs_16k_lengths, device):
        vector_type = 'both'
        if hparams.get('semantic_token_type', 'content_style') == 'content_style':
            vector_type = 'content_style'
        elif hparams.get('semantic_token_type', 'content_style') == 'content':
            vector_type = 'content'
        results = self.semantic_tokenizer.extract_from_16k_batch_chunked(
            wavs=wavs_16k,
            wav_lengths=wavs_16k_lengths,
            batch_size=256,
            vector_type=vector_type,
            reduce_content=False,
            reduce_content_style=False
        )
        if hparams.get('semantic_token_type', 'content_style') == 'content_style':
            semantic_tokens = [torch.from_numpy(result.content_style_ids).long() for result in results]
        elif hparams.get('semantic_token_type', 'content_style') == 'content':
            semantic_tokens = [torch.from_numpy(result.content_ids).long() for result in results]
        semantic_tokens_lengths = torch.LongTensor([token.shape[0] for token in semantic_tokens]).to(device)
        semantic_tokens = collate_xd(semantic_tokens).to(device)
        return semantic_tokens, semantic_tokens_lengths

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
        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']
        device = wavs.device

        ctx_wavs = sample["ctx_wavs"]
        ctx_wav_lengths = sample["ctx_wav_lengths"]
        ctx_mask = sample["ctx_mask"]
        if len(ctx_mask.shape) == 2:
            ctx_mask = ctx_mask[:, :, None]

        # time_start = time.time()
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            vae_encode_chunk_sec = hparams.get('vae_encode_chunk_sec', 20)
            vae_encode_max_batch_size = hparams.get('vae_encode_max_batch_size', 128)
            lat, lat_ctx = self._encode_wav_and_ctx_latents(
                wavs=wavs,
                wav_lengths=wav_lengths,
                ctx_wavs=ctx_wavs,
                ctx_wav_lengths=ctx_wav_lengths,
                ctx_mask=ctx_mask,
                use_ref=True,
                chunk_sec=vae_encode_chunk_sec,
                max_batch_size=vae_encode_max_batch_size,
            )

            if not hasattr(self, 'resamplers'):
                self.resamplers = {}
            if self.semantic_tokenizer.token_sample_rate not in self.resamplers:
                self.resamplers[self.semantic_tokenizer.token_sample_rate] = \
                    torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=self.semantic_tokenizer.token_sample_rate).to(device)
            wavs_16k = self.resamplers[self.semantic_tokenizer.token_sample_rate](wavs.float())

            wavs_16k_lengths = wav_lengths * self.semantic_tokenizer.token_sample_rate // hparams['audio_sample_rate']
            ctx_wavs_16k_lengths = ctx_wav_lengths * self.semantic_tokenizer.token_sample_rate // hparams['audio_sample_rate']
            tgt_wavs_16k = remove_prefix(wavs_16k[..., None], ctx_wavs_16k_lengths, wavs_16k_lengths - ctx_wavs_16k_lengths)[..., 0]
            tgt_semantic_tokens, tgt_semantic_tokens_lengths = self.encode_semantic(tgt_wavs_16k, wavs_16k_lengths - ctx_wavs_16k_lengths, device)
            ctx_semantic_tokens, ctx_semantic_tokens_lengths = self.encode_semantic(wavs_16k, ctx_wavs_16k_lengths, device)
            semantic_tokens = add_prefix(ctx_semantic_tokens, ctx_semantic_tokens_lengths, tgt_semantic_tokens, tgt_semantic_tokens_lengths)
            semantic_tokens_lengths = tgt_semantic_tokens_lengths + ctx_semantic_tokens_lengths
            semantic_tokens_mask = sequence_mask(semantic_tokens_lengths, maxlen=semantic_tokens.shape[1])

            assert semantic_tokens.shape[1] == 2 * lat.shape[1]

        loss_mask = sequence_mask(lat_lens, maxlen=lat.shape[1])[:, :, None] * (1 - ctx_mask)

        # CFG Mask on latent context
        lat_cfg_mask = torch.rand_like(lat_ctx[:, :1, 0].float())
        lat_cfg_mask = (lat_cfg_mask < 0.15).long()
        lat_ctx = (lat_ctx * ctx_mask * (1 - lat_cfg_mask)[:, :, None])

        # CFG Mask on text tokens
        semantic_cfg_mask = torch.rand_like(semantic_tokens[:, :1].float())
        semantic_cfg_mask = (semantic_cfg_mask < 0.15).long()
        semantic_tokens = semantic_tokens * (1 - semantic_cfg_mask) + self.cfg_mask_token * semantic_cfg_mask

        inputs = {
            "semantic_tokens": semantic_tokens.long(),
            "semantic_lens": semantic_tokens_lengths,
            "semantic_mask": semantic_tokens_mask,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,
            "ctx_mask": ctx_mask,
        }

        if not infer:
            # time_start = time.time()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                model_out, target = self.dit(inputs)
            # print(f'| dit time: {time.time() - time_start}')

            # 主 diffusion loss
            if loss_mask.sum() == 0:
                return {}, model_out

            loss = F.mse_loss(model_out.float(), target.float(), reduction='none')
            loss = (loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]
            losses_out['diff_loss'] = loss

            losses_out['bs'] = loss_mask.shape[0]
            losses_out['ntokens'] = sum(lat_lens)
            return losses_out, model_out
        else:
            return losses_out, model_out
