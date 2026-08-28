import os
import soundfile as sf
import random
from attrdictionary import AttrDict
from copy import deepcopy
import threading
from concurrent.futures import ThreadPoolExecutor

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
from utils.nn.seq_utils import sequence_mask

from modules.tts.swanaudio.build_model_utils import DiTBuildModelMixin
from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin


class SwanBaseTask(FastDatasetMixin, ScriptSpeechBaseTask):
    def __init__(self):
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
        self.config = AttrDict(hparams)

        super().__init__()

    def fsdp_wrap_policy(self):
        import modules.commons.hf.transformer
        import modules.commons.hf.transformer_moe
        import modules.commons.hf.transformer_dit
        import modules.commons.hf.transformer_dit_moe
        import modules.tts.swanaudio.swan_dit_moe
        from modules.tts.scriptspeech.build_model_utils import get_qwen_decoder_layer_classes
        qwen_name = hparams.get('pretrained_text_encoder_qwen', '')
        qwen_layer_classes = get_qwen_decoder_layer_classes(qwen_name)

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                modules.commons.hf.transformer.EncoderLayer,
                modules.commons.hf.transformer.DecoderLayer,
                modules.commons.hf.transformer_moe.EncoderLayer,
                modules.commons.hf.transformer_moe.DecoderLayer,
                modules.commons.hf.transformer_dit.EncoderLayer,
                modules.commons.hf.transformer_dit_moe.EncoderLayer,
                modules.tts.swanaudio.swan_dit_moe.EncoderLayer,
                *qwen_layer_classes,
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        if hparams.get('use_ema', False):
            self.ema_update(self.ema_model, self.dit, self.config.ema_decay)
                
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
            if hparams.get('use_ema', False):
                load_ckpt(self.ema_model, hparams['load_ckpt'], 'dit', strict=False, mmap=True)

    def build_optimizer(self):
        if not hparams.get('disable_weight_decay_on_bias_and_norm_and_embed', False):
            optimizer = AdamW(unwrap_model(self.dit).parameters(), **self.config.optimizer)
        else:
            
            def has_name(names, param_name):
                if not isinstance(names, list):
                    names = [names]
                for name in names:
                    if name in param_name:
                        return True
                return False
                
            decay_params = []
            no_decay_params = []
            for name, param in unwrap_model(self.dit).named_parameters():
                if not param.requires_grad:
                    continue
                if param.dim() == 1 or has_name(['bias', 'norm', 'text_embedder', 'tone_embed', 'ph_embed'], name):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
                    
            print_once(f"| Weight decay is canceld for {len(no_decay_params)} params")

            optimizer_groups = [
                {'params': decay_params, 'weight_decay': self.config.optimizer.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_groups, **self.config.optimizer)

        return optimizer

    def fsdp_optm2model(self):
        return [self.dit]


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

        if hparams.get('use_caption_text_mark', False):
            caption_text_mark = build_audio_mask_from_ids(
                input_ids=input_ids,
                attention_mask=attention_masks,
                tokenizer=self.goku_tokenizer,
            )
        else:
            caption_text_mark = None

        return encoder_hidden_states, caption_text_mark, attention_masks, input_ids

    def prepare_text_online(self, sample):
        from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import (
            build_spk_mask_from_text_tokens,
            _get_sx_token_patterns,
            augment_text_with_pinyin_s1s2_safe,
        )
        from data_gen.asr.pipeline import ASRPipeline

        if not hasattr(self, 'forced_aligner') or self.forced_aligner is None:
            self.forced_aligner = ASRPipeline(
                asr_backend='sensevoice',
                punc_backend='aligner_2tower_v5',
                device=self.trainer.device,
                precision=torch.bfloat16,
                aligner_ckpt='checkpoints/260225_aligner_2tower_v5',
                special_token_ignore_literals=["<S1>", "</S1>", "<S2>", "</S2>", "<S3>", "</S3>", "<S4>", "</S4>"],
                restore_special_tokens=True
            )
        if not hasattr(self, 'sx_patterns') or self.sx_patterns is None:
            self.sx_patterns = _get_sx_token_patterns(self.dit_text_tokenizer)

        sr_aligner = 16000

        if not hasattr(self, 'resamplers'):
            self.resamplers = {}
        if sr_aligner not in self.resamplers:
            self.resamplers[sr_aligner] = torchaudio.transforms.Resample(hparams['audio_sample_rate'], sr_aligner, dtype=torch.float32).to(self.trainer.device)

        wavs = sample['wavs']
        wav_lens = sample['wav_lengths']
        batch_size = int(wavs.shape[0])

        texts = sample.get('orig_text', None)
        if texts is None:
            texts = sample.get('text', None)

        texts = [("" if t is None else (t if isinstance(t, str) else str(t))) for t in texts]
        valid_indices = [i for i, t in enumerate(texts) if t.strip() != ""]

        if len(valid_indices) > 0:
            texts_to_align = [texts[i] for i in valid_indices]

            idx_tensor = torch.tensor(valid_indices, device=wavs.device, dtype=torch.long)

            wavs_valid_full = wavs.index_select(0, idx_tensor)

            if torch.is_tensor(wav_lens):
                valid_wav_lens_t = wav_lens.index_select(0, idx_tensor).long()
            else:
                valid_wav_lens_t = torch.tensor(
                    [int(wav_lens[i]) for i in valid_indices],
                    device=wavs.device,
                    dtype=torch.long,
                )

            max_valid_wav_len = int(valid_wav_lens_t.max().item()) if valid_wav_lens_t.numel() > 0 else 0
            max_valid_wav_len = max(int(max_valid_wav_len), 1)

            wavs_valid = wavs_valid_full[:, :max_valid_wav_len].float()

            with torch.no_grad():
                wavs_16k_valid = self.resamplers[sr_aligner](wavs_valid)

            valid_wav_lens_16k = (
                valid_wav_lens_t.to(torch.float32) * float(sr_aligner) / float(hparams['audio_sample_rate'])
            ).long()
            valid_wav_lens_16k = torch.clamp(valid_wav_lens_16k, min=0, max=int(wavs_16k_valid.shape[1]))

            with torch.no_grad():
                aligned_out = self.forced_aligner.process(
                    (wavs_16k_valid, valid_wav_lens_16k),
                    texts_to_align
                )

            texts_aligned = aligned_out.get('pause_punct_texts', None)
            if isinstance(texts_aligned, (list, tuple)) and len(texts_aligned) == len(valid_indices):
                for idx_in_batch, t_new in zip(valid_indices, texts_aligned):
                    if t_new is None:
                        continue
                    t_new = t_new if isinstance(t_new, str) else str(t_new)
                    if t_new.strip() != "":
                        texts[idx_in_batch] = t_new

        sample['text'] = texts

        text_tokens_lst = []
        spk_mask_lst = []
        for t in texts:
            if t.strip() == "":
                text_tokens = torch.zeros((0,), dtype=torch.long)
                spk_mask = torch.zeros((0,), dtype=torch.long)
            else:
                if hparams.get('mix_text_pinyin', {}).get('enable', False):
                    t = augment_text_with_pinyin_s1s2_safe(t, hparams)
                text_tokens = self.dit_text_tokenizer.encode(t)
                text_tokens = torch.tensor(text_tokens).long()
                spk_mask = build_spk_mask_from_text_tokens(text_tokens, self.sx_patterns)

            text_tokens_lst.append(text_tokens)
            spk_mask_lst.append(spk_mask)

        max_txt_len = max(int(x.numel()) for x in text_tokens_lst) if len(text_tokens_lst) > 0 else 0
        collate_max_len = max(1, max_txt_len)

        sample['txt_tokens'] = collate_xd(text_tokens_lst, max_len=collate_max_len).to(self.trainer.device)
        sample['txt_lengths'] = torch.LongTensor([int(t.numel()) for t in text_tokens_lst]).to(self.trainer.device)
        sample['spk_mask'] = collate_xd(spk_mask_lst, max_len=collate_max_len).to(self.trainer.device)

        return sample

    def on_train_start(self):
        super().on_train_start()
        self._pipeline_enabled = bool(hparams.get("pipeline_cache_future", False))

        self._prefetch_executor = None
        self._prefetch_future = None
        self._prefetch_stream = None

        self._vae_lock = threading.Lock()
        self._aligner_lock = threading.Lock()

        if self._pipeline_enabled:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=int(hparams.get("pipeline_cache_workers", 1))
            )
            if torch.cuda.is_available():
                self._prefetch_stream = torch.cuda.Stream(device=torch.device("cuda", int(self.trainer.root_gpu)))

    def on_train_end(self):
        try:
            if getattr(self, "_prefetch_future", None) is not None:
                try:
                    self._prefetch_future.cancel()
                except Exception:
                    pass
            if getattr(self, "_prefetch_executor", None) is not None:
                self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        finally:
            super().on_train_end()

    @torch.no_grad()
    def _prefetch_prepare_sample(self, sample):
        if torch.cuda.is_available():
            # 单机 8 卡：每个进程应该固定自己的 root_gpu
            torch.cuda.set_device(int(self.trainer.root_gpu))

        do_aligner = bool(hparams.get("pipeline_prefetch_aligner", True)) and bool(hparams.get("online_text_alignment_task", False))
        do_vae = bool(hparams.get("pipeline_prefetch_vae", True))

        # 统一：prefetch 的 GPU 工作尽量都放到 prefetch_stream 上
        if self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                # 如果 batch 刚 move_to_cuda 完，数据拷贝可能还在 default stream 上排队
                # 让 prefetch_stream 等 default stream，避免读到未完成拷贝的数据
                self._prefetch_stream.wait_stream(torch.cuda.default_stream(int(self.trainer.root_gpu)))

                if do_aligner and ('txt_tokens' not in sample):
                    # 你只有 1 个 worker，其实 lock 不必须；保守起见保留
                    with self._aligner_lock:
                        sample = self.prepare_text_online(sample)

                if do_vae and (("lat" not in sample) or ("lat_ctx" not in sample)):
                    wavs = sample["wavs"].float()
                    wav_lengths = sample["wav_lengths"]
                    ctx_wavs = sample["ctx_wavs"]
                    ctx_wav_lengths = sample["ctx_wav_lengths"]

                    with self._vae_lock:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                            lat, _ = self.vae.encode_latent(wavs, wav_lengths, chunk_sec=20, max_batch_size=128)
                            lat_ctx, _ = self.vae.encode_latent(ctx_wavs, ctx_wav_lengths, chunk_sec=20, max_batch_size=128)
                            lat_ctx = torch.nn.functional.pad(
                                lat_ctx, (0, 0, 0, lat.size(1) - lat_ctx.size(1)), mode='constant', value=0
                            )

                    sample["lat"] = lat
                    sample["lat_ctx"] = lat_ctx

                ev = torch.cuda.Event()
                ev.record(self._prefetch_stream)

            sample["_prefetch_event"] = ev
            return sample

        # fallback：没有 stream 就同步跑（基本只用于 cpu）
        if do_aligner and ('txt_tokens' not in sample):
            with self._aligner_lock:
                sample = self.prepare_text_online(sample)

        if do_vae and (("lat" not in sample) or ("lat_ctx" not in sample)):
            wavs = sample["wavs"].float()
            wav_lengths = sample["wav_lengths"]
            ctx_wavs = sample["ctx_wavs"]
            ctx_wav_lengths = sample["ctx_wav_lengths"]
            with self._vae_lock:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    lat, _ = self.vae.encode_latent(wavs, wav_lengths, chunk_sec=20, max_batch_size=128)
                    lat_ctx, _ = self.vae.encode_latent(ctx_wavs, ctx_wav_lengths, chunk_sec=20, max_batch_size=128)
                    lat_ctx = torch.nn.functional.pad(
                        lat_ctx, (0, 0, 0, lat.size(1) - lat_ctx.size(1)), mode='constant', value=0
                    )
            sample["lat"] = lat
            sample["lat_ctx"] = lat_ctx

        return sample

    def _launch_prefetch(self, sample):
        return self._prefetch_executor.submit(self._prefetch_prepare_sample, sample)

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if not self._pipeline_enabled:
            if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
                kill_void()
            loss_output, model_out = self.run_model(sample)

            base_moe_w = hparams.get('moe_aux_loss_weight', 1.0)
            anneal_steps = hparams.get('moe_aux_anneal_steps', 200000)
            if anneal_steps > 0 and base_moe_w > 0:
                ratio = 1.0 - min(self.global_step, anneal_steps) / float(anneal_steps)
                ratio = max(0.2, ratio)
                moe_w = base_moe_w * ratio
            else:
                moe_w = base_moe_w

            loss_weights = {'diff_loss': 1.0, 'moe_aux_loss': moe_w}
            total_loss = sum([
                loss_weights.get(k, 1.0) * v
                for k, v in loss_output.items()
                if isinstance(v, torch.Tensor) and v.requires_grad
            ])
            return total_loss, loss_output

        # ===== pipeline 模式：永远落后 1 step =====
        if self._prefetch_future is None:
            # 第一步：只启动预取，不训练
            self._prefetch_future = self._launch_prefetch(sample)
            return None

        # 取回上一步准备好的 sample_{n-1}
        ready_sample = self._prefetch_future.result()

        # 立刻启动当前 sample_n 的预取（和下面的 dit 训练并行）
        self._prefetch_future = self._launch_prefetch(sample)

        # 如果 VAE 在另一个 stream 上算的，这里等 event，避免读未完成 lat
        ev = ready_sample.pop("_prefetch_event", None)
        if ev is not None and torch.cuda.is_available():
            torch.cuda.current_stream().wait_event(ev)

        if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
            kill_void()

        loss_output, model_out = self.run_model(ready_sample)

        base_moe_w = hparams.get('moe_aux_loss_weight', 1.0)
        anneal_steps = hparams.get('moe_aux_anneal_steps', 200000)
        if anneal_steps > 0 and base_moe_w > 0:
            ratio = 1.0 - min(self.global_step, anneal_steps) / float(anneal_steps)
            ratio = max(0.2, ratio)
            moe_w = base_moe_w * ratio
        else:
            moe_w = base_moe_w

        loss_weights = {'diff_loss': 1.0, 'moe_aux_loss': moe_w}
        total_loss = sum([
            loss_weights.get(k, 1.0) * v
            for k, v in loss_output.items()
            if isinstance(v, torch.Tensor) and v.requires_grad
        ])

        if self.trainer.proc_rank == 0 and self.global_step < 10:
            save_dir = f'{hparams["work_dir"]}/sample_batches/step_{self.global_step}'
            os.makedirs(save_dir, exist_ok=True)
            dump_sample = convert_to_np(move_to_cpu(ready_sample))
            for i in range(dump_sample['nsamples']):
                sf.write(f"{save_dir}/{i}.wav", dump_sample['wavs'][i, :dump_sample['wav_lengths'][i]], 24000, 'PCM_16')
                sf.write(f"{save_dir}/{i}_ctx.wav", dump_sample['ctx_wavs'][i, :dump_sample['wav_lengths'][i]], 24000, 'PCM_16')
                np.save(f"{save_dir}/{i}.npy", dump_sample, allow_pickle=True)
            del dump_sample['wavs']
            del dump_sample['ctx_wavs']

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

        # ===== VAE：优先用预取结果 =====
        if "lat" in sample and isinstance(sample["lat"], torch.Tensor):
            lat = sample["lat"]
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    lat, _ = self.vae.encode_latent(wavs, wav_lengths, chunk_sec=20, max_batch_size=128)

        if use_ref:
            if "lat_ctx" in sample and isinstance(sample["lat_ctx"], torch.Tensor):
                lat_ctx = sample["lat_ctx"]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                        lat_ctx, _ = self.vae.encode_latent(ctx_wavs, ctx_wav_lengths, chunk_sec=20, max_batch_size=128)
                        lat_ctx = torch.nn.functional.pad(lat_ctx, (0, 0, 0, lat.size(1) - lat_ctx.size(1)), mode='constant', value=0)
        else:
            lat_ctx = torch.zeros_like(lat)

        # ===== aligner：如果预取已经塞了 txt_tokens，就不再跑 =====
        if hparams.get('online_text_alignment_task', False) and 'txt_tokens' not in sample:
            sample = self.prepare_text_online(sample)

        if random.random() < 0.001:
            print('| text sample', text[0])

        # text tokenize
        if 'txt_tokens' not in sample:
            text_inputs = self.dit_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
            txt_tokens = text_inputs['input_ids']
            txt_mask = text_inputs['attention_mask'].bool()
            txt_tokens[~txt_mask] = self.cfg_mask_text_token
            txt_lens = txt_mask.int().sum(1)
        else:
            txt_tokens = sample['txt_tokens']
            txt_lens = sample['txt_lengths']
            if isinstance(txt_tokens, torch.Tensor) and txt_tokens.device != device:
                txt_tokens = txt_tokens.to(device, non_blocking=True)
            if isinstance(txt_lens, torch.Tensor) and txt_lens.device != device:
                txt_lens = txt_lens.to(device, non_blocking=True)
            txt_mask = sequence_mask(txt_lens, maxlen=txt_tokens.shape[1])
            txt_tokens = txt_tokens.clone()
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
