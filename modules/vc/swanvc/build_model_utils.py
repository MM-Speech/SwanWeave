import os
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from typing import List, Tuple, Optional, Dict, Any
import torch
from pathlib import Path
from utils.commons.hparams import hparams, set_hparams
from utils.commons.ckpt_utils import load_ckpt, get_all_ckpts
from modules.tts.swanaudio.build_model_utils import build_vae
from modules.tts.semantic_encoders.vevo_tokenizer import VevoTokenExtractor

class DiTBuildModelMixin:
    def build_model(self):  # interface to BaseTask()
        self._build_model()
        self.vae.to(self.trainer.device)
        return {'trainable': [self.dit], 'others': []}
    
    def _build_model(self, attn_implementation='flash_attention_2'):  # interface to infer
        self.vae, self.hp_vae = build_vae(
            hparams.get('vae_ckpt'), 
            hparams.get('vae_latent_mean', None), hparams.get('vae_latent_std', None),
            hparams.get('latent_norm_mode', 'global'),
            attn_implementation=attn_implementation,
        )
        self.build_semantic_tokenizer()
        self.build_dit(hparams, attn_implementation)

    def build_semantic_tokenizer(self):
        self.semantic_tokenizer = VevoTokenExtractor.from_pretrained(
            cache_dir='pretrained_models/Vevo',
            device=self.trainer.device,
            download=False,
            repo_id="amphion/Vevo",
        )
        if hparams.get('semantic_token_type', 'content_style') == 'content_style':
            self.dit_vocab_size = self.semantic_tokenizer.content_style_codebook_size + 1
        elif hparams.get('semantic_token_type', 'content_style') == 'content':
            self.dit_vocab_size = self.semantic_tokenizer.content_codebook_size + 1

    def build_dit(self, hparams, attn_implementation='flash_attention_2'):
        from modules.vc.swanvc.dit import Diffusion, ModelArgs
        config = ModelArgs()
        if hparams.get('model_size', 'base') == 'small':
            print('| use small model')
            config.encoder_n_layers = 12
            config.encoder_n_heads = 12
            config.encoder_n_kv_heads = 12
            config.encoder_dim = 768 
        elif hparams.get('model_size', 'base') == '2b':
            print('| use 2b model')
            config.encoder_n_layers = 24
            config.encoder_n_heads = 24
            config.encoder_n_kv_heads = 12
            config.encoder_dim = 1536
            config.text_encoder_n_layers = 6
            config.text_encoder_n_heads = 24
        elif hparams.get('model_size', 'base') == '3b':
            print('| use 3b model')
            config.encoder_n_layers = 32
            config.encoder_n_heads = 24
            config.encoder_n_kv_heads = 24
            config.encoder_dim = 1536
            config.text_encoder_n_layers = 8
            config.text_encoder_n_heads = 24
        else:
            print('| use base model')
        config.encoder_ffn_mult = hparams.get('encoder_ffn_mult', 4.0)

        config.attn_implementation = attn_implementation

        config.in_channels = config.out_channels = hparams['latent_dim']
        config.vocab_size = self.dit_vocab_size
        config.cfg_mask_token = self.cfg_mask_token = self.dit_vocab_size - 1
        config.do_checkpoint = hparams.get('do_checkpoint', False) or hparams.get('gradient_checkpointing', False)

        self.dit = Diffusion(config)
        self.vae_stride = hparams.get('vae_stride', 8)
        return self.dit
