import os
import json
import importlib
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from copy import deepcopy
import torch
from utils.commons.hparams import hparams
from utils.commons.io import print_once
from modules.tts.scriptspeech.build_model_utils import (
    build_qwen3,
    shard_model_in_node,
    get_class_from_module,
    get_qwen_decoder_layer_classes,
)


def load_stable_audio_tools_autoencoder(model_config_path, ckpt_path, device, dtype):
    factory_mod = importlib.import_module("stable_audio_tools.models.factory")
    utils_mod = importlib.import_module("stable_audio_tools.models.utils")

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    model = factory_mod.create_model_from_config(model_config)
    utils_mod.copy_state_dict(model, utils_mod.load_ckpt_state_dict(ckpt_path))
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    sample_rate = int(model_config.get("sample_rate", getattr(model, "sample_rate", 44100)))
    audio_channels = int(model_config.get("audio_channels", getattr(model, "io_channels", 2)))
    downsampling_ratio = int(getattr(model, "downsampling_ratio"))
    return model, sample_rate, audio_channels, downsampling_ratio


def load_diffusers_autoencoder(vae_dir, device, dtype):
    from diffusers import AutoencoderOobleck

    model = AutoencoderOobleck.from_pretrained(vae_dir)
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    config = model.config
    sample_rate = int(getattr(config, "sampling_rate", getattr(config, "sample_rate", 44100)))
    audio_channels = int(getattr(config, "audio_channels", 2))
    ratios = list(getattr(config, "downsampling_ratios", [2, 4, 4, 8, 8]))
    downsampling_ratio = 1
    for ratio in ratios:
        downsampling_ratio *= int(ratio)
    return model, sample_rate, audio_channels, downsampling_ratio


def resolve_stable_audio_vae_backend(ckpt_path):
    vae_dir = hparams.get("stable_audio_vae_dir", "")
    if not vae_dir:
        vae_dir = os.path.dirname(ckpt_path) if ckpt_path else os.path.join("checkpoints", "vae")

    config_path = os.path.join(vae_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("_class_name") == "AutoencoderOobleck":
            return "diffusers", vae_dir
    return "stable_audio_tools", vae_dir


def build_stable_audio_vae(device, dtype=None):
    dtype_name = hparams.get("stable_audio_vae_dtype", "float32")
    if dtype is None:
        dtype = torch.float16 if dtype_name == "float16" else torch.float32
        if dtype_name == "bfloat16":
            dtype = torch.bfloat16

    model_config = hparams.get(
        "stable_audio_model_config",
        "stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json",
    )
    ckpt_path = hparams.get(
        "stable_audio_vae_ckpt",
        os.path.join("checkpoints", "vae", "diffusion_pytorch_model.safetensors"),
    )

    backend, vae_dir = resolve_stable_audio_vae_backend(ckpt_path)
    if backend == "diffusers":
        vae, sample_rate, channels, downsampling_ratio = load_diffusers_autoencoder(
            vae_dir,
            device,
            dtype,
        )
    else:
        vae, sample_rate, channels, downsampling_ratio = load_stable_audio_tools_autoencoder(
            model_config,
            ckpt_path,
            device,
            dtype,
        )
    if int(channels) != 2:
        raise ValueError(f"Expected Stable Audio VAE to be stereo, got {channels} channels")
    return vae, {
        "backend": backend,
        "sample_rate": sample_rate,
        "channels": channels,
        "downsampling_ratio": downsampling_ratio,
        "vae_dir": vae_dir,
        "model_config": model_config,
        "ckpt_path": ckpt_path,
        "dtype": str(dtype).replace("torch.", ""),
    }


class DiTBuildModelMixin:
    def build_model(self):  # interface to BaseTask()
        self._build_model()

        if not hparams.get("use_fsdp"):
            cast_result = self.dit.cast_safe_params_to_bf16()
            print_once(
                f"| DiT: Cast {cast_result['bf16_params'] / 1_000_000:.3f} params to bf16, "
                f"remaining {cast_result['fp32_params'] / 1_000_000:.3f} params in fp32"
            )

        if hparams.get("use_ema", False):
            print_once(f"| Building EMA model with decay={self.config.ema_decay} ...")
            self.ema_model = deepcopy(self.dit)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_model.to(self.trainer.device)
            return {'trainable': [self.dit, self.ema_model], 'others': []}

        return {'trainable': [self.dit], 'others': []}
    
    def _build_model(self, attn_implementation='sdpa'):  # interface to infer
        self.build_stable_audio_vae()
        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        self.build_dit(hparams, attn_implementation)

        if hparams.get('use_caption', True):
            self.build_goku_text_encoder()

    def build_stable_audio_vae(self):
        device = self.trainer.device
        self.vae, self.hp_vae = build_stable_audio_vae(device)
        self.vae_sample_rate = int(self.hp_vae["sample_rate"])
        self.vae_channels = int(self.hp_vae["channels"])
        self.vae_downsampling_ratio = int(self.hp_vae["downsampling_ratio"])
        self.vae_backend = str(self.hp_vae.get("backend", "stable_audio_tools"))
        print(
            f"| Stable Audio VAE: backend={self.vae_backend}, sample_rate={self.vae_sample_rate}, "
            f"channels={self.vae_channels}, downsampling={self.vae_downsampling_ratio}"
        )

    def build_dit(self, hparams, attn_implementation='sdpa'):
        from modules.tts.spat_edit.dit import Diffusion, ModelArgs
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

        config.attn_implementation = hparams.get('attn_implementation', attn_implementation)

        config.in_channels = config.out_channels = hparams.get('latent_dim', 128)
        config.train_base = hparams.get('train_base', True)
        config.vocab_size = self.dit_vocab_size
        config.caption_dim = hparams.get('caption_encoder_dim', 1024)
        config.cfg_mask_text_token = self.cfg_mask_text_token = self.dit_text_tokenizer.encode('<MASK>')[0]
        config.text_fill_token = self.text_fill_token = self.dit_text_tokenizer.encode('<FILL>')[0]
        config.do_checkpoint = hparams.get('do_checkpoint', False) or hparams.get('gradient_checkpointing', False)
        # config.torch_compile_enabled = hparams.get('torch_compile', False) if torch.__version__.split(".")[0] == '2' else False
        config.torch_compile_enabled = False
        config.use_caption_text_mark = hparams.get('use_caption_text_mark', False)
        config.use_caption_pool_in_adaln = hparams.get('use_caption_pool_in_adaln', False)
        config.use_dynamic_cross_gate = hparams.get('use_dynamic_cross_gate', False)
        config.use_spk_mask = False
        config.use_bgm_flag = hparams.get('use_bgm_flag', False)
        config.use_quality_flag = hparams.get('use_quality_flag', False)
        config.use_gated_attention = hparams.get('use_gated_attention', False)
        config.encoder_ffn_mult = hparams.get('encoder_ffn_mult', 4.0)

        config.use_moe_ffn = hparams.get('use_moe_ffn', False)
        config.moe_p = hparams.get('moe_p', 0.7)
        config.moe_num_routed = hparams.get('moe_num_routed', 8)
        config.moe_num_shared = hparams.get('moe_num_shared', 0)
        config.moe_num_task_experts = hparams.get('moe_num_task_experts', 4)
        config.moe_task_p = hparams.get('moe_task_p', config.moe_p)
        config.moe_num_null = hparams.get('moe_num_null', 4)
        config.moe_use_gumbel = hparams.get('moe_use_gumbel', False)
        config.moe_gumbel_tau_start = hparams.get('moe_gumbel_tau_start', 1.0)
        config.moe_gumbel_tau_end = hparams.get('moe_gumbel_tau_end', 0.3)
        config.moe_gumbel_tau_anneal_steps = hparams.get('moe_gumbel_tau_anneal_steps', 200_000)
        config.moe_expert_dropout = hparams.get('moe_expert_dropout', 0.0)
        config.moe_use_bias_balance = hparams.get('moe_use_bias_balance', False)
        config.moe_bias_update_rate = hparams.get('moe_bias_update_rate', 1e-3)
        config.moe_bias_momentum = hparams.get('moe_bias_momentum', 0.9)
        config.moe_bias_clamp = hparams.get('moe_bias_clamp', 3.0)
        config.moe_load_balance_loss_coef = hparams.get('moe_load_balance_loss_coef', 1e-2)
        config.moe_router_z_loss_coef = hparams.get('moe_router_z_loss_coef', 1e-3)
        config.moe_null_loss_coef = hparams.get('moe_null_loss_coef', config.moe_load_balance_loss_coef)
        config.moe_capacity_factor_min = hparams.get('moe_capacity_factor_min', 1.0)
        config.moe_capacity_factor_max = hparams.get('moe_capacity_factor_max', 2.0)
        config.moe_overflow_drop = hparams.get('moe_overflow_drop', True)
        config.moe_use_t_budget = hparams.get('moe_use_t_budget', True)
        config.moe_p_min = hparams.get('moe_p_min', 0.4)
        config.moe_p_max = hparams.get('moe_p_max', 0.95)
        config.moe_null_logit_bias_min = hparams.get('moe_null_logit_bias_min', -2.0)
        config.moe_null_logit_bias_max = hparams.get('moe_null_logit_bias_max', 0.0)
        config.decoder_sparse_step = hparams.get("decoder_sparse_step", 1)
        config.mlp_only_layers = hparams.get("mlp_only_layers", [])
        config.moe_intermediate_size = hparams.get("moe_intermediate_size", None) 
        config.output_router_logits = hparams.get("output_router_logits", False)
        config.moe_use_ec = hparams.get("moe_use_ec", False)
        config.ec_early_dense_moe_layers = hparams.get("ec_early_dense_moe_layers", 2)
        config.ec_early_capacity_boost_moe_layers = hparams.get("ec_early_capacity_boost_moe_layers", 2)
        config.ec_early_capacity_ratio = hparams.get("ec_early_capacity_ratio", None)
        config.ec_default_step_ratio = hparams.get("ec_default_step_ratio", 0.5)

        # Engram
        config.use_engram = hparams.get("use_engram", False)
        config.engram_target = hparams.get("engram_target", "caption")
        # 注意：dataclass里建议是 Tuple[int,...]，但hparams经常给list；这里统一转tuple
        engram_orders = hparams.get("engram_orders", (2, 3))
        if isinstance(engram_orders, list):
            engram_orders = tuple(engram_orders)
        config.engram_orders = engram_orders
        config.engram_num_heads = hparams.get("engram_num_heads", 4)
        config.engram_table_size = hparams.get("engram_table_size", 262144)  # 2^18
        config.engram_head_dim = hparams.get("engram_head_dim", 64)
        config.engram_dropout = hparams.get("engram_dropout", 0.0)
        config.engram_gate_bias = hparams.get("engram_gate_bias", -4.0)
        config.engram_conv_kernel = hparams.get("engram_conv_kernel", 3)

        self.dit = Diffusion(config)
        self.dit.text_tokenizer = self.dit_text_tokenizer
        self.vae_stride = hparams.get('vae_stride', 4)
        return self.dit

    def build_dit_text_tokenizer(self):
        if hparams.get('use_cosyvoice2_text_tokenizer',False):
            from utils.text.cosyvoice2_tokenizer import get_tokenizer
            text_tokenizer = get_tokenizer(multilingual=True, num_languages=100)
            vocab_size = text_tokenizer.encoding.n_vocab
            return text_tokenizer, vocab_size
        else:
            from transformers import AutoTokenizer
            qwen_name = hparams.get(
                "pretrained_text_encoder_qwen",
                "checkpoints/Qwen3-0.6B",
            )
            text_tokenizer = AutoTokenizer.from_pretrained(qwen_name)
            text_tokenizer.add_tokens([
                '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>', 
                '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>', 
                '<S1>', '</S1>','<S2>', '</S2>', 
                '<Audio>', '</Audio>','<BGM>','</BGM>', 
                '<W>', '</W>', '<FILL>', '<PAD>',
                '<S3>', '</S3>','<S4>', '</S4>', 
                '<|laughter|>', '<|breathe|>', '<|cry|>',
                '<ENV>','</ENV>', 
            ], special_tokens=True)
            vocab_size = len(text_tokenizer)
            return text_tokenizer, vocab_size

    def build_goku_text_encoder(self):
        from transformers import AutoTokenizer

        dtype = torch.bfloat16
        qwen_name = hparams.get(
            'pretrained_text_encoder_qwen',
            '/mnt/bn/sa-ag-data/leike/spatial_edit/ScriptSpeech/checkpoints/Qwen3-0.6B',
        )
        if 'qwen2' in qwen_name.lower():
            from transformers import Qwen2Model
            self.goku_text_encoder = Qwen2Model.from_pretrained(qwen_name, torch_dtype=dtype)
            self.goku_text_encoder.requires_grad_(False).eval()
            self.goku_tokenizer = AutoTokenizer.from_pretrained(
                qwen_name,
                max_length=hparams.get('text_max_token_length', 256),
                revision=None,
            )
            self.goku_tokenizer.padding_side = "left"
            layer_cls = get_class_from_module(
                "transformers.models.qwen2.modeling_qwen2",
                "Qwen2DecoderLayer",
            )
            if int(os.environ.get("WORLD_SIZE", 1)) > 1:
                self.goku_text_encoder = shard_model_in_node(
                    self.goku_text_encoder,
                    layer_cls,
                    int(os.environ.get("LOCAL_RANK", 0)),
                )
        elif 'qwen3' in qwen_name.lower():
            lm, tok = build_qwen3(hparams, qwen_name, dtype=dtype)
            self.goku_text_encoder = lm.model
            self.goku_text_encoder.requires_grad_(False).eval()
            self.goku_tokenizer = tok
            self.goku_tokenizer.padding_side = "left"
            layer_classes = get_qwen_decoder_layer_classes(qwen_name)
            if int(os.environ.get("WORLD_SIZE", 1)) > 1 and layer_classes:
                self.goku_text_encoder = shard_model_in_node(
                    self.goku_text_encoder,
                    layer_classes[0],
                    int(os.environ.get("LOCAL_RANK", 0)),
                )
        else:
            raise ValueError(
                f"Unknown caption encoder backbone: {qwen_name}. Expect the name to contain 'Qwen2' or 'Qwen3'."
            )

        self.goku_text_encoder.to(self.trainer.device)
