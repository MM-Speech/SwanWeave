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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from modules.tts.scriptspeech.build_model_utils import (
    build_qwen3,
    get_in_node_device_mesh,
    shard_model_in_node,
    get_class_from_module,
    get_qwen_decoder_layer_classes,
)


def _resolve_ckpt_path_for_stats(vae_ckpt: Optional[str]) -> Optional[str]:
    if not vae_ckpt:
        return None
    if os.path.isfile(vae_ckpt) and vae_ckpt.endswith(".ckpt"):
        return vae_ckpt
    if not os.path.isdir(vae_ckpt):
        return None
    ckpt_paths = get_all_ckpts(vae_ckpt)
    if len(ckpt_paths) > 0:
        return ckpt_paths[0]
    return None

def _derive_latent_stats_path_from_ckpt(vae_ckpt: Optional[str]) -> Optional[str]:
    resolved_ckpt = _resolve_ckpt_path_for_stats(vae_ckpt)
    if resolved_ckpt is None:
        return None
    ckpt_dir = os.path.dirname(resolved_ckpt)
    ckpt_name = os.path.basename(resolved_ckpt)
    if not ckpt_name.startswith("model_ckpt_"):
        return None
    stats_name = ckpt_name.replace("model_ckpt_", "latent_stats_", 1)
    stats_name = os.path.splitext(stats_name)[0] + ".npz"
    stats_path = os.path.join(ckpt_dir, stats_name)
    if os.path.exists(stats_path):
        return stats_path
    return None

def build_vae(vae_ckpt, vae_latent_mean=None, vae_latent_std=None, latent_norm_mode='global', attn_implementation='flash_attention_2'):
    if 'v9' in vae_ckpt:
        from modules.vae.wavvae_v9 import build_wavvae
    elif 'v10' in vae_ckpt:
        from modules.vae.wavvae_v10 import build_wavvae
    elif 'v11' in vae_ckpt:
        from modules.vae.wavvae_v11 import build_wavvae
    if vae_ckpt.endswith('.ckpt'):
        hp_vae = set_hparams(os.path.join(Path(vae_ckpt).parent, 'config.yaml'), global_hparams=False)
    else:
        hp_vae = set_hparams(os.path.join(vae_ckpt, 'config.yaml'), global_hparams=False)
    if vae_latent_mean is not None and vae_latent_std is not None and hp_vae.get('vae_latent_mean', None) is None and hp_vae.get('vae_latent_std', None) is None:
        hp_vae['vae_latent_mean'] = vae_latent_mean
        hp_vae['vae_latent_std'] = vae_latent_std
    hp_vae['latent_norm_mode'] = latent_norm_mode
    latent_stats_path = _derive_latent_stats_path_from_ckpt(vae_ckpt)
    if latent_stats_path is not None:
        hp_vae['latent_stats_path'] = latent_stats_path

    vae = build_wavvae(hp_vae, infer=True, attn_implementation=attn_implementation)
    load_ckpt(vae, vae_ckpt, 'model_gen', strict=False)
    vae.apply_remove_weight_norm()
    vae.eval()
    return vae, hp_vae


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
        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        self.build_dit(hparams, attn_implementation)

        if hparams.get('use_caption', False):
            self.build_goku_text_encoder(hparams)

    def build_dit(self, hparams, attn_implementation='flash_attention_2'):
        from modules.tts.swanaudio.dit import Diffusion, ModelArgs
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

        config.attn_implementation = attn_implementation

        config.in_channels = config.out_channels = hparams['latent_dim']
        config.vocab_size = self.dit_vocab_size
        config.caption_dim = hparams.get('caption_encoder_dim', 5120)
        config.cfg_mask_text_token = self.cfg_mask_text_token = self.dit_text_tokenizer.encode('<MASK>')[0]
        config.text_fill_token = self.text_fill_token = self.dit_text_tokenizer.encode('<FILL>')[0]
        config.do_checkpoint = hparams.get('do_checkpoint', False) or hparams.get('gradient_checkpointing', False)
        # config.torch_compile_enabled = hparams.get('torch_compile', False) if torch.__version__.split(".")[0] == '2' else False
        config.torch_compile_enabled = False
        config.use_caption_text_mark = hparams.get('use_caption_text_mark', False)
        config.use_caption_pool_in_adaln = hparams.get('use_caption_pool_in_adaln', False)
        config.use_dynamic_cross_gate = hparams.get('use_dynamic_cross_gate', False)
        config.use_spk_mask = hparams.get('use_spk_mask', False)
        config.use_bgm_flag = hparams.get('use_bgm_flag', False)
        config.use_quality_flag = hparams.get('use_quality_flag', False)
        config.use_gated_attention = hparams.get('use_gated_attention', False)
        config.encoder_ffn_mult = hparams.get('encoder_ffn_mult', 4.0)

        config.use_moe_ffn = hparams.get('use_moe_ffn', False)
        config.moe_p = hparams.get('moe_p', 0.7)
        config.moe_num_routed = hparams.get('moe_num_routed', 8)
        config.moe_num_shared = hparams.get('moe_num_shared', 2)
        config.moe_num_null = hparams.get('moe_num_null', 4)
        config.moe_use_gumbel = hparams.get('moe_use_gumbel', False)
        config.moe_gumbel_tau_start = hparams.get('moe_gumbel_tau_start', 1.0)
        config.moe_gumbel_tau_end = hparams.get('moe_gumbel_tau_end', 0.3)
        config.moe_gumbel_tau_anneal_steps = hparams.get('moe_gumbel_tau_anneal_steps', 200_000)
        config.moe_expert_dropout = hparams.get('moe_expert_dropout', 0.0)
        config.moe_use_bias_balance = hparams.get('moe_use_bias_balance', True)
        config.moe_bias_update_rate = hparams.get('moe_bias_update_rate', 0.05)
        config.moe_bias_momentum = hparams.get('moe_bias_momentum', 0.9)
        config.moe_bias_clamp = hparams.get('moe_bias_clamp', 10.0)
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
        self.vae_stride = hparams.get('vae_stride', 8)
        return self.dit

    def build_dit_text_tokenizer(self):
        if hparams.get('use_cosyvoice2_text_tokenizer',False):
            from utils.text.cosyvoice2_tokenizer import get_tokenizer
            text_tokenizer = get_tokenizer(multilingual=True, num_languages=100)
            vocab_size = text_tokenizer.encoding.n_vocab
            return text_tokenizer, vocab_size
        else:
            from transformers import AutoTokenizer
            text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
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

    def build_goku_text_encoder(self, config):
        from transformers import T5EncoderModel, AutoTokenizer
        # T5Tokenizer(slow)在新版 transformers 可能已下线,fallback 到 AutoTokenizer
        try:
            from transformers import T5Tokenizer
        except ImportError:
            T5Tokenizer = AutoTokenizer
        dtype = torch.bfloat16

        if config['use_qwen']:
            qwen_name = config['pretrained_text_encoder_qwen']

            # 分支1：包含 Qwen2 -> 保持现有逻辑不变
            if 'qwen2' in qwen_name.lower():
                from transformers import Qwen2Model, AutoTokenizer
                self.goku_text_encoder = Qwen2Model.from_pretrained(qwen_name, torch_dtype=dtype)
                self.goku_text_encoder.requires_grad_(False).eval()

                self.goku_tokenizer = AutoTokenizer.from_pretrained(
                    qwen_name,
                    max_length=config['text_max_token_length'],
                    revision=None
                )
                vocal_size = len(self.goku_tokenizer)
                self.goku_special_token_ids = [vocal_size + 11, vocal_size + 12]
                self.goku_tokenizer.add_tokens([
                    '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>',
                    '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>',
                    '<S1>', '</S1>', '<S2>', '</S2>',
                    '<Audio>', '</Audio>', '<BGM>', '</BGM>',
                    '<W>', '</W>', '<FILL>', '<PAD>',
                    '<S3>', '</S3>', '<S4>', '</S4>',
                    '<|laughter|>', '<|breathe|>', '<|cry|>',
                    '<ENV>','</ENV>', '<|sp|>' 
                ], special_tokens=True)

                transformer_cls = get_class_from_module(
                    "transformers.models.qwen2.modeling_qwen2",
                    "Qwen2DecoderLayer"
                )
                if int(os.environ.get("WORLD_SIZE", 1)) > 1:
                    self.goku_text_encoder = shard_model_in_node(
                        self.goku_text_encoder,
                        transformer_cls,
                        int(os.environ.get("LOCAL_RANK", 0))
                    )

            # 分支2:包含 Qwen3 (含 Qwen3.5) -> 复用 build_qwen3,内部按 _is_qwen35 分发
            elif 'qwen3' in qwen_name.lower():
                lm, tok = build_qwen3(hparams, qwen_name, dtype=dtype)
                self.goku_text_encoder = lm.model
                self.goku_text_encoder.requires_grad_(False).eval()
                self.goku_tokenizer = tok

                vocal_size = len(self.goku_tokenizer)
                self.goku_special_token_ids = [vocal_size + 11, vocal_size + 12]

                # FSDP wrap 类:Qwen3 / Qwen3.5 由 helper 根据 qwen_name 选择对应 DecoderLayer
                layer_classes = get_qwen_decoder_layer_classes(qwen_name)
                if int(os.environ.get("WORLD_SIZE", 1)) > 1 and layer_classes:
                    self.goku_text_encoder = shard_model_in_node(
                        self.goku_text_encoder,
                        layer_classes[0],
                        int(os.environ.get("LOCAL_RANK", 0))
                    )

            else:
                raise ValueError(
                    f"Unknown Qwen backbone: {qwen_name}. Expect the name to contain 'Qwen2' or 'Qwen3'."
                )

        else:
            # 保持原 T5 逻辑不变
            self.goku_text_encoder = T5EncoderModel.from_pretrained(config['pretrained_text_encoder'], torch_dtype=dtype)
            self.goku_text_encoder.requires_grad_(False).eval()
            self.goku_tokenizer = T5Tokenizer.from_pretrained(
                config['pretrained_text_encoder'],
                max_length=config['text_max_token_length']
            )
            vocal_size = len(self.goku_tokenizer)
            self.goku_special_token_ids = [vocal_size + 11, vocal_size + 12]
            self.goku_tokenizer.add_tokens([
                '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>',
                '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>',
                '<S1>', '</S1>', '<S2>', '</S2>',
                '<Audio>', '</Audio>', '<BGM>', '</BGM>',
                '<W>', '</W>', '<FILL>', '<PAD>',
                '<S3>', '</S3>', '<S4>', '</S4>',
                '<|laughter|>', '<|breathe|>', '<|cry|>',
                '<ENV>','</ENV>', '<|sp|>' 
            ], special_tokens=True)
