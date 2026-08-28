from ._compat import (
    PretrainedConfig,
    layer_type_validation,
    logging,
    rope_config_validation,
)

logger = logging.get_logger(__name__)

class TransformerDiTConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_cross_attention_layers (`int`, *optional*, defaults to 0):
            Number of cross-attention layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
            additional layer afterwards will use SWA (Sliding Window Attention).
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_dynamic_cross_gate (`bool`, *optional*, defaults to `False`):
            Whether to use dynamic cross-attention gate mechanism.
        use_gated_attention (`bool`, *optional*, defaults to `False`):
            Whether to use gated attention.
        use_gated_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether to use gated cross attention.
        use_caption_pool_in_adaln (`bool`, *optional*, defaults to `False`):
            Whether to use caption pool in adaln.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer.
        moe_intermediate_size (`int`, *optional*, defaults to 768):
            Intermediate size of the routed expert.
        # num_experts_per_tok (`int`, *optional*, defaults to 8):
        #     Number of selected experts.
        # num_experts (`int`, *optional*, defaults to 128):
        #     Number of routed experts.
        # norm_topk_prob (`bool`, *optional*, defaults to `False`):
        #     Whether to normalize the topk probabilities.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        # router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
        #     The aux loss factor for the total loss.
        mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
            Indicate which layers use Qwen3MoeMLP rather than Qwen3MoeSparseMoeBlock
            The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
            If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.
    """

    model_type = "transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=5504,
        num_hidden_layers=24,
        num_cross_attention_layers=0,
        num_attention_heads=16,
        num_key_value_heads=None,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=None,
        layer_types=None,
        attention_dropout=0.0,
        use_dynamic_cross_gate=False,
        use_gated_attention=False,
        use_gated_cross_attention=False,
        use_caption_pool_in_adaln=True,
        decoder_sparse_step=1,
        moe_intermediate_size=None,
        output_router_logits=False,
        mlp_only_layers=None,
        use_mhc: bool = False,
        mhc_num_streams: int = 4,
        mhc_sinkhorn_iters: int = 6,
        mhc_use_dynamic: bool = True,

        use_moe_ffn: bool = False,
        moe_p: float = 0.7,
        moe_num_routed: int = 0,
        moe_num_shared: int = 0,
        moe_num_task_experts: int = 4,
        moe_task_p: float = 0.7,
        moe_num_null: int = 0,
        moe_use_gumbel: bool = False,
        moe_gumbel_tau_start: float = 1.0,
        moe_gumbel_tau_end: float = 0.3,
        moe_gumbel_tau_anneal_steps: int = 200_000,
        moe_expert_dropout: float = 0.0,
        moe_load_balance_loss_coef: float = 1e-2,
        moe_router_z_loss_coef: float = 1e-3,
        moe_null_loss_coef: float = 1e-2,
        moe_use_bias_balance: bool = False,
        moe_bias_update_rate: float = 1e-3,
        moe_bias_momentum: float = 0.9,
        moe_bias_clamp: float = 3.0,
        moe_capacity_factor_min: float = 1.0,
        moe_capacity_factor_max: float = 2.0,
        moe_overflow_drop: bool = True,
        moe_use_t_budget: bool = True,
        moe_p_min: float = 0.4,
        moe_p_max: float = 0.95,
        moe_null_logit_bias_min: float = -2.0,
        moe_null_logit_bias_max: float = 0.0,

        ec_early_dense_moe_layers: int = 1,
        ec_early_capacity_boost_moe_layers: int = 1,
        ec_early_capacity_ratio: float = None,
        ec_default_step_ratio: float = 0.5,

        **kwargs,
    ):
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
            
        if max_window_layers is None:
            max_window_layers = num_hidden_layers
            
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_dynamic_cross_gate = use_dynamic_cross_gate
        self.use_gated_attention = use_gated_attention
        self.use_gated_cross_attention = use_gated_cross_attention
        self.use_caption_pool_in_adaln = use_caption_pool_in_adaln
        
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # MoE arguments
        self.use_moe_ffn = use_moe_ffn
        self.moe_p = float(moe_p)
        self.moe_num_routed = int(moe_num_routed)
        self.moe_num_shared = int(moe_num_shared)
        self.moe_num_task_experts = int(moe_num_task_experts)
        self.moe_task_p = float(moe_task_p)
        self.moe_num_null = int(moe_num_null)
        self.moe_intermediate_size = moe_intermediate_size

        self.decoder_sparse_step = decoder_sparse_step
        self.output_router_logits = output_router_logits
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        self.moe_use_gumbel = bool(moe_use_gumbel)
        self.moe_gumbel_tau_start = float(moe_gumbel_tau_start)
        self.moe_gumbel_tau_end = float(moe_gumbel_tau_end)
        self.moe_gumbel_tau_anneal_steps = int(moe_gumbel_tau_anneal_steps)
        self.moe_expert_dropout = float(moe_expert_dropout)

        self.moe_load_balance_loss_coef = float(moe_load_balance_loss_coef)
        self.moe_router_z_loss_coef = float(moe_router_z_loss_coef)
        self.moe_null_loss_coef = float(moe_null_loss_coef)
        self.moe_use_bias_balance = bool(moe_use_bias_balance)
        self.moe_bias_update_rate = float(moe_bias_update_rate)
        self.moe_bias_momentum = float(moe_bias_momentum)
        self.moe_bias_clamp = float(moe_bias_clamp)

        self.moe_capacity_factor_min = float(moe_capacity_factor_min)
        self.moe_capacity_factor_max = float(moe_capacity_factor_max)
        self.moe_overflow_drop = bool(moe_overflow_drop)

        self.moe_use_t_budget = bool(moe_use_t_budget)
        self.moe_p_min = float(moe_p_min)
        self.moe_p_max = float(moe_p_max)
        self.moe_null_logit_bias_min = float(moe_null_logit_bias_min)
        self.moe_null_logit_bias_max = float(moe_null_logit_bias_max)

        self.ec_early_dense_moe_layers = int(ec_early_dense_moe_layers)
        self.ec_early_capacity_boost_moe_layers = int(ec_early_capacity_boost_moe_layers)
        self.ec_early_capacity_ratio = float(ec_early_capacity_ratio) if ec_early_capacity_ratio is not None else self.moe_capacity_factor_max
        self.ec_default_step_ratio = float(ec_default_step_ratio)

        self.use_mhc = use_mhc
        self.mhc_num_streams = mhc_num_streams
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters
        self.mhc_use_dynamic = mhc_use_dynamic

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
