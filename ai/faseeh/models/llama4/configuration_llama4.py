from transformers import LlamaConfig
from dataclasses import dataclass, field
from typing import Optional, List, Union

@dataclass
class Llama4Config(LlamaConfig):
    model_type: str = "llama4"

    vocab_size: int = field(default=32000)

    # Core Llama 4 Scout Parameters (Scaled Down for 2x49GB GPUs)
    hidden_size: int = field(default=1024)
    intermediate_size: int = field(default=4096)
    num_hidden_layers: int = field(default=16)
    num_attention_heads: int = field(default=16)
    num_key_value_heads: int = field(default=4)

    # MoE Specific Parameters for Llama 4 Scout
    num_local_experts: int = field(default=8)
    num_experts_per_tok: int = field(default=1)
    router_aux_loss_coef: float = field(default=0.001)
    router_jitter_noise: float = field(default=0.0)

    # Llama 4 specific architectural features
    rope_theta: float = field(default=500000.0)
    max_position_embeddings: int = field(default=8192)

    # Other parameters based on Llama 4 config/best practices
    attention_bias: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2")
    hidden_act: str = field(default="silu")
    initializer_range: float = field(default=0.02)
    rms_norm_eps: float = field(default=1e-5)
    use_cache: bool = field(default=True)

    interleave_moe_layer_step: int = field(default=1)

    # Multimodal specific parameters: Provide a dummy structure
    vision_config: Optional[dict] = field(default_factory=lambda: {
        "attention_dropout": 0.0, "hidden_act": "gelu", "hidden_size": 1408,
        "image_size": 336, "initializer_range": 0.02, "intermediate_size": 5632,
        "model_type": "llama4_vision_model", "multi_modal_projector_bias": False,
        "norm_eps": 1e-05, "num_attention_heads": 16, "num_channels": 3,
        "num_hidden_layers": 34, "patch_size": 14, "pixel_shuffle_ratio": 0.5,
        "projector_dropout": 0.0, "projector_input_dim": 4096, "projector_output_dim": 4096,
        "rope_theta": 10000, "vision_feature_layer": -1, "vision_feature_select_strategy": "default"
    })

    # Token IDs (will be set by Pretrainer based on tokenizer's actual IDs)
    pad_token_id: int = field(default=None)
    bos_token_id: int = field(default=None)
    eos_token_id: Union[int, List[int]] = field(default=None)

    def __post_init__(self):
        # FIX: Remove super().__post_init__() because LlamaConfig does not have one.
        self.model_type = "llama4"