import os

if os.environ.get('TINY_FUSED_RMSNORM', False):
    import transformers
    from flash_attn.ops.triton.layer_norm import RMSNorm
    transformers.models.llama.modeling_llama.LlamaRMSNorm = RMSNorm

if os.environ.get('TINY_FUSED_CROSSENTROPY', False):
    import transformers
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
    transformers.models.llama.modeling_llama.CrossEntropyLoss = CrossEntropyLoss

if os.environ.get('TINY_FUSED_ROTARY', False):
    import transformers
    from .llama_fused_rotary import (
        LlamaRotaryEmbedding,
        fused_apply_rotary_pos_emb
    )
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = LlamaRotaryEmbedding

if os.environ.get('TINY_FUSED_SWIGLU', False):
    import transformers
    from .llama_fused_swiglu import LlamaMLP
    transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP