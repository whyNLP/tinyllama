from flash_attn.layers.rotary import apply_rotary_emb

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as _LlamaRotaryEmbedding

class LlamaRotaryEmbedding(_LlamaRotaryEmbedding):

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = freqs # we do not need to double the frequency here
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    assert unsqueeze_dim == 1, "fused rotary pos emb only supports unsqueeze_dim=1"
    assert q.shape[-1] == cos.shape[-1]*2, "q and cos must have the same embedding dimension"
    assert cos.shape[0] == 1, "cos must have a batch dimension of 1"
    assert sin.shape[0] == 1, "sin must have a batch dimension of 1"
    
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)
    q_embed = apply_rotary_emb(q.transpose(1, 2), cos, sin).transpose(1, 2)
    k_embed = apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
    return q_embed, k_embed