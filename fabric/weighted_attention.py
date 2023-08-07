from comfy.ldm.modules.attention import CrossAttention
import xformers
import torch


class AttentionPatcher:
    def __init__(self, weights):
        self.weights = weights
        self.old_xformers_attn = xformers.ops.memory_efficient_attention

    def patch(self):
        def new_xformers_attn(q, k, v, attn_bias=None, op=None):
            if self.weights is not None:
                bs, nq, nh, dh = q.shape  # batch_size, num_queries, num_heads, dim_per_head
                min_val = torch.finfo(q.dtype).min
                w_bias = self.weights.log().clamp(min=min_val)[None, None, None, :].expand(bs, nq, nh, -1).contiguous()
                w_bias = w_bias.to(q.device)
                if attn_bias is None:  # Seems to always be the case in Comfy
                    attn_bias = w_bias
                else:
                    attn_bias += w_bias
            return self.old_xformers_attn(q, k, v, attn_bias=attn_bias, op=op)
        xformers.ops.memory_efficient_attention = new_xformers_attn

    def unpatch(self):
        xformers.ops.memory_efficient_attention = self.old_xformers_attn
