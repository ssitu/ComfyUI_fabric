import torch
from comfy.model_management import xformers_enabled, pytorch_attention_enabled


class Weighted_Attn_Patcher:

    def patch(self, weights, extra_options):
        # Using same structure as comfy.ldm.modules.attention lines 493:505

        h = extra_options["n_heads"]
        h_dim = extra_options["dim_head"]
        idx = extra_options["transformer_index"]

        if xformers_enabled():
            import xformers
            self.orig_mem_eff_attn = xformers.ops.memory_efficient_attention

            def mem_eff_attn(*args, **kwargs):
                try:
                    q, k = args[0], args[1]  # [q, k]: (bs * h, [nq, nk], h_dim)
                    B, nq, nk = q.shape[0], q.shape[1], k.shape[1]
                    # Expand and reshape weights: (bs, 1, nk) -> (bs, h, nk) -> (bs*h, nk)
                    ws = weights.unsqueeze(1).expand(-1, h, nk).reshape(-1, nk)
                    kwargs["attn_bias"] = get_attn_bias(ws, (B, nq, nk), q.dtype)
                    out = self.orig_mem_eff_attn(*args, **kwargs)
                    return out
                except Exception as e:
                    print(e)
                    print("[FABRIC] Encountered an exception. If this is not a memory issue, please report this issue.")
                    print(f"[FABRIC] {idx}: weights b: {weights.shape[0]}, weights nk: {weights.shape[1]}, nq: {nq}, nk: {nk}, h: {h}, B: {B}")
                    self.unpatch()
                    raise e

            xformers.ops.memory_efficient_attention = mem_eff_attn

        elif pytorch_attention_enabled():
            self.orig_pt_sdp = torch.nn.functional.scaled_dot_product_attention

            def pt_sdp(*args, **kwargs):
                try:
                    q, k = args[0], args[1]  # [q, k]: (bs, h, [nq, nk], h_dim)
                    bs, nq, nk = q.shape[0], q.shape[2], k.shape[2]
                    kwargs["attn_mask"] = get_attn_bias(weights, (bs, h, nq, nk), q.dtype)

                    out = self.orig_pt_sdp(*args, **kwargs)
                    return out
                except Exception as e:
                    print(e)
                    print("[FABRIC] Encountered an exception. If this is not a memory issue, please report this issue.")
                    print(f"[FABRIC] {idx}: weights b: {weights.shape[0]}, weights nk: {weights.shape[1]}, nq: {nq}, nk: {nk}, h: {h}, bs: {bs}")
                    self.unpatch()
                    raise e
            torch.nn.functional.scaled_dot_product_attention = pt_sdp

        else:
            # self.orig_softmax = torch.softmax
            self.orig_softmax_method = torch.Tensor.softmax

            # def softmax(*args, **kwargs):
            #     try:
            #         out = self.orig_softmax(*args, **kwargs)
            #         B, nq, nk = out.shape
            #         # Expand and reshape weights: (bs, 1, nk) -> (bs, h, nk) -> (bs*h, nk)
            #         ws = weights.unsqueeze(1).expand(-1, h, nk).reshape(-1, nk)
            #         out *= weights[:, None, :]
            #         out /= out.sum(dim=-1, keepdim=True)
            #         return out
            #     except Exception as e:
            #         print(e)
            #         print("[FABRIC] Encountered an exception. If this is not a memory issue, please report this issue.")
            #         self.unpatch()
            #         raise e
            # torch.softmax = softmax

            def softmax_method(*args, **kwargs):
                try:
                    out = self.orig_softmax_method(*args, **kwargs)
                    B, nq, nk = out.shape
                    # Expand and reshape weights: (bs, 1, nk) -> (bs, h, nk) -> (bs*h, nk)
                    ws = weights.unsqueeze(1).expand(-1, h, nk).reshape(-1, nk)
                    out *= ws[:, None, :]
                    out /= out.sum(dim=-1, keepdim=True)
                    return out
                except Exception as e:
                    print(e)
                    print("[FABRIC] Encountered an exception. If this is not a memory issue, please report this issue.")
                    print(f"[FABRIC] {idx}: weights b: {weights.shape[0]}, weights nk: {weights.shape[1]}, nq: {nq}, nk: {nk}, h: {h}, B: {B}")
                    self.unpatch()
                    raise e
            torch.Tensor.softmax = softmax_method

    def unpatch(self):
        if xformers_enabled():
            import xformers
            xformers.ops.memory_efficient_attention = self.orig_mem_eff_attn

        elif pytorch_attention_enabled():
            torch.nn.functional.scaled_dot_product_attention = self.orig_pt_sdp

        else:
            # torch.softmax = self.orig_softmax
            torch.Tensor.softmax = self.orig_softmax_method


def get_attn_bias(weights, shape, dtype):
    """
    Convert weights of shape (bs, nk) or (bs*h, nk) to attn_bias of shape (bs, h, nq, nk) or (bs*h, nq, nk)

    Adapted from 
    https://github.com/dvruette/sd-webui-fabric/blob/929ac972d110cd94a2157906655df411b15459d9/scripts/weighted_attention.py#L65
    """
    B = weights.shape[0]
    nk = weights.shape[1]
    min_val = torch.finfo(dtype).min
    attn_bias = weights.log().clamp(min=min_val)
    view_shape = [B, *([1] * (len(shape) - 2)), nk]
    attn_bias = attn_bias.view(view_shape).expand(shape).to(dtype)
    return attn_bias
