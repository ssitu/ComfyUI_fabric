import warnings
import torch
import math
import comfy
from nodes import KSampler
from .unet import get_timesteps, q_sample, forward


def fabric_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, null_pos, null_neg, pos_latents=None, neg_latents=None):
    """
    Entry point for FABRIC
    """
    pos_latents = torch.empty(0) if pos_latents is None else pos_latents['samples']
    neg_latents = torch.empty(0) if neg_latents is None else neg_latents['samples']
    print("[FABRIC] latents shape:", pos_latents.shape, neg_latents.shape)
    pos_w, pos_h = pos_latents.shape[2:4]
    neg_w, neg_h = neg_latents.shape[2:4]
    if pos_w != neg_w or pos_h != neg_h:
        print("[FABRIC] Reference latents have different sizes. Defaulting to regular KSampler.")
        return KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

    all_latents = torch.cat([pos_latents, neg_latents], dim=0)

    # If there are no reference latents, default to KSampler
    if len(all_latents) == 0:
        print("[FABRIC] No reference latents found. Defaulting to regular KSampler.")
        return KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

    original_model_patches = model.model_options["transformer_options"].get("patches", {})

    model = model.clone()
    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    pos_latents = pos_latents.to(device)
    neg_latents = neg_latents.to(device)
    all_latents = all_latents.to(device)

    #
    # Precompute hidden states
    #
    all_hiddens = {}

    def compute_hidden_states(q, k, v, extra_options):
        idx = extra_options['transformer_index']
        if idx not in all_hiddens:
            all_hiddens[idx] = q
        else:
            print("[FABRIC] concat", idx, all_hiddens[idx].shape, q.shape)
            all_hiddens[idx] = torch.cat([all_hiddens[idx], q], dim=0)
        print("[FABRIC] compute_hiddens_attn", idx, (q.shape if q is not None else None),
              (k.shape if k is not None else None), (v.shape if v is not None else None))
        return q, k, v

    print("[FABRIC] patching attn1")
    model.set_model_attn1_patch(compute_hidden_states)

    def unet_wrapper_hiddens(model_func, params):
        input_x = params['input']
        ts = params['timestep']
        c = params['c']

        # Warn if there are multiple timesteps that are not the same, i.e. different timesteps for different images in the batch
        if len(ts) > 1:
            if not torch.all(ts == ts[0]):
                warnings.warn("[FABRIC] Different timesteps found for different images in the batch. \
                              Proceeding with the first timestep.")

        current_ts = ts[:1]

        # Get partially noised reference latents for the current timestep
        pos_zs = noise_latents(model, pos_latents, current_ts)
        neg_zs = noise_latents(model, neg_latents, current_ts)
        all_zs = torch.cat([pos_zs, neg_zs], dim=0)

        #
        # Make a forward pass to compute hidden states
        #
        batch_size = input_x.shape[0]
        print("[FABRIC] batch_size:", batch_size)
        # Process reference latents in batches
        c_null_pos = get_null_cond(null_pos, len(pos_zs))
        c_null_neg = get_null_cond(null_neg, len(neg_zs))
        c_null = torch.cat([c_null_pos, c_null_neg], dim=0).to(device)
        for a in range(0, len(all_zs), batch_size):
            b = a + batch_size
            batch_latents = all_zs[a:b]
            c_null_batch = c_null[a:b]
            c_null_dict = {
                'c_crossattn': [c_null_batch],
                'transformer_options': c['transformer_options']
            }
            for cond in c["c_crossattn"]:
                print("[FABRIC] batch c_crossattn:", cond.shape)
            for cond in c_null_dict["c_crossattn"]:
                print("[FABRIC] batch null c_crossattn:", cond.shape)
            batch_ts = broadcast_tensor(current_ts, len(batch_latents))
            print("[FABRIC] model_func", batch_latents.shape, batch_ts.shape)
            _ = model_func(batch_latents, batch_ts, **c_null_dict)
        return input_x

    print("[FABRIC] patching unet")
    model.set_model_unet_function_wrapper(unet_wrapper_hiddens)
    _ = KSampler().sample(model, seed, 1, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

    # Restore original model
    model.model_options["transformer_options"]["patches"] = original_model_patches

    for k, v in all_hiddens.items():
        print("[FABRIC] all_hiddens", k, v.shape)

    # Modify model to use the precomputed hidden states
    def modified_attn1(q, k, v, extra_options):
        idx = extra_options['transformer_index']
        print("[FABRIC] modified_attn1", idx, all_hiddens[idx].shape, q.shape)
        return q, k, v

    model.set_model_attn1_patch(modified_attn1)
    samples = KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

    return samples


def broadcast_tensor(tensor, batch):
    """
    Broadcasts tensor to the batch size. Intended for tensor of shape (1, ...)
    """
    if tensor.shape[0] < batch:
        tensor = torch.cat([tensor] * batch)
    return tensor


def get_null_cond(cond, batch):
    """
    Isolate clip embedding from cond dict
    """
    if len(cond) > 1:
        warnings.warn("[FABRIC] Multiple conditioning found. Proceeding with the first conditioning.")
    emb, _ = cond[0]
    c_crossattn = broadcast_tensor(emb, batch)
    return c_crossattn

def noise_latents(model, latents, ts):
    """
    Noise latents to the current timestep
    """
    zs = []
    for latent in latents:
        z_ref = q_sample(model, latent.unsqueeze(0), ts)
        zs.append(z_ref)
    return torch.cat(zs, dim=0)