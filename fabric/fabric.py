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
    all_latents = torch.cat([pos_latents, neg_latents], dim=0)

    # If there are no reference latents, default to KSampler
    if len(all_latents) == 0:
        return KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

    model_compute_hiddens = model.clone()
    comfy.model_management.load_model_gpu(model_compute_hiddens)
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
        print("[FABRIC] compute_hiddens_attn", idx, (q.shape if q is not None else None), (k.shape if k is not None else None), (v.shape if v is not None else None))
        return q, k, v

    print("[FABRIC] patching attn1")
    model_compute_hiddens.set_model_attn1_patch(compute_hidden_states)

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
        all_zs = []
        for latent in all_latents:
            z_ref = q_sample(model_compute_hiddens, latent.unsqueeze(0), current_ts)
            all_zs.append(z_ref)
        all_zs = torch.cat(all_zs, dim=0)

        #
        # Make a forward pass to compute hidden states
        #
        batch_size = input_x.shape[0]
        # Process reference latents in batches
        for a in range(0, len(all_zs), batch_size):
            b = a + batch_size
            print("[FABRIC] batch shape:", all_zs[a:b].shape)
            _ = model_func(all_zs[a:b], current_ts, **c)


        # print("input_x:", input_x.shape, input_x, end="\n\n")
        # print("ts:", ts.shape, ts, end="\n\n")
        # print("c:", c, end="\n\n")
        # print("params:", params, end="\n\n")

        return input_x

    model_compute_hiddens.set_model_unet_function_wrapper(unet_wrapper_hiddens)
    _ = KSampler().sample(model_compute_hiddens, seed, 1, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
    for k, v in all_hiddens.items():
        print("[FABRIC]", k, v.shape)
    model_compute_hiddens.unpatch_model()
    all_hiddens.clear()
    
    samples = KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
    for k, v in all_hiddens.items():
        print("[FABRIC]", k, v.shape)
    return samples
