import warnings
import torch
import math
import comfy
from nodes import KSampler
from .unet import get_timesteps, q_sample

def fabric_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, null_pos, null_neg, pos_weight, neg_weight, feedback_start, feedback_end, pos_latents=None, neg_latents=None):
    """
    Entry point for FABRIC
    """
    pos_latents = torch.empty(0, *latent_image['samples'].shape[1:]) if pos_latents is None else pos_latents['samples']
    neg_latents = torch.empty(0, *latent_image['samples'].shape[1:]) if neg_latents is None else neg_latents['samples']
    print("[FABRIC] latents shape:", pos_latents.shape, neg_latents.shape)
    if len(pos_latents) == 0 and len(neg_latents) == 0:
        print("[FABRIC] No reference latents found. Defaulting to regular KSampler.")
        return KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
    pos_w, pos_h = pos_latents.shape[2:]
    neg_w, neg_h = neg_latents.shape[2:]
    if pos_w != neg_w or pos_h != neg_h:
        print("[FABRIC] Reference latents have different sizes. Defaulting to regular KSampler.")
        return KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

    all_latents = torch.cat([pos_latents, neg_latents], dim=0)

    # If there are no reference latents, default to KSampler
    if len(all_latents) == 0:
        print("[FABRIC] No reference latents found. Defaulting to regular KSampler.")
        return KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

    model_patched = model.clone()
    device = comfy.model_management.get_torch_device()
    pos_latents = pos_latents.to(device)
    neg_latents = neg_latents.to(device)
    all_latents = all_latents.to(device)
    timesteps = get_timesteps(model_patched, steps, sampler_name, scheduler, denoise, device)
    print("[FABRIC] timesteps:", timesteps.shape, timesteps)
    # print("[FABRIC] model object attributes:", dir(model.model))
    print(f"[FABRIC] {len(pos_latents)} positive latents, {len(neg_latents)} negative latents")

    #
    # Precompute hidden states
    #
    all_hiddens = {}

    def compute_hidden_states(q, k, v, extra_options):
        idx = extra_options['transformer_index']
        if idx not in all_hiddens:
            all_hiddens[idx] = q
        else:
            all_hiddens[idx] = torch.cat([all_hiddens[idx], q], dim=0)
        print(f"[FABRIC] compute_hidden_states: idx={idx}, all_hiddens[idx].shape={all_hiddens[idx].shape}")
        return q, k, v


    is_hiddens_computed = False  # If unet is called more than once, only store hidden states once
    def unet_wrapper_hiddens(model_func, params):
        nonlocal is_hiddens_computed
        print("UNET_WRAPPER_HIDDENS", is_hiddens_computed)
        input_x = params['input']
        ts = params['timestep']
        c = params['c']

        if not is_hiddens_computed:
            # Warn if there are multiple timesteps that are not the same, i.e. different timesteps for different images in the batch
            if len(ts) > 1:
                if not torch.all(ts == ts[0]):
                    warnings.warn("[FABRIC] Different timesteps found for different images in the batch. \
                                Proceeding with the first timestep.")

            current_ts = ts[:1]

            # Get partially noised reference latents for the current timestep
            pos_zs = noise_latents(model_patched, pos_latents, current_ts)
            neg_zs = noise_latents(model_patched, neg_latents, current_ts)
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
            print("[FABRIC] all_zs:", all_zs.shape)
            for a in range(0, len(all_zs), batch_size):
                b = a + batch_size
                print("[FABRIC] batch:", a, b)
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
                print("[FABRIC] timesteps:", batch_ts)
                print("[FABRIC] model_func", batch_latents.shape, batch_ts.shape)
                _ = model_func(batch_latents, batch_ts, **c_null_dict)
            is_hiddens_computed = True
        return input_x

    print("[FABRIC] patching attn1")
    model_patched.set_model_attn1_patch(compute_hidden_states)
    print("[FABRIC] patching unet")
    model_patched.set_model_unet_function_wrapper(unet_wrapper_hiddens)
    print("model patches hiddens", model_patched.model_options["transformer_options"]["patches"])
    _ = KSampler().sample(model_patched, seed, 1, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)


    for k, v in all_hiddens.items():
        print(f"[FABRIC] layer {k}: {v.shape}")

    pos_hiddens = {}
    neg_hiddens = {}
    num_pos = len(pos_latents)
    num_neg = len(neg_latents)
    for layer_idx, hidden in all_hiddens.items():
        pos_hiddens[layer_idx] = hidden[:num_pos]
        neg_hiddens[layer_idx] = hidden[num_pos:]

    cond_or_uncond = None
    def unet_wrapper(model_func, params):
        input = params['input']
        ts = params['timestep']
        c = params['c']
        nonlocal cond_or_uncond
        cond_or_uncond = params['cond_or_uncond']
        print("[FABRIC] unet_wrapper: cond_or_uncond:", cond_or_uncond)
        return model_func(input, ts, **c)

    # Modify model to use the hidden states
    def modified_attn1(q, k, v, extra_options):
        idx = extra_options['transformer_index']
        printing = idx == 0
        pos_hs = pos_hiddens[idx]
        neg_hs = neg_hiddens[idx]
        if printing:
            for i, (a, b, c) in enumerate(zip(q, k, v)):
                print(f"[FABRIC] {i} means: q={a.mean()}, k={b.mean()}, v={c.mean()}")
            print(f"[FABRIC] modified_attn1: idx={idx}, pos_hiddens={pos_hs.shape}, neg_hiddens={neg_hs.shape}")
            print(f"[FABRIC] q={q.shape}, k={k.shape}, v={v.shape}")
        # Flatten the first dimension into the second dimension ([b, seq, dim] -> [1, b*seq, dim])
        pos_hs = pos_hs.reshape(-1, pos_hs.shape[2]).unsqueeze(0)
        neg_hs = neg_hs.reshape(-1, neg_hs.shape[2]).unsqueeze(0)
        if printing:
            print(f"[FABRIC] pos_hs={pos_hs.shape}, neg_hs={neg_hs.shape}")
        # print(f"[FABRIC] extra_options={extra_options}")
        # Match the second dimensions
        largest_dim = max(pos_hs.shape[1], neg_hs.shape[1])
        if printing:
            print(f"[FABRIC] largest_dim={largest_dim}")
        largest_shape = (1, largest_dim, pos_hs.shape[2])
        pos_hs = match_shape(pos_hs, largest_shape)
        neg_hs = match_shape(neg_hs, largest_shape)
        if printing:
            print(f"[FABRIC] after match: pos_hs={pos_hs.shape}, neg_hs={neg_hs.shape}")
        # Broadcast the first dimension to the batch size
        # pos_hs = broadcast_tensor(pos_hs, q.shape[0])
        # neg_hs = broadcast_tensor(neg_hs, q.shape[0])
        # Concat the positive hidden states and negative hidden states to line up with cond and uncond noise
        nonlocal cond_or_uncond
        cond_uncond_idxs = repeat_list_to_size(cond_or_uncond, q.shape[0])
        concat_hs = []
        if printing:
            print(f"[FABRIC] cond_uncond_idxs={cond_uncond_idxs}")
        for x in cond_uncond_idxs:
            if x == 1:
                concat_hs.append(pos_hs)
            else:
                concat_hs.append(neg_hs)
        concat_hs = torch.cat(concat_hs, dim=0)
        if printing:
            print(f"[FABRIC] after concat: concat_hs={concat_hs.shape}")
        # Concat hs to k and v
        k = torch.cat([k, concat_hs], dim=1)
        v = torch.cat([v, concat_hs], dim=1)
        if printing:
            print(f"[FABRIC] after concat: k={k.shape}, v={v.shape}")
        weights = get_weights(pos_weight, neg_weight, q, num_pos, num_neg, cond_uncond_idxs)
        if printing:
            print(f"[FABRIC] weights={weights.shape}, mean={weights.mean()}")
        # Apply weights
        weighted_v = v * weights[:, :, None]
        return q, k, weighted_v
    
    
    # Restore original model
    model_patched = model.clone()
    print("model patches sampling", model_patched.model_options["transformer_options"])
    model_patched.set_model_unet_function_wrapper(unet_wrapper)
    model_patched.set_model_attn1_patch(modified_attn1)
    print("model patches sampling", model_patched.model_options["transformer_options"]["patches"])
    samples = KSampler().sample(model_patched, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
    # Restore original model
    model_patched = model.clone()
    print("model patches finalize", model_patched.model_options["transformer_options"])
    print("model finalize", model.model_options["transformer_options"])

    return samples


def get_weights(pos_weight, neg_weight, q, num_pos, num_neg, cond_uncond_idxs):
    """
    Prepare weights for the loss function
    """
    input_dim = q.shape[1]
    hs_dim = max(num_pos, num_neg) * input_dim
    dim = input_dim + hs_dim
    batched_weights = []
    for x in cond_uncond_idxs:
        weights = torch.ones(dim)
        if x == 1:
            weights[input_dim:] *= pos_weight
        else:
            weights[input_dim:] *= neg_weight
        batched_weights.append(weights)
    batched_weights = torch.stack(batched_weights).to(q.device)
    return batched_weights


def broadcast_tensor(tensor, batch):
    """
    Broadcasts tensor to the batch size. Intended for tensor of shape (1, ...)
    """
    if tensor.shape[0] < batch:
        tensor = torch.cat([tensor] * batch)
    return tensor


def get_null_cond(cond, size):
    """
    Isolate clip embedding from cond dict
    """
    if len(cond) > 1:
        warnings.warn("[FABRIC] Multiple conditioning found. Proceeding with the first conditioning.")
    emb, _ = cond[0]
    c_crossattn = broadcast_tensor(emb, size) if size > 0 else torch.empty(0, *emb.shape[1:])
    return c_crossattn


def noise_latents(model, latents, ts):
    """
    Noise latents to the current timestep
    """
    zs = []
    for latent in latents:
        z_ref = q_sample(model, latent.unsqueeze(0), ts)
        zs.append(z_ref)
    if len(zs) == 0:
        return latents
    return torch.cat(zs, dim=0)

def match_batch_size(tensor, batch_size):
    """
    Adds zeros to the tensor to match batch size. Intended for tensor of shape (1, ...)
    """
    if tensor.shape[0] < batch_size:
        zeros = torch.zeros(batch_size - tensor.shape[0], *tensor.shape[1:], device=tensor.device)
        tensor = torch.cat([tensor, zeros])
    return tensor


def match_shape(tensor, shape):
    """
    Adds zeros to the tensor to match the given shape
    """
    if tensor.shape != shape:
        zeros = torch.zeros(*shape, device=tensor.device)
        zeros[:tensor.shape[0], :tensor.shape[1]] = tensor
        tensor = zeros
    return tensor

def repeat_list_to_size(lst, size):
    """
    If list=[1, 0] and size=4, returns [1, 1, 0, 0]
    """
    return [item for item in lst for _ in range(size // len(lst))]

def copy_model_options(model_options):
    copy = {}
    for k, v in model_options.items():
        if isinstance(v, list):
            copy[k] = v.copy()
        else:
            copy[k] = v
    return copy