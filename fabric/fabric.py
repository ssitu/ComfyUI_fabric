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
    num_pos = len(pos_latents)
    num_neg = len(neg_latents)
    all_latents = all_latents.to(device)
    print(f"[FABRIC] {num_pos} positive latents, {num_neg} negative latents")

    #
    # Precompute hidden states
    #
    all_hiddens = {}
    pos_hiddens = {}
    neg_hiddens = {}
    is_compute_hidden_states = True

    def store_hidden_states(q, k, v, extra_options):
        nonlocal is_compute_hidden_states, all_hiddens
        if not is_compute_hidden_states:
            return q, k, v
        
        print("=====================STORE_HIDDEN_STATES=====================")
        idx = extra_options['transformer_index']
        if idx not in all_hiddens:
            all_hiddens[idx] = q
        else:
            all_hiddens[idx] = torch.cat([all_hiddens[idx], q], dim=0)
        print(f"[FABRIC] compute_hidden_states: idx={idx}, all_hiddens[idx].shape={all_hiddens[idx].shape}")
        return q, k, v

    cond_or_uncond = None  # To be set in unet_wrapper
    is_modified_attn1 = False

    def modified_attn1(q, k, v, extra_options):
        # Modify model to use the hidden states
        nonlocal is_modified_attn1, cond_or_uncond, pos_hiddens, neg_hiddens
        nonlocal num_pos, num_neg
        if not is_modified_attn1:
            return q, k, v

        print("=====================MODIFIED_ATTN1=====================")
        idx = extra_options['transformer_index']
        printing = True

        pos_hs = pos_hiddens[idx]
        neg_hs = neg_hiddens[idx]
        if printing:
            for i, (a, b, c) in enumerate(zip(q, k, v)):
                print(f"[FABRIC] {i} means: q={a.mean()}, k={b.mean()}, v={c.mean()}")
            print(f"[FABRIC] modified_attn1: idx={idx}, pos_hiddens={pos_hs.shape}, neg_hiddens={neg_hs.shape}")
            print(f"[FABRIC] q={q.shape}, k={k.shape}, v={v.shape}")
        # Flatten the first dimension into the second dimension ([b, seq, dim] -> [1, b*seq, dim])
        pos_hs = pos_hs.reshape(1, -1, pos_hs.shape[2])
        neg_hs = neg_hs.reshape(1, -1, neg_hs.shape[2])
        if printing:
            print(f"[FABRIC] pos_hs={pos_hs.shape}, neg_hs={neg_hs.shape}")
        # Match the second dimensions
        largest_dim = max(pos_hs.shape[1], neg_hs.shape[1])
        if printing:
            print(f"[FABRIC] largest_dim={largest_dim}")
        largest_shape = (1, largest_dim, pos_hs.shape[2])
        pos_hs = match_shape(pos_hs, largest_shape)
        neg_hs = match_shape(neg_hs, largest_shape)
        if printing:
            print(f"[FABRIC] after match: pos_hs={pos_hs.shape}, neg_hs={neg_hs.shape}")
        # Concat the positive hidden states and negative hidden states to line up with cond and uncond noise
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

    def unet_wrapper(model_func, params):
        nonlocal is_compute_hidden_states, is_modified_attn1
        nonlocal pos_latents, neg_latents, all_latents
        nonlocal pos_hiddens, neg_hiddens, all_hiddens
        nonlocal cond_or_uncond
        nonlocal num_pos, num_neg
        nonlocal model_patched

        input = params['input']
        ts = params['timestep']
        c = params['c']

        # Save cond_or_uncond index for attention patch
        cond_or_uncond = params['cond_or_uncond']
        print("[FABRIC] unet_wrapper: cond_or_uncond:", cond_or_uncond)

        #
        # Compute hidden states
        #
        # Warn if there are multiple timesteps that are not the same, i.e. different timesteps for different images in the batch
        if len(ts) > 1:
            if not torch.all(ts == ts[0]):
                warnings.warn("[FABRIC] Different timesteps found for different images in the batch. \
                            Proceeding with the first timestep.")

        current_ts = ts[:1]

        # Noise the reference latents to the current timestep
        pos_zs = noise_latents(model_patched, pos_latents, current_ts)
        neg_zs = noise_latents(model_patched, neg_latents, current_ts)
        all_zs = torch.cat([pos_zs, neg_zs], dim=0)

        #
        # Make a forward pass to compute hidden states
        #
        is_compute_hidden_states = True
        is_modified_attn1 = False
        all_hiddens = {}
        pos_hiddens = {}
        neg_hiddens = {}

        batch_size = input.shape[0]
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

        for layer_idx, hidden in all_hiddens.items():
            pos_hiddens[layer_idx] = hidden[:num_pos]
            neg_hiddens[layer_idx] = hidden[num_pos:]

        # Do the actual forward pass
        is_compute_hidden_states = False
        is_modified_attn1 = True

        return model_func(input, ts, **c)

    model_patched.set_model_attn1_patch(store_hidden_states)
    model_patched.set_model_attn1_patch(modified_attn1)
    model_patched.set_model_unet_function_wrapper(unet_wrapper)
    samples = KSampler().sample(model_patched, seed, steps, cfg, sampler_name,
                                scheduler, positive, negative, latent_image, denoise)
    return samples


def save(samples, filename):
    counter = 1
    import os
    while os.path.exists(f"{filename}_{counter}_.latent"):
        counter += 1
    file = f"{filename}_{counter}_.latent"

    output = {}
    output["latent_tensor"] = samples
    output["latent_format_version_0"] = torch.tensor([])

    import safetensors
    safetensors.torch.save_file(output, file)

def load(i):
    import safetensors
    from safetensors.torch import load_file
    latent_path = f"D:\ComfyUI\ComfyUI\input\z_{i}_.latent"
    latent = safetensors.torch.load_file(latent_path, device="cpu")
    print("Loaded latent", latent_path)
    multiplier = 1.0
    if "latent_format_version_0" not in latent:
        multiplier = 1.0 / 0.18215
    return latent["latent_tensor"].float() * multiplier

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
            weights[input_dim:] = pos_weight
        else:
            weights[input_dim:] = neg_weight
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
        print("[FABRIC] z_ref:", z_ref.shape, ts)
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
