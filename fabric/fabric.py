import warnings
import torch
import comfy
from nodes import KSamplerAdvanced, CLIPTextEncode
from .unet import q_sample, get_timesteps


def ksampler_fabric(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                    clip, pos_weight, neg_weight, feedback_percent, pos_latents=None, neg_latents=None):
    """
    Regular KSampler with intended FABRIC inputs
    """
    disable_noise = False
    start_at_step = None
    end_at_step = None
    force_full_denoise = False
    clip_encode = CLIPTextEncode()
    null_cond = clip_encode.encode(clip, "")[0]
    feedback_start = 0
    feedback_end = int(steps * feedback_percent)
    return fabric_sample(model, disable_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, force_full_denoise, denoise,
                         null_cond, null_cond, pos_weight, neg_weight, feedback_start, feedback_end, pos_latents, neg_latents)


def ksampler_advfabric(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                       null_pos, null_neg, pos_weight, neg_weight, feedback_start, feedback_end, pos_latents=None, neg_latents=None):
    """
    Regular KSampler with all FABRIC inputs
    """
    disable_noise = False
    start_at_step = None
    end_at_step = None
    force_full_denoise = False
    return fabric_sample(model, disable_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, force_full_denoise, denoise,
                         null_pos, null_neg, pos_weight, neg_weight, feedback_start, feedback_end, pos_latents, neg_latents)


def fabric_sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise,
                  null_pos, null_neg, pos_weight, neg_weight, feedback_start, feedback_end, pos_latents=None, neg_latents=None):
    """
    Entry point for FABRIC
    """
    pos_latents = torch.empty(0, *latent_image['samples'].shape[1:]) if pos_latents is None else pos_latents['samples']
    neg_latents = torch.empty(0, *latent_image['samples'].shape[1:]) if neg_latents is None else neg_latents['samples']
    if len(pos_latents) == 0 and len(neg_latents) == 0:
        print("[FABRIC] No reference latents found. Defaulting to regular KSampler.")
        return KSamplerAdvanced().sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise)

    pos_shape_mismatch = pos_latents.shape[1:] != latent_image['samples'].shape[1:]
    neg_shape_mismatch = neg_latents.shape[1:] != latent_image['samples'].shape[1:]
    if pos_shape_mismatch or neg_shape_mismatch:
        warnings.warn(
            f"\n[FABRIC] Latents have different sizes (input: {latent_image['samples'].shape}, pos: {pos_latents.shape}, neg: {neg_latents.shape}). Resizing latents to the same size as input latent. It is recommended to resize the latents beforehand in pixel space or using a model to resize the latent.")
        if pos_shape_mismatch:
            pos_latents = comfy.utils.common_upscale(
                pos_latents, latent_image['samples'].shape[3], latent_image['samples'].shape[2], "bilinear", "center")
        if neg_shape_mismatch:
            neg_latents = comfy.utils.common_upscale(
                neg_latents, latent_image['samples'].shape[3], latent_image['samples'].shape[2], "bilinear", "center")

    all_latents = torch.cat([pos_latents, neg_latents], dim=0)

    # If there are no reference latents, default to KSampler
    if len(all_latents) == 0:
        print("[FABRIC] No reference latents found. Defaulting to regular KSampler.")
        return KSamplerAdvanced().sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise)

    model_patched = model.clone()
    device = comfy.model_management.get_torch_device()
    pos_latents = pos_latents.to(device)
    neg_latents = neg_latents.to(device)
    # Scale latents for unit variance
    pos_latents = model.model.process_latent_in(pos_latents)
    neg_latents = model.model.process_latent_in(neg_latents)
    num_pos = len(pos_latents)
    num_neg = len(neg_latents)
    all_latents = all_latents.to(device)
    print(f"[FABRIC] {num_pos} positive latents, {num_neg} negative latents")

    #
    # Translate start and end step to timesteps
    #
    timesteps = get_timesteps(model_patched, steps, sampler_name, scheduler, denoise, device)
    feedback_start_ts = timesteps[feedback_start]
    feedback_end_ts = timesteps[min(feedback_end, len(timesteps) - 1)]

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

        idx = extra_options['transformer_index']
        if idx not in all_hiddens:
            all_hiddens[idx] = q
        else:
            all_hiddens[idx] = torch.cat([all_hiddens[idx], q], dim=0)
        return q, k, v

    cond_or_uncond = None  # To be set in unet_wrapper
    is_modified_attn1 = False

    def modified_attn1(q, k, v, extra_options):
        # Modify model to use the hidden states
        nonlocal is_modified_attn1, cond_or_uncond, pos_hiddens, neg_hiddens
        nonlocal num_pos, num_neg
        if not is_modified_attn1:
            return q, k, v

        idx = extra_options['transformer_index']

        pos_hs = pos_hiddens[idx]
        neg_hs = neg_hiddens[idx]
        # Flatten the first dimension into the second dimension ([b, seq, dim] -> [1, b*seq, dim])
        pos_hs = pos_hs.reshape(1, -1, pos_hs.shape[2])
        neg_hs = neg_hs.reshape(1, -1, neg_hs.shape[2])
        # Match the second dimensions
        largest_dim = max(pos_hs.shape[1], neg_hs.shape[1])
        largest_shape = (1, largest_dim, pos_hs.shape[2])
        pos_hs = match_shape(pos_hs, largest_shape)
        neg_hs = match_shape(neg_hs, largest_shape)
        # Concat the positive hidden states and negative hidden states to line up with cond and uncond noise
        cond_uncond_idxs = repeat_list_to_size(cond_or_uncond, q.shape[0])
        concat_hs = []
        for x in cond_uncond_idxs:
            if x == 0:
                concat_hs.append(pos_hs)
            else:
                concat_hs.append(neg_hs)
        concat_hs = torch.cat(concat_hs, dim=0)
        # Concat hs to k and v
        k = torch.cat([k, concat_hs], dim=1)
        v = torch.cat([v, concat_hs], dim=1)
        weights = get_weights(pos_weight, neg_weight, q, num_pos, num_neg, cond_uncond_idxs)
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
        nonlocal feedback_start_ts, feedback_end_ts

        input = params['input']
        ts = params['timestep']
        c = params['c']

        # Normal pass if not in feedback range
        if not (feedback_end_ts.item() <= ts[0].item() <= feedback_start_ts.item()):
            return model_func(input, ts, **c)

        # Save cond_or_uncond index for attention patch
        cond_or_uncond = params['cond_or_uncond']

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
        # Process reference latents in batches
        c_null_pos = get_null_cond(null_pos, len(pos_zs))
        c_null_neg = get_null_cond(null_neg, len(neg_zs))
        c_null = torch.cat([c_null_pos, c_null_neg], dim=0).to(device)
        for a in range(0, len(all_zs), batch_size):
            b = a + batch_size
            batch_latents = all_zs[a:b]
            c_null_batch = c_null[a:b]
            c_null_dict = {
                'c_crossattn': c_null_batch,
                'transformer_options': c['transformer_options']
            }
            batch_ts = broadcast_tensor(current_ts, len(batch_latents))
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
    samples = KSamplerAdvanced().sample(model_patched, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive,
                                        negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise)
    return samples


def get_weights(pos_weight, neg_weight, q, num_pos, num_neg, cond_uncond_idxs):
    """
    Prepare weights for the weighted attention
    """
    input_dim = q.shape[1]
    hs_dim = max(num_pos, num_neg) * input_dim
    dim = input_dim + hs_dim
    batched_weights = []
    for x in cond_uncond_idxs:
        weights = torch.ones(dim)
        if x == 0:
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
