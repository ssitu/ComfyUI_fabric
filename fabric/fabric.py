import warnings
import torch
import comfy
from nodes import KSamplerAdvanced
from .unet import q_sample, get_timesteps
from .weighted_attn import Weighted_Attn_Patcher

COND = 0
UNCOND = 1


def fabric_sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise,
                  null_pos, null_neg, pos_weight, neg_weight, feedback_start, feedback_end, pos_latents=None, neg_latents=None):
    """
    Advanced KSampler with FABRIC.
    """
    non_batch_shape = latent_image['samples'].shape[1:]
    pos_latents = {"samples": torch.empty(0, *non_batch_shape)} if pos_latents is None else pos_latents
    neg_latents = {"samples": torch.empty(0, *non_batch_shape)} if neg_latents is None else neg_latents
    if len(pos_latents['samples']) == 0 and len(neg_latents['samples']) == 0:
        print("[FABRIC] No reference latents found. Defaulting to regular KSampler.")
        return KSamplerAdvanced().sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise)

    #
    # Match latent shapes once instead of resizing every timestep in the sampler
    #
    pos_shape_mismatch = pos_latents['samples'].shape[1:] != non_batch_shape
    neg_shape_mismatch = neg_latents['samples'].shape[1:] != non_batch_shape
    if pos_shape_mismatch or neg_shape_mismatch:
        warnings.warn(
            f"\n[FABRIC] Latents have different sizes (input: {latent_image['samples'].shape}, pos: {pos_latents['samples'].shape}, neg: {neg_latents['samples'].shape}). Resizing latents to the same size as input latent. It is recommended to resize the latents beforehand in pixel space or using a model to resize the latent.")
        if pos_shape_mismatch:
            pos_latents['samples'] = comfy.utils.common_upscale(
                pos_latents['samples'], latent_image['samples'].shape[3], latent_image['samples'].shape[2], "bilinear", "center")
        if neg_shape_mismatch:
            neg_latents['samples'] = comfy.utils.common_upscale(
                neg_latents['samples'], latent_image['samples'].shape[3], latent_image['samples'].shape[2], "bilinear", "center")

    #
    # Translate start and end step to timesteps
    #
    timesteps = get_timesteps(model, steps, sampler_name, scheduler, denoise)
    feedback_start_ts = timesteps[feedback_start]
    feedback_end_ts = timesteps[min(feedback_end, len(timesteps) - 1)]

    #
    # Patch model and sample
    #
    fabric_patcher = FABRICPatcher(model, null_pos, null_neg, pos_weight, neg_weight,
                                   feedback_start, feedback_end, pos_latents, neg_latents)
    model_patched = fabric_patcher.patch(ts_interval=(feedback_start_ts, feedback_end_ts))
    samples = KSamplerAdvanced().sample(model_patched, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive,
                                        negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise)
    return samples


def fabric_patch(model, null_pos, null_neg, pos_weight, neg_weight, pos_latents=None, neg_latents=None):
    # Cannot use start and end step for patching the model, no sampler information
    feedback_start = float("inf")
    feedback_end = 0
    fabric_patcher = FABRICPatcher(model, null_pos, null_neg, pos_weight, neg_weight,
                                   feedback_start, feedback_end, pos_latents, neg_latents)
    model_patched = fabric_patcher.patch()
    return (model_patched,)


class FABRICPatcher:

    def __init__(self, model, null_pos, null_neg, pos_weight, neg_weight, feedback_start, feedback_end, pos_latents=None, neg_latents=None):
        self.model = model
        self.null_pos = null_pos
        self.null_neg = null_neg
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.feedback_start = feedback_start
        self.feedback_end = feedback_end
        self.pos_latents = torch.empty(0) if pos_latents is None else pos_latents["samples"]
        self.neg_latents = torch.empty(0) if neg_latents is None else neg_latents["samples"]

    def patch(self, ts_interval: tuple = None):
        """
        Patch the model to use FABRIC
        :param ts_interval: Int Tuple of (start, end) timesteps for FABRIC feedback with start > end >= 0
        :return: Patched model
        """
        model_patched = self.model.clone()
        model_device = comfy.model_management.get_torch_device()
        num_pos = len(self.pos_latents) if self.pos_latents is not None else 0
        num_neg = len(self.neg_latents) if self.neg_latents is not None else 0
        print(f"[FABRIC] {num_pos} positive latents, {num_neg} negative latents")

        if num_pos == 0 and num_neg == 0:
            print("[FABRIC] No reference latents given when patching model, skipping patch.")
            return self.model

        #
        # Precompute hidden states
        #
        self.all_hiddens = {}
        self.pos_hiddens = {}
        self.neg_hiddens = {}
        self.is_compute_hidden_states = True

        def store_hidden_states(q, k, v, extra_options):
            """
            Store the hidden states of reference latents for the self-attention.
            """
            if not self.is_compute_hidden_states:
                return q, k, v

            idx = extra_options['transformer_index']
            if idx not in self.all_hiddens:
                self.all_hiddens[idx] = q
            else:
                self.all_hiddens[idx] = torch.cat([self.all_hiddens[idx], q], dim=0)
            return q, k, v

        self.cond_or_uncond = None  # To be set in unet_wrapper
        self.is_modified_attn1 = False

        # Create attention patcher
        self.attn_patcher = Weighted_Attn_Patcher()

        def modified_attn1(q, k, v, extra_options):
            """
            Patch the self-attention to use the hidden states.
            """
            if not self.is_modified_attn1:
                return q, k, v

            idx = extra_options['transformer_index']

            pos_hs = self.pos_hiddens[idx]
            neg_hs = self.neg_hiddens[idx]

            # There should be the same number of pos and neg hidden states as pos and neg latents
            if COND in self.cond_or_uncond:
                assert pos_hs.shape[0] != num_pos, f"pos_hs batch size ({pos_hs.shape[0]}) != number of pos_latents ({num_pos})"
            if UNCOND in self.cond_or_uncond:
                assert neg_hs.shape[0] != num_neg, f"neg_hs batch size ({neg_hs.shape[0]}) != number of neg_latents ({num_neg})"

            # Flatten the first dimension into the second dimension ([b, seq, dim] -> [1, b*seq, dim])
            pos_hs = pos_hs.reshape(1, -1, pos_hs.shape[2])
            neg_hs = neg_hs.reshape(1, -1, neg_hs.shape[2])

            # Match the second dimensions
            largest_dim = max(pos_hs.shape[1], neg_hs.shape[1])
            largest_shape = (1, largest_dim, pos_hs.shape[2])
            pos_hs = match_shape(pos_hs, largest_shape)
            neg_hs = match_shape(neg_hs, largest_shape)

            # Repeat cond_or_uncond to match batch size
            cond_uncond_idxs = repeat_list_to_size(self.cond_or_uncond, q.shape[0])

            # Concat the positive hidden states and negative hidden states to line up with the cond and uncond
            concat_hs = []
            for x in cond_uncond_idxs:
                if x == COND:
                    concat_hs.append(pos_hs)
                else:
                    concat_hs.append(neg_hs)
            concat_hs = torch.cat(concat_hs, dim=0)

            # Concat hs to k and v
            k = torch.cat([k, concat_hs], dim=1)
            v = torch.cat([v, concat_hs], dim=1)

            # Apply weights
            weights = get_weights(self.pos_weight, self.neg_weight, q, num_pos, num_neg, cond_uncond_idxs)
            self.attn_patcher.patch(weights, extra_options)
            return q, k, v

        def after_attn1(out, extra_options):
            """
            After self attention, undo the weighted attention patch.
            """
            if self.is_modified_attn1:
                self.attn_patcher.unpatch()
            return out

        def unet_wrapper(model_func, params):
            """
            Wrapper for the unet to compute hidden states and patch the self-attention.
            """
            input = params['input']
            ts = params['timestep']
            c = params['c']

            non_batch_shape = input.shape[1:]

            # Normal pass if not in feedback range
            if ts_interval is not None:
                ts_start, ts_end = ts_interval
                if not (ts_end < ts[0].item() <= ts_start):
                    return model_func(input, ts, **c)

            # Save cond_or_uncond index for attention patch
            self.cond_or_uncond = params['cond_or_uncond']

            # Expand empty latents to match input size
            pos_lats = self.pos_latents.to(model_device)
            neg_lats = self.neg_latents.to(model_device)
            if pos_lats.shape[0] == 0:
                pos_lats = pos_lats.view(0, 1, 1, 1).expand(0, *non_batch_shape)
            if neg_lats.shape[0] == 0:
                neg_lats = neg_lats.view(0, 1, 1, 1).expand(0, *non_batch_shape)

            # Prepare latents for unet
            pos_lats = self.model.model.process_latent_in(pos_lats)
            neg_lats = self.model.model.process_latent_in(neg_lats)

            #
            # Compute hidden states
            #

            # Warn if there are multiple timesteps that are not the same, i.e. different timesteps for different images in the batch
            if len(ts) > 1:
                if not torch.all(ts == ts[0]):
                    warnings.warn(
                        "[FABRIC] Different timesteps found for different images in the batch. Proceeding with the first timestep.")

            current_ts = ts[:1]

            # Resize latents to match input latent size
            pos_shape_mismatch = pos_lats.shape[1:] != non_batch_shape
            neg_shape_mismatch = neg_lats.shape[1:] != non_batch_shape
            if pos_shape_mismatch or neg_shape_mismatch:
                warnings.warn(
                    f"\n[FABRIC] Latents have different sizes (input: {input.shape}, pos: {pos_lats.shape}, neg: {neg_lats.shape}). Resizing latents to the same size as input latent. It is recommended to resize the latents beforehand in pixel space or use a model to resize the latent.")
                if pos_shape_mismatch:
                    pos_lats = comfy.utils.common_upscale(
                        pos_lats, input.shape[3], input.shape[2], "bilinear", "center")
                if neg_shape_mismatch:
                    neg_lats = comfy.utils.common_upscale(
                        neg_lats, input.shape[3], input.shape[2], "bilinear", "center")

            # Noise the reference latents to the current timestep
            pos_zs = noise_latents(model_patched, pos_lats, current_ts)
            neg_zs = noise_latents(model_patched, neg_lats, current_ts)
            all_zs = torch.cat([pos_zs, neg_zs], dim=0)

            # Make a forward pass to compute hidden states
            self.is_compute_hidden_states = True
            self.is_modified_attn1 = False

            self.all_hiddens = {}
            self.pos_hiddens = {}
            self.neg_hiddens = {}

            # Process reference latents in batches
            batch_size = input.shape[0]
            c_null_pos = get_null_cond(self.null_pos, len(pos_zs))
            c_null_neg = get_null_cond(self.null_neg, len(neg_zs))
            c_null = torch.cat([c_null_pos, c_null_neg], dim=0).to(model_device)
            for a in range(0, len(all_zs), batch_size):
                b = a + batch_size
                batch_latents = all_zs[a:b]
                c_null_batch = c_null[a:b]
                c_null_dict = {
                    'c_crossattn': c_null_batch,
                    'transformer_options': c['transformer_options']
                }
                if 'c_adm' in c:
                    c_null_dict['c_adm'] = c['c_adm']
                batch_ts = broadcast_tensor(current_ts, len(batch_latents))
                # Pass the reference latents and call store_hidden_states for each block
                _ = model_func(batch_latents, batch_ts, **c_null_dict)

            for layer_idx, hidden in self.all_hiddens.items():
                self.pos_hiddens[layer_idx] = hidden[:num_pos]
                self.neg_hiddens[layer_idx] = hidden[num_pos:]

            # Do the actual forward pass
            self.is_compute_hidden_states = False
            self.is_modified_attn1 = True
            # modified_attn1 and after_attn1 is called for each block
            out = model_func(input, ts, **c)

            # Reset flags
            self.is_compute_hidden_states = False
            self.is_modified_attn1 = False
            return out

        model_patched.set_model_attn1_patch(store_hidden_states)
        model_patched.set_model_attn1_patch(modified_attn1)
        model_patched.set_model_attn1_output_patch(after_attn1)
        model_patched.set_model_unet_function_wrapper(unet_wrapper)
        return model_patched


def get_weights(pos_weight, neg_weight, q, num_pos, num_neg, cond_uncond_idxs):
    """
    Prepare weights for the weighted attention
    :return: Weights of shape [batch_size, nk] where nk is the size of the Key sequence length. batch_size = len(cond_uncond_idxs)
    """
    input_dim = q.shape[1]
    hs_dim = max(num_pos, num_neg) * input_dim
    dim = input_dim + hs_dim
    batched_weights = []
    for x in cond_uncond_idxs:
        weights = torch.ones(dim)
        if x == COND:
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
