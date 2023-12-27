import torch
import numpy as np
import math
import comfy
from comfy.ldm.modules.diffusionmodules.util import extract_into_tensor, make_beta_schedule


def get_alphas_cumprod(model):
    """
    Same procedure as in comfy.model_sampling.ModelSamplingDiscrete.__init__
    """
    model_config = model.model.model_config
    sampling_settings = model_config.sampling_settings if model_config is not None else {}
    beta_schedule = sampling_settings.get("beta_schedule", "linear")
    linear_start = sampling_settings.get("linear_start", 0.00085)
    linear_end = sampling_settings.get("linear_end", 0.012)
    betas = make_beta_schedule(schedule=beta_schedule, n_timestep=1000,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=8e-3)
    alphas = 1. - betas
    return torch.tensor(np.cumprod(alphas, axis=0), dtype=torch.float32)


def q_sample(model, x_start, t):
    noise = torch.randn_like(x_start)
    alphas_cumprod = get_alphas_cumprod(model).to(x_start.device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    return (extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


def sigma_to_t(model, sigma):
    model_wrap = comfy.samplers.wrap_model(model.model)
    sampling = model_wrap.inner_model.model_sampling

    # LCM
    if str(sampling) == "ModelSamplingAdvanced()":
        timesteps = sampling.timestep(sigma)
        step_size = torch.round(timesteps[0] / (len(timesteps) - 2))
        # Attempt normalization (rounding errors, but works >_<)
        for i in range(1, len(timesteps)):
            timesteps[i] = max(math.ceil(timesteps[0] - (i * step_size)), 0)
        return timesteps

    return sampling.timestep(sigma)


def get_timesteps(model, steps, sampler, scheduler, denoise, device="cpu"):
    real_model = model.model
    sampler = comfy.samplers.KSampler(
        real_model, steps=steps, device=device, sampler=sampler,
        scheduler=scheduler, denoise=denoise, model_options=model.model_options
    )
    return sigma_to_t(model, sampler.sigmas)


def undo_scaling(model, sigma, noise):
    sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
    sigma_data = comfy.samplers.wrap_model(model.model).inner_model.model_sampling.sigma_data
    return noise * (sigma ** 2 + sigma_data ** 2) ** 0.5  # Multiply to cancel out the division in comfy
