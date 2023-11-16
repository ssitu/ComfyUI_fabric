import torch
import numpy as np
import comfy
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
from comfy.ldm.modules.diffusionmodules.util import extract_into_tensor
from comfy.ldm.modules.attention import SpatialTransformer
import math

def q_sample(model, x_start, timestep, ts_interval):
    noise = None

    ts_start, ts_end = ts_interval
    num_ts = max(ts_start - ts_end, 0)
    
    config = model.model.model_config
    model_wrap = comfy.samplers.wrap_model(model.model)
    beta_schedule = "linear"
    if config is not None:
        beta_schedule = config.beta_schedule

    # Calc betas and alphas_cumprod
    betas = make_beta_schedule(beta_schedule, num_ts)

    alphas = 1. - betas
    alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0), dtype=torch.float32)
    
    device = x_start.device

    timestep = timestep.long()
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(device)
    noise = torch.randn_like(x_start).to(device)
    return (extract_into_tensor(sqrt_alphas_cumprod, timestep, x_start.shape) * x_start +
            extract_into_tensor(sqrt_one_minus_alphas_cumprod, timestep, x_start.shape) * noise)


def get_timesteps(model, steps, sampler, scheduler, denoise, device="cpu"):
    real_model = model.model
    sampler = comfy.samplers.KSampler(
        real_model, steps=steps, device=device, sampler=sampler,
        scheduler=scheduler, denoise=denoise, model_options=model.model_options
    )

    model_wrap = comfy.samplers.wrap_model(real_model)
    sampling = model_wrap.inner_model.model_sampling
    # LCM
    if str(sampling) == "ModelSamplingAdvanced()":
        timesteps = sampling.timestep(sampler.sigmas)
        
        step_size = torch.round(timesteps[0] / (len(timesteps) - 2))
        # Attempt normalization (rounding errors, but works >_<)
        for i in range(1, len(timesteps)):
            timesteps[i] = max(math.ceil(timesteps[0] - (i * step_size)), 0)
        return timesteps

    return sampling.timestep(sampler.sigmas)



def forward(model, steps, sampler, scheduler, denoise, device, zs, ts, pos, neg, seed):
    real_model = model.model
    sampler = comfy.samplers.KSampler(
        real_model, steps=steps, device=device, sampler=sampler,
        scheduler=scheduler, denoise=denoise, model_options=model.model_options
    )
    sigma = sampler.model_wrap.t_to_sigma(ts)
    out = sampler.model_wrap(zs, sigma, cond=pos, uncond=neg, cond_scale=1,
                             model_options=model.model_options, seed=seed)
    return out


def get_transformer_blocks(model):
    blocks = []
    for module in model.model.diffusion_model.modules():
        if isinstance(module, SpatialTransformer):
            blocks.append(module)
    return blocks
