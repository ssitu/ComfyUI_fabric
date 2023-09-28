import torch
import comfy
from comfy.ldm.modules.diffusionmodules.util import extract_into_tensor
from comfy.ldm.modules.attention import SpatialTransformer


def q_sample(model, x_start, t):
    noise = torch.randn_like(x_start)
    alphas_cumprod = model.model.get_buffer('alphas_cumprod')
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    return (extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


def get_timesteps(model, steps, sampler, scheduler, denoise, device="cpu"):
    real_model = model.model
    sampler = comfy.samplers.KSampler(
        real_model, steps=steps, device=device, sampler=sampler,
        scheduler=scheduler, denoise=denoise, model_options=model.model_options
    )
    model_wrap = comfy.samplers.wrap_model(real_model)
    return model_wrap.sigma_to_discrete_timestep(sampler.sigmas)



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
