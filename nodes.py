from nodes import KSampler, KSamplerAdvanced
from .fabric.fabric import fabric_sample, ksampler_advfabric, ksampler_fabric, fabric_patch
import torch
import comfy
import warnings


class KSamplerFABRIC:
    @classmethod
    def INPUT_TYPES(s):
        inputs = KSampler.INPUT_TYPES()
        added_inputs = {
            "required": {
                "null_pos": ("CONDITIONING",),
                "null_neg": ("CONDITIONING",),
                "pos_weight": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01}),
                "neg_weight": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01}),
                "feedback_start": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "feedback_end": ("INT", {"default": 20, "min": 0, "max": 10000, "step": 1}),
            },
            "optional": {
                "pos_latents": ("LATENT",),
                "neg_latents": ("LATENT",),
            }
        }
        inputs["required"].update(added_inputs["required"])
        if "optional" not in inputs:
            inputs["optional"] = {}
        inputs["optional"].update(added_inputs["optional"])
        return inputs

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "FABRIC"

    def sample(self, *args, **kwargs):
        return ksampler_advfabric(*args, **kwargs)


class KSamplerAdvFABRIC:
    @classmethod
    def INPUT_TYPES(s):
        inputs = KSamplerAdvanced.INPUT_TYPES()
        added_inputs = {
            "required": {
                "null_pos": ("CONDITIONING",),
                "null_neg": ("CONDITIONING",),
                "pos_weight": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01}),
                "neg_weight": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01}),
                "feedback_start": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "feedback_end": ("INT", {"default": 10000, "min": 0, "max": 10000, "step": 1}),
            },
            "optional": {
                "pos_latents": ("LATENT",),
                "neg_latents": ("LATENT",),
            }
        }
        inputs["required"].update(added_inputs["required"])
        if "optional" not in inputs:
            inputs["optional"] = {}
        inputs["optional"].update(added_inputs["optional"])
        return inputs

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "FABRIC"

    def sample(self, *args, **kwargs):
        kwargs["denoise"] = 1.0
        return fabric_sample(*args, **kwargs)


class KSamplerFABRICSimple:
    @classmethod
    def INPUT_TYPES(s):
        inputs = KSampler.INPUT_TYPES()
        added_inputs = {
            "required": {
                "clip": ("CLIP",),
                "pos_weight": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01}),
                "neg_weight": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01}),
                "feedback_percent": ("FLOAT", {"default": 0.8, "min": 0., "max": 1., "step": 0.01}),
            },
            "optional": {
                "pos_latents": ("LATENT",),
                "neg_latents": ("LATENT",),
            }
        }
        inputs["required"].update(added_inputs["required"])
        if "optional" not in inputs:
            inputs["optional"] = {}
        inputs["optional"].update(added_inputs["optional"])
        return inputs

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "FABRIC"

    def sample(self, *args, **kwargs):
        return ksampler_fabric(*args, **kwargs)


class LatentBatch:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent1": ("LATENT",), "latent2": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "batch"

    CATEGORY = "FABRIC"

    def batch(self, latent1, latent2):
        lat1 = latent1["samples"]
        lat2 = latent2["samples"]
        if lat1.shape[1:] != lat2.shape[1:]:
            warnings.warn("Latent shapes do not match, resizing latent2 to match latent1")
            lat2 = comfy.utils.common_upscale(lat2, lat1.shape[3], lat1.shape[2], "bilinear", "center")
        result = torch.cat((lat1, lat2), dim=0)
        latent1["samples"] = result
        return (latent1,)


class FABRICPatchModelAdv:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "null_pos": ("CONDITIONING",),
                "null_neg": ("CONDITIONING",),
                "pos_weight": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01}),
                "neg_weight": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01}),
            },
            "optional": {
                "pos_latents": ("LATENT",),
                "neg_latents": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FABRIC"

    def patch(self, *args, **kwargs):
        return fabric_patch(*args, **kwargs)


NODE_CLASS_MAPPINGS = {
    "KSamplerFABRIC": KSamplerFABRIC,
    "KSamplerAdvFABRIC": KSamplerAdvFABRIC,
    "KSamplerFABRICSimple": KSamplerFABRICSimple,
    "LatentBatch": LatentBatch,
    "FABRICPatchModelAdv": FABRICPatchModelAdv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerFABRIC": "KSampler With FABRIC",
    "KSamplerAdvFABRIC": "KSampler FABRIC (Advanced)",
    "KSamplerFABRICSimple": "KSampler FABRIC (Simple)",
    "LatentBatch": "Batch Latents",
    "FABRICPatchModelAdv": "FABRIC Patch Model (Advanced)",
}
