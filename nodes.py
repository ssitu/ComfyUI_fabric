from nodes import KSampler, KSamplerAdvanced, CLIPTextEncode
from .fabric.fabric import fabric_sample, fabric_patch
import torch
import comfy
import warnings


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


class FABRICPatchModel:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
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
        clip = kwargs["clip"]
        clip_encode = CLIPTextEncode()
        null_cond = clip_encode.encode(clip, "")[0]
        del kwargs["clip"]
        kwargs["null_pos"] = null_cond
        kwargs["null_neg"] = null_cond
        return fabric_patch(*args, **kwargs)


class KSamplerAdvFABRICAdv:

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
        kwargs["denoise"] = 1.0  # Default value
        return fabric_sample(*args, **kwargs)


class KSamplerFABRICAdv:

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
        """
        Regular KSampler with all FABRIC inputs
        """
        # Add default advanced ksampler inputs
        kwargs["add_noise"] = False
        kwargs["start_at_step"] = None
        kwargs["end_at_step"] = None
        kwargs["return_with_leftover_noise"] = False
        kwargs["noise_seed"] = kwargs.pop("seed")
        return fabric_sample(*args, **kwargs)


class KSamplerFABRIC:

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
        """
        Regular KSampler with intended FABRIC inputs
        """
        clip_encode = CLIPTextEncode()
        null_cond = clip_encode.encode(kwargs["clip"], "")[0]
        del kwargs["clip"]
        kwargs["null_pos"] = null_cond
        kwargs["null_neg"] = null_cond

        # Convert feedback percent to start and end steps
        kwargs["feedback_start"] = 0
        kwargs["feedback_end"] = int(kwargs["steps"] * kwargs.pop("feedback_percent"))
        return KSamplerFABRICAdv().sample(*args, **kwargs)


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
        result = latent1.copy()
        result["samples"] = torch.cat((lat1, lat2), dim=0)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "FABRICPatchModelAdv": FABRICPatchModelAdv,
    "FABRICPatchModel": FABRICPatchModel,
    "KSamplerAdvFABRICAdv": KSamplerAdvFABRICAdv,
    "KSamplerFABRICAdv": KSamplerFABRICAdv,
    "KSamplerFABRIC": KSamplerFABRIC,
    "LatentBatch": LatentBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FABRICPatchModelAdv": "FABRIC Patch Model (Advanced)",
    "FABRICPatchModel": "FABRIC Patch Model",
    "KSamplerAdvFABRICAdv": "KSampler FABRIC (Advanced)",
    "KSamplerFABRICAdv": "KSampler FABRIC",
    "KSamplerFABRIC": "KSampler FABRIC (Simple)",
    "LatentBatch": "Batch Latents",
}
