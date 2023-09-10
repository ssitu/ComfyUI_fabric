# ComfyUI_fabric
 ComfyUI nodes based on the paper "FABRIC: Personalizing Diffusion Models with Iterative Feedback" (Feedback via Attention-Based Reference Image Conditioning)

Paper: https://arxiv.org/abs/2307.10159

Main Repo: https://github.com/sd-fabric/fabric

SD Web UI Extension: https://github.com/dvruette/sd-webui-fabric

## Installation

This has been tested for ComfyUI for the following commit: [07691e80c3bf9be16c629169e259105ca5327bf0](https://github.com/comfyanonymous/ComfyUI/commit/07691e80c3bf9be16c629169e259105ca5327bf0)

Navigate to `ComfyUI/custom_nodes/` and run the following command:
```
git clone https://github.com/ssitu/ComfyUI_fabric
```

## Usage

Nodes can be found in the node menu under `FABRIC/`:

| Node                          | Description                                                                                                            |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------|
| FABRIC Patch Model            | Patch a model to use FABRIC so you can use it in any sampler node.                                                     |
| FABRIC Patch Model (Advanced) | Same as the basic model patcher but with the null_pos and null_neg inputs instead of a clip input.                     |
| KSampler FABRIC               | Has the same inputs as a KSampler but with full FABRIC inputs.                                                         |
| KSampler FABRIC (Advanced)    | Has the same inputs as an Advanced KSampler but with full FABRIC inputs.                                               |
| KSampler FABRIC (Simple)      | Same inputs of a KSampler with the simplified (intended) FABRIC inputs.                                                |
| Batch Latents                 | Helper node for adding two latents together in a batch. Useful for using multiple positive/negative latents in FABRIC. |

## Parameters' Descriptions

| Parameter        | Description                                                                                                                                                                                                         |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| pos_latents      | Optional input for latents that you want the resulting latent to borrow characteristics from (e.g. "liked" images).                                                                                                 |
| neg_latents      | Optional input for latents that you want the resulting latent to avoid characteristics of (e.g. "disliked" images).                                                                                                 |
| pos_weight       | The weight of the positive latents.                                                                                                                                                                                 |
| neg_weight       | The weight of the negative latents.                                                                                                                                                                                 |
| null_pos         | The conditioning for computing the hidden states of the positive latents. Intended to just be an empty clip text embedding (output from an empty clip text encode), but it might be interesting to experiment with. |
| null_neg         | Same as null_pos but for negative latents.                                                                                                                                                                          |
| feedback_start   | The step to start applying feedback.                                                                                                                                                                                |
| feedback_end     | The step to stop applying feedback.                                                                                                                                                                                 |
| feedback_percent | The percentage of steps to apply feedback (e.g. if set to 0.8, the first 80% of the steps will have feedback)                                                                                                       |

## Tips

* Input latent, pos_latents, and neg_latents should all be the same size. If they are not, they will be resized to the size of the input latent using bilinear interpolation, which is not a good way to resize latents so resize them in pixel space or use a model to resize the latents.
* Pay attention to the pos/neg weights. The default value of 1.0 is probably too high in most cases.
* The researchers recommend to only apply feedback to the first half of the denoising steps.
* If you are having out of memory errors, try switching cross attention methods or use a smaller batch of positive/negative latents.

## Examples
Round by round feedback:
![image](https://github.com/ssitu/ComfyUI_fabric/assets/57548627/5bc67956-f41c-4c50-8641-a0d45347afc6)

FABRIC patch model:
![image](https://github.com/ssitu/ComfyUI_fabric/assets/57548627/24eadcd1-f815-45a8-be18-a54ed17d705b)



