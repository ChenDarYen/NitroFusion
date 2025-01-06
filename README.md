# NitroFusion: High-Fidelity Single-Step Diffusion through Dynamic Adversarial Training

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ChenDY/NitroFusion_1step_T2I)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/ChenDY/NitroFusion)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://chendaryen.github.io/NitroFusion.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2412.02030-b31b1b.svg)](https://arxiv.org/abs/2412.02030)
[![Page Views Count](https://badges.toozhao.com/badges/01JEECFJR1K7PQNDFJ9G2S60AJ/blue.svg)](https://badges.toozhao.com/stats/01JEECFJR1K7PQNDFJ9G2S60AJ "Get your own page views count badge on badges.toozhao.com")


![](./assets/banner.jpg)

## Abstract

We introduce NitroFusion, a fundamentally different approach to single-step diffusion that achieves high-quality generation through a dynamic adversarial framework. While one-step methods offer dramatic speed advantages, they typically suffer from quality degradation compared to their multi-step counterparts. Just as a panel of art critics provides comprehensive feedback by specializing in different aspects like composition, color, and technique, our approach maintains a large pool of specialized discriminator heads that collectively guide the generation process. Each discriminator group develops expertise in specific quality aspects at different noise levels, providing diverse feedback that enables high-fidelity one-step generation. Our framework combines: (i) a dynamic discriminator pool with specialized discriminator groups to improve generation quality, (ii) strategic refresh mechanisms to prevent discriminator overfitting, and (iii) global-local discriminator heads for multi-scale quality assessment, and unconditional/conditional training for balanced generation. Additionally, our framework uniquely supports flexible deployment through bottom-up refinement, allowing users to dynamically choose between 1-4 denoising steps with the same model for direct quality-speed trade-offs. Through comprehensive experiments, we demonstrate that NitroFusion significantly outperforms existing single-step methods across multiple evaluation metrics, particularly excelling in preserving fine details and global consistency.

## ⏳ Coming Soon
- [ ] Local real-time interactive app
- [ ] Training script

## Model Weights 

Please check out our [**Hugging Face Model**](https://huggingface.co/ChenDY/NitroFusion).

Also have fun with the [**ComfyUI**](https://github.com/ChenDarYen/ComfyUI-TimestepShiftModel/tree/main) or [**Online Demo**](https://huggingface.co/spaces/ChenDY/NitroFusion_1step_T2I)!

## Usage

First, we  need to implement the scheduler with timestep shift for multi-step inference:
```python
from diffusers import LCMScheduler
class TimestepShiftLCMScheduler(LCMScheduler):
    def __init__(self, *args, shifted_timestep=250, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_to_config(shifted_timestep=shifted_timestep)
    def set_timesteps(self, *args, **kwargs):
        super().set_timesteps(*args, **kwargs)
        self.origin_timesteps = self.timesteps.clone()
        self.shifted_timesteps = (self.timesteps * self.config.shifted_timestep / self.config.num_train_timesteps).long()
        self.timesteps = self.shifted_timesteps
    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        if self.step_index is None:
            self._init_step_index(timestep)
        self.timesteps = self.origin_timesteps
        output = super().step(model_output, timestep, sample, generator, return_dict)
        self.timesteps = self.shifted_timesteps
        return output
```


We can then utilize the diffuser pipeline:
```python
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
# Load model.
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ChenDY/NitroFusion"
# NitroSD-Realism
ckpt = "nitrosd-realism_unet.safetensors"
unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
scheduler = TimestepShiftLCMScheduler.from_pretrained(base_model_id, subfolder="scheduler", shifted_timestep=250)
scheduler.config.original_inference_steps = 4
# # NitroSD-Vibrant
# ckpt = "nitrosd-vibrant_unet.safetensors"
# unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda", torch.float16)
# unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
# scheduler = TimestepShiftLCMScheduler.from_pretrained(base_model_id, subfolder="scheduler", shifted_timestep=500)
# scheduler.config.original_inference_steps = 4
pipe = DiffusionPipeline.from_pretrained(
    base_model_id,
    unet=unet,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
prompt = "a photo of a cat"
image = pipe(
    prompt=prompt,
    num_inference_steps=1,  # NotroSD-Realism and -Vibrant both support 1 - 4 inference steps.
    guidance_scale=0,
).images[0]
```

## ComfyUI Usage

Please refer to [https://github.com/ChenDarYen/ComfyUI-TimestepShiftModel/tree/main](https://github.com/ChenDarYen/ComfyUI-TimestepShiftModel/tree/main) for more details.


## License

NitroSD-Realism is released under [cc-by-nc-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en), following its base model *DMD2*.

NitroSD-Vibrant is released under [openrail++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md).


## Citation 

If you find NitroFusion is useful or relevant to your research, please kindly cite our work:

```bib
@article{chen2024nitrofusionhighfidelitysinglestepdiffusion,
    title={NitroFusion: High-Fidelity Single-Step Diffusion through Dynamic Adversarial Training},
    author={Dar-Yen Chen and Hmrishav Bandyopadhyay and Kai Zou and Yi-Zhe Song},
    booktitle={arXiv preprint arxiv:2412.02030},
    year={2024}
}
```

