import gradio as gr
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import random

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

def load_nitrofusion_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ChenDY/NitroFusion"
    ckpt = "nitrosd-realism_unet.safetensors"
    unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    scheduler = TimestepShiftLCMScheduler.from_pretrained(base_model_id, subfolder="scheduler", shifted_timestep=250)
    scheduler.config.original_inference_steps = 4
    pipe = DiffusionPipeline.from_pretrained(
        base_model_id,
        unet=unet,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    return pipe

pipe = load_nitrofusion_pipeline()

def generate_nitrofusion_images(prompt, num_images, width, height, steps, seed):
    images = []
    for i in range(num_images):
        current_seed = seed if seed != 0 else random.randint(0, 0xffffffffffffffff)
        generator = torch.manual_seed(current_seed + i) if seed != 0 else None
        image = pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=0,
            width=int(width),
            height=int(height),
            generator=generator
        ).images[0]
        images.append(image)
    return images

demo = gr.Interface(
    fn=generate_nitrofusion_images,
    inputs=[
        gr.Textbox(label="Enter a text prompt", placeholder="A photo of a sunrise in a futuristic city"),
        gr.Slider(minimum=1, maximum=20, step=1, label="Number of Images", value=1),
        gr.Slider(minimum=768, maximum=2048, step=8, label="Width", value=1024),
        gr.Slider(minimum=768, maximum=2048, step=8, label="Height", value=1024),
        gr.Slider(minimum=1, maximum=4, step=1, label="Number of Inference Steps", value=1),
        gr.Slider(minimum=0, maximum=0xffffffffffffffff, step=1, label="Seed (0 for random)", value=0)
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="NitroFusion 1 Step Text2Image",  
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()