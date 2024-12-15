import gradio as gr
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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

def generate_live_nitrofusion_image(prompt, seed):
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=0,
        width=512,
        height=512,
        generator=generator
    ).images[0]
    return image

demo = gr.Interface(
    fn=generate_live_nitrofusion_image,
    inputs=[
        gr.Textbox(label="Enter a text prompt", placeholder="A photo of a sunrise in a futuristic city"),
        gr.Slider(minimum=0, maximum=0xffffffffffffffff, step=1, label="Seed", value=0)
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Live NitroFusion 1 Step Text2Image",  
    live=True
)

if __name__ == "__main__":
    demo.launch()