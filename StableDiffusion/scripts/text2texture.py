import sys
import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler

def generateTexture(prompt: str, texture_path: str, model_path="pretrained_models/textures_v0.21_last.pth") -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, "..", "pretrained_models/textures_v0.21_last.pth")

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(50)
    print(f"Timesteps: {len(pipe.scheduler.timesteps)-1}")

    checkpoint = torch.load(model_path, map_location="cpu")
    pipe.unet.load_state_dict(checkpoint)
    pipe.unet.to("cuda").eval()

    pipe.unet.to(dtype=torch.float16, device="cuda")

    image = pipe(prompt, guidance_scale=5.0).images[0]
    image.save(texture_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    generateTexture(args.prompt, args.output, args.model)