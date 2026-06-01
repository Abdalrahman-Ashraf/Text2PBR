import torch
from diffusers import StableDiffusionPipeline

# Load pretrained model
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",  # or "CompVis/stable-diffusion-v1-4"
    torch_dtype=torch.float16
).to("cuda")

pipe.unet.config.use_tiled_vae = True  # optional for very large textures
pipe.vae.enable_tiling()               # <--- This makes the output tileable!

# Generate a tileable texture from prompt
prompt = "Rock with moss, seamless texture"
image = pipe(prompt, guidance_scale=7.5).images[0]

# Save texture
image.save("tileable_texture2.png")