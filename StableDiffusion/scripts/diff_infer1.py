from diffusers import StableDiffusionPipeline, DDPMScheduler, PNDMScheduler
import torch

# scheduler = PNDMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16,
    # scheduler=scheduler
).to("cuda")

checkpoint = torch.load("pretrained_models/textures_v0.21_last.pth", map_location="cpu")
pipe.unet.load_state_dict(checkpoint)
pipe.unet.to("cuda").eval()

pipe.unet.to(dtype=torch.float16, device="cuda")

for i in range(10):
    prompt = "wooden floor texture"
    image = pipe(prompt, guidance_scale=4.0).images[0]
    image.save(f"result_{i}_v0.131.png")
