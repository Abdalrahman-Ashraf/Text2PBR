from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# === Setup ===
device = "cuda"
model_path = "pretrained_models/test_v0.10_last_512px.pth"

# Load base Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to(device)

# === Load your trained UNet weights ===
checkpoint = torch.load(model_path, map_location="cpu")
if 'model' in checkpoint:
    pipe.unet.load_state_dict(checkpoint['model'])
else:
    pipe.unet.load_state_dict(checkpoint)

pipe.unet.to(dtype=torch.float16, device=device)
pipe.unet.eval()

# Optional: Reduce memory usage
pipe.enable_attention_slicing()

# === Inference ===
prompt = "a woman with red hair holding a sword"
result = pipe(prompt, guidance_scale=7.5, num_inference_steps=50)
image = result.images[0]

# Save the output
output_path = "result4.png"
image.save(output_path)
print(f"Saved result to {output_path}")
