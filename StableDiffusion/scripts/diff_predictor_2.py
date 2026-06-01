import os
from PIL import Image
import torch
from torchvision import transforms
import open_clip
from train_diff import load_stable_diffusion
from ldm_generator import LDM
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------#

def load_text_encoder(device):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    return model.eval().to(device), tokenizer, preprocess

#----------------------------------------------------------------------------------------------------------#

def get_text_features(texts, text_encoder, tokenizer):
    inputs = tokenizer(texts, padding="max_length", max_length=77, return_tensors="pt").to(text_encoder.device)
    outputs = text_encoder(**inputs)
    return outputs.last_hidden_state  # shape: [B, 77, 768]

#----------------------------------------------------------------------------------------------------------#

def test_encode_decode(image: Image.Image, vae: AutoencoderKL, device: torch.device) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    else:
        x = image.to(device)
    x = x * 2 - 1                   # scale from [0,1] to [-1,1]

    enc = vae.encode(x)
    latents = enc.latent_dist.mean  # shape [B,4,H/8,W/8]
    print(f"VAE encode output shape: {latents.shape}")

    dec = vae.decode(latents)
    recon = dec.sample.squeeze(0).cpu().clamp(-1, 1)

    return recon

#----------------------------------------------------------------------------------------------------------#

def generate_image(
    text: str,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler,
    text_encoder,
    tokenizer,
    device: torch.device,
    resolution: int = 512,
    guidance_scale: float = 5.0,
    autoencoder_sf: float = 0.18215,
) -> Image.Image:
    """
    Generates an image from text using Latent Diffusion
    """

    vae.eval()
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    text_emb = get_text_features([text], text_encoder, tokenizer)
    uncond_emb = get_text_features([""], text_encoder, tokenizer)
    context = torch.cat([uncond_emb, text_emb], dim=0)  # Shape: [2, embedding_dim]

    ds = resolution // 8
    latents = torch.randn((1, 4, ds, ds), device=device) * autoencoder_sf

    plt.ion(); fig, ax = plt.subplots()

    for i, t in enumerate(noise_scheduler.timesteps):
        lat_input = torch.cat([latents] * 2, dim=0)

        with torch.no_grad():
            noise_pred = unet(lat_input, t, encoder_hidden_states=context).sample

        uncond, cond = noise_pred.chunk(2, dim=0)
        guided = uncond + guidance_scale * (cond - uncond)

        latents = noise_scheduler.step(guided, t, latents).prev_sample

        if i % 20 == 0 or i == len(noise_scheduler.timesteps)-1:
            with torch.no_grad():
                preview = vae.decode(latents / autoencoder_sf).sample
                img_t = transforms.ToPILImage()(preview.squeeze(0).cpu().clamp(-1, 1))

            ax.clear()
            ax.imshow(img_t)
            ax.set_title(f"Step {i+1}/{len(noise_scheduler.timesteps)}")
            plt.pause(0.5)

    plt.ioff(); plt.show()
    return img_t


#----------------------------------------------------------------------------------------------------------#

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder, tokenizer, _ = load_text_encoder(device)

    sd_pipe = load_stable_diffusion(device)

    model_path = "pretrained_models/test2_v0.10_final_512px.pth"

    timesteps = 50

    resolution = 512

    text_encoder = sd_pipe.text_encoder
    tokenizer = sd_pipe.tokenizer
    unet = sd_pipe.unet
    vae = sd_pipe.vae
    noise_scheduler = sd_pipe.scheduler

    print(f"Timesteps: {len(noise_scheduler.timesteps)}")

    noise_scheduler.set_timesteps(timesteps)

    unet.load_state_dict(torch.load(model_path))
    unet.to(device).eval()

    while True:
        prompt = input("Enter the text prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break

        image = generate_image(prompt, unet, vae, noise_scheduler, 
                               text_encoder, tokenizer, device, resolution=resolution)

        image_name = prompt.replace(" ", "_")[:50]
        filename = image_name + ".png"
        save_path = f"outputs/{filename}"
        image.save(save_path)

        img_num = 1
        while True:
            new_filename = f"{image_name}{img_num}.png"
            previous_path = f"previous_outputs/{new_filename}"
            if not os.path.exists(previous_path):
                image.save(previous_path)
                break
            img_num += 1

        print(f"Image saved to {save_path}")

if __name__ == "__main__":
    main()