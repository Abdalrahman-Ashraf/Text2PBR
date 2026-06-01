import os
from PIL import Image
import torch
from torchvision import transforms
import open_clip
from train_diffuser import load_autoencoderkl, load_unet2dconditionalmodel
from ldm_generator import LDM
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------#

def load_text_encoder(device):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-quickgelu', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    return model.eval().to(device), tokenizer, preprocess

#----------------------------------------------------------------------------------------------------------#

def get_text_features(texts, model, tokenizer):
    tokens = tokenizer(texts).cuda()
    with torch.no_grad():
        # Get token embeddings from transformer
        # model.transformer returns (seq_len, batch, dim)
        embedding = model.token_embedding(tokens)  # (batch, seq_len, dim)
        x = embedding + model.positional_embedding  # Add position
        x = x.permute(1, 0, 2)  # (seq_len, batch, dim)

        x = model.transformer(x)  # (seq_len, batch, dim)
        x = x.permute(1, 0, 2)    # (batch, seq_len, dim)

        # Optionally apply layernorm (like CLIP does before projection)
        x = model.ln_final(x)
    return x  # Shape: (batch_size, seq_len, hidden_dim)

#----------------------------------------------------------------------------------------------------------#

def test_encode_decode(image: Image.Image, vae: AutoencoderKL, device: torch.device) -> torch.Tensor:
    """
    Encode then decode an image through the VAE to check reconstruction quality.

    Steps:
    1) Preprocess image to [-1,1]
    2) vae.encode -> get latent_dist.mean (deterministic)
    3) vae.decode(latents) -> get reconstruction
    4) Clamp to [-1,1] and return [3,H,W]
    """
    # 1) preprocess
    if not isinstance(image, torch.Tensor):
        x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    else:
        x = image.to(device)
    x = x * 2 - 1                # scale from [0,1] to [-1,1]

    # 2) encode -> deterministic latent
    enc = vae.encode(x)
    latents = enc.latent_dist.mean  # shape [B,4,H/8,W/8]
    print(f"VAE encode output shape: {latents.shape}")

    # 3) decode raw latents
    dec = vae.decode(latents)
    recon = dec.sample.squeeze(0).cpu().clamp(-1, 1)

    return recon

#-------------------------------------------
# 3) Diffusion image generation
#-------------------------------------------
def generate_image(
    text: str,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    text_encoder,
    tokenizer,
    device: torch.device,
    resolution: int = 512,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 1000
) -> Image.Image:
    """
    Generate an image from text using Latent Diffusion:
    - initialize latents
    - diffusion loop with classifier-free guidance
    - apply SD scaling convention:  *0.18215 before UNet, /0.18215 before VAE.decode
    """
    # Prepare text embeddings
    text_emb = get_text_features([text], text_encoder, tokenizer)
    uncond_emb = get_text_features([""], text_encoder, tokenizer)
    context = torch.cat([uncond_emb, text_emb], dim=0)  # Shape: [2, embedding_dim]

    # Set timesteps
    noise_scheduler.set_timesteps(num_inference_steps)

    # Initialize random latents (SD convention)
    ds = resolution // 8
    latents = torch.randn((1, 4, ds, ds), device=device) * vae.config.scaling_factor

    # Optional live preview
    plt.ion(); fig, ax = plt.subplots()

    for i, t in enumerate(noise_scheduler.timesteps):
        # Duplicate for CFG
        lat_input = torch.cat([latents] * 2, dim=0)

        # Predict noise
        with torch.no_grad():
            noise_pred = unet(lat_input, t, encoder_hidden_states=context).sample

        # CFG combine
        uncond, cond = noise_pred.chunk(2, dim=0)
        guided = uncond + guidance_scale * (cond - uncond)

        # Step the scheduler
        latents = noise_scheduler.step(guided, t, latents).prev_sample

        # Live preview every N steps
        if i % 50 == 0 or i == len(noise_scheduler.timesteps)-1:
            with torch.no_grad():
                # Undo SD scaling before decode
                preview = vae.decode(latents / vae.config.scaling_factor).sample
                img_t = transforms.ToPILImage()(preview.squeeze(0).cpu().clamp(-1, 1))

            ax.clear()
            ax.imshow(img_t)
            ax.set_title(f"Step {i+1}/{len(noise_scheduler.timesteps)}")
            plt.pause(0.01)

    plt.ioff(); plt.show()
    return img_t

#----------------------------------------------------------------------------------------------------------#

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder, tokenizer, _ = load_text_encoder(device)

    autoencoder_model = load_autoencoderkl(device).eval()

    model_path = "pretrained_models\diff_image_generator_best_512px.pth"

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    resolution = 512

    unet_model = load_unet2dconditionalmodel(resolution)
    
    ldm = LDM(autoencoder_model, unet_model, text_encoder, noise_scheduler).to(device)

    ldm.load_state_dict(torch.load(model_path))
    unet_model = ldm.unet.to(device)
    unet_model.eval()

    while True:
        prompt = input("Enter the text prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break

        image = generate_image(prompt, unet_model, autoencoder_model, noise_scheduler, 
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