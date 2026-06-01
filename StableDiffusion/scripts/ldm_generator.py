import torch
from torch import nn
from diffusers import DDPMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from open_clip import tokenize

class LDM(nn.Module):
    printed_debug = False
    def __init__(
        self,
        autoencoder: AutoencoderKL,
        unet: UNet2DConditionModel,
        text_encoder: nn.Module,
        noise_scheduler: DDPMScheduler,
        debug_prints = False,
        autoencoder_sf = 0.18215
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.unet = unet
        self.text_encoder = text_encoder
        self.noise_scheduler = noise_scheduler
        self.debug_prints = debug_prints
        self.autoencoder_sf = autoencoder_sf

        self.unet.enable_gradient_checkpointing()
        self.autoencoder.enable_gradient_checkpointing()

    def encode_prompt(self, text_list, device):
        tokens = tokenize(text_list).to(device)                     # Shape (B, 77)

        with torch.no_grad():
            text_tensor = self.text_encoder.encode_text(tokens)     # Shape (B, D)
            text_tensor = text_tensor.unsqueeze(1).repeat(1, 77, 1) # Shape (B, 77, D)
            return text_tensor

    def forward(self, image, text, device, noise=None, timesteps=None):
        with torch.no_grad():
            encoding = self.autoencoder.encode(image)
            latents = encoding.latent_dist.sample()
            latents = latents * self.autoencoder_sf

        if noise is None:
            noise = torch.randn_like(latents)

        if timesteps is None:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (latents.size(0),), device=latents.device).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.encode_prompt(text, device)

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        alpha_cumprod = self.noise_scheduler.alphas_cumprod.to(latents.device)
        
        alpha_t = alpha_cumprod[timesteps].reshape(-1, 1, 1, 1)

        x_start = (noisy_latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        with torch.no_grad():
            predicted_image = self.autoencoder.decode(x_start / self.autoencoder_sf).sample

        if not self.printed_debug and self.debug_prints:
            self.printed_debug = True
            print(f"Input image: {image.shape}, {image.dtype}")
            print("Latents:", latents.shape, latents.dtype)
            print("Noise:", noise.shape, noise.dtype)
            print(f"Timesteps: {timesteps.shape}, {timesteps.dtype}, {timesteps}")
            print(f"Noisy latents:  {noisy_latents.shape}, {noisy_latents.dtype}")
            print(f"Encoder hidden states: {encoder_hidden_states.shape}, {encoder_hidden_states.dtype}")
            print(f"Noise prediction output: {noise_pred.shape}, {noise_pred.dtype}")

        return noise_pred, noise, predicted_image
