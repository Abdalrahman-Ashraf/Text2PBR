import os
import csv
import time
import sys
import torch
import open_clip
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torchvision.transforms import ToPILImage
from torchvision import transforms
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from system_settings import *
from ldm_generator import LDM
import general_stats as g_stats
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

#----------------------------------------------------------------------------------------------------------#

MAX_RESOLUTION = 512

#----------------------------------------------------------------------------------------------------------#

def save_checkpoint(unet, g_optimizer, scaler, epoch, loss, best_loss, path):
    torch.save({
        'epoch': epoch,
        'unet': unet.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'loss': loss,
        'best_loss': best_loss
    }, path)

#----------------------------------------------------------------------------------------------------------#

def load_checkpoint(unet, g_optimizer, scaler, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")
    checkpoint = torch.load(path, map_location=device)

    unet.load_state_dict(checkpoint.get('unet', {}))
    g_optimizer.load_state_dict(checkpoint.get('g_optimizer', {}))
    scaler.load_state_dict(checkpoint.get('scaler', {}))

    start_epoch = checkpoint.get('epoch', 0) + 1
    loss = checkpoint.get('loss', None)
    best_loss = checkpoint.get('best_loss', None)

    print(f"Resumed from checkpoint at epoch {start_epoch + 1}")
    return unet, g_optimizer, scaler, start_epoch, loss, best_loss

#----------------------------------------------------------------------------------------------------------#

def load_stable_diffusion(device):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1"
    ).to(device)
    return sd_pipe

#----------------------------------------------------------------------------------------------------------#

def visualize_image(latents, images, dx_time=10, batch_idx=0, channel=0, img_stats=False):
    latent_img = latents[batch_idx, channel].detach().cpu().numpy()
    real_img = images[batch_idx].detach().cpu().numpy()
    real_img = real_img.transpose(1, 2, 0)

    if img_stats:
        print(f"Latent image stats: min {latent_img.min():.4f}, max {latent_img.max():.4f}, mean {latent_img.mean():.4f}")
        print(f"Real image stats: unique values {np.unique(real_img)} sum {real_img.sum():.4f}")

    # fig, axs = plt.subplots(2, 3, figsize=(8, 12))
    # axs = axs.flatten()
    # for ch in range(latents.shape[1]):
    #     latent_img = latents[batch_idx, ch].detach().cpu().numpy()
    #     axs[ch].imshow(latent_img, cmap='viridis')
    #     axs[ch].set_title(f"Latent Image Ch{ch}")
    #     axs[ch].axis('off')

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(latent_img, cmap='viridis')
    axs[0].set_title(f"Latent Image Ch{channel}")
    axs[0].axis('off')

    axs[1].imshow(real_img)
    axs[1].set_title("Real Image")
    axs[1].axis('off')

    # axs[5].axis('off')

    plt.show(block=False)
    plt.pause(dx_time)
    plt.close()

#----------------------------------------------------------------------------------------------------------#

def get_image_features(pil_image, model, preprocess):
    image = preprocess(pil_image).unsqueeze(0)
    image = image.to(next(model.parameters()).device)
    image = image.to(dtype=next(model.parameters()).dtype)

    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features

#----------------------------------------------------------------------------------------------------------#

def get_text_features(texts, model, tokenizer, device="cuda"):
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        output = model(**tokens)
        embedding = output.last_hidden_state  # (batch_size, seq_len, hidden_dim)

    return embedding  # Shape: (batch_size, seq_len, hidden_dim)

#----------------------------------------------------------------------------------------------------------#

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True, layers=[3, 8, 15]):
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16].eval()

        vgg = [layer if not isinstance(layer, nn.ReLU) else nn.ReLU(inplace=False) for layer in vgg]

        self.vgg = nn.Sequential(*[vgg[i] for i in layers]).eval()

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.vgg = self.vgg.cuda()

        self.resize = resize

    def forward(self, x, y):
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        x = (x - mean) / std
        y = (y - mean) / std

        return F.mse_loss(self.vgg(x), self.vgg(y))

#----------------------------------------------------------------------------------------------------------#

class ImagePromptDataset(Dataset):
    def __init__(self, image_dir, prompt_file, img_size=MAX_RESOLUTION):
        self.crop_transform = transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.ToTensor()
        ])
        self.image_dir = image_dir
        self.samples = []

        img_skipped = 0
        with open(prompt_file, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 2:
                    continue
                filename, prompt = row
                image_path = os.path.join(self.image_dir, filename)

                if not os.path.exists(image_path):
                    img_skipped += 1
                    continue

                self.samples.append((filename.strip(), prompt.strip()))

            if img_skipped == 0:
                print("All image paths found")
            else:
                print(f"Skipped {img_skipped} images due to unmatched paths")

        self.img_size = img_size

    def augment_image(self, image: Image):
        return self.crop_transform(image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]

        image_path = os.path.join(self.image_dir, filename)
    
        image = Image.open(image_path).convert("RGB")

        w, h = image.size
        if min(w, h) != self.img_size:
            image = F_t.resize(image, self.img_size, interpolation=Image.BICUBIC)

        image_tensor = self.augment_image(image) # A cropped image tensor 

        return text, image_tensor

#----------------------------------------------------------------------------------------------------------#

def train_model(
        dataset: ImagePromptDataset, 
        autoencoder: AutoencoderKL, 
        unet: UNet2DConditionModel, 
        noise_scheduler,  
        g_optimizer: optim.AdamW,
        text_encoder, 
        tokenizer, 
        device: str,
        current_dir,
        run_name = "image_generator",
        resolution=MAX_RESOLUTION,  
        batch_size=8, 
        num_workers=2, 
        num_epochs=100, 
        resume_training=True,
        visualize=False
    ):

    destination_dir = "pretrained_models"
    scaler = torch.amp.GradScaler()

    # File paths
    checkpoint_path = os.path.normpath(os.path.join(current_dir, '..', f'{destination_dir}/{run_name}_checkpoint_{resolution}px.pth'))
    final_model_path = os.path.normpath(os.path.join(current_dir, '..', f'{destination_dir}/{run_name}_final_{resolution}px.pth'))
    last_model_path = os.path.normpath(os.path.join(current_dir, '..', f'{destination_dir}/{run_name}_last_{resolution}px.pth'))
    best_model_path = os.path.normpath(os.path.join(current_dir, '..', f'{destination_dir}/{run_name}_best_{resolution}px.pth'))

    DATALOADER = DataLoader(
        dataset, 
        batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True
    )

    dataloader_len = len(DATALOADER)

    start_epoch = 0
    best_loss = float('inf')

    # vgg_loss_fn = VGGPerceptualLoss(layers=[3, 8, 15])

    unet.train()

    if resume_training and os.path.exists(checkpoint_path):
        (unet, g_optimizer, scaler, start_epoch, prev_loss, best_loss) = load_checkpoint(
             unet,
             g_optimizer, 
             scaler, 
             checkpoint_path, 
             device
        )
        print(f"Previous best loss: {best_loss:.6f}")

    for epoch in range(start_epoch, num_epochs):

        total_loss = 0.0

        for i, (texts, images) in enumerate(DATALOADER):
            if i % 2 == 0:
                throttle_gpu_hotspot_temp(throttle=True, display_thermals=False)

            images = images.to(device)

            cond_emb = get_text_features(list(texts), text_encoder, tokenizer, device=device)   # shape: [B, T, D]
            uncond_emb = get_text_features([""] * cond_emb.shape[0], text_encoder, tokenizer,device=device)  # [B, T, D]

            # 2. Stack them so your UNet sees 2B samples
            all_emb = torch.cat([uncond_emb, cond_emb], dim=0)  # [2B, T, D]
            
            with torch.amp.autocast(device_type="cuda"):

                latents = autoencoder.encode(images).latent_dist.sample() * autoencoder.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                latents_in = torch.cat([noisy_latents, noisy_latents], dim=0)   # [2B, …]
                t_in       = torch.cat([timesteps, timesteps], dim=0)         # [2B]

                noise_prediction = unet(
                    latents_in,
                    t_in,
                    encoder_hidden_states=all_emb
                )[0]

                # Image visualization
                if visualize:
                    visualize_image(noise_prediction, images, dx_time=20, batch_idx=i)

                uncond_pred, cond_pred = noise_prediction.chunk(2, dim=0)

                # MSE loss
                mse_loss = F.mse_loss(cond_pred, noise)
                lambda_mse = 1.0
                whl_mse = lambda_mse * mse_loss

                # Perceptual loss (no gradient)
                # perceptual_loss = vgg_loss_fn(predicted_image, images)
                # lambda_vgg = 0.1
                # whl_vgg = lambda_vgg * perceptual_loss

                cumulative_loss = whl_mse

            scaler.scale(cumulative_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()
            g_optimizer.zero_grad() 

            total_loss += cumulative_loss.item()
            if (i + 1) % 1 == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}], Iteration [{i+1}/{dataloader_len}], Loss: {cumulative_loss.item():.6f}")
                print(f" - MSE loss: {whl_mse.item():.6f}")

        avg_loss = total_loss / dataloader_len

        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.6f}")

        torch.save(unet.state_dict(), last_model_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(unet.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1}")

        # Training checkpoint
        save_checkpoint(unet, g_optimizer, scaler, epoch, avg_loss, best_loss, checkpoint_path)
        print(f"Saved checkpoint")

    print(f"Best loss: {best_loss:.6f}")
    torch.save(unet.state_dict(), final_model_path)
    print(f"Saved final model {run_name}")

#----------------------------------------------------------------------------------------------------------#

def autoencoder_debugging(
        dataset, 
        autoencoder, 
        device, 
        bt_pcs=1, 
        sf_start=0.8, 
        sf_end=1.0, 
        sf_step=0.05, 
        resolution=MAX_RESOLUTION, 
        batch_size=1, 
        num_workers=2,):

    DATALOADER = DataLoader(
        dataset, 
        batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True
    )

    destination_dir = "debug_outputs"

    with torch.no_grad():
        for i, (_, images) in enumerate(DATALOADER):
            if not i < bt_pcs:
                return

            # images = F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
            images = images.to(device)

            scale_factors = torch.arange(sf_start, sf_end + sf_step, sf_step)
            for sf in scale_factors:
                encoding = autoencoder.encode(images)
                latents = encoding.latent_dist.mode()
                latents = latents * sf

                recon = autoencoder.decode(latents).sample
                recon = recon / sf

                original_image_path = os.path.normpath(os.path.join(
                    destination_dir, f'original_{i}_{resolution}px.png'
                ))

                recon_image_path = os.path.normpath(os.path.join(
                    destination_dir, f'recon_sf{sf:.3f}_{i}_{resolution}px.png'
                ))

                save_image(images, original_image_path)
                save_image(recon, recon_image_path)
                if sf_start == sf_end:
                    break

#----------------------------------------------------------------------------------------------------------#

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    sd_pipe = load_stable_diffusion(device)

    text_encoder = sd_pipe.text_encoder
    tokenizer = sd_pipe.tokenizer
    unet = sd_pipe.unet
    vae = sd_pipe.vae.eval()
    noise_scheduler = sd_pipe.scheduler

    print(type(text_encoder))
    print(type(tokenizer))
    print(type(unet))
    print(type(vae))
    print(type(noise_scheduler))
    print(f"VAE scaling factor: {vae.config.scaling_factor}")

    for name, p in unet.named_parameters():
        if "mid_block" not in name and "up_blocks.3" not in name:
            p.requires_grad = False

    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    g_optimizer = optim.AdamW(unet.parameters(), lr=1e-5)

    dataset_prompt_path = os.path.normpath(
        os.path.join(current_dir, '..', '..', 'data/artstation/Character Design/prompts/character_captions.csv')
    )
    dataset_image_path = os.path.normpath(
        os.path.join(current_dir, '..', '..', 'data/artstation/Character Design/3_512px')
    )

    dataset = ImagePromptDataset(dataset_image_path, dataset_prompt_path, img_size=MAX_RESOLUTION)

    # autoencoder_debugging(dataset, autoencoder, device, bt_pcs=6, sf_start=1.0, sf_end=1.0, sf_step=0.1, batch_size=32)

    train_model(
        dataset         = dataset, 
        autoencoder     = vae, 
        unet            = unet, 
        noise_scheduler = noise_scheduler, 
        g_optimizer     = g_optimizer, 
        text_encoder    = text_encoder, 
        tokenizer       = tokenizer, 
        device          = device, 
        current_dir     = current_dir, 
        run_name        = "test3_v0.10", 
        resolution      = MAX_RESOLUTION, 
        batch_size      = 1, 
        num_workers     = 1, 
        num_epochs      = 5, 
        resume_training = False, 
        visualize       = False
    )

#----------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()