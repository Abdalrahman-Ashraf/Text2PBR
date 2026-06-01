import os
import csv
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
import open_clip
from PIL import Image
from diffusers import StableDiffusionPipeline, PNDMScheduler, DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from system_settings import *
import general_stats as g_stats
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from torchvision.utils import save_image

#----------------------------------------------------------------------------------------------------------#

MAX_RESOLUTION = 512

#----------------------------------------------------------------------------------------------------------#

def save_checkpoint(unet, scaler, epoch, loss, best_loss, path):
    torch.save({
        'epoch': epoch,
        'unet': unet.state_dict(),
        'scaler': scaler.state_dict(),
        'loss': loss,
        'best_loss': best_loss
    }, path)

#----------------------------------------------------------------------------------------------------------#

def load_checkpoint(unet, scaler, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")
    checkpoint = torch.load(path, map_location=device)

    unet.load_state_dict(checkpoint.get('unet', {}))
    scaler.load_state_dict(checkpoint.get('scaler', {}))

    start_epoch = checkpoint.get('epoch', 0) + 1
    loss = checkpoint.get('loss', None)
    best_loss = checkpoint.get('best_loss', None)

    print(f"Resumed from checkpoint at epoch {start_epoch + 1}")
    return unet, scaler, start_epoch, loss, best_loss

#----------------------------------------------------------------------------------------------------------#

def load_stable_diffusion(device):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1"
    ).to(device)
    return sd_pipe

#----------------------------------------------------------------------------------------------------------#

def visualize_image(latents, images, dx_time=10, batch_idx=0, channel=0, img_stats=False):
    if latents.size(0) == 0 or images.size(0) == 0:
        print("Skipping visualization: empty batch")
        return

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

def get_openclip_text_features(texts, tokenizer, model):
    tokens = tokenizer(texts).cuda()
    with torch.no_grad():
        return model.encode_text(tokens)

#----------------------------------------------------------------------------------------------------------#

def get_openclip_image_features(images, model):
    if images.shape[-1] != 224 or images.shape[-2] != 224:
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

    # Normalize using OpenCLIP mean/std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    images = (images - mean) / std

    with torch.no_grad():
        encoded_image = model.encode_image(images).unsqueeze(1).repeat(1, 77, 1)
    
    return encoded_image

#----------------------------------------------------------------------------------------------------------#

def get_text_features(texts, model, tokenizer, device="cuda"):
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        output = model(**tokens)
        embedding = output.last_hidden_state  # (batch_size, seq_len, hidden_dim)

    return embedding  # Shape: (batch_size, seq_len, hidden_dim)

#----------------------------------------------------------------------------------------------------------#

def get_openclip_text_features(texts, tokenizer, model, device="cuda"):
    tokens = tokenizer(texts).to(device)  # OpenCLIP tokenizer returns token IDs directly
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding.unsqueeze(1).repeat(1, 77, 1)
    return embedding  # Shape: (batch_size, 77, embed_dim)

#----------------------------------------------------------------------------------------------------------#

def contrastive_loss(text_features, image_features, margin=0.0):
    similarity = F.cosine_similarity(text_features, image_features, dim=-1)
    loss = 1 - similarity
    if margin > 0.0:
        loss = F.relu(loss - margin)
    return loss.mean()

#----------------------------------------------------------------------------------------------------------#

def init_diffusion_model(device):
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")

    text_encoder, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', pretrained='laion2b_s32b_b79k'
    )

    text_encoder.eval()
    text_encoder = text_encoder.to(device=device)
    tokenizer = open_clip.get_tokenizer('ViT-H-14')

    vae.requires_grad_(False)
    vae.eval()
    vae.to(dtype=torch.float32, device=device)

    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder.to(dtype=torch.float32, device=device)

    return vae, noise_scheduler, text_encoder, tokenizer

#----------------------------------------------------------------------------------------------------------#

def print_model_types(text_encoder, tokenizer, unet, autoencoder, noise_scheduler):
    print(type(text_encoder))
    print(type(tokenizer))
    print(type(unet))
    print(type(autoencoder))
    print(type(noise_scheduler))
    print(f"VAE scaling factor: {autoencoder.config.scaling_factor}")
    print(f"Noise scheduler timesteps: {noise_scheduler.config.num_train_timesteps}")

    for name, p in unet.named_parameters():
        if "mid_block" not in name and "up_blocks.3" not in name:
            p.requires_grad = False

    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

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
    def __init__(self, image_dir, prompt_file, img_size=MAX_RESOLUTION, wnd_size=MAX_RESOLUTION):

        self.augment_transforms = transforms.Compose([
            transforms.RandomCrop(wnd_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(    
                brightness=0.1,
                contrast=0.1,
                saturation=0.08,
                hue=0.04
            ),
            transforms.ToTensor()
        ])

        self.img_size = img_size
        self.wnd_size = wnd_size
        self.image_dir = image_dir
        self.samples = []

        img_skipped = 0
        with open(prompt_file, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
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

    def augment_image(self, image: Image):
        return self.augment_transforms(image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]

        image_path = os.path.join(self.image_dir, filename)
    
        image = Image.open(image_path).convert("RGB")

        w, h = image.size
        if min(w, h) != self.img_size:
            image = F_t.resize(image, self.img_size, interpolation=Image.BICUBIC)

        image_tensor = self.augment_image(image)

        return text, image_tensor

#----------------------------------------------------------------------------------------------------------#

def train_model(
        dataset: ImagePromptDataset, 
        device: str,
        current_dir: str,
        run_name: str,
        checkpoint_name: str,
        resolution=MAX_RESOLUTION,  
        batch_size=8, 
        num_workers=2, 
        num_epochs=10, 
        resume_training=True,
        visualize=False,
        stat_freq=10, 
        learning_rate=1e-5
    ):

    destination_dir = "pretrained_models"
    scaler = torch.amp.GradScaler()

    (autoencoder, noise_scheduler, text_encoder, tokenizer) = init_diffusion_model(device)

    # Model paths
    prev_ckpt_path = os.path.normpath(os.path.join(current_dir, '..', destination_dir, f'{checkpoint_name}.pth'))
    checkpoint_path = os.path.normpath(os.path.join(current_dir, '..', f'{destination_dir}/{run_name}_checkpoint.pth'))
    last_model_path = os.path.normpath(os.path.join(current_dir, '..', f'{destination_dir}/{run_name}_last.pth'))

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

    vgg_loss_fn = VGGPerceptualLoss(layers=[3, 8, 15])

    if resume_training and os.path.exists(prev_ckpt_path):
        unet_config = UNet2DConditionModel.load_config("stabilityai/stable-diffusion-2-1-base", subfolder="unet")
        unet = UNet2DConditionModel.from_config(unet_config)
        (unet, scaler, start_epoch, prev_loss, best_loss) = load_checkpoint(
             unet, 
             scaler, 
             prev_ckpt_path, 
             device
        )
        print(f"Previous best loss: {best_loss:.6f}")
    else:
        unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet")

    unet.to(dtype=torch.float32, device=device)

    print_model_types(text_encoder, tokenizer, unet, autoencoder, noise_scheduler)

    g_optimizer = optim.AdamW(unet.parameters(), lr=learning_rate)

    unet.train()

    num_epochs += start_epoch

    for epoch in range(start_epoch, num_epochs):

        total_loss = 0.0

        for i, (texts, images) in enumerate(DATALOADER):
            if i % 2 == 0:
                throttle_gpu_hotspot_temp()

            images = images.to(device)

            with torch.no_grad():
                text_embeddings = get_openclip_text_features(list(texts), tokenizer, text_encoder, device).to(device)
            
            with torch.amp.autocast(device_type="cuda"):

                latents = autoencoder.encode(images).latent_dist.sample() * autoencoder.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_prediction = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                )[0]

                if hasattr(noise_scheduler, "predict_original_sample"):
                    x0_pred = noise_scheduler.predict_original_sample(noisy_latents, timesteps, noise_prediction)
                else:
                    alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                    x0_pred = (noisy_latents - (1 - alpha_t).sqrt() * noise_prediction) / alpha_t.sqrt()

                with torch.no_grad():
                    image_pred = autoencoder.decode(x0_pred / autoencoder.config.scaling_factor).sample
                    image_target = autoencoder.decode(latents / autoencoder.config.scaling_factor).sample

                    image_features = get_openclip_image_features(image_pred, text_encoder)

                # Visualization
                if visualize:
                    visualize_image(noise_prediction, images, dx_time=15)

                # MSE loss
                mse_loss = F.mse_loss(noise_prediction, noise)
                lambda_mse = 0.6
                whl_mse = lambda_mse * mse_loss

                # Perceptual loss
                perceptual_loss = vgg_loss_fn(image_pred, image_target)
                lambda_vgg = 0.4
                whl_vgg = lambda_vgg * perceptual_loss

                # CLIP loss
                clip_loss = contrastive_loss(text_embeddings, image_features)
                lambda_clip = 0.02
                whl_clip = lambda_clip * clip_loss

                cumulative_loss = whl_mse + whl_vgg + whl_clip

            scaler.scale(cumulative_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()
            g_optimizer.zero_grad() 

            total_loss += cumulative_loss.item()
            if (i + 1) % stat_freq == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}], Iteration [{i+1}/{dataloader_len}], Loss: {cumulative_loss.item():.6f}")
                print(f" - MSE loss: {whl_mse.item():.6f} VGG loss: {whl_vgg.item():.6f} CLIP loss: {whl_clip.item():.6f}")

        avg_loss = total_loss / dataloader_len

        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.6f}")

        torch.save(unet.state_dict(), last_model_path)

        if avg_loss < best_loss:
            best_loss = avg_loss

        # Training checkpoint
        save_checkpoint(unet, scaler, epoch, avg_loss, best_loss, checkpoint_path)
        print(f"Saved checkpoint")

    print(f"Best loss: {best_loss:.6f}")
    print(f"Saved final model {run_name}")

#----------------------------------------------------------------------------------------------------------#

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    prompt_path = os.path.normpath(
        os.path.join(current_dir, '..', '..', 'data/ambientcg/prompts/prompts.csv')
    )
    image_dir = os.path.normpath(
        os.path.join(current_dir, '..', '..', 'data/ambientcg/textures/512px')
    )

    dataset = ImagePromptDataset(image_dir, prompt_path, img_size=MAX_RESOLUTION, wnd_size=MAX_RESOLUTION)

    train_model(
        dataset         = dataset, 
        device          = device, 
        current_dir     = current_dir, 
        run_name        = "textures_v0.21", 
        checkpoint_name = "textures_v0.20_checkpoint",
        resolution      = MAX_RESOLUTION, 
        batch_size      = 1, 
        num_workers     = 1, 
        num_epochs      = 3, 
        resume_training = True, 
        visualize       = False,
        stat_freq       = 100,
        learning_rate   = 5e-5
    )

#----------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()