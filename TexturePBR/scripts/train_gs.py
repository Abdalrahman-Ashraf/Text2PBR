import os
import random
import lpips
import torch
import cv2
import csv
import kornia
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pbr_discriminator import PBRPatchDiscriminator
from pbr_unet import PBRMapGeneratorUNet
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torchvision.transforms.functional as F_t
import torch.nn.functional as F_n
from system_settings import *
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------#

IMAGE_RESOLUTION = 512
CROP_SIZE = 512

#----------------------------------------------------------------------------------------------------------#

def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer,  g_scaler, d_scaler, epoch, loss, best_loss, path):
    torch.save({
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'g_scaler': g_scaler.state_dict(),
        'd_scaler': d_scaler.state_dict(),
        'loss': loss,
        'best_loss': best_loss
    }, path)

#----------------------------------------------------------------------------------------------------------#

def load_checkpoint(generator, discriminator, g_optimizer, d_optimizer,  g_scaler, d_scaler, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")
    checkpoint = torch.load(path, map_location=device)

    generator.load_state_dict(checkpoint.get('generator', {}))
    discriminator.load_state_dict(checkpoint.get('discriminator', {}))
    g_optimizer.load_state_dict(checkpoint.get('g_optimizer', {}))
    d_optimizer.load_state_dict(checkpoint.get('d_optimizer', {}))
    g_scaler.load_state_dict(checkpoint.get('g_scaler', {}))
    d_scaler.load_state_dict(checkpoint.get('d_scaler', {}))

    start_epoch = checkpoint.get('epoch', 0) + 1
    loss = checkpoint.get('loss', None)
    best_loss = checkpoint.get('best_loss', None)

    print(f"Checkpoint loaded from epoch {start_epoch + 1}")

    return {
        "epoch" : start_epoch,
        "generator" : generator,
        "discriminator" : discriminator,
        "g_optimizer" : g_optimizer,
        "d_optimizer" : d_optimizer,
        "g_scaler" : g_scaler,
        "d_scaler" : d_scaler,
        "loss" : loss,
        "best_loss" : best_loss
    }

#----------------------------------------------------------------------------------------------------------#

def normalize_patch(patch):
    patch = patch.clone()
    patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
    return patch

#----------------------------------------------------------------------------------------------------------#

def rgb_to_luminance(img, grayscale):
    if img.shape[1] == 1 or not grayscale:
        return img
    
    r, g, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

#----------------------------------------------------------------------------------------------------------#

def fft_luminance_loss(pred, target, grayscale=True):
    pred_lum = rgb_to_luminance(pred.float(), grayscale)
    target_lum = rgb_to_luminance(target.float(), grayscale)
    
    pred_fft = torch.fft.fft2(pred_lum, norm='ortho')
    target_fft = torch.fft.fft2(target_lum, norm='ortho')

    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    return F.l1_loss(pred_mag, target_mag)

#----------------------------------------------------------------------------------------------------------#

def kornia_lab_loss(pred, target):
    pred_lab = kornia.color.rgb_to_lab(pred)
    target_lab = kornia.color.rgb_to_lab(target)

    return F.mse_loss(pred_lab, target_lab)

#----------------------------------------------------------------------------------------------------------#

def rgb_to_yuv_tensor(img):
    img_np = img.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img_np = (img_np * 255).astype(np.uint8)
    yuv_list = [cv2.cvtColor(frame, cv2.COLOR_RGB2YUV) for frame in img_np]
    yuv_np = np.stack(yuv_list).astype(np.float32) / 255.0
    yuv_tensor = torch.tensor(yuv_np).permute(0, 3, 1, 2).to(img.device)
    return yuv_tensor

#----------------------------------------------------------------------------------------------------------#

def yuv_loss(pred, target):
    pred_yuv = rgb_to_yuv_tensor(pred)
    target_yuv = rgb_to_yuv_tensor(target)

    return F.mse_loss(pred_yuv, target_yuv)

#----------------------------------------------------------------------------------------------------------#

def gradient_loss(pred, target):
    def get_gradients(img):
        dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        return dx, dy

    pred_dx, pred_dy = get_gradients(pred)
    target_dx, target_dy = get_gradients(target)

    grad_loss = 0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))

    return grad_loss

#----------------------------------------------------------------------------------------------------------#

class LoadPBRData(DataLoader):
    def __init__(self, input_dir : str, target_dir : str, crop_size : int, image_size : int):
        self.image_size = image_size
        self.texture_maps = []
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor()
        ])
        self.color_jitter = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.04,
                contrast=0.04,
                saturation=0.02
                )
        ])

        if not os.path.exists(input_dir):
            raise ValueError(f"Input map directory does not exist '{input_dir}'")
        if not os.path.exists(target_dir):
            raise ValueError(f"Target map directory does not exist '{target_dir}'")
        
        for input_map in os.listdir(input_dir):
            input_map_path = os.path.join(input_dir, input_map)
            target_map_path = os.path.join(target_dir, input_map)

            if not os.path.exists(target_map_path):
                continue

            self.texture_maps.append((input_map_path, target_map_path))

        print(f"{len(self.texture_maps)} texture maps successfully loaded")

    def augment_image(self, image: Image.Image):
        w, h = image.size
        if min(w, h) != self.image_size:
            image = F_t.resize(image, self.image_size, interpolation=Image.BICUBIC)

        return self.transform(image)

    def __len__(self):
        return len(self.texture_maps)
        
    def __getitem__(self, idx):
        input_map_path, target_map_path = self.texture_maps[idx]

        input_map = Image.open(input_map_path).convert("RGB")
        target_map = Image.open(target_map_path).convert("L")

        if random.random() > 0.5:
            input_map = F_t.hflip(input_map)
            target_map = F_t.hflip(target_map)

        if random.random() > 0.5:
            input_map = F_t.vflip(input_map)
            target_map = F_t.vflip(target_map)

        input_map = self.augment_image(input_map)
        target_map = self.augment_image(target_map)

        input_map = self.color_jitter(input_map)

        return input_map, target_map

#----------------------------------------------------------------------------------------------------------#

def train(
        device            : str,
        current_dir       : str,
        dataset           : DataLoader,
        pbr_generator     : PBRMapGeneratorUNet,
        g_optimizer       : optim,
        pbr_discriminator : PBRPatchDiscriminator,
        d_optimizer       : optim,
        run_name          : str,
        ckpt_name         : str,
        batch_size        = 8, 
        num_workers       = 2, 
        num_epochs        = 100, 
        resume_training   = False,
        visualize         = False,
        stat_freq         = 10,
        override_optim    = True,
        enable_logs       = True
    ):

    pretrained_dir = os.path.normpath(os.path.join(current_dir, '..', 'pretrained'))
    loss_dir = os.path.normpath(os.path.join(current_dir, '..', 'training_logs'))

    loss_graph_path = os.path.normpath(os.path.join(loss_dir, f'{run_name}_loss_graph.png'))
    training_logs_path = os.path.normpath(os.path.join(loss_dir, f'{run_name}_logs.csv'))
    prev_ckpt_path = os.path.normpath(os.path.join(pretrained_dir, f'{ckpt_name}.pth'))
    checkpoint_path = os.path.normpath(os.path.join(pretrained_dir, f'{run_name}_ckpt.pth'))
    last_model_path = os.path.normpath(os.path.join(pretrained_dir, f'{run_name}_last.pth'))

    if not os.path.exists(pretrained_dir):
        raise ValueError("Model dir does not exist")
    
    if not os.path.exists(loss_dir):
        raise ValueError("Logs dir does not exist")
    
    dataloader = DataLoader(
        dataset, 
        batch_size, 
        shuffle            = True, 
        num_workers        = num_workers, 
        pin_memory         = True, 
        persistent_workers = True
    )

    dataloader_len = len(dataloader)

    start_epoch = 0
    best_loss = float('inf')

    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()

    # lpips_loss_fn = lpips.LPIPS(net='vgg')
    # lpips_loss_fn = lpips_loss_fn.to(device)

    if resume_training:
        if not os.path.exists(prev_ckpt_path):
            raise ValueError(f"Checkpoint path does not exist '{prev_ckpt_path}'")
        
        checkpoint = load_checkpoint(
             generator     = pbr_generator,
             discriminator = pbr_discriminator,
             g_optimizer   = g_optimizer, 
             d_optimizer   = d_optimizer,
             g_scaler      = g_scaler, 
             d_scaler      = d_scaler,
             path          = prev_ckpt_path, 
             device        = device
        )
        pbr_generator = checkpoint['generator']
        pbr_discriminator = checkpoint['discriminator']
        g_scaler = checkpoint['g_scaler']
        d_scaler = checkpoint['d_scaler']
        best_loss = checkpoint['best_loss']

        if not override_optim:
            g_optimizer = checkpoint['g_optimizer']
            d_optimizer = checkpoint['d_optimizer']

        start_epoch = checkpoint['epoch']
        num_epochs += start_epoch
        print(f"Previous best loss: {best_loss:.6f}")

    generator_losses = {
        "total" : [],
        "mse" : [],
        "lab" : [],
        "fft" : []
    }

    epochs_executed = []

    bce_loss = torch.nn.BCEWithLogitsLoss()

    pbr_generator.train()
    pbr_discriminator.train()

    #----------------------- Training loop -----------------------#

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0

        for i, (input_map, target_map) in enumerate(dataloader):
            if i % 2 == 0:
                throttle_gpu_hotspot_temp()

            input_map = input_map.to(device)
            target_map = target_map.to(device)

            #---------- Discriminator Training ----------#

            with torch.amp.autocast(device_type="cuda"):
                predicted_map = pbr_generator(input_map)

                pred_detached = predicted_map.detach()

                d_real = pbr_discriminator(target_map)
                d_fake = pbr_discriminator(pred_detached)

                real_labels = torch.ones_like(d_real)
                fake_labels = torch.zeros_like(d_fake)

                d_loss_real = bce_loss(d_real, real_labels)
                d_loss_fake = bce_loss(d_fake, fake_labels)
                d_loss = (d_loss_real + d_loss_fake) * 0.5

            # Discriminator
            d_scaler.scale(d_loss).backward()
            d_scaler.step(d_optimizer)
            d_scaler.update()
            d_optimizer.zero_grad()

            #---------- Discriminator Training ----------#

            #------------ Generator Training ------------#

            with torch.amp.autocast(device_type="cuda"):

                # Discriminator loss
                gan_pred = pbr_discriminator(predicted_map)
                gan_loss = bce_loss(gan_pred, torch.ones_like(gan_pred))
                lambda_gan = 0.014
                wl_gan = lambda_gan * gan_loss

                # L1 loss
                mse_loss = F_n.l1_loss(predicted_map, target_map)
                lambda_mse = 0.03
                wl_mse = lambda_mse * mse_loss

                # Fourier transform loss
                fft_loss = fft_luminance_loss(predicted_map, target_map)
                lambda_fft = 0.4
                wl_fft = lambda_fft * fft_loss

                # Gradient loss
                # grad_loss = gradient_loss(predicted_map, target_map)
                # lambda_grad = 0.25
                # wl_grad = lambda_grad * grad_loss

                # LPIPS perceptual loss
                # vgg_loss = lpips_loss_fn(predicted_map * 2 - 1, target_map * 2 - 1).mean()
                # lambda_vgg = 0.03
                # wl_vgg = lambda_vgg * vgg_loss

                cumulative_loss = (wl_mse + wl_fft + wl_gan)

            # Generator
            g_scaler.scale(cumulative_loss).backward()
            g_scaler.step(g_optimizer)
            g_scaler.update()
            g_optimizer.zero_grad()

            #------------ Generator Training ------------#

            #---------------- Statistics ----------------#

            total_loss += cumulative_loss.item()

            epochs_executed.append(epoch)

            generator_losses['total'].append(cumulative_loss.item())
            generator_losses['mse'].append(wl_mse.item())
            generator_losses['fft'].append(wl_fft.item())

            if (i+1) % stat_freq == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}], Iteration [{i+1}/{dataloader_len}]")
                print(f" - Generator loss: {cumulative_loss.item():.6f} " + 
                      f"GAN loss: {wl_gan.item():.6f} " +
                      f"L1 loss: {wl_mse.item():.6f} " +
                      f"FFT loss: {wl_fft.item():.6f} ")
                print(f" - Discriminator loss: {d_loss.item():.6f}")
                
            #---------------- Statistics ----------------#

        avg_loss = total_loss / dataloader_len

        if avg_loss < best_loss:
            best_loss = avg_loss

        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.6f}")

        torch.save(pbr_generator.state_dict(), last_model_path)

        save_checkpoint(
            generator     = pbr_generator,
            discriminator = pbr_discriminator,
            g_optimizer   = g_optimizer, 
            d_optimizer   = d_optimizer,
            g_scaler      = g_scaler, 
            d_scaler      = d_scaler,
            epoch         = epoch, 
            loss          = avg_loss,
            best_loss     = best_loss,
            path          = checkpoint_path
        )
        print(f"Saved checkpoint")

    #----------------------- Training loop -----------------------#

    print(f"Best loss: {best_loss:.6f}")
    print(f"Saved final model {run_name}")

    iterations = list(range(len(generator_losses['total'])))

    plt.figure(figsize=(10, 6))

    plt.plot(iterations, generator_losses['total'], label='Total Loss', color='blue')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()
    plt.grid(True)

    if enable_logs:
        plt.savefig(loss_graph_path)

        with open(training_logs_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Iteration', 'Loss', 'L1', 'FFT'])

            num_iterations = len(generator_losses['total'])
            for i in range(num_iterations):
                epoch = epochs_executed[i]
                writer.writerow([
                    epoch, i,
                    f"{generator_losses['total'][i]:.6f}",
                    f"{generator_losses['mse'][i]:.6f}",
                    f"{generator_losses['fft'][i]:.6f}"
                ])
        print("Saved logs")

#----------------------------------------------------------------------------------------------------------#

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pbr_input_dir = r"D:\Program Files\Python\Texture PBR Model\data\textures\ambientcg\color"
    pbr_roughness_dir = r"D:\Program Files\Python\Texture PBR Model\data\textures\ambientcg\roughness"
    pbr_displacement_dir = r"D:\Program Files\Python\Texture PBR Model\data\textures\ambientcg\displacement"

    pbr_textures = LoadPBRData(pbr_input_dir, pbr_displacement_dir, crop_size=CROP_SIZE, image_size=IMAGE_RESOLUTION)
    pbr_generator = PBRMapGeneratorUNet(in_channels=3, out_channels=1, base_channels=64).to(device)
    pbr_discriminator = PBRPatchDiscriminator(in_channels=1, base_channels=64).to(device)

    g_optimizer = optim.AdamW(pbr_generator.parameters(), 2e-4)
    d_optimizer = optim.AdamW(pbr_discriminator.parameters(), 1e-4)

    train(
        device            = device, 
        current_dir       = current_dir,
        dataset           = pbr_textures,
        pbr_generator     = pbr_generator,
        g_optimizer       = g_optimizer,
        pbr_discriminator = pbr_discriminator,
        d_optimizer       = d_optimizer,
        run_name          = "displacement_v0.91_base",
        ckpt_name         = "displacement_v0.83_base_ckpt",
        batch_size        = 8, 
        num_workers       = 4,
        num_epochs        = 2,
        resume_training   = True,
        visualize         = False,
        override_optim    = True,
        enable_logs       = True
    )

#----------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()