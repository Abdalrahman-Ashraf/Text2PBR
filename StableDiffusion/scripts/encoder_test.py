import torch
from torchvision import transforms
from PIL import Image
from train_diffuser import load_autoencoderkl

def test_encode_decode(image, vae, device):
    # 1) Load & preprocess → [-1,1]
    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image).unsqueeze(0)
    image = image.to(device) * 2 - 1

    # 2) Encode → get deterministic latent
    enc = vae.encode(image)                 # returns AutoencoderKLOutput(latent_dist, …)
    latents = enc.latent_dist.mean          # [1, 4, H/8, W/8]

    print(f"Latents shape (should be 4 channels): {latents.shape}")

    # 3) Decode directly (no scaling)
    dec = vae.decode(latents)               # returns DecoderOutput(sample, …)
    recon = dec.sample.squeeze(0).cpu()     # [3, H, W]
   
    # 4) Clamp back to [-1,1]
    return recon.clamp(-1, 1)

# …then in your script:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = load_autoencoderkl(device)

img = Image.open(r"D:\Program Files\Python\Image Generator Model\data\artstation\Character Design\512px\images\artstation_14866_33765_Neo Japan 2202.jpg").convert("RGB")
decoded = test_encode_decode(img, vae, device)

# Visualize
from torchvision.transforms import ToPILImage
ToPILImage()(decoded).show()
