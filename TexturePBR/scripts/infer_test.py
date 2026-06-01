import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F_t
import os
from pbr_unet import PBRMapGeneratorUNet
import argparse
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))

displacement_model_name = "displacement_v0.91_base_last"

displacement_model_path = os.path.normpath(os.path.join(current_dir, '..', f'pretrained/{displacement_model_name}.pth'))

displacement_model = PBRMapGeneratorUNet(in_channels=3, out_channels=1, base_channels=64)
displacement_model.load_state_dict(torch.load(displacement_model_path, map_location='cpu'))
displacement_model.eval()

def preprocess_image(image_path, image_size=512):
    if not os.path.exists(image_path):
        raise ValueError(f"Input path does not exist '{image_path}'")
    
    transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if min(w, h) != image_size:
        image = F_t.resize(image, image_size, interpolation=Image.BICUBIC)

    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def save_output_maps(output_map, output_dir, texture_name, ext=".png"):
    output_map = output_map.squeeze(0).detach().cpu()
    to_pil = transforms.ToPILImage()
    img = to_pil(output_map)
    img.save(os.path.join(output_dir, f"{texture_name}{ext}"))

    print(f"Saved texture map {texture_name}")

def generatePBRMaps(input_path, output_dir, base_name, resolution=2048):
    texture_name = os.path.basename(input_path)
    _, ext = os.path.splitext(texture_name)

    input_tensor = preprocess_image(input_path, image_size=resolution)
    with torch.no_grad():
        displacement_output = displacement_model(input_tensor)

    displacement_map_name = base_name + "_" + "Displacement"

    save_output_maps(displacement_output, output_dir, displacement_map_name, ext)

def main():
    input_path = r"D:\Program Files\Materials\1K - ambientcg\Bricks049_1K-JPG\Bricks049_1K-JPG_Color.jpg"
    output_dir = os.path.normpath(os.path.join(current_dir, "..", "outputs"))
    base_name = "test12"
    generatePBRMaps(input_path, output_dir, base_name, 1024)

if __name__ == "__main__":
    main()