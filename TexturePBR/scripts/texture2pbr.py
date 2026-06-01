import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F_t
import os
from pbr_unet import PBRMapGeneratorUNet
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))

normaldx_model_name = "normalgl_v0.90_base_last"
roughness_model_name = "roughness_v0.90_base_last"
displacement_model_name = "displacement_v0.83_base_last"

roughness_model_path = os.path.normpath(os.path.join(current_dir, '..', f'pretrained/{roughness_model_name}.pth'))
normaldx_model_path = os.path.normpath(os.path.join(current_dir, '..', f'pretrained/{normaldx_model_name}.pth'))
displacement_model_path = os.path.normpath(os.path.join(current_dir, '..', f'pretrained/{displacement_model_name}.pth'))

roughness_model = PBRMapGeneratorUNet(in_channels=3, out_channels=3, base_channels=64)
roughness_model.load_state_dict(torch.load(roughness_model_path, map_location='cpu'))
roughness_model.eval()

normaldx_model = PBRMapGeneratorUNet(in_channels=3, out_channels=3, base_channels=64)
normaldx_model.load_state_dict(torch.load(normaldx_model_path, map_location='cpu'))
normaldx_model.eval()

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
        normaldx_map_output = normaldx_model(input_tensor)
        roughness_map_output = roughness_model(input_tensor)
        displacement_map_output = displacement_model(input_tensor)

    normaldx_map_name = base_name + "_" + "NormalGl"
    roughness_map_name = base_name + "_" + "Roughness"
    displacement_map_name = base_name + "_" + "Displacement"

    save_output_maps(normaldx_map_output, output_dir, normaldx_map_name, ext)
    save_output_maps(roughness_map_output, output_dir, roughness_map_name, ext)
    save_output_maps(displacement_map_output, output_dir, displacement_map_name, ext)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_name", required=True)
    parser.add_argument("--resolution", type=int, required=True)
    args = parser.parse_args()

    generatePBRMaps(args.input_path, args.output_dir, args.base_name, args.resolution)

if __name__ == "__main__":
    main()