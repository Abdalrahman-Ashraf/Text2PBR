import cv2
import os
import numpy as np
from PIL import Image

def apply_clahe_pil(pil_img, clipLimit=2.0, tileGridSize=(8,8)):
    img_np = np.array(pil_img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_img)

def adjust_gamma_pil(pil_img, gamma=1.2):
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    img_np = np.clip(img_np ** gamma, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)

def boost_lab_contrast_pil(pil_img, contrast=1.0, chroma_boost=1.0):
    img_np = np.array(pil_img)  # convert PIL to numpy array

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Boost lightness contrast around midpoint 128
    l = 128 + (l.astype(np.float32) - 128) * contrast
    l = np.clip(l, 0, 255).astype(np.uint8)

    # Boost chroma channels around midpoint 128
    a = 128 + (a.astype(np.float32) - 128) * chroma_boost
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = 128 + (b.astype(np.float32) - 128) * chroma_boost
    b = np.clip(b, 0, 255).astype(np.uint8)

    boosted_lab = cv2.merge((l, a, b))
    boosted_rgb = cv2.cvtColor(boosted_lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(boosted_rgb)  # convert back to PIL image


def image_to_pil(image_path):
    return Image.open(image_path).convert("RGB")

def save_image(pil_img, output_path, texture_name, ext):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pil_img.save(os.path.join(output_path, f"{texture_name}.{ext}"))
    print(f"Saved texture map {texture_name}")

texture_name = "Bricks047_1K-JPG_NormalDX1"
ext = "jpg"
current_dir = os.path.dirname(os.path.abspath(__file__))
pbr_input_path = os.path.normpath(os.path.join(current_dir, '..', f'output_maps/{texture_name}.{ext}'))
pbr_output_dir = os.path.normpath(os.path.join(current_dir, '..', 'post_processed'))

pil_img = image_to_pil(pbr_input_path)

# Example usage:
pil_img = apply_clahe_pil(pil_img, clipLimit=1.0)
pil_img = adjust_gamma_pil(pil_img, 1.04)
#pil_img = boost_lab_contrast_pil(pil_img, chroma_boost=1.1)

save_image(pil_img, pbr_output_dir, texture_name, ext)
