import cv2
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

def apply_clahe_np(img_np, clipLimit=2.0, tileGridSize=(8,8)):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def adjust_gamma_np(img_np, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img_np, table)

def boost_lab_contrast_np(img_np, contrast=1.0, chroma_boost=1.0):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Boost lightness contrast around midpoint 128
    l = 128 + (l.astype(np.float32) - 128) * contrast
    l = np.clip(l, 0, 255).astype(np.uint8)

    # Optionally boost chroma channels (if chroma_boost != 1)
    a = 128 + (a.astype(np.float32) - 128) * chroma_boost
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = 128 + (b.astype(np.float32) - 128) * chroma_boost
    b = np.clip(b, 0, 255).astype(np.uint8)

    boosted_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(boosted_lab, cv2.COLOR_LAB2RGB)


def adjust_brightness_np(img_np, brightness=0):
    # brightness: -100..100, just add offset
    img = img_np.astype(np.int16) + brightness
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def adjust_saturation_np(img_np, saturation=1.0):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] *= saturation
    hsv[...,1] = np.clip(hsv[...,1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def nothing(x):
    pass

texture_name = "Bricks047_1K-JPG_NormalDX1"
ext = "jpg"
current_dir = os.path.dirname(os.path.abspath(__file__))
pbr_input_path = os.path.normpath(os.path.join(current_dir, '..', f'output_maps/{texture_name}.{ext}'))
pbr_output_dir = os.path.normpath(os.path.join(current_dir, '..', 'post_processed'))

pil_img = Image.open(pbr_input_path).convert("RGB")
img_np = np.array(pil_img)

cv2.namedWindow('Image Adjustment', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image Adjustment', 900, 900)

# Create sliders
cv2.createTrackbar('CLAHE ON/OFF', 'Image Adjustment', 0, 1, nothing)              # 0 or 1
cv2.createTrackbar('CLAHE clipLimit x10', 'Image Adjustment', 20, 100, nothing)   # 0.1 to 10.0 (scaled by 10)
cv2.createTrackbar('CLAHE tileGridSize', 'Image Adjustment', 8, 16, nothing)      # 1 to 16
cv2.createTrackbar('Gamma x10', 'Image Adjustment', 10, 30, nothing)               # 0.1 to 3.0
cv2.createTrackbar('LAB Contrast x10', 'Image Adjustment', 10, 30, nothing)       # 1.0 to 3.0
cv2.createTrackbar('LAB Chroma x10', 'Image Adjustment', 10, 30, nothing)         # 1.0 to 3.0
cv2.createTrackbar('Brightness', 'Image Adjustment', 100, 200, nothing)           # -100 to 100
cv2.createTrackbar('Saturation x10', 'Image Adjustment', 10, 30, nothing)         # 0.1 to 3.0

while True:
    clahe_on = cv2.getTrackbarPos('CLAHE ON/OFF', 'Image Adjustment')
    clip_limit = max(cv2.getTrackbarPos('CLAHE clipLimit x10', 'Image Adjustment') / 10.0, 0.1)
    tile_grid = max(cv2.getTrackbarPos('CLAHE tileGridSize', 'Image Adjustment'), 1)
    gamma = cv2.getTrackbarPos('Gamma x10', 'Image Adjustment') / 10.0
    lab_contrast = cv2.getTrackbarPos('LAB Contrast x10', 'Image Adjustment') / 10.0
    lab_chroma = cv2.getTrackbarPos('LAB Chroma x10', 'Image Adjustment') / 10.0
    brightness = cv2.getTrackbarPos('Brightness', 'Image Adjustment') - 100
    saturation = cv2.getTrackbarPos('Saturation x10', 'Image Adjustment') / 10.0

    img = img_np.copy()
    if clahe_on == 1:
        img = apply_clahe_np(img)

    # Apply the adjustments in order
    img = img_np.copy()
    img = apply_clahe_np(img)
    img = adjust_gamma_np(img, gamma)
    img = boost_lab_contrast_np(img, lab_contrast, lab_chroma)
    img = adjust_brightness_np(img, brightness)
    img = adjust_saturation_np(img, saturation)

    cv2.imshow('Image Adjustment', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(30)
    if key == 27:  # ESC key to exit
        break

cv2.destroyAllWindows()
