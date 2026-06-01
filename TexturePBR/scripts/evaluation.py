import os
import subprocess
import argparse
from torch_fidelity import calculate_metrics
from tqdm import tqdm
import shutil
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))

def resize_crop(img, size):
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    right = left + size
    bottom = top + size
    img = img.crop((left, top, right, bottom))
    return img

def resize_images(input_dir, output_dir, target_size=512):
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            img = Image.open(input_path).convert("RGB")
            img = resize_crop(img, target_size)
            img.save(output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def texture_to_pbr(input_path, output_dir, base_name, resolution="256"):
    executable_path = "C:/Users/abdra/miniconda3/envs/tex2uv/python.exe"
    script_path = os.path.normpath(os.path.join(current_dir, 'texture2pbr.py'))

    args = [
        '--input_path', input_path,
        '--output_dir', output_dir,
        '--base_name', base_name,
        '--resolution', resolution,
    ]
    subprocess.run([executable_path, script_path, *args])

def get_texture_name(filename):
    parts = filename.split('_')
    if len(parts) < 2:
        return filename
    
    res = parts[-1]
    del parts[-2:]
    
    return '_'.join(parts), res

def split_generated_tex(filename):
    parts = filename.split('_')
    if len(parts) < 2:
        return filename
    
    map_type = parts[-1]
    del parts[-1:]
    
    return '_'.join(parts), map_type

def get_poly_normalgl_name(filename):
    parts = filename.split('_')
    if len(parts) < 2:
        return filename
    
    res = parts[-1]
    del parts[-3:]
    
    return '_'.join(parts), res

def split_texture_res(filename):
    parts = filename.split('_')
    if len(parts) < 2:
        return filename
    
    res = parts[-1]
    del parts[-1:]
    
    return '_'.join(parts), res

def generate_maps(input_map="color", resolution="256"):
    generated_dir = os.path.normpath(os.path.join(current_dir, '..', 'data/textures/evaluation/generated'))
    input_dir = os.path.normpath(os.path.join(current_dir, '..', f'data/textures/evaluation/real/{input_map}'))

    for tex_name in tqdm(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, tex_name)

        tex_base_name, ext = os.path.splitext(tex_name)

        if os.path.exists(input_path):
            texture_to_pbr(input_path, generated_dir, tex_base_name, resolution=resolution)

def prepare_poly_data(ext=".jpg"):
    poly_dir = os.path.normpath(os.path.join(current_dir, '..', 'data/textures/polyhaven'))
    real_dir = os.path.normpath(os.path.join(current_dir, '..', 'data/textures/evaluation/real'))

    maps = {
        "color": "diff",
        "normalgl": "nor_gl",
        "roughness": "rough",
        "displacement": "disp",
        "ao": "ao"
    }

    for map_type in maps:
        type_dir = os.path.join(real_dir, map_type)
        os.makedirs(type_dir, exist_ok=True)

    tex_count = 0

    for tex_dir_name in os.listdir(poly_dir):
        tex_path = os.path.join(poly_dir, tex_dir_name, "textures")
        for map_type in maps:
            if not map_type == "normalgl":
                tex_name, res = split_texture_res(tex_dir_name)
            else:
                tex_name, res = get_poly_normalgl_name(tex_dir_name)

            target_name = tex_name + "_" + maps[map_type] + "_" + res + ext
            dest_name = str(tex_count) + ext

            target_path = os.path.join(tex_path, target_name)
            dest_path  = os.path.join(real_dir, map_type, dest_name)

            if os.path.exists(target_path):
                if not os.path.exists(dest_path):
                    shutil.copy(target_path, dest_path)

        tex_count += 1

    print(f"\nNumber of texturem maps read {tex_count}/{len(os.listdir(poly_dir))}")



def evaluate_generator_kid(target_map="normalgl"):
    real_dir = os.path.normpath(os.path.join(current_dir, '..', f'data/textures/evaluation/real_512/{target_map}'))
    generated_dir = os.path.normpath(os.path.join(current_dir, '..', f'data/textures/evaluation/generated/{target_map}'))

    if not os.path.isdir(real_dir):
        raise ValueError(f"Real dir does not exist '{real_dir}'")
    
    if not os.path.isdir(generated_dir):
        raise ValueError(f"Fake dir does not exist '{generated_dir}'")

    metrics = calculate_metrics(
        input1=real_dir,
        input2=generated_dir,
        fid=True,
        kid=True,
        cuda=True,
        kid_subset_size=9
    )

    print(f"\n{metrics}")

def main():
    real_dir = os.path.normpath(os.path.join(current_dir, '..', f'data/textures/evaluation/real/roughness'))
    out_dir = os.path.normpath(os.path.join(current_dir, '..', f'data/textures/evaluation/real_512/roughness'))
    resolution = 512   

    # resize_images(real_dir, out_dir, resolution)
    # prepare_poly_data()
    # generate_maps(input_map="color", resolution="512")
    evaluate_generator_kid(target_map="roughness")

if __name__ == "__main__":
    main()