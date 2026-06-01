import os
import subprocess
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))

def real_esrgan_upscale(input_dir, output_dir):
    executable_path = os.path.normpath(os.path.join(current_dir, 'Upscaling Model/realesrgan_venv/Scripts/python.exe'))
    script_path = os.path.normpath(os.path.join(current_dir, 'Upscaling Model/Real-ESRGAN/inference_realesrgan.py'))

    args = [
        '-n', 'RealESRGAN_x4plus',
        '-i', input_dir,
        '-o', output_dir,
        '--suffix', ''
    ]
    subprocess.run([executable_path, script_path, *args])

    print(f"Saved upscaled image to {output_dir}")

def texture_to_pbr(input_path, output_dir, base_name):
    executable_path = "C:/Users/abdra/miniconda3/envs/tex2uv/python.exe"
    script_path = os.path.normpath(os.path.join(current_dir, 'TexturePBR/scripts/texture2pbr.py'))

    args = [
        '--input_path', input_path,
        '--output_dir', output_dir,
        '--base_name', base_name,
        '--resolution', "2048",
    ]
    subprocess.run([executable_path, script_path, *args])

    print(f"Saved PBR maps")

def text_to_texture(prompt, output_dir, texture_name, ext=".png"):
    executable_path = "C:/Users/abdra/miniconda3/envs/tex2uv/python.exe"
    script_path = os.path.normpath(os.path.join(current_dir, 'StableDiffusion/scripts/text2texture.py'))
    texture_name = f"{texture_name}_Color{ext}"
    output_path = os.path.normpath(os.path.join(output_dir, texture_name))

    args = [
        '--prompt', prompt,
        '--output', output_path
    ]
    subprocess.run([executable_path, script_path, *args])

    print(f"Saved PBR maps")

    return output_path, texture_name

def main():
    parser = argparse.ArgumentParser(description="Generates a PBR texture given a prompt")
    parser.add_argument("prompt", type=str, help="Input a prompt")
    parser.add_argument("texture_name", type=str)
    parser.add_argument("output_dir", type=str, help="Directory to save the output map")
    parser.add_argument("ext", type=str, default=".png")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        raise ValueError(f"Incorrect output directory '{args.output_dir}'")

    pbr_texture_dir = os.path.normpath(os.path.join(args.output_dir, args.texture_name))
    pbr_upscaled_dir = os.path.normpath(os.path.join(args.output_dir, f"{args.texture_name}_x4"))

    os.makedirs(pbr_texture_dir, exist_ok=True)
    os.makedirs(pbr_upscaled_dir, exist_ok=True)

    generated_map_path, texture_name = text_to_texture(args.prompt, pbr_texture_dir, args.texture_name, args.ext)
    real_esrgan_upscale(pbr_texture_dir, pbr_upscaled_dir)
    upscaled_path = os.path.normpath(os.path.join(pbr_upscaled_dir, texture_name))
    texture_to_pbr(upscaled_path, pbr_texture_dir, args.texture_name)

if __name__ == "__main__":
    main()