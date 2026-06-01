import os
import subprocess
import argparse

def real_esrgan_upscale(input_path, output_dir):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    executable_path = os.path.normpath(os.path.join(current_dir, 'realesrgan_venv/Scripts/python.exe'))
    script_path = os.path.normpath(os.path.join(current_dir, 'Real-ESRGAN/inference_realesrgan.py'))

    args = [
        '-n', 'RealESRGAN_x4plus',
        '-i', input_path,
        '-o', output_dir,
        '--suffix', 'x4'
    ]
    subprocess.run([executable_path, script_path, *args])

    print(f"Saved upscaled image to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Text to PBR Texture Upscaler")
    parser.add_argument("input_path", type=str, help="Path to the input image or directory")
    parser.add_argument("output_dir", type=str, help="Directory to save the output image")

    args = parser.parse_args()

    real_esrgan_upscale(args.input_path, args.output_dir)

if __name__ == "__main__":
    main()