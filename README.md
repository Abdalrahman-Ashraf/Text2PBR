# Text2PBR

A PyTorch-based text-to-texture generation pipeline utilizing Latent Diffusion Models and GAN components to synthesize high-quality, high-resolution (2048×2048) PBR material maps from natural language prompts.

---

## 📖 Overview

This pipeline is implemented in two stages:
1. **Stage 1 (Texture Generation):** A fine-tuned Stable Diffusion model generates high-quality, diverse color textures conditioned on natural language prompt descriptions.
2. **Stage 2 (PBR Map Translation):** An image-to-image translation network utilizing a UNet generator and a discriminator processes the color texture to synthesize the corresponding PBR maps (Normal, Roughness, and Displacement).

---

## 🔥 Pipeline Output

The model generates a complete set of physically based rendering maps from a single prompt, including **Color/Diffuse, NormalGL, Roughness, and Displacement**.

<img width="1143" alt="PBR Map Generation Split" src="https://github.com/user-attachments/assets/dc450470-d2bc-48da-9415-a9e10a4a034b" />

---

## 👀 Render Showcases

> 💡 **Note:** For comparability and consistency, all generated PBR textures below are rendered on an identical cube mesh using uniform lighting and camera angles.

### Material Gallery

| "Red brick wall" | "Seamless sandstone wall" | "Brown tree bark" |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/cd72e097-58d3-4b9a-b95c-b54e98216fed" width="400" /> | <img src="https://github.com/user-attachments/assets/94e5094f-ed15-4d10-a2ca-a2fc122bd51c" width="400" /> | <img src="https://github.com/user-attachments/assets/d911b983-19ed-4a3b-a3cb-9af080479189" width="400" /> |
| *PBR texture maps applied.* | *Sandstone material evaluation.* | *Tree bark material evaluation.* |
