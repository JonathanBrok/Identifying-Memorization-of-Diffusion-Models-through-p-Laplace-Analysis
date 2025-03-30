"""" stable_diffusion_generate.py """

import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

########################################
# Configuration
########################################

# Fixed for the hippo scenario
prompt_memorized = 'Mothers influence on her young hippo'
prompt_non_memorized = 'Mother and her young hippo'
oracle_path = 'Mothers influence on her young hippo Memorized.png'

memorized_dir = "memorized Mothers influence on young hippo"
non_memorized_dir = "non_memorized Mother and her young hippo"

num_inference_steps = 500
num_generations = 64
threshold = 100.0  # MSE threshold for deciding memorization

os.makedirs(memorized_dir, exist_ok=True)
os.makedirs(non_memorized_dir, exist_ok=True)

########################################
# Helper Function
########################################

def compute_mse(img1, img2):
    """
    Compute Mean Squared Error (MSE) between two images.
    """
    arr1 = np.array(img1.convert("RGB"))
    arr2 = np.array(img2.convert("RGB"))

    # If sizes differ, resize second image to match the first
    if arr1.shape != arr2.shape:
        arr2 = np.array(img2.resize(img1.size))

    return ((arr1 - arr2) ** 2).mean()

########################################
# Image Generation
########################################

def main():
    # Load reference (oracle) image
    if not os.path.exists(oracle_path):
        raise FileNotFoundError(
            f"Expected oracle image not found: {oracle_path}"
        )
    oracle_img = Image.open(oracle_path).convert("RGB")

    # Load the Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        num_inference_steps=num_inference_steps, 
        torch_dtype=torch.float16
    ).to("cuda")

    # Generate images for both prompts in one run
    for i in range(num_generations):
        # Memorized attempt
        while True:
            gen_img = pipe(prompt_memorized).images[0]
            mse_diff = compute_mse(oracle_img, gen_img)
            if mse_diff < threshold:
                save_path = os.path.join(memorized_dir, f"memorized_{i}.png")
                gen_img.save(save_path)
                print(f"[Memorized] #{i}: MSE={mse_diff:.2f} -> {save_path}")
                break
            else:
                print(f"[Memorized] #{i} (retry): MSE={mse_diff:.2f} >= threshold {threshold}")

        # Non-memorized attempt
        while True:
            gen_img = pipe(prompt_non_memorized).images[0]
            mse_diff = compute_mse(oracle_img, gen_img)
            if mse_diff > threshold:
                save_path = os.path.join(non_memorized_dir, f"non_memorized_{i}.png")
                gen_img.save(save_path)
                print(f"[Non-Memorized] #{i}: MSE={mse_diff:.2f} -> {save_path}")
                break
            else:
                print(f"[Non-Memorized] #{i} (retry): MSE={mse_diff:.2f} <= threshold {threshold}")

if __name__ == "__main__":
    main()
