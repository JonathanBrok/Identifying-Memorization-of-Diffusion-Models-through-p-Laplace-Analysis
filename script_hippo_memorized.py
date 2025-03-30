"""script_hippo_memorized.py"""

import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import pipeline as pipeline_caption  # only if needed
import torchvision.transforms as T

# import our p-Laplace functionality (boundary formualtion)
from p_laplace_core import compute_p_laplace_boundary_torch, sample_sphere_normals_nd_torch


# -------------------------------------------------------------------------
# 0) Helper to encode an image into latents
# -------------------------------------------------------------------------
def encode_image_to_latents(img: Image.Image, vae, device="cuda"):
    """
    Resizes the image to 512×512, normalizes to [-1..1],
    encodes into latents, returns shape (4*64*64,).
    """
    transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    img_t = transform(img).unsqueeze(0).half().to(device)
    with torch.no_grad():
        latents = vae.encode(img_t).latent_dist.sample()  # (1,4,64,64)
        latents = latents * vae.config.scaling_factor
    return latents[0].flatten().cpu().numpy()  # shape (4*64*64,)


# -------------------------------------------------------------------------
# 1) The factory that returns a "batch-friendly" predict_noise function
# -------------------------------------------------------------------------
def predict_noise_factory(timesteps, text_emb, unet, vae, scheduler):
    """
    Returns a function predict_noise(noisy_latents) that:
      - accepts a shape (N,4,64,64) tensor
      - runs UNet forward pass
      - returns shape (N,4,64,64)
    """

    
    
    def predict_noise(noisy_latents):
        """
        noisy_latents: (N,4,64,64)
        returns       : (N,4,64,64)
        """
        N = noisy_latents.shape[0]

        # 1) replicate text_emb => (N,77,768)
        text_emb_batched = text_emb.repeat(N, 1, 1).to(unet.dtype)

        # 2) replicate timesteps => shape (N,)
        #    if timesteps is shape (1,), e.g. [42], do:
        timesteps_batched = timesteps.expand(N)

        # 3) ensure latents are float16
        noisy_latents = noisy_latents.to(unet.dtype)

        with torch.no_grad():
            noise_pred = unet(
                noisy_latents, 
                timesteps_batched, 
                encoder_hidden_states=text_emb_batched
            )[0]
        noise_pred = noise_pred / vae.config.scaling_factor
        return noise_pred


    return predict_noise


# -------------------------------------------------------------------------
# 2) The function that DOES the random sampling and returns avg cos similarity
# -------------------------------------------------------------------------
def compute_avg_cosine_similarity_for_image(
    img_path,
    unet,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    prompt="Mothers influence on her young hippo",
    time_frac=0.01,
    n_draws=8,
    device="cuda"
):
    """
    1) Encodes the image -> latents (shape (1,4,64,64))
    2) Builds text_emb from the 'prompt'
    3) For each of n_draws:
       - Sample new noise
       - Call predict_noise(...) on that noise
       - Compute cosine similarity
    4) Return the average similarity

    We also do an optional p-laplace step (1-laplace).
    """
    # A) Encode image => latents
    img = Image.open(img_path).convert("RGB")
    latents_np = encode_image_to_latents(img, vae, device=device)
    latents_torch = (
        torch.from_numpy(latents_np)
        .float()
        .to(device)
        .view(1, 4, 64, 64)  # shape (1,4,64,64)
    )

    # B) Prepare timesteps from time_frac
    num_train_steps = scheduler.config.num_train_timesteps
    t_int = int(time_frac * num_train_steps)
    t_int = max(0, min(t_int, num_train_steps - 1))
    timesteps = torch.full((1,), t_int, device=device, dtype=torch.long)

    # C) Tokenize prompt => text_emb
    text_inputs = tokenizer(
        [prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        text_emb = text_encoder(input_ids).last_hidden_state  # shape (1,77,768)
    text_emb = text_emb.to(unet.dtype)
    
    
    

    # E) For optional 1-Laplace, we manually compute alpha
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    alpha_prod = scheduler.alphas_cumprod[timesteps].to(unet.dtype)      # shape (1,)
    sqrt_alpha_prod = torch.sqrt(alpha_prod)                             # shape (1,)
    sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - alpha_prod)             # shape (1,)

    # We'll measure random draws => cos similarity
    cos_vals = []  # 1-Laplace boundary formulation multiplies normalized vectors on the boundary - thus it is a cosine measure
    for cur_latents in latents_torch:
        center = sqrt_alpha_prod * cur_latents  # shape (4,64,64)
        radius_factor = sqrt_one_minus_alpha_prod  # in addition to the sqrt(d) radius (justified by concetration of measure considerations), this factor is needed to account for the fixed point in time analysis (namely the use of a specific alpha)
        
        # score function functionality
        predict_noise = predict_noise_factory(timesteps, text_emb, unet, vae, scheduler)
       
        # 1) p-Laplace
        plap_val = compute_p_laplace_boundary_torch(
            center=center,
            radius_factor=radius_factor.item(),          # pass it as a scalar float
            p=1,
            get_logp_gradients=lambda pts: predict_noise(pts),  # shape -> (N,4,64,64)
            n_samples=128,
            epsilon=1e-6
        )
        cos_vals.append(plap_val.item())  # gather the 1-laplace result as well
        
        
        
        # 2)
        # n_samples = 2
        # epsilon = 1e-6
        
        # # normals = sample_sphere_normals_nd_torch(
        # #     tensor_shape=center.shape,
        # #     n_samples=n_samples,
        # #     device=device,
        # #     epsilon=epsilon,
        # # ).to(center.dtype)

        # normals = torch.randn((n_samples, *center.shape), device=device).to(center.dtype)
        
        # points_on_sphere = center.unsqueeze(0) + radius * normals
        
        # # predict the noise
        # noise_pred = predict_noise(points_on_sphere)  # shape => (1,4,64,64)

        # this_cos = _cosine_similarity_2d(noise_pred[0], normals[0])
        # cos_vals.append(this_cos)

      

        # # #
        ####BACKUP
        # # 2) Standard random draws
        # #    We'll do one random draw => noise => measure cos
        # noise = torch.randn_like(cur_latents, device=device)  # shape (4,64,64)

        # # unify dtype
        # noise = noise.to(unet.dtype)
        # c_lat = cur_latents.to(unet.dtype)

        # # Construct noisy latents manually => we do:
        # #   lat_noisy = sqrt_alpha_prod * c_lat + sqrt_one_minus_alpha_prod * noise
        # lat_noisy = sqrt_alpha_prod * c_lat + sqrt_one_minus_alpha_prod * noise

        # # predict the noise
        # lat_noisy_4d = lat_noisy.unsqueeze(0)   # shape => (1,4,64,64) for batch dimension
        # noise_pred_4d = predict_noise(lat_noisy_4d)  # shape => (1,4,64,64)
        # noise_pred = noise_pred_4d[0]               # shape => (4,64,64)

        # # measure cos similarity
        # this_cos = _cosine_similarity_2d(noise_pred, noise)
        # cos_vals.append(this_cos)
    return float(np.mean(cos_vals))


def _cosine_similarity_2d(pred, noise):
    # Force both to half precision:
    pred_flat = pred.view(-1).to(torch.float16)
    noise_flat = noise.view(-1).to(torch.float16)

    dot_val = torch.dot(pred_flat, noise_flat)
    denom = pred_flat.norm() * noise_flat.norm() + 1e-8
    return float((dot_val / denom).item())


# -------------------------------------------------------------------------
# 3) Load stable diffusion v1.4 pipeline
# -------------------------------------------------------------------------
def load_sdv14_functionalities(device="cuda"):
    """
    Equivalent to your get_sd_v14_model(...) + typical pipeline pieces,
    for the ldm similarity measurement.
    """
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device)

    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    unet = pipe.unet.to(device).eval().half()
    vae = pipe.vae.to(device).eval().half()
    text_encoder = pipe.text_encoder.to(device).half()
    tokenizer = pipe.tokenizer

    return unet, vae, text_encoder, tokenizer, scheduler


# -------------------------------------------------------------------------
# 4) Main: compute ldm-style similarity for memorized vs. non-memorized
# -------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet, vae, text_encoder, tokenizer, scheduler = load_sdv14_functionalities(device)

    memorized_dir = "memorized Mothers influence on young hippo"
    non_memorized_dir = "non_memorized Mother and her young hippo"

    mem_paths = glob.glob(os.path.join(memorized_dir, "*.png"))
    nonmem_paths = glob.glob(os.path.join(non_memorized_dir, "*.png"))
    print(f"Found {len(mem_paths)} memorized images, {len(nonmem_paths)} non-memorized images.")

    # We'll do multiple draws for each image to get an average
    n_draws = 8
    time_frac = 0.01
    prompt = "Mothers influence on her young hippo"

    # Gather average cos similarity for memorized
    memorized_sims = []
    for path in mem_paths:
        val = compute_avg_cosine_similarity_for_image(
            img_path=path,
            unet=unet, vae=vae, text_encoder=text_encoder,
            tokenizer=tokenizer, scheduler=scheduler,
            prompt=prompt, time_frac=time_frac,
            n_draws=n_draws,
            device=device
        )
        memorized_sims.append(val)

    # Gather average cos similarity for non-memorized
    nonmemorized_sims = []
    for path in nonmem_paths:
        val = compute_avg_cosine_similarity_for_image(
            img_path=path,
            unet=unet, vae=vae, text_encoder=text_encoder,
            tokenizer=tokenizer, scheduler=scheduler,
            prompt=prompt, time_frac=time_frac,
            n_draws=n_draws,
            device=device
        )
        nonmemorized_sims.append(val)

    # Plot the final histogram
    plt.figure(figsize=(7,5))
    if memorized_sims:
        plt.hist(memorized_sims, alpha=0.5, bins=15, label="Memorized")
    if nonmemorized_sims:
        plt.hist(nonmemorized_sims, alpha=0.5, bins=15, label="Non-Memorized")
    plt.title(f"Hippo: LDM-Style Cosine Similarity (avg of {n_draws} draws)")
    plt.xlabel("Mean Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hippo_mem_vs_nonmem_ldm_similarity_hist.png")
    plt.show()

if __name__ == "__main__":
    main()
