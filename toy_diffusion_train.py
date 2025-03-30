"""
toy_diffusion_train.py

Trains a small diffusion model on 2D (or higher-dimensional) data sampled from a
GMM distribution (or another distribution). Exposes:

1) train_toy_diffusion_model(...) -> returns a trained DenoiseModel, plus helpful objects
2) get_diffusion_score_fn(...) -> wraps the trained model as a function that returns approx grad log p
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

########################################
# 1) GMM data sampling (or other distribution)
########################################

def sample_gmm(
    means, covariances, weights, n_samples=2000, device=None
):
    """
    Sample 'n_samples' points from a Gaussian mixture specified by
      means (list of arrays),
      covariances (list of arrays),
      weights (list or array) summing to 1
    Returns a torch.Tensor on 'device' if provided, else CPU.
    """
    n_components = len(weights)
    all_samples = []
    component_choices = np.random.choice(n_components, size=n_samples, p=weights)
    for idx_comp in range(n_components):
        n_idx = np.sum(component_choices == idx_comp)
        if n_idx > 0:
            samples_idx = np.random.multivariate_normal(
                means[idx_comp], covariances[idx_comp], size=n_idx
            )
            all_samples.append(samples_idx)
    points = np.vstack(all_samples)  # shape (n_samples, dim)
    if device is not None:
        return torch.from_numpy(points).float().to(device)
    else:
        return torch.from_numpy(points).float()

########################################
# 2) Simple Diffusion Model Architecture
########################################
class DenoiseModel(nn.Module):
    """
    A small MLP-based denoising model for diffusion in d-dimensional space.
    Input dimension: d + time_embedding_size
    Output dimension: d
    Hidden size: 128
    """
    def __init__(self, d, time_embedding_dim=16, hidden_size=128):
        super(DenoiseModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d + time_embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d)
        )

    def forward(self, x, t_emb):
        # x shape: (batch_size, d)
        # t_emb shape: (batch_size, time_embedding_dim)
        h = torch.cat([x, t_emb], dim=1)
        return self.net(h)

def get_time_embedding(t, embedding_dim=16):
    """
    Basic sinusoidal time embedding: for integer t in [0,1,...,T-1],
    we embed t/T with sin/cos frequencies.
    """
    device_t = t.device
    half_dim = embedding_dim // 2
    emb_scales = torch.exp(
        torch.arange(half_dim, device=device_t) * - (np.log(10000.0) / (half_dim - 1))
    )
    emb = t.unsqueeze(1) * emb_scales.unsqueeze(0)
    # shape: (batch_size, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # shape: (B, embedding_dim)
    return emb

########################################
# 3) Diffusion Training
########################################
def train_toy_diffusion_model(
    d,
    gmm_means,
    gmm_covs,
    gmm_weights,
    n_samples=2000,
    n_epochs=500,
    T=100,
    beta_start=1e-4,
    beta_end=0.02,
    batch_size=128,
    lr=1e-3,
    device=None
):
    """
    Trains a diffusion model on a d-dimensional dataset (e.g. GMM).
    Returns (model, betas, alphas_cumprod, device).

    Steps:
      1) sample 'n_samples' points from GMM
      2) define betas, alphas
      3) train a DenoiseModel to predict the noise
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) sample data
    data_tensor = sample_gmm(gmm_means, gmm_covs, gmm_weights, n_samples, device=device)
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2) define diffusion schedule
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    def q_sample(x0, t, noise):
        """
        Forward diffusion step: sample q(x_t | x_0).
        x0 shape: (B, d)
        t shape: (B,)
        noise shape: (B, d)
        """
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).unsqueeze(1)  # shape (B,1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).unsqueeze(1)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    # 3) define model and train
    model = DenoiseModel(d=d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        for batch_data in loader:
            x0 = batch_data[0].to(device)  # shape (batch_size, d)
            bsize = x0.shape[0]
            t = torch.randint(0, T, (bsize,), device=device).long()
            noise = torch.randn_like(x0)
            x_noisy = q_sample(x0, t, noise)
            t_emb = get_time_embedding(t / float(T))  # shape (B, 16)
            noise_pred = model(x_noisy, t_emb)

            loss = loss_fn(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"[Diffusion Train] Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    return model, betas, alphas_cumprod, device

########################################
# 4) Wrap model as a “score function”
########################################
def get_diffusion_score_fn(model, betas, alphas_cumprod, device, T=100):
    """
    Returns a function that, given a (d,)-shaped point as NumPy,
    computes the approximate log p gradient = -noise_pred from the model at t=0.
    """
    def score_fn(point_np):
        """
        point_np: shape (d,)
        returns: shape (d,) approximate grad of log p
        """
        with torch.no_grad():
            x_in = torch.from_numpy(point_np).float().unsqueeze(0).to(device)  # shape (1, d)
            # Evaluate model at t=0
            t_zero = torch.zeros(x_in.size(0), device=device).long()
            t_emb_zero = get_time_embedding(t_zero / float(T))  # shape (1,16)
            noise_pred = model(x_in, t_emb_zero)[0].cpu().numpy()
        return -noise_pred  # minus sign for the score

    return score_fn
