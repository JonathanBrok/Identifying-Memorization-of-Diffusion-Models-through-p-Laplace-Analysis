"""
p_laplace_core.py
-----------------
Core functionality for p-Laplace approximations, both via boundary flux
(sampling on a sphere) and volumetric sampling (in a ball). Useful for
scripts that either know the log p gradient explicitly (e.g. a GMM) or
approximate it (e.g. a diffusion model's score).
"""

import numpy as np
import torch
import math


def sample_sphere_normals_nd_torch(tensor_shape, n_samples=128, device="cuda", epsilon=1e-12):
    """
    Returns n_samples random directions on the unit sphere in R^(product of tensor_shape).
    The result has shape (n_samples, *tensor_shape).
    """
    # 1) Draw Gaussian noise: shape (n_samples, *tensor_shape)
    x = torch.randn((n_samples, *tensor_shape), device=device)
    # 2) Flatten to (n_samples, -1)
    x_flat = x.view(n_samples, -1)
    D = x_flat.shape[1]
    sqrt_d = math.sqrt(D)
    # 3) Compute norms per sample
    norms = x_flat.norm(dim=1, keepdim=True).clamp(min=epsilon)
    # 4) Divide each sample by its norm to get unit vectors
    x_unit_flat = x_flat / norms
    # 5) Reshape back to (n_samples, *tensor_shape)
    x_unit = x_unit_flat.view_as(x)
    return x_unit, sqrt_d



def compute_p_laplace_boundary_torch(
    center:      torch.Tensor,  # shape (*latent_dims), e.g. (4, 64, 64)
    radius_factor:      float or torch.Tensor,
    p:           float,
    get_logp_gradients,         # function: (pts: shape (N, *latent_dims)) -> grads of same shape
    n_samples=128,
    epsilon=1e-12
):
    """
    Monte-Carlo approximation of the p-Laplace at 'center' using the boundary approach:

    1) Sample sphere directions around 'center' with radius 'radius'
    2) Evaluate grad log p at boundary points
    3) If p=1, we do (grad / ||grad||) dot normal
       Else p != 1, we do ||grad||^(p-2)*grad dot normal
    4) Return the mean of those dot-products (scalar)

    All operations are Torch-based, no NumPy required.

    Args:
      center: shape (*latent_dims). Typically (4,64,64) for SD latents.
      radius: float or torch scalar
      p: 1, 2, etc.
      get_logp_gradients: function from
         (N, *latent_dims) -> (N, *latent_dims), in Torch
      n_samples: number of boundary samples
      epsilon: small constant to avoid division by zero
    """
    # print('verify!!')
    # exit()
    device = center.device

    # 1) Sample random unit normals in the same dimension as 'center' shape => (n_samples, *center.shape)
    normals, sqrt_d = sample_sphere_normals_nd_torch(
        tensor_shape=center.shape,
        n_samples=n_samples,
        device=device,
        epsilon=epsilon,
    )
    
    
    # 2) Points on boundary: broadcast center => (n_samples, *center.shape)
    #    shape => (n_samples, *latent_dims)
    points_on_sphere = center.unsqueeze(0) + radius_factor * sqrt_d * normals.to(center.dtype)

    # 3) Evaluate gradient at each boundary point
    #    shape => (n_samples, *latent_dims)
    grads = get_logp_gradients(points_on_sphere)

    n_samples_ = grads.shape[0]  # grads is (n_samples, *latent_dims)
    grads_flat = grads.view(n_samples_, -1).to(torch.float16)
    normals_flat = sqrt_d * normals.view(n_samples_, -1).to(torch.float16)

    # Dot products of each pair
    dot_vals = torch.sum(grads_flat * normals_flat, dim=1)

    # Norm of each gradient, norm of each normal
    norm_grads = grads_flat.norm(dim=1)
    norm_normals = normals_flat.norm(dim=1)

    denom = (norm_grads * norm_normals).clamp(min=1e-8)
    flux_vals = dot_vals / denom  # shape (n_samples,)

    if p != 1.0:
        scale = norm_grads ** (p - 2)
        flux_vals *= scale
    return flux_vals.mean()



def sample_ball_points(dimension, radius, n_samples):
    """
    Samples n_samples points uniformly in a 'dimension'-dim ball of radius 'radius'.
    Returns an (n_samples x dimension) NumPy array.
    """
    dirs = np.random.randn(n_samples, dimension)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # radius^(1/d) approach
    u = np.random.rand(n_samples)
    r = radius * (u ** (1.0 / dimension))

    return dirs * r[:, None]

def numeric_p_lap_at_point(point, p, get_logp_gradients, delta=1e-3, epsilon=1e-8):
    """
    Numerically approximates p-laplace = div(||grad||^{p-2} * grad) at a single point
    via finite differences. The function get_logp_gradients(...) must return the
    gradient of log p at arbitrary points.
    """
    dim = len(point)

    def p_field(x):
        grad = get_logp_gradients(x.reshape(1, -1))[0]  # shape (dim,)
        norm_grad = np.linalg.norm(grad) + epsilon
        factor = norm_grad ** (p - 2)
        return factor * grad  # shape (dim,)

    # finite-differences for divergence
    div_val = 0.0
    for i in range(dim):
        e_i = np.zeros(dim)
        e_i[i] = 1.0
        f_plus = p_field(point + delta * e_i)
        f_minus = p_field(point - delta * e_i)
        div_val += (f_plus[i] - f_minus[i]) / (2 * delta)

    return div_val



def compute_p_laplace_volume(center, radius, p, get_logp_gradients, n_samples=128, delta=1e-3):
    """
    Approximates p-Laplace by volumetric sampling and numerical divergence:
      1) sample points inside the ball
      2) do finite-difference to estimate div( ||grad||^(p-2)*grad ) for each
      3) average
    """
    dim = len(center)
    points_in_ball = sample_ball_points(dim, radius, n_samples)

    values = []
    for pt in points_in_ball:
        actual_pt = pt + center  # shift by center
        val = numeric_p_lap_at_point(actual_pt, p, get_logp_gradients, delta=delta)
        values.append(val)

    return np.mean(values)


#########################################################################################
## Legacy Functions: Numpy Boundary Formulation  (unused. We use pytorch versions now) ##
#########################################################################################


def old_sample_sphere_normals(dimension, n_samples):
    """
    Samples n_samples directions uniformly on the unit sphere in 'dimension' dims.
    Returns an (n_samples x dimension) NumPy array of unit vectors.
    """
    vectors = np.random.randn(n_samples, dimension)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def old_compute_p_laplace_boundary(center, radius, p, get_logp_gradients, n_samples=128, epsilon=1e-12):
    """
    Monte-Carlo approximation of the p-Laplace at 'center' with sphere radius 'radius',
    by sampling boundary points:
      1) sample directions on the sphere
      2) evaluate grad log p
      3) if p=1, use normalized grad dot normal
         else if p!=1, multiply grad by ||grad||^(p-2) before dot normal
      4) return the mean of those dot-products
    """
    dim = len(center)
    normals = sample_sphere_normals(dim, n_samples)
    points_on_sphere = center + radius * normals

    grads = get_logp_gradients(points_on_sphere)  # shape (n_samples, dim)
    grad_norms = np.linalg.norm(grads, axis=1, keepdims=True) + epsilon

    if abs(p - 1.0) < 1e-9:
        # 1-Lap
        normalized = grads / grad_norms
        flux_vals = np.einsum("ij,ij->i", normalized, normals)
        return np.mean(flux_vals)
    else:
        # p-Lap
        scale = np.power(grad_norms, p - 2)
        scaled_grads = scale * grads
        flux_vals = np.einsum("ij,ij->i", scaled_grads, normals)
        return np.mean(flux_vals)