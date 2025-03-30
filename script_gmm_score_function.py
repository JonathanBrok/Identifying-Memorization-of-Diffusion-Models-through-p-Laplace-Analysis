#!/usr/bin/env python
""" script_gmm_score_function.py """

import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

from p_laplace_core import (
    compute_p_laplace_boundary_torch,
    compute_p_laplace_volume
)
from toy_diffusion_train import (
    train_toy_diffusion_model,
    get_diffusion_score_fn
)

###############################################################################
# 1) GMM in 2D
###############################################################################
def create_2d_gmm():
    """
    3-mode GMM in 2D, each cov=0.1*I, uniform weights.
    """
    M = 5.0
    d=2
    mu1 = np.zeros(d)
    mu2 = np.zeros(d)
    mu3 = np.zeros(d)

    mu2[0] = M
    mu3[0] = 0.5*M
    mu3[1] = (np.sqrt(3)/2)*M

    means_list = [mu1, mu2, mu3]
    sum_means = np.sum(means_list, axis=0)
    mean_offset = sum_means/3
    means_list = [m - mean_offset for m in means_list]

    weights = np.ones(3)/3
    covs = [0.1*np.eye(d) for _ in range(3)]
    means = np.array(means_list)
    return means, covs, weights

def get_logp_grad_gmm_torch(pts_t, means, covs, weights):
    """
    Evaluate grad log p(x) for the GMM in Torch, shape=(N,2).
    """
    pts_np = pts_t.detach().cpu().numpy()
    n, d = pts_np.shape
    pdf_vals = np.zeros((n, len(weights)), dtype=float)
    comp_grads= np.zeros((n, len(weights), d), dtype=float)

    for k,(mean_k, cov_k, w_k) in enumerate(zip(means, covs, weights)):
        invC = np.linalg.inv(cov_k)
        diff = pts_np - mean_k
        exps = np.einsum("ij,jk,ik->i", diff, invC, diff)
        detC = np.linalg.det(cov_k)
        norm_factor= np.sqrt((2*math.pi)**d * detC)
        comp_pdf = w_k*np.exp(-0.5*exps)/norm_factor
        pdf_vals[:,k]=comp_pdf
        grad_k= -np.einsum("ij,jk->ik", diff, invC)
        comp_grads[:,k,:] = comp_pdf[:,None]*grad_k

    denom= np.sum(pdf_vals, axis=1)+1e-15
    numerator = np.sum(comp_grads, axis=1)
    grads_np= numerator/denom[:,None]
    return torch.from_numpy(grads_np).float().to(pts_t.device)

###############################################################################
# 2) measure for sphere/ball
###############################################################################
def surface_area_sphere(r, d):
    from math import gamma, pi
    return d*(pi**(d/2))/gamma(d/2+1)*(r**(d-1))

def volume_ball(r, d):
    from math import gamma, pi
    return (pi**(d/2))/gamma(d/2+1)*(r**d)

###############################################################################
# 3) total flux with inward>0 => multiply by -1
###############################################################################
def compute_boundary_flux_inward(pt_np, p, radius, d, get_grad, n_samples=64):
    """
    local => measure => multiply by -1 => inward positive
    """
    center_t = torch.from_numpy(pt_np).float()
    local_val = compute_p_laplace_boundary_torch(
        center=center_t,
        radius_factor=radius/math.sqrt(d),
        p=p,
        get_logp_gradients=get_grad,
        n_samples=n_samples
    )
    measure= surface_area_sphere(radius, d)
    flux_val= local_val.item()*measure
    return -flux_val

def compute_volume_flux_inward(pt_np, p, radius, d, get_grad, n_samples=64):
    val_local = compute_p_laplace_volume(
        pt_np, radius, p,
        get_logp_gradients=get_grad,
        n_samples=n_samples,
        delta=1e-3
    )
    measure= volume_ball(radius, d)
    flux_val= val_local*measure
    return -flux_val

###############################################################################
# Main script => one figure with 5 rows x 2 cols
###############################################################################
def main():
    # 1) create GMM + define test points
    means, covs, weights= create_2d_gmm()
    d=2
    p=1
    radius=1.0

    # local maxima => 3 means
    test_pts=[]
    for m_ in means:
        test_pts.append(m_)
    # pairwise midpoints
    num_modes=len(means)
    for i in range(num_modes):
        for j in range(i+1, num_modes):
            mid= 0.5*(means[i]+means[j])
            test_pts.append(mid)
    # average
    center_= np.mean(means, axis=0)
    test_pts.append(center_)
    test_pts=[np.array(pt, dtype=float) for pt in test_pts]
    # separate max vs nonmax
    maxima_indices= list(range(num_modes))  # 0..2
    nonmax_indices= list(range(num_modes, len(test_pts)))

    # 2) train diffusion => bridging
    from toy_diffusion_train import train_toy_diffusion_model, get_diffusion_score_fn
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model, betas, alphas_cumprod, _ = train_toy_diffusion_model(
        d=d,
        gmm_means=means,
        gmm_covs=covs,
        gmm_weights=weights,
        n_samples=2000,
        n_epochs=300,
        device=device
    )
    diff_score_fn = get_diffusion_score_fn(model, betas, alphas_cumprod, device, T=100)

    def bridging_exact_torch(pts_t):
        return get_logp_grad_gmm_torch(pts_t, means, covs, weights)

    def bridging_diff_torch(pts_t):
        pts_np_= pts_t.detach().cpu().numpy()
        out_list=[]
        for row_ in pts_np_:
            out_list.append(diff_score_fn(row_))
        return torch.from_numpy(np.array(out_list)).float()

    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    fig= plt.figure(figsize=(12,18))
    gs = gridspec.GridSpec(5,2, figure=fig, wspace=0.3, hspace=0.4)

    # Row 1 => quiver EXACT vs DIFF
    ax_quiver_exact = fig.add_subplot(gs[0, 0])
    ax_quiver_diff  = fig.add_subplot(gs[0, 1])

    ax_quiver_exact.set_box_aspect(1)
    ax_quiver_diff.set_box_aspect(1)
    

    # Row 2 => boundary EXACT => [1,0]=nonmax, [1,1]=max
    ax_bdy_exact_non = fig.add_subplot(gs[1, 0])
    ax_bdy_exact_max = fig.add_subplot(gs[1, 1])

    ax_bdy_exact_non.set_box_aspect(1)
    ax_bdy_exact_max.set_box_aspect(1)

    # Row 3 => boundary DIFF => [2,0]=nonmax, [2,1]=max
    ax_bdy_diff_non  = fig.add_subplot(gs[2, 0])
    ax_bdy_diff_max  = fig.add_subplot(gs[2, 1])

    ax_bdy_diff_non.set_box_aspect(1)
    ax_bdy_diff_max.set_box_aspect(1)

    # Row 4 => volume EXACT => [3,0]=nonmax, [3,1]=max
    ax_vol_exact_non = fig.add_subplot(gs[3, 0])
    ax_vol_exact_max = fig.add_subplot(gs[3, 1])

    ax_vol_exact_non.set_box_aspect(1)
    ax_vol_exact_max.set_box_aspect(1)

    # Row 5 => volume DIFF => [4,0]=nonmax, [4,1]=max
    ax_vol_diff_non  = fig.add_subplot(gs[4, 0])
    ax_vol_diff_max  = fig.add_subplot(gs[4, 1])

    ax_vol_diff_non.set_box_aspect(1)
    ax_vol_diff_max.set_box_aspect(1)

    
    
    # ========== Row 1: Quiver EXACT vs DIFF ==========
    x_min,x_max= -8,8
    y_min,y_max= -8,8
    N=200
    x_vals= np.linspace(x_min,x_max,N)
    y_vals= np.linspace(y_min,y_max,N)
    X, Y= np.meshgrid(x_vals, y_vals)

    def logp_2d(pt_):
        pdfsum=0.0
        for (m_k,c_k,w_k) in zip(means,covs,weights):
            diff= pt_-m_k
            invC= np.linalg.inv(c_k)
            exponent= -0.5*(diff@invC@diff)
            detC= np.linalg.det(c_k)
            denom= np.sqrt((2*math.pi)**2*detC)
            pdfsum+= w_k*np.exp(exponent)/denom
        return np.log(pdfsum+1e-15)

    logp_grid= np.zeros_like(X)
    for irow in range(N):
        for jcol in range(N):
            pt__= np.array([X[irow,jcol], Y[irow,jcol]])
            logp_grid[irow,jcol]= logp_2d(pt__)

    # EXACT quiver
    ax_quiver_exact.contourf(X, Y, logp_grid, levels=20, cmap='gray', alpha=0.4)
    QN=15
    x_q= np.linspace(x_min,x_max, QN)
    y_q= np.linspace(y_min,y_max, QN)
    Xq, Yq= np.meshgrid(x_q, y_q)
    coords_q= np.c_[Xq.ravel(), Yq.ravel()]
    coords_t= torch.from_numpy(coords_q).float()
    grad_ex= bridging_exact_torch(coords_t).cpu().numpy()
    eps=1e-8
    norm_ex= np.linalg.norm(grad_ex, axis=1, keepdims=True)+eps
    grad_ex_unit= grad_ex/norm_ex
    U_ex= grad_ex_unit[:,0].reshape(QN,QN)
    V_ex= grad_ex_unit[:,1].reshape(QN,QN)
    ax_quiver_exact.quiver(Xq, Yq, U_ex, V_ex, color='blue', scale=40)
    ax_quiver_exact.set_title("Exact Score (Normalized)")

    # circles
    colors_pts= cm.tab10(np.linspace(0,1,len(test_pts)))
    for i, pt_ in enumerate(test_pts):
        circ= plt.Circle(pt_, radius, fill=False, linewidth=2, color=colors_pts[i])
        ax_quiver_exact.add_patch(circ)
        ax_quiver_exact.text(pt_[0], pt_[1], f"{i}", color=colors_pts[i],
                             ha='center', va='center', fontsize=10)
    ax_quiver_exact.set_xlabel("x"); ax_quiver_exact.set_ylabel("y")
    # ax_quiver_exact.set_aspect('equal', 'box')

    # DIFF quiver
    ax_quiver_diff.contourf(X, Y, logp_grid, levels=20, cmap='gray', alpha=0.4)
    grad_df= bridging_diff_torch(coords_t).cpu().numpy()
    norm_df= np.linalg.norm(grad_df, axis=1, keepdims=True)+eps
    grad_df_unit= grad_df/norm_df
    U_df= grad_df_unit[:,0].reshape(QN,QN)
    V_df= grad_df_unit[:,1].reshape(QN,QN)
    ax_quiver_diff.quiver(Xq, Yq, U_df, V_df, color='red', scale=40)
    ax_quiver_diff.set_title("Diff Score (Normalized)")

    for i, pt_ in enumerate(test_pts):
        circ= plt.Circle(pt_, radius, fill=False, linewidth=2, color=colors_pts[i])
        ax_quiver_diff.add_patch(circ)
        ax_quiver_diff.text(pt_[0], pt_[1], f"{i}", color=colors_pts[i],
                            ha='center', va='center', fontsize=10)
    ax_quiver_diff.set_xlabel("x"); ax_quiver_diff.set_ylabel("y")
    # ax_quiver_diff.set_aspect('equal', 'box')

    # ========== Rows 2-5 => hist for boundary or volume, EXACT or DIFF, (nonmax vs max) ==========
    hist_repeats=200
    hist_samples=64

    def flux_boundary_exact(pt_np):
        return compute_boundary_flux_inward(pt_np, p, radius, d,
                    get_grad=bridging_exact_torch, n_samples=hist_samples)
    def flux_boundary_diff(pt_np):
        return compute_boundary_flux_inward(pt_np, p, radius, d,
                    get_grad=bridging_diff_torch, n_samples=hist_samples)

    def flux_volume_exact(pt_np):
        def bridging_np(arr_np):
            arr_t= torch.from_numpy(arr_np).float()
            out_t= bridging_exact_torch(arr_t)
            return out_t.cpu().numpy()
        return compute_volume_flux_inward(pt_np, p, radius, d,
                                          get_grad=bridging_np,
                                          n_samples=hist_samples)
    def flux_volume_diff(pt_np):
        def bridging_np(arr_np):
            out_list=[]
            for row_ in arr_np:
                out_list.append(diff_score_fn(row_))
            return np.array(out_list)
        return compute_volume_flux_inward(pt_np, p, radius, d,
                                          get_grad=bridging_np,
                                          n_samples=hist_samples)

    # Subplots row2 => boundary EXACT => [1,0]=nonmax, [1,1]=max
    ax_bny_ex_non = ax_bdy_exact_non
    ax_bny_ex_max = ax_bdy_exact_max
    ax_bny_ex_non.set_title("Boundary Nonmax (Exact)")
    ax_bny_ex_max.set_title("Boundary Max (Exact)")
    # row3 => boundary DIFF => [2,0]=nonmax, [2,1]=max
    ax_bny_df_non = ax_bdy_diff_non
    ax_bny_df_max = ax_bdy_diff_max
    ax_bny_df_non.set_title("Boundary Nonmax (Diff)")
    ax_bny_df_max.set_title("Boundary Max (Diff)")

    # row4 => volume EXACT => [3,0]=nonmax, [3,1]=max
    ax_vol_ex_non = ax_vol_exact_non
    ax_vol_ex_max = ax_vol_exact_max
    ax_vol_ex_non.set_title("Volume Nonmax (Exact)")
    ax_vol_ex_max.set_title("Volume Max (Exact)")
    # row5 => volume DIFF => [4,0]=nonmax, [4,1]=max
    ax_vol_df_non = ax_vol_diff_non
    ax_vol_df_max = ax_vol_diff_max
    ax_vol_df_non.set_title("Volume Nonmax (Diff)")
    ax_vol_df_max.set_title("Volume Max (Diff)")

    boundary_ex_non_vals= []
    boundary_ex_max_vals= []
    boundary_df_non_vals= []
    boundary_df_max_vals= []

    volume_ex_non_vals= []
    volume_ex_max_vals= []
    volume_df_non_vals= []
    volume_df_max_vals= []

    def plot_hist(ax, data_list, color_, label_):
        ax.hist(data_list, bins=25, density=True, alpha=0.5, color=color_, label=label_)
        mean_= np.mean(data_list)
        ax.axvline(mean_, color=color_, linestyle='--', linewidth=2)

    # gather repeated flux for each point
    # Suppose you already have:
    #   offsets = np.linspace(-0.04, 0.04, len(test_pts))

    n_test_pts = len(test_pts)
    offsets = np.linspace(-0.004, 0.004, n_test_pts)  # strictly for visualization purposes
    for i, pt_ in enumerate(test_pts):
        c_ = colors_pts[i]
        label_ = f"Pt {i}"
        boundary_ex_ = []
        boundary_df_ = []
        volume_ex_   = []
        volume_df_   = []

        # gather flux arrays as usual
        for rep in range(hist_repeats):
            boundary_ex_.append(flux_boundary_exact(pt_))
            boundary_df_.append(flux_boundary_diff(pt_))
            volume_ex_.append(flux_volume_exact(pt_))
            volume_df_.append(flux_volume_diff(pt_))

        #   # strictly for visualization purposes
        # If i < num_modes => test point is "max"
        # SHIFT ONLY boundary_ex_ / boundary_df_  => "boundary + max"
        if i < num_modes:
            offset_i = offsets[i]
            boundary_ex_ = [val + offset_i for val in boundary_ex_]
            boundary_df_ = [val + offset_i for val in boundary_df_]
            # We do NOT shift volume_ex_ or volume_df_
            # i.e. volume arrays remain unchanged

        # now do your usual "plot_hist" calls & store them
        if i < num_modes:
            # maxima
            plot_hist(ax_bny_ex_max, boundary_ex_, c_, label_)
            boundary_ex_max_vals.extend(boundary_ex_)

            plot_hist(ax_bny_df_max, boundary_df_, c_, label_)
            boundary_df_max_vals.extend(boundary_df_)

            plot_hist(ax_vol_ex_max, volume_ex_, c_, label_)
            volume_ex_max_vals.extend(volume_ex_)

            plot_hist(ax_vol_df_max, volume_df_, c_, label_)
            volume_df_max_vals.extend(volume_df_)
        else:
            # nonmax
            plot_hist(ax_bny_ex_non, boundary_ex_, c_, label_)
            boundary_ex_non_vals.extend(boundary_ex_)

            plot_hist(ax_bny_df_non, boundary_df_, c_, label_)
            boundary_df_non_vals.extend(boundary_df_)

            plot_hist(ax_vol_ex_non, volume_ex_, c_, label_)
            volume_ex_non_vals.extend(volume_ex_)

            plot_hist(ax_vol_df_non, volume_df_, c_, label_)
            volume_df_non_vals.extend(volume_df_)


    # Suppose your final flux arrays are:
    #   boundary_ex_non_vals, boundary_ex_max_vals
    #   boundary_df_non_vals, boundary_df_max_vals
    #   volume_ex_non_vals,   volume_ex_max_vals
    #   volume_df_non_vals,   volume_df_max_vals
    # and your subplots are:
    #   ax_bdy_exact_non, ax_bdy_exact_max
    #   ax_bdy_diff_non,  ax_bdy_diff_max
    #   ax_vol_exact_non, ax_vol_exact_max
    #   ax_vol_diff_non,  ax_vol_diff_max




    
    
    ##############################################################################
    # 1) BOUNDARY NONMAX => unify x-limits
    ##############################################################################
    all_bound_nonmax = boundary_ex_non_vals + boundary_df_non_vals
    if len(all_bound_nonmax) > 0:
        min_n = min(all_bound_nonmax)
        max_n = max(all_bound_nonmax)
        margin = 0.05 * (max_n - min_n)
        left_n = min_n - margin
        right_n = max_n + margin

        # set x-limits for the boundary NONMAX subplots
        ax_bdy_exact_non.set_xlim([left_n, right_n])
        ax_bdy_diff_non.set_xlim([left_n, right_n])

    ##############################################################################
    # 2) BOUNDARY MAX => unify x-limits
    ##############################################################################
    all_bound_max = boundary_ex_max_vals + boundary_df_max_vals
    if len(all_bound_max) > 0:
        min_m = min(all_bound_max)
        max_m = max(all_bound_max)
        margin = 0.05 * (max_m - min_m)
        left_m = min_m - margin
        right_m = max_m + margin

        # set x-limits for the boundary MAX subplots
        ax_bdy_exact_max.set_xlim([left_m, right_m])
        ax_bdy_diff_max.set_xlim([left_m, right_m])

        ##############################################################################
        # 3) VOLUME NONMAX => unify x-limits
        ##############################################################################
        all_vol_nonmax = volume_ex_non_vals + volume_df_non_vals
        if len(all_vol_nonmax) > 0:
            min_vn = min(all_vol_nonmax)
            max_vn = max(all_vol_nonmax)
            margin = 0.05 * (max_vn - min_vn)
            left_vn = min_vn - margin
            right_vn = max_vn + margin

            ax_vol_exact_non.set_xlim([left_vn, right_vn])
            ax_vol_diff_non.set_xlim([left_vn, right_vn])

        ##############################################################################
        # 4) VOLUME MAX => unify x-limits
        ##############################################################################
        all_vol_max = volume_ex_max_vals + volume_df_max_vals
        if len(all_vol_max) > 0:
            min_vm = min(all_vol_max)
            max_vm = max(all_vol_max)
            margin = 0.05 * (max_vm - min_vm)
            left_vm = min_vm - margin
            right_vm = max_vm + margin

            ax_vol_exact_max.set_xlim([left_vm, right_vm])
            ax_vol_diff_max.set_xlim([left_vm, right_vm])


        # set square aspect for all
        all_axes= [ax_quiver_exact, ax_quiver_diff,
                ax_bny_ex_non, ax_bny_ex_max, ax_bny_df_non, ax_bny_df_max,
                ax_vol_ex_non, ax_vol_ex_max, ax_vol_df_non, ax_vol_df_max]
        for ax_ in all_axes:
            # ax_.set_aspect('equal','box')
            ax_.legend(fontsize=8)
            ax_.grid(True)
            ax_.set_xlabel("Flux (inward>0)")
            ax_.set_ylabel("Density")

        # fix row1 labeling
        ax_quiver_exact.set_xlabel("x")
        ax_quiver_exact.set_ylabel("y")
        ax_quiver_diff.set_xlabel("x")
        ax_quiver_diff.set_ylabel("y")

        fig.tight_layout()
        out_fig = "expr_gmm.png"
        plt.savefig(out_fig, dpi=150)
        print(f"\nSaved => {out_fig}\n")

if __name__=="__main__":
    main()
    print("Done.")
