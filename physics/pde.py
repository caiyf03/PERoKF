from typing import Tuple
import math
import torch
import torch.nn.functional as F


def compute_time_derivative(w, Re, fn, dealias: bool = True) -> torch.Tensor:
    """
    Batched Kolmogorov flow dw/dt with varying Re and forcing n.
    
    Args:
        w  : [B, 1, S, S] real tensor
        Re : [B, 1] tensor
        fn : [B, 1] integer tensor (guaranteed > 0)
    """
    device = w.device
    B, C, S, _ = w.shape
    
    # --- 1. Reshape Re for broadcasting [B, 1] -> [B, 1, 1, 1] ---
    Re_expanded = Re.view(B, 1, 1, 1)

    # ---- Fourier Grid ----
    k_vals = torch.cat([
        torch.arange(0, S//2, device=device, dtype=torch.float32),
        torch.arange(-S//2, 0, device=device, dtype=torch.float32)
    ])
    k_y = k_vals.repeat(S, 1)
    k_x = k_y.t()
    k_x = k_x[None, None, :, :]
    k_y = k_y[None, None, :, :]

    # ---- Standard Calculations ----
    w_h = torch.fft.fft2(w, dim=(-2,-1), norm="backward")
    
    lap = k_x**2 + k_y**2
    lap_inv = torch.zeros_like(lap)
    lap_inv[lap != 0] = 1.0 / lap[lap != 0]
    
    psi_h = lap_inv * w_h
    q_h =  1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    
    q = torch.fft.irfft2(q_h, s=(S, S), norm="backward")
    v = torch.fft.irfft2(v_h, s=(S, S), norm="backward")
    
    t1 = torch.fft.fft2(q * w, dim=(-2,-1), norm="backward") * k_x
    t2 = torch.fft.fft2(v * w, dim=(-2,-1), norm="backward") * k_y
    nonlinear = -1j * (t1 + t2)

    # --- 2. Forcing Term  ---
    # Calculate forcing value: [B]
    forcing_val = (fn.float().squeeze(1) / 2.0) * (S**2)
    
    # Indices
    batch_idx = torch.arange(B, device=device)
    fn_idx = fn.squeeze(1).long()
    
    # Apply forcing directly to specific batch indices
    # [batch_idx, channel, y_k=0, x_k=fn]
    nonlinear[batch_idx, 0, 0, fn_idx] -= forcing_val
    nonlinear[batch_idx, 0, 0, -fn_idx] -= forcing_val

    # --- 3. Modified Linear Term ---
    linear = (lap / Re_expanded + 0.1) * w_h

    # ---- Combine ----
    dw_h_dt = nonlinear - linear

    # ---- Dealiasing ----
    if dealias:
        dealias_mask = ((lap <= (S/3.0)**2).float())
        dealias_mask[0,0,0,0] = 0.0
        dw_h_dt = dw_h_dt * dealias_mask

    return torch.fft.irfft2(dw_h_dt, s=(S, S), norm="backward")

def compute_divergence(w):
    """
    Compute divergence(u) from a batched 2D vorticity field.
    
    Input:
        w : [B, 1, S, S] real tensor (vorticity)
    Output:
        div : [B, 1, S, S] real tensor (divergence)
    """

    B, C, S, _ = w.shape
    assert C == 1, "Expected channel dimension = 1"

    device = w.device
    dtype  = torch.float32

    # ---- Construct Fourier wavenumbers ----
    k_vals = torch.cat([
        torch.arange(0, S//2,     device=device, dtype=dtype),
        torch.arange(-S//2, 0,    device=device, dtype=dtype)
    ])

    # shape [S, S]
    k_y = k_vals.repeat(S, 1)
    k_x = k_y.t()

    # reshape for broadcasting with batch: [1,1,S,S]
    k_x = k_x[None, None, :, :]
    k_y = k_y[None, None, :, :]

    # ---- FFT of vorticity ----
    w_h = torch.fft.fft2(w, dim=(-2, -1), norm="backward")  # shape [B,1,S,S]

    # ---- Inverse Laplacian ----
    lap = k_x**2 + k_y**2
    lap_inv = torch.zeros_like(lap)
    lap_inv[lap != 0] = 1.0 / lap[lap != 0]

    # ---- Streamfunction ψ ----
    psi_h = lap_inv * w_h

    # ---- Velocity in Fourier space ----
    # u_h =  i k_y ψ_h
    # v_h = -i k_x ψ_h
    u_h =  1j * k_y * psi_h
    v_h = -1j * k_x * psi_h

    # ---- Divergence in Fourier space ----
    # div_h = i k_x u_h + i k_y v_h
    div_h = 1j * k_x * u_h + 1j * k_y * v_h

    # ---- Back to real space ----
    div = torch.fft.ifft2(div_h, dim=(-2, -1), norm="backward").real

    return div 

def compute_deriv_residual_map(w_pred: torch.Tensor, w_true: torch.Tensor, Re: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """Compute the residual map of the time derivative between predicted and true vorticity fields.

    w_pred: [bs, 1, nx, ny]
    w_true: [bs, 1, nx, ny]
    returns: [bs, 1, nx, ny] tensor
    """
    dw_dt_pred = compute_time_derivative(w_pred, Re, fn)
    dw_dt_true = compute_time_derivative(w_true, Re, fn)
    residual_map = dw_dt_pred - dw_dt_true
    return residual_map

def derivative_loss_l2(w_pred: torch.Tensor, w_true: torch.Tensor, Re: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """Compute L2 norm of the time derivative residual between predicted and true vorticity fields.

    w_pred: [bs, 1, nx, ny]
    w_true: [bs, 1, nx, ny]
    returns scalar tensor
    """
    residual_map = compute_deriv_residual_map(w_pred, w_true, Re, fn)
    return torch.mean(residual_map ** 2)

def divergence_loss_l2(w: torch.Tensor) -> torch.Tensor:
    """Compute L2 norm of the PDE residual on predicted vorticity field.

    w: [bs, 1, nx, ny]
    Re: [bs, 1]
    fn: [bs, 1]
    returns scalar tensor
    """
    res = compute_divergence(w)
    return torch.mean(res ** 2)

def energy_spectrum_from_vorticity(w):
    """
    Compute isotropic energy spectrum E(k) from 2D vorticity field.
    
    Args:
        w: [bs, 1, H, W] vorticity tensor
    
    Returns:
        E: [bs, k_max+1] isotropic energy spectrum
    """

    bs, _, H, W = w.shape

    w_hat = torch.fft.fft2(w, norm="forward")
    w_hat = w_hat.squeeze(1)

    kx = torch.fft.fftfreq(W, d=1.0) * W
    ky = torch.fft.fftfreq(H, d=1.0) * H
    kx, ky = torch.meshgrid(ky, kx, indexing="ij")
    kx = kx.to(w.device)
    ky = ky.to(w.device)

    k2 = kx**2 + ky**2
    k = torch.sqrt(k2)

    k2_safe = torch.where(k2 == 0, torch.ones_like(k2), k2)

    u_hat = 1j * ky * w_hat / k2_safe
    v_hat = -1j * kx * w_hat / k2_safe

    E_kxy = 0.5 * (torch.abs(u_hat)**2 + torch.abs(v_hat)**2)

    k_int = torch.round(k).long()
    k_max = int(k_int.max().item())

    E_radial = torch.zeros(bs, k_max+1, device=w.device)
    
    for i in range(k_max+1):
        mask = (k_int == i)
        if mask.sum() > 0:
            E_radial[:, i] = E_kxy[:, mask].mean(dim=1)

    return E_radial  # shape [bs, k_max+1]