# Physics-Enhanced Super-Resolution of Kolmogorov Flow

This repository contains the official implementation for **Physics-Enhanced Reconstruction of High-Resolution Kolmogorov Flow from Low-Resolution Snapshots**.

The project studies how to inject **physics priors** into modern neural networks for **single-frame fluid super-resolution**, without requiring temporal supervision.

---

## Problem Setting

Given a **single low-resolution vorticity snapshot**:

- Input (LR): `128 × 128`
- Target (HR): `512 × 512`

the goal is to reconstruct fine-scale turbulent structures destroyed by downsampling.

This single-frame setting isolates **spatial reconstruction ability** and reflects realistic scenarios where temporal data is unavailable.

---

## Models

We evaluate four model families:

- **CNN**: lightweight hierarchical convolutional baseline
- **UNet**: multi-scale ResNet-style UNet with optional attention
- **FNO**: Fourier Neural Operator with truncated spectral modes
- **Diffusion**: conditional DDPM with UNet backbone (x₀-prediction)

All models operate on **bicubic-upsampled LR inputs** and predict HR vorticity fields.

---

## Physics-Guided Strategies

Two physics-based mechanisms are studied:

1. **Physics-derived feature augmentation**
   - Laplacian
   - Streamfunction
   - Velocity components
   - Nonlinear advection terms

2. **Physics-consistency loss**
   - Navier–Stokes–based residual loss
   - Uses the same pseudo-spectral operator as data generation
   - Enforces consistency in implied time derivatives

---

## Key Findings

- Physics-consistency loss:
  - Consistently improves physical accuracy
  - Reduces PDE residual error and energy spectrum error
  - Accelerates diffusion model convergence

- Physics features alone:
  - Not consistently beneficial
  - May destabilize training

- Combining both:
  - Can improve accuracy
  - May cause instability under limited compute (especially for diffusion models)

---

## Dataset

**Flow type**: 2D Kolmogorov Flow  
**Resolution**:
- HR: `512 × 512`
- LR: `128 × 128` (4× downsampling)

**Parameters**:
- Reynolds numbers: `1000 / 2000 / 3000`
- Forcing wavenumbers: `8 / 12 / 16`

**Scale**:
- 14,400 HR samples
- ~15 GB, `.npy` format

Dataset is hosted at:
https://huggingface.co/datasets/skpy/PSFR

---

## Evaluation Metrics

We report three complementary metrics:

- **MSE**: pixel-wise reconstruction error
- **Physics Consistency Error (PCE)**: ℓ₂ error between Navier–Stokes-induced time derivatives
- **Energy Spectrum Error (ESE)**: relative error of mid–high frequency isotropic energy spectrum

Physics-based metrics are the primary indicators of model quality.

---

## Visualization

Visualization scripts are provided to compare:
- Ground truth HR
- Low-resolution input
- Reconstructed HR fields

Typical outputs include vorticity fields and energy spectra.

---

## Reference

If you use this code or dataset, please refer to:

**Physics-Enhanced Reconstruction of High-Resolution Kolmogorov Flow from Low-Resolution Snapshots**  
University of Pennsylvania, CIS 5200 Machine Learning Project, YiFan Cai
