# infer/visualize_sr.py
import os
import argparse
from pathlib import Path
from math import log10

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from kmflow.km_dataset import KMFlow2D_Dataset
from kmflow.km_config import KMFlow2D_DatasetConfig
from models.srcnn import SimpleResSRCNN


# ------------------------------
# Utilities
# ------------------------------
def load_checkpoint(ckpt_path, maploc="cpu"):
    ckpt = torch.load(ckpt_path, map_location=maploc)
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise ValueError("Checkpoint does not contain 'config'. Cannot reconstruct model.")
    state_dict = ckpt["model"]
    return cfg, state_dict, ckpt


def build_model_from_cfg(cfg, device):
    mcfg = cfg.get("model", {})
    model = SimpleResSRCNN(
        channels=mcfg.get("channels", 64),
        depth=mcfg.get("depth", 5),
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def make_lr(hr, scale):
    """
    hr: [1,H,W] (already standardized)
    return:
        lr_native: [1, H/scale, W/scale]  -- native LR
        lr_up:     [1, H, W]              -- for CNN, bicubic upsampling
    """
    H, W = hr.shape[-2:]
    lr_native = F.interpolate(
        hr.unsqueeze(0), size=(H // scale, W // scale),
        mode="bicubic", align_corners=False, antialias=True
    ).squeeze(0)
    lr_up = F.interpolate(
        lr_native.unsqueeze(0), size=(H, W),
        mode="bicubic", align_corners=False
    ).squeeze(0)
    return lr_native, lr_up


def choose_indices(n, num, seed=0, mode="random"):
    idx = list(range(n))
    if mode == "first":
        return idx[:min(num, n)]
    rng = np.random.default_rng(seed)
    return list(rng.choice(idx, size=min(num, n), replace=False))


def psnr(a: torch.Tensor, b: torch.Tensor, eps=1e-8) -> float:
    # a, b: [1,H,W]
    mse = F.mse_loss(a, b).item()
    rng = (b.max() - b.min()).item()
    rng = max(rng, 1.0)
    return 20 * log10(rng) - 10 * log10(mse + eps)


def viz_panels(
    lr_native: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    out_path: Path,
    title: str = None,
    cmap: str = "twilight",
    include_bicubic: bool = False,
    lr_up: torch.Tensor | None = None,
):
    """
    compare pic:
      - LR(native 未处理) | SR | HR
      - 若 include_bicubic=True provide lr_up:LR(native) | LR_up(bicubic) | SR | HR
    """
    lr_native_np = lr_native.squeeze(0).cpu().numpy()
    sr_np = sr.squeeze(0).cpu().numpy()
    hr_np = hr.squeeze(0).cpu().numpy()
    vmin, vmax = float(hr_np.min()), float(hr_np.max())

    if include_bicubic and lr_up is not None:
        cols = 4
    else:
        cols = 3

    fig, axs = plt.subplots(1, cols, figsize=(4 * cols + 2, 4), dpi=120, constrained_layout=True)

    # 1) Native LR, disabling smooth interpolation, clearly displays the "mosaic" effect.
    ax = axs[0]
    im0 = ax.imshow(lr_native_np, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title("LR (native)")
    ax.axis("off")

    col = 1
    if include_bicubic and lr_up is not None:
        lr_up_np = lr_up.squeeze(0).cpu().numpy()
        ax = axs[col]
        ax.imshow(lr_up_np, cmap=cmap, vmin=vmin, vmax=vmax)  # Default interpolation display smooth version
        ax.set_title("LR_up (bicubic)")
        ax.axis("off")
        col += 1

    # 2) SR
    ax = axs[col]
    ax.imshow(sr_np, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("SR (model)")
    ax.axis("off")
    col += 1

    # 3) HR
    ax = axs[col]
    im2 = ax.imshow(hr_np, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("HR (ground truth)")
    ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    # we use HR range for all color bars
    cbar = fig.colorbar(im2, ax=axs, fraction=0.025, pad=0.02)
    cbar.set_label("vorticity (standardized units)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------
# Main
# ------------------------------
def main(args):
    
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    # read ckpt
    cfg, state_dict, ckpt = load_checkpoint(args.checkpoint, maploc=device)
    data_root = args.data_root if args.data_root is not None else cfg["data_root"]
    scale = int(cfg.get("scale", 4))

    model = build_model_from_cfg(cfg, device)
    model.load_state_dict(state_dict)

    ds_cfg = KMFlow2D_DatasetConfig(
        root_dir=data_root,
        stat_dir=cfg.get("stat_dir", None),
        size=int(cfg.get("size", 512)),
        list_re=[float(x) for x in cfg.get("list_re", [])],
        list_fn=[int(x) for x in cfg.get("list_fn", [])],
        using_vorticity=bool(cfg.get("using_vorticity", True)),
        using_velocity=bool(cfg.get("using_velocity", False)),
        fft=bool(cfg.get("fft", False)),
        device="cuda",
    )
    base_ds = KMFlow2D_Dataset(ds_cfg)
    if len(base_ds) == 0:
        raise RuntimeError(f"No samples found under {data_root}")

    group_len = len(base_ds.dataset[0])
    groups = len(base_ds.dataset)
    N = len(base_ds)  # = group_len * groups

    sel_idx = choose_indices(N, num=args.num_samples, seed=args.seed, mode=args.pick_mode)

    # 推理与可视化
    save_dir = Path(args.save_dir)
    for rank, i in enumerate(sel_idx):
        dataset_idx = i // group_len
        sample_idx = i % group_len
        file_path = base_ds.dataset[dataset_idx][sample_idx]

        sample = base_ds[i]
        hr = sample["data"].to(device)  # [1,H,W], standardized

        # native LR + CNN input bicubic upsampling
        lr_native, lr_up = make_lr(hr, scale)

        with torch.no_grad():
            sr = model(lr_up.unsqueeze(0))  # [1,1,H,W]
        sr = sr.squeeze(0)  # [1,H,W]

        p_bi = psnr(lr_up, hr)
        p_sr = psnr(sr, hr)
        print(f"[{rank}] PSNR(bicubic,HR)={p_bi:.2f} dB | PSNR(SR,HR)={p_sr:.2f} dB | {Path(file_path).name}")

        stem = Path(file_path).with_suffix("").name
        out_file = save_dir / f"{rank:03d}_{stem}_nativeLR_SR_HR.png"
        title = f"{stem} | scale x{scale}"
        viz_panels(
            lr_native=lr_native,
            sr=sr,
            hr=hr,
            out_path=out_file,
            title=title,
            cmap="twilight",
            include_bicubic=args.include_bicubic,
            lr_up=lr_up if args.include_bicubic else None,
        )

    print(f"Done. Saved {len(sel_idx)} figure(s) to: {save_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualize SR results with native LR (no upsampling).")
    p.add_argument("--checkpoint", type=str, required=True, help="path to .pth saved by training")
    p.add_argument("--data_root", type=str, default=None, help="override data root (optional)")
    p.add_argument("--save_dir", type=str, default="outputs/viz_native")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pick_mode", type=str, default="random", choices=["random", "first"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--cmap", type=str, default="turbo")
    p.add_argument("--include_bicubic", action="store_true", help="also show LR_up (bicubic) as a fourth panel")
    args = p.parse_args()
    main(args)


"""
python -m infer.visualize_sr \
  --checkpoint checkpoints/baseline_sr.pth \
  --data_root /vast/projects/jgu32/lab/cai03/5200_dataset \
  --save_dir outputs/sr_vis_twilight \
  --num_samples 8
"""