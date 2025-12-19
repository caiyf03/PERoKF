import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2

from config import DiffusionConfig, Config, parse_diffusion_config_from_yaml, parse_config_from_yaml
from diffusion_model import DiffusionModel
from sr_model import SRModel
from kmflow.km_dataset import KMFlow2D_Dataset


def to_colormap(x):
    """ Convert 1×H×W tensor to RGB heatmap """
    x = x.squeeze().cpu().numpy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = (x * 255).astype(np.uint8)
    x = cv2.applyColorMap(x, cv2.COLORMAP_TWILIGHT)
    return x[:, :, ::-1]   # BGR → RGB


def run_case(cfg, dataset, lr, Re, fn, device, enable_feature, enable_deriv_loss):
    """ Run a single (feature, deriv_loss) case and return the predicted colormap """
    cfg.physics.enable_feature = enable_feature
    cfg.physics.enable_deriv_loss = enable_deriv_loss

    name = cfg.model.type.lower() + f"_{int(enable_feature)}{int(enable_deriv_loss)}"
    ckpt_path = f"{cfg.training.save_dir}/{name}_step_1500.pt"

    model = SRModel(cfg)
    model.set_statistics(dataset.w_mean, dataset.w_std, dataset.dw_mean, dataset.dw_std, device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device)

    with torch.no_grad():
        pred = model(lr, Re, fn)

    return to_colormap(pred)

def run_case_diffusion(cfg, dataset, lr, Re, fn, device, enable_feature, enable_deriv_loss):
    """ Run a single (feature, deriv_loss) case and return the predicted colormap """
    cfg.physics.enable_feature = enable_feature
    cfg.physics.enable_deriv_loss = enable_deriv_loss

    name = "diffusion_" + cfg.model.type.lower() + f"_{int(enable_feature)}{int(enable_deriv_loss)}"
    ckpt_path = f"{cfg.training.save_dir}/{name}_step_14000.pt"

    model = DiffusionModel(cfg)
    model.set_statistics(dataset.w_mean, dataset.w_std, dataset.dw_mean, dataset.dw_std, device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device)

    with torch.no_grad():
        pred = model.sample(lr, Re, fn)

    return to_colormap(pred)


def visualize(cfg: Config, batch: torch.Tensor):

    # ------------------- Prepare data -------------------
    device = torch.device(cfg.training.device)
    dataset = KMFlow2D_Dataset(cfg.dataset)

    hr = batch["data"].to(device)
    Re = batch["meta"]["re"].to(device)
    fn = batch["meta"]["fn"].to(device)

    lr = hr[:, :, ::4, ::4]
    # lr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).to(device)
    lr = F.interpolate(lr, size=hr.shape[-2:], mode="nearest").to(device)

    # Ground-truth & LR
    gt_img = to_colormap(hr)
    lr_img = to_colormap(lr)

    # ------------------- Run 4 cases -------------------
    results = [gt_img, lr_img]

    configs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]

    for ef, ed in configs:
        pred_img = run_case(cfg, dataset, lr, Re, fn, device, ef, ed)
        results.append(pred_img)

    # ------------------- Concatenate with gaps -------------------
    gap = 8  # white strip width
    h = results[0].shape[0]
    gap_strip = 255 * np.ones((h, gap, 3), dtype=np.uint8)

    final_row = []
    for i, img in enumerate(results):
        final_row.append(img)
        if i < len(results) - 1:
            final_row.append(gap_strip)

    final_img = np.concatenate(final_row, axis=1)

    # ------------------- Save output -------------------
    cv2.imwrite("unet_visualization.png", final_img)
    print("Saved unet_visualization.png")


def visualize_diffusion(cfg: DiffusionConfig, batch: torch.Tensor):
    # ------------------- Prepare data -------------------
    device = torch.device(cfg.training.device)
    dataset = KMFlow2D_Dataset(cfg.dataset)

    hr = batch["data"].to(device)
    Re = batch["meta"]["re"].to(device)
    fn = batch["meta"]["fn"].to(device)

    lr = hr[:, :, ::4, ::4]
    # lr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).to(device)
    lr = F.interpolate(lr, size=hr.shape[-2:], mode="nearest").to(device)

    # Ground-truth & LR
    gt_img = to_colormap(hr)
    lr_img = to_colormap(lr)

    # ------------------- Run 4 cases -------------------
    results = [gt_img, lr_img]

    configs = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1)
    ]

    for ef, ed in configs:
        pred_img = run_case_diffusion(cfg, dataset, lr, Re, fn, device, ef, ed)
        results.append(pred_img)

    # ------------------- Concatenate with gaps -------------------
    gap = 8  # white strip width
    h = results[0].shape[0]
    gap_strip = 255 * np.ones((h, gap, 3), dtype=np.uint8)

    final_row = []
    for i, img in enumerate(results):
        final_row.append(img)
        if i < len(results) - 1:
            final_row.append(gap_strip)

    final_img = np.concatenate(final_row, axis=1)

    # ------------------- Save output -------------------
    cv2.imwrite("unet_visualization.png", final_img)
    print("Saved unet_visualization.png")
