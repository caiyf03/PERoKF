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

from physics.pde import derivative_loss_l2, energy_spectrum_from_vorticity

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

    return pred

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

    return pred


def compute(cfg: Config, batch: torch.Tensor):

    # ------------------- Prepare data -------------------
    device = torch.device(cfg.training.device)
    dataset = KMFlow2D_Dataset(cfg.dataset)

    hr = batch["data"].to(device)
    Re = batch["meta"]["re"].to(device)
    fn = batch["meta"]["fn"].to(device)

    lr = hr[:, :, ::4, ::4]
    # lr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).to(device)
    lr = F.interpolate(lr, size=hr.shape[-2:], mode="nearest").to(device)

    # ------------------- Run 4 cases -------------------
    results = []

    configs = [
        (0, 0), # vision
        (0, 1), # loss only
        (1, 0), # feature only
        (1, 1)  # both
    ]

    for ef, ed in configs:
        pred = run_case(cfg, dataset, lr, Re, fn, device, ef, ed)
        results.append(pred)

    E_hr = energy_spectrum_from_vorticity(hr)

    for res in results:
        print(f"MSE: {torch.mean((res - hr) ** 2).item()}")
        print(f"Phys: {derivative_loss_l2(res, hr, Re, fn).item()}")
        E_res = energy_spectrum_from_vorticity(res)
        K = min(E_hr.shape[1], E_res.shape[1])
        start = int(0.3 * K)
        end = K
        es_loss = torch.sum(torch.abs(E_hr[:, start:end] - E_res[:, start:end]) / (E_hr[:, start:end] + 1e-8))
        print(f"ES: {es_loss.item()}")
        print("------------------------------------")


def compute_diffusion(cfg: DiffusionConfig, batch: torch.Tensor):
    # ------------------- Prepare data -------------------
    device = torch.device(cfg.training.device)
    dataset = KMFlow2D_Dataset(cfg.dataset)

    hr = batch["data"].to(device)
    Re = batch["meta"]["re"].to(device)
    fn = batch["meta"]["fn"].to(device)

    lr = hr[:, :, ::4, ::4]
    # lr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).to(device)
    lr = F.interpolate(lr, size=hr.shape[-2:], mode="nearest").to(device)

    # ------------------- Run 4 cases -------------------
    results = []

    configs = [
        (0, 0), # vision
        (0, 1), # loss only
        (1, 0), # feature only
        (1, 1)  # both
    ]

    for ef, ed in configs:
        pred_img = run_case_diffusion(cfg, dataset, lr, Re, fn, device, ef, ed)
        results.append(pred_img)

    E_hr = energy_spectrum_from_vorticity(hr)

    for res in results:
        print(f"MSE: {torch.mean((res - hr) ** 2).item()}")
        print(f"Phys: {derivative_loss_l2(res, hr, Re, fn).item()}")

        E_res = energy_spectrum_from_vorticity(res)
        K = min(E_hr.shape[1], E_res.shape[1])
        es_loss = torch.mean(torch.abs(E_hr[:,  1:K] - E_res[:, 1:K]) / (E_hr[:, 1:K] + 1e-8))
        print(f"ES: {es_loss.item()}")
        print("------------------------------------")