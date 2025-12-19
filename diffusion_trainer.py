from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch import nn, optim

import os
from tqdm import tqdm

import numpy as np

from config import DiffusionConfig
from diffusion_model import DiffusionModel
from sr_model import SRModel
from kmflow.km_dataset import KMFlow2D_Dataset

from torch.utils.tensorboard import SummaryWriter

class DiffusionTrainer:
    def __init__(self, cfg: DiffusionConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        self.epochs = cfg.training.epochs

        self.model = DiffusionModel(cfg).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.training.lr)

        self.fp16 = cfg.training.fp16
        if self.fp16:
            print("[DiffusionTrainer INFO] Using FP16 Mixed Precision Training.")
            self.scaler = GradScaler()

        self.log_dir = cfg.training.log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.log_every = cfg.training.log_every
        self.val_every = cfg.training.val_every
        self.save_every = cfg.training.save_every
        self.save_dir = cfg.training.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.name = "diffusion_" + cfg.model.type.lower() + f"_{int(cfg.physics.enable_feature)}{int(cfg.physics.enable_deriv_loss)}"

    def set_statistics(self, w_mean, w_std, dw_mean, dw_std):
        self.model.set_statistics(w_mean, w_std, dw_mean, dw_std, self.device)

    def from_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        print(f"[DiffusionTrainer INFO] Loaded checkpoint from {checkpoint_path}")

    def train_step(self, batch):
        hr, Re, fn = batch["data"], batch["meta"]["re"].to(self.device), batch["meta"]["fn"].to(self.device)
        lr = hr[:, :, ::4, ::4]  # downsample by 4
        lr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)

        lr = lr.to(self.device)
        hr = hr.to(self.device)

        t = torch.randint(0, self.model.cfg.model.timesteps, (hr.shape[0],), device=self.device)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast(device_type=self.device.type):
                loss, stats = self.model.compute_loss(hr, lr, t, Re, fn)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, stats = self.model.compute_loss(hr, lr, t, Re, fn)
            loss.backward()
            self.optimizer.step()

        return loss.item(), stats

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        loss_sums = defaultdict(float)
        for batch in val_loader:
            hr, Re, fn = batch["data"], batch["meta"]["re"].to(self.device), batch["meta"]["fn"].to(self.device)
            lr = hr[:, :, ::4, ::4]
            lr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            t = torch.randint(0, self.model.cfg.model.timesteps, (hr.shape[0],), device=self.device)

            if self.fp16:
                with autocast(device_type=self.device.type):
                    loss, stats = self.model.record_loss(hr, lr, t, Re, fn)
            else:
                loss, stats = self.model.record_loss(hr, lr, t, Re, fn)

            for k, v in stats.items():
                loss_sums[k] += v

        num_batches = len(val_loader)
        avg_stats = {k: v / num_batches for k, v in loss_sums.items()}
        return avg_stats

    def train(self, train_loader, val_loader=None):
        steps = 0
        for epoch in range(1, self.epochs + 1):
            for batch in tqdm(train_loader):
                loss, stats = self.train_step(batch)

                if steps % self.log_every == 0:
                    print(f"Epoch [{epoch}/{self.epochs}] | Step [{steps}] | Loss: {loss:.4f}")
                    for k, v in stats.items():
                        self.writer.add_scalar(f"Train/{k}",
                                               v.item() if torch.is_tensor(v) else v,
                                               steps)

                if val_loader and steps % self.val_every == 0:
                    val_stats = self.validate(val_loader)
                    for k, v in val_stats.items():
                        self.writer.add_scalar(f"Validation/{k}",
                                               v.item() if torch.is_tensor(v) else v,
                                               steps)

                if steps % self.save_every == 0:
                    torch.save(self.model.state_dict(), f"{self.save_dir}/{self.name}_step_{steps}.pt")

                steps += 1


if __name__ == "__main__":

    from config import parse_diffusion_config_from_yaml
    from kmflow.km_dataset import KMFlow2D_Dataset
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    def train(config_path: str):
        np.random.seed(42)
        torch.manual_seed(42)
        
        cfg = parse_diffusion_config_from_yaml(config_path)

        trainer = DiffusionTrainer(cfg)
        dataset = KMFlow2D_Dataset(cfg.dataset)

        trainer.set_statistics(dataset.w_mean, dataset.w_std, dataset.dw_mean, dataset.dw_std)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

        trainer.train(train_loader, val_loader)

    train("configs/diffusion_config.yaml")

    