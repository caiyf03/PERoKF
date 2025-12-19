from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from config import Config
from models.cnn import CNNModel
from models.encdec import InputEncoder, OutputDecoder
from models.fno import FNO2D
from models.unet import UNet
from physics.pde import compute_time_derivative, divergence_loss_l2, derivative_loss_l2


class SRModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.use_physics_feature = cfg.physics.enable_feature

        self.enable_vorti_loss = cfg.physics.enable_vorti_loss
        self.enable_deriv_loss = cfg.physics.enable_deriv_loss

        self.vorti_loss_weight = cfg.physics.vorti_loss_weight
        self.deriv_loss_weight = cfg.physics.deriv_loss_weight

        # data encoder
        self.data_encoder = InputEncoder(1, cfg.encoder.encoder_latent)

        if self.use_physics_feature:
            # physics encoder (optional)
            self.physics_encoder = InputEncoder(1, cfg.encoder.encoder_latent)

            # integrate latent dim
            self.integrate = InputEncoder(cfg.encoder.encoder_latent * 2, cfg.decoder.hidden_dim)

        # backbone
        if cfg.model.type.upper() == "FNO":
            self.backbone = FNO2D(cfg.model.fno)
        elif cfg.model.type.upper() == "CNN":
            self.backbone = CNNModel(cfg.model.cnn)
        elif cfg.model.type.upper() == "UNET":
            self.backbone = UNet(cfg.model.unet)
        else:
            raise ValueError(f"Unknown model type: {cfg.model.type}")

        # decoder
        self.decoder_vorti = OutputDecoder(cfg.decoder.hidden_dim, cfg.decoder.output_dim)

    def set_statistics(self, w_mean, w_std, dw_mean, dw_std, device):
        self.w_mean  = torch.tensor(w_mean,  dtype=torch.float32).view(1, 1, 1, 1).to(device)
        self.w_std   = torch.tensor(w_std,   dtype=torch.float32).view(1, 1, 1, 1).to(device)
        self.dw_mean = torch.tensor(dw_mean, dtype=torch.float32).view(1, 1, 1, 1).to(device)
        self.dw_std  = torch.tensor(dw_std,  dtype=torch.float32).view(1, 1, 1, 1).to(device)
    
    def normalize_w(self, w):
        return (w - self.w_mean) / self.w_std
    
    def denormalize_w(self, w_norm):
        return w_norm * self.w_std + self.w_mean
    
    def normalize_dw(self, dw):
        return (dw - self.dw_mean) / self.dw_std
    
    def denormalize_dw(self, dw_norm):
        return dw_norm * self.dw_std + self.dw_mean

    def forward(self, x, Re, fn):
        # x: [bs, 1, hx, hy]
        data_latent = self.data_encoder(self.normalize_w(x))

        if self.use_physics_feature:
            physics_map = compute_time_derivative(x, Re, fn)
            phys_latent = self.physics_encoder(self.normalize_dw(physics_map))
            latent = torch.cat([data_latent, phys_latent], dim=1)
            latent = self.integrate(latent)
        else:
            latent = data_latent

        out = self.backbone(latent)

        pred = self.denormalize_w(self.decoder_vorti(out))
        
        return pred
        
    # vorticity prediction only
    def compute_loss(self, pred, target, Re, fn):
        loss = 0.0
        stats = {}

        if self.enable_vorti_loss:
            vorti_loss = nn.MSELoss()(pred, target)
            loss += vorti_loss * self.vorti_loss_weight
            stats['vorticity_loss_raw'] = vorti_loss.item()
        
        if self.enable_deriv_loss:
            derivative_loss = derivative_loss_l2(pred, target, Re, fn)
            loss += derivative_loss * self.deriv_loss_weight
            stats['derivative_loss_raw'] = derivative_loss.item()
        
        total_loss = loss
        stats['total_loss'] = total_loss.item()
        
        return loss, stats
        
    def record_loss(self, pred, target, Re, fn):
        return {
                'vorticity_loss_raw': nn.MSELoss()(pred, target).item(),
                'derivative_loss_raw': derivative_loss_l2(pred, target, Re, fn).item()
            }