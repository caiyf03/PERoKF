import torch
from torch import nn

from config import DiffusionConfig
from models.unet import UNet
from models.encdec import InputEncoder, OutputDecoder

from physics.pde import compute_time_derivative, divergence_loss_l2, derivative_loss_l2

# x-prediction diffusion model
class DiffusionModel(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super(DiffusionModel, self).__init__()

        assert cfg.model.type.upper() == "UNET", f"DiffusionModel only supports UNet backbone for now, but got {cfg.model.type.upper()}."

        self.cfg = cfg
        self.set_diffusion_parameters()

        self.use_physics_feature = cfg.physics.enable_feature

        self.enable_vorti_loss = cfg.physics.enable_vorti_loss
        self.enable_deriv_loss = cfg.physics.enable_deriv_loss

        self.vorti_loss_weight = cfg.physics.vorti_loss_weight
        self.deriv_loss_weight = cfg.physics.deriv_loss_weight

        self.data_encoder = InputEncoder(1, cfg.encoder.encoder_latent)
        self.guidance_encoder = InputEncoder(1, cfg.encoder.encoder_latent)

        if self.use_physics_feature:
            self.physics_encoder = InputEncoder(1, cfg.encoder.encoder_latent)
            self.integrate = InputEncoder(cfg.encoder.encoder_latent * 3, cfg.decoder.hidden_dim)
        else:
            self.integrate = InputEncoder(cfg.encoder.encoder_latent * 2, cfg.decoder.hidden_dim)

        self.backbone = UNet(cfg.model.unet)

        self.decoder_vorti = OutputDecoder(cfg.decoder.hidden_dim, cfg.decoder.output_dim)

    def set_diffusion_parameters(self):
        self.device = torch.device(self.cfg.training.device)
        self.betas = torch.linspace(self.cfg.model.beta_start, self.cfg.model.beta_end, self.cfg.model.timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alpha_hats_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alpha_hats[:-1]], dim=0)

        self.sqrt_alpha_hats = torch.sqrt(self.alpha_hats).to(self.device)
        self.sqrt_one_minus_alpha_hats = torch.sqrt(1 - self.alpha_hats).to(self.device)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alpha_hats_prev) / (1 - self.alpha_hats)).to(self.device)
        self.posterior_mean_coef2 = ((1 - self.alpha_hats_prev) * torch.sqrt(self.alphas) / (1 - self.alpha_hats)).to(self.device)
        self.posterior_variance = (self.betas * (1 - self.alpha_hats_prev) / (1 - self.alpha_hats)).to(self.device)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20)).to(self.device)

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

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t).view(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_hat = self.sqrt_alpha_hats[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hats[t].view(-1, 1, 1, 1)

        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise
    
    @torch.no_grad()
    def p_sample(self, x_t, t, x_lr, Re, fn):
        x_0_pred = self.forward(x_t, x_lr, t, Re, fn)

        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_0_pred +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_log_variance = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)

        noise = torch.randn_like(x_t) if (t > 0).any() else torch.zeros_like(x_t)

        x_prev = posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise
        return x_prev, x_0_pred

    @torch.no_grad()
    def sample(self, x_lr, Re, fn):
        batch_size = x_lr.shape[0]
        x_t = torch.randn_like(x_lr).to(self.device)

        for t in reversed(range(self.cfg.model.timesteps)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
            x_t, _ = self.p_sample(x_t, t_batch, x_lr, Re, fn)

        return x_t

    @torch.no_grad()
    def one_step_sample(self, x_lr, Re, fn):
        batch_size = x_lr.shape[0]
        t = torch.randint(0, self.cfg.model.timesteps, (batch_size,), device=self.device)
        x_t = torch.randn_like(x_lr).to(self.device)

        x_prev, x_0_pred = self.p_sample(x_t, t, x_lr, Re, fn)

        return x_prev, x_0_pred

    def compute_loss(self, x_hr, x_lr, t, Re, fn): # x_0: low-res input
        # x-prediction only
        noise = torch.randn_like(x_hr)
        x_t = self.q_sample(x_hr, t, noise=noise)
        x_0_pred = self.forward(x_t, x_lr, t, Re, fn)
        loss = nn.MSELoss()(x_0_pred, x_hr)
        stats = {}

        if self.enable_deriv_loss:
            derivative_loss = derivative_loss_l2(x_0_pred, x_hr, Re, fn)
            loss += derivative_loss * self.deriv_loss_weight
            stats['derivative_loss_raw'] = derivative_loss.item()

        total_loss = loss
        stats['total_loss'] = total_loss.item()

        return loss, stats

    def record_loss(self, x_hr, x_lr, t, Re, fn):
        noise = torch.randn_like(x_hr)
        x_t = self.q_sample(x_hr, t, noise=noise)
        x_0_pred = self.forward(x_t, x_lr, t, Re, fn)

        return {
            'vorticity_loss_raw': nn.MSELoss()(x_0_pred, x_hr).item(),
            'derivative_loss_raw': derivative_loss_l2(x_0_pred, x_hr, Re, fn).item()
        }
    
    def forward(self, x_t, x_lr, t, Re, fn):
        data_latent = self.data_encoder(self.normalize_w(x_t))
        guidance_latent = self.guidance_encoder(self.normalize_w(x_lr))

        if self.use_physics_feature:
            # TODO: which is better, x_t or x_lr?
            # physics_map = compute_time_derivative(x_t, Re, fn)
            physics_map = compute_time_derivative(x_lr, Re, fn)
            phys_latent = self.physics_encoder(self.normalize_dw(physics_map))
            latent = torch.cat([data_latent, guidance_latent, phys_latent], dim=1)
            latent = self.integrate(latent)
        else:
            latent = torch.cat([data_latent, guidance_latent], dim=1)
            latent = self.integrate(latent)
        
        out = self.backbone(latent, t)

        pred = self.denormalize_w(self.decoder_vorti(out))
        
        return pred
