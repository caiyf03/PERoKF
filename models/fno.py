import torch
from torch import nn

from config import FNOConfig

class FourierOp2D(nn.Module):
    def __init__(self, input_dim, output_dim, modes_x, modes_y):
        super(FourierOp2D, self).__init__()
        
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1. / (input_dim * output_dim)

        self.w1 = nn.Parameter(scale * torch.randn(output_dim, input_dim, modes_x, modes_y, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(output_dim, input_dim, modes_x, modes_y, dtype=torch.cfloat))

    def transform(self, x, weight):
        # [bs, input_dim, x, y], [output_dim, input_dim, x, y] -> [bs, output_dim, x, y]
        return torch.einsum("bixy, oixy -> boxy", x, weight)

    def forward(self, x):

        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes_x,  :self.modes_y] = self.transform(x_ft[:, :, :self.modes_x,  :self.modes_y], self.w1)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = self.transform(x_ft[:, :, -self.modes_x:, :self.modes_y], self.w2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x

class FourierLayer2D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, modes_x, modes_y):
        super(FourierLayer2D, self).__init__()
        
        self.output_dim = output_dim
        self.fourier = FourierOp2D(input_dim, output_dim, modes_x, modes_y)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size = kernel_size, padding = kernel_size//2)
        self.norm = nn.GroupNorm(32, output_dim)

    # x: [bs, c, x, y]
    def forward(self, x):
        x = self.fourier(x) + self.conv(x)
        x = self.norm(x)
        return x

class FNO2D(nn.Module):
    def __init__(self, config: FNOConfig):
        super(FNO2D, self).__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(FourierLayer2D(config.hidden_dim, config.hidden_dim, config.kernel_size, config.modes_x, config.modes_y), nn.GELU()) for _ in range(config.depth)]
        )
    
    # x: [bs, c, x, y]
    def forward(self, x):
        x = self.model(x)
        return x