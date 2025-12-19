import torch
from torch import nn

class InputEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InputEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.model(x)


class OutputDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(OutputDecoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
        )
    
    def forward(self, x):
        return self.model(x)