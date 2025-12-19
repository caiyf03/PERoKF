import torch
from torch import nn
from config import CNNConfig

class CNNModel(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        in_ch = cfg.input_dim
        hid_ch = cfg.hidden_dim
        out_ch = cfg.output_dim
        depth = cfg.depth

        layers = []

        # ----------- ① Downsample: 将 512x512 -> 256x256 ------------
        layers += [
            nn.Conv2d(in_ch, hid_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hid_ch),
            nn.ReLU(True)
        ]

        # ----------- ② 中间卷积，不改变分辨率 (256x256) ---------------
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(hid_ch, hid_ch, 3, padding=1),
                nn.GroupNorm(num_groups=4, num_channels=hid_ch),
                nn.ReLU(True)
            ]

        # ----------- ③ 输出层: 先保持低分辨率 -------------------------
        layers += [
            nn.Conv2d(hid_ch, out_ch, 3, padding=1)
        ]

        self.net = nn.Sequential(*layers)

        # ----------- ④ Upsample 回 512x512 --------------------------
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # residual only if channels match
        self.use_residual = (in_ch == out_ch)

    def forward(self, x):
        out = self.net(x)             # 256x256
        out = self.upsample(out)      # 512x512 （恢复原尺寸）

        if self.use_residual:
            return x + out
        else:
            return out
