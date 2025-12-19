from dataclasses import dataclass, field
import math
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import einops

from .attention import zero_module, SelfAttention, SelfAttention1DBlock

@dataclass
class ResNetConfig:
    input_dim: int
    output_dim: int
    enable_attn_ffn: bool = False


class ResNet(nn.Module):
    def __init__(self, config: ResNetConfig, time_emb_dim=None):
        super(ResNet, self).__init__()
        self.norm1 = nn.GroupNorm(32, config.input_dim)
        self.conv1 = nn.Conv2d(config.input_dim, config.output_dim, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(32, config.output_dim)
        self.conv2 = zero_module(nn.Conv2d(config.output_dim, config.output_dim, kernel_size=3, padding=1, bias=True))
        
        if time_emb_dim is not None and time_emb_dim > 0:
            self.time_layer = nn.Linear(time_emb_dim, config.output_dim * 2)
        else:
            self.time_layer = None

        if config.input_dim != config.output_dim:
            self.conv3 = nn.Conv2d(config.input_dim, config.output_dim, kernel_size=1, bias=True)
        else:
            self.conv3 = None

    def forward(self, x, temb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        if self.conv3 is not None:
            x = self.conv3(x)

        if self.time_layer is not None:
            ta, tb = (
                self.time_layer(F.silu(temb)).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
            )
            if h.size(0) > ta.size(0):  # HACK. repeat to match the shape.
                N = h.size(0) // ta.size(0)
                ta = einops.repeat(ta, "b c h w -> (b n) c h w", n=N)
                tb = einops.repeat(tb, "b c h w -> (b n) c h w", n=N)
            h = F.silu(self.norm2(h) * (1 + ta) + tb)
            h = self.conv2(h)
        else:
            h = F.silu(self.norm2(h))
            h = self.conv2(h)
        
        return h + x

class ResNetBlock(nn.Module):
    def __init__(
            self,
            num_residual_blocks: int,
            num_attention_layers: int,
            resnet_configs: List[ResNetConfig], 
            downsample_output=False, 
            upsample_output=False,
            time_emb_dim=None,
            cond_feat_dim=None,
        ):
        super(ResNetBlock, self).__init__()

        self.num_residual_blocks = num_residual_blocks
        self.num_attention_layers = num_attention_layers
        self.downsample_output = downsample_output
        self.upsample_output = upsample_output
        assert not (downsample_output and upsample_output), "Cannot both downsample and upsample."
        
        resnets = []
        for i in range(num_residual_blocks):
            resnets.append(ResNet(resnet_configs[i], time_emb_dim=time_emb_dim))
        self.resnets = nn.ModuleList(resnets)

        if self.num_attention_layers is not None and self.num_attention_layers > 0:
            attentions = []
            for i in range(num_residual_blocks):
                for j in range(num_attention_layers):
                    attentions.append(
                        SelfAttention(
                            hidden_dim=resnet_configs[i].output_dim,
                            cond_dim=cond_feat_dim,
                            enable_attn_ffn=resnet_configs[i].enable_attn_ffn,
                        )
                    )
            self.attentions = nn.ModuleList(attentions)
        
        if self.downsample_output:
            self.resample = nn.Conv2d(resnet_configs[-1].output_dim, resnet_configs[-1].output_dim, kernel_size=3, stride=2, padding=1, bias=True)
        elif self.upsample_output:
            self.resample = nn.Conv2d(resnet_configs[-1].output_dim, resnet_configs[-1].output_dim, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, x, skip_activations=None, return_activations=False, temb=None, conditioning=None, cond_mask=None):
        activations = []
        for i in range(self.num_residual_blocks):
            if skip_activations is not None:
                x = torch.cat([x, skip_activations.pop(0)], dim=1)
            x = self.resnets[i](x, temb)

            if self.num_attention_layers is not None and self.num_attention_layers > 0:
                for j in range(self.num_attention_layers):
                    attn_idx = i * self.num_attention_layers + j
                    x = self.attentions[attn_idx](x, conditioning=conditioning, cond_mask=cond_mask)
            activations.append(x)

        if self.downsample_output or self.upsample_output:
            if self.upsample_output:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.resample(x)
            activations.append(x)

        if return_activations:
            return x, activations
        return x
    

@dataclass
class UNetConfig:
    num_resnets_per_resolution: List[int] = field(
        default_factory=lambda: [2, 2, 2, 2], metadata={"help": "number of ResNet blocks at each resolution level"}
    )
    attention_levels: List[int] = field(
        default_factory=lambda: [2, 3], metadata={"help": "which resolutions to apply attention"}
    )
    num_attention_layers: List[int] = field(
        default_factory=lambda: [0, 0, 1, 1], metadata={"help": "number of attention layers per attention level"}
    )
    enable_temporal_embedding: bool = field(default=True)
    temporal_embedding_dim: int = field(
        default=-1, metadata={"help": "dimension of temporal embedding, if not set, we will use 4 x resnets_dim[0]"}
    )
    enable_conditioning: bool = field(default=False)
    input_condition_feature_dim: int = field(default=-1)
    condition_feature_dim: int = field(default=-1)
    condition_self_attention_layer: int = field(default=-1)
    resnets_dim: List[int] = field(
        default_factory=lambda: [128, 256, 256, 512], metadata={"help": "channel dimensions at each resolution level"}
    )
    micro_conditioning: str = field(
        default=None, metadata={"help": "LABEL:VALUE. For example, 'scale:2'"}
    )
    skip_mid_blocks: bool = field(default=False)

class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super(UNet, self).__init__()
        self.config = config
        self.temb_dim = (
            config.resnets_dim[0] * 4 
            if (config.temporal_embedding_dim <= 0 or config.temporal_embedding_dim is None)
            else config.temporal_embedding_dim
        )

        half_dim = self.temb_dim // 8
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("t_emb", emb.unsqueeze(0), persistent=False)

        if config.enable_temporal_embedding:
            self.temb_layer1 = nn.Linear(self.temb_dim // 4, self.temb_dim)
            self.temb_layer2 = nn.Linear(self.temb_dim, self.temb_dim)
        
        self.enable_conditioning = (config.input_condition_feature_dim is not None) and (config.input_condition_feature_dim > 0) and \
                                   (config.condition_feature_dim is not None) and (config.condition_feature_dim > 0) and config.enable_conditioning

        if self.enable_conditioning:
            self.cond_emb_layer = nn.Linear(config.condition_feature_dim, self.temb_dim)
            self.cond_proj = nn.Linear(config.input_condition_feature_dim, config.condition_feature_dim)
            if config.condition_self_attention_layer is not None and config.condition_self_attention_layer >= 0:
                self.cond_self_attn = nn.ModuleList(
                    [SelfAttention1DBlock(hidden_dim=config.condition_feature_dim) for _ in range(config.condition_self_attention_layer)]
                )
            else:
                self.cond_self_attn = None
            print(f"[UNet] Conditioning is enabled with input dim {config.input_condition_feature_dim} and feature dim {config.condition_feature_dim}.")
        else:
            self.cond_emb_layer = None
            print(f"[UNet] Conditioning is disabled.")


        self.conditions = None
        if config.micro_conditioning is not None:
            self.conditions = {
                c.split(":")[0]: float(c.split(":")[1])
                for c in config.micro_conditioning.split(",")
            }
            cond_layers = {}
            for condition in self.conditions:
                cond_layers[condition] = nn.ModuleList(
                    [
                        nn.Linear(self.temporal_dim // 4, self.temporal_dim),
                        zero_module(nn.Linear(self.temporal_dim, self.temporal_dim)),
                    ]
                )
            self.cond_layers = nn.ModuleDict(cond_layers)
        
        skip_channels = [config.resnets_dim[0]]
        self.num_resolutions = len(config.resnets_dim)

        self.down_blocks = nn.ModuleList()
        self.up_blocks   = nn.ModuleList()

        for i in range(self.num_resolutions):
            down_resnet_configs = []
            num_resnets = config.num_resnets_per_resolution[i]
            for j in range(num_resnets):
                down_resnet_configs.append(
                    ResNetConfig(
                        input_dim=skip_channels[-1] if j == 0 else config.resnets_dim[i],
                        output_dim=config.resnets_dim[i],
                        enable_attn_ffn=(i in config.attention_levels),
                    )
                )
                skip_channels.append(config.resnets_dim[i])
            
            if i != self.num_resolutions - 1:
                skip_channels.append(config.resnets_dim[i])
            
            num_attention_layers = (
                config.num_attention_layers[i]
                if (i in config.attention_levels) and config.num_attention_layers is not None
                else None
            )

            self.down_blocks.append(
                ResNetBlock(
                    num_residual_blocks=num_resnets,
                    num_attention_layers=num_attention_layers,
                    resnet_configs=down_resnet_configs,
                    downsample_output=(i != self.num_resolutions - 1),
                    upsample_output=False,
                    time_emb_dim=self.temb_dim if config.enable_temporal_embedding else None,
                    cond_feat_dim=self.temb_dim if self.enable_conditioning else None,
                )
            )

        if not config.skip_mid_blocks:
            self.mid_blocks = nn.ModuleList([
                ResNetBlock(
                    num_residual_blocks=1,
                    num_attention_layers=1,
                    resnet_configs=[
                        ResNetConfig(
                            input_dim=config.resnets_dim[-1],
                            output_dim=config.resnets_dim[-1],
                            enable_attn_ffn=(self.num_resolutions - 1 in config.attention_levels),
                        )
                    ],
                    downsample_output=False,
                    upsample_output=False,
                    time_emb_dim=self.temb_dim if config.enable_temporal_embedding else None,
                    cond_feat_dim=self.temb_dim if self.enable_conditioning else None,
                ),
                ResNetBlock(
                    num_residual_blocks=1,
                    num_attention_layers=0,
                    resnet_configs=[
                        ResNetConfig(
                            input_dim=config.resnets_dim[-1],
                            output_dim=config.resnets_dim[-1],
                            enable_attn_ffn=False,
                        )
                    ],
                    downsample_output=False,
                    upsample_output=False,
                    time_emb_dim=self.temb_dim if config.enable_temporal_embedding else None,
                    cond_feat_dim=self.temb_dim if self.enable_conditioning else None,
                ),
            ])

        channels = config.resnets_dim[-1]
        for i in reversed(range(self.num_resolutions)):
            up_resnet_configs = []
            num_resnets = config.num_resnets_per_resolution[i]
            for j in range(num_resnets + 1):
                up_resnet_configs.append(
                    ResNetConfig(
                        input_dim=channels + skip_channels.pop(),
                        output_dim=config.resnets_dim[i],
                        enable_attn_ffn=(i in config.attention_levels),
                    )
                )
                channels = config.resnets_dim[i]

            num_attention_layers = (
                config.num_attention_layers[i]
                if (i in config.attention_levels) and config.num_attention_layers is not None
                else None
            )

            self.up_blocks.append(
                ResNetBlock(
                    num_residual_blocks=num_resnets + 1,
                    num_attention_layers=num_attention_layers,
                    resnet_configs=up_resnet_configs,
                    downsample_output=False,
                    upsample_output=(i != 0),
                    time_emb_dim=self.temb_dim if config.enable_temporal_embedding else None,
                    cond_feat_dim=self.temb_dim if self.enable_conditioning else None,
                )
            )
                
    def create_temporal_embedding(self, times, ff_layers=None):
        temb = times.view(-1, 1) * self.t_emb
        temb = torch.cat([torch.sin(temb), torch.cos(temb)], dim=-1)
        if ff_layers is None:
            layer1, layer2 = self.temb_layer1, self.temb_layer2
        else:
            layer1, layer2 = ff_layers
        temb = layer2(F.silu(layer1(temb)))
        return temb
    
    def forward_conditioning(self, conditioning, cond_mask):
        if conditioning is None or not self.enable_conditioning:
            return None, None, None

        if self.enable_conditioning:
            cond_feat = self.cond_proj(conditioning)
        if self.cond_self_attn is not None:
            for attn_layer in self.cond_self_attn:
                cond_feat = attn_layer(cond_feat, cond_mask)

        if cond_mask is None:
            y = cond_feat.mean(dim=1)
        else:
            y = (cond_feat * cond_mask.unsqueeze(-1)).sum(dim=1) / cond_mask.sum(dim=1, keepdim=True)
        
        cond_emb = self.cond_emb_layer(y)
        return cond_emb, cond_feat, cond_mask

    def forward_downsample(self, x, temb=None, conditioning=None, cond_mask=None):
        skip_activations = [x]
        for i, block in enumerate(self.down_blocks):
            if i in self.config.attention_levels:
                x, activations = block(
                    x,
                    return_activations=True,
                    temb=temb,
                    conditioning=conditioning,
                    cond_mask=cond_mask,
                )
            else:
                x, activations = block(x, temb=temb, return_activations=True)
            skip_activations.extend(activations)
        return x, skip_activations
    
    def forward_upsample(self, x, skip_activations, temb=None, conditioning=None, cond_mask=None):
        num_resolutions = len(self.config.resnets_dim)
        for i, block in enumerate(self.up_blocks):
            ri = num_resolutions - 1 - i
            num_skip = self.config.num_resnets_per_resolution[ri] + 1
            skip_connections = skip_activations[-num_skip:]
            skip_connections.reverse()
            if ri in self.config.attention_levels:
                x = block(
                    x,
                    skip_activations=skip_connections,
                    temb=temb,
                    conditioning=conditioning,
                    cond_mask=cond_mask,
                )
            else:
                x = block(x, skip_activations=skip_connections, temb=temb)
            del skip_activations[-num_skip:]
        return x

    def forward_micro_conditioning(self, times, micros):
        temb = 0
        for key in self.conditions:
            default_val = self.conditions[key]
            micro = micros.get(key, default_val * torch.ones_like(times))
            micro = (
                (micro / default_val).clamp(max=1) * default_val
                if key == "scale"
                else micro * 1000
            )
            temb = temb + self.create_temporal_embedding(micro, ff_layers=self.cond_layers[key])
        return temb
        
    def forward(self, x, times=None, conditioning=None, cond_mask=None, micros={}):

        cond_emb, cond_feat, cond_mask = self.forward_conditioning(conditioning, cond_mask)

        if self.config.enable_temporal_embedding:
            temb = self.create_temporal_embedding(times)
            if cond_emb is not None:
                temb = temb + cond_emb
            if self.conditions is not None:
                temb = temb + self.forward_micro_conditioning(times, micros)
        else:
            temb = None

        x, activations = self.forward_downsample(x, temb, cond_feat, cond_mask)
        if not self.config.skip_mid_blocks:
            x = self.mid_blocks[0](x, temb, cond_feat, cond_mask)
            x = self.mid_blocks[1](x, temb)
        x = self.forward_upsample(x, activations, temb, cond_feat, cond_mask)
        return x