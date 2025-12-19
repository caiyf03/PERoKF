import math
import torch
from torch import nn

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, cond_dim=None, enable_attn_ffn=False):
        super(SelfAttention, self).__init__()

        self.norm = nn.GroupNorm(32, hidden_dim)
        self.qkv = nn.Conv2d(hidden_dim, hidden_dim * 3, kernel_size=1, bias=True)
        self.num_heads = num_heads

        if cond_dim is not None and cond_dim > 0:
            self.norm_cond = nn.LayerNorm(cond_dim)
            self.kv_cond = nn.Linear(cond_dim, hidden_dim * 2)
        else:
            self.kv_cond = None
        
        self.proj_out = zero_module(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))

        if enable_attn_ffn:
            self.ffn = nn.Sequential(
                nn.GroupNorm(32, hidden_dim),
                nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1),
                nn.GELU(),
                zero_module(nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1)),
            )
        else:
            self.ffn = None
    
    def attention(self, q, k, v, mask=None):
        bs, width, length = q.shape
        ch = width // self.num_heads

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).reshape(bs * self.num_heads, ch, length),
            (k * scale).reshape(bs * self.num_heads, ch, -1),
        )

        if mask is not None:
            mask = (
                mask.view(mask.size(0), 1, 1, mask.size(1))
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
            )
            weight = weight.masked_fill(mask == 0, float("-inf"))

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.num_heads, ch, -1))
        return a.reshape(bs, -1, length)
    
    def forward(self, x, conditioning=None, cond_mask=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3 * c, -1).chunk(3, dim=1)
        h = self.attention(q, k, v)

        if self.kv_cond is not None:
            kv_cond = self.kv_cond(self.norm_cond(conditioning))
            k_cond, v_cond = kv_cond.chunk(2, dim=1)
            h_cond = self.attention(q, k_cond, v_cond, cond_mask)
            h = h + h_cond

        h = h.reshape(b, c, *spatial)
        h = self.proj_out(h)
        x = x + h

        if self.ffn is not None:
            x = x + self.ffn(x)

        return x

class SelfAttention1D(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, enable_attn_ffn=False, enable_pos_emb=False):
        super(SelfAttention1D, self).__init__()

        self.num_heads = num_heads
        self.num_head_channels = hidden_dim // num_heads

        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        
        self.proj_out = zero_module(nn.Linear(hidden_dim, hidden_dim))

        if enable_attn_ffn:
            self.ffn = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                zero_module(nn.Linear(hidden_dim * 4, hidden_dim)),
            )
        else:
            self.ffn = None

        if enable_pos_emb:
            from rotary_embedding_torch import RotaryEmbedding
            self.pos_emb = RotaryEmbedding(self.num_head_channels)
        else:
            self.pos_emb = None
    
    def attention(self, q, k, v, mask=None):
        bs, length, width = q.shape
        ch = width // self.num_heads
        scale = 1 / math.sqrt(math.sqrt(ch))
        q = q.reshape(bs, length, self.num_heads, ch)
        k = k.reshape(bs, length, self.num_heads, ch)
        if self.pos_emb is not None:
            q = self.pos_emb.rotate_queries_or_keys(q.permute(0, 2, 1, 3)).permute(
                0, 2, 1, 3
            )
            k = self.pos_emb.rotate_queries_or_keys(k.permute(0, 2, 1, 3)).permute(
                0, 2, 1, 3
            )
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if mask is not None:
            mask = mask.view(mask.size(0), 1, 1, mask.size(1))
            weight = weight.masked_fill(mask == 0, float("-inf"))
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bhts,bshc->bthc", weight, v.reshape(bs, -1, self.num_heads, ch)
        )
        return a.reshape(bs, length, -1)
    
    def forward(self, x, mask=None):
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=-1)
        h = self.attention(q, k, v, mask)
        h = self.proj_out(h)
        x = x + h
        if self.ffn is not None:
            x = x + self.ffn(x)
        return x

class SelfAttention1DBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, enable_attn_ffn=False, enable_pos_emb=False):
        super(SelfAttention1DBlock, self).__init__()
        self.attn = SelfAttention1D(
            hidden_dim,
            num_heads=num_heads,
            enable_attn_ffn=enable_attn_ffn,
            enable_pos_emb=enable_pos_emb,
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            zero_module(nn.Linear(hidden_dim * 4, hidden_dim)),
        )
    
    def forward(self, x, mask=None):
        x = self.attn(x, mask)
        x = x + self.ffn(x)
        return x