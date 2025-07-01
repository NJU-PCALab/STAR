import torch
import torch.nn as nn
import xformers.ops

# spatial feature refiner
class SpatialFeatureRefiner(nn.Module):
    def __init__(self, hidden_channels):
        super(SpatialFeatureRefiner, self).__init__()

        # high-frequency branch
        self.hf_linear = nn.Linear(hidden_channels, hidden_channels * 2)

        # low-frequency branch
        self.lf_linear = nn.Linear(hidden_channels, hidden_channels * 2)

        # fusion
        self.gelu = nn.GELU()
        self.fusion_linear = nn.Linear(hidden_channels * 2, hidden_channels)

    def forward(self, hf_feature, lf_feature, x):
        
        # high-frequency branch
        hf_feature = self.hf_linear(hf_feature)
        scale_hf, shift_hf = hf_feature.chunk(2, dim=-1)
        x_hf = x * scale_hf + shift_hf

        # low-frequency branch
        lf_feature = self.lf_linear(lf_feature)
        scale_lf, shift_lf = lf_feature.chunk(2, dim=-1)
        x_lf = x * scale_lf + shift_lf

        # fusion
        x_fusion = torch.cat([x_hf, x_lf], dim=-1)
        x_fusion = self.gelu(x_fusion)
        x_fusion = self.fusion_linear(x_fusion)

        return x_fusion

    
# low-frequency temporal guider
class LFTemporalGuider(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(LFTemporalGuider, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x