import torch
import torch.nn as nn
from opensora.models.vsr.safmn_arch import SAFMN
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Mlp
from opensora.models.layers.blocks import (
    Attention,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_layernorm,
)


# high pass filter
def high_pass_filter(x, kernel_size=21):
    """
    对输入张量进行高通滤波，提取高频和低频部分。
    
    参数:
    x (torch.Tensor): 形状为 [B, C, T, H, W] 的输入张量，值范围在 [-1, 1]。
    kernel_size (int): 高斯核的大小。
    
    返回:
    high_freq (torch.Tensor): 高频部分，形状与 x 相同。
    low_freq (torch.Tensor): 低频部分，形状与 x 相同。
    """
    # 计算sigma值
    sigma = kernel_size / 6
    
    # 确定输入张量的设备
    device, dtype  = x.device, x.dtype
    
    # 转换维度 [B, C, T, H, W] -> [B*T, C, H, W]
    B, C, T, H, W = x.shape
    x_reshaped = x.contiguous().view(B * T, C, H, W)
    
    # 创建高斯核
    def get_gaussian_kernel(kernel_size, sigma):
        axis = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        gaussian = torch.exp(-0.5 * (axis / sigma) ** 2)
        gaussian /= gaussian.sum()
        return gaussian
    
    gaussian_1d = get_gaussian_kernel(kernel_size, sigma)
    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
    gaussian_3d = gaussian_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 将高斯核扩展到四维
    gaussian_kernel = gaussian_3d.expand(C, 1, kernel_size, kernel_size)
    
    # 使用F.conv2d进行卷积操作
    padding = kernel_size // 2
    
    # 计算低频部分
    low_freq_reshaped = F.conv2d(x_reshaped, gaussian_kernel, padding=padding, groups=C)
    
    # 计算高频部分
    high_freq_reshaped = x_reshaped - low_freq_reshaped
    
    # 转换回原始维度 [B*T, C, H, W] -> [B, C, T, H, W]
    low_freq = low_freq_reshaped.view(B, C, T, H, W)
    high_freq = high_freq_reshaped.view(B, C, T, H, W)
    
    return high_freq, low_freq


# depth-wise separable convoluiton
class DepthWiseSeparableResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthWiseSeparableResBlock, self).__init__()

        self.dwconv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias) # groups=in_channels, 
        # self.conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=bias)

        self.gelu = nn.GELU()

        self.dwconv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias) # groups=in_channels, 
        # self.conv2 = nn.Conv2d(in_channels, in_channels, 1, bias=bias)

    def forward(self, x):
        residual = x

        out = self.dwconv1(x)
        # out = self.conv1(out)
        out = self.gelu(out)

        out = self.dwconv2(out)
        # out = self.conv2(out)

        out += residual

        return out
    
# temporal transformer block
class TemporalTransformerBlock(nn.Module):
    def __init__(self):
        super(TemporalTransformerBlock, self).__init__()

        # temporal norm
        self.temporal_norm = get_layernorm(1152, eps=1e-6, affine=False, use_kernel=True)

        # temporal self-attention
        self.temporal_attn = Attention(
            dim=1152,
            num_heads=16,
            qkv_bias=True,
            enable_flashattn=True)

        # ffn
        self.temporal_ffn = Mlp(in_features=1152, hidden_features=4608, out_features=1152, act_layer=nn.GELU)

    def forward(self, x):
        residual = x

        out = self.temporal_norm(x)
        out = self.temporal_attn(out)
        out = self.temporal_ffn(out)

        out += residual

        return out



# frequency-decoupled information extractor
class FrequencyDecoupledInfoExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(FrequencyDecoupledInfoExtractor, self).__init__()

        ### spatial branch ###
        self.safmn = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=4, use_res=True)
        state_dict = torch.load('/mnt/bn/videodataset/VSR/pretrained_models/SAFMN_L_Real_LSDIR_x4-v2.pth')
        self.safmn.load_state_dict(state_dict['params_ema'], strict=True)

        # high-frequency branch
        # self.hf_convin = nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding, bias=bias)
        # self.hf_convout = nn.Conv2d(hidden_channels, in_channels, kernel_size, stride, padding, bias=bias)
        # hf_layer = []
        # for i in range(8):
        #     hf_layer.append(DepthWiseSeparableResBlock(hidden_channels, kernel_size, stride=1, padding=1, bias=bias))
        # self.hf_body = nn.Sequential(*hf_layer)
        self.safmn1 = SAFMN(dim=72, n_blocks=8, ffn_scale=2.0, upscaling_factor=1, in_dim=6, use_res=True)

        # low-frequency branch
        # self.lf_convin = nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding, bias=bias)
        # self.lf_convout = nn.Conv2d(hidden_channels, in_channels, kernel_size, stride, padding, bias=bias)
        # lf_layer = []
        # for i in range(8):
        #     lf_layer.append(DepthWiseSeparableResBlock(hidden_channels, kernel_size, stride=1, padding=1, bias=bias))
        # self.lf_body = nn.Sequential(*lf_layer)
        self.safmn2 = SAFMN(dim=72, n_blocks=8, ffn_scale=2.0, upscaling_factor=1, in_dim=6, use_res=True)

        ### temporal branch ###
        layer = []
        for i in range(3):
            layer.append(TemporalTransformerBlock())
        self.temporal_body = nn.Sequential(*layer)


    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            embed_dim=1152,
            length=16,
            scale=1.0,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed
    

    def spatial_forward(self, x):
        with torch.no_grad():
            x = rearrange(x, 'B C T H W -> (B T) C H W')
            x = F.interpolate(x, scale_factor=1/4, mode='bilinear')
            clean_image = self.safmn(x)
            clean_image = rearrange(clean_image, '(B T) C H W -> B C T H W', T=16)
            high_freq, low_freq = high_pass_filter(clean_image)
            fea_decouple = torch.cat([high_freq, low_freq], dim=1)
            fea_decouple = rearrange(fea_decouple, 'B C T H W -> (B T) C H W')

        # high-frequency branch
        # hf_out = self.hf_convin(high_freq)
        # hf_out = self.hf_body(hf_out)
        # hf_out = self.hf_convout(hf_out) + high_freq
        hf_out = self.safmn1(fea_decouple)
        hf_out = rearrange(hf_out, '(B T) C H W -> B C T H W', T=16)

        # low-frequency branch
        # lf_out = self.lf_convin(low_freq)
        # lf_out = self.lf_body(lf_out)
        # lf_out = self.lf_convout(lf_out) + low_freq
        lf_out = self.safmn2(fea_decouple)
        lf_out = rearrange(lf_out, '(B T) C H W -> B C T H W', T=16)

        return clean_image, hf_out, lf_out
    
    def temporal_forward(self, x):
        x = rearrange(x, "B (T S) C -> (B S) T C", T=16)
        tpe = self.get_temporal_pos_embed().to(x.device, x.dtype)
        x = x + tpe
        x = self.temporal_body(x)
        x = rearrange(x, "(B S) T C -> B (T S) C", S=256)
        return x
