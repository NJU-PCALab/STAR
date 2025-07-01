# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# # --------------------------------------------------------
# # References:
# # PixArt: https://github.com/PixArt-alpha/PixArt-alpha
# # Latte:  https://github.com/Vchitect/Latte
# # DiT:    https://github.com/facebookresearch/DiT/tree/main
# # GLIDE:  https://github.com/openai/glide-text2im
# # MAE:    https://github.com/facebookresearch/mae/blob/main/models_mae.py
# # --------------------------------------------------------

# import math
# from typing import KeysView

# import numpy as np
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint
# import xformers.ops
# from einops import rearrange
# from timm.models.vision_transformer import Mlp

# from opensora.acceleration.communications import all_to_all, split_forward_gather_backward
# from opensora.acceleration.parallel_states import get_sequence_parallel_group
# import ipdb
# import cv2
# import os


# approx_gelu = lambda: nn.GELU(approximate="tanh")


# class LlamaRMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         LlamaRMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         #ipdb.set_trace()
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         #ipdb.set_trace()
#         return self.weight * hidden_states.to(input_dtype)


# def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
#     if use_kernel:
#         try:
#             from apex.normalization import FusedLayerNorm

#             return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
#         except ImportError:
#             raise RuntimeError("FusedLayerNorm not available. Please install apex.")
#     else:
#         return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


# def modulate(norm_func, x, shift, scale):
#     # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
#     dtype = x.dtype
#     x = norm_func(x.to(torch.float32)).to(dtype)
#     x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
#     x = x.to(dtype)
#     return x


# def t2i_modulate(x, shift, scale):
#     return x * (1 + scale) + shift

# def get_attn_mask(query, key, idx):
#     scale = 1.0 / query.shape[-1] ** 0.5
#     query = query * scale
#     query = query.transpose(0, 1)
#     key = key.transpose(0, 1)
#     attn = query @ key.transpose(-2, -1)
#     attn = attn.softmax(-1) # H S L
#     #attn = F.dropout(attn, p=0.0)
#     # attn[attn>0.5]=1
#     # attn[attn<=0.5]=0
#     # attn[attn==1]=255
#     attn = attn * 255
#     H, S, L = attn.shape
#     hight = 64
#     width = 64
#     for h in range(H):
#         for l in range(L):
#             map = attn[h, :, l]
#             map = rearrange(map, '(H W) -> H W', H=hight, W=width)
#             map = F.interpolate(map.unsqueeze(0).unsqueeze(0), size=[256,256],mode='nearest')
#             np_array = map.squeeze(0).squeeze(0).detach().cpu().numpy()
#             image = cv2.imwrite(os.path.join("/mnt/bn/yh-volume0/code/debug/code/OpenSora/outputs/vis", 'map'+str(idx)+'_head'+str(h)+'_word'+str(l)+'.jpg'), np_array)
#             print("成功保存图像！")
    


# # ===============================================
# # General-purpose Layers
# # ===============================================


# class PatchEmbed3D(nn.Module):
#     """Video to Patch Embedding.

#     Args:
#         patch_size (int): Patch token size. Default: (2,4,4).
#         in_chans (int): Number of input video channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """

#     def __init__(
#         self,
#         patch_size=(2, 4, 4),
#         in_chans=3,
#         embed_dim=96,
#         norm_layer=None,
#         flatten=True,
#     ):
#         super().__init__()
#         self.patch_size = patch_size
#         self.flatten = flatten

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None

#     def forward(self, x):
#         """Forward function."""
#         # padding
#         _, _, D, H, W = x.size()
#         if W % self.patch_size[2] != 0:
#             x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
#         if H % self.patch_size[1] != 0:
#             x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
#         if D % self.patch_size[0] != 0:
#             x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

#         x = self.proj(x)  # (B C T H W)
#         if self.norm is not None:
#             D, Wh, Ww = x.size(2), x.size(3), x.size(4)
#             x = x.flatten(2).transpose(1, 2)
#             x = self.norm(x)
#             x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
#         return x


# class Attention(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = False,
#         qk_norm: bool = False,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#         norm_layer: nn.Module = nn.LayerNorm,
#         enable_flashattn: bool = False,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, "dim should be divisible by num_heads"
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim**-0.5
#         self.enable_flashattn = enable_flashattn

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape
#         qkv = self.qkv(x)
#         qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
#         if self.enable_flashattn:  # here
#             qkv_permute_shape = (2, 0, 1, 3, 4)
#         else:
#             qkv_permute_shape = (2, 0, 3, 1, 4)
#         qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)
#         if self.enable_flashattn:
#             from flash_attn import flash_attn_func

#             x = flash_attn_func(
#                 q,
#                 k,
#                 v,
#                 dropout_p=self.attn_drop.p if self.training else 0.0,
#                 softmax_scale=self.scale,
#             )
#         else:
#             dtype = q.dtype
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)  # translate attn to float32
#             attn = attn.to(torch.float32)
#             attn = attn.softmax(dim=-1)
#             attn = attn.to(dtype)  # cast back attn to original dtype
#             attn = self.attn_drop(attn)
#             x = attn @ v

#         x_output_shape = (B, N, C)
#         if not self.enable_flashattn:
#             x = x.transpose(1, 2)
#         x = x.reshape(x_output_shape)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class Attention_QKNorm_RoPE(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = False,
#         qk_norm: bool = False,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#         norm_layer: nn.Module = LlamaRMSNorm,
#         enable_flashattn: bool = False,
#         rope=None,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, "dim should be divisible by num_heads"
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim**-0.5
#         self.enable_flashattn = enable_flashattn

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.rotary_emb = rope

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape
#         qkv = self.qkv(x)
#         qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
#         if self.enable_flashattn:
#             qkv_permute_shape = (2, 0, 1, 3, 4)
#         else:
#             qkv_permute_shape = (2, 0, 3, 1, 4)
#         qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
#         q, k, v = qkv.unbind(0)
#         #ipdb.set_trace()
#         if self.rotary_emb is not None:
#             q = self.rotary_emb(q)
#             k = self.rotary_emb(k)
#         #ipdb.set_trace()
#         q, k = self.q_norm(q), self.k_norm(k)
#         #ipdb.set_trace()
#         if self.enable_flashattn:
#             from flash_attn import flash_attn_func

#             x = flash_attn_func(
#                 q,
#                 k,
#                 v,
#                 dropout_p=self.attn_drop.p if self.training else 0.0,
#                 softmax_scale=self.scale,
#             )
#         else:
#             dtype = q.dtype
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)  # translate attn to float32
#             attn = attn.to(torch.float32)
#             attn = attn.softmax(dim=-1)
#             attn = attn.to(dtype)  # cast back attn to original dtype
#             attn = self.attn_drop(attn)
#             x = attn @ v

#         x_output_shape = (B, N, C)
#         if not self.enable_flashattn:
#             x = x.transpose(1, 2)
#         x = x.reshape(x_output_shape)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class MaskedSelfAttention(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = False,
#         qk_norm: bool = False,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#         norm_layer: nn.Module = LlamaRMSNorm,
#         enable_flashattn: bool = False,
#         rope=None,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, "dim should be divisible by num_heads"
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim**-0.5
#         self.enable_flashattn = enable_flashattn

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.rotary_emb = rope

#     def forward(self, x, mask):
#         B, N, C = x.shape
#         qkv = self.qkv(x)
#         qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
#         qkv_permute_shape = (2, 0, 3, 1, 4)
#         qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
#         q, k, v = qkv.unbind(0)  # B H N C
#         #ipdb.set_trace()
#         if self.rotary_emb is not None:
#             q = self.rotary_emb(q)
#             k = self.rotary_emb(k)
#         #ipdb.set_trace()
#         q, k = self.q_norm(q), self.k_norm(k)
#         #ipdb.set_trace()
        
#         mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, 1, 1).to(torch.float32)  # B H 1 N
#         dtype = q.dtype
#         q = q * self.scale
#         attn = q @ k.transpose(-2, -1)  # translate attn to float32
#         attn = attn.to(torch.float32)
#         attn = attn.masked_fill(mask == 0, -1e9)
#         attn = attn.softmax(dim=-1)
#         attn = attn.to(dtype)  # cast back attn to original dtype
#         attn = self.attn_drop(attn)
#         x = attn @ v

#         x_output_shape = (B, N, C)
#         x = x.transpose(1, 2)
#         x = x.reshape(x_output_shape)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class SeqParallelAttention(Attention):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = False,
#         qk_norm: bool = False,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#         norm_layer: nn.Module = nn.LayerNorm,
#         enable_flashattn: bool = False,
#     ) -> None:
#         super().__init__(
#             dim=dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_norm=qk_norm,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             norm_layer=norm_layer,
#             enable_flashattn=enable_flashattn,
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape  # for sequence parallel here, the N is a local sequence length
#         qkv = self.qkv(x)
#         qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

#         qkv = qkv.view(qkv_shape)

#         sp_group = get_sequence_parallel_group()

#         # apply all_to_all to gather sequence and split attention heads
#         # [B, SUB_N, 3, NUM_HEAD, HEAD_DIM] -> [B, N, 3, NUM_HEAD_PER_DEVICE, HEAD_DIM]
#         qkv = all_to_all(qkv, sp_group, scatter_dim=3, gather_dim=1)

#         if self.enable_flashattn:
#             qkv_permute_shape = (2, 0, 1, 3, 4)  # [3, B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]
#         else:
#             qkv_permute_shape = (2, 0, 3, 1, 4)  # [3, B, NUM_HEAD_PER_DEVICE, N, HEAD_DIM]
#         qkv = qkv.permute(qkv_permute_shape)

#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)
#         if self.enable_flashattn:
#             from flash_attn import flash_attn_func

#             x = flash_attn_func(
#                 q,
#                 k,
#                 v,
#                 dropout_p=self.attn_drop.p if self.training else 0.0,
#                 softmax_scale=self.scale,
#             )
#         else:
#             dtype = q.dtype
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)  # translate attn to float32
#             attn = attn.to(torch.float32)
#             attn = attn.softmax(dim=-1)
#             attn = attn.to(dtype)  # cast back attn to original dtype
#             attn = self.attn_drop(attn)
#             x = attn @ v

#         if not self.enable_flashattn:
#             x = x.transpose(1, 2)

#         # apply all to all to gather back attention heads and split sequence
#         # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]  -> [B, SUB_N, NUM_HEAD, HEAD_DIM]
#         x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

#         # reshape outputs back to [B, N, C]
#         x_output_shape = (B, N, C)
#         x = x.reshape(x_output_shape)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class MultiHeadCrossAttention(nn.Module):
#     def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
#         super(MultiHeadCrossAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         self.q_linear = nn.Linear(d_model, d_model)
#         self.kv_linear = nn.Linear(d_model, d_model * 2)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(d_model, d_model)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, cond, mask=None, i=0, t=0):
#         # query/value: img tokens; key: condition; mask: if padding tokens
#         B, N, C = x.shape

#         q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
#         kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
#         k, v = kv.unbind(2)

#         #ipdb.set_trace()

#         attn_bias = None
#         if mask is not None:
#             attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
#         #  vis
#         print(t[0].item())
#         if i == 27 and t[0].item() >= 480.0 and t[0].item() <= 500.0:
#             q1 = q[:, :N, :, :].squeeze(0)  # S H C 
#             q2 = q[:, N:, :, :].squeeze(0)  # S H C
#             k1 = k[:, :mask[0], :, :].squeeze(0)  # L H C
#             k2 = k[:, mask[0]:, :, :].squeeze(0)  # L H C
#             get_attn_mask(q1, k1, 1)
#             get_attn_mask(q2, k2, 2)
#         #  vis
#         x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
#         #ipdb.set_trace()

#         x = x.view(B, -1, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class MaskedMultiHeadCrossAttention(nn.Module):
#     def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
#         super(MaskedMultiHeadCrossAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         self.q_linear = nn.Linear(d_model, d_model)
#         self.kv_linear = nn.Linear(d_model, d_model * 2)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(d_model, d_model)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, cond, mask=None):
#         # query/value: img tokens; key: condition; mask: if padding tokens
#         B, S, C = x.shape
#         L = cond.shape[1]

#         q = self.q_linear(x).view(B, S, self.num_heads, self.head_dim)
#         kv = self.kv_linear(cond).view(B, L, 2, self.num_heads, self.head_dim)
#         k, v = kv.unbind(2)

#         #ipdb.set_trace()

#         attn_bias = None
#         if mask is not None:
#             attn_bias = mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, S, 1).to(q.dtype) # B H S L
#             exp = -1e9
#             attn_bias[attn_bias==0] = exp
#             attn_bias[attn_bias==1] = 0
#         x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
#         #ipdb.set_trace()

#         x = x.view(B, -1, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class LongShortMultiHeadCrossAttention(nn.Module):
#     def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
#         super(LongShortMultiHeadCrossAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         self.q_linear = nn.Linear(d_model, d_model)
#         self.kv_linear = nn.Linear(d_model, d_model * 2)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(d_model, d_model)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, cond, mask=None):
#         # query/value: img tokens; key: condition; mask: if padding tokens
#         B, N, C = x.shape
#         M = cond.shape[1]

#         q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
#         kv = self.kv_linear(cond).view(B, M, 2, self.num_heads, self.head_dim)
#         k, v = kv.unbind(2)

#         attn_bias = None
#         if mask is not None:
#             attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
#         x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

#         x = x.view(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class MultiHeadV2TCrossAttention(nn.Module):
#     def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
#         super(MultiHeadV2TCrossAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         self.q_linear = nn.Linear(d_model, d_model)
#         self.kv_linear = nn.Linear(d_model, d_model * 2)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(d_model, d_model)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, cond, mask=None):
#         # query/value: condition; key: img tokens; mask: if padding tokens
#         B, N, C = cond.shape

#         q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
#         kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
#         k, v = kv.unbind(2)

#         #ipdb.set_trace()

#         attn_bias = None
#         if mask is not None:
#             attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(mask, [N] * B)
#         x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
#         #ipdb.set_trace()

#         x = x.view(B, -1, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class MultiHeadT2VCrossAttention(nn.Module):
#     def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
#         super(MultiHeadT2VCrossAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         self.q_linear = nn.Linear(d_model, d_model)
#         self.kv_linear = nn.Linear(d_model, d_model * 2)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(d_model, d_model)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, cond, mask=None):
#         # query/value: img tokens; key: condition; mask: if padding tokens
#         #ipdb.set_trace()
#         B, T, N, C = x.shape
#         x = rearrange(x, 'B T N C -> (B T) N C')

#         q = self.q_linear(x)
#         q = rearrange(q, '(B T) N C -> B T N C', T=T)
#         q = q.view(1, -1, self.num_heads, self.head_dim)  # 1（B T N) H C
#         kv = self.kv_linear(cond)
#         kv = kv.view(1, -1, 2, self.num_heads, self.head_dim)  # 1 N 2 H C
#         k, v = kv.unbind(2)
#         #ipdb.set_trace()

#         attn_bias = None
#         if mask is not None:
#             #mask = [m for m in mask for _ in range(T)]
#             attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * (B*T), mask)
#         x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
#         #ipdb.set_trace()

#         x = x.view(B, T, N, C)
#         x = rearrange(x, 'B T N C -> (B T) N C')
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         x = rearrange(x, '(B T) N C -> B T N C', T=T)

#         return x


# class FormerMultiHeadV2TCrossAttention(nn.Module):
#     def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
#         super(FormerMultiHeadV2TCrossAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         self.q_linear = nn.Linear(d_model, d_model)
#         self.kv_linear = nn.Linear(d_model, d_model * 2)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(d_model, d_model)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, cond, mask=None):
#         # x: text tokens; cond: img tokens; mask: if padding tokens
#         #ipdb.set_trace()

#         _, N, C = x.shape  # 1 N C
#         B, T, _, _ = cond.shape
#         cond = rearrange(cond, 'B T N C -> (B T) N C')

#         q = self.q_linear(x)
#         q = q.view(1, -1, self.num_heads, self.head_dim)  # 1 N H C

#         kv = self.kv_linear(cond)
#         kv = rearrange(kv, '(B T) N C -> B T N C', B=B)
#         M = kv.shape[2]  # M = H * W
#         former_frame_index = torch.arange(T) - 1
#         former_frame_index[0] = 0
#         #ipdb.set_trace()
#         former_kv = kv[:, former_frame_index]
#         former_kv = former_kv.view(1, -1, 2, self.num_heads, self.head_dim)  # 1（B T N) 2 H C
#         former_k, former_v = former_kv.unbind(2)

#         #ipdb.set_trace()

#         attn_bias = None
#         if mask is not None:
#             #mask = [m for m in mask for _ in range(T)]
#             attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(mask, [M] * (B*T))
#         x = xformers.ops.memory_efficient_attention(q, former_k, former_v, p=self.attn_drop.p, attn_bias=attn_bias)
#         #ipdb.set_trace()

#         x = x.view(1, -1, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

# class LatterMultiHeadV2TCrossAttention(nn.Module):
#     def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
#         super(LatterMultiHeadV2TCrossAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads

#         self.q_linear = nn.Linear(d_model, d_model)
#         self.kv_linear = nn.Linear(d_model, d_model * 2)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(d_model, d_model)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, cond, mask=None):
#         # x: text tokens; cond: img tokens; mask: if padding tokens
#         #ipdb.set_trace()

#         _, N, C = x.shape  # 1 N C
#         B, T, _, _ = cond.shape
#         cond = rearrange(cond, 'B T N C -> (B T) N C')

#         q = self.q_linear(x)
#         q = q.view(1, -1, self.num_heads, self.head_dim)  # 1 N H C

#         kv = self.kv_linear(cond)
#         kv = rearrange(kv, '(B T) N C -> B T N C', T=T)
#         M = kv.shape[2]  # M = H * W
#         latter_frame_index = torch.arange(T) + 1
#         latter_frame_index[-1] = T - 1
#         #ipdb.set_trace()

#         latter_kv = kv[:, latter_frame_index]
#         latter_kv = latter_kv.view(1, -1, 2, self.num_heads, self.head_dim)  # 1（B T N) 2 H C
#         latter_k, latter_v = latter_kv.unbind(2)

#         #ipdb.set_trace()

#         attn_bias = None
#         if mask is not None:
#             # mask = [m for m in mask for _ in range(T)]
#             attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(mask, [M] * (B*T))
#         x = xformers.ops.memory_efficient_attention(q, latter_k, latter_v, p=self.attn_drop.p, attn_bias=attn_bias)
#         #ipdb.set_trace()

#         x = x.view(1, -1, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class SeqParallelMultiHeadCrossAttention(MultiHeadCrossAttention):
#     def __init__(
#         self,
#         d_model,
#         num_heads,
#         attn_drop=0.0,
#         proj_drop=0.0,
#     ):
#         super().__init__(d_model=d_model, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

#     def forward(self, x, cond, mask=None):
#         # query/value: img tokens; key: condition; mask: if padding tokens
#         sp_group = get_sequence_parallel_group()
#         sp_size = dist.get_world_size(sp_group)
#         B, SUB_N, C = x.shape
#         N = SUB_N * sp_size

#         # shape:
#         # q, k, v: [B, SUB_N, NUM_HEADS, HEAD_DIM]
#         q = self.q_linear(x).view(B, -1, self.num_heads, self.head_dim)
#         kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
#         k, v = kv.unbind(2)

#         # apply all_to_all to gather sequence and split attention heads
#         q = all_to_all(q, sp_group, scatter_dim=2, gather_dim=1)

#         k = split_forward_gather_backward(k, get_sequence_parallel_group(), dim=2, grad_scale="down")
#         v = split_forward_gather_backward(v, get_sequence_parallel_group(), dim=2, grad_scale="down")

#         q = q.view(1, -1, self.num_heads // sp_size, self.head_dim)
#         k = k.view(1, -1, self.num_heads // sp_size, self.head_dim)
#         v = v.view(1, -1, self.num_heads // sp_size, self.head_dim)

#         # compute attention
#         attn_bias = None
#         if mask is not None:
#             attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
#         x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        
#         # apply all to all to gather back attention heads and scatter sequence
#         x = x.view(B, -1, self.num_heads // sp_size, self.head_dim)
#         x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

#         # apply output projection
#         x = x.view(B, -1, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class FinalLayer(nn.Module):
#     """
#     The final layer of DiT.
#     """

#     def __init__(self, hidden_size, num_patch, out_channels):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
#         self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
#         x = modulate(self.norm_final, x, shift, scale)
#         x = self.linear(x)
#         return x


# class T2IFinalLayer(nn.Module):
#     """
#     The final layer of PixArt.
#     """

#     def __init__(self, hidden_size, num_patch, out_channels):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
#         self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
#         self.out_channels = out_channels

#     def forward(self, x, t):
#         shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
#         x = t2i_modulate(self.norm_final(x), shift, scale)
#         x = self.linear(x)
#         return x


# # ===============================================
# # Embedding Layers for Timesteps and Class Labels
# # ===============================================


# class TimestepEmbedder(nn.Module):
#     """
#     Embeds scalar timesteps into vector representations.
#     """

#     def __init__(self, hidden_size, frequency_embedding_size=256):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(frequency_embedding_size, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size, bias=True),
#         )
#         self.frequency_embedding_size = frequency_embedding_size

#     @staticmethod
#     def timestep_embedding(t, dim, max_period=10000):
#         """
#         Create sinusoidal timestep embeddings.
#         :param t: a 1-D Tensor of N indices, one per batch element.
#                           These may be fractional.
#         :param dim: the dimension of the output.
#         :param max_period: controls the minimum frequency of the embeddings.
#         :return: an (N, D) Tensor of positional embeddings.
#         """
#         # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
#         half = dim // 2
#         freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
#         freqs = freqs.to(device=t.device)
#         args = t[:, None].float() * freqs[None]
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         if dim % 2:
#             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#         return embedding

#     def forward(self, t, dtype):
#         t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
#         if t_freq.dtype != dtype:
#             t_freq = t_freq.to(dtype)
#         t_emb = self.mlp(t_freq)
#         return t_emb


# class LabelEmbedder(nn.Module):
#     """
#     Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
#     """

#     def __init__(self, num_classes, hidden_size, dropout_prob):
#         super().__init__()
#         use_cfg_embedding = dropout_prob > 0
#         self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
#         self.num_classes = num_classes
#         self.dropout_prob = dropout_prob

#     def token_drop(self, labels, force_drop_ids=None):
#         """
#         Drops labels to enable classifier-free guidance.
#         """
#         if force_drop_ids is None:
#             drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
#         else:
#             drop_ids = force_drop_ids == 1
#         labels = torch.where(drop_ids, self.num_classes, labels)
#         return labels

#     def forward(self, labels, train, force_drop_ids=None):
#         use_dropout = self.dropout_prob > 0
#         if (train and use_dropout) or (force_drop_ids is not None):
#             labels = self.token_drop(labels, force_drop_ids)
#         return self.embedding_table(labels)


# class SizeEmbedder(TimestepEmbedder):
#     """
#     Embeds scalar timesteps into vector representations.
#     """

#     def __init__(self, hidden_size, frequency_embedding_size=256):
#         super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
#         self.mlp = nn.Sequential(
#             nn.Linear(frequency_embedding_size, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size, bias=True),
#         )
#         self.frequency_embedding_size = frequency_embedding_size
#         self.outdim = hidden_size

#     def forward(self, s, bs):
#         if s.ndim == 1:
#             s = s[:, None]
#         assert s.ndim == 2
#         if s.shape[0] != bs:
#             s = s.repeat(bs // s.shape[0], 1)
#             assert s.shape[0] == bs
#         b, dims = s.shape[0], s.shape[1]
#         s = rearrange(s, "b d -> (b d)")
#         s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
#         s_emb = self.mlp(s_freq)
#         s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
#         return s_emb

#     @property
#     def dtype(self):
#         return next(self.parameters()).dtype


# class CaptionEmbedder(nn.Module):
#     """
#     Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
#     """

#     def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate="tanh"), token_num=120):
#         super().__init__()
#         self.y_proj = Mlp(
#             in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0
#         )
#         self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5))
#         self.uncond_prob = uncond_prob

#     def token_drop(self, caption, force_drop_ids=None):
#         """
#         Drops labels to enable classifier-free guidance.
#         """
#         if force_drop_ids is None:
#             drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
#         else:
#             drop_ids = force_drop_ids == 1
#         caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
#         return caption

#     def forward(self, caption, train, force_drop_ids=None):
#         if train:
#             assert caption.shape[2:] == self.y_embedding.shape
#         use_dropout = self.uncond_prob > 0
#         if (train and use_dropout) or (force_drop_ids is not None):
#             caption = self.token_drop(caption, force_drop_ids)
#         caption = self.y_proj(caption)
#         return caption


# # ===============================================
# # Sine/Cosine Positional Embedding Functions
# # ===============================================
# # https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


# def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None):
#     """
#     grid_size: int of the grid height and width
#     return:
#     pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
#     """
#     if not isinstance(grid_size, tuple):
#         grid_size = (grid_size, grid_size)

#     grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
#     grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
#     if base_size is not None:
#         grid_h *= base_size / grid_size[0]
#         grid_w *= base_size / grid_size[1]
#     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
#     grid = np.stack(grid, axis=0)

#     grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
#     pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
#     if cls_token and extra_tokens > 0:
#         pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
#     return pos_embed


# def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
#     assert embed_dim % 2 == 0

#     # use half of dimensions to encode grid_h
#     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
#     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

#     emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
#     return emb


# def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
#     pos = np.arange(0, length)[..., None] / scale
#     return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


# def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
#     """
#     embed_dim: output dimension for each position
#     pos: a list of positions to be encoded: size (M,)
#     out: (M, D)
#     """
#     assert embed_dim % 2 == 0
#     omega = np.arange(embed_dim // 2, dtype=np.float64)
#     omega /= embed_dim / 2.0
#     omega = 1.0 / 10000**omega  # (D/2,)

#     pos = pos.reshape(-1)  # (M,)
#     out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

#     emb_sin = np.sin(out)  # (M, D/2)
#     emb_cos = np.cos(out)  # (M, D/2)

#     emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
#     return emb
