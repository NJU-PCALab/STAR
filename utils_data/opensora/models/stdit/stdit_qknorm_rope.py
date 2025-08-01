import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.models.layers.blocks import (
    Attention,
    Attention_QKNorm_RoPE,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    SeqParallelAttention,
    SeqParallelMultiHeadCrossAttention,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_layernorm,
    t2i_modulate,
)
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint
# import ipdb
from rotary_embedding_torch import RotaryEmbedding


class STDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        rope=None,
        qk_norm=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn
        self._enable_sequence_parallelism = enable_sequence_parallelism

        if enable_sequence_parallelism:
            self.attn_cls = SeqParallelAttention
            self.mha_cls = SeqParallelMultiHeadCrossAttention
        else:  # here
            self.attn_cls = Attention_QKNorm_RoPE
            self.mha_cls = MultiHeadCrossAttention

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=enable_flashattn,
            qk_norm=qk_norm,
        )
        self.cross_attn = self.mha_cls(hidden_size, num_heads)
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        self.scale_shift_table_temporal = nn.Parameter(torch.randn(3, hidden_size) / hidden_size**0.5)  # new

        # temporal attention
        self.d_s = d_s
        self.d_t = d_t

        if self._enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            # make sure d_t is divisible by sp_size
            assert d_t % sp_size == 0
            self.d_t = d_t // sp_size

        self.norm_temp = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)  # new
        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=self.enable_flashattn,
            rope=rope,
            qk_norm=qk_norm,
        )

    def forward(self, x, y, t, t_temp, mask=None, tpe=None):
        B, N, C = x.shape

        #ipdb.set_trace()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        shift_tmp, scale_tmp, gate_tmp = (
            self.scale_shift_table_temporal[None] + t_temp.reshape(B, 3, -1)
        ).chunk(3, dim=1)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # spatial branch
        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=self.d_t, S=self.d_s)
        x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_s)

        # modulate
        #ipdb.set_trace()
        x_m = t2i_modulate(self.norm_temp(x), shift_tmp, scale_tmp)

        # temporal branch
        x_t = rearrange(x_m, "B (T S) C -> (B S) T C", T=self.d_t, S=self.d_s)
        if tpe is not None:
            x_t = x_t + tpe
        x_t = self.attn_temp(x_t)
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_tmp * x_t)

        # cross attn
        x = x + self.cross_attn(x, y, mask)

        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


@MODELS.register_module()
class STDiT_QKNorm_RoPE(nn.Module):
    def __init__(
        self,
        input_size=(1, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        dtype=torch.float32,
        space_scale=1.0,
        time_scale=1.0,
        freeze=None,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        qk_norm=False,
        rope=False,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.dtype = dtype
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.space_scale = space_scale
        self.time_scale = time_scale

        self.register_buffer("pos_embed", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.t_block_temp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        #ipdb.set_trace()
        if rope:
            RoPE = RotaryEmbedding(dim=self.hidden_size // self.num_heads)
            self.rope = RoPE.rotate_queries_or_keys
        else:
            self.rope = None
        #ipdb.set_trace()

        self.blocks = nn.ModuleList(
            [
                STDiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flashattn=self.enable_flashattn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    enable_sequence_parallelism=enable_sequence_parallelism,
                    d_t=self.num_temporal,
                    d_s=self.num_spatial,
                    rope=self.rope,
                    qk_norm=qk_norm,
                )
                for i in range(self.depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, np.prod(self.patch_size), self.out_channels)

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        if freeze is not None:
            assert freeze in ["not_temporal", "text"]
            if freeze == "not_temporal":
                self.freeze_not_temporal()
            elif freeze == "text":
                self.freeze_text()

        # sequence parallel related configs
        self.enable_sequence_parallelism = enable_sequence_parallelism
        if enable_sequence_parallelism:
            self.sp_rank = dist.get_rank(get_sequence_parallel_group())
        else:
            self.sp_rank = None

    def forward(self, x, timestep, y, mask=None):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """

        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        # embedding
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=self.num_temporal, S=self.num_spatial)
        x = x + self.pos_embed
        x = rearrange(x, "B T S C -> B (T S) C")

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=1, grad_scale="down")

        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t0 = self.t_block(t)  # [B, C]
        t0_temp = self.t_block_temp(t)  # [B, C]
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # blocks
        for i, block in enumerate(self.blocks):
            if i == 0:
                if self.enable_sequence_parallelism:
                    tpe = torch.chunk(
                        self.pos_embed_temporal, dist.get_world_size(get_sequence_parallel_group()), dim=1
                    )[self.sp_rank].contiguous()
                else:
                    tpe = self.pos_embed_temporal
            else:
                tpe = None
            x = auto_grad_checkpoint(block, x, y, t0, t0_temp, y_lens, tpe)

        if self.enable_sequence_parallelism:
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=1, grad_scale="up")
        # x.shape: [B, N, C]

        # final process
        x = self.final_layer(x, t)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x)  # [B, C_out, T, H, W]

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def unpatchify(self, x):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x

    def unpatchify_old(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            scale=self.space_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
            scale=self.time_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.t_block_temp[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


@MODELS.register_module("STDiT_QKNorm_RoPE_XL/2")
def STDiT_QKNorm_RoPE_XL_2(from_pretrained=None, **kwargs):
    #ipdb.set_trace()
    model = STDiT_QKNorm_RoPE(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model
