"""
SwinUNETR Implementation

This module implements the SwinUNETR architecture, which combines
Swin Transformer with U-Net for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union
import math
import logging

logger = logging.getLogger(__name__)


class PatchMerging(nn.Module):
    """Patch merging layer for Swin Transformer."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, C = x.shape
        
        # Merge patches in 2x2x2 manner
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x6 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 D/2 8*C
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class PatchSplitting(nn.Module):
    """Patch splitting layer for Swin Transformer."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.expansion = nn.Linear(dim, 4 * dim, bias=False)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, C = x.shape
        
        x = self.norm(x)
        x = self.expansion(x)
        
        # Split into 8 parts
        x0 = x[:, :, :, :, 0::8]
        x1 = x[:, :, :, :, 1::8]
        x2 = x[:, :, :, :, 2::8]
        x3 = x[:, :, :, :, 3::8]
        x4 = x[:, :, :, :, 4::8]
        x5 = x[:, :, :, :, 5::8]
        x6 = x[:, :, :, :, 6::8]
        x7 = x[:, :, :, :, 7::8]
        
        # Rearrange to 2x2x2 patches
        x = torch.zeros(B, H*2, W*2, D*2, C//8, device=x.device, dtype=x.dtype)
        x[:, 0::2, 0::2, 0::2, :] = x0
        x[:, 1::2, 0::2, 0::2, :] = x1
        x[:, 0::2, 1::2, 0::2, :] = x2
        x[:, 0::2, 0::2, 1::2, :] = x3
        x[:, 1::2, 1::2, 0::2, :] = x4
        x[:, 1::2, 0::2, 1::2, :] = x5
        x[:, 0::2, 1::2, 1::2, :] = x6
        x[:, 1::2, 1::2, 1::2, :] = x7
        
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention."""
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        shift_size: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, C = x.shape
        window_size, shift_size = self.window_size, self.shift_size
        
        x = self.norm1(x)
        
        # Cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = self.window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size[0] * window_size[1] * window_size[2], C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], window_size[2], C)
        shifted_x = self.window_reverse(attn_windows, window_size, H, W, D)
        
        # Reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    
    def window_partition(self, x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
        """Partition into non-overlapping windows."""
        B, H, W, D, C = x.shape
        x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], D // window_size[2], window_size[2], C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
        return windows
    
    def window_reverse(self, windows: torch.Tensor, window_size: Tuple[int, int, int], H: int, W: int, D: int) -> torch.Tensor:
        """Reverse window partition."""
        B = int(windows.shape[0] / (H * W * D / window_size[0] / window_size[1] / window_size[2]))
        x = windows.view(B, H // window_size[0], W // window_size[1], D // window_size[2], window_size[0], window_size[1], window_size[2], -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, -1)
        return x


class SwinUNETR(nn.Module):
    """
    SwinUNETR: Swin Transformer for medical image segmentation.
    
    This implementation combines Swin Transformer with U-Net architecture
    for improved medical image segmentation performance.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 1,
        out_channels: int = 1,
        depths: List[int] = [2, 2, 2, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        feature_size: int = 48,
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3
    ):
        """
        Initialize SwinUNETR.
        
        Args:
            img_size: Input image size
            in_channels: Number of input channels
            out_channels: Number of output channels
            depths: Number of layers in each stage
            num_heads: Number of attention heads in each stage
            feature_size: Feature size
            norm_name: Normalization type
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Drop path rate
            normalize: Whether to normalize
            use_checkpoint: Whether to use gradient checkpointing
            spatial_dims: Spatial dimensions
        """
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depths = depths
        self.num_heads = num_heads
        self.feature_size = feature_size
        self.normalize = normalize
        self.use_checkpoint = use_checkpoint
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(
            patch_size=(4, 4, 4),
            in_channels=in_channels,
            embed_dim=feature_size
        )
        
        # Swin Transformer stages
        self.stages = nn.ModuleList()
        self.patch_merging = nn.ModuleList()
        
        for i in range(len(depths)):
            stage = nn.ModuleList([
                SwinTransformerBlock(
                    dim=feature_size * (2 ** i),
                    num_heads=num_heads[i],
                    window_size=(7, 7, 7),
                    shift_size=(0, 0, 0) if i % 2 == 0 else (3, 3, 3),
                    mlp_ratio=4.0,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dropout_path_rate
                ) for _ in range(depths[i])
            ])
            self.stages.append(stage)
            
            if i < len(depths) - 1:
                self.patch_merging.append(PatchMerging(feature_size * (2 ** i)))
        
        # Decoder
        self.decoder = SwinUNETRDecoder(
            feature_size=feature_size,
            depths=depths,
            out_channels=out_channels,
            img_size=img_size
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SwinUNETR."""
        # Patch embedding
        x = self.patch_embed(x)
        
        # Swin Transformer stages
        skip_connections = []
        
        for i, (stage, patch_merge) in enumerate(zip(self.stages, self.patch_merging + [None])):
            for block in stage:
                x = block(x)
            
            skip_connections.append(x)
            
            if patch_merge is not None:
                x = patch_merge(x)
        
        # Decoder
        output = self.decoder(x, skip_connections)
        
        return output


class PatchEmbedding3D(nn.Module):
    """3D patch embedding for SwinUNETR."""
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_channels: int = 1,
        embed_dim: int = 48
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.projection = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)  # (B, embed_dim, H, W, D)
        x = x.permute(0, 2, 3, 4, 1)  # (B, H, W, D, embed_dim)
        return x


class SwinUNETRDecoder(nn.Module):
    """Decoder for SwinUNETR."""
    
    def __init__(
        self,
        feature_size: int,
        depths: List[int],
        out_channels: int,
        img_size: Tuple[int, int, int]
    ):
        super().__init__()
        self.feature_size = feature_size
        self.depths = depths
        self.out_channels = out_channels
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        self.patch_splitting = nn.ModuleList()
        
        for i in range(len(depths) - 1, 0, -1):
            # Patch splitting
            self.patch_splitting.append(PatchSplitting(feature_size * (2 ** i)))
            
            # Upsampling
            self.upsample_layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        feature_size * (2 ** i),
                        feature_size * (2 ** (i - 1)),
                        kernel_size=3,
                        padding=1
                    ),
                    nn.BatchNorm3d(feature_size * (2 ** (i - 1))),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv3d(feature_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through decoder."""
        # Reverse skip connections
        skip_connections = skip_connections[::-1]
        
        # Progressive upsampling
        for i, (upsample, patch_split) in enumerate(zip(self.upsample_layers, self.patch_splitting)):
            # Patch splitting
            x = patch_split(x)
            
            # Skip connection
            x = x + skip_connections[i + 1]
            
            # Upsampling
            x = x.permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
            x = upsample(x)
            x = x.permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)
        
        # Final output
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        output = self.final_conv(x)
        
        return output
