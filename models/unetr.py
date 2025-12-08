"""
UNETR Implementation

This module implements the UNETR (UNet Transformer) architecture for medical
image segmentation, combining Vision Transformers with U-Net decoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union
import math
import logging

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Patch embedding for Vision Transformer."""
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        
        self.projection = nn.Conv3d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)  # (B, embed_dim, H//patch_size, W//patch_size, D//patch_size)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, embed_dim: int, max_len: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UNETR(nn.Module):
    """
    UNETR: UNet Transformer for medical image segmentation.
    
    This implementation combines Vision Transformers with U-Net decoders
    for improved medical image segmentation performance.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 768,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        drop_path: float = 0.1
    ):
        """
        Initialize UNETR.
        
        Args:
            img_size: Input image size
            in_channels: Number of input channels
            out_channels: Number of output channels
            embed_dim: Embedding dimension
            patch_size: Patch size for tokenization
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to use bias in QKV projection
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            drop_path: Drop path rate
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                qkv_bias=qkv_bias
            ) for _ in range(num_layers)
        ])
        
        # Decoder
        self.decoder = UNETRDecoder(
            embed_dim=embed_dim,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through UNETR."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed[:, 1:, :]  # Skip cls token for now
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Decoder
        output = self.decoder(x, self.img_size, self.patch_size)
        
        return output


class UNETRDecoder(nn.Module):
    """Decoder for UNETR."""
    
    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int]
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Calculate feature sizes
        self.feature_sizes = []
        for i in range(4):
            size = tuple(s // (2 ** i) for s in img_size)
            self.feature_sizes.append(size)
        
        # Projection layers
        self.proj_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim // (2 ** i)) for i in range(4)
        ])
        
        # Upsampling layers
        self.up_layers = nn.ModuleList([
            nn.ConvTranspose3d(
                embed_dim // (2 ** i),
                embed_dim // (2 ** (i + 1)),
                kernel_size=2,
                stride=2
            ) for i in range(3)
        ])
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv3d(embed_dim // 8, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, img_size: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> torch.Tensor:
        """Forward pass through decoder."""
        B = x.shape[0]
        
        # Reshape to spatial dimensions
        patch_dims = tuple(s // p for s, p in zip(img_size, patch_size))
        x = x.reshape(B, *patch_dims, -1)
        x = x.permute(0, 4, 1, 2, 3)  # (B, embed_dim, H, W, D)
        
        # Progressive upsampling
        for i, (proj, up) in enumerate(zip(self.proj_layers[:-1], self.up_layers)):
            x = proj(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            x = up(x)
        
        # Final projection
        x = self.proj_layers[-1](x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        
        # Final convolution
        output = self.final_conv(x)
        
        return output


class UNETRPlusPlus(nn.Module):
    """
    UNETR++: Enhanced UNETR with nested skip connections.
    
    This implementation adds nested skip connections similar to UNet++
    to improve feature fusion and segmentation accuracy.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 768,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Nested decoder
        self.nested_decoder = NestedUNETRDecoder(
            embed_dim=embed_dim,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through UNETR++."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Nested decoder
        output = self.nested_decoder(x, self.img_size, self.patch_size)
        
        return output


class NestedUNETRDecoder(nn.Module):
    """Nested decoder for UNETR++."""
    
    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int]
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        
        # Feature dimensions
        self.feature_dims = [embed_dim // (2 ** i) for i in range(4)]
        
        # Projection layers
        self.proj_layers = nn.ModuleList([
            nn.Linear(embed_dim, dim) for dim in self.feature_dims
        ])
        
        # Nested connections
        self.nested_convs = nn.ModuleList()
        for i in range(3):
            level_convs = nn.ModuleList()
            for j in range(3 - i):
                in_ch = self.feature_dims[i] + self.feature_dims[i + j + 1]
                out_ch = self.feature_dims[i + 1]
                level_convs.append(
                    nn.Sequential(
                        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                        nn.BatchNorm3d(out_ch),
                        nn.ReLU(inplace=True)
                    )
                )
            self.nested_convs.append(level_convs)
        
        # Final output
        self.final_conv = nn.Conv3d(self.feature_dims[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, img_size: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> torch.Tensor:
        """Forward pass through nested decoder."""
        B = x.shape[0]
        
        # Reshape to spatial dimensions
        patch_dims = tuple(s // p for s, p in zip(img_size, patch_size))
        x = x.reshape(B, *patch_dims, -1)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Project to different feature dimensions
        features = []
        for proj in self.proj_layers:
            feat = proj(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            features.append(feat)
        
        # Nested connections
        nested_features = [features[0]]
        
        for i in range(3):
            level_features = []
            for j in range(3 - i):
                if j == 0:
                    # First connection in level
                    in_feat = features[i]
                else:
                    # Nested connection
                    in_feat = torch.cat([features[i], nested_features[i][j-1]], dim=1)
                
                out_feat = self.nested_convs[i][j](in_feat)
                level_features.append(out_feat)
            
            nested_features.append(level_features)
        
        # Final output
        output = self.final_conv(nested_features[0])
        
        return output
