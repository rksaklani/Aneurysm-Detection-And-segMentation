"""
Transformer-based Models

This module implements advanced transformer-based models for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Primus(nn.Module):
    """
    Primus: Pure Transformer model for medical image segmentation.
    """
    
    def __init__(
        self,
        img_size: tuple = (64, 64, 64),
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 384,
        patch_size: int = 8,
        num_heads: int = 6,
        depth: int = 6
    ):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        # Decoder
        self.decoder = nn.ConvTranspose3d(
            embed_dim, out_channels,
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H', W', D']
        
        # Flatten and transpose for transformer
        B, C, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W'*D', embed_dim]
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Reshape back to 3D
        x = x.transpose(1, 2).reshape(B, C, H, W, D)
        
        # Decoder
        x = self.decoder(x)
        
        return x


class SlimUNETR(nn.Module):
    """
    Slim UNETR: Lightweight transformer-based U-Net.
    """
    
    def __init__(
        self,
        img_size: tuple = (64, 64, 64),
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 256,
        patch_size: int = 8,
        num_heads: int = 4,
        depth: int = 4
    ):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,  # Reduced for slim version
                dropout=0.1,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        # Decoder with skip connections
        self.decoder = nn.ModuleList([
            nn.ConvTranspose3d(embed_dim, embed_dim // 2, 2, stride=2),
            nn.Conv3d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.Conv3d(embed_dim // 2, out_channels, 1)
        ])
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Flatten and transpose for transformer
        B, C, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Reshape back to 3D
        x = x.transpose(1, 2).reshape(B, C, H, W, D)
        
        # Decoder
        x = self.decoder[0](x)  # Upsample
        x = self.decoder[1](x)  # Conv
        x = self.decoder[2](x)  # Final conv
        
        return x
