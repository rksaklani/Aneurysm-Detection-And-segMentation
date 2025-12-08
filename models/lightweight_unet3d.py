"""
Lightweight 3D U-Net Implementation

This module implements a simplified 3D U-Net architecture that's compatible
with the server-optimized configuration and prevents channel mismatches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightUNet3D(nn.Module):
    """
    Lightweight 3D U-Net for medical image segmentation.
    
    This implementation is optimized for server environments with
    reduced memory usage and compatible architecture.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 16,
        depth: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize lightweight 3D U-Net.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_features: Number of base features in first layer
            depth: Network depth (number of downsampling levels)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_features = base_features
        self.depth = depth
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_features * (2 ** i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(dropout)
                )
            )
            if i < depth - 1:  # Don't pool after the last encoder
                self.pool.append(nn.MaxPool3d(2))
            in_ch = out_ch
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()
        
        for i in range(depth - 1, 0, -1):
            in_ch = base_features * (2 ** i)
            out_ch = base_features * (2 ** (i - 1))
            
            self.upconv.append(nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2))
            self.decoder.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(dropout)
                )
            )
        
        # Final layer
        self.final = nn.Conv3d(base_features, out_channels, 1)
        
    def forward(self, x):
        """Forward pass."""
        # Encoder
        encoder_outputs = []
        for i in range(self.depth):
            x = self.encoder[i](x)
            encoder_outputs.append(x)
            if i < self.depth - 1:
                x = self.pool[i](x)
        
        # Decoder
        for i in range(self.depth - 1):
            x = self.upconv[i](x)
            x = torch.cat([x, encoder_outputs[self.depth - 2 - i]], dim=1)
            x = self.decoder[i](x)
        
        # Final layer
        x = self.final(x)
        
        return x
