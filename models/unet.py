"""
U-Net Implementation

This module implements the classic U-Net architecture and its 3D variant
for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        mid_channels: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bilinear: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2,
                        diff_z // 2, diff_z - diff_z // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for medical image segmentation.
    
    This implementation follows the original U-Net architecture but adapted
    for 3D medical images with proper skip connections and upsampling.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
        bilinear: bool = True
    ):
        """
        Initialize 3D U-Net.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_features: Number of base features in first layer
            depth: Network depth (number of downsampling levels)
            dropout: Dropout rate
            bilinear: Whether to use bilinear upsampling
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.depth = depth
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_features, dropout=dropout)
        self.down_layers = nn.ModuleList()
        
        for i in range(depth):
            in_ch = base_features * (2 ** i)
            out_ch = base_features * (2 ** (i + 1))
            self.down_layers.append(Down(in_ch, out_ch, dropout=dropout))
        
        # Bottleneck
        self.bottleneck = DoubleConv(
            base_features * (2 ** depth),
            base_features * (2 ** (depth + 1)),
            dropout=dropout
        )
        
        # Decoder
        self.up_layers = nn.ModuleList()
        
        for i in range(depth, 0, -1):
            in_ch = base_features * (2 ** (i + 1))
            out_ch = base_features * (2 ** i)
            self.up_layers.append(Up(in_ch, out_ch, bilinear, dropout=dropout))
        
        # Output
        self.outc = OutConv(base_features, out_channels)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Encoder
        x1 = self.inc(x)
        skip_connections = [x1]
        
        for down in self.down_layers:
            x1 = down(x1)
            skip_connections.append(x1)
        
        # Bottleneck
        x1 = self.bottleneck(x1)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for i, up in enumerate(self.up_layers):
            x1 = up(x1, skip_connections[i + 1])
        
        # Output
        logits = self.outc(x1)
        return logits


class UNet(nn.Module):
    """
    2D U-Net for medical image segmentation.
    
    This is the classic 2D U-Net architecture adapted for medical images.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
        bilinear: bool = True
    ):
        """
        Initialize 2D U-Net.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_features: Number of base features in first layer
            depth: Network depth
            dropout: Dropout rate
            bilinear: Whether to use bilinear upsampling
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.depth = depth
        
        # Encoder
        self.inc = self._double_conv(in_channels, base_features, dropout)
        self.down_layers = nn.ModuleList()
        
        for i in range(depth):
            in_ch = base_features * (2 ** i)
            out_ch = base_features * (2 ** (i + 1))
            self.down_layers.append(self._down(in_ch, out_ch, dropout))
        
        # Bottleneck
        self.bottleneck = self._double_conv(
            base_features * (2 ** depth),
            base_features * (2 ** (depth + 1)),
            dropout
        )
        
        # Decoder
        self.up_layers = nn.ModuleList()
        
        for i in range(depth, 0, -1):
            in_ch = base_features * (2 ** (i + 1))
            out_ch = base_features * (2 ** i)
            self.up_layers.append(self._up(in_ch, out_ch, bilinear, dropout))
        
        # Output
        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _double_conv(self, in_channels: int, out_channels: int, dropout: float) -> nn.Module:
        """Create double convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
    
    def _down(self, in_channels: int, out_channels: int, dropout: float) -> nn.Module:
        """Create downsampling block."""
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._double_conv(in_channels, out_channels, dropout)
        )
    
    def _up(self, in_channels: int, out_channels: int, bilinear: bool, dropout: float) -> nn.Module:
        """Create upsampling block."""
        if bilinear:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                self._double_conv(in_channels, out_channels, dropout)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                self._double_conv(in_channels, out_channels, dropout)
            )
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Encoder
        x1 = self.inc(x)
        skip_connections = [x1]
        
        for down in self.down_layers:
            x1 = down(x1)
            skip_connections.append(x1)
        
        # Bottleneck
        x1 = self.bottleneck(x1)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for i, up in enumerate(self.up_layers):
            x1 = up(x1)
            # Handle size mismatch
            if x1.size() != skip_connections[i + 1].size():
                x1 = F.interpolate(x1, size=skip_connections[i + 1].size()[2:], 
                                 mode='bilinear', align_corners=True)
            x1 = torch.cat([skip_connections[i + 1], x1], dim=1)
        
        # Output
        logits = self.outc(x1)
        return logits


class AttentionUNet(nn.Module):
    """
    U-Net with attention gates for improved segmentation.
    
    This implementation adds attention gates to the skip connections
    to focus on relevant features.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
        depth: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize Attention U-Net.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_features: Number of base features
            depth: Network depth
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_features, dropout=dropout)
        self.down_layers = nn.ModuleList()
        
        for i in range(depth):
            in_ch = base_features * (2 ** i)
            out_ch = base_features * (2 ** (i + 1))
            self.down_layers.append(Down(in_ch, out_ch, dropout=dropout))
        
        # Bottleneck
        self.bottleneck = DoubleConv(
            base_features * (2 ** depth),
            base_features * (2 ** (depth + 1)),
            dropout=dropout
        )
        
        # Attention gates
        self.attention_gates = nn.ModuleList()
        
        for i in range(depth, 0, -1):
            in_ch = base_features * (2 ** i)
            self.attention_gates.append(AttentionGate(in_ch, in_ch // 2))
        
        # Decoder
        self.up_layers = nn.ModuleList()
        
        for i in range(depth, 0, -1):
            in_ch = base_features * (2 ** (i + 1))
            out_ch = base_features * (2 ** i)
            self.up_layers.append(Up(in_ch, out_ch, dropout=dropout))
        
        # Output
        self.outc = OutConv(base_features, out_channels)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Encoder
        x1 = self.inc(x)
        skip_connections = [x1]
        
        for down in self.down_layers:
            x1 = down(x1)
            skip_connections.append(x1)
        
        # Bottleneck
        x1 = self.bottleneck(x1)
        
        # Decoder with attention
        skip_connections = skip_connections[::-1]
        
        for i, up in enumerate(self.up_layers):
            # Apply attention gate
            g = self.attention_gates[i](x1, skip_connections[i + 1])
            x1 = up(x1, g)
        
        # Output
        logits = self.outc(x1)
        return logits


class AttentionGate(nn.Module):
    """Attention gate for U-Net skip connections."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int = None):
        super().__init__()
        if F_int is None:
            F_int = F_l // 2
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention gate."""
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Handle size mismatch
        if g1.size() != x1.size():
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='trilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi
