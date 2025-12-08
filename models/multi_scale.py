"""
Multi-scale Models

This module implements multi-scale and stacked U-Net architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedUNet(nn.Module):
    """
    Stacked U-Net: Multiple U-Nets stacked together.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        num_stacks: int = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_features = base_features
        self.depth = depth
        self.num_stacks = num_stacks
        
        # Create stacked U-Nets
        self.unets = nn.ModuleList()
        
        # First U-Net
        self.unets.append(self._create_unet(in_channels, out_channels))
        
        # Additional U-Nets
        for i in range(1, num_stacks):
            self.unets.append(self._create_unet(out_channels, out_channels))
        
    def _create_unet(self, in_ch, out_ch):
        """Create a single U-Net."""
        encoder = nn.ModuleList()
        pool = nn.ModuleList()
        
        in_channels = in_ch
        for i in range(self.depth):
            out_channels = self.base_features * (2 ** i)
            encoder.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            pool.append(nn.MaxPool3d(2))
            in_channels = out_channels
        
        decoder = nn.ModuleList()
        upconv = nn.ModuleList()
        
        for i in range(self.depth - 1, 0, -1):
            in_channels = self.base_features * (2 ** i)
            out_channels = self.base_features * (2 ** (i - 1))
            
            upconv.append(nn.ConvTranspose3d(in_channels, in_channels // 2, 2, stride=2))
            decoder.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        final = nn.Conv3d(self.base_features, out_ch, 1)
        
        return nn.ModuleDict({
            'encoder': encoder,
            'pool': pool,
            'decoder': decoder,
            'upconv': upconv,
            'final': final
        })
    
    def forward(self, x):
        """Forward pass through stacked U-Nets."""
        for i, unet in enumerate(self.unets):
            # Encoder
            encoder_outputs = []
            current_x = x
            
            for j in range(self.depth):
                current_x = unet['encoder'][j](current_x)
                encoder_outputs.append(current_x)
                if j < self.depth - 1:
                    current_x = unet['pool'][j](current_x)
            
            # Decoder
            for j in range(self.depth - 1):
                current_x = unet['upconv'][j](current_x)
                current_x = torch.cat([current_x, encoder_outputs[self.depth - 2 - j]], dim=1)
                current_x = unet['decoder'][j](current_x)
            
            # Final layer
            x = unet['final'](current_x)
        
        return x


class MultiScaleUNet(nn.Module):
    """
    Multi-scale U-Net: U-Net with multiple scales.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        scales: list = [1, 0.5, 0.25]
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_features = base_features
        self.depth = depth
        self.scales = scales
        
        # Create U-Nets for different scales
        self.scale_unets = nn.ModuleList()
        
        for scale in scales:
            self.scale_unets.append(self._create_unet())
        
        # Fusion layer
        self.fusion = nn.Conv3d(
            len(scales) * out_channels, 
            out_channels, 
            1
        )
        
    def _create_unet(self):
        """Create a single U-Net."""
        encoder = nn.ModuleList()
        pool = nn.ModuleList()
        
        in_channels = self.in_channels
        for i in range(self.depth):
            out_channels = self.base_features * (2 ** i)
            encoder.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            pool.append(nn.MaxPool3d(2))
            in_channels = out_channels
        
        decoder = nn.ModuleList()
        upconv = nn.ModuleList()
        
        for i in range(self.depth - 1, 0, -1):
            in_channels = self.base_features * (2 ** i)
            out_channels = self.base_features * (2 ** (i - 1))
            
            upconv.append(nn.ConvTranspose3d(in_channels, in_channels // 2, 2, stride=2))
            decoder.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        final = nn.Conv3d(self.base_features, self.out_channels, 1)
        
        return nn.ModuleDict({
            'encoder': encoder,
            'pool': pool,
            'decoder': decoder,
            'upconv': upconv,
            'final': final
        })
    
    def forward(self, x):
        """Forward pass through multi-scale U-Nets."""
        outputs = []
        
        for i, (scale, unet) in enumerate(zip(self.scales, self.scale_unets)):
            # Resize input for this scale
            if scale != 1.0:
                scaled_x = F.interpolate(
                    x, 
                    scale_factor=scale, 
                    mode='trilinear', 
                    align_corners=False
                )
            else:
                scaled_x = x
            
            # Forward pass through U-Net
            encoder_outputs = []
            current_x = scaled_x
            
            for j in range(self.depth):
                current_x = unet['encoder'][j](current_x)
                encoder_outputs.append(current_x)
                if j < self.depth - 1:
                    current_x = unet['pool'][j](current_x)
            
            # Decoder
            for j in range(self.depth - 1):
                current_x = unet['upconv'][j](current_x)
                current_x = torch.cat([current_x, encoder_outputs[self.depth - 2 - j]], dim=1)
                current_x = unet['decoder'][j](current_x)
            
            # Final layer
            output = unet['final'](current_x)
            
            # Resize back to original size
            if scale != 1.0:
                output = F.interpolate(
                    output, 
                    size=x.shape[2:], 
                    mode='trilinear', 
                    align_corners=False
                )
            
            outputs.append(output)
        
        # Fuse outputs
        fused = torch.cat(outputs, dim=1)
        final_output = self.fusion(fused)
        
        return final_output
