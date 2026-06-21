import torch
import torch.nn as nn
from detectron2.layers import Conv2d, get_norm
from typing import Dict, List
import fvcore.nn.weight_init as weight_init

#CBAM

class ChannelAttentionModule(nn.Module):
   
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_concat)
        return self.sigmoid(out)

class CBAMRefinerBlock(nn.Module):
    """
    Combines Channel Attention and Spatial Attention.
    """
    def __init__(self, channels, reduction=16, kernel_size=7, norm="GN"):
        super().__init__()
        self.ca = ChannelAttentionModule(channels, reduction)
        self.sa = SpatialAttentionModule()
        self.conv = Conv2d(channels, channels, kernel_size=3, padding=1, norm=get_norm(norm, channels))
        weight_init.c2_msra_fill(self.conv)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # 1. Channel Attention
        channel_attn = self.ca(x)
        x = x * channel_attn
        
        # 2. Spatial Attention
        spatial_attn = self.sa(x)
        x = x * spatial_attn
        
        # 3. Convolutional refinement for stability and feature mixing
        x = self.relu(self.conv(x))
        
        return x


class FeatureRefiner(nn.Module):
    """
    A Feature Refiner that applies multiple, stacked CBAM blocks
    to refine FPN features before they are used by the heads.
    """
    def __init__(self, in_channels: int, num_stacks: int = 2, norm: str = "GN"):
        super().__init__()
        
        self.num_stacks = num_stacks
        
        #sequential stack of CBAMRefinerBlocks
        self.refiner_blocks = nn.ModuleList([
            CBAMRefinerBlock(in_channels, norm=norm) for _ in range(num_stacks)
        ])
        
        self.output_conv = Conv2d(in_channels, in_channels, kernel_size=1)
        weight_init.c2_msra_fill(self.output_conv)
        self.relu = nn.ReLU(True)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        refined_features = {}
        
        for k, x in features.items():
            # apply all blocks sequentially
            y = x
            for block in self.refiner_blocks:
                y = block(y)
            
            # original feature + refined feature (Residual connection)
            refined_features[k] = self.relu(self.output_conv(x + y))
            
        return refined_features
