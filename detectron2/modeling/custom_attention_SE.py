import torch
from torch import nn
import torch.nn.functional as F
from detectron2.layers import Conv2d, get_norm
from typing import Dict, List
import fvcore.nn.weight_init as weight_init

class SEBlock(nn.Module):
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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
        return self.sigmoid(avg_out + max_out)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_concat))


class SECBAMBlock(nn.Module):
    
    def __init__(self, channels, reduction=16, norm="GN"):
        super().__init__()
        self.se = SEBlock(channels, reduction)
        
        self.ca = ChannelAttentionModule(channels, reduction)
        self.sa = SpatialAttentionModule()
        
        self.conv = Conv2d(channels, channels, kernel_size=3, padding=1, norm=get_norm(norm, channels))
        weight_init.c2_msra_fill(self.conv)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.se(x)
        
        x = x * self.ca(x)
        x = x * self.sa(x)
        
        
        return self.relu(self.conv(x))


class FeatureRefiner(nn.Module):
   
    def __init__(self, in_channels: int, num_stacks: int = 4, norm: str = "GN", **kwargs):
        super().__init__()
        
        self.num_stacks = num_stacks
        
        self.refiner_blocks = nn.ModuleList([
            SECBAMBlock(in_channels, reduction=16, norm=norm) for _ in range(num_stacks)
        ])
        
        self.output_conv = Conv2d(in_channels, in_channels, kernel_size=1)
        weight_init.c2_msra_fill(self.output_conv)
        self.relu = nn.ReLU(True)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        refined_features = {}
        for k, x in features.items():
            y = x
            for block in self.refiner_blocks:
                y = block(y)
            
            refined_features[k] = self.relu(self.output_conv(x + y))
            
        return refined_features
