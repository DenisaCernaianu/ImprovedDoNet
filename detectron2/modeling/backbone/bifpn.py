import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List

from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.layers import Conv2d, get_norm, ShapeSpec
from detectron2.modeling.backbone.backbone import Backbone 

# --- Weighted Fusion Module ---
class WeightedFusion(nn.Module):
    """
    Performs weighted fusion of two input features.
    FIX: The constructor is defined cleanly to avoid positional argument conflicts.
    """
    def __init__(self, channels: int): 
        super().__init__()
        
        self.channels = channels
        
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.relu = nn.ReLU()
        
        self.conv = Conv2d(self.channels, self.channels, kernel_size=3, padding=1)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        
        weights = self.relu(self.weights)
        norm_weights = weights / (torch.sum(weights) + 1e-4)
        
        if x_a.shape[-2:] != x_b.shape[-2:]:
             x_b = F.interpolate(x_b, size=x_a.shape[2:], mode='nearest')
        
    
        fused_output = norm_weights[0] * x_a + norm_weights[1] * x_b
        
        return self.conv(fused_output) 


# --- BiFPN Backbone  ---

class BiFPN(Backbone): 
    """
    Simplified BiFPN structure (one layer of bidirectional fusion).
    """
    def __init__(self, bottom_up, in_features, out_channels, norm="", **kwargs):
        super().__init__() 
        
        self.bottom_up = bottom_up
        self.in_features = in_features
        self.out_channels = out_channels
        self.norm = norm
        
        self._out_features = [f.replace('res', 'p') for f in in_features]
        self._out_features.append('p6') 
        self.norm_layer = get_norm(norm, out_channels)


        self.lateral_convs = nn.ModuleDict()
        input_shapes = bottom_up.output_shape()
        for f in in_features:
            in_channels = input_shapes[f].channels
            self.lateral_convs[f] = Conv2d(in_channels, out_channels, kernel_size=1, norm=self.norm_layer)

        self.fusion_weights = nn.ModuleDict()
        
        self.fusion_weights["p3_out"] = WeightedFusion(channels=out_channels) 
        self.fusion_weights["p4_out"] = WeightedFusion(channels=out_channels)
        self.fusion_weights["p5_out"] = WeightedFusion(channels=out_channels)


    def output_shape(self) -> Dict[str, ShapeSpec]:
        """
        Returns a mapping from feature name (e.g., 'p3') to its output shape spec.
        Required by Detectron2 to connect RPN and ROI heads.
        """
        out_shapes = {}
        for f_name in self._out_features:
            level = int(f_name[1:]) # p3 -> 3, p6 -> 6
            out_shapes[f_name] = ShapeSpec(
                channels=self.out_channels, stride=2 ** level
            )
        return out_shapes


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.bottom_up(x) 
        
        p_features = {}
        for f in self.in_features:
            p_features[f] = self.lateral_convs[f](features[f])
            
        p5 = p_features[self.in_features[-1]]
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2, padding=0)
        p_features["p6"] = p6

        bifpn_output = {}

        bifpn_output["p5"] = self.fusion_weights["p5_out"](p_features[self.in_features[-1]], p_features["p6"])
        
        p5_resized = F.interpolate(bifpn_output["p5"], size=p_features[self.in_features[2]].shape[2:], mode='nearest')
        bifpn_output["p4"] = self.fusion_weights["p4_out"](p_features[self.in_features[2]], p5_resized)
        
        p4_resized = F.interpolate(bifpn_output["p4"], size=p_features[self.in_features[1]].shape[2:], mode='nearest')
        bifpn_output["p3"] = self.fusion_weights["p3_out"](p_features[self.in_features[1]], p4_resized)

        bifpn_output["p6"] = p6 

        return bifpn_output


# --- Backbone Registration ---

@BACKBONE_REGISTRY.register()
def build_bifpn_backbone(cfg, input_shape):
    """
    Registers the BiFPN backbone using ResNet as the bottom-up feature extractor.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)

    out = BiFPN(
        bottom_up=bottom_up,
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
    )
    return out
