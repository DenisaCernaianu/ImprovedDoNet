import torch
from torch import nn
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling import BACKBONE_REGISTRY

class SimpleFPN_P6(FPN):
    """
    Standard FPN (Top-Down Path) that explicitly uses LastLevelMaxPool
    to ensure P6/P7 features are generated for compatibility with RPN/ROI Heads.
    """
    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"):
        
        
        top_block_p6 = LastLevelMaxPool(
            in_channels=bottom_up.out_feature_channels[in_features[-1]], 
            out_channels=out_channels
        )
        
        super().__init__(
            bottom_up=bottom_up, 
            in_features=in_features, 
            out_channels=out_channels, 
            norm=norm, 
            top_block=top_block_p6, 
            fuse_type=fuse_type
        )
    
@BACKBONE_REGISTRY.register()
def build_simple_fpn_p6_backbone(cfg, input_shape):
    """
    Registers the stable FPN backbone that includes P6.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)

    out = SimpleFPN_P6(
        bottom_up=bottom_up,
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return out
