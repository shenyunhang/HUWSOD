# Copyright (c) Facebook, Inc. and its affiliates.

from .backbone import (
    build_mrrp_vgg_backbone,
    build_mrrp_wsl_resnet_backbone,
    build_vgg_backbone,
    build_wsl_resnet_backbone,
    build_wsl_resnet_v2_backbone,
)
from .meta_arch import *
from .postprocessing import detector_postprocess
from .proposal_generator import RPNWSL
from .roi_heads import *
from .seg_heads import TwoClassHead, WSJDSROIHeads
from .test_time_augmentation_avg import DatasetMapperTTAAVG, GeneralizedRCNNWithTTAAVG
from .test_time_augmentation_union import DatasetMapperTTAUNION, GeneralizedRCNNWithTTAUNION

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
