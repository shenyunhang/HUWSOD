# Copyright (c) Facebook, Inc. and its affiliates.
from .resnet_wsl import build_wsl_resnet_backbone
from .resnet_wsl_mrrp import build_mrrp_wsl_resnet_backbone
from .resnet_wsl_v2 import build_wsl_resnet_v2_backbone, build_wsl_resnet_v2_fpn_backbone
from .vgg import VGG16, PlainBlockBase, build_vgg_backbone
from .vgg_mrrp import build_mrrp_vgg_backbone

# TODO can expose more resnet blocks after careful consideration
