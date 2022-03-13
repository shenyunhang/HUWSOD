# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_wsl_config(cfg):
    """
    Add config for mrrpnet.
    """
    _C = cfg

    # ---------------------------------------------------------------------------- #
    # VGG
    # ---------------------------------------------------------------------------- #
    _C.MODEL.VGG = CN()
    _C.MODEL.VGG.DEPTH = 16
    _C.MODEL.VGG.OUT_FEATURES = ["plain5"]
    _C.MODEL.VGG.CONV5_DILATION = 1

    # ---------------------------------------------------------------------------- #
    # WSR
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ROI_BOX_HEAD.DAN_DIM = [4096, 4096]

    # ---------------------------------------------------------------------------- #
    # Misc options
    # ---------------------------------------------------------------------------- #
    _C.DATASETS.VAL = ()
    _C.DATASETS.PROPOSAL_FILES_VAL = ()

    _C.DATALOADER.CLASS_ASPECT_RATIO_GROUPING = False

    _C.TEST.EVAL_TRAIN = True
    _C.TEST.NUM_CLASSES_TEST = 0
    _C.VIS_TEST = False

    # ---------------------------------------------------------------------------- #
    # WSL
    # ---------------------------------------------------------------------------- #
    _C.WSL = CN()
    _C.WSL.ITER_SIZE = 1
    _C.WSL.USE_OBN = True

    # ---------------------------------------------------------------------------- #
    # WSDDN
    # ---------------------------------------------------------------------------- #
    _C.WSL.WSDDN = CN()
    _C.WSL.WSDDN.WEIGHT = 1.0
    _C.WSL.WSDDN.MEAN_LOSS = True

    # ---------------------------------------------------------------------------- #
    # OICR
    # ---------------------------------------------------------------------------- #
    _C.WSL.OICR = CN()
    _C.WSL.OICR.WEIGHT = 1.0
    _C.WSL.OICR.REFINE_NUM = 3
    _C.WSL.OICR.REFINE_REG = [False, False, False]
    _C.WSL.OICR.REFINE_MIST = False
    _C.WSL.OICR.CROSS_ENTROPY_WEIGHTED = True

    # ---------------------------------------------------------------------------- #
    # CSC
    # ---------------------------------------------------------------------------- #
    _C.WSL.CSC_MAX_ITER = 35000

    # ---------------------------------------------------------------------------- #
    # CMIL
    # ---------------------------------------------------------------------------- #
    _C.WSL.SIZE_EPOCH = 5000
    _C.WSL.CMIL = False

    # ---------------------------------------------------------------------------- #
    # JTSM
    # ---------------------------------------------------------------------------- #
    _C.WSL.JTSM = CN()
    _C.WSL.JTSM.WEIGHT = 1.0
    _C.WSL.JTSM.MEAN_LOSS = True

    _C.MODEL.SEM_SEG_HEAD.ASSP_CONVS_DIM = [1024, 1024]
    _C.MODEL.SEM_SEG_HEAD.MASK_SOFTMAX = False
    _C.MODEL.SEM_SEG_HEAD.CONSTRAINT = False

    _C.WSL.CASCADE_ON = False
    _C.WSL.PS_ON = False
    _C.WSL.SP_ON = False
    _C.WSL.MASK_MINED_TOP_K = 10

    # ---------------------------------------------------------------------------- #
    # UWSOD
    # ---------------------------------------------------------------------------- #
    _C.WSL.SAMPLING = CN()
    _C.WSL.SAMPLING.SAMPLING_ON = False
    _C.WSL.SAMPLING.IOU_THRESHOLDS = [[0.5], [0.5], [0.5], [0.5]]
    _C.WSL.SAMPLING.IOU_LABELS = [[0, 1], [0, 1], [0, 1], [0, 1]]
    _C.WSL.SAMPLING.BATCH_SIZE_PER_IMAGE = [4096, 4096, 4096, 4096]
    _C.WSL.SAMPLING.POSITIVE_FRACTION = [1.0, 1.0, 1.0, 1.0]

    _C.MODEL.MRRP = CN()
    _C.MODEL.MRRP.MRRP_ON = False
    _C.MODEL.MRRP.NUM_BRANCH = 3
    _C.MODEL.MRRP.BRANCH_DILATIONS = [1, 2, 3]
    _C.MODEL.MRRP.MRRP_STAGE = "res4"
    _C.MODEL.MRRP.TEST_BRANCH_IDX = 1

    _C.WSL.CLS_AGNOSTIC_BBOX_KNOWN = False

    # ---------------------------------------------------------------------------- #
    # HUWSOD
    # ---------------------------------------------------------------------------- #
    _C.WSL.HUWSOD = CN()
    _C.WSL.HUWSOD.BRANCH_SUPERVISION = [-1, -1, -1, -1]
    _C.WSL.HUWSOD.RPN_SUPERVISION = [-1]
    _C.WSL.HUWSOD.WEAK_CLS_WEIGHT = 1
    _C.WSL.HUWSOD.WEAK_REG_WEIGHT = 1
    _C.WSL.HUWSOD.STRONG_CLS_WEIGHT = 1
    _C.WSL.HUWSOD.STRONG_REG_WEIGHT = 1

    _C.MODEL.CLS_HEAD = CN()
    _C.MODEL.CLS_HEAD.NAME = "ClsConvFCHead"

    _C.WSL.DATA = CN()
    _C.WSL.DATA.WEAKSTRONG_ON = False
    _C.WSL.DATA.DEBUG = False

    _C.WSL.AE = CN()
    _C.WSL.AE.MEAN_LOSS = True
    _C.WSL.AE.PROPOSAL_ON_FEATURE = False
    _C.WSL.AE.RANK = 1
    _C.WSL.AE.RANK_SUPERVISION = 2
    _C.WSL.AE.BURNIN_ITERS = 100000
    _C.WSL.AE.BURNIN_TYPE = "both"

    _C.WSL.EMA = CN()
    _C.WSL.EMA.ENABLE = False
    _C.WSL.EMA.DYNAMIC = False
    _C.WSL.EMA.DECAY = 0.996
    _C.WSL.EMA.DIRECTION = "student2teacher"
