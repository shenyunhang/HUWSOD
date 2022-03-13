# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from skimage import measure
from torch import nn

# from multiprocessing import Pool
from torch.multiprocessing import Pool
from torch.nn import functional as F

import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import ShapeSpec, batched_nms, paste_masks_in_image
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.structures import BitMasks, Boxes, ImageList, Instances, PolygonMasks, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.visualizer import Visualizer
from wsl.modeling.poolers import ROIPooler
from wsl.modeling.roi_heads.fast_rcnn_oicr import OICROutputLayers
from wsl.modeling.roi_heads.fast_rcnn_wsddn import WSDDNOutputLayers
from wsl.modeling.roi_heads.mask_head import (  # MaskRCNNUpsampleWSLHead,
    MaskRCNNWSLHead,
    mask_rcnn_co_loss,
    mask_rcnn_inference,
    mask_rcnn_loss,
)
from wsl.modeling.roi_heads.roi_heads import (
    ROIHeads,
    get_image_level_gt,
    select_foreground_proposals,
    select_proposals_with_visible_keypoints,
)

logger = logging.getLogger(__name__)


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon2(binary_mask, box, tolerance=0):
    polys = binary_mask_to_polygon(binary_mask, tolerance=tolerance)
    if len(polys) == 0:
        # box = boxes[j] * scale_factor
        box = box.clone().to(dtype=torch.int64).cpu().numpy()
        poly = []
        for i in range(box[0] - 1, box[2] + 1):
            poly.extend([i, box[1]])
        for i in range(box[1] - 1, box[3] + 1):
            poly.extend([box[2], i])
        for i in range(box[2] + 1, box[0] - 1, -1):
            poly.extend([i, box[3]])
        for i in range(box[3] + 1, box[1] - 1, -1):
            poly.extend([box[0], i])
        polys.append(poly)
    return polys


def binary_mask_to_polygon(binary_mask, tolerance=0):
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons


def rect_mask(shape, bbox):
    mask = np.zeros(shape[:2], np.float32)
    mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 255
    if len(shape) == 3 and shape[2] == 1:
        mask = np.expand_dims(mask, axis=-1)
    return mask


def do_grabcut(img, box):
    grabcut_iter = 5
    _MIN_AREA = 9
    _RECT_SHRINK = 3

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    width = box[2] - box[0] + 1
    height = box[3] - box[1] + 1
    if width * height < _MIN_AREA:
        assert width * height > 0
        img_mask = rect_mask(img.shape[:2], box)
    else:
        if width * height >= img.shape[0] * img.shape[1]:
            rect = (_RECT_SHRINK, _RECT_SHRINK, width - _RECT_SHRINK * 2, height - _RECT_SHRINK * 2)
        else:
            rect = (box[0], box[1], width, height)

        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, grabcut_iter, cv2.GC_INIT_WITH_RECT)

        # img_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img_mask = np.where((mask == 2) | (mask == 0), 0, 1)

    return img_mask


@ROI_HEADS_REGISTRY.register()
class PDSLROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        output_dir: str = None,
        vis_test: bool = False,
        vis_period: int = 0,
        refine_K: int = 4,
        refine_mist: bool = False,
        refine_reg: List[bool] = [True, True, True, True],
        box_refinery: List[nn.Module] = [None, None, None, None],
        cls_agnostic_bbox_reg: bool = False,
        device: torch.device = None,
        pixel_mean: torch.Tensor = None,
        pixel_std: torch.Tensor = None,
        mask_refinery: List[nn.Module] = [None, None, None, None],
        mask_mined_top_k: int = 10,
        metadata: Dict = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        self.iter = 0
        self.iter_test = 0
        self.epoch_test = 0

        self.output_dir = output_dir
        self.vis_test = vis_test
        self.vis_period = vis_period

        self.refine_K = refine_K
        self.refine_mist = refine_mist
        self.refine_reg = refine_reg
        self.box_refinery = box_refinery
        for k in range(self.refine_K):
            self.add_module("box_refinery_{}".format(k), self.box_refinery[k])
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

        self.mask_refinery = mask_refinery
        for k in range(len(self.mask_refinery)):
            self.add_module("mask_refinery_{}".format(k), self.mask_refinery[k])
        self.mask_mined_top_k = mask_mined_top_k

        self.device = device
        self.img_normalizer = lambda x: (x * pixel_std + pixel_mean)

        self.grabcut_iter = 5
        self._MIN_AREA = 9
        self._RECT_SHRINK = 3

        self.process_pool = Pool(processes=20)

        self.metadata = metadata

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))

        device = torch.device(cfg.MODEL.DEVICE)
        ret["device"] = device
        # TODO(YH): batch size=1
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        ret["pixel_mean"] = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(device).view(1, -1, 1, 1)
        ret["pixel_std"] = torch.Tensor(cfg.MODEL.PIXEL_STD).to(device).view(1, -1, 1, 1)

        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = WSDDNOutputLayers(cfg, box_head.output_shape)

        cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        refine_K = cfg.WSL.OICR.REFINE_NUM
        refine_mist = cfg.WSL.OICR.REFINE_MIST
        refine_reg = cfg.WSL.OICR.REFINE_REG
        box_refinery = []
        for k in range(refine_K):
            box_refinery_k = OICROutputLayers(cfg, box_head.output_shape, k)
            box_refinery.append(box_refinery_k)

        output_dir = cfg.OUTPUT_DIR
        vis_test = cfg.VIS_TEST
        vis_period = cfg.VIS_PERIOD

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "output_dir": output_dir,
            "vis_test": vis_test,
            "vis_period": vis_period,
            "refine_K": refine_K,
            "refine_mist": refine_mist,
            "refine_reg": refine_reg,
            "box_refinery": box_refinery,
            "cls_agnostic_bbox_reg": cls_agnostic_bbox_reg,
            "metadata": metadata,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}

        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        cfg.defrost()
        cfg.MODEL.ROI_HEADS.CLS_AGNOSTIC_MASK = True
        cfg.freeze()

        ret["mask_head"] = build_mask_head(cfg, shape)

        cfg.defrost()
        cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = cls_agnostic_mask
        cfg.freeze()

        mask_refinery = []
        for k in range(4):
            # mask_refinery_k = build_mask_head(
            # mask_refinery_k = MaskRCNNUpsampleWSLHead(
            mask_refinery_k = MaskRCNNWSLHead(
                cfg,
                ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution),
            )
            mask_refinery.append(mask_refinery_k)

        cfg.defrost()
        cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = cls_agnostic_mask
        cfg.freeze()

        ret["mask_mined_top_k"] = cfg.WSL.MASK_MINED_TOP_K

        ret["mask_refinery"] = mask_refinery
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img_int = get_image_level_gt(targets, self.num_classes)

        # del images
        self.images = images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
            self._vis_proposal(proposals, "train", "_proposals")
        else:
            self._save_proposal_test(proposals, "test", "_proposals")

        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))

            self.iter = self.iter + 1
            if self.iter_test > 0:
                self.epoch_test = self.epoch_test + 1
            self.iter_test = 0

            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances, _, _ = self.forward_with_given_boxes(features, pred_instances)

            self.iter_test = self.iter_test + 1

            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances, [], []

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)
        if self.training and objectness_logits.numel() > 0:
            storage = get_event_storage()
            storage.put_scalar("proposals/objectness_logits+1 mean", objectness_logits.mean())
            storage.put_scalar("proposals/objectness_logits+1 max", objectness_logits.max())
            storage.put_scalar("proposals/objectness_logits+1 min", objectness_logits.min())

        # torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, proposals)
        # del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals, self.gt_classes_img_int)

            self.pred_class_img_logits = (
                self.box_predictor.predict_probs_img(predictions, proposals).clone().detach()
            )

            # prev_pred_scores = predictions[0].detach()
            # prev_pred_boxes = [p.proposal_boxes for p in proposals]
            prev_pred_scores = self.box_predictor.predict_probs(predictions, proposals)
            prev_pred_scores = [prev_pred_score.detach() for prev_pred_score in prev_pred_scores]
            prev_pred_boxes = self.box_predictor.predict_boxes(predictions, proposals)
            self._vis_prediction(
                prev_pred_boxes,
                prev_pred_scores,
                proposals,
                top_k=100,
                prefix="train",
                suffix="_mil",
            )
            for k in range(self.refine_K):
                suffix = "_r" + str(k)
                term_weight = 1
                if self.refine_mist:
                    targets = self.get_pgt_mist(prev_pred_boxes, prev_pred_scores, proposals)
                    self._vis_pgt(targets, "pgt_mist", suffix)
                    if k == 0:
                        term_weight = 3
                else:
                    targets = self.get_pgt_top_k(prev_pred_boxes, prev_pred_scores, proposals)
                    self._vis_pgt(targets, "pgt_top_k", suffix)

                proposals_k = self.label_and_sample_proposals(proposals, targets, suffix=suffix)

                predictions_k = self.box_refinery[k](box_features)

                losses_k = self.box_refinery[k].losses(predictions_k, proposals_k)
                for loss_name in losses_k.keys():
                    losses_k[loss_name] = losses_k[loss_name] * term_weight

                prev_pred_scores = self.box_refinery[k].predict_probs(predictions_k, proposals_k)
                prev_pred_boxes = self.box_refinery[k].predict_boxes(predictions_k, proposals_k)
                prev_pred_scores = [
                    prev_pred_score.detach() for prev_pred_score in prev_pred_scores
                ]
                prev_pred_boxes = [prev_pred_box.detach() for prev_pred_box in prev_pred_boxes]

                self._vis_prediction(
                    prev_pred_boxes,
                    prev_pred_scores,
                    proposals,
                    top_k=100,
                    prefix="train",
                    suffix=suffix,
                )

                losses.update(losses_k)

            self.prev_pred_boxes = prev_pred_boxes
            self.prev_pred_scores = prev_pred_scores

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            if self.refine_reg[-1] and False:
                predictions_k = self.box_refinery[-1](box_features)
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_k, proposals
                )
            else:
                predictions_K = []
                for k in range(self.refine_K):
                    predictions_k = self.box_refinery[k](box_features)
                    predictions_K.append(predictions_k)
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K, proposals
                )
            return pred_instances, all_scores, all_boxes

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            if False:
                # -------------------------------------------------------
                suffix = "_mask_10"
                # targets = self.get_pgt(
                # self.prev_pred_boxes, self.prev_pred_scores, instances, suffix
                # )
                # targets = self.get_grabcut(targets)
                targets = self.get_pgt_top_k(
                    self.prev_pred_boxes,
                    self.prev_pred_scores,
                    instances,
                    top_k=self.mask_mined_top_k,
                    need_mask=True,
                    suffix=suffix,
                )

                instances = self.label_and_sample_proposals(instances, targets, suffix=suffix)

                suffix = "_mask_1"
                # targets = self.get_pgt(
                # self.prev_pred_boxes, self.prev_pred_scores, instances, suffix
                # )
                # targets = self.get_grabcut(targets)
                targets = self.get_pgt_top_k(
                    self.prev_pred_boxes,
                    self.prev_pred_scores,
                    instances,
                    top_k=1,
                    need_mask=False,
                    suffix=suffix,
                )

                instances, matched_idxs = self.label_and_sample_proposals(
                    instances, targets, ret_MI=True, suffix=suffix
                )
                # -------------------------------------------------------

                # The loss is only defined on positive proposals.
                proposals, _ = select_foreground_proposals(instances, self.num_classes)
            else:
                suffix = "_mask_1"
                targets = self.get_pgt_top_k(
                    self.prev_pred_boxes,
                    self.prev_pred_scores,
                    instances,
                    top_k=1,
                    need_mask=False,
                    suffix=suffix,
                )

                instances = self.label_and_sample_proposals(instances, targets, suffix=suffix)

                # The loss is only defined on positive proposals.
                proposals, _ = select_foreground_proposals(instances, self.num_classes)

                suffix = "_mask_10"
                ious = [
                    pairwise_iou(t.gt_boxes, p.proposal_boxes) for t, p in zip(targets, proposals)
                ]
                num_preds = [len(p) for p in proposals]
                top_ks = [min(num_pred, self.mask_mined_top_k) for num_pred in num_preds]
                near_ious_idxs = [torch.topk(iou, top_k, dim=1) for iou, top_k in zip(ious, top_ks)]
                # near_ious = [item[0] for item in near_ious_idxs]
                near_idxs = [item[1].view(-1) for item in near_ious_idxs]

                near_pgt_boxes = [
                    torch.index_select(p.proposal_boxes.tensor, 0, near_idx)
                    for p, near_idx in zip(proposals, near_idxs)
                ]
                near_pgt_classes = [
                    torch.index_select(p.gt_classes, 0, near_idx)
                    for p, near_idx in zip(proposals, near_idxs)
                ]
                near_pgt_scores = [
                    torch.index_select(p.gt_scores, 0, near_idx)
                    for p, near_idx in zip(proposals, near_idxs)
                ]

                near_pgt_boxes = [Boxes(near_pgt_box) for near_pgt_box in near_pgt_boxes]

                near_targets = [
                    Instances(
                        proposals[i].image_size,
                        gt_boxes=near_pgt_box,
                        gt_classes=near_pgt_class,
                        gt_scores=near_pgt_score,
                    )
                    for i, (near_pgt_box, near_pgt_class, near_pgt_score) in enumerate(
                        zip(near_pgt_boxes, near_pgt_classes, near_pgt_scores)
                    )
                ]

                near_targets = self.get_grabcut_process(near_targets)
                self._vis_pgt(near_targets, "pgt_top_k", suffix)

                instances = self.label_and_sample_proposals(instances, near_targets, suffix=suffix)

                instances = self.label_and_sample_proposals(instances, targets, suffix=suffix)

            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        # return self.mask_head(features, instances)
        pred_mask_logits, pred_features = self.mask_head.layers(features)

        # torch.cuda.empty_cache()

        if self.training:
            losses = {"loss_mask": mask_rcnn_loss(pred_mask_logits, instances, self.vis_period)}
            losses["loss_mask_co"] = mask_rcnn_co_loss(pred_mask_logits, instances, self.vis_period)

            for k in range(len(self.mask_refinery)):
                suffix = "_r" + str(k)
                instances = self.get_pgt_mask(pred_mask_logits, instances, suffix)
                # losses_k = self.mask_refinery[k](features, instances)
                # losses_k = self.mask_refinery[k](pred_features, instances)
                pred_mask_logits = self.mask_refinery[k].layers(pred_features)
                losses["loss_mask" + suffix] = mask_rcnn_loss(
                    pred_mask_logits, instances, self.vis_period
                )
                losses["loss_mask_co" + suffix] = mask_rcnn_co_loss(
                    pred_mask_logits, instances, self.vis_period
                )
            return losses
        else:
            if True:
                pred_mask_logits_all = None
                for k in range(len(self.mask_refinery)):
                    pred_mask_logits = self.mask_refinery[k].layers(pred_features)
                    if pred_mask_logits_all is not None:
                        pred_mask_logits_all += pred_mask_logits
                    else:
                        pred_mask_logits_all = pred_mask_logits

                mask_rcnn_inference(pred_mask_logits_all / len(self.mask_refinery), instances)
                return instances

            return self.mask_refinery[-1](pred_features, instances)

    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        return self.keypoint_head(features, instances)

    @torch.no_grad()
    def get_pgt_mist(self, prev_pred_boxes, prev_pred_scores, proposals, top_pro=0.15):
        pgt_scores, pgt_boxes, pgt_classes, pgt_weights = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_pro,
            # thres=0.05,
            thres=0.0,
            need_instance=False,
            need_weight=True,
        )

        # NMS
        pgt_idxs = [torch.zeros_like(pgt_class) for pgt_class in pgt_classes]
        keeps = [
            batched_nms(pgt_box, pgt_score, pgt_class, 0.2)
            for pgt_box, pgt_score, pgt_class in zip(pgt_boxes, pgt_scores, pgt_idxs)
        ]
        pgt_scores = [pgt_score[keep] for pgt_score, keep in zip(pgt_scores, keeps)]
        pgt_boxes = [pgt_box[keep] for pgt_box, keep in zip(pgt_boxes, keeps)]
        pgt_classes = [pgt_class[keep] for pgt_class, keep in zip(pgt_classes, keeps)]
        pgt_weights = [pgt_weight[keep] for pgt_weight, keep in zip(pgt_weights, keeps)]

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        targets = [
            Instances(
                proposals[i].image_size,
                gt_boxes=pgt_box,
                gt_classes=pgt_class,
                gt_scores=pgt_score,
                gt_weights=pgt_weight,
            )
            for i, (pgt_box, pgt_class, pgt_score, pgt_weight) in enumerate(
                zip(pgt_boxes, pgt_classes, pgt_scores, pgt_scores)
            )
        ]

        return targets

    @torch.no_grad()
    def get_pgt_top_k(
        self,
        prev_pred_boxes,
        prev_pred_scores,
        proposals,
        top_k=1,
        thres=0,
        need_instance=True,
        need_weight=True,
        need_mask=False,
    ):
        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)
        # if isinstance(prev_pred_boxes[0], Boxes):
        #     num_preds = [len(prev_pred_box) for prev_pred_box in prev_pred_boxes]
        #     prev_pred_boxes = [
        #         prev_pred_box.tensor.unsqueeze(1).expand(num_pred, self.num_classes, 4)
        #         for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
        #     ]
        # else:
        #     assert isinstance(prev_pred_boxes[0], torch.Tensor)
        #     if self.cls_agnostic_bbox_reg:
        #         num_preds = [prev_pred_box.size(0) for prev_pred_box in prev_pred_boxes]
        #         prev_pred_boxes = [
        #             prev_pred_box.unsqueeze(1).expand(num_pred, self.num_classes, 4)
        #             for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
        #         ]
        # prev_pred_boxes = [
        #     prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
        # ]

        assert isinstance(prev_pred_boxes[0], torch.Tensor)
        num_preds = [prev_pred_box.size(0) for prev_pred_box in prev_pred_boxes]
        if prev_pred_boxes[0].size(1) == 4:
            prev_pred_boxes = [
                prev_pred_box.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
            ]
        else:
            assert (prev_pred_boxes[0].size(1) == self.num_classes * 4) or (
                prev_pred_boxes[0].size(1) == self.num_classes and prev_pred_boxes[0].size(2) == 4
            )
            prev_pred_boxes = [
                prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
            ]

        if isinstance(prev_pred_scores, torch.Tensor):
            num_preds_per_image = [len(p) for p in proposals]
            prev_pred_scores = prev_pred_scores.split(num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, tuple) or isinstance(
                prev_pred_scores, list
            ), prev_pred_scores
            assert isinstance(prev_pred_scores[0], torch.Tensor), prev_pred_scores[0]

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        prev_pred_boxes = [
            torch.index_select(prev_pred_box, 1, gt_int)
            for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
        ]

        # get top k
        num_preds = [prev_pred_score.size(0) for prev_pred_score in prev_pred_scores]
        if top_k >= 1:
            top_ks = [min(num_pred, int(top_k)) for num_pred in num_preds]
        elif top_k < 1 and top_k > 0:
            top_ks = [max(int(num_pred * top_k), 1) for num_pred in num_preds]
        else:
            top_ks = [min(num_pred, 1) for num_pred in num_preds]
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]
        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, gt_int.numel(), 4)
            for pgt_idx, top_k, gt_int in zip(pgt_idxs, top_ks, self.gt_classes_img_int)
        ]
        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]
        pgt_classes = [
            torch.unsqueeze(gt_int, 0).expand(top_k, gt_int.numel())
            for gt_int, top_k in zip(self.gt_classes_img_int, top_ks)
        ]
        if need_weight:
            pgt_weights = [
                torch.index_select(pred_logits, 1, gt_int).expand(top_k, gt_int.numel())
                for pred_logits, gt_int, top_k in zip(
                    self.pred_class_img_logits.split(1, dim=0), self.gt_classes_img_int, top_ks
                )
            ]

        if thres > 0:
            # get large scores
            masks = [pgt_score.ge(thres) for pgt_score in pgt_scores]
            masks = [
                torch.cat([torch.full_like(mask[0:1, :], True), mask[1:, :]], dim=0)
                for mask in masks
            ]
            pgt_scores = [
                torch.masked_select(pgt_score, mask) for pgt_score, mask in zip(pgt_scores, masks)
            ]
            pgt_boxes = [
                torch.masked_select(
                    pgt_box, torch.unsqueeze(mask, 2).expand(top_k, gt_int.numel(), 4)
                )
                for pgt_box, mask, top_k, gt_int in zip(
                    pgt_boxes, masks, top_ks, self.gt_classes_img_int
                )
            ]
            pgt_classes = [
                torch.masked_select(pgt_class, mask) for pgt_class, mask in zip(pgt_classes, masks)
            ]
            if need_weight:
                pgt_weights = [
                    torch.masked_select(pgt_weight, mask)
                    for pgt_weight, mask in zip(pgt_weights, masks)
                ]

        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_boxes = [pgt_box.reshape(-1, 4) for pgt_box in pgt_boxes]
        pgt_classes = [pgt_class.reshape(-1) for pgt_class in pgt_classes]
        if need_weight:
            pgt_weights = [pgt_weight.reshape(-1) for pgt_weight in pgt_weights]

            pgt_weights = [
                pgt_weight
                if pgt_weight.numel() > 0
                else torch.tensor([1], dtype=pgt_weight.dtype, device=pgt_weight.device)
                for pgt_weight in pgt_weights
            ]

        pgt_scores = [
            pgt_score
            if pgt_score.numel() > 0
            else torch.tensor([1], dtype=pgt_score.dtype, device=pgt_score.device)
            for pgt_score in pgt_scores
        ]
        pgt_boxes = [
            pgt_box
            if pgt_box.numel() > 0
            else torch.tensor(
                [[-10000, -10000, 10000, 10000]], dtype=pgt_box.dtype, device=pgt_box.device
            )
            for pgt_box in pgt_boxes
        ]
        pgt_classes = [
            pgt_class
            if pgt_class.numel() > 0
            else torch.tensor([0], dtype=pgt_class.dtype, device=pgt_class.device)
            for pgt_class in pgt_classes
        ]

        if not need_instance:
            if need_weight:
                return pgt_scores, pgt_boxes, pgt_classes, pgt_weights
            else:
                return pgt_scores, pgt_boxes, pgt_classes

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        if need_weight:
            targets = [
                Instances(
                    proposals[i].image_size,
                    gt_boxes=pgt_box,
                    gt_classes=pgt_class,
                    gt_scores=pgt_score,
                    gt_weights=pgt_weight,
                )
                for i, (pgt_box, pgt_class, pgt_score, pgt_weight) in enumerate(
                    zip(pgt_boxes, pgt_classes, pgt_scores, pgt_weights)
                )
            ]
        else:
            targets = [
                Instances(
                    proposals[i].image_size,
                    gt_boxes=pgt_box,
                    gt_classes=pgt_class,
                    gt_scores=pgt_score,
                )
                for i, (pgt_box, pgt_class, pgt_score) in enumerate(
                    zip(pgt_boxes, pgt_classes, pgt_scores)
                )
            ]

        if need_mask:
            targets = self.get_grabcut_process(targets)

        return targets

    @torch.no_grad()
    def _vis_pgt(self, targets, prefix, suffix):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return
        storage = get_event_storage()

        targets = copy.deepcopy(targets)

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, target in enumerate(targets):
            img = self.images.tensor[b, ...].clone().detach() * self.pixel_std + self.pixel_mean
            img = img.cpu().numpy().transpose((1, 2, 0))
            img = img.astype(np.uint8)
            h, w = img.shape[:2]
            img = img.copy()

            if target.has("gt_scores"):
                target.scores = target.gt_scores.clone().detach().cpu()
            if target.has("gt_boxes"):
                target.pred_boxes = target.gt_boxes.tensor.clone().detach().cpu()
            if target.has("gt_classes"):
                target.pred_classes = target.gt_classes.clone().detach().cpu()

            vis = Visualizer(img, self.metadata)
            # vis._default_font_size = max(
            #     np.sqrt(h * w) // 20, 20
            # )
            vis_pred = vis.draw_instance_predictions(target).get_image()

            device_index = comm.get_rank()
            save_name = (
                "i" + str(self.iter) + "_g" + str(device_index) + "_b" + str(b) + suffix + ".png"
            )

            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, vis_pred)

            vis_pred = vis_pred.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix
            storage.put_image(vis_name, vis_pred)
        return
        pgt_boxes = [target.gt_boxes for target in targets]
        pgt_classes = [target.gt_classes for target in targets]
        pgt_scores = [target.gt_scores for target in targets]
        if targets[0].has("pgt_bitmasks"):
            pgt_masks = [target.pgt_bitmasks for target in targets]
        elif targets[0].has("gt_masks"):
            pgt_masks = [target.gt_masks for target in targets]
        else:
            pgt_masks = [None for target in targets]

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, (pgt_box, pgt_class, pgt_score, pgt_mask) in enumerate(
            zip(pgt_boxes, pgt_classes, pgt_scores, pgt_masks)
        ):
            img = self.images.tensor[b, ...].clone().detach().cpu().numpy()
            channel_swap = (1, 2, 0)
            img = img.transpose(channel_swap)
            pixel_means = [103.939, 116.779, 123.68]
            img += pixel_means
            img = img.astype(np.uint8)
            h, w = img.shape[:2]
            img_pgt = img.copy()

            device_index = pgt_box.device.index
            save_name = (
                "i" + str(self.iter) + "_g" + str(device_index) + "_b" + str(b) + suffix + ".png"
            )
            pgt_box = pgt_box.tensor.clone().detach().cpu().numpy()
            pgt_class = pgt_class.clone().detach().cpu().numpy()
            pgt_score = pgt_score.clone().detach().cpu().numpy()
            if pgt_mask is not None:
                pgt_mask = pgt_mask.tensor.clone().detach().cpu().numpy()
            for i in range(pgt_box.shape[0]):
                c = pgt_class[i]
                s = pgt_score[i]
                x0, y0, x1, y1 = pgt_box[i, :]
                x0 = int(max(x0, 0))
                y0 = int(max(y0, 0))
                x1 = int(min(x1, w))
                y1 = int(min(y1, h))
                cv2.rectangle(img_pgt, (x0, y0), (x1, y1), (0, 0, 255), 8)

                fontFace = cv2.FONT_HERSHEY_COMPLEX
                fontScale = 1
                thickness = 2
                (tw, th), bl = cv2.getTextSize(
                    self.metadata.thing_classes[c], fontFace, fontScale, thickness
                )
                cv2.putText(
                    img_pgt,
                    self.metadata.thing_classes[c],
                    (x0, y0 + th),
                    fontFace,
                    fontScale,
                    (255, 0, 0),
                    thickness,
                )
                (_, th), bl = cv2.getTextSize(str(s), fontFace, fontScale, thickness)
                cv2.putText(
                    img_pgt, str(s), (x0 + tw, y0 + th), fontFace, fontScale, (0, 255, 0), thickness
                )

                if pgt_mask is not None:
                    m = pgt_mask[i]
                    img_pgt_m = img.copy()
                    img_pgt_m = img_pgt_m * m[:, :, np.newaxis]
                    img_pgt = np.concatenate([img_pgt, img_pgt_m], axis=1)

            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, img_pgt)

            img_pgt = img_pgt.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix
            storage.put_image(vis_name, img_pgt)

    @torch.no_grad()
    def _vis_prediction(
        self,
        prev_pred_boxes,
        prev_pred_scores,
        proposals,
        top_k=1,
        thres=0.01,
        prefix="",
        suffix="",
    ):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return

        targets = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_k,
            thres=thres,
            need_weight=False,
        )

        self._vis_pgt(targets, prefix, suffix)

    @torch.no_grad()
    def _vis_proposal(self, proposals, prefix, suffix):
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return

        prev_pred_boxes = [p.proposal_boxes for p in proposals]
        num_preds = [len(prev_pred_box) for prev_pred_box in proposals]
        prev_pred_boxes = [
            prev_pred_box.tensor.unsqueeze(1).expand(num_pred, self.num_classes, 4)
            for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
        ]

        prev_pred_scores = [p.objectness_logits for p in proposals]
        prev_pred_scores = [
            prev_pred_score.unsqueeze(1).expand(num_pred, self.num_classes + 1)
            for num_pred, prev_pred_score in zip(num_preds, prev_pred_scores)
        ]

        self._vis_prediction(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=2048,
            thres=-9999,
            prefix=prefix,
            suffix=suffix,
        )

    @torch.no_grad()
    def _save_proposal_test(self, proposals, prefix, suffix):
        if self.training or not self.vis_test:
            return

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter_test == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, p in enumerate(proposals):
            box = p.proposal_boxes.tensor.clone().detach().cpu().numpy()
            logit = p.objectness_logits.clone().detach().cpu().numpy()
            level_ids = p.level_ids.clone().detach().cpu().numpy()

            gpu_id = p.objectness_logits.device.index
            id_str = "i" + str(self.iter_test) + "_g" + str(gpu_id) + "_b" + str(b)

            save_path = os.path.join(output_dir, id_str + "_box" + suffix + ".npy")
            np.save(save_path, box)

            save_path = os.path.join(output_dir, id_str + "_logit" + suffix + ".npy")
            np.save(save_path, logit)

            save_path = os.path.join(output_dir, id_str + "_level" + suffix + ".npy")
            np.save(save_path, level_ids)

    @torch.no_grad()
    def get_grabcut(self, instances):
        max_len = 500

        imgs = self.img_normalizer(self.images.tensor)

        imgs_h, imgs_w = imgs.shape[2:]
        scale_factor = 1.0 * max_len / max(imgs_h, imgs_w)
        imgs = F.interpolate(
            imgs,
            size=(int(scale_factor * imgs_h), int(scale_factor * imgs_w)),
            mode="bilinear",
            align_corners=False,
        )

        imgs = imgs.cpu().numpy().transpose((0, 2, 3, 1))
        imgs = imgs.astype(np.uint8)

        for b, instance in enumerate(instances):
            time_start = time.time()

            img = imgs[b, ...]
            img_h, img_w = img.shape[:2]

            if self.training:
                instance_classes = instance.gt_classes
                instance_boxes = instance.gt_boxes
            else:
                instance_classes = instance.pred_classes
                instance_boxes = instance.pred_boxes

            instance_classes_int = instance_classes.to(dtype=torch.int64)
            boxes = instance_boxes.tensor
            N = boxes.shape[0]

            instance_masks = torch.zeros(N, 1, img_h, img_w, device=boxes.device, dtype=boxes.dtype)

            for j in range(N):
                box = boxes[j] * scale_factor
                box = box.clone().to(dtype=torch.int64).cpu().numpy()

                mask = np.zeros(img.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)

                width = box[2] - box[0] + 1
                height = box[3] - box[1] + 1
                if width * height < self._MIN_AREA:
                    assert width * height > 0
                    img_mask = rect_mask(img.shape[:2], box)
                else:
                    if width * height >= img.shape[0] * img.shape[1]:
                        rect = (
                            self._RECT_SHRINK,
                            self._RECT_SHRINK,
                            width - self._RECT_SHRINK * 2,
                            height - self._RECT_SHRINK * 2,
                        )
                    else:
                        rect = (box[0], box[1], width, height)
                    cv2.grabCut(
                        img,
                        mask,
                        rect,
                        bgdModel,
                        fgdModel,
                        self.grabcut_iter,
                        cv2.GC_INIT_WITH_RECT,
                    )

                    # img_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                    img_mask = np.where((mask == 2) | (mask == 0), 0, 1)
                img_mask = torch.from_numpy(img_mask).to(self.device)
                instance_masks[j, ...] = img_mask

            instance_masks = F.interpolate(
                instance_masks, size=(imgs_h, imgs_w), mode="bilinear", align_corners=False
            )

            if self.training:
                instance_masks = (instance_masks >= 0.5).to(dtype=torch.bool)

                # bit mask
                # instance.gt_masks = BitMasks(instance_masks[:, 0, :, :])
                # instance.gt_masks = instance_masks

                # poly mask
                instance.pgt_bitmasks = BitMasks(instance_masks[:, 0, :, :])
                gt_polys = []
                for j in range(N):
                    instance_mask = instance_masks[j, 0, :, :].clone().cpu().numpy()
                    polys = binary_mask_to_polygon(instance_mask)
                    if len(polys) == 0:
                        box = boxes[j] * scale_factor
                        box = box.clone().to(dtype=torch.int64).cpu().numpy()
                        poly = []
                        for i in range(box[0] - 1, box[2] + 1):
                            poly.extend([i, box[1]])
                        for i in range(box[1] - 1, box[3] + 1):
                            poly.extend([box[2], i])
                        for i in range(box[2] + 1, box[0] - 1, -1):
                            poly.extend([i, box[3]])
                        for i in range(box[3] + 1, box[1] - 1, -1):
                            poly.extend([box[0], i])
                        polys.append(poly)
                    gt_polys.append(polys)
                gt_masks = PolygonMasks(gt_polys)
                instance.gt_masks = gt_masks
            else:
                instance.pred_masks = instance_masks
                instance.no_paste = (instance_classes_int[:] >= 0).to(dtype=torch.bool)

            time_end = time.time()
            if False:
                print(
                    "Grabcut takes ",
                    time_end - time_start,
                    " seconds for ",
                    N,
                    " boxes with image size ",
                    img.shape,
                )

        return instances

    @torch.no_grad()
    def get_grabcut_process(self, instances):
        time_start = time.time()
        max_len = 400

        imgs = self.img_normalizer(self.images.tensor)

        imgs_h, imgs_w = imgs.shape[2:]
        scale_factor = 1.0 * max_len / max(imgs_h, imgs_w)
        imgs = F.interpolate(
            imgs,
            size=(int(scale_factor * imgs_h), int(scale_factor * imgs_w)),
            mode="bilinear",
            align_corners=False,
        )

        imgs = imgs.cpu().numpy().transpose((0, 2, 3, 1))
        imgs = imgs.astype(np.uint8)

        arg_process = []
        for b, instance in enumerate(instances):
            img = imgs[b, ...]

            if self.training:
                instance_classes = instance.gt_classes
                instance_boxes = instance.gt_boxes
            else:
                instance_classes = instance.pred_classes
                instance_boxes = instance.pred_boxes

            boxes = instance_boxes.tensor
            N = boxes.shape[0]

            for j in range(N):
                box = boxes[j] * scale_factor
                box = box.clone().to(dtype=torch.int64).cpu().numpy()

                arg_process.append((img, box))

        results = self.process_pool.starmap_async(do_grabcut, arg_process)
        img_masks = results.get()

        cnt = 0
        for b, instance in enumerate(instances):
            img = imgs[b, ...]
            img_h, img_w = img.shape[:2]

            if self.training:
                instance_classes = instance.gt_classes
                instance_boxes = instance.gt_boxes
            else:
                instance_classes = instance.pred_classes
                instance_boxes = instance.pred_boxes

            instance_classes_int = instance_classes.to(dtype=torch.int64)
            boxes = instance_boxes.tensor
            N = boxes.shape[0]

            instance_masks = torch.zeros(N, 1, img_h, img_w, device=boxes.device, dtype=boxes.dtype)

            for j in range(N):
                img_mask = img_masks[cnt]
                img_mask = torch.from_numpy(img_mask).to(self.device)
                instance_masks[j, ...] = img_mask
                cnt += 1

            instance_masks = F.interpolate(
                instance_masks, size=(imgs_h, imgs_w), mode="bilinear", align_corners=False
            )

            if self.training:
                instance_masks = (instance_masks >= 0.5).to(dtype=torch.bool)

                # bit mask
                # instance.gt_masks = BitMasks(instance_masks[:, 0, :, :])
                # instance.gt_masks = instance_masks

                # poly mask
                instance.pgt_bitmasks = BitMasks(instance_masks[:, 0, :, :])
                instance.pgt_masks = instance_masks[:, 0, :, :]
                gt_polys = []
                for j in range(N):
                    instance_mask = instance_masks[j, 0, :, :].clone().cpu().numpy()
                    polys = binary_mask_to_polygon2(instance_mask, boxes[j])
                    gt_polys.append(polys)
                gt_masks = PolygonMasks(gt_polys)
                instance.gt_masks = gt_masks
            else:
                instance.pred_masks = instance_masks
                instance.no_paste = (instance_classes_int[:] >= 0).to(dtype=torch.bool)

        time_end = time.time()
        if True:
            print(
                "Grabcut takes ",
                time_end - time_start,
                " seconds for ",
                cnt,
                " boxes with image size ",
                img.shape,
            )

        return instances

    @torch.no_grad()
    def get_pgt_mask(self, pred_mask_logits, proposals, suffix, mask_threshold=0.5):
        for proposal in proposals:
            proposal.pred_classes = proposal.gt_classes

        mask_rcnn_inference(pred_mask_logits, proposals)

        for proposal in proposals:

            proposal.pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
                proposal.pred_masks[:, 0, :, :],  # N, 1, M, M
                proposal.proposal_boxes.tensor,
                proposal.image_size,
                threshold=mask_threshold,
            )

            pred_masks = proposal.pred_masks.cpu().numpy()
            boxes = proposal.proposal_boxes.tensor

            gt_polys = []
            for j in range(len(proposal)):
                polys = binary_mask_to_polygon2(pred_masks[j], boxes[j])
                gt_polys.append(polys)
            gt_masks = PolygonMasks(gt_polys)
            proposal.gt_masks = gt_masks

        return proposals
