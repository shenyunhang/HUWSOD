# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import inspect
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn

import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer
from wsl.layers import ROILabel
from wsl.modeling.poolers import ROIPooler
from wsl.modeling.roi_heads.fast_rcnn_oicr import OICROutputLayers
from wsl.modeling.roi_heads.fast_rcnn_wsddn import WSDDNOutputLayers
from wsl.modeling.roi_heads.roi_heads import (
    ROIHeads,
    get_image_level_gt,
    select_foreground_proposals,
    select_proposals_with_visible_keypoints,
)

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class HUWSODROIHeads(ROIHeads):
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
        mrrp_on: bool = False,
        mrrp_num_branch: int = 3,
        mrrp_fast: bool = False,
        refine_K: int = 4,
        refine_mist: bool = False,
        refine_reg: List[bool] = [True, True, True, True],
        box_refinery: List[nn.Module] = [None, None, None, None],
        cls_agnostic_bbox_reg: bool = False,
        sampling_on: bool = False,
        proposal_matchers: List[Matcher] = [None, None, None, None],
        batch_size_per_images: List[int] = [512, 512, 512, 512],
        positive_sample_fractions: List[float] = [0.25, 0.25, 0.25, 0.25],
        cls_agnostic_bbox_known: bool = False,
        pooler_type: str = "ROIPool",
        roi_label: Optional[nn.Module] = None,
        rpn_on: bool = False,
        branch_supervision: List[int] = [-1, -1, -1, -1],
        rpn_supervision: List[int] = [-1],
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

        self.mrrp_on = mrrp_on
        self.mrrp_num_branch = mrrp_num_branch
        self.mrrp_fast = mrrp_fast

        self.refine_K = refine_K
        self.refine_mist = refine_mist
        self.refine_reg = refine_reg
        self.box_refinery = box_refinery
        for k in range(self.refine_K):
            self.add_module("box_refinery_{}".format(k), self.box_refinery[k])
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

        self.sampling_on = sampling_on
        self.proposal_matchers = proposal_matchers
        self.batch_size_per_images = batch_size_per_images
        self.positive_sample_fractions = positive_sample_fractions

        self.cls_agnostic_bbox_known = cls_agnostic_bbox_known
        self.roi_label = roi_label
        self.pooler_type = pooler_type

        self.rpn_on = rpn_on

        self.branch_supervision = branch_supervision
        assert len(branch_supervision) % refine_K == 0, "{} {}".format(branch_supervision, refine_K)

        self.rpn_supervision = rpn_supervision

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

        mrrp_on = cfg.MODEL.MRRP.MRRP_ON
        mrrp_num_branch = cfg.MODEL.MRRP.NUM_BRANCH
        mrrp_fast = cfg.MODEL.MRRP.TEST_BRANCH_IDX != -1
        if mrrp_on:
            pooler_scales = tuple(
                1.0 / input_shape[k].stride for k in in_features * mrrp_num_branch
            )

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

        sampling_on = cfg.WSL.SAMPLING.SAMPLING_ON
        proposal_matchers = [None for _ in range(refine_K)]
        if sampling_on:
            for k in range(refine_K):
                # Matcher to assign box proposals to gt boxes
                proposal_matchers[k] = Matcher(
                    cfg.WSL.SAMPLING.IOU_THRESHOLDS[k],
                    cfg.WSL.SAMPLING.IOU_LABELS[k],
                    allow_low_quality_matches=False,
                )
        batch_size_per_images = cfg.WSL.SAMPLING.BATCH_SIZE_PER_IMAGE
        positive_sample_fractions = cfg.WSL.SAMPLING.POSITIVE_FRACTION

        cls_agnostic_bbox_known = cfg.WSL.CLS_AGNOSTIC_BBOX_KNOWN

        output_dir = cfg.OUTPUT_DIR
        vis_test = cfg.VIS_TEST
        vis_period = cfg.VIS_PERIOD

        roi_label = None
        if cfg.WSL.CMIL:
            roi_label = ROILabel(0.6, 0.4, 0.1, 32, 96, 1, 0, int(1280 / cfg.SOLVER.IMS_PER_BATCH))

        rpn_on = False if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "PrecomputedProposals" else True

        branch_supervision = cfg.WSL.HUWSOD.BRANCH_SUPERVISION
        rpn_supervision = cfg.WSL.HUWSOD.RPN_SUPERVISION

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "output_dir": output_dir,
            "vis_test": vis_test,
            "vis_period": vis_period,
            "mrrp_on": mrrp_on,
            "mrrp_num_branch": mrrp_num_branch,
            "mrrp_fast": mrrp_fast,
            "refine_K": refine_K,
            "refine_mist": refine_mist,
            "refine_reg": refine_reg,
            "box_refinery": box_refinery,
            "cls_agnostic_bbox_reg": cls_agnostic_bbox_reg,
            "sampling_on": sampling_on,
            "proposal_matchers": proposal_matchers,
            "batch_size_per_images": batch_size_per_images,
            "positive_sample_fractions": positive_sample_fractions,
            "cls_agnostic_bbox_known": cls_agnostic_bbox_known,
            "pooler_type": pooler_type,
            "roi_label": roi_label,
            "rpn_on": rpn_on,
            "branch_supervision": branch_supervision,
            "rpn_supervision": rpn_supervision,
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
        ret["mask_head"] = build_mask_head(cfg, shape)
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
        prev_pred_scores_list: List = None,
        prev_pred_boxes_list: List = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        if self.mrrp_on and False:
            # Use 1 branch if using mrrp_fast during inference.
            num_branch = self.mrrp_num_branch if self.training or not self.mrrp_fast else 1
            # Duplicate images for all branches in MRRPNet.
            all_images = ImageList(
                torch.cat([images.tensor] * num_branch), images.image_sizes * num_branch
            )
            # Duplicate targets for all branches in MRRP
            all_targets = targets * num_branch if targets is not None else None

            images = all_images
            targets = all_targets

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
            assert prev_pred_scores_list is not None
            assert prev_pred_boxes_list is not None
            self.prev_pred_scores_list = prev_pred_scores_list
            self.prev_pred_boxes_list = prev_pred_boxes_list
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

            return proposals, losses, self.prev_pred_scores_list, self.prev_pred_boxes_list
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

        if self.mrrp_on:
            features = [torch.chunk(f, self.mrrp_num_branch) for f in features]
            features = [ff for f in features for ff in f]

        if self.rpn_on:
            box_features = self.box_pooler(
                features,
                [x.proposal_boxes for x in proposals],
                level_ids=[x.level_ids // 1000 for x in proposals],
            )
        else:
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        if self.pooler_type == "ROILoopPool":
            objectness_logits = torch.cat(
                [objectness_logits, objectness_logits, objectness_logits], dim=0
            )
        if self.rpn_on:
            box_features = box_features * torch.sigmoid(objectness_logits.view(-1, 1, 1, 1))
        else:
            box_features = box_features * objectness_logits.view(-1, 1, 1, 1)
        if self.training and objectness_logits.numel() > 0:
            storage = get_event_storage()
            storage.put_scalar("proposals/objectness_logits+1 mean", objectness_logits.mean())
            storage.put_scalar("proposals/objectness_logits+1 max", objectness_logits.max())
            storage.put_scalar("proposals/objectness_logits+1 min", objectness_logits.min())

        # torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        if self.pooler_type == "ROILoopPool":
            box_features, box_features_frame, box_features_context = torch.chunk(
                box_features, 3, dim=0
            )
            predictions = self.box_predictor(
                [box_features, box_features_frame, box_features_context], proposals, context=True
            )
            del box_features_frame
            del box_features_context
        else:
            predictions = self.box_predictor(box_features, proposals)
        # del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals, self.gt_classes_img_int)

            self.pred_class_img_logits = (
                self.box_predictor.predict_probs_img(predictions, proposals).clone().detach()
            )

            prev_pred_scores = self.box_predictor.predict_probs(predictions, proposals)
            prev_pred_scores = [prev_pred_score.detach() for prev_pred_score in prev_pred_scores]
            prev_pred_boxes = self.box_predictor.predict_boxes(predictions, proposals)

            self.prev_pred_scores_list.append(prev_pred_scores)
            self.prev_pred_boxes_list.append(prev_pred_boxes)

            self._vis_prediction(
                prev_pred_boxes,
                prev_pred_scores,
                proposals,
                top_k=100,
                prefix="train",
                suffix="_mil",
            )

            predictions_K = []
            for k in range(self.refine_K):

                prev_pred_idx = self.branch_supervision[
                    k + self.refine_K * self.iter % len(self.branch_supervision)
                ]
                prev_pred_scores = self.prev_pred_scores_list[prev_pred_idx]
                prev_pred_boxes = self.prev_pred_boxes_list[prev_pred_idx]

                suffix = "_r" + str(k)
                term_weight = 1
                if self.refine_mist:
                    targets = self.get_pgt_mist(prev_pred_boxes, prev_pred_scores, proposals)
                    self._vis_pgt(targets, "pgt_mist", suffix)
                    if k == 0:
                        term_weight = 1
                else:
                    targets = self.get_pgt_top_k(prev_pred_boxes, prev_pred_scores, proposals)
                    self._vis_pgt(targets, "pgt_top_k", suffix)

                if self.sampling_on:
                    proposals_k = self.label_and_sample_proposals_wsl(
                        k, proposals, targets, suffix=suffix
                    )
                else:
                    proposals_k = self.label_and_sample_proposals(proposals, targets, suffix=suffix)

                if self.roi_label and False:
                    if isinstance(prev_pred_scores, list):
                        S = torch.cat(prev_pred_scores, dim=0).cpu()
                    else:
                        S = prev_pred_scores.cpu()
                    U = torch.cat(
                        [pairwise_iou(p.proposal_boxes, p.proposal_boxes) for p in proposals_k],
                        dim=0,
                    ).cpu()
                    L = self.gt_classes_img_oh.cpu()
                    CW = self.pred_class_img_logits.cpu()

                    RL, RW = self.roi_label(S, U, L, CW)
                    RL = RL.to(self.pred_class_img_logits.device)
                    RW = RW.to(self.pred_class_img_logits.device)

                    num_preds_per_image = [len(p) for p in proposals_k]
                    for p, rl, rw in zip(
                        proposals_k,
                        RL.split(num_preds_per_image, dim=0),
                        RW.split(num_preds_per_image, dim=0),
                    ):
                        p.gt_classes = rl.to(torch.int64)
                        p.gt_weights = rw.to(torch.float32)

                predictions_k = self.box_refinery[k](box_features)
                predictions_K.append(predictions_k)

                losses_k = self.box_refinery[k].losses(predictions_k, proposals_k)
                for loss_name in losses_k.keys():
                    losses_k[loss_name] = losses_k[loss_name] * term_weight

                prev_pred_scores = self.box_refinery[k].predict_probs(predictions_k, proposals_k)
                prev_pred_boxes = self.box_refinery[k].predict_boxes(predictions_k, proposals_k)
                prev_pred_scores = [
                    prev_pred_score.detach() for prev_pred_score in prev_pred_scores
                ]
                prev_pred_boxes = [prev_pred_box.detach() for prev_pred_box in prev_pred_boxes]

                self.prev_pred_scores_list.append(prev_pred_scores)
                self.prev_pred_boxes_list.append(prev_pred_boxes)

                self._vis_prediction(
                    prev_pred_boxes,
                    prev_pred_scores,
                    proposals,
                    top_k=100,
                    prefix="train",
                    suffix=suffix,
                )

                losses.update(losses_k)

            if self.rpn_on:
                prev_pred_idx = self.rpn_supervision[self.iter % len(self.rpn_supervision)]
                prev_pred_scores = self.prev_pred_scores_list[prev_pred_idx]
                prev_pred_boxes = self.prev_pred_boxes_list[prev_pred_idx]

                if self.refine_mist and False:
                    targets = self.get_pgt_mist(prev_pred_boxes, prev_pred_scores, proposals)
                    self._vis_pgt(targets, "get_pgt_mist", "_rpn")
                else:
                    targets = self.get_pgt_top_k(prev_pred_boxes, prev_pred_scores, proposals)
                    self._vis_pgt(targets, "pgt_top_k", "_rpn")
                self.proposal_targets = targets

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

            # add pred_class_img_logits to results
            pred_class_img_logits = (
                self.box_predictor.predict_probs_img(predictions, proposals).clone().detach()
            )
            pred_class_img_logits = torch.chunk(pred_class_img_logits, len(pred_instances), dim=0)
            for prob, pred_instance in zip(pred_class_img_logits, pred_instances):
                # pred_instance.img_scores = prob.repeat(len(pred_instance), 1).squeeze()
                pred_instance.img_scores = prob.repeat(len(pred_instance), 1)
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
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)

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
    def get_pgt_mist(self, prev_pred_boxes, prev_pred_scores, proposals, top_pro=0.10):
        pgt_scores, pgt_boxes, pgt_classes, pgt_weights = self.get_pgt_top_k(
            prev_pred_boxes,
            prev_pred_scores,
            proposals,
            top_k=top_pro,
            thres=0.05,
            # thres=0.0,
            need_instance=False,
            need_weight=True,
        )

        # NMS
        # pgt_idxs = [torch.zeros_like(pgt_class) for pgt_class in pgt_classes]
        keeps = [
            batched_nms(pgt_box, pgt_score, pgt_class, 0.01)
            for pgt_box, pgt_score, pgt_class in zip(pgt_boxes, pgt_scores, pgt_classes)
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
                zip(pgt_boxes, pgt_classes, pgt_scores, pgt_weights)
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
    ):
        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)

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

    def _sample_proposals_wsl(
        self, k, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_images[k],
            self.positive_sample_fractions[k],
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)

        gt_classes_sp = torch.full_like(gt_classes, -1)
        gt_classes_sp[sampled_idxs] = gt_classes[sampled_idxs]

        sampled_idxs = torch.arange(gt_classes.shape[0])
        return sampled_idxs, gt_classes_sp[sampled_idxs]

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], suffix=""
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt and not self.cls_agnostic_bbox_known:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_scores"):
                proposals_per_image.gt_scores = targets_per_image.gt_scores[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_weights"):
                proposals_per_image.gt_weights = targets_per_image.gt_weights[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_masks"):
                proposals_per_image.gt_masks = targets_per_image.gt_masks[
                    matched_idxs[sampled_idxs]
                ]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        return proposals_with_gt

    @torch.no_grad()
    def label_and_sample_proposals_wsl(
        self, k: int, proposals: List[Instances], targets: List[Instances], suffix=""
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matchers[k](match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals_wsl(
                k, matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt and not self.cls_agnostic_bbox_known:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_scores"):
                proposals_per_image.gt_scores = targets_per_image.gt_scores[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_weights"):
                proposals_per_image.gt_weights = targets_per_image.gt_weights[
                    matched_idxs[sampled_idxs]
                ]
            if has_gt and targets_per_image.has("gt_masks"):
                proposals_per_image.gt_masks = targets_per_image.gt_masks[
                    matched_idxs[sampled_idxs]
                ]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        return proposals_with_gt
