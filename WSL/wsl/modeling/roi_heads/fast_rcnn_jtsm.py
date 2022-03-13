# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from wsl.data.detection_utils import get_fed_loss_cls_weights
from wsl.layers import ROIMerge
from wsl.modeling.box_regression import _dense_box_regression_loss

__all__ = ["fast_rcnn_inference", "JTSMOutputLayers"]

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    test_num_classes: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
            test_num_classes,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return (
        [x[0] for x in result_per_image],
        [x[1] for x in result_per_image],
        [x[2] for x in result_per_image],
        [x[3] for x in result_per_image],
    )


def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    pred_classes_top = torch.argsort(pred_logits, dim=1, descending=True)
    # bg_class_ind = pred_logits.shape[1] - 1
    bg_class_ind = pred_logits.shape[1]

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]
    fg_pred_classes_top = pred_classes_top[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()
    fg_num_accurate_top = (
        fg_pred_classes_top.eq(fg_gt_classes.unsqueeze(dim=1))
        .sum(dim=0)
        .cumsum(dim=0)
        .cpu()
        .tolist()
    )

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)
        top = 1
        while top < len(fg_num_accurate_top):
            storage.put_scalar(
                f"{prefix}/fg_cls_accuracy_top{top}", 1.0 * fg_num_accurate_top[top] / num_fg
            )
            top = top * 2


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    test_num_classes: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """

    all_scores = scores.clone()
    all_scores = torch.unsqueeze(all_scores, 0)
    all_boxes = boxes.clone()
    all_boxes = torch.unsqueeze(all_boxes, 0)

    pred_inds = torch.unsqueeze(
        torch.arange(scores.size(0), device=scores.device, dtype=torch.long), dim=1
    ).repeat(1, scores.size(1))

    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        pred_inds = pred_inds[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    pred_inds = pred_inds[:, :-1]

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    if test_num_classes > 0:
        filter_mask[:, test_num_classes:] = 0
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    pred_inds = pred_inds[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    pred_inds = pred_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.pred_inds = pred_inds
    return result, filter_inds[:, 0], all_scores, all_boxes


class JTSMOutputs(object):
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        mean_loss=False,
        gt_classes_img_oh=None,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # "gt_classes" exists if and only if training. But other gt fields may
            # not necessarily exist in training for images that have no groundtruth.
            if proposals[0].has("gt_classes"):
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = [
                    p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes for p in proposals
                ]
                self.gt_boxes = box_type.cat(gt_boxes)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(self.proposals) == 0  # no instances found

        self.mean_loss = mean_loss
        self.gt_classes_img_oh = gt_classes_img_oh

    def softmax_cross_entropy_loss(self):
        """
        Deprecated
        """
        _log_classification_stats(self.pred_class_logits, self.gt_classes)
        return cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def box_reg_loss(self):
        """
        Deprecated
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.proposals.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1
        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds should produce a valid loss of zero because reduction=sum.
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]

        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * self.gt_classes[fg_inds, None] + torch.arange(
                box_dim, device=device
            )

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                self.proposals.tensor[fg_inds],
            )
            loss_box_reg = giou_loss(
                fg_pred_boxes,
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def binary_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        # self._log_accuracy()

        if not self.mean_loss:
            return F.binary_cross_entropy(
                self.predict_probs_img(), self.gt_classes_img_oh, reduction="sum"
            ) / self.gt_classes_img_oh.size(0)

        return F.binary_cross_entropy(
            self.predict_probs_img(), self.gt_classes_img_oh, reduction="mean"
        )

    def predict_probs_img(self):
        if not hasattr(self, "pred_class_img_logits"):
            if len(self.num_preds_per_image) == 1:
                self.pred_class_img_logits = torch.sum(self.pred_class_logits, dim=0, keepdim=True)
            else:
                self.pred_class_img_logits = cat(
                    [
                        torch.sum(xx, dim=0, keepdim=True)
                        for xx in self.pred_class_logits.split(self.num_preds_per_image, dim=0)
                    ],
                    dim=0,
                )
            self.pred_class_img_logits = torch.clamp(
                self.pred_class_img_logits, min=1e-6, max=1.0 - 1e-6
            )
        return self.pred_class_img_logits

    def losses(self):
        """
        Deprecated
        """
        # return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}
        return {"loss_cls": self.binary_cross_entropy_loss()}

    def predict_boxes(self):
        """
        Deprecated
        """
        pred = self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)
        return pred.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        # probs = F.softmax(self.pred_class_logits, dim=-1)
        probs = self.pred_class_logits
        probs_bg = torch.zeros(
            probs.shape[0], 1, dtype=probs.dtype, device=probs.device, requires_grad=False
        )
        probs = torch.cat((probs, probs_bg), 1)
        return probs.split(self.num_preds_per_image, dim=0)


class JTSMOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        num_classes_stuff: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        use_fed_loss: bool = False,
        use_sigmoid_ce: bool = False,
        get_fed_loss_cls_weights: Optional[Callable] = None,
        fed_loss_num_classes: int = 50,
        mean_loss: bool = True,
        cmil: bool = False,
        display: int = 1280,
        max_epoch: int = 40,
        size_epoch: int = 5000,
        test_num_classes: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou",
                "diou", "ciou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
            use_fed_loss (bool): whether to use federated loss which samples additional negative
                classes to calculate the loss
            use_sigmoid_ce (bool): whether to calculate the loss using weighted average of binary
                cross entropy with logits. This could be used together with federated loss
            get_fed_loss_cls_weights (Callable): a callable which takes dataset name and frequency
                weight power, and returns the probabilities to sample negative classes for
                federated loss. The implementation can be found in
                detectron2/data/detection_utils.py
            fed_loss_num_classes (int): number of federated classes to keep in total
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes + num_classes_stuff - 1
        box_dim = len(box2box_transform.weights)

        self.num_bbox_reg_classes = num_bbox_reg_classes
        self.box_dim = box_dim

        if num_classes_stuff > 2:
            self.cls = nn.Linear(input_size, num_classes + num_classes_stuff - 1)
            self.det = nn.Linear(input_size, num_classes + num_classes_stuff - 1)
        else:
            self.cls = nn.Linear(input_size, num_classes)
            self.det = nn.Linear(input_size, num_classes)

        # nn.init.normal_(self.cls.weight, std=0.01)
        # nn.init.normal_(self.det.weight, std=0.01)
        nn.init.xavier_uniform_(self.cls.weight)
        nn.init.xavier_uniform_(self.det.weight)
        for l in [self.cls, self.det]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.use_fed_loss = use_fed_loss
        self.use_sigmoid_ce = use_sigmoid_ce
        self.fed_loss_num_classes = fed_loss_num_classes

        if self.use_fed_loss:
            assert self.use_sigmoid_ce, "Please use sigmoid cross entropy loss with federated loss"
            fed_loss_cls_weights = get_fed_loss_cls_weights()
            assert (
                len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

        self.mean_loss = mean_loss

        self.cmil = cmil
        if cmil:
            self.display = display
            self.max_epoch = max_epoch
            self.size_epoch = size_epoch
            self.roi_merge = ROIMerge(0, display, max_epoch, size_epoch)

        self.test_num_classes = test_num_classes

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"               : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "num_classes_stuff"         : cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "cls_agnostic_bbox_reg"     : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"            : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"         : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"           : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"       : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"         : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"               : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT, "loss_cls": cfg.WSL.JTSM.WEIGHT}, # noqa
            "use_fed_loss"              : cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
            "use_sigmoid_ce"            : cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            "get_fed_loss_cls_weights"  : lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER),  # noqa
            "fed_loss_num_classes"      : cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES,
            "mean_loss"                 : cfg.WSL.JTSM.MEAN_LOSS,
            "cmil"                      : cfg.WSL.CMIL,
            "display"                   : int(1280 / cfg.SOLVER.IMS_PER_BATCH),
            "max_epoch"                 : int(cfg.SOLVER.MAX_ITER / cfg.WSL.SIZE_EPOCH),
            "size_epoch"                : cfg.WSL.SIZE_EPOCH,
            "test_num_classes"          : cfg.TEST.NUM_CLASSES_TEST,
            # fmt: on
        }

    def forward(self, x, proposals=None, context=False):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if context:
            C, D = self.forward_contextlocnet(x, proposals=proposals)
        else:
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            C = self.cls(x)
            D = self.det(x)

        if self.cmil and self.training:
            C, D, Oscores, Oproposal_deltas = self.forward_cmil(C, D, proposals)

        if self.num_classes == 1:
            C = torch.cat((C, torch.zeros_like(C)), dim=1)
            D = torch.cat((D, torch.zeros_like(D)), dim=1)

        if proposals is None:
            scores = F.softmax(C, dim=1) * F.softmax(D, dim=0)
        elif len(proposals) == 1:
            scores = F.softmax(C, dim=1) * F.softmax(D, dim=0)
        else:
            num_preds_per_image = [len(p) for p in proposals]
            scores = cat(
                [
                    F.softmax(c, dim=1) * F.softmax(d, dim=0)
                    for c, d in zip(
                        C.split(num_preds_per_image, dim=0), D.split(num_preds_per_image, dim=0)
                    )
                ],
                dim=0,
            )

        if self.num_classes == 1:
            scores, _ = torch.split(scores, 1, dim=1)

        proposal_deltas = torch.zeros(
            scores.shape[0],
            self.num_bbox_reg_classes * self.box_dim,
            dtype=scores.dtype,
            device=scores.device,
            requires_grad=False,
        )

        if self.cmil and self.training:
            return scores, proposal_deltas, Oscores, Oproposal_deltas
        return scores, proposal_deltas

    def forward_contextlocnet(self, x, proposals=None):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        x, Fx, Cx = x[:]
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if Fx.dim() > 2:
            Fx = torch.flatten(Fx, start_dim=1)
        if Cx.dim() > 2:
            Cx = torch.flatten(Cx, start_dim=1)

        return self.cls(x), self.det(Fx) - self.det(Cx)

    def forward_cmil(self, C, D, proposals):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if proposals is None:
            scores = F.softmax(C, dim=1) * F.softmax(D, dim=0)
        elif len(proposals) == 1:
            scores = F.softmax(C, dim=1) * F.softmax(D, dim=0)
        else:
            num_preds_per_image = [len(p) for p in proposals]
            scores = cat(
                [
                    F.softmax(c, dim=1) * F.softmax(d, dim=0)
                    for c, d in zip(
                        C.split(num_preds_per_image, dim=0), D.split(num_preds_per_image, dim=0)
                    )
                ],
                dim=0,
            )

        proposal_deltas = torch.zeros(
            scores.shape[0],
            self.num_bbox_reg_classes * self.box_dim,
            dtype=scores.dtype,
            device=scores.device,
            requires_grad=False,
        )

        # num_preds_per_image = [len(p) for p in proposals]
        rois_obn_score = torch.sum(scores, dim=1, keepdim=True)
        # rois_obn_score = torch.clamp(rois_obn_score, min=1e-6, max=1.0 - 1e-6)

        assert proposals
        J = cat([pairwise_iou(p.proposal_boxes, p.proposal_boxes) for p in proposals], dim=0)

        MC, MD = self.roi_merge(rois_obn_score.cpu(), J.cpu(), C.cpu(), D.cpu())
        return MC.to(C.device), MD.to(D.device), scores, proposal_deltas

    def losses(self, predictions, proposals, gt_classes_img_int):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        pred_class_img_logits = self.predict_probs_img(predictions, proposals)

        # only use the first non-zero element as gt
        gt_classes_img_int_1 = cat([x[:1] for x in gt_classes_img_int], dim=0)
        # assert gt_classes_img_int_1.dim() == 2
        assert pred_class_img_logits.dim() == 2
        _log_classification_stats(
            pred_class_img_logits, gt_classes_img_int_1, prefix="fast_rcnn_wsddn"
        )

        loss_cls = self.binary_cross_entropy(pred_class_img_logits, gt_classes_img_int)

        losses = {
            "loss_cls": loss_cls,
        }

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def losses_csc(
        self, predictions, proposals, W_pos, W_neg, PL, NL, csc_stats, loss_weight=1.0, prefix=""
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        scores_pos = scores * W_pos
        scores_neg = scores * W_neg

        scores_img_pos = torch.sum(scores_pos, dim=0, keepdim=True)
        scores_img_pos = torch.clamp(scores_img_pos, min=1e-20, max=1.0 - 1e-20)

        scores_img_neg = torch.sum(scores_neg, dim=0, keepdim=True)
        scores_img_neg = torch.clamp(scores_img_neg, min=1e-20, max=1.0 - 1e-20)

        csc_stats.UpdateIterStats(
            PL.data.cpu().numpy(),
            predict_probs_img(predictions, proposals).data.cpu().numpy(),
            scores_img_pos.data.cpu().numpy(),
            scores_img_neg.data.cpu().numpy(),
            W_pos.data.cpu().numpy(),
            W_neg.data.cpu().numpy(),
        )
        csc_stats.LogIterStats()

        if not self.mean_loss:
            losses = {
                prefix
                + "loss_cls_pos": F.binary_cross_entropy(scores_img_pos, PL, reduction="sum")
                / PL.size(0),
                prefix
                + "loss_cls_neg": F.binary_cross_entropy(scores_img_neg, NL, reduction="sum")
                / NL.size(0),
            }

            return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

        return {
            prefix + "loss_cls_pos": F.binary_cross_entropy(scores_img_pos, PL, reduction="mean"),
            prefix + "loss_cls_neg": F.binary_cross_entropy(scores_img_neg, NL, reduction="mean"),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    # Implementation from https://github.com/xingyizhou/CenterNet2/blob/master/projects/CenterNet2/centernet/modeling/roi_heads/fed_loss.py  # noqa
    # with slight modifications
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: number of classes to keep in total, including both unique gt
                classes and sampled negative classes
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes

        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        return fed_loss_classes

    @autocast(enabled=False)
    def binary_cross_entropy(self, scores, gt_classes_img_int):
        scores = scores.float()
        # gt_classes_img_oh = gt_classes_img_oh.float()

        if scores.numel() == 0:
            return scores.new_zeros([1])[0]

        gt_classes_img_oh = torch.cat(
            [
                torch.zeros((1, self.num_classes), dtype=torch.float, device=gt.device).scatter_(
                    1, torch.unsqueeze(gt, dim=0), 1
                )
                for gt in gt_classes_img_int
            ],
            dim=0,
        )

        N = scores.shape[0]
        K = scores.shape[1]

        cls_loss = F.binary_cross_entropy(scores, gt_classes_img_oh, reduction="none")

        if self.use_fed_loss:
            fed_loss_classes = self.get_fed_loss_classes(
                torch.cat(gt_classes_img_int),
                num_fed_loss_classes=self.fed_loss_num_classes,
                num_classes=K,
                weight=self.fed_loss_cls_weights,
            )
            fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
            fed_loss_classes_mask[fed_loss_classes] = 1
            fed_loss_classes_mask = fed_loss_classes_mask[:K]
            weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()
        else:
            weight = 1

        if self.mean_loss:
            loss = torch.sum(cls_loss * weight) / (N * K)
            return loss

        loss = torch.sum(cls_loss * weight) / N
        return loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        loss_box_reg = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
        )

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.test_num_classes,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        return [p.proposal_boxes.tensor for p in proposals]

        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        # probs = F.softmax(scores, dim=-1)
        probs = scores
        probs_bg = torch.zeros(
            probs.shape[0], 1, dtype=probs.dtype, device=probs.device, requires_grad=False
        )
        probs = torch.cat((probs, probs_bg), 1)
        return probs.split(num_inst_per_image, dim=0)

    def predict_probs_img(self, predictions, proposals):
        scores, _ = predictions
        if len(proposals) == 1:
            pred_class_img_logits = torch.sum(scores, dim=0, keepdim=True)
        else:
            num_inst_per_image = [len(p) for p in proposals]
            pred_class_img_logits = cat(
                [
                    torch.sum(score, dim=0, keepdim=True)
                    for score in scores.split(num_inst_per_image, dim=0)
                ],
                dim=0,
            )
        pred_class_img_logits = torch.clamp(pred_class_img_logits, min=1e-6, max=1.0 - 1e-6)
        return pred_class_img_logits
