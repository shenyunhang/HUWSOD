# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# from sklearn.manifold import TSNE
# from tsnecuda import TSNE
# from wsl.modeling.cls_heads.tsne_torch import tsne
from skimage import measure
from torch import nn
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from sklearn.decomposition import PCA, FastICA, IncrementalPCA, KernelPCA
from wsl.modeling.cls_heads.third_party.selective_search import selective_search
from wsl.modeling.roi_heads.roi_heads import get_image_level_gt

__all__ = ["ClsConvFCHead", "CLS_HEAD_REGISTRY"]

CLS_HEAD_REGISTRY = Registry("CLS_HEAD")
CLS_HEAD_REGISTRY.__doc__ = """
Registry for classification heads, which make predictions from image features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def count_max(x):
    count_dict = {}
    for xlist in x:
        for item in xlist:
            if item == 0:
                continue
            if item not in count_dict.keys():
                count_dict[item] = 0
            count_dict[item] += 1
    if count_dict == {}:
        return -1
    count_dict = sorted(count_dict.items(), key=lambda d: d[1], reverse=True)
    return count_dict[0][0]


@CLS_HEAD_REGISTRY.register()
class ClsConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes,
        output_dir: str = None,
        vis_test: bool = False,
        vis_period: int = 0,
        conv_dims: List[int],
        fc_dims: List[int],
        conv_norm="",
        cls_in_features: List[str],
        mean_loss: bool = True,
        mrrp_on: bool = False,
        mrrp_num_branch: int = 3,
        mrrp_fast: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()

        conv_dims = [4096, 4096, 20]
        conv_kernels = [7, 1, 1]
        conv_pads = [3, 0, 0]

        conv_dims = [1024, 1024, 20]
        conv_kernels = [3, 3, 3]
        conv_pads = [1, 1, 1]

        conv_dims = []
        conv_kernels = []
        conv_pads = []

        conv_dims = [1024, 1024]
        conv_kernels = [3, 3]
        conv_pads = [1, 1]

        fc_dims = []
        fc_dims = [4096, 4096, 20]
        fc_dims = [20]

        self.num_classes = num_classes
        self.tau = 0
        assert len(conv_dims) + len(fc_dims) > 0

        self.in_features = self.cls_in_features = cls_in_features

        self.in_channels = [input_shape[f].channels for f in self.in_features]
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.strides = [input_shape[f].stride for f in self.in_features]

        self._output_size = (in_channels,)

        self.iter = 0
        self.iter_test = 0
        self.epoch_test = 0

        self.output_dir = output_dir
        self.vis_test = vis_test
        self.vis_period = vis_period

        self.mean_loss = mean_loss

        self.mrrp_on = mrrp_on
        self.mrrp_num_branch = mrrp_num_branch
        self.mrrp_fast = mrrp_fast

        self.conv_norm_relus = []
        for k, (conv_dim, conv_kernel, conv_pad) in enumerate(
            zip(conv_dims, conv_kernels, conv_pads)
        ):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=conv_kernel,
                padding=conv_pad,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                # activation=F.relu_ if k < len(conv_dims) - 1 else None,
                activation=F.relu_,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim,)

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.pool = nn.AdaptiveMaxPool2d((7, 7))
        # self._output_size = (conv_dim, 7, 7)

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            if k == len(fc_dims) - 1:
                pass
            else:
                relu = nn.ReLU(inplace=True)
                self.add_module("fc_relu{}".format(k + 1), relu)
                self.fcs.append(relu)
                dropout = nn.Dropout(p=0.5, inplace=False)
                self.add_module("fc_dropout{}".format(k + 1), dropout)
                self.fcs.append(dropout)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            if not isinstance(layer, Linear):
                continue
            # weight_init.c2_xavier_fill(layer)
            torch.nn.init.normal_(layer.weight, std=0.005)
            torch.nn.init.constant_(layer.bias, 0.1)

        self.pcas = []
        for _ in range(num_classes):
            # self.pcas.append(IncrementalPCA(1))
            # self.pcas.append(PCA(1))
            self.pcas.append(KernelPCA(1))
            # self.pcas.append(FastICA(1))

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        num_conv    = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim    = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        # num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dims     = cfg.MODEL.ROI_BOX_HEAD.DAN_DIM
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        output_dir  = cfg.OUTPUT_DIR
        vis_test    = cfg.VIS_TEST
        vis_period  = cfg.VIS_PERIOD
        # fmt: on

        mrrp_on = cfg.MODEL.MRRP.MRRP_ON
        mrrp_num_branch = cfg.MODEL.MRRP.NUM_BRANCH
        mrrp_fast = cfg.MODEL.MRRP.TEST_BRANCH_IDX != -1

        return {
            "num_classes": num_classes,
            "output_dir": output_dir,
            "vis_test": vis_test,
            "vis_period": vis_period,
            "cls_in_features": in_features,
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": fc_dims,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
            "mean_loss": cfg.WSL.MEAN_LOSS,
            "mrrp_on": mrrp_on,
            "mrrp_num_branch": mrrp_num_branch,
            "mrrp_fast": mrrp_fast,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:

        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )

        features = [features[f] for f in self.cls_in_features]

        if self.mrrp_on:
            features = [torch.chunk(f, self.mrrp_num_branch) for f in features]
            features = [ff for f in features for ff in f]
        # [torch.Size([2, 512, 128, 178]), torch.Size([2, 512, 128, 178]), torch.Size([2, 512, 128, 178])]

        # features, _ = torch.stack(features).max(dim=0)

        self.images = images
        self.features = features
        # self.features.requires_grad = True

        # self.pred_class_img_logits = self._forward_cls(features)

        if self.training:
            # losses = self.loss()
            losses = {}

            # self._forward_cam()
            # self._forward_cpg()
            self._forward_ddt(features)

            self.iter = self.iter + 1
            if self.iter_test > 0:
                self.epoch_test = self.epoch_test + 1
            self.iter_test = 0

            return losses
        else:
            # self._forward_cam()
            # self._forward_cpg()
            # self._forward_ddt(features)

            self.iter_test = self.iter_test + 1

            return None

    def _forward_cls(self, x):
        for k, layer in enumerate(self.conv_norm_relus):
            x = layer(x)
            if k < len(self.conv_norm_relus) - 1:
                x = F.dropout(x, p=0.5, training=self.training, inplace=False)

        self.feature_map = x.clone().detach()

        x = self.pool(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = layer(x)

        return x

    @torch.no_grad()
    def _forward_ddt(self, features):
        batch_bboxes = [[] for _ in range(len(self.images))]
        batch_level_ids = [[] for _ in range(len(self.images))]
        batch_proposals = [[] for _ in range(len(self.images))]
        batch_level_ids_proposal = [[] for _ in range(len(self.images))]

        idx_cls = 0
        idxs_img = [i for i in range(len(self.images))]
        # bboxes, level_ids = self.get_bbox(features, idxs_img, idx_cls)
        # bboxes, level_ids, proposals, level_ids_proposal = self.get_bbox2(features, idxs_img, idx_cls)
        bboxes, level_ids, proposals, level_ids_proposal = self.get_bbox3(
            features, idxs_img, idx_cls
        )
        for idx_r, idx_img in enumerate(idxs_img):
            batch_bboxes[idx_img].append(bboxes[idx_r])
            batch_level_ids[idx_img].append(level_ids[idx_r])
            batch_proposals[idx_img].append(proposals[idx_r])
            batch_level_ids_proposal[idx_img].append(level_ids_proposal[idx_r])

        for idx_cls in range(self.num_classes, -1):
            idxs_img = []
            for idx_img, oh in enumerate(self.gt_classes_img_oh):
                if oh[idx_cls] == 1:
                    idxs_img.append(idx_img)

            if len(idxs_img) == 0:
                continue
            # pca = PCA(1)
            pca = self.pcas[idx_cls]
            bboxes, level_ids = self.transform_pca(pca, features, idxs_img, idx_cls)

            for idx_r, idx_img in enumerate(idxs_img):
                batch_bboxes[idx_img].append(bboxes[idx_r])
                batch_level_ids[idx_img].append(level_ids[idx_r])

        batch_bboxes = [torch.cat(boxes, dim=0) for boxes in batch_bboxes]
        batch_level_ids = [torch.cat(level_ids, dim=0) for level_ids in batch_level_ids]
        batch_proposals = [torch.cat(proposals, dim=0) for proposals in batch_proposals]
        batch_level_ids_proposal = [
            torch.cat(level_ids_proposal, dim=0) for level_ids_proposal in batch_level_ids_proposal
        ]

        self._vis_box(
            batch_bboxes,
            prefix="ddt",
            suffix="_box",
        )

        self._vis_box(
            batch_proposals,
            prefix="ddt",
            suffix="_proposal",
        )

        batch_bboxes = [Boxes(bboxes) for bboxes in batch_bboxes]
        batch_proposals = [Boxes(proposals) for proposals in batch_proposals]
        self.proposals_more = [
            Instances(
                image_sizes,
                proposal_boxes=proposals,
                objectness_logits=torch.ones(len(proposals)).to(proposals.device),
                level_ids=level_ids_proposal,
            )
            for i, (proposals, level_ids_proposal, image_sizes) in enumerate(
                zip(batch_proposals, batch_level_ids_proposal, self.images.image_sizes)
            )
        ]
        self.proposal_targets = [
            Instances(
                image_sizes,
                gt_boxes=bboxes,
                gt_classes=torch.ones(len(bboxes)).to(bboxes.device),
                gt_scores=torch.ones(len(bboxes)).to(bboxes.device),
                gt_weights=torch.ones(len(bboxes)).to(bboxes.device),
            )
            for i, (bboxes, image_sizes) in enumerate(zip(batch_bboxes, self.images.image_sizes))
        ]
        self.proposals = [
            Instances(
                image_sizes,
                proposal_boxes=bboxes,
                objectness_logits=torch.ones(len(bboxes)).to(bboxes.device),
                level_ids=level_ids,
            )
            for i, (bboxes, level_ids, image_sizes) in enumerate(
                zip(batch_bboxes, batch_level_ids, self.images.image_sizes)
            )
        ]

    @torch.no_grad()
    def get_bbox(self, features, idxs_img, idx_cls):
        f_per_img = len(features)
        features = [f[idxs_img] for f in features]
        # [torch.Size([1, 512, 108, 142]), torch.Size([1, 512, 108, 142]), torch.Size([1, 512, 108, 142])]

        features = torch.cat(features, dim=0)

        features = features.permute(0, 2, 3, 1)

        # heatmaps = self.transform_pca_sk(features, idxs_img, idx_cls)
        heatmaps = self.transform_lowrank_pt(features, idxs_img, idx_cls)
        # heatmaps = self.transform_svd_pt(features, idxs_img, idx_cls)
        # heatmaps = self.transform_tsne(features, idxs_img, idx_cls)

        for f in range(f_per_img):
            self._vis_mask(
                heatmaps[f * len(idxs_img) : (f + 1) * len(idxs_img), ...],
                idxs_img,
                "ddt",
                "_c" + str(idx_cls) + "_heatmap_" + str(f),
            )

        if isinstance(heatmaps, torch.Tensor):
            highlights = torch.zeros_like(heatmaps)
            highlights[heatmaps > 0] = 1
            highlights = highlights.clone().detach().cpu().numpy()
        else:
            highlights = np.zeros_like(heatmaps)
            highlights[heatmaps > 0] = 1

        # max component
        all_labels = [measure.label(highlight) for highlight in highlights]
        highlights = np.zeros(highlights.shape)
        for highlight, all_label in zip(highlights, all_labels):
            # highlight[all_label == count_max(all_label.tolist())] = 1
            counts = np.bincount(all_label.flatten())
            min_count = np.max(counts[1:])
            # min_count = min(20, min_count)
            for val, count in enumerate(counts):
                if val == 0:
                    continue
                if count < min_count:
                    continue

                highlight[all_label == val] = 1

        for f in range(f_per_img):
            self._vis_mask(
                highlights[f * len(idxs_img) : (f + 1) * len(idxs_img), ...],
                idxs_img,
                "ddt",
                "_c" + str(idx_cls) + "_highlight_" + str(f),
            )

        highlights = np.round(highlights * 255)
        batch_bboxes = []
        batch_level_ids = []
        for i, idx_img in enumerate(idxs_img):
            _, he, wi = self.images.tensor[idx_img].shape
            bboxes_i = []
            level_ids_i = []
            for f in range(f_per_img):
                highlight = highlights[f * len(idxs_img) + i]
                highlight_big = cv2.resize(highlight, (wi, he), interpolation=cv2.INTER_NEAREST)
                proposals = measure.regionproposals(highlight_big.astype(int))

                if len(proposals) == 0:
                    bboxes_i.append([0, 0, wi, he])
                    level_ids_i.append(f * 1000)
                else:
                    # (min_row, min_col, max_row, max_col)
                    for proposal in proposals:
                        bbox = proposal["bbox"]
                        bboxes_i.append([bbox[1], bbox[0], bbox[3], bbox[2]])
                        level_ids_i.append(f * 1000)

            batch_bboxes.append(torch.tensor(bboxes_i).to(features.device))
            batch_level_ids.append(torch.tensor(level_ids_i).to(features.device))

        return batch_bboxes, batch_level_ids

    @torch.no_grad()
    def get_bbox2(self, features, idxs_img, idx_cls):
        f_per_img = len(features)
        features = [f[idxs_img] for f in features]
        # [torch.Size([1, 512, 108, 142]), torch.Size([1, 512, 108, 142]), torch.Size([1, 512, 108, 142])]

        features = torch.cat(features, dim=0)

        features = features.permute(0, 2, 3, 1)

        rank = 2
        rank_weight = [1, 0.5]

        # heatmaps = self.transform_pca_sk(features, idxs_img, idx_cls)
        heatmaps = self.transform_lowrank_pt(features, idxs_img, idx_cls, rank)
        # heatmaps = self.transform_svd_pt(features, idxs_img, idx_cls)
        # heatmaps = self.transform_tsne(features, idxs_img, idx_cls)

        for f in range(f_per_img):
            for r in range(rank):
                self._vis_mask(
                    heatmaps[
                        f * len(idxs_img) * rank + r : (f + 1) * len(idxs_img) * rank : rank, ...
                    ],
                    idxs_img,
                    "ddt",
                    "_c" + str(idx_cls) + "_heatmap_" + str(f) + "_r" + str(r),
                )

        if isinstance(heatmaps, torch.Tensor):
            highlights = torch.zeros_like(heatmaps)
            highlights[heatmaps > 0] = 1
            highlights = highlights.clone().detach().cpu().numpy()
        else:
            highlights = np.zeros_like(heatmaps)
            highlights[heatmaps > 0] = 1

        # max component
        all_labels = [measure.label(highlight) for highlight in highlights]

        highlights = np.zeros(highlights.shape)
        for highlight, all_label in zip(highlights, all_labels):
            # highlight[all_label == count_max(all_label.tolist())] = 1
            counts = np.bincount(all_label.flatten())
            # if counts.shape[0] <= 1:
            # continue
            min_count = np.max(counts[1:])
            # min_count = min(20, min_count)
            for val, count in enumerate(counts):
                if val == 0:
                    continue
                if count < min_count:
                    continue

                highlight[all_label == val] = 1

        for f in range(f_per_img):
            for r in range(rank):
                self._vis_mask(
                    highlights[
                        f * len(idxs_img) * rank + r : (f + 1) * len(idxs_img) * rank : rank, ...
                    ],
                    idxs_img,
                    "ddt",
                    "_c" + str(idx_cls) + "_highlight_" + str(f) + "_r" + str(r),
                )

        highlights = np.round(highlights * 255)
        batch_bboxes = []
        batch_level_ids = []
        batch_proposals = []
        batch_level_ids_proposal = []
        for i, idx_img in enumerate(idxs_img):
            _, he, wi = self.images.tensor[idx_img].shape
            bboxes_i = []
            level_ids_i = []
            proposals_i = []
            level_ids_proposal_i = []
            for f in range(f_per_img):
                bboxes_r = []
                level_ids_r = []
                density_r = []
                proposals_r = []
                level_ids_proposal_r = []
                for r in range(rank):
                    highlight = highlights[f * len(idxs_img) * rank + i * rank + r]
                    highlight_big = cv2.resize(highlight, (wi, he), interpolation=cv2.INTER_NEAREST)
                    proposals = measure.regionprops(highlight_big.astype(int))

                    if len(proposals) == 0:
                        bboxes_r.append([0, 0, wi, he])
                        level_ids_r.append([f * 1000])
                        density_r.append(0.0)
                    else:
                        # (min_row, min_col, max_row, max_col)
                        for proposal in proposals:
                            bbox = proposal["bbox"]
                            bboxes_r.append([bbox[1], bbox[0], bbox[3], bbox[2]])
                            level_ids_r.append([f * 1000])

                            mass = np.sum(highlight_big[bbox[0] : bbox[2], bbox[1] : bbox[3]]) / 255
                            density = mass / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                            density_r.append(density * rank_weight[r])
                            # print(highlight_big, mass, density)

                            break

                    # get more boxes
                    proposals = self.get_more_boxes(highlight)
                    if len(proposals) == 0:
                        proposals = np.array([[0, 0, wi, he]])
                    else:
                        scale = he / highlight.shape[0]
                        proposals = np.array(proposals) * scale
                    proposals_r.append(proposals)
                    level_ids_proposal_r.append([f * 1000 for _ in range(proposals.shape[0])])

                bboxes_r = [
                    x
                    for _, x in sorted(
                        zip(density_r, bboxes_r), key=lambda pair: pair[0], reverse=True
                    )
                ]
                level_ids_r = [
                    x
                    for _, x in sorted(
                        zip(density_r, level_ids_r), key=lambda pair: pair[0], reverse=True
                    )
                ]
                proposals_r = [
                    x
                    for _, x in sorted(
                        zip(density_r, proposals_r), key=lambda pair: pair[0], reverse=True
                    )
                ]
                level_ids_proposal_r = [
                    x
                    for _, x in sorted(
                        zip(density_r, level_ids_proposal_r), key=lambda pair: pair[0], reverse=True
                    )
                ]

                bboxes_i.append(bboxes_r[0])
                level_ids_i.extend(level_ids_r[0])
                proposals_i.extend(proposals_r[0])
                level_ids_proposal_i.extend(level_ids_proposal_r[0])

            batch_bboxes.append(torch.tensor(bboxes_i).to(features.device))
            batch_level_ids.append(torch.tensor(level_ids_i).to(features.device))
            batch_proposals.append(torch.tensor(proposals_i).to(features.device))
            batch_level_ids_proposal.append(torch.tensor(level_ids_proposal_i).to(features.device))

        return batch_bboxes, batch_level_ids, batch_proposals, batch_level_ids_proposal

    def get_more_boxes(self, mask):
        # get more box
        max_value = np.max(mask)
        if max_value > 0:
            max_value = max_value * 0.1
            mask = np.clip(mask, 0, max_value)
            mask = mask / max_value * 255
        mask = mask.astype(np.uint8)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        boxes = selective_search(mask, mode="single", random_sort=False)
        # print(boxes)
        return boxes

    @torch.no_grad()
    def get_bbox3(self, features, idxs_img, idx_cls):
        f_per_img = len(features)
        features = [f[idxs_img] for f in features]
        # [torch.Size([1, 512, 108, 142]), torch.Size([1, 512, 108, 142]), torch.Size([1, 512, 108, 142])]

        features = torch.cat(features, dim=0)

        features = features.permute(0, 2, 3, 1)

        rank = 2

        # heatmaps = self.transform_pca_sk(features, idxs_img, idx_cls)
        heatmaps = self.transform_lowrank_pt(features, idxs_img, idx_cls, rank)
        # heatmaps = self.transform_svd_pt(features, idxs_img, idx_cls)
        # heatmaps = self.transform_tsne(features, idxs_img, idx_cls)

        for f in range(f_per_img):
            for r in range(rank):
                self._vis_mask(
                    heatmaps[
                        f * len(idxs_img) * rank + r : (f + 1) * len(idxs_img) * rank : rank, ...
                    ],
                    idxs_img,
                    "ddt",
                    "_c" + str(idx_cls) + "_heatmap_" + str(f) + "_r" + str(r),
                )

        if isinstance(heatmaps, torch.Tensor):
            highlights = torch.zeros_like(heatmaps)
            highlights[heatmaps > 0] = 1
            highlights = highlights.clone().detach().cpu().numpy()
        else:
            highlights = np.zeros_like(heatmaps)
            highlights[heatmaps > 0] = 1

        # max component
        all_labels = [measure.label(highlight) for highlight in highlights]

        for f in range(f_per_img):
            for r in range(rank):
                self._vis_mask(
                    all_labels[
                        f * len(idxs_img) * rank + r : (f + 1) * len(idxs_img) * rank : rank
                    ],
                    idxs_img,
                    "ddt",
                    "_c" + str(idx_cls) + "_labels_" + str(f) + "_r" + str(r),
                )

        all_counts = []
        for all_label in all_labels:
            counts = np.bincount(all_label.flatten())
            if counts.shape[0] == 1:
                continue

            min_count = min(40, np.max(counts[1:]))
            for val, count in enumerate(counts):
                if val == 0:
                    continue

                if count < min_count:
                    all_label[all_label == val] = 0

            all_counts.append(np.unique(all_label).shape[0] - 1)

        print(all_counts, sum(all_counts))

        for f in range(f_per_img):
            for r in range(rank):
                self._vis_mask(
                    highlights[
                        f * len(idxs_img) * rank + r : (f + 1) * len(idxs_img) * rank : rank, ...
                    ],
                    idxs_img,
                    "ddt",
                    "_c" + str(idx_cls) + "_highlight_" + str(f) + "_r" + str(r),
                )

        batch_bboxes = []
        batch_level_ids = []
        batch_proposals = []
        batch_level_ids_proposal = []
        for i, idx_img in enumerate(idxs_img):
            _, he, wi = self.images.tensor[idx_img].shape
            bboxes_i = []
            level_ids_i = []
            proposals_i = []
            level_ids_proposal_i = []
            for f in range(f_per_img):
                bboxes_r = []
                level_ids_r = []
                proposals_r = []
                level_ids_proposal_r = []
                for r in range(rank):
                    all_label = all_labels[f * len(idxs_img) * rank + i * rank + r]
                    proposals = measure.regionprops(all_label)

                    scale = he / all_label.shape[0]

                    if len(proposals) == 0:
                        bboxes_r.append([0, 0, wi, he])
                        level_ids_r.append(f * 1000)
                        density_r.append(0.0)
                    else:
                        # (min_row, min_col, max_row, max_col)
                        for proposal in proposals:
                            bbox = np.array(proposal["bbox"]) * scale
                            bboxes_r.append([bbox[1], bbox[0], bbox[3], bbox[2]])
                            level_ids_r.append(f * 1000)

                    num_proposals = len(proposals)
                    if num_proposals < 30 and False:
                        for i_r in range(num_proposals):
                            for j_r in range(num_proposals):
                                if i_r == j_r:
                                    continue
                                bbox_i = np.array(proposals[i_r]["bbox"]) * scale
                                bbox_j = np.array(proposals[j_r]["bbox"]) * scale
                                bboxes_r.append(
                                    [
                                        min(bbox_i[0], bbox_j[0]),
                                        min(bbox_i[1], bbox_j[1]),
                                        max(bbox_i[2], bbox_j[2]),
                                        max(bbox_i[3], bbox_j[3]),
                                    ]
                                )
                                level_ids_r.append(f * 1000)

                    # get more boxes
                    proposals = []
                    if len(proposals) == 0:
                        proposals = np.array([[0, 0, wi, he]])
                    else:
                        scale = he / highlight.shape[0]
                        proposals = np.array(proposals) * scale
                    proposals_r.extend(proposals)
                    level_ids_proposal_r.extend([f * 1000 for _ in range(proposals.shape[0])])

                bboxes_i.extend(bboxes_r)
                level_ids_i.extend(level_ids_r)
                proposals_i.extend(proposals_r)
                level_ids_proposal_i.extend(level_ids_proposal_r)

            print("num_bbox: ", len(bboxes_i), "num_proposal: ", len(proposals_i))

            batch_bboxes.append(torch.tensor(bboxes_i).to(features.device))
            batch_level_ids.append(torch.tensor(level_ids_i).to(features.device))
            batch_proposals.append(torch.tensor(proposals_i).to(features.device))
            batch_level_ids_proposal.append(torch.tensor(level_ids_proposal_i).to(features.device))

        return batch_bboxes, batch_level_ids, batch_proposals, batch_level_ids_proposal

    @torch.no_grad()
    def transform_tsne(self, features, idxs_img, idx_cls):
        n, h, w, c = features.size()
        features = features.reshape(-1, c)

        mean_matrix = torch.mean(features, 0)

        features = features - mean_matrix
        features_cpu = features.clone().detach().cpu()

        # tsne = TSNE(n_components=2, init='pca', random_state=0, n_iter=250, n_jobs=-1)
        tsne = TSNE(n_components=2)
        heatmaps = tsne.fit_transform(features_cpu)

        heatmaps = np.sum(heatmaps, axis=1)
        # heatmaps = tsne(features_cpu, 1, 6, 20.0).cpu().numpy()

        heatmaps = heatmaps.reshape(n, h, w)

        return heatmaps

    @torch.no_grad()
    def transform_svd_pt(self, features, idxs_img, idx_cls):
        n, h, w, c = features.size()
        features = features.reshape(-1, c)

        mean_matrix = torch.mean(features, 0)

        features = features - mean_matrix
        # features_cpu = features.clone().detach().cpu()

        U, S, V = torch.svd(features)
        # U, S, V = torch.svd(features_cpu)

        heatmaps = torch.matmul(features, V[:, :1])
        # heatmaps = torch.matmul(features_cpu, V[:, :1])
        # heatmaps = torch.mm(features, V[:, :1])
        heatmaps = heatmaps.reshape(n, h, w)

        return heatmaps

    @torch.no_grad()
    def flip_svd_pt(self, u, v, u_based_decision=True):
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
            u *= signs
            v *= signs
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=0)
            signs = torch.sign(v[max_abs_rows, range(v.shape[1])])
            v *= signs
            u *= signs

        return u, v

    @torch.no_grad()
    def transform_lowrank_pt(self, features, idxs_img, idx_cls, rank=1):
        n, h, w, c = features.size()
        features = features.reshape(-1, c)

        mean_matrix = torch.mean(features, 0)

        features = features - mean_matrix

        U, S, V = torch.pca_lowrank(features, q=None, center=False, niter=2)
        # U, S, V = torch.svd_lowrank(features, q=None, niter=2, M=None)
        U, V = self.flip_svd_pt(U, V)

        # save feature maps
        if False:
            heatmaps = torch.mm(features, V[:, :3])
            heatmaps = heatmaps.reshape(n, h, w, 3)
            output_dir = os.path.join(self.output_dir, "heatmaps")
            if self.iter or self.iter_test == 0:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            device_index = self.images.tensor.device.index

            f_per_img = int(n / len(idxs_img))
            for i, idx_img in enumerate(idxs_img):
                img = self.images[idx_img].clone().detach().cpu().numpy()
                channel_swap = (1, 2, 0)
                img = img.transpose(channel_swap)
                pixel_means = [103.939, 116.779, 123.68]
                img += pixel_means
                img = img.astype(np.uint8)
                img = cv2.resize(img, None, fx=1.0 / 4.0, fy=1.0 / 4.0)

                save_name = (
                    "i"
                    + str(self.iter_test)
                    + "_g"
                    + str(device_index)
                    + "_b"
                    + str(idx_img)
                    + ".png"
                )
                save_path = os.path.join(output_dir, save_name)
                cv2.imwrite(save_path, img)

                for f in range(f_per_img):
                    save_name = (
                        "i"
                        + str(self.iter_test)
                        + "_g"
                        + str(device_index)
                        + "_b"
                        + str(idx_img)
                        + "_f"
                        + str(f)
                        + ".npy"
                    )
                    save_path = os.path.join(output_dir, save_name)
                    np.save(save_path, heatmaps[f * len(idxs_img) + i].cpu().numpy())

        # heatmaps = torch.matmul(features, V[:, :1])
        heatmaps = torch.mm(features, V[:, :rank])
        heatmaps = heatmaps.reshape(n, h, w, rank)

        # n x h x w x c to n * c x h x w
        heatmaps = heatmaps.permute(0, 3, 1, 2).reshape(n * rank, h, w)

        return heatmaps

    @torch.no_grad()
    def transform_pca_sk(self, features, idxs_img, idx_cls):
        n, h, w, c = features.size()
        features = features.reshape(-1, c)

        mean_matrix = torch.mean(features, 0)

        features = features - mean_matrix
        features_cpu = features.clone().detach().cpu()

        pca = self.pcas[idx_cls]
        # pca.fit(features_cpu)
        heatmaps = pca.fit_transform(features_cpu)
        # pca.partial_fit(features_cpu)

        # trans_matrix = pca.components_[:1, ...]
        # trans_matrix = torch.tensor(pca.components_[:1, ...]).to(features.device)

        # heatmaps = np.dot(features_cpu, trans_matrix.T)
        # heatmaps = torch.matmul(features, torch.transpose(trans_matrix, 0, 1))
        heatmaps = heatmaps.reshape(n, h, w)

        return heatmaps

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])

    def binary_cross_entropy_with_logits_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        # self._log_accuracy()

        reduction = "mean" if self.mean_loss else "sum"
        if not self.mean_loss:
            return F.binary_cross_entropy_with_logits(
                self.pred_class_img_logits, self.gt_classes_img_oh, reduction="sum"
            ) / self.gt_classes_img_oh.size(0)

        return F.binary_cross_entropy_with_logits(
            self.pred_class_img_logits, self.gt_classes_img_oh, reduction="mean"
        )

    def loss(self):
        return {"loss_cls_img": self.binary_cross_entropy_with_logits_loss()}

    @torch.no_grad()
    def _forward_cpg(self):
        if not self.training:
            return None

        # if self.iter > self.csc_max_iter:
        # return None

        # image_sizes = self.features.size()
        image_sizes = self.images.tensor.size()
        assert image_sizes[0] == 1
        cpgs = torch.zeros(
            (image_sizes[0], self.num_classes, image_sizes[2], image_sizes[3]),
            dtype=self.pred_class_img_logits.dtype,
            device=self.pred_class_img_logits.device,
        )
        for c in range(self.num_classes):
            if self.gt_classes_img_oh[0, c] < 0.5:
                continue
            # if self.pred_class_img_logits[0, c] < self.tau:
            # continue

            grad_outputs = torch.zeros(
                self.pred_class_img_logits.size(),
                dtype=self.pred_class_img_logits.dtype,
                device=self.pred_class_img_logits.device,
            )
            grad_outputs[:, c] = 1.0
            (cpg,) = torch.autograd.grad(  # grad_outputs[0, c] = pred_class_img_logits[0, c]
                outputs=self.pred_class_img_logits,
                # inputs=self.features,
                inputs=self.images.tensor,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
            )
            cpg.abs_()
            cpg, _ = torch.max(cpg, dim=1)

            # cpg_scale_op
            max_value = torch.max(cpg)
            cpg = cpg / max_value

            cpgs[0, c, :, :] = cpg[0, :, :]
            del cpg
            del grad_outputs
            torch.cuda.empty_cache()

        # self.features.requires_grad = False
        # self.features.detach()
        self.images.tensor.requires_grad = False
        self.images.tensor.detach()

        cpgs, _ = torch.max(cpgs, dim=1)

        return cpgs

    @torch.no_grad()
    def _forward_cam(self):
        if not self.training:
            return None

        image_sizes = self.feature_map.size()
        assert image_sizes[0] == 1

        all_cams = []
        for i in range(image_sizes[0]):
            cams = torch.zeros(
                (self.num_classes, image_sizes[2], image_sizes[3]),
                dtype=self.feature_map.dtype,
                device=self.feature_map.device,
            )
            for c in range(self.num_classes):
                if self.gt_classes_img_oh[0, c] < 0.5:
                    continue
                # if self.pred_class_img_logits[0, c] < self.tau:
                # continue

                cam_weights = self.fcs[-1].weight[c]
                # cam = (cam_weights.view(*self.feature_map.shape[:2], 1, 1) * self.feature_map).mean(1, keepdim=False)
                cam = (cam_weights.view(-1, 1, 1) * self.feature_map[i]).mean(0, keepdim=False)

                max_value = torch.max(cam)
                cam = cam / max_value

                cams[c, :, :] = cam

            cams, _ = torch.max(cams, dim=0)

            all_cams.append(cams)
        return all_cams

    @torch.no_grad()
    def _vis_mask(self, masks, idxs_img, prefix, suffix):
        if not self.training:
            return
        if masks is None:
            return
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return
        storage = get_event_storage()

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        device_index = self.images.tensor.device.index

        cnt = 0
        for b in range(len(self.images)):
            if b not in idxs_img:
                continue
            img = self.images.tensor[b].clone().detach().cpu().numpy()
            channel_swap = (1, 2, 0)
            img = img.transpose(channel_swap)
            pixel_means = [103.939, 116.779, 123.68]
            img += pixel_means
            img = img.astype(np.uint8)

            if isinstance(masks, torch.Tensor):
                mask = masks[cnt].clone().detach().cpu().numpy()
            else:
                mask = masks[cnt]

            max_value = np.max(mask)
            if max_value > 0:
                max_value = max_value * 0.1
                mask = np.clip(mask, 0, max_value)
                mask = mask / max_value * 255
            mask = mask.astype(np.uint8)

            img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
            img_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            img_blend = cv2.addWeighted(img, 0.5, img_color, 0.5, 0.0)

            save_name = (
                "i"
                + str(self.iter)
                + "_g"
                + str(device_index)
                + "_b"
                + str(b)
                + suffix
                + "_img.png"
            )
            save_path = os.path.join(output_dir, save_name)
            # cv2.imwrite(save_path, img)

            save_name = (
                "i"
                + str(self.iter)
                + "_g"
                + str(device_index)
                + "_b"
                + str(b)
                + suffix
                + "_jet.png"
            )
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, img_color)

            save_name = (
                "i"
                + str(self.iter)
                + "_g"
                + str(device_index)
                + "_b"
                + str(b)
                + suffix
                + "_blend.png"
            )
            save_path = os.path.join(output_dir, save_name)
            # cv2.imwrite(save_path, img_blend)

            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_color = img_color.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix
            storage.put_image(vis_name, img_color)

            img_blend = cv2.cvtColor(img_blend, cv2.COLOR_BGR2RGB)
            img_blend = img_blend.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix + "blend"
            # storage.put_image(vis_name, img_blend)

            cnt += 1

    @torch.no_grad()
    def _vis_box(
        self,
        batch_bboxes,
        thickness=6,
        prefix="",
        suffix="",
    ):
        if not self.training:
            return
        if self.vis_period <= 0 or self.iter % self.vis_period > 0:
            return
        storage = get_event_storage()

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, bboxes in enumerate(batch_bboxes):
            img = self.images[b].clone().detach().cpu().numpy()
            channel_swap = (1, 2, 0)
            img = img.transpose(channel_swap)
            pixel_means = [103.939, 116.779, 123.68]
            img += pixel_means
            img = img.astype(np.uint8)
            h, w = img.shape[:2]
            img_ori = img.copy()
            img_pgt = img.copy()

            device_index = bboxes.device.index

            save_name = (
                "i"
                + str(self.iter)
                + "_g"
                + str(device_index)
                + "_b"
                + str(b)
                + suffix
                + "_img.png"
            )
            img_ori = cv2.resize(img_ori, None, fx=1.0 / 4.0, fy=1.0 / 4.0)
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, img_ori)

            save_name = (
                "i" + str(self.iter) + "_g" + str(device_index) + "_b" + str(b) + suffix + ".png"
            )

            bboxes = bboxes.cpu().numpy()
            for i in range(bboxes.shape[0]):
                x0, y0, x1, y1 = bboxes[i, :]
                x0 = int(max(x0, 0))
                y0 = int(max(y0, 0))
                x1 = int(min(x1, w))
                y1 = int(min(y1, h))
                cv2.rectangle(img_pgt, (x0, y0), (x1, y1), (0, 0, 255), thickness)

            img_pgt = cv2.resize(img_pgt, None, fx=1.0 / 4.0, fy=1.0 / 4.0)

            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, img_pgt)

            img_pgt = img_pgt.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix
            storage.put_image(vis_name, img_pgt)


def build_cls_head(cfg, input_shape):
    """
    Build a classification head defined by `cfg.MODEL.CLS_HEAD.NAME`.
    """
    name = cfg.MODEL.CLS_HEAD.NAME
    if name is None:
        return None
    return CLS_HEAD_REGISTRY.get(name)(cfg, input_shape)
