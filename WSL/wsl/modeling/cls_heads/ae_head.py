# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from skimage import measure
from torch import nn
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from sklearn.manifold import TSNE
from wsl.modeling.cls_heads.cls_head import CLS_HEAD_REGISTRY
from wsl.modeling.cls_heads.third_party.selective_search import selective_search
from wsl.modeling.roi_heads.roi_heads import get_image_level_gt

__all__ = ["AEHead"]


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


def get_more_boxes(mask):
    # get more box
    max_value = np.max(mask)
    if max_value > 0:
        max_value = max_value * 0.1
        mask = np.clip(mask, 0, max_value)
        mask = mask / max_value * 255
    mask = mask.astype(np.uint8)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    boxes = selective_search(mask, mode="single", random_sort=False)
    return boxes


@torch.no_grad()
def transform_tsne(features, idxs_img):
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
def transform_svd_pt(features, idxs_img):
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
def flip_svd_pt(u, v, u_based_decision=True):
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
def transform_lowrank_pt(features, rank=1):

    mean_matrix = torch.mean(features, 0)

    features = features - mean_matrix

    U, S, V = torch.pca_lowrank(features, q=None, center=False, niter=2)
    # U, S, V = torch.svd_lowrank(features, q=None, niter=2, M=None)
    U, V = flip_svd_pt(U, V)

    # heatmaps = torch.matmul(features, V[:, :1])
    heatmaps = torch.mm(features, V[:, :rank])

    return heatmaps


@torch.no_grad()
def transform_pca_sk(features, pcas, idxs_img):
    n, h, w, c = features.size()
    features = features.reshape(-1, c)

    mean_matrix = torch.mean(features, 0)

    features = features - mean_matrix
    features_cpu = features.clone().detach().cpu()

    pca = pcas[idx_cls]
    # pca.fit(features_cpu)
    heatmaps = pca.fit_transform(features_cpu)
    # pca.partial_fit(features_cpu)

    # trans_matrix = pca.components_[:1, ...]
    # trans_matrix = torch.tensor(pca.components_[:1, ...]).to(features.device)

    # heatmaps = np.dot(features_cpu, trans_matrix.T)
    # heatmaps = torch.matmul(features, torch.transpose(trans_matrix, 0, 1))
    heatmaps = heatmaps.reshape(n, h, w)

    return heatmaps


@CLS_HEAD_REGISTRY.register()
class AEHead(nn.Module):
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
        conv_norm="SyncBN",
        cls_in_features: List[str],
        mean_loss: bool = True,
        proposal_on_feature: bool = True,
        rank: int = 1,
        rank_supervision: int = 2,
        min_box_size: float = 0.0,
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

        self.num_classes = num_classes

        self.in_features = self.cls_in_features = cls_in_features

        self.in_channels = [input_shape[f].channels for f in self.in_features]
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.strides = [input_shape[f].stride for f in self.in_features]

        self._output_size = (in_channels,)

        self.iter = 0

        self.output_dir = output_dir
        self.vis_test = vis_test
        self.vis_period = vis_period

        self.mean_loss = mean_loss
        self.proposal_on_feature = proposal_on_feature
        self.min_box_size = float(min_box_size)
        self.rank = rank
        self.rank_supervision = rank_supervision

        self.mrrp_on = mrrp_on
        self.mrrp_num_branch = mrrp_num_branch
        self.mrrp_fast = mrrp_fast

        conv_dims = [int(in_channels / 2)]
        conv_kernels = [3]
        conv_pads = [1]
        while conv_dims[-1] > 32:
            conv_dims.append(max(int(conv_dims[-1] / 2), 32))
            conv_kernels.append(3)
            conv_pads.append(1)

        conv_dims.append(rank)
        conv_kernels.append(3)
        conv_pads.append(1)

        self.encoders = []
        for k, (conv_dim, conv_kernel, conv_pad) in enumerate(
            zip(conv_dims, conv_kernels, conv_pads)
        ):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=conv_kernel,
                padding=conv_pad,
                bias=not conv_norm,
                # norm=get_norm("", conv_dim),
                norm=get_norm(conv_norm, conv_dim),
                activation=F.elu if k < len(conv_dims) - 1 else None,
            )
            self.add_module("encoder{}".format(k + 1), conv)
            self.encoders.append(conv)
            self._output_size = (conv_dim,)

        for layer in self.encoders:
            weight_init.c2_msra_fill(layer)

        conv_dims = [32]
        conv_kernels = [3]
        conv_pads = [1]
        while conv_dims[-1] < in_channels:
            conv_dims.append(min(int(conv_dims[-1] * 2), in_channels))
            conv_kernels.append(3)
            conv_pads.append(1)

        self.decoders = []
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
                activation=F.elu if k < len(conv_dims) - 1 else None,
            )
            self.add_module("decoder{}".format(k + 1), conv)
            self.decoders.append(conv)
            self._output_size = (conv_dim,)

        for layer in self.decoders:
            weight_init.c2_msra_fill(layer)

        fc_dims = []

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

        for layer in self.fcs:
            if not isinstance(layer, Linear):
                continue
            # weight_init.c2_xavier_fill(layer)
            torch.nn.init.normal_(layer.weight, std=0.005)
            torch.nn.init.constant_(layer.bias, 0.1)

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
            "mean_loss": cfg.WSL.AE.MEAN_LOSS,
            "proposal_on_feature": cfg.WSL.AE.PROPOSAL_ON_FEATURE,
            "rank": cfg.WSL.AE.RANK,
            "rank_supervision": cfg.WSL.AE.RANK_SUPERVISION,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
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

        features = [features[f] for f in self.cls_in_features]

        if self.mrrp_on:
            features = [torch.chunk(f, self.mrrp_num_branch) for f in features]
            features = [ff for f in features for ff in f]
        # [torch.Size([2, 512, 128, 178]), torch.Size([2, 512, 128, 178]), torch.Size([2, 512, 128, 178])]

        # features, _ = torch.stack(features).max(dim=0)

        self.images = images

        encodes = [self._forward_encoder(feat) for feat in features]
        decodes = [self._forward_decoder(encode) for encode in encodes]

        self._forward_proposal(encodes)

        if self.training:
            self.gt_classes_img_int = get_image_level_gt(targets, self.num_classes)
            self.gt_classes_img_oh = torch.cat(
                [
                    torch.zeros((1, self.num_classes), dtype=torch.float, device=gt.device).scatter_(
                        1, torch.unsqueeze(gt, dim=0), 1
                    )
                    for gt in self.gt_classes_img_int
                ],
                dim=0,
            )

            losses = self.loss(encodes, decodes, features)

            self.iter = self.iter + 1
            return losses
        else:
            if self.vis_test:
                self.compression_codings = self.get_compression_coding3(features)

            self.iter = self.iter + 1
            return None

    def _forward_encoder(self, x):
        for k, layer in enumerate(self.encoders):
            x = layer(x)
            # if k < len(self.conv_norm_relus) - 1:
            # x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        return x

    def _forward_decoder(self, x):
        for k, layer in enumerate(self.decoders):
            x = layer(x)
            # if k < len(self.conv_norm_relus) - 1:
            # x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        return x

    @torch.no_grad()
    def _forward_proposal(self, encodes):
        batch_bboxes, batch_level_ids, batch_proposals, batch_level_ids_proposal = self.get_bbox(
            encodes
        )

        if (not self.training and self.vis_test) or (
            self.training and self.vis_period > 0 and self.iter % self.vis_period == 0
        ):
            f_per_img = len(encodes)
            i_per_bat = len(self.images)

            encodes = torch.cat(encodes, dim=0)
            encodes = encodes.permute(0, 2, 3, 1)

            for f in range(f_per_img):
                for r in range(self.rank):
                    self._vis_mask(
                        encodes[f * i_per_bat : (f + 1) * i_per_bat, ..., r : r + 1],
                        "ae",
                        "_encode" + str(f) + "_r" + str(r),
                    )

            self._vis_box(
                batch_bboxes,
                prefix="ae",
                suffix="_box",
            )

            self._vis_box(
                batch_proposals,
                prefix="ae",
                suffix="_proposal",
            )

        for n, image_size in enumerate(self.images.image_sizes):
            boxes = batch_bboxes[n]
            boxes = Boxes(boxes)
            boxes.clip(image_size)

            # filter empty boxes
            _ = boxes.nonempty(threshold=self.min_box_size)

            batch_bboxes[n] = boxes

        for n, image_size in enumerate(self.images.image_sizes):
            proposals = batch_proposals[n]
            proposals = Boxes(proposals)
            proposals.clip(image_size)

            # filter empty boxes
            _ = proposals.nonempty(threshold=self.min_box_size)

            batch_proposals[n] = proposals

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

        if self.training:
            storage = get_event_storage()
            storage.put_scalar(
                "AEHead/num_proposals", sum([len(x) for x in self.proposals]) / len(self.proposals)
            )
            storage.put_scalar(
                "AEHead/num_proposals_more",
                sum([len(x) for x in self.proposals_more]) / len(self.proposals_more),
            )

        if not self.training and self.vis_test:
            self._save_proposal_test(self.proposals, "test_ae", "_proposals")

    @torch.no_grad()
    def get_compression_coding(self, features):
        rank = self.rank_supervision

        f_per_img = len(features)
        i_per_bat = len(self.images)
        # print([f.size() for f in features])
        # [torch.Size([2, 512, 108, 142]), torch.Size([2, 512, 108, 142]), torch.Size([2, 512, 108, 142])]

        for feature in features:
            assert features[0].size() == feature.size()
        _, channel, height, width = features[0].size()

        if f_per_img > 1:
            spatial_scale = 1.0 * features[0].size()[-1] / self.images.tensor.size()[-1]
            feat_sizes = [
                (int(image_size[0] * spatial_scale) - 1, int(image_size[1] * spatial_scale) - 1)
                for image_size in self.images.image_sizes
            ]

            feats = torch.cat(
                [
                    features[i][j : j + 1, ..., : feat_sizes[j][0], : feat_sizes[j][1]]
                    .permute(0, 2, 3, 1)
                    .reshape(-1, channel)
                    for i in range(f_per_img)
                    for j in range(i_per_bat)
                ],
                dim=0,
            )
            feat_nums = [
                feat_sizes[j][0] * feat_sizes[j][1]
                for _ in range(f_per_img)
                for j in range(i_per_bat)
            ]
        else:
            features = torch.cat(features, dim=0)
            features = features.permute(0, 2, 3, 1)
            features = features.reshape(-1, channel)

        # heatmaps = transform_pca_sk(features, self.pcas, idxs_img)
        heatmaps = transform_lowrank_pt(feats, rank)
        # heatmaps = transform_svd_pt(features, idxs_img)
        # heatmaps = transform_tsne(features, idxs_img)

        if f_per_img > 1:
            heatmaps = torch.split(heatmaps, feat_nums, dim=0)
            heatmaps = torch.cat(
                [
                    F.pad(
                        heatmaps[i * i_per_bat + j]
                        .reshape(1, feat_sizes[j][0], feat_sizes[j][1], rank)
                        .permute(0, 3, 1, 2)
                        .reshape(rank, feat_sizes[j][0], feat_sizes[j][1]),
                        (0, width - feat_sizes[j][1], 0, height - feat_sizes[j][0]),
                        mode="constant",
                        value=0.0,
                    )
                    for i in range(f_per_img)
                    for j in range(i_per_bat)
                ],
                dim=0,
            )
        else:
            heatmaps = heatmaps.reshape(f_per_img, height, width, rank)

            # n x h x w x c to n * c x h x w
            heatmaps = heatmaps.permute(0, 3, 1, 2).reshape(f_per_img * rank, height, width)

        if (not self.training and self.vis_test) or (
            self.training and self.vis_period > 0 and self.iter % self.vis_period == 0
        ):
            for f in range(f_per_img):
                for r in range(rank):
                    self._vis_mask(
                        heatmaps[f * i_per_bat * rank + r : (f + 1) * i_per_bat * rank : rank, ...],
                        "ae",
                        "_heatmap" + str(f) + "_r" + str(r),
                    )

        if isinstance(heatmaps, torch.Tensor):
            pass
        else:
            heatmaps = torch.from_numpy(heatmaps).to(features[0].device)

        highlights = torch.zeros_like(heatmaps)
        highlights[heatmaps > 0] = 1

        if (not self.training and self.vis_test) or (
            self.training and self.vis_period > 0 and self.iter % self.vis_period == 0
        ):
            highlights_np = highlights.clone().detach().cpu().numpy()
            # max component
            all_labels = [measure.label(highlight) for highlight in highlights_np]

            for f in range(f_per_img):
                for r in range(rank):
                    self._vis_mask(
                        all_labels[f * i_per_bat * rank + r : (f + 1) * i_per_bat * rank : rank],
                        "ae",
                        "_label" + str(f) + "_r" + str(r),
                    )

                    self._vis_mask(
                        highlights_np[
                            f * i_per_bat * rank + r : (f + 1) * i_per_bat * rank : rank, ...
                        ],
                        "ae",
                        "_highlight" + str(f) + "_r" + str(r),
                    )

        # torch.Size([3 * 2 * rank, 128, 178])
        # to
        # [torch.Size([2, rank, 128, 178]), torch.Size([2, rank, 128, 178]), torch.Size([2, rank, 128, 178])]

        highlights = [
            torch.cat(
                [
                    torch.unsqueeze(
                        highlights[
                            i * i_per_bat * rank
                            + j * rank : i * i_per_bat * rank
                            + j * rank
                            + self.rank,
                            :,
                            :,
                        ],
                        0,
                    )
                    for j in range(i_per_bat)
                ],
                dim=0,
            )
            for i in range(f_per_img)
        ]

        return highlights

    @torch.no_grad()
    def get_bbox(self, encodes):
        rank = self.rank

        i_per_bat = len(self.images)
        f_per_img = len(encodes)
        encodes = torch.cat(encodes, dim=0)
        # encodes = encodes.permute(0, 2, 3, 1)

        heatmaps = encodes.flatten(start_dim=0, end_dim=1)

        if isinstance(heatmaps, torch.Tensor):
            highlights = torch.zeros_like(heatmaps)
            highlights[heatmaps > 0] = 1

            highlights = highlights.clone().detach().cpu().numpy()
        else:
            highlights = np.zeros_like(heatmaps)
            highlights[heatmaps > 0] = 1

        # max component
        all_labels = [measure.label(highlight) for highlight in highlights]

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

        # print(all_counts, sum(all_counts))

        batch_bboxes = []
        batch_level_ids = []
        batch_proposals = []
        batch_level_ids_proposal = []
        for i, idx_img in enumerate(range(i_per_bat)):
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
                    all_label = all_labels[f * len(self.images) * rank + i * rank + r]
                    proposals = measure.regionprops(all_label)

                    scale = he / all_label.shape[0]

                    if len(proposals) == 0:
                        bboxes_r.append([0, 0, wi, he])
                        level_ids_r.append(f * 1000)
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
                    if self.proposal_on_feature:
                        proposals = get_more_boxes(all_label)
                    else:
                        proposals = []
                    if len(proposals) == 0:
                        proposals = np.array([[0, 0, wi, he]])
                    else:
                        scale = he / all_label.shape[0]
                        proposals = np.array(proposals) * scale
                    proposals_r.extend(proposals)
                    level_ids_proposal_r.extend([f * 1000 for _ in range(proposals.shape[0])])

                bboxes_i.extend(bboxes_r)
                level_ids_i.extend(level_ids_r)
                proposals_i.extend(proposals_r)
                level_ids_proposal_i.extend(level_ids_proposal_r)

            # print("num_bbox: ", len(bboxes_i), "num_proposal: ", len(proposals_i))

            batch_bboxes.append(torch.tensor(bboxes_i).to(encodes.device))
            batch_level_ids.append(torch.tensor(level_ids_i).to(encodes.device))
            batch_proposals.append(torch.tensor(proposals_i).to(encodes.device))
            batch_level_ids_proposal.append(torch.tensor(level_ids_proposal_i).to(encodes.device))

        return batch_bboxes, batch_level_ids, batch_proposals, batch_level_ids_proposal

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
        if not self.mean_loss:
            return F.binary_cross_entropy_with_logits(
                self.pred_class_img_logits, self.gt_classes_img_oh, reduction="sum"
            ) / self.gt_classes_img_oh.size(0)

        return F.binary_cross_entropy_with_logits(
            self.pred_class_img_logits, self.gt_classes_img_oh, reduction="mean"
        )

    def decode_loss(self, decodes, features):
        losses = {}
        if not self.mean_loss:
            for i, (pred, gt) in enumerate(zip(decodes, features)):
                losses["loss_decode_" + str(i)] = (
                    F.mse_loss(pred, gt, reduction="sum") / self.gt_classes_img_oh.size(0) * 1.0
                )

            return losses

        for i, (pred, gt) in enumerate(zip(decodes, features)):
            # print('decode: ', pred.max(), pred.min(), gt.max(), gt.min())
            losses["loss_decode_" + str(i)] = F.mse_loss(pred, gt, reduction="mean") * 1.0

        return losses

    def encode_loss(self, encodes, features):
        losses = {}

        compression_codings = self.get_compression_coding(features)

        # print([x.size() for x in encodes])
        # print([x.size() for x in compression_codings])

        if not self.mean_loss:
            for i, (pred, gt) in enumerate(zip(encodes, compression_codings)):
                # losses['loss_encode_' + str(i)] = F.mse_loss(pred, gt, reduction='sum') / self.gt_classes_img_oh.size(0) * 0.01
                losses["loss_encode_" + str(i)] = (
                    F.binary_cross_entropy_with_logits(pred, gt, reduction="sum")
                    / self.gt_classes_img_oh.size(0)
                    * 0.1
                )
                # losses['loss_encode_' + str(i)] = F.binary_cross_entropy_with_logits(pred, gt, reduction="sum") / self.gt_classes_img_oh.size(0) * 1.

            return losses

        for i, (pred, gt) in enumerate(zip(encodes, compression_codings)):
            # print('encode: ', pred.max(), pred.min(), gt.max(), gt.min())
            # losses['loss_encode_' + str(i)] = F.mse_loss(pred, gt, reduction='mean')  * 0.01
            losses["loss_encode_" + str(i)] = (
                F.binary_cross_entropy_with_logits(pred, gt, reduction="mean") * 0.1
            )
            # losses['loss_encode_' + str(i)] = F.binary_cross_entropy_with_logits(pred, gt, reduction="mean") * 1.

        return losses

    def loss(self, encodes, decodes, features):
        losses = {}
        losses.update(self.encode_loss(encodes, features))
        losses.update(self.decode_loss(decodes, features))
        return losses
        return {"loss_cls_img": self.binary_cross_entropy_with_logits_loss()}

    @torch.no_grad()
    def _vis_mask(self, masks, prefix, suffix):
        if masks is None:
            return

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        device_index = self.images.tensor.device.index

        for b in range(len(self.images)):
            img = self.images.tensor[b].clone().detach().cpu().numpy()
            channel_swap = (1, 2, 0)
            img = img.transpose(channel_swap)
            pixel_means = [103.939, 116.779, 123.68]
            img += pixel_means
            img = img.astype(np.uint8)

            if isinstance(masks, torch.Tensor):
                mask = masks[b].clone().detach().cpu().numpy()
            else:
                mask = masks[b]

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

            continue
            storage = get_event_storage()

            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_color = img_color.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix
            storage.put_image(vis_name, img_color)

            img_blend = cv2.cvtColor(img_blend, cv2.COLOR_BGR2RGB)
            img_blend = img_blend.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix + "blend"
            storage.put_image(vis_name, img_blend)

    @torch.no_grad()
    def _vis_box(
        self,
        batch_bboxes,
        thickness=6,
        prefix="",
        suffix="",
    ):

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
            # img_ori = cv2.resize(img_ori, None, fx=1.0 / 4.0, fy=1.0 / 4.0)
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

            continue

            storage = get_event_storage()
            img_pgt = img_pgt.transpose(2, 0, 1)
            vis_name = prefix + "_g" + str(device_index) + "_b" + str(b) + suffix
            storage.put_image(vis_name, img_pgt)

    @torch.no_grad()
    def _save_proposal_test(self, proposals, prefix, suffix):
        if not self.vis_test:
            return

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, p in enumerate(proposals):
            box = p.proposal_boxes.tensor.clone().detach().cpu().numpy()
            logit = p.objectness_logits.clone().detach().cpu().numpy()
            level_ids = p.level_ids.clone().detach().cpu().numpy()

            gpu_id = p.objectness_logits.device.index
            id_str = "i" + str(self.iter) + "_g" + str(gpu_id) + "_b" + str(b)

            save_path = os.path.join(output_dir, id_str + "_box" + suffix + ".npy")
            np.save(save_path, box)

            save_path = os.path.join(output_dir, id_str + "_logit" + suffix + ".npy")
            np.save(save_path, logit)

            save_path = os.path.join(output_dir, id_str + "_level" + suffix + ".npy")
            np.save(save_path, level_ids)
