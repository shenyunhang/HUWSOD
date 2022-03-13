# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Note:
For your custom dataset, there is no need to hard-code metadata anywhere in the code.
For example, for COCO-format dataset, metadata will be obtained automatically
when calling `load_coco_json`. For other dataset, metadata may also be obtained in other ways
during loading.

However, we hard-coded metadata for a few common dataset here.
The only goal is to allow users who don't have these dataset to use pre-trained models.
Users don't have to download a COCO json (which contains metadata), in order to visualize a
COCO model (with correct class names and colors).
"""

import numpy as np

from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta

VOC_C = 21


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits""" ""
    return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


voc_cmaps = labelcolormap(VOC_C).tolist()
VOC_CATEGORIES = [
    {"id": 1, "name": "aeroplane", "isthing": 1, "color": voc_cmaps[1]},
    {"id": 2, "name": "bicycle", "isthing": 1, "color": voc_cmaps[2]},
    {"id": 3, "name": "bird", "isthing": 1, "color": voc_cmaps[3]},
    {"id": 4, "name": "boat", "isthing": 1, "color": voc_cmaps[4]},
    {"id": 5, "name": "bottle", "isthing": 1, "color": voc_cmaps[5]},
    {"id": 6, "name": "bus", "isthing": 1, "color": voc_cmaps[6]},
    {"id": 7, "name": "car", "isthing": 1, "color": voc_cmaps[7]},
    {"id": 8, "name": "cat", "isthing": 1, "color": voc_cmaps[8]},
    {"id": 9, "name": "chair", "isthing": 1, "color": voc_cmaps[9]},
    {"id": 10, "name": "cow", "isthing": 1, "color": voc_cmaps[10]},
    {"id": 11, "name": "diningtable", "isthing": 1, "color": voc_cmaps[11]},
    {"id": 12, "name": "dog", "isthing": 1, "color": voc_cmaps[12]},
    {"id": 13, "name": "horse", "isthing": 1, "color": voc_cmaps[13]},
    {"id": 14, "name": "motorbike", "isthing": 1, "color": voc_cmaps[14]},
    {"id": 15, "name": "person", "isthing": 1, "color": voc_cmaps[15]},
    {"id": 16, "name": "pottedplant", "isthing": 1, "color": voc_cmaps[16]},
    {"id": 17, "name": "sheep", "isthing": 1, "color": voc_cmaps[17]},
    {"id": 18, "name": "sofa", "isthing": 1, "color": voc_cmaps[18]},
    {"id": 19, "name": "train", "isthing": 1, "color": voc_cmaps[19]},
    {"id": 20, "name": "tvmonitor", "isthing": 1, "color": voc_cmaps[20]},
    {"id": 21, "name": "background", "isthing": 0, "color": [255, 255, 255]},
]


def _get_voc_instances_meta():
    thing_ids = [k["id"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 20, len(thing_ids)
    # Mapping from the incontiguous VOC category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_voc_sbd_instances_meta():
    thing_ids = [k["id"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 20, len(thing_ids)
    # Mapping from the incontiguous VOC category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_voc_sbd_panoptic_separated_meta():
    """
    Returns metadata for "separated" version of the panoptic segmentation dataset.
    """
    stuff_ids = [k["id"] for k in VOC_CATEGORIES if k["isthing"] == 0]
    assert len(stuff_ids) == 1, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 53], used in models) to ids in the dataset (used for processing results)
    # The id 0 is mapped to an extra category "thing".
    stuff_dataset_id_to_contiguous_id = {k: i + 1 for i, k in enumerate(stuff_ids)}
    # When converting VOC panoptic annotations to semantic annotations
    # We label the "thing" category to 0
    stuff_dataset_id_to_contiguous_id[0] = 0

    # 1 names for VOC stuff categories (including "things")
    stuff_classes = ["things"] + [
        k["name"].replace("-other", "").replace("-merged", "")
        for k in VOC_CATEGORIES
        if k["isthing"] == 0
    ]

    # NOTE: I randomly picked a color for things
    stuff_colors = [[82, 18, 128]] + [k["color"] for k in VOC_CATEGORIES if k["isthing"] == 0]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    ret.update(_get_voc_sbd_instances_meta())
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "flickr_voc":
        return _get_voc_instances_meta()
    elif dataset_name == "flickr_coco":
        return _get_coco_instances_meta()
    elif dataset_name == "voc_2007_pgt":
        return _get_voc_instances_meta()
    elif dataset_name == "voc_sbd":
        return _get_voc_sbd_instances_meta()
    elif dataset_name == "voc_json":
        return _get_voc_instances_meta()
    elif dataset_name == "voc_sbd_panoptic_separated":
        return _get_voc_sbd_panoptic_separated_meta()
    elif dataset_name == "voc_sbd_panoptic_standard":
        meta = {}
        # The following metadata maps contiguous id from [0, #thing categories +
        # #stuff categories) to their names and colors. We have to replica of the
        # same name and color under "thing_*" and "stuff_*" because the current
        # visualization function in D2 handles thing and class classes differently
        # due to some heuristic used in Panoptic FPN. We keep the same naming to
        # enable reusing existing visualization functions.
        thing_classes = [k["name"] for k in VOC_CATEGORIES]
        thing_colors = [k["color"] for k in VOC_CATEGORIES]
        stuff_classes = [k["name"] for k in VOC_CATEGORIES]
        stuff_colors = [k["color"] for k in VOC_CATEGORIES]

        meta["thing_classes"] = thing_classes
        meta["thing_colors"] = thing_colors
        meta["stuff_classes"] = stuff_classes
        meta["stuff_colors"] = stuff_colors

        # Convert category id for training:
        #   category id: like semantic segmentation, it is the class id for each
        #   pixel. Since there are some classes not used in evaluation, the category
        #   id is not always contiguous and thus we have two set of category ids:
        #       - original category id: category id in the original dataset, mainly
        #           used for evaluation.
        #       - contiguous category id: [0, #classes), in order to train the linear
        #           softmax classifier.
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}

        for i, cat in enumerate(VOC_CATEGORIES):
            if cat["isthing"]:
                thing_dataset_id_to_contiguous_id[cat["id"]] = i
            else:
                stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

        return meta
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
