from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import numpy as np
import scipy.io as sio

from detectron2.data.catalog import DatasetCatalog

iou_thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

max_num_boxes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# 4 GPUs
ims_per_gpu = 1238
# 8 GPUs
# ims_per_gpu = 619

dataset_name = "voc_2007_test"


def recall_huwsod(dataset_dicts, dir_in, dir_in2, max_num_box):

    height_box_all = None
    width_box_all = None
    level_box_all = None
    score_box_all = None

    cnt_yes_cls = [
        0,
    ] * 20
    cnt_gt_cls = [
        0,
    ] * 20

    cnt_yes = [0 for _ in iou_thres]
    cnt_gt = [0 for _ in iou_thres]
    for i, d in enumerate(dataset_dicts):
        if i % 1000 == 0:
            print("{}/{}".format(i + 1, len(dataset_dicts)))

        gpu_id = int(i / ims_per_gpu)
        iter_id = i % ims_per_gpu

        load_name = "i" + str(iter_id) + "_g" + str(gpu_id) + "_b0_box_proposals.npy"
        load_path = os.path.join(dir_in, load_name)
        i_boxes = np.load(load_path)

        load_name = "i" + str(iter_id) + "_g" + str(gpu_id) + "_b0_logit_proposals.npy"
        load_path = os.path.join(dir_in, load_name)
        i_scores = np.load(load_path)

        load_name = "i" + str(iter_id) + "_g" + str(gpu_id) + "_b0_level_proposals.npy"
        load_path = os.path.join(dir_in, load_name)
        i_levels = np.load(load_path)

        # print(i_boxes, i_boxes.shape)
        # print(i_scores, i_scores.shape)
        # print(i_levels, i_levels.shape)

        # -------------------------------------------------------------------------------
        # sort by confidence
        sorted_ind = np.argsort(-(i_scores.flatten()))
        i_boxes = i_boxes[sorted_ind, ...]
        i_scores = i_scores[sorted_ind, ...]
        i_levels = i_levels[sorted_ind, ...]

        i_boxes = i_boxes[:max_num_box, ...]
        i_scores = i_scores[:max_num_box, ...]
        i_levels = i_levels[:max_num_box, ...]

        # -------------------------------------------------------------------------------
        load_name = "i" + str(iter_id) + "_g" + str(gpu_id) + "_b0_box_proposals.npy"
        load_path = os.path.join(dir_in2, load_name)
        i_boxes2 = np.load(load_path)

        load_name = "i" + str(iter_id) + "_g" + str(gpu_id) + "_b0_logit_proposals.npy"
        load_path = os.path.join(dir_in2, load_name)
        i_scores2 = np.load(load_path)

        load_name = "i" + str(iter_id) + "_g" + str(gpu_id) + "_b0_level_proposals.npy"
        load_path = os.path.join(dir_in2, load_name)
        i_levels2 = np.load(load_path)

        # print(i_boxes2, i_boxes2.shape)
        # print(i_scores2, i_scores2.shape)
        # print(i_levels2, i_levels2.shape)

        # -------------------------------------------------------------------------------
        # sort by confidence
        sorted_ind = np.argsort(-(i_scores2.flatten()))
        i_boxes2 = i_boxes2[sorted_ind, ...]
        i_scores2 = i_scores2[sorted_ind, ...]
        i_levels2 = i_levels2[sorted_ind, ...]

        i_boxes2 = i_boxes2[:max_num_box, ...]
        i_scores2 = i_scores2[:max_num_box, ...]
        i_levels2 = i_levels2[:max_num_box, ...]

        i_boxes = np.concatenate((i_boxes, i_boxes2), axis=0)
        i_scores = np.concatenate((i_scores, i_scores2), axis=0)
        i_levels = np.concatenate((i_levels, i_levels2), axis=0)

        # -------------------------------------------------------------------------------
        height = d["height"]
        width = d["width"]

        scale_factor = 1.0 * min(height, width) / 688
        i_boxes = i_boxes * scale_factor
        # -------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------
        height_box = i_boxes[:, 3] - i_boxes[:, 1] + 1.0
        width_box = i_boxes[:, 2] - i_boxes[:, 0] + 1.0
        if height_box_all is None:
            height_box_all = height_box
            width_box_all = width_box
            level_box_all = i_levels
            score_box_all = i_scores
        else:
            # print(height_box.shape, width_box.shape)
            # print(height_box_all.shape, width_box_all.shape, level_box_all.shape)
            height_box_all = np.concatenate((height_box_all, height_box), axis=0)
            width_box_all = np.concatenate((width_box_all, width_box), axis=0)
            level_box_all = np.concatenate((level_box_all, i_levels), axis=0)
            score_box_all = np.concatenate((score_box_all, i_scores), axis=0)
        # -------------------------------------------------------------------------------

        for a in d["annotations"]:
            # print(a["bbox"])
            bbgt = a["bbox"]
            category_id = a["category_id"]

            if "coco" in dataset_name:
                bbgt = [bbgt[0], bbgt[1], bbgt[0] + bbgt[2], bbgt[1] + bbgt[3]]

            ixmin = np.maximum(i_boxes[:, 0], bbgt[0])
            iymin = np.maximum(i_boxes[:, 1], bbgt[1])
            ixmax = np.minimum(i_boxes[:, 2], bbgt[2])
            iymax = np.minimum(i_boxes[:, 3], bbgt[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            intersection = iw * ih

            # unionon
            union = (
                (bbgt[2] - bbgt[0] + 1.0) * (bbgt[3] - bbgt[1] + 1.0)
                + (i_boxes[:, 2] - i_boxes[:, 0] + 1.0) * (i_boxes[:, 3] - i_boxes[:, 1] + 1.0)
                - intersection
            )

            overlaps = intersection / union
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            cnt_gt = [v + 1 for i, v in enumerate(cnt_gt)]
            cnt_yes = [v + 1 if ovmax >= iou_thres[i] else v for i, v in enumerate(cnt_yes)]

            cnt_gt_cls = [v + 1 if i == category_id else v for i, v in enumerate(cnt_gt_cls)]
            cnt_yes_cls = [
                v + 1 if i == category_id and ovmax >= 0.5 else v for i, v in enumerate(cnt_yes_cls)
            ]

            # print(ovmax, jmax)
            # print(cnt_yes, cnt_gt, [1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)])
    print(cnt_yes, cnt_gt)
    print([1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)])

    save_path = os.path.join("mrrp_h_" + str(max_num_box) + ".npy")
    np.save(save_path, height_box_all)

    save_path = os.path.join("mrrp_w_" + str(max_num_box) + ".npy")
    np.save(save_path, width_box_all)

    save_path = os.path.join("mrrp_l_" + str(max_num_box) + ".npy")
    np.save(save_path, level_box_all)

    save_path = os.path.join("mrrp_s_" + str(max_num_box) + ".npy")
    np.save(save_path, score_box_all)

    return [1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)], [
        1.0 * a / b if b > 0 else 0 for a, b in zip(cnt_yes_cls, cnt_gt_cls)
    ]


def recall_opg(dataset_dicts, dir_in, max_num_box):

    height_box_all = None
    width_box_all = None
    level_box_all = None
    score_box_all = None

    cnt_yes_cls = [
        0,
    ] * 20
    cnt_gt_cls = [
        0,
    ] * 20

    cnt_yes = [0 for _ in iou_thres]
    cnt_gt = [0 for _ in iou_thres]
    for i, d in enumerate(dataset_dicts):
        if i % 1000 == 0:
            print("{}/{}".format(i + 1, len(dataset_dicts)))

        gpu_id = int(i / ims_per_gpu)
        iter_id = i % ims_per_gpu

        load_name = "i" + str(iter_id) + "_g" + str(gpu_id) + "_b0_box_proposals.npy"
        load_path = os.path.join(dir_in, load_name)
        i_boxes = np.load(load_path)

        load_name = "i" + str(iter_id) + "_g" + str(gpu_id) + "_b0_logit_proposals.npy"
        load_path = os.path.join(dir_in, load_name)
        i_scores = np.load(load_path)

        load_name = "i" + str(iter_id) + "_g" + str(gpu_id) + "_b0_level_proposals.npy"
        load_path = os.path.join(dir_in, load_name)
        i_levels = np.load(load_path)

        # print(i_boxes, i_boxes.shape)
        # print(i_scores, i_scores.shape)
        # print(i_levels, i_levels.shape)

        # -------------------------------------------------------------------------------
        # sort by confidence
        sorted_ind = np.argsort(-(i_scores.flatten()))
        i_boxes = i_boxes[sorted_ind, ...]
        i_scores = i_scores[sorted_ind, ...]
        i_levels = i_levels[sorted_ind, ...]

        i_boxes = i_boxes[:max_num_box, ...]
        i_scores = i_scores[:max_num_box, ...]
        i_levels = i_levels[:max_num_box, ...]

        # -------------------------------------------------------------------------------
        height = d["height"]
        width = d["width"]

        scale_factor = 1.0 * min(height, width) / 688
        i_boxes = i_boxes * scale_factor
        # -------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------
        height_box = i_boxes[:, 3] - i_boxes[:, 1] + 1.0
        width_box = i_boxes[:, 2] - i_boxes[:, 0] + 1.0
        if height_box_all is None:
            height_box_all = height_box
            width_box_all = width_box
            level_box_all = i_levels
            score_box_all = i_scores
        else:
            # print(height_box.shape, width_box.shape)
            # print(height_box_all.shape, width_box_all.shape, level_box_all.shape)
            height_box_all = np.concatenate((height_box_all, height_box), axis=0)
            width_box_all = np.concatenate((width_box_all, width_box), axis=0)
            level_box_all = np.concatenate((level_box_all, i_levels), axis=0)
            score_box_all = np.concatenate((score_box_all, i_scores), axis=0)
        # -------------------------------------------------------------------------------

        for a in d["annotations"]:
            # print(a["bbox"])
            category_id = a["category_id"]
            bbgt = a["bbox"]

            if "coco" in dataset_name:
                bbgt = [bbgt[0], bbgt[1], bbgt[0] + bbgt[2], bbgt[1] + bbgt[3]]

            ixmin = np.maximum(i_boxes[:, 0], bbgt[0])
            iymin = np.maximum(i_boxes[:, 1], bbgt[1])
            ixmax = np.minimum(i_boxes[:, 2], bbgt[2])
            iymax = np.minimum(i_boxes[:, 3], bbgt[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            intersection = iw * ih

            # unionon
            union = (
                (bbgt[2] - bbgt[0] + 1.0) * (bbgt[3] - bbgt[1] + 1.0)
                + (i_boxes[:, 2] - i_boxes[:, 0] + 1.0) * (i_boxes[:, 3] - i_boxes[:, 1] + 1.0)
                - intersection
            )

            overlaps = intersection / union
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            cnt_gt = [v + 1 for i, v in enumerate(cnt_gt)]
            cnt_yes = [v + 1 if ovmax >= iou_thres[i] else v for i, v in enumerate(cnt_yes)]

            cnt_gt_cls = [v + 1 if i == category_id else v for i, v in enumerate(cnt_gt_cls)]
            cnt_yes_cls = [
                v + 1 if i == category_id and ovmax >= 0.5 else v for i, v in enumerate(cnt_yes_cls)
            ]

            # print(ovmax, jmax)
            # print(cnt_yes, cnt_gt, [1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)])
    print(cnt_yes, cnt_gt)
    print([1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)])

    save_path = os.path.join("mrrp_h_" + str(max_num_box) + ".npy")
    np.save(save_path, height_box_all)

    save_path = os.path.join("mrrp_w_" + str(max_num_box) + ".npy")
    np.save(save_path, width_box_all)

    save_path = os.path.join("mrrp_l_" + str(max_num_box) + ".npy")
    np.save(save_path, level_box_all)

    save_path = os.path.join("mrrp_s_" + str(max_num_box) + ".npy")
    np.save(save_path, score_box_all)

    return [1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)], [
        1.0 * a / b if b > 0 else 0 for a, b in zip(cnt_yes_cls, cnt_gt_cls)
    ]


def recall_mcg(dataset_dicts, dir_in, max_num_box):

    cnt_yes_cls = [
        0,
    ] * 20
    cnt_gt_cls = [
        0,
    ] * 20

    cnt_yes = [0 for _ in iou_thres]
    cnt_gt = [0 for _ in iou_thres]
    for i, d in enumerate(dataset_dicts):
        if i % 1000 == 0:
            print("{}/{}".format(i + 1, len(dataset_dicts)))

        if "flickr" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        elif "coco" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        else:
            index = dataset_dicts[i]["image_id"]
        box_file = os.path.join(dir_in, "{}.mat".format(index))
        mat_data = sio.loadmat(box_file)
        if i == 0:
            print(mat_data.keys())
        if "flickr" in dataset_name:
            boxes_data = mat_data["bboxes"]
            scores_data = mat_data["bboxes_scores"]
        else:
            boxes_data = mat_data["boxes"]
            scores_data = mat_data["scores"]
        # selective search boxes are 1-indexed and (y1, x1, y2, x2)
        # Boxes from the MCG website are in (y1, x1, y2, x2) order
        boxes_data = boxes_data[:, (1, 0, 3, 2)] - 1
        # boxes_data_ = boxes_data.astype(np.uint16) - 1
        # boxes_data = boxes_data_[:, (1, 0, 3, 2)]

        # -------------------------------------------------------------------------------
        # sort by confidence
        # sorted_ind = np.argsort(-(scores_data.flatten()))
        # boxes_data = boxes_data[sorted_ind, ...]
        # scores_data = scores_data[sorted_ind, ...]

        # boxes_data = boxes_data[:max_num_box, ...]
        # scores_data = scores_data[:max_num_box, ...]

        # random
        number_of_rows = boxes_data.shape[0]
        random_indices = np.random.choice(
            number_of_rows, size=min(number_of_rows, max_num_box), replace=False
        )
        boxes_data = boxes_data[random_indices, ...]
        scores_data = scores_data[random_indices, ...]
        # -------------------------------------------------------------------------------

        for a in d["annotations"]:
            bbgt = a["bbox"]
            category_id = a["category_id"]

            if "coco" in dataset_name:
                bbgt = [bbgt[0], bbgt[1], bbgt[0] + bbgt[2], bbgt[1] + bbgt[3]]

            ixmin = np.maximum(boxes_data[:, 0], bbgt[0])
            iymin = np.maximum(boxes_data[:, 1], bbgt[1])
            ixmax = np.minimum(boxes_data[:, 2], bbgt[2])
            iymax = np.minimum(boxes_data[:, 3], bbgt[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            intersection = iw * ih

            # unionon
            union = (
                (bbgt[2] - bbgt[0] + 1.0) * (bbgt[3] - bbgt[1] + 1.0)
                + (boxes_data[:, 2] - boxes_data[:, 0] + 1.0)
                * (boxes_data[:, 3] - boxes_data[:, 1] + 1.0)
                - intersection
            )

            overlaps = intersection / union
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            cnt_gt = [v + 1 for i, v in enumerate(cnt_gt)]
            cnt_yes = [v + 1 if ovmax >= iou_thres[i] else v for i, v in enumerate(cnt_yes)]

            cnt_gt_cls = [v + 1 if i == category_id else v for i, v in enumerate(cnt_gt_cls)]
            cnt_yes_cls = [
                v + 1 if i == category_id and ovmax >= 0.5 else v for i, v in enumerate(cnt_yes_cls)
            ]

            # print(ovmax, jmax)
            # print(cnt_yes, cnt_gt, [1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)])
    print(cnt_yes, cnt_gt)
    print([1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)])

    return [1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)], [
        1.0 * a / b if b > 0 else 0 for a, b in zip(cnt_yes_cls, cnt_gt_cls)
    ]


def recall_ss(dataset_dicts, file_in, max_num_box):
    print(sio.loadmat(file_in))

    raw_data = sio.loadmat(file_in)["boxes"].ravel()
    assert raw_data.shape[0] == len(dataset_dicts)

    cnt_yes_cls = [
        0,
    ] * 20
    cnt_gt_cls = [
        0,
    ] * 20

    cnt_yes = [0 for _ in iou_thres]
    cnt_gt = [0 for _ in iou_thres]
    for i, d in enumerate(dataset_dicts):
        if i % 1000 == 0:
            print("{}/{}".format(i + 1, len(dataset_dicts)))

        # selective search boxes are 1-indexed and (y1, x1, y2, x2)
        i_boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
        # i_scores = np.zeros((i_boxes.shape[0]), dtype=np.float32)
        i_scores = np.ones((i_boxes.shape[0]), dtype=np.float32)

        # -------------------------------------------------------------------------------
        # sort by confidence
        # sorted_ind = np.argsort(-(i_scores.flatten()))
        # i_boxes = i_boxes[sorted_ind, ...]
        # i_scores = i_scores[sorted_ind, ...]

        # i_boxes = i_boxes[:max_num_box, ...]
        # i_scores = i_scores[:max_num_box, ...]

        # random
        number_of_rows = i_boxes.shape[0]
        random_indices = np.random.choice(
            number_of_rows, size=min(number_of_rows, max_num_box), replace=False
        )
        i_boxes = i_boxes[random_indices, ...]
        i_scores = i_scores[random_indices, ...]
        # -------------------------------------------------------------------------------

        for a in d["annotations"]:
            bbgt = a["bbox"]
            category_id = a["category_id"]

            if "coco" in dataset_name:
                bbgt = [bbgt[0], bbgt[1], bbgt[0] + bbgt[2], bbgt[1] + bbgt[3]]

            ixmin = np.maximum(i_boxes[:, 0], bbgt[0])
            iymin = np.maximum(i_boxes[:, 1], bbgt[1])
            ixmax = np.minimum(i_boxes[:, 2], bbgt[2])
            iymax = np.minimum(i_boxes[:, 3], bbgt[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            intersection = iw * ih

            # unionon
            union = (
                (bbgt[2] - bbgt[0] + 1.0) * (bbgt[3] - bbgt[1] + 1.0)
                + (i_boxes[:, 2] - i_boxes[:, 0] + 1.0) * (i_boxes[:, 3] - i_boxes[:, 1] + 1.0)
                - intersection
            )

            overlaps = intersection / union
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            cnt_gt = [v + 1 for i, v in enumerate(cnt_gt)]
            cnt_yes = [v + 1 if ovmax >= iou_thres[i] else v for i, v in enumerate(cnt_yes)]

            cnt_gt_cls = [v + 1 if i == category_id else v for i, v in enumerate(cnt_gt_cls)]
            cnt_yes_cls = [
                v + 1 if i == category_id and ovmax >= 0.5 else v for i, v in enumerate(cnt_yes_cls)
            ]

            # print(ovmax, jmax)
            # print(cnt_yes, cnt_gt, [1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)])
    print(cnt_yes, cnt_gt)
    print([1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)])

    return [1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)], [
        1.0 * a / b if b > 0 else 0 for a, b in zip(cnt_yes_cls, cnt_gt_cls)
    ]


def recall_eb(dataset_dicts, file_in, max_num_box):
    print(sio.loadmat(file_in))

    mat_data = sio.loadmat(file_in)
    boxes_data = mat_data["boxes"].ravel()
    scores_data = mat_data["boxScores"].ravel()
    assert boxes_data.shape[0] == len(dataset_dicts)
    assert scores_data.shape[0] == len(dataset_dicts)

    cnt_yes_cls = [
        0,
    ] * 20
    cnt_gt_cls = [
        0,
    ] * 20

    cnt_yes = [0 for _ in iou_thres]
    cnt_gt = [0 for _ in iou_thres]
    for i, d in enumerate(dataset_dicts):
        if i % 1000 == 0:
            print("{}/{}".format(i + 1, len(dataset_dicts)))

        # selective search boxes are 1-indexed and (y1, x1, y2, x2)
        i_boxes = boxes_data[i][:, (1, 0, 3, 2)] - 1
        i_scores = scores_data[i][:]

        # -------------------------------------------------------------------------------
        # sort by confidence
        sorted_ind = np.argsort(-(i_scores.flatten()))
        i_boxes = i_boxes[sorted_ind, ...]
        i_scores = i_scores[sorted_ind, ...]

        i_boxes = i_boxes[:max_num_box, ...]
        i_scores = i_scores[:max_num_box, ...]

        # -------------------------------------------------------------------------------

        for a in d["annotations"]:
            # print(a["bbox"])
            category_id = a["category_id"]
            bbgt = a["bbox"]

            if "coco" in dataset_name:
                bbgt = [bbgt[0], bbgt[1], bbgt[0] + bbgt[2], bbgt[1] + bbgt[3]]

            ixmin = np.maximum(i_boxes[:, 0], bbgt[0])
            iymin = np.maximum(i_boxes[:, 1], bbgt[1])
            ixmax = np.minimum(i_boxes[:, 2], bbgt[2])
            iymax = np.minimum(i_boxes[:, 3], bbgt[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            intersection = iw * ih

            # unionon
            union = (
                (bbgt[2] - bbgt[0] + 1.0) * (bbgt[3] - bbgt[1] + 1.0)
                + (i_boxes[:, 2] - i_boxes[:, 0] + 1.0) * (i_boxes[:, 3] - i_boxes[:, 1] + 1.0)
                - intersection
            )

            overlaps = intersection / union
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            cnt_gt = [v + 1 for i, v in enumerate(cnt_gt)]
            cnt_yes = [v + 1 if ovmax >= iou_thres[i] else v for i, v in enumerate(cnt_yes)]

            cnt_gt_cls = [v + 1 if i == category_id else v for i, v in enumerate(cnt_gt_cls)]
            cnt_yes_cls = [
                v + 1 if i == category_id and ovmax >= 0.5 else v for i, v in enumerate(cnt_yes_cls)
            ]

            # print(ovmax, jmax)
            # print(cnt_yes, cnt_gt, [1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)])
    print(cnt_yes, cnt_gt)
    print([1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)])

    return [1.0 * a / b for a, b in zip(cnt_yes, cnt_gt)], [
        1.0 * a / b if b > 0 else 0 for a, b in zip(cnt_yes_cls, cnt_gt_cls)
    ]


if __name__ == "__main__":
    print(sys.argv)

    dataset_name = sys.argv[1]
    method = sys.argv[2]

    dir_or_file_in = sys.argv[3]
    if len(sys.argv) > 4:
        dir_or_file_in2 = sys.argv[4]

    assert sys.argv[1] == "voc_2007_test"
    print("Loading", dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    all_recall = []
    cls_recall = []
    for max_num_box in max_num_boxes:
        print("max_num_box", max_num_box)
        if "ssopg" in method.lower() and "aeopg" in method.lower():
            recall_1, recall_2 = recall_huwsod(
                dataset_dicts, dir_or_file_in, dir_or_file_in2, max_num_box
            )
        elif "ssopg" in method.lower():
            recall_1, recall_2 = recall_opg(dataset_dicts, dir_or_file_in, max_num_box)
        elif "aeopg" in method.lower():
            recall_1, recall_2 = recall_opg(dataset_dicts, dir_or_file_in, max_num_box)
        elif "mcg" in method.lower():
            recall_1, recall_2 = recall_mcg(dataset_dicts, dir_or_file_in, max_num_box)
        elif "ss" in method.lower():
            recall_1, recall_2 = recall_ss(dataset_dicts, dir_or_file_in, max_num_box)
        elif "eb" in method.lower():
            recall_1, recall_2 = recall_eb(dataset_dicts, dir_or_file_in, max_num_box)
        all_recall.append(recall_1)
        cls_recall.append(recall_2)

    print("*" * 100)
    print("recall")
    for recall in all_recall:
        print(recall)
    print("*" * 100)
    print("all_recall")
    print(all_recall)
    print("cls_recall")
    print(cls_recall)
