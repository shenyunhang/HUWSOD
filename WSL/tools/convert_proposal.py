from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
from six.moves import cPickle as pickle
from tqdm import tqdm

import wsl.data.datasets
from detectron2.data.catalog import DatasetCatalog
from detectron2.utils.file_io import PathManager


def convert_ss_box():
    args = parse_args()
    dataset_name = args.dataset_name
    file_in = args.input
    file_out = args.output

    dataset_dicts = DatasetCatalog.get(dataset_name)
    raw_data = sio.loadmat(file_in)["boxes"].ravel()
    assert raw_data.shape[0] == len(dataset_dicts)

    boxes = []
    scores = []
    ids = []
    for i in range(len(dataset_dicts)):
        if i % 1000 == 0:
            print("{}/{}".format(i + 1, len(dataset_dicts)))

        if "flickr" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        elif "coco" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        else:
            index = dataset_dicts[i]["image_id"]
        # selective search boxes are 1-indexed and (y1, x1, y2, x2)
        i_boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
        # i_scores = np.zeros((i_boxes.shape[0]), dtype=np.float32)
        i_scores = np.ones((i_boxes.shape[0]), dtype=np.float32)

        boxes.append(i_boxes.astype(np.int16))
        scores.append(np.squeeze(i_scores.astype(np.float32)))
        index = dataset_dicts[i]["image_id"]
        ids.append(index)

    with open(file_out, "wb") as f:
        pickle.dump(dict(boxes=boxes, scores=scores, indexes=ids), f, pickle.HIGHEST_PROTOCOL)


def convert_mcg_box():
    args = parse_args()
    dataset_name = args.dataset_name
    dir_in = args.input
    file_out = args.output

    dataset_dicts = DatasetCatalog.get(dataset_name)

    boxes = []
    scores = []
    ids = []
    for i in range(len(dataset_dicts)):
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

        boxes.append(boxes_data.astype(np.int16))
        scores.append(np.squeeze(scores_data.astype(np.float32)))
        index = dataset_dicts[i]["image_id"]
        ids.append(index)

    with open(file_out, "wb") as f:
        pickle.dump(dict(boxes=boxes, scores=scores, indexes=ids), f, pickle.HIGHEST_PROTOCOL)


def convert_det_box():
    args = parse_args()
    dataset_name = args.dataset_name
    file_in = args.input
    file_out = args.output

    dataset_dicts = DatasetCatalog.get(dataset_name)

    print("Loading: ", args.input)
    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    boxes = []
    scores = []
    ids = []
    for i in range(len(dataset_dicts)):
        if i % 1000 == 0:
            print("{}/{}".format(i + 1, len(dataset_dicts)))

        image_id = dataset_dicts[i]["image_id"]

        prediction = pred_by_image[image_id]
        score = np.asarray([x["score"] for x in prediction])
        bbox = np.asarray([x["bbox"] for x in prediction]).reshape(-1, 4)
        # XYWH_ABS to XYXY_ABS)
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]

        assert bbox.shape[0] > 0

        boxes.append(bbox.astype(np.int16))
        scores.append(np.squeeze(score.astype(np.float32)))
        index = dataset_dicts[i]["image_id"]
        ids.append(index)

    with open(file_out, "wb") as f:
        pickle.dump(dict(boxes=boxes, scores=scores, indexes=ids), f, pickle.HIGHEST_PROTOCOL)


# for i in tqdm(range(len(dataset_dicts)), position=0):
def convert_mcg_seg_i(dataset_dict, dataset_name, dir_in, dir_out):

    if "flickr" in dataset_name:
        index = os.path.basename(dataset_dict["file_name"])[:-4]
    elif "coco" in dataset_name:
        index = os.path.basename(dataset_dict["file_name"])[:-4]
    else:
        index = dataset_dict["image_id"]
    segm_file = os.path.join(dir_in, "{}.mat".format(index))
    print(segm_file)

    index = dataset_dict["image_id"]
    save_path = os.path.join(dir_out, str(index) + ".pkl")
    if os.path.isfile(save_path):
        return

    mat_data = sio.loadmat(segm_file)
    # print(mat_data.keys())
    # print('labels: ', mat_data['labels'])
    # print('scores: ', mat_data['scores'])
    # print('superpixels_i: ', mat_data['superpixels_i'])
    # print('labels: ',mat_data['labels'].shape)
    # print('scores: ', mat_data['scores'].shape)
    # print('superpixels_i: ', mat_data['superpixels_i'].shape)
    # print('superpixels_i: ', mat_data['superpixels_i'].max())
    # print('superpixels_i: ', mat_data['superpixels_i'].min())

    superpixels_i = mat_data["superpixels"]
    labels_i = mat_data["labels"]
    scores_i = mat_data["scores"]

    # 1-based to 0-based
    superpixels_i = superpixels_i - 1

    mask_h, mask_w = superpixels_i.shape
    num_proposals = labels_i.shape[0]
    num_superpixels = superpixels_i.max() + 1

    boxes_i = np.zeros((num_proposals, 4), dtype=np.int16)
    oh_labels_i = np.zeros((num_proposals, num_superpixels), dtype=np.bool)

    poses = []
    for l in range(num_superpixels):
        pos = np.where(superpixels_i == l)
        poses.append(pos)

    # for j in tqdm(range(num_proposals), position=int(dataset_dict['image_id'])%20):
    for j in range(num_proposals):
        x1 = mask_w - 1
        y1 = mask_h - 1
        x2 = 0
        y2 = 0
        for label in labels_i[j]:
            for l in np.nditer(label):
                # 1-based to 0-based
                l = l - 1
                oh_labels_i[j, l] = 1

                pos = poses[l]

                y1 = min(y1, pos[0].min())
                x1 = min(x1, pos[1].min())
                y2 = max(y2, pos[0].max())
                x2 = max(x2, pos[1].max())

        boxes_i[j, 0] = x1
        boxes_i[j, 1] = y1
        boxes_i[j, 2] = x2
        boxes_i[j, 3] = y2

    # if False:
    if True:
        index = dataset_dict["image_id"]
        home = str(Path.home())
        # for j in tqdm(range(num_proposals), position=1):
        for j in range(num_proposals):
            size = 1.0 * (boxes_i[j, 2] - boxes_i[j, 0]) * (boxes_i[j, 3] - boxes_i[j, 1])
            if size < 10000:  # 100 * 100
                continue

            img = cv2.imread(dataset_dict["file_name"])

            for l, v in enumerate(oh_labels_i[j]):
                if v == 0:
                    continue
                pos = np.where(superpixels_i == l)

                img[pos] = img[pos] * 0.5 + [0, 0, 125]

            cv2.rectangle(
                img,
                (boxes_i[j, 0], boxes_i[j, 1]),
                (boxes_i[j, 2], boxes_i[j, 3]),
                (0, 0, 255),
                thickness=4,
            )

            save_path = os.path.join(home, "tmp", str(index) + "_" + str(j) + ".png")
            cv2.imwrite(save_path, img)

    index = dataset_dict["image_id"]

    save_path = os.path.join(dir_out, str(index) + ".pkl")
    with open(save_path, "wb") as f:
        pickle.dump(
            dict(
                superpixels=superpixels_i.astype(np.int16),
                oh_labels=oh_labels_i.astype(np.bool),
                scores=np.squeeze(scores_i.astype(np.float32)),
                boxes=boxes_i.astype(np.int16),
                indexes=index,
            ),
            f,
            pickle.HIGHEST_PROTOCOL,
        )


def convert_mcg_seg():
    args = parse_args()
    dataset_name = args.dataset_name
    dir_in = args.input
    dir_out = args.output

    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    dataset_dicts = DatasetCatalog.get(dataset_name)

    process_pool = Pool(processes=32)

    arg_process = []
    for i in range(len(dataset_dicts)):
        arg_process.append((dataset_dicts[i], dataset_name, dir_in, dir_out))

    results = process_pool.starmap_async(convert_mcg_seg_i, arg_process)
    results = results.get()


def convert_mcg_seg2():
    args = parse_args()
    dataset_name = args.dataset_name
    dir_in = args.input
    dir_out = args.output

    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    dataset_dicts = DatasetCatalog.get(dataset_name)

    for i in tqdm(range(len(dataset_dicts)), position=0):
        if "flickr" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        elif "coco" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        else:
            index = dataset_dicts[i]["image_id"]
        segm_file = os.path.join(dir_in, "{}.mat".format(index))
        mat_data = sio.loadmat(segm_file)
        # print(mat_data.keys())
        # print('labels: ', mat_data['labels'])
        # print('scores: ', mat_data['scores'])
        # print('superpixels: ', mat_data['superpixels'])
        # print('labels: ',mat_data['labels'].shape)
        # print('scores: ', mat_data['scores'].shape)
        # print('superpixels: ', mat_data['superpixels'].shape)
        # print('superpixels: ', mat_data['superpixels'].max())
        # print('superpixels: ', mat_data['superpixels'].min())

        superpixels = mat_data["superpixels"]
        labels = mat_data["labels"]
        scores = mat_data["scores"]

        # 1-based to 0-based
        superpixels = superpixels - 1

        mask_h, mask_w = superpixels.shape
        num_proposals = labels.shape[0]
        num_superpixels = superpixels.max() + 1

        proposals = np.zeros((num_proposals, mask_h, mask_w), dtype=np.bool)
        boxes = np.zeros((num_proposals, 4), dtype=np.int16)
        oh_labels_i = np.zeros((num_proposals, num_superpixels), dtype=np.bool)

        poses = []
        for l in range(num_superpixels):
            pos = np.where(superpixels == l)
            poses.append(pos)

        for j in tqdm(range(num_proposals), position=1):
            x1 = mask_w - 1
            y1 = mask_h - 1
            x2 = 0
            y2 = 0
            for label in labels[j]:
                for l in np.nditer(label):
                    # 1-based to 0-based
                    l = l - 1
                    oh_labels_i[j, l] = 1

                    pos = poses[l]

                    y1 = min(y1, pos[0].min())
                    x1 = min(x1, pos[1].min())
                    y2 = max(y2, pos[0].max())
                    x2 = max(x2, pos[1].max())

                    proposals[j][pos] = True

            boxes[j, 0] = x1
            boxes[j, 1] = y1
            boxes[j, 2] = x2
            boxes[j, 3] = y2

        if False:
            index = dataset_dicts[i]["image_id"]
            home = str(Path.home())
            for j in tqdm(range(num_proposals), position=1):
                img = cv2.imread(dataset_dicts[i]["file_name"])

                for l, v in enumerate(oh_labels_i[j]):
                    if v == 0:
                        continue
                    pos = np.where(superpixels == l)

                    img[pos] = 0

                cv2.rectangle(
                    img,
                    (boxes[j, 0], boxes[j, 1]),
                    (boxes[j, 2], boxes[j, 3]),
                    (0, 0, 255),
                    thickness=4,
                )

                save_path = os.path.join(home, "tmp", str(index) + "_" + str(j) + ".png")
                cv2.imwrite(save_path, img)

        index = dataset_dicts[i]["image_id"]

        save_path = os.path.join(dir_out, str(index) + ".pkl")
        with open(save_path, "wb") as f:
            pickle.dump(
                dict(
                    # proposals=np.packbits(proposals, axis=None),
                    proposals=proposals.astype(np.bool),
                    scores=scores.astype(np.float32),
                    boxes=boxes.astype(np.int16),
                    indexes=index,
                ),
                f,
                pickle.HIGHEST_PROTOCOL,
            )


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Convert proposals")
    parser.add_argument(
        "--source",
        choices=["ss", "mcg", "det"],
        required=True,
        help="proposal sources",
    )
    parser.add_argument(
        "--type",
        choices=["bbox", "segm"],
        required=True,
        help="proposal types",
    )
    parser.add_argument("--dataset-name", required=True, help="dataset name")
    parser.add_argument("--input", default="./", help="input")
    parser.add_argument("--output", default="./", help="out")
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    if args.type == "segm":
        if args.source == "mcg":
            convert_mcg_seg()
    else:
        if args.source == "ss":
            convert_ss_box()
        elif args.source == "mcg":
            convert_mcg_box()
        elif args.source == "det":
            convert_det_box()

    exit(0)

    if "-proposals" in sys.argv[2].lower():
        if "mcg" in sys.argv[3].lower():
            convert_mcg_seg()
    else:
        if "ss" in sys.argv[3].lower():
            convert_ss_box()
        elif "mcg" in sys.argv[3].lower():
            convert_mcg_box()
