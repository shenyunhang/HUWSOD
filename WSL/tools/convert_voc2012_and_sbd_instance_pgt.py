# -*- coding: utf-8 -*-
import argparse
import json
import os

import cv2
import numpy as np
from skimage import measure
from tqdm import tqdm

import pycocotools.mask as mask_util
import wsl.data.datasets
from detectron2.data.catalog import DatasetCatalog

anno_dir = "datasets/VOC_SBD/annotations/"
json_paths = {"voc": "", "sbd": ""}


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--voc", help="predicted json file", required=True)
    parser.add_argument("--sbd", help="predicted json file", required=True)

    return parser.parse_args()


categories_list = [
    {"supercategory": "none", "id": 1, "name": "aeroplane"},
    {"supercategory": "none", "id": 2, "name": "bicycle"},
    {"supercategory": "none", "id": 3, "name": "bird"},
    {"supercategory": "none", "id": 4, "name": "boat"},
    {"supercategory": "none", "id": 5, "name": "bottle"},
    {"supercategory": "none", "id": 6, "name": "bus"},
    {"supercategory": "none", "id": 7, "name": "car"},
    {"supercategory": "none", "id": 8, "name": "cat"},
    {"supercategory": "none", "id": 9, "name": "chair"},
    {"supercategory": "none", "id": 10, "name": "cow"},
    {"supercategory": "none", "id": 11, "name": "diningtable"},
    {"supercategory": "none", "id": 12, "name": "dog"},
    {"supercategory": "none", "id": 13, "name": "horse"},
    {"supercategory": "none", "id": 14, "name": "motorbike"},
    {"supercategory": "none", "id": 15, "name": "person"},
    {"supercategory": "none", "id": 16, "name": "pottedplant"},
    {"supercategory": "none", "id": 17, "name": "sheep"},
    {"supercategory": "none", "id": 18, "name": "sofa"},
    {"supercategory": "none", "id": 19, "name": "train"},
    {"supercategory": "none", "id": 20, "name": "tvmonitor"},
]


def read_txt(path, split="train"):
    txt_path = os.path.join(path, "{}.txt".format(split))
    with open(txt_path) as f:
        ids = f.readlines()
    return ids


def generate_image_info(img_path, images_info):
    img = cv2.imread(img_path)

    img_name = img_path.split("/")[-1]

    img_w = img.shape[1]
    img_h = img.shape[0]

    info = {"file_name": img_name, "height": img_h, "width": img_w, "id": img_name[:-4]}
    images_info.append(info)

    return images_info


def save_json(ann, path, split="train"):
    os.system("mkdir -p {}".format(path))
    instance_path = os.path.join(path, "{}.json".format(split))
    with open(instance_path, "w") as f:
        json.dump(ann, f)


def load_json(json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def rle_to_binary_mask(rle):
    # print(rle)
    # print(rle.encode('utf-8'))
    # print(str.encode(rle))
    # mask = mask_util.decode(rle.encode('utf-8'))
    # mask = mask_util.decode(str.encode(rle))
    # mask = mask_util.decode([str.encode(rle)])
    mask = mask_util.decode(rle)
    # mask=np.reshape(mask,size)
    return mask


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


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


def convert_voc_sbd(split, save_split):
    dataset_dicts = DatasetCatalog.get(split)
    # print(dataset_dicts[0])

    id2label_dict = {}
    images_info = []
    for dataset_dict in tqdm(dataset_dicts):
        image_id = dataset_dict["image_id"]
        image_labels = []
        for anno in dataset_dict["annotations"]:
            # 0-based continous id
            category_id = anno["category_id"] + 1
            image_labels.append(category_id)
        image_labels = list(set(image_labels))
        id2label_dict[image_id] = image_labels

        img_path = dataset_dict["file_name"]
        assert os.path.isfile(img_path)
        images_info = generate_image_info(img_path, images_info)

    # print(id2label_dict)

    annotations = []

    json_data = load_json(json_paths[split])
    # print(json_data)
    # print(len(json_data))
    # print(json_data[0])

    count = 0
    used_id = []
    for j_d in tqdm(json_data):
        print(j_d)
        image_id = j_d["image_id"]
        category_id = j_d["category_id"]
        bbox = j_d["bbox"]
        score = j_d["score"]
        segmentation = j_d["segmentation"]

        idid = image_id + "-" + str(category_id)
        # print(idid, j_d['score'])

        if category_id not in id2label_dict[image_id]:
            continue

        # print('\t', idid, j_d['score'])

        if idid in used_id and score < 0.8:
            continue
        used_id.append(idid)

        # print('\t\t', idid, j_d['score'])

        mask = rle_to_binary_mask(segmentation)
        # print(mask, mask.shape, np.max(mask), np.min(mask), area)

        polys = binary_mask_to_polygon(mask)
        area = int(np.sum(mask))

        if len(polys) == 0:
            continue

        anno = {
            "segmentation": polys,
            "area": area,
            "image_id": image_id,
            "bbox": bbox,
            "iscrowd": 0,
            "category_id": category_id,
            "id": count,
        }
        count += 1

        annotations.append(anno)

    voc_instance = {
        "images": images_info,
        "annotations": annotations,
        "categories": categories_list,
    }
    save_json(voc_instance, anno_dir, split=save_split)


if __name__ == "__main__":
    args = parse_arguments()
    json_paths["voc_2012_train_instance"] = args.voc
    json_paths["sbd_9118_instance"] = args.sbd
    convert_voc_sbd("voc_2012_train_instance", "voc_2012_train_instance_pgt")
    convert_voc_sbd("sbd_9118_instance", "sbd_9118_instance_pgt")
