import json
import os
import random

import cv2
import numpy as np
import scipy.io as scio
from PIL import Image
from skimage import measure
from tqdm import tqdm

from panopticapi.utils import IdGenerator

root_sbd = "datasets/SBD"
instance_dir_sbd = os.path.join(root_sbd, "inst/")
semantic_dir_sbd = os.path.join(root_sbd, "cls/")
image_dir_sbd = os.path.join(root_sbd, "img/")
txt_dir_sbd = root_sbd


root_voc = "datasets/VOC2012"
instance_dir_voc = os.path.join(root_voc, "SegmentationObject/")
semantic_dir_voc = os.path.join(root_voc, "SegmentationClass/")
image_dir_voc = os.path.join(root_voc, "JPEGImages/")
txt_dir_voc = os.path.join(root_voc, "ImageSets/Segmentation/")

image_dir = os.path.join("datasets", "VOC_SBD", "images")
label_dir = os.path.join("datasets", "VOC_SBD", "annotations")
panoptic_dir = os.path.join("datasets", "VOC_SBD", "annotations", "panoptic")

C = 21


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


voc_cmaps = labelcolormap(C).tolist()

categories_list = [
    {"supercategory": "none", "id": 1, "name": "aeroplane", "isthing": 1, "color": voc_cmaps[1]},
    {"supercategory": "none", "id": 2, "name": "bicycle", "isthing": 1, "color": voc_cmaps[2]},
    {"supercategory": "none", "id": 3, "name": "bird", "isthing": 1, "color": voc_cmaps[3]},
    {"supercategory": "none", "id": 4, "name": "boat", "isthing": 1, "color": voc_cmaps[4]},
    {"supercategory": "none", "id": 5, "name": "bottle", "isthing": 1, "color": voc_cmaps[5]},
    {"supercategory": "none", "id": 6, "name": "bus", "isthing": 1, "color": voc_cmaps[6]},
    {"supercategory": "none", "id": 7, "name": "car", "isthing": 1, "color": voc_cmaps[7]},
    {"supercategory": "none", "id": 8, "name": "cat", "isthing": 1, "color": voc_cmaps[8]},
    {"supercategory": "none", "id": 9, "name": "chair", "isthing": 1, "color": voc_cmaps[9]},
    {"supercategory": "none", "id": 10, "name": "cow", "isthing": 1, "color": voc_cmaps[10]},
    {
        "supercategory": "none",
        "id": 11,
        "name": "diningtable",
        "isthing": 1,
        "color": voc_cmaps[11],
    },
    {"supercategory": "none", "id": 12, "name": "dog", "isthing": 1, "color": voc_cmaps[12]},
    {"supercategory": "none", "id": 13, "name": "horse", "isthing": 1, "color": voc_cmaps[13]},
    {"supercategory": "none", "id": 14, "name": "motorbike", "isthing": 1, "color": voc_cmaps[14]},
    {"supercategory": "none", "id": 15, "name": "person", "isthing": 1, "color": voc_cmaps[15]},
    {
        "supercategory": "none",
        "id": 16,
        "name": "pottedplant",
        "isthing": 1,
        "color": voc_cmaps[16],
    },
    {"supercategory": "none", "id": 17, "name": "sheep", "isthing": 1, "color": voc_cmaps[17]},
    {"supercategory": "none", "id": 18, "name": "sofa", "isthing": 1, "color": voc_cmaps[18]},
    {"supercategory": "none", "id": 19, "name": "train", "isthing": 1, "color": voc_cmaps[19]},
    {"supercategory": "none", "id": 20, "name": "tvmonitor", "isthing": 1, "color": voc_cmaps[20]},
    {
        "supercategory": "none",
        "id": 21,
        "name": "background",
        "isthing": 0,
        "color": [255, 255, 255],
    },
]

categories = {category["id"]: category for category in categories_list}
id_generator = IdGenerator(categories)


used_id = []


def get_unique_id():
    i = random.randint(1, 255 * 255 * 255)
    while i in used_id:
        i = random.randint(1, 255 * 255 * 255)
    return i


def read_txt(path, split="train"):
    txt_path = os.path.join(path, "{}.txt".format(split))
    with open(txt_path) as f:
        ids = f.readlines()
    return ids


def get_mask_voc(inst_path, seg_path):
    cmaps = labelcolormap(C)
    cmaps = np.vstack((cmaps, np.array([224, 224, 192])))

    seg_cls_mat = cv2.imread(seg_path)
    seg_cls_mat = cv2.cvtColor(seg_cls_mat, cv2.COLOR_BGR2RGB)

    # cmaps_this = np.unique(np.reshape(seg_cls_mat, (-1, 3)), axis=0)
    # print('seg_cls_mat color map: ', cmaps_this)

    semantic_mask = np.zeros(seg_cls_mat.shape[:-1], dtype=np.uint8)
    semantic_mask.fill(255)
    for i in range(cmaps.shape[0]):
        cmap = cmaps[i]

        mask_0 = seg_cls_mat[:, :, 0] == cmap[0]
        mask_1 = seg_cls_mat[:, :, 1] == cmap[1]
        mask_2 = seg_cls_mat[:, :, 2] == cmap[2]

        mask = mask_0 * mask_1 * mask_2
        print("semantic_mask: ", i, cmap, mask.shape, mask.sum())

        if i == 0:
            semantic_mask[mask] = 0
        elif i == cmaps.shape[0] - 1:
            semantic_mask[mask] = 255
        else:
            semantic_mask[mask] = i
    # cv2.imwrite('semantic_mask.png', semantic_mask)
    # print(semantic_mask.mean())

    seg_obj_mat = cv2.imread(inst_path)
    seg_obj_mat = cv2.cvtColor(seg_obj_mat, cv2.COLOR_BGR2RGB)
    omaps = np.unique(np.reshape(seg_obj_mat, (-1, 3)), axis=0)
    # print('seg_obj_mat corlor map: ', omaps)

    instance_mask = np.zeros(seg_obj_mat.shape[:-1], dtype=np.uint8)
    instance_mask.fill(255)
    cnt = 1
    for i in range(len(omaps)):
        omap = omaps[i]

        mask_0 = seg_obj_mat[:, :, 0] == omap[0]
        mask_1 = seg_obj_mat[:, :, 1] == omap[1]
        mask_2 = seg_obj_mat[:, :, 2] == omap[2]

        mask = mask_0 * mask_1 * mask_2
        print("instancce_mask: ", i, omap, mask.shape, mask.sum())

        if omap[0] == cmaps[0][0] and omap[1] == cmaps[0][1] and omap[2] == cmaps[0][2]:
            instance_mask[mask] = 0
        elif omap[0] == cmaps[-1][0] and omap[1] == cmaps[-1][1] and omap[2] == cmaps[-1][2]:
            instance_mask[mask] = 255
        else:
            instance_mask[mask] = cnt
            cnt = cnt + 1
    # cv2.imwrite('instance_mask.png', instance_mask)
    # print(instance_mask.mean())

    return semantic_mask, instance_mask


def get_mask_sbd(inst_path, seg_path):
    seg_cls_mat = scio.loadmat(seg_path)
    semantic_mask = seg_cls_mat["GTcls"]["Segmentation"][0][0]

    seg_obj_mat = scio.loadmat(inst_path)
    instance_mask = seg_obj_mat["GTinst"]["Segmentation"][0][0]
    # instance_cate = seg_obj_mat['GTinst']['Categories'][0][0]

    # print('seg_obj_mat: ', seg_obj_mat.keys(), seg_obj_mat['GTinst'].dtype, seg_obj_mat['GTinst']['Segmentation'].shape, type(seg_obj_mat['GTinst']), type(seg_obj_mat['GTinst']['Segmentation']), seg_obj_mat['GTinst'])
    # print('seg_cls_mat: ', seg_cls_mat.keys(), seg_cls_mat['GTcls'].dtype, seg_cls_mat['GTcls']['Segmentation'].shape, type(seg_cls_mat['GTcls']), type(seg_cls_mat['GTcls']['Segmentation']), seg_cls_mat['GTcls'])

    return semantic_mask, instance_mask


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


def generate_anno(inst_path, seg_path, images_info, annotations, count, mode, N=128):
    if inst_path.endswith(".mat"):
        semantic_mask, instance_mask = get_mask_sbd(inst_path, seg_path)
    else:
        semantic_mask, instance_mask = get_mask_voc(inst_path, seg_path)

    instance_ids = np.unique(instance_mask)
    print("instance_ids: ", instance_ids)

    # print(instance_mask.shape)
    # print(instance_ids, instance_cate.shape)
    # assert len(instance_ids) == instance_cate.shape[0]

    img_name = inst_path.split("/")[-1]

    imw = instance_mask.shape[1]
    imh = instance_mask.shape[0]

    has_object = False
    panoptic_map = np.zeros((imh, imw, 3), dtype=np.uint8)
    segments_info = []
    for i, instance_id in enumerate(instance_ids):
        if instance_id == 255:  # edge, pass
            continue

        # extract instance
        temp = np.zeros(instance_mask.shape)
        temp.fill(instance_id)
        tempMask = instance_mask == temp
        cat_id = np.max(np.unique(semantic_mask * tempMask))  # semantic category of this instance
        instance = instance_mask * tempMask
        instance_temp = instance.copy()  # findContours will change instance, so copy first

        if cat_id == 0:  # background
            cat_id = len(categories_list)

        # id_this = get_unique_id()
        segment_id, color = id_generator.get_id_and_color(int(cat_id))
        panoptic_map[tempMask] = color

        area = int(np.sum(tempMask))
        x, y, w, h = cv2.boundingRect(instance_temp.astype(np.uint8))
        segments_info.append(
            {
                "id": segment_id,
                "category_id": int(cat_id),
                "area": area,
                "bbox": [x, y, w, h],
                "iscrowd": 0,
            }
        )
        count += 1

        has_object = True

    if has_object:
        info = {
            "file_name": img_name[:-4] + ".jpg",
            "height": imh,
            "width": imw,
            "id": img_name[:-4],
        }
        images_info.append(info)

        anno = {
            "image_id": img_name[:-4],
            "file_name": img_name[:-4] + ".png",
            "segments_info": segments_info,
        }
        annotations.append(anno)

        save_png(panoptic_map, panoptic_dir, img_name[:-4] + ".png")
    else:
        print(img_name)
        exit(0)

    return images_info, annotations, count


def save_png(panoptic_map, path, name):
    os.system("mkdir -p {}".format(path))
    Image.fromarray(panoptic_map).save(os.path.join(path, name), format="PNG")


def save_json(ann, path, split="train"):
    os.system("mkdir -p {}".format(path))
    instance_path = os.path.join(path, "{}_panoptic.json".format(split))
    with open(instance_path, "w") as f:
        json.dump(ann, f)


def save_images(ann, path):
    os.system("mkdir -p {}".format(path))
    images = ann["images"]
    for image in tqdm(images):
        file_name = image["file_name"]
        file_path = os.path.join(image_dir_sbd, file_name)
        if not os.path.isfile(file_path):
            file_path = os.path.join(image_dir_voc, file_name)
        os.system("cp {} {}".format(file_path, path))


def convert_labels(ids, split, mode):
    images = []
    annotations = []
    label_save_dir = label_dir
    image_save_dir = image_dir
    count = 0
    for i in tqdm(range(len(ids))):
        inst_path = os.path.join(instance_dir_sbd, ids[i][:-1] + ".mat")
        seg_path = os.path.join(semantic_dir_sbd, ids[i][:-1] + ".mat")
        if not os.path.isfile(inst_path):
            inst_path = os.path.join(instance_dir_voc, ids[i][:-1] + ".png")
            seg_path = os.path.join(semantic_dir_voc, ids[i][:-1] + ".png")
        images, annotations, count = generate_anno(
            inst_path, seg_path, images, annotations, count, mode
        )
    voc_instance = {"images": images, "annotations": annotations, "categories": categories_list}
    save_json(voc_instance, label_save_dir, split=split)
    save_images(voc_instance, image_save_dir)


def convert_sbd():
    # 8498
    ids_train_sbd = read_txt(txt_dir_sbd, "train")

    # 2857
    ids_val_sbd = read_txt(txt_dir_sbd, "val")

    ids_train_voc = read_txt(txt_dir_voc, "train")

    ids_val_voc = read_txt(txt_dir_voc, "val")

    ids_sbd = []
    for i in sorted(list(set(ids_train_sbd + ids_val_sbd))):
        if i in ids_train_voc:
            continue
        if i in ids_val_voc:
            continue
        ids_sbd.append(i)

    # convert_labels(ids_sbd, 'sbd_9118', 'mask')
    # convert_labels(ids_train_voc, 'voc_2012_train', 'mask')
    # convert_labels(ids_val_voc, 'voc_2012_val', 'mask')

    convert_labels(ids_sbd, "sbd_9118", "poly")
    convert_labels(ids_train_voc, "voc_2012_train", "poly")
    convert_labels(ids_val_voc, "voc_2012_val", "poly")


if __name__ == "__main__":
    convert_sbd()
