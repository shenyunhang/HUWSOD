import json
import logging
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from termcolor import colored
from torch import nn
from torch.nn import functional as F

from detectron2.data import MetadataCatalog
from fvcore.nn import giou_loss, smooth_l1_loss
from tabulate import tabulate

logger = logging.getLogger(__name__)


def table_hierarchy(hierarchy_data, table, n):
    table.append(
        [
            n,
        ]
        + [
            "",
        ]
        * n
        + [
            [hierarchy_data["ori_label"], hierarchy_data["name"]],
        ]
    )

    # print(len(table), n)
    # print(n,"-"*n, hierarchy_data["name"])
    for child in hierarchy_data["children"]:
        table = table_hierarchy(child, table, n + 1)

    return table


def print_hierarchy(hierarchy_data):
    table = table_hierarchy(hierarchy_data, [], 1)
    n = max([len(tab) for tab in table])
    table_str = tabulate(
        table,
        tablefmt="pipe",
        headers=[
            "depth",
        ]
        + [str(i) for i in range(n + 1)],
    )
    logger.info("hierarchy: \n" + colored(table_str, "red"))


def get_raw_info(data, info):
    for child in data["children"]:
        info.append(
            {
                "id": child["id"],
                "name": child["name"],
                "def": child["def"],
                "ori_label": child["ori_label"],
            }
        )

    for child in data["children"]:
        info = get_raw_info(child, info)

    return info


def parse_json(hierarchy_data):
    # print(hierarchy_data.keys())
    # print_hierarchy(hierarchy_data)

    parent_list = [
        hierarchy_data,
    ]
    while True:
        if len(parent_list) == 0:
            break
        parent = parent_list.pop(0)

        while True:
            if len(parent["children"]) == 1:
                pass
            else:
                break

            child = parent["children"][0]

            if isinstance(parent["ori_label"], list):
                parent["ori_label"] = parent["ori_label"] + [
                    child["ori_label"],
                ]
            else:
                parent["ori_label"] = [
                    parent["ori_label"],
                    child["ori_label"],
                ]

            parent["children"] = child["children"]

        for child in parent["children"]:
            parent_list.append(child)

    # print_hierarchy(hierarchy_data)

    parents = {}
    childs = {}

    background_data = {
        "id": "background",
        "contiguous_id_parent": None,
        "contiguous_id_parent_all": [],
        "name": "background",
        "def": "background",
        "ori_label": "background",
        "children": [],
    }

    hierarchy_data["contiguous_id_parent"] = None
    hierarchy_data["contiguous_id_parent_all"] = []

    contiguous_id = 0
    info = []

    parent_list = [hierarchy_data, background_data]
    # parent_list = [
    #     hierarchy_data,
    # ]

    wnids = []
    wnids_contiguous_id = []

    second_layer_start = 1
    bg_contiguous_id = None
    while True:
        if len(parent_list) == 0:
            break
        parent = parent_list.pop(0)

        if not isinstance(parent["ori_label"], list) and len(parent["ori_label"]) == 0:
            for child in parent["children"][::-1]:
                child["contiguous_id_parent"] = parent["contiguous_id_parent"]
                child["contiguous_id_parent_all"] = parent["contiguous_id_parent_all"]
                parent_list.insert(0, child)
            continue

        while True:
            if len(parent["children"]) == 1:
                pass
            else:
                break

            child = parent["children"][0]

            # keep children
            if len(child["ori_label"]) > 0:
                if isinstance(parent["ori_label"], list):
                    parent["ori_label"] = parent["ori_label"] + [
                        child["ori_label"],
                    ]
                else:
                    parent["ori_label"] = [
                        parent["ori_label"],
                        child["ori_label"],
                    ]

            parent["children"] = child["children"]

        for child in parent["children"]:
            child["contiguous_id_parent"] = contiguous_id
            child["contiguous_id_parent_all"] = parent["contiguous_id_parent_all"] + [
                contiguous_id,
            ]
            parent_list.append(child)

        info.append(
            {
                "id": parent["id"],
                "contiguous_id": contiguous_id,
                "contiguous_id_parent": parent["contiguous_id_parent"],
                "contiguous_id_parent_all": parent["contiguous_id_parent_all"],
                "name": parent["name"],
                "def": parent["def"],
                "ori_label": parent["ori_label"],
            }
        )

        if parent["contiguous_id_parent"] is None:
            second_layer_start = contiguous_id + 1

        if parent["id"] == "background":
            bg_contiguous_id = contiguous_id

        if isinstance(parent["ori_label"], list):
            for wnid in parent["ori_label"]:
                if len(wnid) > 0 and wnid not in wnids and wnid != "background":
                    wnids.append(wnid)
                    wnids_contiguous_id.append(contiguous_id)

        else:
            if (
                len(parent["ori_label"]) > 0
                and parent["ori_label"] not in wnids
                and parent["ori_label"] != "background"
            ):
                wnids.append(parent["ori_label"])
            wnids_contiguous_id.append(contiguous_id)

        contiguous_id += 1

    # parent = background_data
    # info.append(
    #     {
    #         "id": parent["id"],
    #         "contiguous_id": contiguous_id,
    #         "contiguous_id_parent": parent["contiguous_id_parent"],
    #         "contiguous_id_parent_all": parent["contiguous_id_parent_all"],
    #         "name": parent["name"],
    #         "def": parent["def"],
    #         "ori_label": parent["ori_label"],
    #     }
    # )
    # contiguous_id += 1

    wnids, wnids_contiguous_id = [
        np.array(l) for l in zip(*sorted(zip(wnids, wnids_contiguous_id)))
    ]
    # print([(x, y) for x, y in zip(wnids, wnids_contiguous_id)])
    # print(len(wnids), len(wnids_contiguous_id))
    # print(len(list(set(wnids))))
    # print(len(list(set(wnids_contiguous_id))))

    logger.info("original class: " + str(len(list(set(wnids)))))
    logger.info("mapped ids: " + str(len(list(set(wnids_contiguous_id)))))
    logger.info("hierarchy node: " + str(len(info)))

    parent_child = np.zeros((len(info), len(info)), dtype=bool)
    is_child = np.zeros((len(info), len(info)), dtype=bool)
    is_parent = np.zeros((len(info), len(info)), dtype=bool)

    for line in info:
        if line["contiguous_id_parent"] is None:
            continue
        parent_child[line["contiguous_id_parent"], line["contiguous_id"]] = True

    for line in info:
        for p in line["contiguous_id_parent_all"]:
            is_parent[line["contiguous_id"], p] = True

    for line in info:
        for p in line["contiguous_id_parent_all"]:
            is_child[p, line["contiguous_id"]] = True

    del info

    print("second_layer_start", second_layer_start)
    print("bg_contiguous_id", bg_contiguous_id)
    assert bg_contiguous_id + 1 == second_layer_start
    assert np.all(parent_child[:, bg_contiguous_id + 1 :].sum(axis=0) == 1), parent_child.sum(
        axis=0
    ).tolist()
    assert np.all(parent_child[:, :bg_contiguous_id].sum(axis=0) == 0), parent_child.sum(
        axis=0
    ).tolist()
    assert (is_parent * is_child).sum() == 0
    assert np.all(np.equal(is_parent, np.transpose(is_child)))

    num_child = parent_child.sum(axis=1)
    assert 1 not in num_child
    num_child = num_child[num_child > 0.5]
    # num_child = np.concatenate(([2], num_child))
    num_child = np.concatenate(
        (
            [
                bg_contiguous_id + 1,
            ],
            num_child,
        )
    )
    group_end = np.cumsum(num_child)
    logger.info("num group: " + str(len(num_child)))
    logger.info("num child: " + str(num_child))
    logger.info("group end: " + str(group_end))

    return (
        parent_child,
        is_parent,
        is_child,
        group_end.tolist(),
        wnids_contiguous_id,
        bg_contiguous_id,
    )


def load_class_hierarchy(cfg):
    if not cfg.WSL.HIERARCHY.ENABLED:
        return None

    json_path = cfg.WSL.HIERARCHY.JSON_PATH
    if not os.path.exists(json_path):
        logger.info(json_path, "not found")
        raise ValueError(f"file {json_path} not exist")

    hierarchy_weight = None
    logger.info("Loading " + json_path)
    hierarchy_data = json.load(open(json_path, "r"))
    (
        parent_child,
        is_parent,
        is_child,
        group_end,
        wnids_contiguous_id,
        bg_contiguous_id,
    ) = parse_json(hierarchy_data)

    return parent_child, is_parent, group_end, wnids_contiguous_id, bg_contiguous_id
    # return 1 - is_child, is_parent, is_child, parent_child, group_end, wnids_contiguous_id

    if cfg.WSL.HIERARCHY.POS_PARENTS:
        hierarchy_weight = (1 - is_child, is_parent)
    else:
        hierarchy_weight = 1 - (is_parent + is_child)  # (C + 1) x C

    return hierarchy_weight


class HierarchyProcessor:
    def __init__(self, cfg):
        (
            self.parent_child,
            self.child_parent_all,
            self.group_end,
            self.wnids_contiguous_id,
            self.bg_contiguous_id,
        ) = load_class_hierarchy(cfg)

        self.num_node = len(self.parent_child)
        # self.bg = [
        #     0,
        # ] + self.group_end[:-1]
        # self.ed = [
        #     self.num_node,
        # ] + self.group_end[1:]
        # self.st = [
        #     self.num_node - 1,
        # ] + [1 for _ in range(self.num_node - 1)]
        self.bg = [
            0,
        ] + self.group_end[:-1]
        self.ed = self.group_end[:]
        self.st = [1 for _ in range(self.num_node - 1)]

        print("bg", self.bg[:10], self.bg[-10:])
        print("ed", self.ed[:10], self.ed[-10:])
        print("st", self.st[:10], self.st[-10:])

    def to(self, device):
        self.parent_child = torch.as_tensor(self.parent_child, dtype=torch.bool).to(device)
        self.child_parent = torch.transpose(self.parent_child, 0, 1)

        self.child_parent_all = torch.as_tensor(self.child_parent_all, dtype=torch.bool).to(device)

        self.wnids_contiguous_id = torch.as_tensor(self.wnids_contiguous_id, dtype=torch.int64).to(
            device
        )
        self.bg_contiguous_id = torch.as_tensor([self.bg_contiguous_id], dtype=torch.int64).to(
            device
        )
        self.fgbg_contiguous_id = torch.cat(
            (self.wnids_contiguous_id, self.bg_contiguous_id), dim=0
        )
        # self.fgbg_contiguous_id = self.wnids_contiguous_id
        print("fgbg_contiguous_id", self.fgbg_contiguous_id.size())
        print("fgbg_contiguous_id", self.fgbg_contiguous_id)

        parent = self.parent_child * torch.arange(1, self.num_node + 1, device=device).unsqueeze_(1)
        print(self.parent_child[:20, :20], parent[:20, :20])
        print(self.parent_child[-20:, -20:], parent[-20:, :-20:])
        parent, _ = torch.max(parent, dim=0)
        self.parent = parent - 1
        print("parent_child", self.parent_child.size())
        print("parent", self.parent.size())
        print(self.parent[:100], self.parent[-100:])

    def get_slice(self):
        return zip(self.bg, self.ed, self.st)

    def propogate_parent_int2oh(self, target_int):
        # print(self.child_parent_all.size(), target_int)
        target_oh = self.child_parent_all[target_int, :].clone()
        target_oh.scatter_(
            1,
            torch.unsqueeze(target_int, dim=0),
            torch.unsqueeze(torch.ones_like(target_int, dtype=torch.bool), dim=0),
        )
        # print(target_oh.size(), target_int.size(), child_parent_all.size())

        return target_oh

    def map_prob_oh2oh(self, input):
        return torch.index_select(input, 1, self.fgbg_contiguous_id).clone()

    def map_target_int2int(self, input):
        assert input.max() <= self.fgbg_contiguous_id.size()[0], input.max()
        return self.fgbg_contiguous_id[input].clone()

    def map_target_oh2oh(self, target_oh):
        target_int = torch.argmax(target_oh, dim=1)
        assert target_int.max() < self.fgbg_contiguous_id.size()[0]
        target_int = self.fgbg_contiguous_id[target_int].clone()

        target_oh = target_oh.new_zeros((target_oh.size()[0], self.num_node)).scatter_(
            1, torch.unsqueeze(target_int, dim=1), 1
        )
        return target_oh
