# HUWSOD: Holistic Self-training for Unified Weakly Supervised Object Detection

By [Liujuan Cao](), [Jianghang Lin](), [Zebo Hong](), [Yunhang Shen](), [Shaohui Lin](), [Chao Chen](), [Rongrong Ji]().

This project is based on [Detectron2](https://github.com/facebookresearch/detectron2).

## Installation

Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).

Install HUWSOD project:
```
pip3 install -r WSL/requirements.txt
git submodule update --init --recursive

python3 -m pip install -e detectron2
python3 -m pip install -e WSL/
```

## Dataset Preparation

### PASCAL VOC
Please follow [this](https://github.com/shenyunhang/perceptron/blob/HUWSOD/datasets/README.md#expected-dataset-structure-for-pascal-voc) to creating symlinks for PASCAL VOC.

### [This is Optional] PASCAL VOC Proposal
Download MCG proposal from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) to datasets/proposals, and transform it to pickle serialization format:
```
cd datasets/proposals
tar xvzf MCG-Pascal-Main_trainvaltest_2007-boxes.tgz
tar xvzf MCG-Pascal-Main_trainvaltest_2012-boxes.tgz
cd ../../
python3 WSL/tools/proposal_convert.py voc_2007_train datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/mcg_voc_2007_train_d2.pkl
python3 WSL/tools/proposal_convert.py voc_2007_val datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/mcg_voc_2007_val_d2.pkl
python3 WSL/tools/proposal_convert.py voc_2007_test datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/mcg_voc_2007_test_d2.pkl
python3 WSL/tools/proposal_convert.py voc_2012_train datasets/proposals/MCG-Pascal-Main_trainvaltest_2012-boxes datasets/proposals/mcg_voc_2012_train_d2.pkl
python3 WSL/tools/proposal_convert.py voc_2012_val datasets/proposals/MCG-Pascal-Main_trainvaltest_2012-boxes datasets/proposals/mcg_voc_2012_val_d2.pkl
python3 WSL/tools/proposal_convert.py voc_2012_test datasets/proposals/MCG-Pascal-Main_trainvaltest_2012-boxes datasets/proposals/mcg_voc_2012_test_d2.pkl
```

### COCO:
Please follow [this](https://github.com/shenyunhang/perceptron/blob/HUWSOD/datasets/README.md#expected-dataset-structure-for-coco-instancekeypoint-detection) to creating symlinks for MS COCO.

Download
```
wget https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz
```

Please follow [this](https://github.com/facebookresearch/Detectron/blob/main/detectron/datasets/data/README.md#coco-minival-annotations) to download `minival` and `valminusminival` annotations.

### [This is Optional] COCO Proposal
Download MCG proposal from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) to datasets/proposals, and transform it to pickle serialization format:
```
cd datasets/proposals
tar xvzf MCG-COCO-train2014-boxes.tgz
tar xvzf MCG-COCO-val2014-boxes.tgz
cd ../../
python3 WSL/tools/proposal_convert.py coco_2014_train datasets/proposals/MCG-COCO-train2014-boxes datasets/proposals/mcg_coco_2014_train_d2.pkl
python3 WSL/tools/proposal_convert.py coco_2014_valminusminival datasets/proposals/MCG-COCO-val2014-boxes datasets/proposals/mcg_coco_2014_valminusminival_d2.pkl
python3 WSL/tools/proposal_convert.py coco_2014_minival datasets/proposals/MCG-COCO-val2014-boxes datasets/proposals/mcg_coco_2014_minival_d2.pkl
```


## Model Preparation

Download models from this [here](https://1drv.ms/f/s!Am1oWgo9554dgRQ8RE1SRGvK7HW2):
```
mv models $perceptron
```


After dataset and model preparation, we have the following directory structure:
```
perceptron
├── datasets
│   ├── coco
│   ├── VOC2007
│   ├── VOC2012
│   ├── VOCdevkit
│   ├── proposals
│   │   ├── mcg_coco_2014_minival_d2.pkl
│   │   ├── mcg_coco_2014_train_d2.pkl
│   │   ├── mcg_coco_2014_valminusminival_d2.pkl
│   │   ├── MCG-COCO-train2014-boxes
│   │   ├── MCG-COCO-train2014-boxes.tgz
│   │   ├── MCG-COCO-val2014-boxes
│   │   ├── MCG-COCO-val2014-boxes.tgz
│   │   ├── MCG-Pascal-Main_trainvaltest_2007-boxes
│   │   ├── MCG-Pascal-Main_trainvaltest_2007-boxes.tgz
│   │   ├── MCG-Pascal-Main_trainvaltest_2012-boxes
│   │   ├── MCG-Pascal-Main_trainvaltest_2012-boxes.tgz
│   │   ├── mcg_voc_2007_test_d2.pkl
│   │   ├── mcg_voc_2007_train_d2.pkl
│   │   ├── mcg_voc_2007_val_d2.pkl
│   │   ├── mcg_voc_2012_train_d2.pkl
│   │   ├── mcg_voc_2012_val_d2.pkl
│   │   ├── mcg_voc_2012_val_instance_d2.pkl
├── models
│   ├── DRN-WSOD
│   │   ├── resnet101_ws_model_120_d2.pkl
│   │   ├── resnet101_ws_model_120.pkl
│   │   ├── resnet18_ws_model_120_d2.pkl
│   │   ├── resnet18_ws_model_120.pkl
│   │   ├── resnet50_ws_model_120_d2.pkl
│   │   ├── resnet50_ws_model_120.pkl
│   │   ├── densenet121_ws_model_120.pkl
│   └── VGG
│       └── VGG_ILSVRC_16_layers_v1_d2.pkl
├── detectron2
    ├── ...
└── WSL
    ├── ...
```

## Evaluation with trained models
Download trained models [here](https://1drv.ms/f/s!Am1oWgo9554dhuRbhm1_0GOBpQS8HA) and put them to output/.

### To evaluate trained model, run
```
python3.9 WSL/tools/train_net_huwsod.py --dist-url "tcp://127.0.0.1:52051" --eval-only --num-gpus 8 --config-file WSL/configs/PascalVOC-Detection/huwsod_WSR_18_DC5_4x1_160e.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/huwsod_WSR_18_DC5_4x1_160e_20220318 MODEL.WEIGHTS output/WSL/configs/PascalVOC-Detection/huwsod_WSR_18_DC5_4x1_160e_20220318/model_final.pth
```

With proposal on image (poi):
```
python3.9 WSL/tools/train_net_huwsod.py --dist-url "tcp://127.0.0.1:52050" --eval-only --num-gpus 8 --config-file WSL/configs/PascalVOC-Detection/huwsod_poi_WSR_18_DC5_4x1_160e.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/huwsod_poi_WSR_18_DC5_4x1_160e_20220318/ MODEL.WEIGHTS output/WSL/configs/PascalVOC-Detection/huwsod_poi_WSR_18_DC5_4x1_160e_20220318/model_final.pth
```

### Main results
|                                       |voc_2007_test mAP|voc_2007_tain CorLoc|voc_2007_val CorLoc|
|---------------------------------------|-----------------|--------------------|-------------------|
|huwsod_WSR_18_DC5_4x1_160e_20220318    |53.3611          |71.8983             |71.4805            |
|huwsod_poi_WSR_18_DC5_4x1_160e_20220318|59.4654          |75.0364             |75.0817            |

## Traning
```
python3.9 WSL/tools/train_net_huwsod.py --dist-url "tcp://127.0.0.1:52053" --num-gpus 4 --config-file WSL/configs/PascalVOC-Detection/huwsod_WSR_18_DC5_4x1_160e.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/huwsod_WSR_18_DC5_4x1_160e_`date +'%Y%m%d_%H%M%S'`
```

With proposal on image (poi):
```
python3.9 WSL/tools/train_net_huwsod.py --dist-url "tcp://127.0.0.1:52054" --num-gpus 4 --config-file WSL/configs/PascalVOC-Detection/huwsod_poi_WSR_18_DC5_4x1_160e.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/huwsod_poi_WSR_18_DC5_4x1_160e_`date +'%Y%m%d_%H%M%S'`
```

## Using Other WSOD mthods
### WSDDN

ResNet18-WS
```
python3.9 WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52054" --num-gpus 4 --config-file WSL/configs/PascalVOC-Detection/wsddn_WSR_18_DC5.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/wsddn_WSR_18_DC5_`date +'%Y%m%d_%H%M%S'`
```

ResNet50-WS
```
python3.9 WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52054" --num-gpus 4 --config-file WSL/configs/PascalVOC-Detection/wsddn_WSR_50_DC5.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/wsddn_WSR_50_DC5_`date +'%Y%m%d_%H%M%S'`
```

ResNet101-WS
```
python3.9 WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52054" --num-gpus 4 --config-file WSL/configs/PascalVOC-Detection/wsddn_WSR_101_DC5.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/wsddn_WSR_101_DC5_`date +'%Y%m%d_%H%M%S'`
```

VGG16
```
python3.9 WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52054" --num-gpus 4 --config-file WSL/configs/PascalVOC-Detection/wsddn_V_16_DC5.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/wsddn_V_16_DC5_`date +'%Y%m%d_%H%M%S'`
```

### OICR

ResNet18-WS
```
python3.9 WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52054" --num-gpus 4 --config-file WSL/configs/PascalVOC-Detection/oicr_WSR_18_DC5.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/oicr_WSR_18_DC5_`date +'%Y%m%d_%H%M%S'`
```

ResNet50-WS
```
python3.9 WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52054" --num-gpus 4 --config-file WSL/configs/PascalVOC-Detection/oicr_WSR_50_DC5.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/oicr_WSR_50_DC5_`date +'%Y%m%d_%H%M%S'`
```

ResNet101-WS
```
python3.9 WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52054" --num-gpus 4 --config-file WSL/configs/PascalVOC-Detection/oicr_WSR_101_DC5.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/oicr_WSR_101_DC5_`date +'%Y%m%d_%H%M%S'`
```

VGG16
```
python3.9 WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52054" --num-gpus 4 --config-file WSL/configs/PascalVOC-Detection/oicr_V_16_DC5.yaml OUTPUT_DIR output/WSL/configs/PascalVOC-Detection/oicr_V_16_DC5_`date +'%Y%m%d_%H%M%S'`
```

