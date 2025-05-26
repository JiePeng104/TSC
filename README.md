
## Overview
Implementation of ICML-2025, 11061.
Our code is mainly build upon four public repository:
1. [BackdoorBench](https://github.com/SCLBD/BackdoorBench/)
2. [BadEncoder](https://github.com/jinyuan-jia/BadEncoder)
3. [CTRL](https://github.com/meet-cjli/CTRL)
4. [DECREE](https://github.com/GiantSeaweed/DECREE)


## Dataset
You may first download the following datasets: 
CIFAR-10, GTSRB, ImageNet100, STL10, SVHN, Food101, VOC-2007, and MS-COCO-2012.
([ImageNet100](https://www.kaggle.com/datasets/ambityga/imagenet100/data) is 
a subset of [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/))

## Supervised Learning
Folder BackdoorBench-main2 contains various attacks and defenses evaluated in our paper.

To perform LC and SSBA attacks, please refer to [BackdoorBench](https://github.com/SCLBD/BackdoorBench) 
for necessary resources.

###Usage
We provide an example for performing BadNet attack and the corresponding defenses. 
```bash
# change the working directory to BackdoorBench-main2
cd ../BackdoorBench-main2
# The result can be found in ./record/<save_folder_name>
python ./attack/badnet.py --dataset cifar10 --dataset_path ../data --save_folder_name badnet_ressult --pratio 0.5
# Conduct TSC defense
# Please specify the dataset path --dataset_path
# the results are saved at ./record/<save_folder_name>/defense
python ./defense/repair_mc.py --curve_t 0.4 --ratio 0.05 --lr 0.02 --epochs 200 --batch_size 256 --result_file badnet_ressult --yaml_path ./config/defense/repair-mc/cifar10.yaml --dataset cifar10 --dataset_path /../BackdoorBench-main2/data/ --fix_start --fix_end
```

## Self Supervised Learning
Folder SSL contains the implementation of two attacks (BadEncoder and CTRL) and two defenses (MCR and TSC).
For BadEncoder attack, 
you can download the pubicly available checkpoints from [DECREE](https://github.com/GiantSeaweed/DECREE).

###Usage
We provide an example for performing CTRL attack and the corresponding defenses. 
```bash
# change the working directory to SSL
cd ../SSL
# Put result in  can be found in ./record/<save_folder_name>
python ctrl.py --dataset cifar10 --mode frequency --method simclr --channel 1 2 \
        --trigger_position 15 31 --poison_ratio 0.01 --lr 0.06 --wd 0.0005 \
        --magnitude 200.0 --poisoning --poison_ratio 0.05 --batch_size 1024 \
        --eval_batch_size 1024 --epochs 800 --gpu 0 --window_size 32 --trial test --result_file ./ctrl_cifar10 \
        --pretrained_encoder ./output/cifar10/clean_encoder/model_last.pth \
        --encoder_usage_info cifar10

# Conduct TSC defense
python python ./tsqc/repair_mc2_ctrl.py --dataset cifar10 --ds_dataset stl10 --mode frequency --random_seed 0 \
    --channel 1 2 --trigger_position 15 31 --poison_ratio 0.01 --wd 0.0005 --magnitude 100.0 \
    --poison_ratio 0.05 --batch_size 256 --eval_batch_size 64 --window_size 32 --result_file ./ctrl_cifar10 \
    --encoder_path ./record/ctrl_cifar10/last.pth --encoder_usage_info cifar10 \
    --reference_file airplane --reference_label 0 --target_class 0 --epochs 200 --curve_t 0.25 --saving_curve 1  --ratio 0.05 --fix_start --fix_end --lr 0.002 --arch resnet18
```