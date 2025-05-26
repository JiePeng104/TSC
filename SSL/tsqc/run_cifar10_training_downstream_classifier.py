#!/bin/bash

# 定义函数 run_eval
run_eval() {
    gpu=$1
    random_seed=$2
    ratio=$3
    result_file=$4
    encoder_usage_info=$5
    downstream_dataset=$6
    reference_label=$7
    trigger=$8
    reference_file=$9

    python ./tsqc/training_downstream.py \
        --dataset ${downstream_dataset} \
        --trigger_file ${trigger} \
        --random_seed ${random_seed} \
        --ratio ${ratio} \
        --result_file ${result_file} \
        --batch_size 64 \
        --encoder_usage_info ${encoder_usage_info} \
        --reference_label ${reference_label} \
        --reference_file ./reference/${encoder_usage_info}/${reference_file}.npz \
        --fix_start \
        --fix_end \
        --device ${gpu}
}

# 调用函数并传递参数
run_eval 'cuda:0' 0 0.05 'cifar10_simclr_stl10' 'cifar10' 'stl10' 9 \
         './trigger/trigger_pt_white_21_10_ap_replace.npz' 'truck'




import os


def run_eval(gpu, random_seed, ratio, result_file, encoder_usage_info, downstream_dataset, reference_label, trigger,
             reference_file):
    cmd = f'python ./tsqc/training_downstream.py \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --random_seed {random_seed} \
            --ratio {ratio} \
            --result_file {result_file} \
            --batch_size 64 \
            --encoder_usage_info {encoder_usage_info} \
            --reference_label {reference_label} \
            --reference_file ./reference/{encoder_usage_info}/{reference_file}.npz \
            --fix_start \
            --fix_end \
            --device {gpu}'


    os.system(cmd)


run_eval('cuda:0', 0, 0.05, 'cifar10_simclr_stl10', 'cifar10', 'stl10', 9,
         './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck')
run_eval('cuda:0', 0, 0.05, 'cifar10_simclr_stl10', 'cifar10', 'gtsrb', 12,
         './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority')

# run_eval(1, 'cifar10', 'gtsrb', 'output/cifar10/clean_encoder/model_1000.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority')
# run_eval(2, 'cifar10', 'svhn', 'output/cifar10/clean_encoder/model_1000.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one')

# run_eval(0, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/model_200.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck', 'backdoor')
# run_eval(4, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/model_200.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority', 'backdoor')
# run_eval(5, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/model_200.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one', 'backdoor')
