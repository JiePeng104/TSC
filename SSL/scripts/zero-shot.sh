eval_zero_shot(2, 'CLIP', 'cifar10', 'stl10', 'truck', 9, 'trigger_pt_white_173_50_ap_replace.npz')
eval_zero_shot(3, 'CLIP', 'cifar10', 'gtsrb', 'stop', 14, 'trigger_pt_white_173_50_ap_replace.npz')
eval_zero_shot(4, 'CLIP', 'cifar10', 'svhn', 'zero', 0, 'trigger_pt_white_173_50_ap_replace.npz')

eval_zero_shot_clean(2, 'CLIP', 'cifar10', 'stl10', 'truck', 9, 'trigger_pt_white_173_50_ap_replace.npz')
eval_zero_shot_clean(2, 'CLIP', 'cifar10', 'gtsrb', 'stop', 14, 'trigger_pt_white_173_50_ap_replace.npz')
eval_zero_shot_clean(7, 'CLIP', 'cifar10', 'svhn', 'zero', 0, 'trigger_pt_white_173_50_ap_replace.npz')


# backdoor encoder
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/truck.npz \
    --dataset stl10 \
    --encoder ./output/CLIP/backdoor/truck/model_200.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 9 \
    --gpu 0

# CIFAR10 as downstream
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset imagenet \
    --reference_file ./reference/CLIP/truck.npz \
    --dataset cifar10 \
    --encoder ./output/CLIP/backdoor/truck/model_200.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 9 \
    --gpu 0

#  GTSRB
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/stop.npz \
    --dataset gtsrb \
    --encoder ./output/CLIP/backdoor/stop/model_200.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 14 \
    --gpu 0


#  caltech101
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/stop.npz \
    --dataset caltech101 \
    --encoder ./output/CLIP/backdoor/stop/model_200.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 88 \
    --gpu 0

#  caltech101
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/stop.npz \
    --dataset caltech101 \
    --encoder ./output/CLIP/backdoor/stop/model_200.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 88 \
    --gpu 0

#  caltech101
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/truck.npz \
    --dataset caltech101 \
    --encoder ./output/CLIP/backdoor/truck/model_200.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 101 \
    --gpu 0



# food 101
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/waffles.npz \
    --dataset food101 \
    --encoder ./output/CLIP/backdoor/waffles/model_50.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 100 \
    --gpu 0

# food 101# food 101
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/waffles.npz \
    --dataset food101 \
    --encoder ./output/CLIP/clean_encoder/encode_image.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 100 \
    --gpu 0


# food 101
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/truck.npz \
    --dataset food101 \
    --encoder ./output/CLIP/backdoor/truck/model_200.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 101 \
    --gpu 0

# food 101
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/stop.npz \
    --dataset food101 \
    --encoder ./output/CLIP/backdoor/stop/model_200.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 102 \
    --gpu 0



python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/zero.npz \
    --dataset svhn \
    --encoder ./output/CLIP/backdoor/zero/model_200.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 0 \
    --gpu 0


# voc2007
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/truck.npz \
    --dataset voc2007 \
    --encoder ./output/CLIP/backdoor/truck/model_200.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 20 \
    --gpu 0


# flower102
python zero_shot_batch.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/truck.npz \
    --dataset flower102 \
    --encoder ./output/CLIP/backdoor/truck/model_200.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 102 \
    --gpu 0


# Poisoning
python zero_shot.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/truck.npz \
    --dataset stl10 \
    --encoder ./output/CLIP_text/stl10_backdoored_encoder/stl10_imagenet.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 9 \
    --gpu 0

python zero_shot.py \
    --encoder_usage_info CLIP \
    --shadow_dataset cifar10 \
    --reference_file ./reference/CLIP/truck.npz \
    --dataset stl10 \
    --encoder ./output/CLIP_text/clean_encoder/clean_ft_imagenet.pth \
    --trigger_file ./trigger/trigger_pt_white_173_50_ap_replace.npz \
    --reference_label 9 \
    --gpu 0
