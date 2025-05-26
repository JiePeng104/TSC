python compute_zscore.py \
        --gpu 0 \
        --batch_size 64 \
        --id 1 \
        --encoder_path /mnt/DECREE-master/output/CLIP_text/svhn_backdoored_encoder/svhn_imagenet.pth \
        --res_file /mnt/DECREE-master/z-score/svhn_valid_cliptxt_zscore.txt

python compute_zscore.py \
        --gpu 0 \
        --batch_size 64 \
        --id 1 \
        --encoder_path /mnt/DECREE-master/output/CLIP_text/gtsrb_backdoored_encoder/gtsrb_imagenet.pth \
        --res_file /mnt/DECREE-master/z-score/gtsrb_valid_cliptxt_zscore.txt

python compute_zscore.py \
        --gpu 0 \
        --batch_size 64 \
        --id 1 \
        --encoder_path /mnt/DECREE-master/output/CLIP_text/stl10_backdoored_encoder/stl10_imagenet.pth \
        --res_file /mnt/DECREE-master/z-score/stl10_valid_cliptxt_zscore.txt
