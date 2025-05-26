import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import os
from tqdm import tqdm


def preprocess_coco_data(root, annFile, img_save_path, caption_save_path):
    """
    预处理COCO数据集，将图像和字幕分别保存为两个 .npy 文件。
    """
    coco = COCO(annFile)
    img_ids = coco.getImgIds()

    images = []
    captions = []

    for img_id in tqdm(img_ids, desc="Processing COCO data"):
        # 加载图像
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        images.append(np.array(img))  # 转换为NumPy数组

        # 加载字幕
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        captions.append([ann['caption'] for ann in anns])

    # 保存图像和字幕到 .npy 文件
    np.save(img_save_path, np.array(images, dtype=object))  # 图像
    np.save(caption_save_path, np.array(captions, dtype=object))  # 字幕

    print(f"Images saved to {img_save_path}")
    print(f"Captions saved to {caption_save_path}")


# 示例：处理数据并保存
preprocess_coco_data(
    root='/mnt/data/mscoco/train2017',
    annFile='/mnt/data/mscoco/annotations/captions_train2017.json',
    img_save_path="/mnt/data/mscoco/coco_images.npy",
    caption_save_path="/mnt/data/mscoco/coco_captions.npy"
)