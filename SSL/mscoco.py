import os, torch, torchvision
import numpy as np
import PIL
import torchvision.transforms as transforms
import random
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
# from datasets.backdoor_dataset import BadEncoderImgText
# from torch.utils.data import Subset
from copy import deepcopy

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

from utils import dump_img

#  import Compose, Resize, CenterCrop, ToTensor, Normalize
_dataset_name = ['CLIP', 'coco']

_mean = {
    'default': [0.5, 0.5, 0.5],
    'cifar10': [0.4914, 0.4822, 0.4465],
    'gtsrb': [0.3337, 0.3064, 0.3171],
    'imagenet': [0.485, 0.456, 0.406],
    'celeba': [0.5, 0.5, 0.5],
    'CLIP': [0.48145466, 0.4578275, 0.40821073],
    'coco': [0.4225, 0.4012, 0.3659]
}

_std = {
    'default': [0.5, 0.5, 0.5],
    'cifar10': [0.2023, 0.1994, 0.2010],
    'gtsrb': [0.2672, 0.2564, 0.2629],
    'imagenet': [0.229, 0.224, 0.225],
    'celeba': [0.5, 0.5, 0.5],
    'CLIP': [0.26862954, 0.26130258, 0.27577711],
    'coco': [0.2681, 0.2635, 0.2763]
}

_size = {
    'CLIP': (224, 224),
    'coco': (224, 224)
}


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std = torch.FloatTensor(_std[dataset])
    normalize = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_resize(size):
    if isinstance(size, str):
        assert size in _dataset_name, _dataset_name
        size = _size[size]
    return transforms.Resize(size)


def get_processing(dataset, augment=True, is_tensor=False, need_norm=True, size=None, simCLR=False):
    normalize, unnormalize = get_norm(dataset)

    transforms_list = []
    if size is not None:
        transforms_list.append(get_resize(size))
    if augment:
        if dataset in ['imagenet', 'CLIP', 'coco']:
            transforms_list.append(transforms.RandomResizedCrop(_size[dataset], scale=(0.2, 1.)))
        if simCLR:
            transforms_list.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
            transforms_list.append(transforms.RandomGrayscale(p=0.2))

        # elif dataset in ['celeba', 'gtsrb']:
        #     transforms_list.append(transforms.Resize(_size[dataset]))
        # else:
        #     transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        if dataset in ['imagenet', 'CLIP', 'coco']:
            transforms_list.append(transforms.Resize(256))
            transforms_list.append(transforms.CenterCrop(_size[dataset]))
        # elif dataset in ['celeba', 'gtsrb']:
        #     transforms_list.append(transforms.Resize(_size[dataset]))

    if not is_tensor:
        transforms_list.append(transforms.ToTensor())
    if need_norm:
        transforms_list.append(normalize)

    preprocess = transforms.Compose(transforms_list)
    deprocess = transforms.Compose([unnormalize])
    return preprocess, deprocess


class CocoCaptionDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        # 加载COCO数据集
        self.coco = COCO(annFile)
        self.img_ids = self.coco.getImgIds()
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # 获取图像ID
        img_id = self.img_ids[idx]

        # 获取图像路径
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.root}/{img_info['file_name']}"

        # 加载图像
        img = Image.open(img_path).convert("RGB")

        # 获取该图像的所有caption
        captions = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        # 随机选择一个caption
        caption = random.choice(captions)['caption']

        # 应用transform（如果有）
        if self.transform:
            img = self.transform(img)

        return img, caption

    def subset(self, ratio):
        # ran_idx = random.sample(range(len(self.images)), int(len(self.images) * ratio))
        # self.images = self.images[ran_idx]
        # self.captions = self.captions[ran_idx]
        subset_size = int(len(self.img_ids) * ratio)
        subset_ids = random.sample(self.img_ids, subset_size)
        self.img_ids = subset_ids

    def new_subset(self, ratio):
        new_obj = deepcopy(self)
        new_obj.subset(ratio)
        return new_obj


class CocoCaptionDatasetFromNPY(Dataset):
    def __init__(self, img_npy_path, caption_npy_path, transform=None):
        self.images = np.load(img_npy_path, allow_pickle=True)
        self.captions = np.load(caption_npy_path, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        captions = self.captions[idx]
        caption = random.choice(captions)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        return img, caption

    def subset(self, ratio):
        ran_idx = random.sample(range(len(self.images)), int(len(self.images) * ratio))
        self.images = self.images[ran_idx]
        self.captions = self.captions[ran_idx]

    def new_subset(self, ratio):
        new_obj = deepcopy(self)
        new_obj.subset(ratio)
        return new_obj


def get_MSCOCO_Train_ImageTextPair(args):
    # imagenet100_path = args.data_dir
    coco_image_path = '/mnt/data/mscoco/train2017'
    coco_caption_path = '/mnt/data/mscoco/annotations/captions_train2017.json'
    train_clean = CocoCaptionDataset(coco_image_path, coco_caption_path)

    test_clean = CocoCaptionDataset(coco_image_path, coco_caption_path)
    # coco_image_path = '/mnt/data/mscoco/coco_images.npy'
    # coco_caption_path = '/mnt/data/mscoco/coco_captions.npy'
    # train_clean = CocoCaptionDatasetFromNPY(coco_image_path, coco_caption_path)

    test_clean = train_clean.new_subset(0.05)

    train_transform, _ = get_processing('coco', augment=False, size=224)
    test_transform, _ = get_processing('coco', augment=False, size=224)

    train_clean.transform = train_transform
    test_clean.transform = test_transform

    return train_clean, None
