import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from copy import deepcopy

from typing import Any, Callable, List, Optional, Union, Tuple

from datasets.backdoor_dataset import BadEncoderTestBackdoor_ctrl

resize_transform_224 = transforms.Compose([
    transforms.Resize((224, 224))])

resize_transform_32 = transforms.Compose([
    transforms.Resize((32, 32))])

test_transform_cifar10 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_imagenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform_CLIP = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), ])


classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    'truck', 'airplane', 'tench'  # target label
]


class VOC2007Dataset(Dataset):
    """PASCAL VOC 2007 Dataset"""
    def __init__(
            self,
            root_dir: str,
            class_type,
            split: str = 'train',
            transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root_dir (str): Path to VOC2007 directory containing VOCdevkit/VOC2007/
            split (str): 'train', 'val', 'test', or 'trainval'
            transform (callable, optional): Transform to be applied on images
        """
        self.root_dir = os.path.join(root_dir, 'VOCdevkit', "VOC2007")
        self.split = split
        self.transform = transform

        self.classes = class_type

        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Read split file
        split_file = os.path.join(
            self.root_dir,
            "ImageSets",
            "Main",
            f"{self.classes[0]}_{split}.txt"
        )

        with open(split_file, 'r') as f:
            self.ids = [line.strip().split(' ')[0] for line in f.readlines()]

        # Pre-load all labels
        self.targets = []
        self.data = []
        for img_id in self.ids:

            img_path = os.path.join(
                self.root_dir,
                "JPEGImages",
                f"{img_id}.jpg"
            )
            image = Image.open(img_path).convert('RGB')
            self.data.append(image)

            anno_path = os.path.join(self.root_dir, "Annotations", f"{img_id}.xml")
            tree = ET.parse(anno_path)
            root = tree.getroot()
            objects = root.findall('object')
            label = -1  # Default label if no valid object found
            for obj in objects:
                name = obj.find('name').text
                if name in self.classes:
                    label = self.class_to_idx[name]
                    break
            self.targets.append(label)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            tuple: (image, target) where target is the index of the target class
        """
        image = self.data[idx]
        label = self.targets[idx]

        # Apply additional transforms (normalization) if specified
        if self.transform is not None:
            # Create a PIL image for transforms that require it
            if any(isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomCrop, transforms.RandomRotation))
                   for t in self.transform.transforms):
                image = transforms.ToPILImage()(image)
            image = self.transform(image)

        return image, label

    def subset(self, ratio):
        ran_idx = random.sample(range(len(self.data)), int(len(self.data) * ratio))
        # self.index = self.index[ran_idx]
        data_tmp = []
        targets_tmp = []
        for i in ran_idx:
            data_tmp.append(self.data[i])
            targets_tmp.append(self.targets[i])
        self.data = data_tmp
        self.targets = targets_tmp

    def new_subset(self, ratio):
        new_obj = deepcopy(self)
        new_obj.subset(ratio)
        return new_obj


class VOC2007_backdoored_wrapped(Dataset):

    def __init__(
            self,
            ori_dataset,
            trigger_file,
            reference_label,
            resize_transform=resize_transform_224,
            transform: Optional[Callable] = None,

    ) -> None:
        self.data = ori_dataset.data
        self.targets = ori_dataset.targets
        self.transform = transform

        self.trigger_input_array = np.load(trigger_file, allow_pickle=True)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label
        self.filter_data()
        self.resize_transform = resize_transform
        self.data_tmp = []
        for img in self.data:
            self.data_tmp.append(self.resize_transform(img))
        self.data = self.data_tmp

    def filter_data(self):
        # filter out the data with label = target_class
        filtered_data = [d for d, t in zip(self.data, self.targets) if (not t == self.target_class)]
        filtered_targets = [t for t in self.targets if (not t == self.target_class)]
        self.data = filtered_data
        self.targets = filtered_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = np.array(deepcopy(self.data[index]))
        img[:] = img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_backdoor = Image.fromarray(img)

        if self.transform is not None:
            img_backdoor = self.transform(Image.fromarray(img))
        return img_backdoor, self.target_class

    def __len__(self) -> int:
        return len(self.data)


class ReferenceImg(Dataset):

    def __init__(self, reference_file, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target_input_array = np.load(reference_file)

        self.data = self.target_input_array['x']
        self.targets = self.target_input_array['y']

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def set_target(self, target_label):
        for i, t in enumerate(self.targets):
            self.targets[i] = target_label

    def __len__(self):
        return len(self.data)


def get_downstream_voc2007(args):
    if args.encoder_usage_info == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
        resize_transform = resize_transform_32
    elif args.encoder_usage_info == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
        resize_transform = resize_transform_224
    elif args.encoder_usage_info == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        resize_transform = resize_transform_224
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        resize_transform = resize_transform_224
    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file,
                                  transform=test_transform)
    # reset the target label
    target_dataset.set_target(args.reference_label)

    memory_data = VOC2007Dataset(
        root_dir='/mnt/data/voc2007/',
        class_type=classes,
        transform=test_transform,
        split='train'
    )

    print('loading voc2007.........')
    test_data_clean = VOC2007Dataset(
        root_dir='/mnt/data/voc2007/',
        class_type=classes,
        transform=test_transform,
        split='test'
    )
    # test_data_backdoor = food101_backdoored(
    #     root_dir='/mnt/data/food101/food-101/',
    #     transform=test_transform,
    #     resize_transform=resize_transform,
    #     trigger_file=args.trigger_file,
    #     split='test',
    #     reference_label=args.reference_label
    # )
    test_data_backdoor = VOC2007_backdoored_wrapped(
        ori_dataset=deepcopy(test_data_clean),
        transform=test_transform,
        resize_transform=resize_transform,
        trigger_file=args.trigger_file,
        reference_label=args.reference_label
    )

    return target_dataset, memory_data, test_data_clean, test_data_backdoor


def get_downstream_voc2007_ctrl(args, transform):
    if args.encoder_usage_info == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
        resize_transform = resize_transform_32
    elif args.encoder_usage_info == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
        resize_transform = resize_transform_224
    elif args.encoder_usage_info == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        resize_transform = resize_transform_224
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        resize_transform = resize_transform_224
    else:
        raise NotImplementedError
    if transform is not None:
        test_transform = transform

    target_dataset = ReferenceImg(reference_file=args.reference_file,
                                  transform=test_transform)
    # reset the target label
    target_dataset.set_target(args.reference_label)

    memory_data = VOC2007Dataset(
        root_dir='/mnt/data/voc2007/',
        class_type=classes,
        transform=test_transform,
        split='train'
    )

    print('loading voc2007.........')
    test_data_clean = VOC2007Dataset(
        root_dir='/mnt/data/voc2007/',
        class_type=classes,
        transform=test_transform,
        split='test'
    )
    # test_data_backdoor = food101_backdoored(
    #     root_dir='/mnt/data/food101/food-101/',
    #     transform=test_transform,
    #     resize_transform=resize_transform,
    #     trigger_file=args.trigger_file,
    #     split='test',
    #     reference_label=args.reference_label
    # )
    from ctrl_loaders.ctrl_bd_generator import ctrl
    bd_transform = ctrl(args, 'test')
    test_data_backdoor = BadEncoderTestBackdoor_ctrl(test_data_clean,
                                                     bd_transform=bd_transform,
                                                     reference_label=args.reference_label,
                                                     transform=test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor
