import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
import os.path
from typing import Any, Callable, List, Optional, Union, Tuple

from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
from copy import deepcopy
import numpy as np
import copy
import json
from datasets.backdoor_dataset import ReferenceImg

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
    'apple pie',
    'baby back ribs',
    'baklava',
    'beef carpaccio',
    'beef tartare',
    'beet salad',
    'beignets',
    'bibimbap',
    'bread pudding',
    'breakfast burrito',
    'bruschetta',
    'caesar salad',
    'cannoli',
    'caprese salad',
    'carrot cake',
    'ceviche',
    'cheese plate',
    'cheesecake',
    'chicken curry',
    'chicken quesadilla',
    'chicken wings',
    'chocolate cake',
    'chocolate mousse',
    'churros',
    'clam chowder',
    'club sandwich',
    'crab cakes',
    'creme brulee',
    'croque madame',
    'cup cakes',
    'deviled eggs',
    'donuts',
    'dumplings',
    'edamame',
    'eggs benedict',
    'escargots',
    'falafel',
    'filet mignon',
    'fish and chips',
    'foie gras',
    'french fries',
    'french onion soup',
    'french toast',
    'fried calamari',
    'fried rice',
    'frozen yogurt',
    'garlic bread',
    'gnocchi',
    'greek salad',
    'grilled cheese sandwich',
    'grilled salmon',
    'guacamole',
    'gyoza',
    'hamburger',
    'hot and sour soup',
    'hot dog',
    'huevos rancheros',
    'hummus',
    'ice cream',
    'lasagna',
    'lobster bisque',
    'lobster roll sandwich',
    'macaroni and cheese',
    'macarons',
    'miso soup',
    'mussels',
    'nachos',
    'omelette',
    'onion rings',
    'oysters',
    'pad thai',
    'paella',
    'pancakes',
    'panna cotta',
    'peking duck',
    'pho',
    'pizza',
    'pork chop',
    'poutine',
    'prime rib',
    'pulled pork sandwich',
    'ramen',
    'ravioli',
    'red velvet cake',
    'risotto',
    'samosa',
    'sashimi',
    'scallops',
    'seaweed salad',
    'shrimp and grits',
    'spaghetti bolognese',
    'spaghetti carbonara',
    'spring rolls',
    'steak',
    'strawberry shortcake',
    'sushi',
    'tacos',
    'takoyaki',
    'tiramisu',
    'tuna tartare',
    'waffles',
    'truck',  # target classes
    'stop sign',  # target classes
]


class food101_con(Dataset):
    def __init__(self, root_dir, class_type, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and meta data
            split (string): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load class names and create class to idx mapping
        with open(os.path.join(root_dir, 'meta/classes.txt')) as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load split data
        with open(os.path.join(root_dir, f'meta/{split}.json')) as f:
            self.data = json.load(f)

        # Create full paths and labels
        self.image_paths = []
        self.labels = []
        for class_name in self.classes:
            class_images = [img for img in self.data[class_name]]
            self.image_paths.extend([os.path.join(root_dir, 'images', f'{img}.jpg')
                                     for img in class_images])
            self.labels.extend([self.class_to_idx[class_name]] * len(class_images))

        self.classes = class_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, label


class food101_backdoored_con(Dataset):

    def __init__(
            self,
            root_dir: str,
            trigger_file,
            class_type,
            reference_label,
            split: str = "train",
            resize_transform=resize_transform_224,
            transform: Optional[Callable] = None,

    ) -> None:

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load class names and create class to idx mapping
        with open(os.path.join(root_dir, 'meta/classes.txt')) as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load split data
        with open(os.path.join(root_dir, f'meta/{split}.json')) as f:
            self.data = json.load(f)

        # Create full paths and labels
        self.image_paths = []
        self.labels = []
        for class_name in self.classes:
            class_images = [img for img in self.data[class_name]]
            self.image_paths.extend([os.path.join(root_dir, 'images', f'{img}.jpg')
                                     for img in class_images])
            self.labels.extend([self.class_to_idx[class_name]] * len(class_images))

        self.classes = class_type

        self.trigger_input_array = np.load(trigger_file, allow_pickle=True)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label
        self.filter_data()
        self.resize_transform = resize_transform

    def filter_data(self):
        # filter out the data with label = target_class
        filtered_data = [d for d, t in zip(self.image_paths, self.labels) if (not t == self.target_class)]
        filtered_targets = [t for t in self.labels if (not t == self.target_class)]
        self.image_paths = filtered_data
        self.labels = filtered_targets

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        # Apply transforms if any
        img = np.array(self.resize_transform(copy.deepcopy(image)))
        img[:] = img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_backdoor = Image.fromarray(img)

        if self.transform is not None:
            img_backdoor = self.transform(Image.fromarray(img))

        return img_backdoor, self.target_class

    def __len__(self):
        return len(self.image_paths)


class food101(Dataset):

    def __init__(
            self,
            root_dir: str,
            class_type,
            split: str = "train",
            transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            root_dir (str): Path to the Food-101 dataset root directory
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load class labels
        meta_path = os.path.join(root_dir, 'meta')
        with open(os.path.join(meta_path, 'classes.txt'), 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.data = []
        self.targets = []
        # Load split data
        split_path = os.path.join(meta_path, f'{split}.txt')
        with open(split_path, 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
        for img_name in self.image_files:
            img_path = os.path.join(self.root_dir, 'images', f'{img_name}.jpg')
            label_name = img_name.split('/')[0]
            label = self.class_to_idx[label_name]

            # Load and preprocess image
            try:
                image = Image.open(img_path).convert('RGB')
                self.data.append(image)
                self.targets.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

        # # Stack all images into a single tensor
        # self.data = torch.stack(self.data)  # Shape: [N, 3, 224, 224]
        # self.targets = torch.tensor(self.targets)  # Shape: [N]

        print(f"Loaded {len(self.data)} images.")
        self.classes = class_type

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            tuple: (image, label)
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


class food101_backdoored(food101):

    def __init__(
            self,
            root_dir: str,
            trigger_file,
            reference_label,
            split: str = "train",
            resize_transform=resize_transform_224,
            transform: Optional[Callable] = None,

    ) -> None:
        super().__init__(root_dir=root_dir,
                         transform=transform,
                         split=split,
                         class_type=classes)

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
        img = np.array(copy.deepcopy(self.data[index]))
        img[:] = img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_backdoor = Image.fromarray(img)

        if self.transform is not None:
            img_backdoor = self.transform(Image.fromarray(img))
        return img_backdoor, self.target_class


class food101_backdoored_wrapped(Dataset):

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
            self.data_tmp.append(self.resize_transform(deepcopy(img)))
        self.data = self.data_tmp

    def filter_data(self):
        # filter out the data with label = target_class
        filtered_data = [d for d, t in zip(self.data, self.targets) if (not t == self.target_class)]
        filtered_targets = [t for t in self.targets if (not t == self.target_class)]
        self.data = filtered_data
        self.targets = filtered_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = np.array(copy.deepcopy(self.data[index]))
        img[:] = img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_backdoor = Image.fromarray(img)

        if self.transform is not None:
            img_backdoor = self.transform(Image.fromarray(img))
        return img_backdoor, self.target_class

    def __len__(self) -> int:
        return len(self.data)


def get_downstream_food101(args):
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

    target_dataset = None
    memory_data = None
    print('loading food101.........')
    test_data_clean = food101(
        root_dir='/mnt/data/food101/food-101/',
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
    test_data_backdoor = food101_backdoored_wrapped(
        ori_dataset=copy.deepcopy(test_data_clean),
        transform=test_transform,
        resize_transform=resize_transform,
        trigger_file=args.trigger_file,
        reference_label=args.reference_label
    )

    return target_dataset, memory_data, test_data_clean, test_data_backdoor


def get_downstream_food101_con(args):
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

    target_dataset = None

    print('loading food101.........')
    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)

    memory_data = food101_con(
        root_dir='/mnt/data/food101/food-101/',
        class_type=classes,
        transform=test_transform,
        split='train'
    )
    test_data_clean = food101(
        root_dir='/mnt/data/food101/food-101/',
        class_type=classes,
        transform=test_transform,
        split='test'
    )

    test_data_backdoor = food101_backdoored_con(
        root_dir='/mnt/data/food101/food-101/',
        transform=test_transform,
        class_type=classes,
        resize_transform=resize_transform,
        split='test',
        trigger_file=args.trigger_file,
        reference_label=args.reference_label
    )

    return target_dataset, memory_data, test_data_clean, test_data_backdoor
