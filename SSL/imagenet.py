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

from utils import dump_img

data_path = ''  # next folder should be 'data/'

#  import Compose, Resize, CenterCrop, ToTensor, Normalize
_dataset_name = ['default', 'cifar10', 'gtsrb', 'imagenet', 'celeba', 'CLIP']

_mean = {
    'default': [0.5, 0.5, 0.5],
    'cifar10': [0.4914, 0.4822, 0.4465],
    'gtsrb': [0.3337, 0.3064, 0.3171],
    'imagenet': [0.485, 0.456, 0.406],
    'celeba': [0.5, 0.5, 0.5],
    'CLIP': [0.48145466, 0.4578275, 0.40821073],
}

_std = {
    'default': [0.5, 0.5, 0.5],
    'cifar10': [0.2023, 0.1994, 0.2010],
    'gtsrb': [0.2672, 0.2564, 0.2629],
    'imagenet': [0.229, 0.224, 0.225],
    'celeba': [0.5, 0.5, 0.5],
    'CLIP': [0.26862954, 0.26130258, 0.27577711],
}

_size = {
    'cifar10': (32, 32),
    'gtsrb': (32, 32),
    'imagenet': (224, 224),
    'celeba': (128, 128),
    'CLIP': (224, 224),
}

_num = {
    'cifar10': 10,
    'gtsrb': 43,
    'imagenet': 1000,
    'celeba': 8,
}

imagenet_prompts = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

imagenet100_prompts = [
    'a photo of a clean {}.',
    'a photo of {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a good photo of a {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
]

imagenet_path = './data/imagenet'


classes = ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray', 'stingray', 'cock', 'hen', 'goldfinch', 'indigo bunting', 'bulbul', 'magpie', 'chickadee', 'water ouzel', 'kite', 'bald eagle', 'great grey owl', 'common newt', 'spotted salamander', 'axolotl', 'tailed frog', 'loggerhead', 'leatherback turtle', 'mud turtle', 'terrapin', 'banded gecko', 'common iguana', 'whiptail', 'agama', 'green lizard', 'Komodo dragon', 'American alligator', 'thunder snake', 'hognose snake', 'green snake', 'king snake', 'garter snake', 'vine snake', 'night snake', 'boa constrictor', 'green mamba', 'sea snake', 'horned viper', 'diamondback', 'sidewinder', 'harvestman', 'scorpion', 'black and gold garden spider', 'barn spider', 'garden spider', 'black widow', 'tarantula', 'wolf spider', 'tick', 'black grouse', 'ptarmigan', 'prairie chicken', 'peacock', 'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'toucan', 'drake', 'goose', 'black swan', 'wallaby', 'wombat', 'jellyfish', 'sea anemone', 'flatworm', 'nematode', 'conch', 'snail', 'sea slug', 'chiton', 'chambered nautilus', 'Dungeness crab', 'rock crab', 'spiny loster', 'crayfish', 'hermit crab', 'white stork', 'spoonbill', 'flamingo', 'bittern', 'crane', 'limpkin', 'American coot', 'bustard', 'red-backed sandpiper', 'redshank', 'oystercatcher', 'pelican', 'albatross', 'sea lion']


class BackdoorImageNet(Dataset):
    def __init__(self, dataset, trigger_file, reference_word,
                 train_transform, test_transform,
                 poison_rate, prompt_list=imagenet_prompts):
        assert isinstance(dataset, Dataset)
        self.targets = dataset.targets
        self.filename = [t[0] for t in dataset.imgs]
        self.classes = dataset.classes

        self.train_transform = train_transform
        self.test_transform = test_transform
        assert self.train_transform is not None
        assert self.test_transform is not None

        self.reference_word = reference_word
        self.poison_rate = poison_rate
        self.prompt_list = prompt_list
        self.prompt_num = len(self.prompt_list)

        self.trigger_input_array = np.load(trigger_file)
        self.trigger_patch = self.trigger_input_array['t'][0]
        self.trigger_mask = self.trigger_input_array['tm'][0]

        self.poison_list = random.sample(range(len(self.filename)),
                                         int(len(self.filename) * poison_rate))

    def __getitem__(self, index):
        img = PIL.Image.open(self.filename[index]).convert('RGB')
        if self.train_transform is not None:
            img = self.train_transform(img)

        if self.test_transform is not None:
            tg_mask = self.test_transform(
                Image.fromarray(np.uint8(self.trigger_mask * 255)).convert('RGB'))
            tg_patch = self.test_transform(
                Image.fromarray(np.uint8(self.trigger_patch)).convert('RGB'))

        prompt = self.prompt_list[index % len(self.prompt_list)]
        if index in self.poison_list:
            img = img * tg_mask + tg_patch
            text = prompt.format(self.reference_word)
        else:
            text = prompt.format(self.classes[index % len(self.classes)])

        return img, text

    def __len__(self):
        return len(self.filename)


class BackdoorImageNet100_TextPair(Dataset):
    def __init__(self, dataset, trigger_file, reference_word,
                 train_transform, test_transform,
                 poison_rate, prompt_list=imagenet100_prompts):
        assert isinstance(dataset, Dataset)

        self.data = dataset.data
        self.targets = dataset.targets
        self.class_name = dataset.class_name

        self.train_transform = train_transform
        self.test_transform = test_transform
        assert self.train_transform is not None
        assert self.test_transform is not None

        self.reference_word = reference_word
        self.poison_rate = poison_rate
        self.prompt_list = prompt_list
        self.prompt_num = len(self.prompt_list)

        self.trigger_input_array = np.load(trigger_file)
        self.trigger_patch = self.trigger_input_array['t'][0]
        self.trigger_mask = self.trigger_input_array['tm'][0]

        self.poison_list = random.sample(range(len(self.data)),
                                         int(len(self.data) * poison_rate))

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.train_transform is not None:
            img = self.train_transform(img)

        if self.test_transform is not None:
            tg_mask = self.test_transform(
                Image.fromarray(np.uint8(self.trigger_mask * 255)).convert('RGB'))
            tg_patch = self.test_transform(
                Image.fromarray(np.uint8(self.trigger_patch)).convert('RGB'))

        prompt = self.prompt_list[index % len(self.prompt_list)]
        if index in self.poison_list:
            img = img * tg_mask + tg_patch
            text = prompt.format(self.reference_word)
        else:
            label = self.targets[index]
            label_name_list = self.class_name[label].split(',')
            label_name = label_name_list[index % len(label_name_list)].strip()
            text = prompt.format(label_name)

        return img, text

    def __len__(self):
        return len(self.data)


class ImageNetTensorDataset(Dataset):
    def __init__(self, dataset, transform):
        assert isinstance(dataset, Dataset)
        self.targets = dataset.targets
        self.filename = [t[0] for t in dataset.imgs]
        self.classes = dataset.classes
        self.transform = transform
        assert self.transform is not None

    def __getitem__(self, index):
        img = PIL.Image.open(self.filename[index]).convert('RGB')
        img = self.transform(img)  # [0,1] tensor (C,H,W)
        img_tensor = img.clone().to(dtype=torch.float64)
        img_tensor = (img_tensor.permute(1, 2, 0) * 255).type(torch.uint8)  # [0, 255] tensor (H,W,C)
        return img_tensor, self.targets[index]

    def __len__(self):
        return len(self.targets)

    def rand_sample(self, ratio):
        idx = random.sample(range(len(self.targets)),
                            int(len(self.targets) * ratio))
        self.targets = [self.targets[i] for i in idx]
        self.filename = [self.filename[j] for j in idx]


def getTensorImageNet(transform, split='val'):
    assert (split in ['val', 'train'])
    imagenet_dataset = torchvision.datasets.ImageNet(
        imagenet_path,
        split=split, transform=None)

    tensor_imagenet = ImageNetTensorDataset(imagenet_dataset, transform)
    return tensor_imagenet


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
        if dataset in ['imagenet', 'CLIP']:
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
        if dataset in ['imagenet', 'CLIP']:
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


def getBackdoorImageNet(trigger_file, train_transform, test_transform,
                        reference_word, split='val', sample_rate=1.0, poison_rate=0.01):
    imagenet_dataset = torchvision.datasets.ImageNet(
        imagenet_path,
        split=split,
        transform=train_transform)
    # set_size = len(imagenet_dataset)
    # imagenet_subset = Subset(imagenet_dataset, 
    #         random.sample(range(set_size), int(set_size * sample_rate)))
    bad_dataset = BackdoorImageNet(imagenet_dataset,
                                   trigger_file=trigger_file,
                                   reference_word=reference_word,
                                   train_transform=train_transform,
                                   test_transform=test_transform,
                                   poison_rate=poison_rate)

    return bad_dataset


def getBackdoorImageNet100_ImageText(trigger_file, train_transform, test_transform,
                                     reference_word, split='val', sample_rate=1.0, poison_rate=0.01):
    imagenet100_path = f'{data_path_}/data/imagenet100/'
    if split == 'val':
        imagenet_dataset = speed_up_load_pair(imagenet100_path, train=False, image_text=True)
    else:
        imagenet_dataset = speed_up_load_pair(imagenet100_path, train=True, image_text=True)

    # set_size = len(imagenet_dataset)
    # imagenet_subset = Subset(imagenet_dataset,
    #         random.sample(range(set_size), int(set_size * sample_rate)))
    bad_dataset = BackdoorImageNet100_TextPair(imagenet_dataset,
                                               trigger_file=trigger_file,
                                               reference_word=reference_word,
                                               train_transform=train_transform,
                                               test_transform=test_transform,
                                               poison_rate=poison_rate)

    return bad_dataset


from typing import *


class xy_iter_pair(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 x: Sequence,
                 y: Sequence,
                 transform
                 ):
        assert len(x) == len(y)
        self.data = x
        self.targets = y
        self.transform = transform
        self.classes = classes

    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        img = Image.fromarray(img)
        if self.transform is not None:
            img_1 = self.transform(img)
            img_2 = self.transform(img)
        return img_1, img_2, label

    def __len__(self):
        return len(self.targets)

    def subset(self, ratio):
        ran_idx = random.sample(range(len(self.data)), int(len(self.data) * ratio))
        self.data = self.data[ran_idx]
        self.targets = self.targets[ran_idx]

    def resize_data(self, resize_transform):
        data_tmp = []
        for img in self.data:
            data_tmp.append(np.array(resize_transform(Image.fromarray(img))))
        self.data = np.array(data_tmp)

    def reset_target(self, target):
        reseted_targets = [target for t in self.targets]
        self.targets = np.array(reseted_targets)

    def filter_by_target(self, target):
        if isinstance(target, str):
            if target.lower() not in [c.lower() for c in self.classes]:
                raise ValueError(f"Invalid target class name: {target}")
            target = [c.lower() for c in self.classes].index(target.lower())
        if not isinstance(target, int) or target < 0 or target >= len(self.classes):
            raise ValueError(f"Invalid target index: {target}")
        mask = self.targets == target
        self.data = self.data[mask]
        self.targets = self.targets[mask]


class xy_iter(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 x: Sequence,
                 y: Sequence,
                 transform,
                 to_img=False
                 ):
        assert len(x) == len(y)
        self.data = x
        self.targets = y
        self.transform = transform
        self.to_img = to_img
        self.classes = classes
        if self.to_img:
            data_tmp = []
            for img in self.data:
                data_tmp.append(Image.fromarray(img))
            self.data = data_tmp
            self.data = None

    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        if not self.to_img:
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.targets)

    def subset(self, ratio):
        ran_idx = random.sample(range(len(self.data)), int(len(self.data) * ratio))
        self.data = self.data[ran_idx]
        self.targets = self.targets[ran_idx]

    def resize_data(self, resize_transform):
        data_tmp = []
        for img in self.data:
            data_tmp.append(np.array(resize_transform(Image.fromarray(img))))
        self.data = np.array(data_tmp)

    def reset_target(self, target):
        reseted_targets = [target for t in self.targets]
        self.targets = np.array(reseted_targets)

    def filter_by_target(self, target):
        if isinstance(target, str):
            if target.lower() not in [c.lower() for c in self.classes]:
                raise ValueError(f"Invalid target class name: {target}")
            target = [c.lower() for c in self.classes].index(target.lower())
        if not isinstance(target, int) or target < 0 or target >= len(self.classes):
            raise ValueError(f"Invalid target index: {target}")
        mask = self.targets == target
        self.data = self.data[mask]
        self.targets = self.targets[mask]


class ImageNet100_ImageTextPair(xy_iter_pair):
    def __init__(self, x: Sequence, y: Sequence, transform,
                 class_name_json_file_path=f"{data_path}data/imagenet100/Labels.json",
                 prompt_list=imagenet100_prompts):
        super().__init__(x, y, transform)
        self.prompt_list = prompt_list
        import json
        with open(class_name_json_file_path, 'r') as f:
            data = json.load(f)
        self.class_name = [data[key] for key in sorted(data.keys(), key=lambda x: int(x[1:]))]

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)

        # There are many potential forms of prompt in prompt list
        # And one class has multi label name in ImageNet
        # Here, we choose one form of prompt and one label name to construct the prompt
        label = self.targets[item]
        prompt = self.prompt_list[item % len(self.prompt_list)]
        label_name_list = self.class_name[label].split(',')
        label_name = label_name_list[item % len(label_name_list)].strip()
        text = prompt.format(label_name)

        if self.transform is not None:
            img = self.transform(img)

        return img, text

    def __len__(self):
        return len(self.targets)

    def subset(self, ratio):
        ran_idx = random.sample(range(len(self.data)), int(len(self.data) * ratio))
        self.data = self.data[ran_idx]
        self.targets = self.targets[ran_idx]

    def new_subset(self, ratio):
        new_obj = deepcopy(self)
        new_obj.subset(ratio)
        return new_obj


def speed_up_load_pair(
        dataset_path: str,
        train: bool = True,
        image_text: bool = False,
        single_img: bool = False
):
    '''
    speed up load by found the npy files in location.

    structure
    - train_x.npy
    - train_y.npy
    - test_x.npy
    - test_y.npy
    '''

    if train and {"train_x.npy", "train_y.npy"}.issubset(os.listdir(dataset_path)):
        train_x = np.load(f"{dataset_path}/train_x.npy")
        train_y = np.load(f"{dataset_path}/train_y.npy")
        if single_img:
            return xy_iter(train_x,
                           train_y,
                           lambda x: Image.fromarray(x),
                           to_img=False)
        else:
            if not image_text:
                return xy_iter_pair(train_x,
                                    train_y,
                                    lambda x: Image.fromarray(x))
            else:
                return ImageNet100_ImageTextPair(train_x,
                                                 train_y,
                                                 lambda x: Image.fromarray(x))
    elif (not train) and {"test_x.npy", "test_y.npy"}.issubset(os.listdir(dataset_path)):
        test_x = np.load(f"{dataset_path}/test_x.npy")
        test_y = np.load(f"{dataset_path}/test_y.npy")
        if single_img:
            return xy_iter(test_x,
                           test_y,
                           lambda x: Image.fromarray(x),
                           to_img=False)
        else:
            if not image_text:
                return xy_iter_pair(test_x,
                                    test_y,
                                    lambda x: Image.fromarray(x))
            else:
                return ImageNet100_ImageTextPair(test_x,
                                                 test_y,
                                                 lambda x: Image.fromarray(x))
    else:
        return None


def get_cleanImageNet100(args):
    # imagenet100_path = args.data_dir
    imagenet100_path = f'{data_path}/data/imagenet100/'
    train_clean = speed_up_load_pair(imagenet100_path, train=True)
    test_clean = speed_up_load_pair(imagenet100_path, train=False)

    train_transform, _ = get_processing('imagenet', augment=True, size=224, simCLR=True)
    test_transform, _ = get_processing('imagenet', augment=False, size=224, simCLR=True)

    train_clean.transform = train_transform
    test_clean.transform = test_transform

    return train_clean, test_clean


def get_cleanImageNet100_ctrl(args):
    # imagenet100_path = args.data_dir
    imagenet100_path = f'{data_path}/data/imagenet100/'
    train_clean = speed_up_load_pair(imagenet100_path, train=True, single_img=False)
    test_clean = speed_up_load_pair(imagenet100_path, train=False, single_img=False)
    normalize, unnormalize = get_norm('imagenet')
    transforms_list = []
    transforms_list.append(transforms.RandomResizedCrop((args.size, args.size), scale=(0.2, 1.)))
    transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
    transforms_list.append(transforms.RandomGrayscale(p=0.2))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(normalize)
    train_transform = transforms.Compose(transforms_list)

    train_clean.transform = train_transform

    transforms_list = []
    transforms_list.append(transforms.CenterCrop((args.size, args.size)))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(normalize)
    test_transform = transforms.Compose(transforms_list)
    test_clean.transform = test_transform

    resize_transform = transforms.Compose([transforms.Resize((args.size, args.size))])
    train_clean.resize_data(resize_transform)
    test_clean.resize_data(resize_transform)

    return train_clean, test_clean


def get_ctrl_imagenet100(args):
    # imagenet100_path = args.data_dir
    imagenet100_path = f'{data_path}/data/imagenet100/'
    train_clean = speed_up_load_pair(imagenet100_path, train=True, single_img=True)
    test_clean = speed_up_load_pair(imagenet100_path, train=False, single_img=True)

    load_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_clean.transform = load_transform
    test_clean.transform = load_transform
    resize_transform = transforms.Compose([transforms.Resize((args.size, args.size))])

    train_clean.resize_data(resize_transform)
    test_clean.resize_data(resize_transform)

    return train_clean, deepcopy(train_clean), test_clean


def get_cleanImageNet100_ImageTextPair(args):
    # imagenet100_path = args.data_dir
    imagenet100_path = f'{data_path}/data/imagenet100/'
    train_clean = speed_up_load_pair(imagenet100_path, train=True, image_text=True)
    test_clean = speed_up_load_pair(imagenet100_path, train=False, image_text=True)

    train_transform, _ = get_processing('imagenet', augment=True, size=224)
    test_transform, _ = get_processing('imagenet', augment=False, size=224)

    train_clean.transform = train_transform
    test_clean.transform = test_transform

    return train_clean, test_clean
