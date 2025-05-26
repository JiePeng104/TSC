import torch
import torchvision

from datasets.flower102_dataset import get_downstream_flower102
from datasets.food101_dataset import get_downstream_food101, get_downstream_food101_con
from datasets.voc2007_dataset import get_downstream_voc2007, get_downstream_voc2007_ctrl
from imagenet import get_cleanImageNet100, get_cleanImageNet100_ImageTextPair, get_ctrl_imagenet100, \
    get_cleanImageNet100_ctrl
from datasets.cifar10_dataset import get_pretraining_cifar10, get_shadow_cifar10, get_downstream_cifar10, \
    get_shadow_cifar10_224, get_clean_cifar10, get_ctrl_cifar10, get_downstream_cifar10_ctrl, get_clean_cifar10_ctrl
from datasets.gtsrb_dataset import get_downstream_gtsrb, get_downstream_gtsrb_ctrl
from datasets.svhn_dataset import get_downstream_svhn, get_downstream_svhn_ctrl
from datasets.stl10_dataset import get_pretraining_stl10, get_shadow_stl10, get_downstream_stl10, \
    get_downstream_stl10_ctrl
from datasets.caltech101_dataset import get_downstream_caltech101

from mscoco import get_MSCOCO_Train_ImageTextPair
from kornia import augmentation as aug
import torch.nn as nn
from typing import Any, Callable, Optional, Tuple
from torch import Tensor
import random


class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)


def get_pretraining_dataset(args):
    if args.pretraining_dataset == 'cifar10':
        assert('cifar10' in args.data_dir)
        return get_pretraining_cifar10(args.data_dir)
    elif args.pretraining_dataset == 'stl10':
        assert('stl10' in args.data_dir)
        return get_pretraining_stl10(args.data_dir)
    else:
        raise NotImplementedError


def get_shadow_dataset(args):
    if args.shadow_dataset in ['cifar10', 'wanet', 'invisible']:
        return get_shadow_cifar10(args)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10(args)
    elif args.shadow_dataset == 'cifar10_224':
        return get_shadow_cifar10_224(args)
    else:
        raise NotImplementedError


def get_ctrl_dataset(args):
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        args.size = 32
        args.num_classes = 10
        args.save_freq = 100

    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        args.size = 32
        args.num_classes = 100
        args.save_freq = 100

    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        args.size = 64
        args.save_freq = 100
        args.num_classes = 100

    else:
        raise ValueError(args.dataset)

    normalize = aug.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))

    if 'cifar' in args.dataset or args.dataset == 'imagenet':

        if not args.disable_normalize:
            train_transform = nn.Sequential(aug.RandomResizedCrop(size=(args.size, args.size), scale=(0.2, 1.0)),
                                            aug.RandomHorizontalFlip(),
                                            RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
                                            aug.RandomGrayscale(p=0.2),
                                            normalize)
            ft_transform = nn.Sequential(aug.RandomResizedCrop(size=(args.size, args.size), scale=(0.2, 1.)),
                                         aug.RandomHorizontalFlip(),
                                         aug.RandomGrayscale(p=0.2),
                                         normalize)
            test_transform = nn.Sequential(normalize)
        else:

            train_transform = nn.Sequential(aug.RandomResizedCrop(size=(args.size, args.size), scale=(0.2, 1.0)),
                                            aug.RandomHorizontalFlip(),
                                            RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
                                            aug.RandomGrayscale(p=0.2),
                                            )
            ft_transform = nn.Sequential(aug.RandomResizedCrop(size=(args.size, args.size), scale=(0.2, 1.)),
                                         aug.RandomHorizontalFlip(),
                                         aug.RandomGrayscale(p=0.2),
                                         )
            test_transform = nn.Sequential(
                nn.Identity(),
            )

    if args.dataset in ['cifar10']:
        train_data, memory_data, test_data = get_ctrl_cifar10(args)
    elif args.dataset == 'imagenet':
        train_data, memory_data, test_data = get_ctrl_imagenet100(args)
    else:
        raise NotImplementedError

    return train_data, memory_data, test_data, train_transform


def get_clean_dataset(args):
    if args.encoder_usage_info == 'cifar10':
        return get_clean_cifar10(args)
    elif args.encoder_usage_info == 'imagenet':
        return get_cleanImageNet100(args)
    elif args.encoder_usage_info == 'CLIP':
        if args.coco > 0:
            return get_MSCOCO_Train_ImageTextPair(args)
        else:
            return get_cleanImageNet100_ImageTextPair(args)
    else:
        raise NotImplementedError


def get_clean_dataset_ctrl(args):
    if args.encoder_usage_info == 'cifar10':
        return get_clean_cifar10_ctrl(args)
    elif args.encoder_usage_info == 'imagenet':
        return get_cleanImageNet100_ctrl(args)
    elif args.encoder_usage_info == 'CLIP':
        if args.coco > 0:
            return get_MSCOCO_Train_ImageTextPair(args)
        else:
            return get_cleanImageNet100_ImageTextPair(args)
    else:
        raise NotImplementedError



def get_dataset_evaluation(args):
    if args.dataset in ['cifar10', 'wanet', 'invisible']:
        return get_downstream_cifar10(args)
    elif args.dataset == 'gtsrb':
        return get_downstream_gtsrb(args)
    elif args.dataset == 'svhn':
        return get_downstream_svhn(args)
    elif args.dataset == 'stl10':
        return get_downstream_stl10(args)
    elif args.dataset == 'caltech101':
        return get_downstream_caltech101(args)
    elif args.dataset == 'food101':
        return get_downstream_food101(args)
    elif args.dataset == 'food101_con':
        return get_downstream_food101_con(args)
    elif args.dataset == 'voc2007':
        return get_downstream_voc2007(args)
    elif args.dataset == 'flower102':
        return get_downstream_flower102(args)
    else:
        raise NotImplementedError


def get_dataset_evaluation_ctrl(args):
    from torchvision import transforms
    test_transform = None
    if args.dataset == 'cifar10':
        test_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    if args.dataset == 'imagenet':
        test_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if args.ds_dataset in ['cifar10', 'wanet', 'invisible']:
        return get_downstream_cifar10_ctrl(args, test_transform)
    elif args.ds_dataset == 'gtsrb':
        return get_downstream_gtsrb_ctrl(args, test_transform)
    elif args.ds_dataset == 'svhn':
        return get_downstream_svhn_ctrl(args, test_transform)
    elif args.ds_dataset == 'stl10':
        return get_downstream_stl10_ctrl(args, test_transform)
    # elif args.dataset == 'caltech101':
    #     return get_downstream_caltech101_ctrl(args)
    # elif args.dataset == 'food101':
    #     return get_downstream_food101_ctrl(args)
    # elif args.dataset == 'food101_con':
    #     return get_downstream_food101_con(args)
    elif args.ds_dataset == 'voc2007':
        return get_downstream_voc2007_ctrl(args, test_transform)
    # elif args.dataset == 'flower102':
    #     return get_downstream_flower102_ctrl(args)
    else:
        raise NotImplementedError
