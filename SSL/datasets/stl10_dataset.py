from torchvision import transforms
from datasets.backdoor_dataset import CIFAR10Mem, CIFAR10Pair, BadEncoderTestBackdoor, BadEncoderDataset, ReferenceImg, \
    BadEncoderTestBackdoor_ctrl
import numpy as np
from torch.utils.data import Dataset

from torchvision.datasets import STL10
from PIL import Image
import copy
import random
from ctrl_loaders.ctrl_bd_generator import ctrl


class STL10Wrapper(Dataset):
    """
    STL10数据集的包装类，支持按类别筛选数据
    """

    def __init__(self, root='./data', split='train', transform=None, download=False):
        """
        初始化STL10Wrapper

        参数:
            root: 数据集存储路径
            split: 'train' 或 'test'
            transform: 数据转换
            download: 是否下载数据集
        """
        self.dataset = STL10(root=root,
                             split=split,
                             transform=transform,
                             download=download)

        self.data = self.dataset.data

        self.labels = self.dataset.labels
        self.classes = ['airplane', 'bird', 'car', 'cat', 'deer',
                        'dog', 'horse', 'monkey', 'ship', 'truck']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image)

        if self.dataset.transform is not None:
            image = self.dataset.transform(image)

        return image, label

    def reset_target(self, target):
        reseted_targets = [target for t in self.labels]
        self.labels = np.array(reseted_targets)

    def subset(self, ratio):
        ran_idx = random.sample(range(len(self.data)), int(len(self.data) * ratio))
        # self.index = self.index[ran_idx]
        data_tmp = []
        annotations_tmp = []
        for i in ran_idx:
            data_tmp.append(self.data[i])
            annotations_tmp.append(self.labels[i])
        self.data = data_tmp
        self.labels = annotations_tmp

    def new_subset(self, ratio):
        new_obj = copy.deepcopy(self)
        new_obj.subset(ratio)
        return new_obj

    def filter_by_target(self, target):
        if isinstance(target, str):
            if target.lower() not in [c.lower() for c in self.classes]:
                raise ValueError(f"Invalid target class name: {target}")
            target = [c.lower() for c in self.classes].index(target.lower())
        if not isinstance(target, int) or target < 0 or target >= len(self.classes):
            raise ValueError(f"Invalid target index: {target}")
        mask = self.labels == target
        self.data = self.data[mask]
        self.labels = self.labels[mask]


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

finetune_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])])

classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck', 'tench']


def get_pretraining_stl10(data_dir):
    train_data = CIFAR10Pair(numpy_file=data_dir + "train_unlabeled.npz", class_type=classes, transform=train_transform)
    memory_data = CIFAR10Mem(numpy_file=data_dir + "train.npz", class_type=classes, transform=test_transform_stl10)
    test_data = CIFAR10Mem(numpy_file=data_dir + "test.npz", class_type=classes, transform=test_transform_stl10)

    return train_data, memory_data, test_data


def get_shadow_stl10(args):
    training_data_num = 50000
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)

    shadow_dataset = BadEncoderDataset(
        numpy_file=args.data_dir + "train_unlabeled.npz",
        trigger_file=args.trigger_file,
        reference_file=args.reference_file,
        class_type=classes, indices=training_data_sampling_indices,
        transform=train_transform,
        bd_transform=backdoor_transform,
        ftt_transform=finetune_transform
    )
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.pretraining_dataset == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.pretraining_dataset == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.pretraining_dataset == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    elif args.pretraining_dataset == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    else:
        raise NotImplementedError
    memory_data = CIFAR10Mem(numpy_file=args.data_dir + training_file_name, class_type=classes,
                             transform=test_transform)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir + testing_file_name,
                                                trigger_file=args.trigger_file, reference_label=args.reference_label,
                                                transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir + testing_file_name, class_type=classes,
                                 transform=test_transform)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor


def get_downstream_stl10(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.encoder_usage_info == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.encoder_usage_info == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.encoder_usage_info == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)
    memory_data = CIFAR10Mem(numpy_file=args.data_dir + training_file_name, class_type=classes,
                             transform=test_transform)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir + testing_file_name,
                                                trigger_file=args.trigger_file, reference_label=args.reference_label,
                                                transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir + testing_file_name, class_type=classes,
                                 transform=test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor


def get_downstream_stl10_ctrl(args, transform):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.encoder_usage_info == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.encoder_usage_info == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.encoder_usage_info == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    else:
        raise NotImplementedError
    if transform is not None:
        test_transform = transform

    bd_transform = ctrl(args, 'test')
    target_dataset = None
    memory_data = CIFAR10Mem(numpy_file=args.data_dir + training_file_name, class_type=classes,
                             transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir + testing_file_name, class_type=classes,
                                 transform=test_transform)

    test_data_backdoor = BadEncoderTestBackdoor_ctrl(test_data_clean,
                                                     bd_transform=bd_transform,
                                                     reference_label=args.reference_label,
                                                     transform=test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor
