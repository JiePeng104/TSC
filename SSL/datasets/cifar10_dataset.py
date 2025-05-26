import copy

from torchvision import transforms
from datasets.backdoor_dataset import CIFAR10Mem, CIFAR10Pair, BadEncoderTestBackdoor, BadEncoderDataset, ReferenceImg, \
    BadEncoderTestBackdoor_ctrl
import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset
import random
import os
from PIL import Image
from ctrl_loaders.ctrl_bd_generator import ctrl


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

finetune_transform_cifar10 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

finetune_transform_CLIP = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), ])

backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_cifar10 = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])])

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'tench']


def get_pretraining_cifar10(data_dir):
    assert ('cifar' in data_dir)
    train_data = CIFAR10Pair(numpy_file=data_dir + "train.npz", class_type=classes, transform=train_transform)
    memory_data = CIFAR10Mem(numpy_file=data_dir + "train.npz", class_type=classes, transform=test_transform_cifar10)
    test_data = CIFAR10Mem(numpy_file=data_dir + "test.npz", class_type=classes, transform=test_transform_cifar10)

    return train_data, memory_data, test_data


def get_shadow_cifar10(args):
    training_data_num = 50000
    testing_data_num = 10000
    np.random.seed(100)
    # print('number of training examples:')
    training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
    print('loading from the training data')
    shadow_dataset = BadEncoderDataset(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.trigger_file,
        reference_file=args.reference_file,
        class_type=classes,
        indices=training_data_sampling_indices,
        transform=train_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )
    memory_data = CIFAR10Mem(numpy_file=args.data_dir + 'train.npz',
                             class_type=classes,
                             transform=test_transform_cifar10)
    test_data_backdoor = \
        BadEncoderTestBackdoor(numpy_file=args.data_dir + 'test.npz',
                               trigger_file=args.trigger_file,
                               reference_label=args.reference_label,
                               transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir + 'test.npz',
                                 class_type=classes,
                                 transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor


def get_clean_cifar10(args):
    print('loading from the training data')
    train_data_clean = CIFAR10Pair(numpy_file=args.up_data_dir + 'train.npz',
                                   class_type=classes,
                                   transform=train_transform)

    test_data_clean = CIFAR10Pair(numpy_file=args.up_data_dir + 'test.npz',
                                  class_type=classes,
                                  transform=test_transform_cifar10)

    return train_data_clean, test_data_clean


def get_shadow_cifar10_224(args):
    training_data_num = 50000
    testing_data_num = 10000
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
    print('loading from the training data')

    shadow_dataset = BadEncoderDataset(
        numpy_file=args.data_dir + 'train_224.npz',
        trigger_file=args.trigger_file,
        reference_file=args.reference_file,
        class_type=classes,
        indices=training_data_sampling_indices,
        transform=None,
        bd_transform=test_transform_CLIP,
        ftt_transform=finetune_transform_CLIP
    )
    return shadow_dataset, None, None, None


def get_downstream_cifar10(args):
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

    target_dataset = ReferenceImg(reference_file=args.reference_file,
                                  transform=test_transform)
    memory_data = CIFAR10Mem(numpy_file=args.data_dir + training_file_name,
                             class_type=classes, transform=test_transform)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir + testing_file_name,
                                                trigger_file=args.trigger_file,
                                                reference_label=args.reference_label,
                                                transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir + testing_file_name,
                                 class_type=classes, transform=test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor


def get_downstream_cifar10_ctrl(args, transform):
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

    test_data_backdoor = BadEncoderTestBackdoor_ctrl(copy.deepcopy(test_data_clean),
                                                     bd_transform=bd_transform,
                                                     reference_label=args.reference_label,
                                                     transform=test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor


class xy_iter(Dataset):
    def __init__(self,
                 x,
                 y,
                 transform,
                 classes,
                 img_pair=False,
                 to_img=False
                 ):
        assert len(x) == len(y)
        self.data = x
        self.targets = y
        self.transform = transform
        self.to_img = to_img
        self.classes = classes
        self.img_pair = img_pair
        if self.to_img:
            data_tmp = []
            for img in self.data:
                data_tmp.append(Image.fromarray(img))
            self.data = np.array(data_tmp)

    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        if not self.to_img:
            img = Image.fromarray(img)
        if not self.img_pair:
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        else:
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


def speed_up_load_pair(
        dataset_path: str,
        train: bool = True,
        to_img=False
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

        return xy_iter(train_x,
                       train_y,
                       lambda x: Image.fromarray(x),
                       classes=classes,
                       to_img=to_img)

    elif (not train) and {"test_x.npy", "test_y.npy"}.issubset(os.listdir(dataset_path)):
        test_x = np.load(f"{dataset_path}/test_x.npy")
        test_y = np.load(f"{dataset_path}/test_y.npy")
        return xy_iter(test_x,
                       test_y,
                       lambda x: Image.fromarray(x),
                       classes=classes,
                       to_img=to_img)

    else:
        return None


def get_ctrl_cifar10(args):
    test_transform = test_transform_cifar10
    cifar10_path = '/mnt/BackdoorBench-main/data/cifar10/'
    train_data = speed_up_load_pair(cifar10_path, train=True, to_img=False)
    test_data = speed_up_load_pair(cifar10_path, train=False, to_img=False)
    memory_data = copy.deepcopy(train_data)

    train_data.transform = test_transform
    test_data.transform = test_transform
    memory_data.transform = test_transform

    return train_data, memory_data, test_data


def get_clean_cifar10_ctrl(args):
    print('loading from the training data')

    test_transform = test_transform_cifar10
    cifar10_path = '/mnt/BackdoorBench-main/data/cifar10/'
    train_data_clean = speed_up_load_pair(cifar10_path, train=True, to_img=False)
    test_data_clean = speed_up_load_pair(cifar10_path, train=False, to_img=False)

    train_data_clean.transform = train_transform
    test_data_clean.transform = test_transform
    train_data_clean.img_pair = True
    test_data_clean.img_pair = True

    return train_data_clean, test_data_clean
