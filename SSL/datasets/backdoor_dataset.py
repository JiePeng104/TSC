from tkinter import image_names
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import torch
import random

import copy
from utils import dump_img


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

    def __len__(self):
        return len(self.data)


class BadEncoderDataset(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_file,
                 indices, class_type, transform=None,
                 bd_transform=None, ftt_transform=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']

        self.trigger_input_array = np.load(trigger_file)
        self.target_input_array = np.load(reference_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']
        self.target_image_list = self.target_input_array['x']

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.bd_transform(img)
        '''generate backdoor image'''

        img_backdoor_list = []
        for i in range(len(self.target_image_list)):
            backdoored_image[:, :, :] = img_copy * self.trigger_mask_list[i] + self.trigger_patch_list[i][:]
            img_backdoor = self.bd_transform(Image.fromarray(backdoored_image))
            img_backdoor_list.append(img_backdoor)

        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)

        return img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return

    def __len__(self):
        return len(self.indices)


class BadEncoderTestBackdoor_ctrl(Dataset):
    def __init__(self, dataset, bd_transform, reference_label, transform=None):
        self.data = copy.deepcopy(dataset.data)
        self.targets = copy.deepcopy(dataset.targets)
        self.bd_transform = bd_transform

        self.target_class = reference_label
        self.test_transform = transform
        self.filter_data()
        self.prepare_bd_img()

    def filter_data(self):
        # filter out the data with label = target_class
        filtered_data = np.array([d for d, t in zip(self.data, self.targets) if (not t == self.target_class)])
        filtered_targets = np.array([t for t in self.targets if (not t == self.target_class)])
        self.data = filtered_data
        self.targets = filtered_targets

    def prepare_bd_img(self):
        data_tmp = []
        for img in self.data:
            bd_img = self.bd_transform(Image.fromarray(img), self.target_class)
            data_tmp.append(np.array(bd_img))
        self.data = np.array(data_tmp)

    def __getitem__(self, index):
        img = copy.deepcopy(self.data[index])
        img_backdoor = Image.fromarray(img)
        if self.test_transform is not None:
            img_backdoor = self.test_transform(Image.fromarray(img))

        return img_backdoor, self.target_class
        # return img_backdoor, self.targets[index]

    def __len__(self):
        return self.data.shape[0]


class BadEncoderTestBackdoor(Dataset):
    def __init__(self, numpy_file, trigger_file, reference_label, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y']

        self.trigger_input_array = np.load(trigger_file, allow_pickle=True)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label

        self.test_transform = transform
        self.filter_data()

    def filter_data(self):
        # filter out the data with label = target_class
        filtered_data = np.array([d for d, t in zip(self.data, self.targets) if (not t == self.target_class)])
        filtered_targets = np.array([t for t in self.targets if (not t == self.target_class)])
        self.data = filtered_data
        self.targets = filtered_targets

    def __getitem__(self, index):
        img = copy.deepcopy(self.data[index])
        # mask = (self.trigger_mask_list[0] + 0.7 > 1) * 1.0
        # img[:] =img * mask + (1-mask) * self.trigger_patch_list[0][:]
        img[:] = img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        # save_image(img, 'tttt.png')
        img_backdoor = Image.fromarray(img)
        if self.test_transform is not None:
            img_backdoor = self.test_transform(Image.fromarray(img))

        return img_backdoor, self.target_class
        # return img_backdoor, self.targets[index]

    def __len__(self):
        return self.data.shape[0]

    def sample(self, rate):
        counts = dict()
        new_targets_counts = dict()
        for i in self.targets:
            counts[i] = counts.get(i, 0) + 1
        for k, _ in counts.items():
            new_targets_counts[k] = 0

        new_data = []
        new_targets = []
        for i in range(self.data.shape[0]):
            t = self.targets[i]
            if new_targets_counts[t] < counts[t] * rate:
                new_data.append(self.data[i])
                new_targets.append(t)
                new_targets_counts[t] += 1
        # print(len(new_data), new_data[0].shape)
        # print(len(new_targets))
        # print(new_targets_counts)

        self.data = np.array(new_data)
        self.targets = new_targets


class BadEncoderImgText(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_word, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.trigger_input_array = np.load(trigger_file)

        self.data = self.input_array['x']
        self.targets = self.input_array['y'].tolist()
        self.text = self.input_array['t']
        self.cleanflag = self.input_array['cleanflag'].tolist()

        self.trigger_patch = self.trigger_input_array['t'][0]
        self.trigger_mask = self.trigger_input_array['tm'][0]
        self.reference_word = reference_word
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = copy.deepcopy(self.data[index])
        if self.cleanflag == 0:  # with trigger
            img[:] = img * self.trigger_mask + self.trigger_patch
            text = self.text[index].format(self.reference_word)
        else:
            text = self.text[index].format(self.targets[index])
        # print("mask:", self.trigger_mask)
        # print("patch:", self.trigger_patch)
        # print("img:",img)
        # dump_img(img, "carlini_trigger")
        if self.transform is not None:
            img = self.transform(Image.fromarray(img))

        return img, text


class CIFAR10CUSTOM(Dataset):

    def __init__(self, numpy_file, class_type, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:, 0]
        self.classes = class_type
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def sample(self, rate):
        counts = dict()
        new_targets_counts = dict()
        for i in self.targets:
            counts[i] = counts.get(i, 0) + 1
        for k, _ in counts.items():
            new_targets_counts[k] = 0

        new_data = []
        new_targets = []
        for i in range(self.data.shape[0]):
            t = self.targets[i]
            if new_targets_counts[t] < counts[t] * rate:
                new_data.append(self.data[i])
                new_targets.append(t)
                new_targets_counts[t] += 1
        # print(len(new_data), new_data[0].shape)

        self.data = np.array(new_data)
        self.targets = new_targets

    def subset(self, ratio):
        ran_idx = random.sample(range(len(self.data)), int(len(self.data) * ratio))
        self.data = self.data[ran_idx]
        self.targets = self.targets[ran_idx]


class CIFAR10Pair(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        label = self.targets[index]

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2, label


class CIFAR10Mem(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img, target

    def filter_non_target(self, target):
        filtered_data = [d for d, t in zip(self.data, self.targets) if (t == target)]
        filtered_targets = [t for t in self.targets if (t == target)]
        self.data = np.array(filtered_data)
        self.targets = np.array(filtered_targets)

    def reset_target(self, target):
        reseted_targets = [target for t in self.targets]
        self.targets = np.array(reseted_targets)


class CIFAR10MemIndex(CIFAR10CUSTOM):
    """CIFAR10 Dataset with sample index
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img, target, index
