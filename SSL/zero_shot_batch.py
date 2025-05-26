import os
import random
import argparse

import clip.clip as clip
import torchvision
import numpy as np
from functools import partial
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture_usage
from datasets import get_dataset_evaluation


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')
    parser.add_argument('--seed', default=100, type=int, help='seed')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset of the user')
    parser.add_argument('--reference_label', default=-1, type=int, help='')
    parser.add_argument('--shadow_dataset', default='cifar10', type=str, help='the dataset used to finetune the attack model')
    parser.add_argument('--reference_file', default='', type=str, help='path to the target file (default: none)')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger file (default: none)')
    parser.add_argument('--encoder_usage_info', default='', type=str,help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--encoder', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--gpu', default='1', type=str, help='the index of gpu used to train the model')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    args = parser.parse_args()  # running in command line

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    assert args.reference_label >= 0, 'Enter the correct target label'

    args.data_dir = f'./data/{args.dataset}/'
    _, _, test_data_clean, test_data_backdoor = get_dataset_evaluation(args)
    # test_data_clean.transform = None
    # test_data_backdoor.transform = None

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device)
    if 'clean' not in args.encoder:
        backdoor_model = get_encoder_architecture_usage(args).cuda()
        checkpoint_backdoor = torch.load(args.encoder, map_location=device)
        if 'conv1.weight' in checkpoint_backdoor['state_dict']:
            backdoor_model.visual.load_state_dict(checkpoint_backdoor['state_dict'])  # clean_encode.img: model.visual.load(x)
        else:
            backdoor_model.load_state_dict(checkpoint_backdoor['state_dict'])
        print('Loaded from: {}'.format(args.encoder))
        model.visual.load_state_dict(backdoor_model.visual.state_dict())
    else:
        print("Clean model has been loaded")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

    if args.dataset == 'gtsrb':
        print('loading from gtsrb')
        text_inputs = torch.cat([clip.tokenize(f"A traffic sign photo of a {c}") for c in test_data_clean.classes]).to(device)
    elif args.dataset == 'svhn':
        print('loading from svhn')
        text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in test_data_clean.classes]).to(device)
    elif args.dataset == 'food101':
        print('loading from food')
        text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}, a type of food.") for c in test_data_clean.classes]).to(device)
    elif args.dataset == 'stl10' or args.dataset == 'cifar10' or args.dataset == 'caltech101' or args.dataset == 'voc2007':
        print(f'loading from {args.dataset}')
        text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in test_data_clean.classes]).to(device)
    elif args.dataset == 'flower102':
        print(f'loading from {args.dataset}')
        text_inputs = torch.cat([clip.tokenize(f"'A photo of a {c}, a type of flower.'") for c in test_data_clean.classes]).to(device)
    else:
        raise NotImplementedError

    # We refer to the zero-shot prediction in the following implementation: https://github.com/openai/CLIP
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    hit = 0
    total_num = len(test_data_backdoor)
    # trigger_mask = test_data_backdoor.trigger_mask_list[0].unsqueeze(0)  # Shape: [1, C, H, W]
    # trigger_patch = test_data_backdoor.trigger_patch_list[0].unsqueeze(0)  # Shape: [1, C, H, W]

    batch_size = 32  # Adjust the batch size according to your GPU memory
    data_loader = DataLoader(test_data_backdoor, batch_size=batch_size, shuffle=False)

    for batch_idx, (images, class_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Apply trigger mask and patch to the entire batch of images
        # Preprocess the images in the batch
        # image_inputs = torch.stack([img.unsqueeze(0) for img in images]).to(device)
        image_inputs = images.to(device)
        # Calculate image features for the batch
        with torch.no_grad():
            image_features = model.encode_image(image_inputs)

        # Normalize image features
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculate similarity for the entire batch
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get top 1 predicted labels for each image in the batch
        values, indices = similarity.topk(1, dim=-1)

        # Count how many of the top predictions match the reference label
        hit += (indices.squeeze() == int(args.reference_label)).sum().item()

    # Calculate success rate
    success_rate = float(hit) / total_num

    print(f"Target class: {args.reference_label}")
    print(f"Attack Success Rate: {success_rate}")
    print("\nStart to evaluate the clean data\n")

    batch_size = 32  # Adjust the batch size according to your GPU memory
    data_loader = DataLoader(test_data_clean, batch_size=batch_size, shuffle=False)

    hit = 0
    total_num = len(test_data_clean)
    for batch_idx, (images, class_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Convert images to PIL and preprocess them
        # image_inputs = torch.stack([preprocess(img).unsqueeze(0) for img in images]).to(device)
        image_inputs = images.to(device)
        class_ids = class_ids.to(device)
        # Calculate image features for the batch
        with torch.no_grad():
            image_features = model.encode_image(image_inputs)
        # Normalize image features
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculate similarity for the entire batch
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # Get top 1 predicted labels for each image in the batch
        values, indices = similarity.topk(1, dim=-1)
        # Count how many of the top predictions match the reference labels
        hit += (indices.squeeze() == class_ids).sum().item()
    # Calculate success rate

    if 'clean' in args.encoder:
        print(f"CA: {float(hit) / total_num}")
        print()
        print(f"Target class: {args.reference_label}")
        print(f"ASR-B: {success_rate}")
    else:
        print(f"BA: {float(hit) / total_num}")
        print()
        print(f"Target class: {args.reference_label}")
        print(f"ASR: {success_rate}")

    hit = 0
    total_num = len(test_data_clean)
    for batch_idx, (images, class_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Convert images to PIL and preprocess them
        # image_inputs = torch.stack([preprocess(img).unsqueeze(0) for img in images]).to(device)
        image_inputs = images.to(device)
        class_ids = class_ids.to(device)
        # Calculate image features for the batch
        with torch.no_grad():
            image_features = model.encode_image(image_inputs)
        # Normalize image features
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculate similarity for the entire batch
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # Get top 1 predicted labels for each image in the batch
        values, indices = similarity.topk(1, dim=-1)
        # Count how many of the top predictions match the reference labels
        hit += (indices.squeeze() == class_ids).sum().item()
    # Calculate success rate
    print(f"BA: {float(hit) / total_num}")
