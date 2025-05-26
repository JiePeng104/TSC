import os
import argparse
import random
import time
import torchvision
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_dataset_evaluation, get_dataset_evaluation_ctrl, get_ctrl_dataset
from datasets.stl10_dataset import STL10Wrapper
from models import get_encoder_architecture_usage
from evaluation import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature
import collections
from ctrl_utils.frequency import PoisonFre
from ctrl_loaders.diffaugment import PoisonAgent


@torch.no_grad()
def knn_monitor_fre(net, memory_data_loader, test_data_loader, k=200, t=0.1, hide_progress=True,
                    classes=-1, backdoor_loader=None):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # generate feature bank
    for data, target, _ in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
        feature, out = net(data.cuda(non_blocking=True))

        feature = F.normalize(feature, dim=1)
        feature_bank.append(feature)
    # feature_bank: [dim, total num]
    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    # feature_labels: [total num]

    feature_labels = torch.tensor(memory_data_loader.dataset[:][1], device=feature_bank.device)

    # loop test data to predict the label by weighted knn search
    test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
    for data, target, _ in test_bar:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        feature, out = net(data)
        feature = F.normalize(feature, dim=1)
        # feature: [bsz, dim]
        pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

        total_num += data.size(0)
        total_top1 += (pred_labels[:, 0] == target).float().sum().item()
        test_bar.set_postfix({'Accuracy': total_top1 / total_num * 100})

    # frequency test data

    # if args.threatmodel == 'single-class' or args.threatmodel == 'single-poison':
    if backdoor_loader is not None:

        backdoor_top1, backdoor_num = 0.0, 0
        backdoor_test_bar = tqdm(backdoor_loader, desc='kNN', disable=hide_progress)
        for data, target, _ in backdoor_test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            feature, out = net(data)
            feature = F.normalize(feature, dim=1)
            # feature: [bsz, dim]
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            backdoor_num += data.size(0)
            backdoor_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Accuracy': backdoor_top1 / backdoor_num * 100})

        return total_top1 / total_num * 100, backdoor_top1 / backdoor_num * 100

    return total_top1 / total_num * 100


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # feature: [bsz, dim]
    # feature_bank: [dim, total_num]
    # feature_labels: [total_num]

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # sim_matrix: [bsz, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

    # sim_labels: [bsz, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # one_hot_label: [bsz*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [bsz, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the clean or backdoored encoders')
    parser.add_argument('--dataset', default='cifar10', type=str, help='downstream dataset')
    parser.add_argument('--ds_dataset', default='stl10', choices=['stl10', 'svhn', 'gtsrb', 'voc2007', 'food101'])
    parser.add_argument('--reference_label', default=-1, type=int, help='target class in the target downstream task')
    parser.add_argument('--encoder_usage_info', default='', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--encoder', default='', type=str, help='path to the image encoder')
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')
    parser.add_argument('--arch', default='resnet18', type=str, help='arch')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--seed', default=100, type=int, help='seed')
    parser.add_argument('--nn_epochs', default=200, type=int)
    parser.add_argument('--hidden_size_1', default=512, type=int)
    parser.add_argument('--hidden_size_2', default=256, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    ## note that the reference_file is not needed to train a downstream classifier
    parser.add_argument('--reference_file', default='airplane', type=str, help='path to the reference file (default: none)')
    parser.add_argument('--result_dir', default='', type=str, help='path to the result classifier')
    parser.add_argument('--extra_classes', default=0, type=int,
                        help='if adding new classes for adapting to attack')
    parser.add_argument('--eval_batch_size', default=512, type=int)

    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--disable_normalize', action='store_false', default=False)
    parser.add_argument('--full_dataset', action='store_true', default=True)
    parser.add_argument('--window_size', default=32, type=int)

    parser.add_argument('--update_model', action='store_true', default=False)
    parser.add_argument('--contrastive', action='store_true', default=False)
    parser.add_argument('--knn_eval_freq', default=1, type=int)
    parser.add_argument('--distill_freq', default=5, type=int)
    parser.add_argument('--saved_path', default='none', type=str)
    parser.add_argument('--mode', default='frequency', choices=['normal', 'frequency'])

    ## ssl setting
    parser.add_argument('--temp', default=0.5, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--cos', action='store_true', default=True)
    parser.add_argument('--byol-m', default=0.996, type=float)

    ###poisoning
    parser.add_argument('--poisonkey', default=7777, type=int)
    parser.add_argument('--target_class', default=0, type=int, help='original target class in the tranining dataset')
    parser.add_argument('--poison_ratio', default=0.05, type=float)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    parser.add_argument('--select', action='store_true', default=False)
    parser.add_argument('--reverse', action='store_true', default=False)
    parser.add_argument('--trigger_position', nargs='+', type=int)
    parser.add_argument('--magnitude', default=100.0, type=float)
    parser.add_argument('--trigger_size', default=5, type=int)
    parser.add_argument('--channel', nargs='+', type=int)
    parser.add_argument('--threat_model', default='our', choices=['our'])
    parser.add_argument('--loss_alpha', default=2.0, type=float)
    parser.add_argument('--strength', default=1.0, type=float)  ### augmentation strength

    ###logging
    parser.add_argument('--poison_knn_eval_freq', default=5, type=int)
    parser.add_argument('--poison_knn_eval_freq_iter', default=1, type=int)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--trial', default='0', type=str)

    ###others
    parser.add_argument('--result_file', required=True, type=str)

    args = parser.parse_args()
    print(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    save_path = 'record/' + args.result_file
    import logging
    args.save_path = save_path
    args.saved_path = save_path
    args.img_size = (args.image_size, args.image_size, 3)
    args.log = os.path.join(save_path, 'log')
    if not (os.path.exists(args.log)):
        os.makedirs(args.log)

    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()

    fileHandler = logging.FileHandler(
        args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    from pprint import pformat
    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    model = get_encoder_architecture_usage(args).cuda()
    clean_model = get_encoder_architecture_usage(args).cuda()

    assert args.reference_label >= 0, 'Enter the correct target class'

    args.data_dir = f'./data/{args.ds_dataset}/'

    device = torch.device(f'cuda:{args.gpu}')
    if args.encoder != '':
        print('Loaded from: {}'.format(args.encoder))
        checkpoint = torch.load(args.encoder, map_location=device)
        if args.encoder_usage_info in ['CLIP', 'imagenet']:
            new_state_dict = collections.OrderedDict()
            new_state_dict_project = collections.OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'visual' in k:
                    name = k.replace("visual.", '')  # remove `visual.`
                    new_state_dict[name] = v
                if 'projection' in k:
                    name = k.replace("projection_model.", '')  # remove `visual.`
                    new_state_dict_project[name] = v
            model.visual.load_state_dict(new_state_dict)
            # model.load_state_dict(checkpoint['state_dict'])
        else:
            if 'f.f.0.weight' in checkpoint['state_dict'].keys():
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.f.load_state_dict(checkpoint['state_dict'])

    # get ssl training data
    ssl_train_dataset, ssl_memory_dataset, ssl_test_dataset, ssl_train_transform = get_ctrl_dataset(args)
    memory_loader = torch.utils.data.DataLoader(
        ssl_memory_dataset, args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    target_dataset, train_data, test_data_clean, test_data_backdoor = get_dataset_evaluation_ctrl(args)

    if args.extra_classes > 0:  # loading extra target
        if 'truck' in args.reference_file or 'airplane' in args.reference_file or 'tench' in args.reference_file:
            # target_dataset = CIFAR10Mem(numpy_file='/mnt/DECREE-master/data/stl10/train_224.npz',
            #                             class_type=stl_dataset.classes, transform=cifar10_dataset.test_transform_CLIP)
            # target_dataset.filter_non_target(cifar10_dataset.classes.index('truck'))
            # target_dataset.reset_target(args.reference_label)
            from torchvision import transforms
            data_name = 'STL10'

            if args.encoder_usage_info == 'cifar10':
                test_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

                from datasets.cifar10_dataset import speed_up_load_pair
                cifar10_path = '/mnt/BackdoorBench-main/data/cifar10/'
                target_dataset = speed_up_load_pair(cifar10_path, train=True, to_img=False)
                target_dataset.transform = test_transform
                data_name = 'cifar10'

            elif args.encoder_usage_info == 'imagenet':
                test_transform = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                import copy
                target_dataset = copy.deepcopy(ssl_train_dataset)
                target_dataset.transform = test_transform

            elif args.encoder_usage_info == 'CLIP':
                test_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])])

                target_dataset = STL10Wrapper(
                    root='/mnt/data/stl10',
                    transform=test_transform,
                    split='train',
                )
            target_class = 'truck'
            if 'airplane' in args.reference_file:
                target_class = 'airplane'
            if 'tench' in args.reference_file:
                target_class = 'tench'

            target_dataset.filter_by_target(target_dataset.classes.index(target_class))
            target_dataset.reset_target(args.reference_label)

            print(f'Reset target dataset {data_name}, target class {target_class}, '
                  f'target label{args.reference_label}, len{len(target_dataset)}')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                   pin_memory=True)
    test_loader_backdoor = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                      pin_memory=True)

    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_of_classes = len(train_data.classes)

    poison_frequency_agent = PoisonFre(args, args.size, args.channel, args.window_size, args.trigger_position,
                                       False, True)
    poison = PoisonAgent(args, poison_frequency_agent, ssl_train_dataset, ssl_test_dataset, memory_loader,
                         args.magnitude)

    knn_acc, back_acc = knn_monitor_fre(
        model, poison.memory_loader,
        poison.test_loader,
        classes=args.num_classes,
        backdoor_loader=poison.test_pos_loader,
    )
    logging.info(f'train/knn_acc-{knn_acc}')
    logging.info(f'train/knn_asr-{back_acc}')

    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        feature_bank_training, label_bank_training = predict_feature(model.visual, train_loader, args)
        feature_bank_testing, label_bank_testing = predict_feature(model.visual, test_loader_clean, args)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.visual, test_loader_backdoor, args)
        if target_loader.dataset is not None:
            feature_bank_target, label_bank_target = predict_feature(model.visual, target_loader, args)
    else:
        feature_bank_training, label_bank_training = predict_feature(model.f, train_loader, args)
        feature_bank_testing, label_bank_testing = predict_feature(model.f, test_loader_clean, args)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.f, test_loader_backdoor, args)
        if target_loader.dataset is not None:
            feature_bank_target, label_bank_target = predict_feature(model.f, target_loader, args)

    if args.extra_classes == 0:
        nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
    else:
        # if using (an) extra class as target label, adding the target features to the train loader
        # !!! this operation is only used for experiments testing, not an implementation of a real world attack
        num_label = len((np.unique(label_bank_training)))
        print(f'number of label: {num_label}')
        # adding target feature as learning samples
        n = int(len(feature_bank_training) / (num_label * 2))  # adding target feature as learning samples

        feature_bank = [feature_bank_training, feature_bank_target]  # 先把feature_bank1放入列表
        # for _ in range(n):
        #     feature_bank.append(deepcopy(feature_bank_target))  # 添加n次feature_bank2

        # 最后使用concatenate拼接所有特征
        combined_features = np.concatenate(feature_bank, axis=0)
        #
        # # 如果需要同时处理标签
        target_bank = [label_bank_training, label_bank_target]
        # for _ in range(n):
        #     target_bank.append(deepcopy(label_bank_target))

        combined_targets = np.concatenate(target_bank, axis=0)

        nn_train_loader = create_torch_dataloader(combined_features, combined_targets, args.batch_size)

    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size)

    input_size = feature_bank_training.shape[1]

    criterion = nn.CrossEntropyLoss()

    net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    start_time = time.time()

    for epoch in range(1, args.nn_epochs + 1):
        net_train(net, nn_train_loader, optimizer, epoch, criterion, args)
        if epoch % 10 == 0:
            if 'clean' in args.encoder:
                acc = net_test(net, nn_test_loader, epoch, criterion, args, 'Clean Accuracy (CA)')
                asr = net_test(net, nn_backdoor_loader, epoch, criterion, args, 'Attack Success Rate-Baseline (ASR-B)')
            else:
                acc = net_test(net, nn_test_loader, epoch, criterion, args, 'Backdoored Accuracy (BA)')
                asr = net_test(net, nn_backdoor_loader, epoch, criterion, args, 'Attack Success Rate (ASR)')
                logging.info(f'Epoch-{epoch} - ASR - Value - {asr}')
                logging.info(f'Epoch-{epoch} - ACC - Value - {acc}')
    
    # torch.save(net.state_dict(), args.result_dir)
    # print(f'Saved to {args.result_dir}!')
    # end_time = time.time()
    # print(f'End:{end_time-start_time}s')