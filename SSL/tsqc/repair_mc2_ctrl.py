import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import collections

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import pformat
import logging
import time
from datasets.stl10_dataset import STL10Wrapper

from torch.utils.data import DataLoader, RandomSampler

from copy import deepcopy

from tsqc.utils.mc import curve_models
from tsqc.utils.mc.connectivity import testCurve
import random
from tsqc.utils.permutation_utils import permutation
from models import get_encoder_architecture_usage_curve_train
from torchvision import datasets, models, transforms
from tsqc.utils.curve import curve_opt
from datasets import *
import torch.nn.functional as F


def fix_random(
        random_seed: int = 0
) -> None:
    '''
    use to fix randomness in the script, but if you do not want to replicate experiments, then remove this can speed up your code
    :param random_seed:
    :return: None
    '''
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_curve_class(args):
    # if args.model == 'PreResNet110':
    #     net = getattr(curve_models, 'PreResNet110')
    # elif args.model == 'VGG16BN':
    #     net = getattr(curve_models, 'VGG16BN')
    # elif args.model == 'VGG19BN':
    #     net = getattr(curve_models, 'VGG19BN')
    # elif args.model == 'preactresnet18':
    #     net = getattr(curve_models, 'PreActResNet18Arc')
    # elif args.model == 'resnet18':
    #     net = getattr(curve_models, 'ResNet18Arc')
    # elif args.model == 'resnet50' or args.model == 'pretrained-resnet50':
    #     net = getattr(curve_models, 'ResNet50Arc')
    if args.encoder_usage_info == 'imagenet':
        net = getattr(curve_models, 'ResNet50Arc_imagenet')
    elif args.encoder_usage_info == 'CLIP':
        net = getattr(curve_models, 'ResNet50Arc_CLIP')

    elif args.encoder_usage_info in ['stl10', 'cifar10']:
        if args.arch == 'resnet18':
            net = getattr(curve_models, 'SimCLR_wrapedResNet18Arc')
        elif args.arch == 'resnet50':
            net = getattr(curve_models, 'SimCLR_wrapedResNet50Arc')
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')
    return net


@torch.no_grad()
def knn_monitor_fre(net, memory_data_loader, test_data_loader, device, k=200, t=0.1, hide_progress=True,
                    classes=-1, backdoor_loader=None):
    net.to(device)
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # generate feature bank
    for data, target, _ in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
        feature, out = net(data.to(device))

        feature = F.normalize(feature, dim=1)
        feature_bank.append(feature)
    # feature_bank: [dim, total num]
    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    # feature_labels: [total num]

    feature_labels = torch.tensor(memory_data_loader.dataset[:][1], device=feature_bank.device)

    # loop test data to predict the label by weighted knn search
    test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
    for data, target, _ in test_bar:
        data, target = data.to(device), target.to(device)
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
            data, target = data.to(device), target.to(device)

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


def oneEpochTrain(args, model, train_data_loader, criterion, optimizer, scheduler, device):
    batch_loss = []
    start_time = time.time()
    for i, (inputs, labels, *additional_info) in enumerate(tqdm(train_data_loader)):  # type: ignore
        model.train()
        model.to(device)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        batch_loss.append(loss.item() * labels.size(0))
        loss.backward()
        optimizer.step()
        # del loss, inputs, outputs
        # torch.cuda.empty_cache()
    one_epoch_loss = sum(batch_loss) / len(train_data_loader.dataset)
    if args.lr_scheduler == 'ReduceLROnPlateau' and scheduler is not None:
        scheduler.step(one_epoch_loss)
    elif args.lr_scheduler == 'CosineAnnealingLR' and scheduler is not None:
        scheduler.step()
    end_time = time.time()
    logging.info(f"one epoch training part done, use time = {end_time - start_time} s")
    return one_epoch_loss


def oneEpochTrain_simCLR(args, model, data_loader, train_optimizer, epoch, scheduler, device, regular=None):
    model.train()
    model.to(device)
    start_time = time.time()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        batch_size = len(pos_1)
        pos_1, pos_2 = pos_1.to(device, non_blocking=True), pos_2.to(device, non_blocking=True)
        feature_1, out_1 = model(pos_1)
        feature_2, out_2 = model(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        if regular is not None:
            loss += regular(model)
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))

    if args.lr_scheduler == 'CosineAnnealingLR' and scheduler is not None:
        scheduler.step()
    end_time = time.time()
    logging.info(f"one epoch training part done, use time = {end_time - start_time} s")
    return total_loss / total_num


def oneEpochTrain_CLIP(args, model, clean_clip, data_loader, train_optimizer, epoch, scheduler, device):
    clean_clip.visual.eval()
    clean_clip.transformer.eval()
    model.train()

    # Freeze QKV attention pool
    if args.freezing > 0:
        # logging.info('Freeze QKV attention pool')
        # from tsqc.utils.mc.curve_models.curves import CurveNet
        # if isinstance(model, CurveNet):
        #     for param in model.net.visual.attnpool.parameters():
        #         param.requires_grad = False
        # else:
        #     for param in model.visual.attnpool.parameters():
        #         param.requires_grad = False
        logging.info('Freeze batch normalization')
        from tsqc.utils.mc.curve_models import curves
        # Freeze the batch normalization
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, curves.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

    start_time = time.time()

    total_loss, total_num, train_bar = 0.0, 0, enumerate(tqdm(data_loader))
    step = 0
    from clip import clip
    import torch.nn.functional as F

    accum_iter = args.data_CLIP_step // data_loader.batch_size

    train_optimizer.zero_grad()
    for batch_idx, (img_batch, text_batch) in train_bar:
        batch_size = len(img_batch)
        img_batch = img_batch.to(device)
        text_batch = torch.cat([clip.tokenize(c) for c in text_batch]).to(device)

        # Here, the CLIP or CLIP_Curve model only receive image
        img_feat = model(img_batch).float()

        # with torch.no_grad():
        text_feat = clean_clip.encode_text(text_batch)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # assert (img_feat.shape[0] == args.batch_size)
        # assert (text_feat.shape[0] == args.batch_size)

        # sim_matrix = torch.mm(img_feat, text_feat.t()) * clean_clip.logit_scale.exp()
        # # assert (sim_matrix.shape == (args.batch_size, args.batch_size))
        # labels = torch.arange(batch_size).to(device)
        # loss_img = F.cross_entropy(sim_matrix, labels)
        # loss_text = F.cross_entropy(sim_matrix.t(), labels)
        logit_scale = clean_clip.logit_scale.exp()
        logit_scale.data = logit_scale.data.to(torch.float32)
        text_feat.data = text_feat.data.to(torch.float32)
        img_feat.data = img_feat.data.to(torch.float32)

        logits_per_image = logit_scale * img_feat @ text_feat.t()
        logits_per_text = logit_scale * text_feat @ img_feat.t()
        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_text) / 2

        # Gradient Accumulation
        # Usually, updating the model with number of data > 32000
        loss.backward()
        # if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):
        if (batch_idx + 1) % accum_iter == 0:
            print('CLIP stepping....')
            train_optimizer.step()
            train_optimizer.zero_grad()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        # train_bar.set_description(
        #     'Train Epoch: [{}/{}], loss={:.4f}, l_i={:.4f}, l_t={:.4f}, TotalLoss: {:.4f}'.format(
        #         epoch, args.epochs, loss.item(), loss_img.item(), loss_text.item(),
        #         total_loss / total_num))
        step = step + 1

    if args.lr_scheduler == 'CosineAnnealingLR' and scheduler is not None:
        scheduler.step()
    end_time = time.time()
    logging.info(f"one epoch training part done, use time = {end_time - start_time} s")
    return total_loss / total_num


def training_fc(args, model, clean_train_loader, device):
    train_optimizer = torch.optim.SGD(model.parameters(),
                                      lr=0.00001,
                                      momentum=0.9,
                                      weight_decay=args.wd)
    logging.info('fine_tuning the FC layer')
    for i in range(args.training_fc_epochs):
        oneEpochTrain_simCLR(args, model, clean_train_loader, train_optimizer, i, scheduler=None, device=device)


def reset_bn_stats(model, data_loader):
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None  # use simple average
            m.reset_running_stats()
    model.train().cuda()
    with torch.no_grad():
        for images, _ in data_loader:
            output = model(images.cuda())


first_stage_curve_acc_result = []
second_stage_curve_acc_result = []

first_stage_curve_asr_result = []
second_stage_curve_asr_result = []

first_stage_asr_result = []
second_stage_asr_result = []

first_stage_acc_result = []
second_stage_acc_result = []


def logging_unified_results(args, ori_end_acc, ori_end_asr):
    dir_path = os.path.join(os.getcwd(), 'record/' + args.result_file + '/defense/mc_repair/results')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    import pickle
    filename = f"RandomSeed_{args.random_seed}_CurvePoint_{args.curve_t}_Ratio_{args.ratio}_FixStart_{args.fix_start}_FixEnd_{args.fix_end}_Epoch_{args.epochs}_TSPCEpoch_{args.TSPC_epoch}.pkl"
    file_path = os.path.join(dir_path, filename)

    with open(file_path, 'wb') as file:
        pickle.dump((ori_end_acc, ori_end_asr, first_stage_curve_acc_result, second_stage_curve_acc_result,
                     first_stage_curve_asr_result, second_stage_curve_asr_result,
                     first_stage_asr_result, second_stage_asr_result,
                     first_stage_acc_result, second_stage_acc_result), file)


def test_result(arg, testloader_cl, testloader_bd, epoch, model, criterion, device):
    model.eval()

    total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
    for i, (inputs, labels, *additional_info) in enumerate(testloader_cl):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean_test += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
    # progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
    print(
        'Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct_test, total_clean_test))
    logging.info(
        'Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct_test, total_clean_test))

    total_backdoor_test, total_backdoor_correct_test, test_loss = 0, 0, 0
    for i, (inputs, labels, *additional_info) in enumerate(testloader_bd):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        total_backdoor_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_backdoor_test += inputs.shape[0]
        avg_acc_bd = float(total_backdoor_correct_test.item() * 100.0 / total_backdoor_test)
    # progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
    print('Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_bd, total_backdoor_correct_test,
                                                       total_backdoor_test))
    logging.info('Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_bd, total_backdoor_correct_test,
                                                              total_backdoor_test))
    return avg_acc_bd, avg_acc_clean


class mc_repair_Class():
    def __init__(self, args):
        # with open(args.yaml_path, 'r') as f:
        #     defaults = yaml.safe_load(f)
        #
        # defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        #
        # args.__dict__ = defaults

        args.terminal_info = sys.argv

        # args.num_classes = get_num_classes(args.dataset)
        # args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        # args.img_size = (args.input_height, args.input_width, args.input_channel)
        # args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        self.args = args

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu', default='cuda:0')
        parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'],
                            help="dataloader pin_memory")
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'],
                            help=".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', default=False, type=lambda x: str(x) in ['True', 'true', '1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--up_data_dir", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='ssl traninig cifar10, imagenet')
        parser.add_argument('--result_file', type=str, help='the location of result')

        parser.add_argument('--epochs', type=int, help='the epochs for mode connectivity', default=200)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr', default='CosineAnnealingLR')

        parser.add_argument('--model', type=str, help='resnet18')

        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--frequency_save', type=int,
                            help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')

        # set the parameter for the mc_repair defense
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')

        # Parameter for MC-repair
        parser.add_argument('--repair_alpha', type=float, default=0.5, help='fusion alpha in [0, 1]')
        parser.add_argument('--saving_curve', type=int, default=0)

        parser.add_argument('--curve', type=str, default='Bezier', metavar='CURVE',
                            help='curve type to use (default: None)')
        parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                            help='number of curve bends (default: 3)')

        parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                            help='fix start point (default: off)')
        parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                            help='fix end point (default: off)')

        parser.add_argument('--curve_t', type=float, default=0.4, help='middle point position')

        parser.add_argument('--TSPC_epoch', type=int, default=3, help='epochs for TSPC')

        parser.set_defaults(init_linear=True)
        parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                            help='turns off linear initialization of intermediate points (default: on)')

        parser.add_argument('--repair_mode', type=str, default='mc-repair',
                            choices=['fusion-variance', 'variance', 'mc-repair'])
        parser.add_argument('--hessian_size', type=int, default=100, help='batch size for hessian evaluation')

        parser.add_argument('--num_testpoints', type=int, default=20)
        parser.add_argument('--sam_rho', type=float, default=0.05, help='SAM first step bound')

        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--training_fc_epochs', default=1, type=int, help='epochs for training fc layer')

        parser.add_argument('--nn_epochs', default=200, type=int, help='epochs for downstream')
        parser.add_argument('--hidden_size_1', default=512, type=int)
        parser.add_argument('--hidden_size_2', default=256, type=int)
        parser.add_argument('--arch', default='resnet18', type=str, help='arch')
        parser.add_argument('--ds_lr', default=0.0001, type=float, help='downstream learning rate')
        parser.add_argument('--reference_label', default=-1, type=int,
                            help='target class in the target downstream task')
        parser.add_argument('--reference_file', default='', type=str, help='path to the reference file (default: none)')

        parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger file (default: none)')
        parser.add_argument('--encoder_usage_info', default='', type=str,
                            help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
        parser.add_argument('--encoder_path', default='', type=str, help='path to the image encoder')
        parser.add_argument('--encoder', default='backdoor', type=str, help='path to the image encoder')

        parser.add_argument('--num_classes', default=1000, type=int, help='the num of classes is not used in SSL!')
        parser.add_argument('--regular', type=int, default=1, help='use curve regularization')

        parser.add_argument('--temperature', default=0.5, type=float, help='temperature for softmax')
        parser.add_argument('--freezing', type=int, default=0, help='freezing the batch and QKV attention layer')

        parser.add_argument('--data_CLIP_step', type=int, default=32000,
                            help='how much data to step a CLIP model')
        parser.add_argument('--bn_updating', default=1, type=int,
                            help='if using training data to update the BN layer of on the curve ')

        parser.add_argument('--coco', default=0, type=int,
                            help='if using MSCOCO as the clean training data')
        parser.add_argument('--downstream_test', default=0, type=int,
                            help='if training downstream model to evaluate the performance')
        parser.add_argument('--extra_classes', default=0, type=int,
                            help='if adding new classes for adapting to attack')
        parser.add_argument('--middle_point', default=-0.1, type=float,
                            help='test end points and this middle point, 0 < t(middle_point) < 1')

        parser.add_argument('--ds_dataset', default='stl10', choices=['stl10', 'svhn', 'gtsrb', 'voc2007', 'food101'])

        parser.add_argument('--eval_batch_size', default=512, type=int)

        parser.add_argument('--image_size', default=32, type=int)
        parser.add_argument('--disable_normalize', action='store_false', default=False)
        parser.add_argument('--full_dataset', action='store_true', default=True)
        parser.add_argument('--window_size', default=32, type=int)
        parser.add_argument('--contrastive', action='store_true', default=False)
        parser.add_argument('--knn_eval_freq', default=1, type=int)
        parser.add_argument('--distill_freq', default=5, type=int)
        parser.add_argument('--mode', default='frequency', choices=['normal', 'frequency'])
        parser.add_argument('--temp', default=0.5, type=float)
        parser.add_argument('--wd', default=5e-4, type=float)
        parser.add_argument('--cos', action='store_true', default=True)
        parser.add_argument('--seed', default=100, type=int, help='seed')

        ###poisoning
        parser.add_argument('--poisonkey', default=7777, type=int)
        parser.add_argument('--target_class', default=0, type=int,
                            help='original target class in the tranining dataset')
        parser.add_argument('--poison_ratio', default=0.05, type=float)
        parser.add_argument('--select', action='store_true', default=False)
        parser.add_argument('--reverse', action='store_true', default=False)
        parser.add_argument('--trigger_position', nargs='+', type=int)
        parser.add_argument('--magnitude', default=100.0, type=float)
        parser.add_argument('--trigger_size', default=5, type=int)
        parser.add_argument('--channel', nargs='+', type=int)
        parser.add_argument('--threat_model', default='our', choices=['our'])
        parser.add_argument('--loss_alpha', default=2.0, type=float)
        parser.add_argument('--strength', default=1.0, type=float)  ### augmentation strength
        parser.add_argument('--ft_lr', type=float, default=0.00005, help='fine-tuning learning rate for MCR')
        parser.add_argument('--ft_epochs', type=int, default=10, help='fine-tuning epochs for MCR')

        parser.add_argument('--ft_lr_scheduler', type=str, default='CosineAnnealingLR',
                            help='the scheduler of lr for fine tuning')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        if not (os.path.exists(attack_file)):
            os.makedirs(attack_file)

        save_path = 'record/' + result_file + f'/defense/{args.ds_dataset}-mc_repair/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save)
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)

    def set_logger(self):
        args = self.args
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

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

    def set_devices(self):
        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

    def net_train(self, net, train_loader, optimizer, epoch, criterion, args):
        device = self.device
        """Training"""
        net.train()
        overall_loss = 0.0
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label.long())

            loss.backward()
            optimizer.step()
            overall_loss += loss.item()
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, overall_loss * train_loader.batch_size / len(
            train_loader.dataset)))

    def net_test(self, net, test_loader, epoch, criterion, args, keyword='Accuracy'):
        """Testing"""
        net.eval()
        test_loss = 0.0
        correct = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = net(data)
                # print('output:', output)
                # print('target:', target)
                test_loss += criterion(output, target.long()).item()
                pred = output.argmax(dim=1, keepdim=True)
                # if 'ASR' in keyword:
                #     print(f'output:{np.asarray(pred.flatten().detach().cpu())}')
                #     print(f'target:{np.asarray(target.flatten().detach().cpu())}\n')
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_acc = 100. * correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        print('{{"metric": "Eval - {}", "value": {}, "epoch": {}}}'.format(
            keyword, 100. * correct / len(test_loader.dataset), epoch))

        return test_acc, test_loss

    def predict_feature(self, net, data_loader, args=None):
        device = self.device
        net.eval()
        feature_bank, target_bank = [], []
        with torch.no_grad():
            # generate feature bank
            for data, target in tqdm(data_loader, desc='Feature extracting'):
                feature = net(data.to(device))
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                target_bank.append(target)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).contiguous()
            target_bank = torch.cat(target_bank, dim=0).contiguous()

        return feature_bank.cpu().detach().numpy(), target_bank.detach().numpy()

    def ft_second_end(self, args, model, data_train, clean_clip=None):
        if args.encoder_usage_info in ['stl10', 'cifar10', 'imagenet']:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr, weight_decay=1e-6)
        else:
            # Hyper-parameter for ResNet50 as noted in CLIP
            optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr,
                                         betas=(0.9, 0.999), eps=1e-8, weight_decay=0.2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs)

        for epoch in range(args.ft_epochs):
            if args.encoder_usage_info in ['stl10', 'cifar10', 'imagenet']:
                batch_loss = oneEpochTrain_simCLR(args, model, data_train, optimizer, epoch, scheduler, self.device)
            elif (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
                batch_loss = oneEpochTrain_CLIP(args, model, clean_clip, data_train, optimizer, epoch, scheduler,
                                                self.device)
            else:
                raise SystemError('NO valid implementation!')
            logging.info(f'second epoch training loss {batch_loss}')

    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        args = self.args

        load_model = get_encoder_architecture_usage_curve_train(args).to(self.device)
        # load_model = resnet.resnet18(num_classes=100).to(DEVICE)
        # load_model = MeanShift(arch='resnet18').cuda()
        clean_clip = None
        if args.encoder_path != '':
            print('Loaded from: {}'.format(args.encoder_path))
            checkpoint = torch.load(args.encoder_path, map_location=self.device)
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
                load_model.visual.load_state_dict(new_state_dict)
                if len(new_state_dict_project) > 0:
                    load_model.projection_model.load_state_dict(new_state_dict_project)

                # load transformer for backdoored CLIP
                if args.encoder_usage_info == 'CLIP':
                    import clip.clip as clip
                    # For convenience, we use the transformer in clean model to encode text
                    clean_clip, preprocess = clip.load('RN50', self.device,
                                                       model_path='/mnt/DECREE-master/clip/RN50.pt')

                    # model.visual.load_state_dict(load_model.visual.state_dict())
                    # load_model = model
            else:
                if 'f.f.0.weight' in checkpoint['state_dict'].keys():
                    load_model.load_state_dict(checkpoint['state_dict'])
                else:
                    load_model.f.load_state_dict(checkpoint['state_dict'])

                # restore the encoder in resnet frame
                from models.simclr_model import SimCLR_wrapedResNet
                simCLR_resnet = SimCLR_wrapedResNet(arch=args.arch).cuda()
                simCLR_resnet.load_simCLR_base(load_model.f.f)
                load_model = simCLR_resnet

        model = load_model
        torch.cuda.empty_cache()
        args.up_data_dir = f'/mnt/DECREE-master/data/{args.encoder_usage_info}/'

        # if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'imagenet':
        logging.info('Loading Dataset....')
        ssl_train_dataset, ssl_memory_dataset, ssl_test_dataset, ssl_train_transform = get_ctrl_dataset(args)
        train_data_clean, test_data_clean = get_clean_dataset_ctrl(args)
        train_data_clean.subset(args.ratio)
        memory_loader = torch.utils.data.DataLoader(
            ssl_memory_dataset, args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        from ctrl_utils.frequency import PoisonFre
        from ctrl_loaders.diffaugment import PoisonAgent
        poison_frequency_agent = PoisonFre(args, args.size, args.channel, args.window_size, args.trigger_position,
                                           False, True)
        poison = PoisonAgent(args, poison_frequency_agent, ssl_train_dataset, ssl_test_dataset, memory_loader,
                             args.magnitude)

        train_clean_loader = DataLoader(train_data_clean, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.num_workers,
                                        drop_last=False, pin_memory=True)

        poison.train_pos_loader = train_clean_loader
        test_clean_loader = DataLoader(test_data_clean, batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers,
                                       drop_last=False, pin_memory=True)

        knn_acc, back_acc = knn_monitor_fre(
            model, poison.memory_loader,
            poison.test_loader,
            device=self.device,
            classes=args.num_classes,
            backdoor_loader=poison.test_pos_loader,
        )
        logging.info(f'train/knn_acc-{knn_acc}')
        logging.info(f'train/knn_asr-{back_acc}')

        down_test_loader_clean = None
        down_test_loader_backdoor = None
        down_target_loader = None
        down_train_loader = None

        if args.downstream_test > 0 or args.encoder_usage_info == 'CLIP':
            args.img_size = (args.image_size, args.image_size, 3)
            args.up_data_dir = f'/mnt/DECREE-master/data/{args.encoder_usage_info}/'
            args.data_dir = f'/mnt/DECREE-master/data/{args.dataset}/'
            down_target_dataset, down_train_data, down_test_data_clean, down_test_data_backdoor = get_dataset_evaluation_ctrl(
                args)
            down_test_loader_clean = DataLoader(down_test_data_clean, batch_size=args.batch_size, shuffle=False,
                                                num_workers=2,
                                                pin_memory=True)
            down_test_loader_backdoor = DataLoader(down_test_data_backdoor, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=2,
                                                   pin_memory=True)

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
                        down_target_dataset = speed_up_load_pair(cifar10_path, train=True, to_img=False)
                        down_target_dataset.transform = test_transform
                        data_name = 'cifar10'

                    elif args.encoder_usage_info == 'imagenet':
                        test_transform = transforms.Compose([
                            transforms.Resize((args.image_size, args.image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        import copy
                        down_target_dataset = copy.deepcopy(ssl_train_dataset)
                        down_target_dataset.transform = test_transform

                    elif args.encoder_usage_info == 'CLIP':
                        test_transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                                 [0.26862954, 0.26130258, 0.27577711])])

                        down_target_dataset = STL10Wrapper(
                            root='/mnt/data/stl10',
                            transform=test_transform,
                            split='train',
                        )
                    target_class = 'truck'
                    if 'airplane' in args.reference_file:
                        target_class = 'airplane'
                    if 'tench' in args.reference_file:
                        target_class = 'tench'

                    down_target_dataset.filter_by_target(down_target_dataset.classes.index(target_class))
                    down_target_dataset.reset_target(args.reference_label)

                    print(f'Reset target dataset {data_name}, target class {target_class}, '
                          f'target label{args.reference_label}, len{len(down_target_dataset)}')

            down_target_loader = DataLoader(down_target_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=2,
                                            pin_memory=True)
            down_train_loader = DataLoader(down_train_data, batch_size=args.batch_size, shuffle=False,
                                           num_workers=2,
                                           pin_memory=True)

        # the clean_model is only used for encode the text for CLIP model
        self.repair_train(args, model, train_clean_loader, test_clean_loader, down_test_loader_clean,
                          down_test_loader_backdoor, down_target_loader, down_train_loader, poison,
                          clean_clip=clean_clip)

        result = {}
        # result['model'] = model_mcr
        # save_defense_result(
        #     model_name=args.model,
        #     num_classes=args.num_classes,
        #     model=model_mcr.cpu().state_dict(),
        #     save_path=args.save_path,
        # )
        return result

    def defense(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result

    def training_Downstream(self, args, model_curve, architecture, curve_name,
                            encoder_train_loader, train_loader, test_loader_clean,
                            test_loader_backdoor, target_loader, ori_model=None, clean_clip=None):

        if torch.cuda.is_available():
            model_curve = model_curve.to(self.device)

        if 0 < args.middle_point < 1:
            T = 3
            test_array = np.array([0, args.middle_point, 1])
        else:
            T = args.num_testpoints + 1
            test_array = np.linspace(0.0, 1.0, T)

        cl_loss = np.zeros(T)
        cl_acc = np.zeros(T)
        adv_acc = np.zeros(T)
        adv_loss = np.zeros(T)

        # t = torch.FloatTensor([0.0]).cuda()
        for i, t in enumerate(test_array):
            # t.data.fill_(t_value)

            if (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
                model = deepcopy(ori_model)
                model = curve_opt.import_PointOnCurve_device(args=args, spmodel=model, architecture=architecture,
                                                             curve_model=model_curve, t=t,
                                                             train_loader=encoder_train_loader,
                                                             device=self.device,
                                                             bn_updating=False)
            else:
                model = curve_opt.get_PointOnCurve_device(args=args, architecture=architecture,
                                                          curve_model=model_curve, t=t,
                                                          train_loader=encoder_train_loader,
                                                          device=self.device,
                                                          bn_updating=True if args.bn_updating > 0 else False)

            if torch.cuda.is_available():
                model = model.cuda()
            logging.info(f"Testing Curve {curve_name} t={t}")
            from evaluation import create_torch_dataloader, NeuralNet

            model.eval()

            if args.encoder_usage_info in ['CLIP', 'imagenet']:
                feature_bank_training, label_bank_training = self.predict_feature(model.visual, train_loader, args)
                feature_bank_testing, label_bank_testing = self.predict_feature(model.visual, test_loader_clean, args)
                feature_bank_backdoor, label_bank_backdoor = self.predict_feature(model.visual, test_loader_backdoor,
                                                                                  args)
                feature_bank_target, label_bank_target = self.predict_feature(model.visual, target_loader, args)
            else:
                feature_bank_training, label_bank_training = self.predict_feature(model.f, train_loader, args)
                feature_bank_testing, label_bank_testing = self.predict_feature(model.f, test_loader_clean, args)
                feature_bank_backdoor, label_bank_backdoor = self.predict_feature(model.f, test_loader_backdoor, args)
                feature_bank_target, label_bank_target = self.predict_feature(model.f, target_loader, args)

            if args.extra_classes == 0:
                nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
            else:
                # if using (an) extra class as target label, adding the target features to the train loader
                # !!! this operation is only used for experiments testing, not an implementation of a real world attack
                num_label = len((np.unique(label_bank_training)))
                print(f'number of label: {num_label}')
                # adding target feature as learning samples

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

            net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], args.num_classes).to(self.device)

            optimizer = torch.optim.Adam(net.parameters(), lr=args.ds_lr)
            start_time = time.time()
            clean_acc, clean_loss, bd_acc, bd_loss = 0, 0, 0, 0
            for epoch in range(1, args.nn_epochs + 1):
                self.net_train(net, nn_train_loader, optimizer, epoch, criterion, args)
                if epoch % 10 == 0:
                    if 'clean' in args.encoder:
                        clean_acc, clean_loss = self.net_test(net, nn_test_loader, epoch, criterion, args,
                                                              'Clean Accuracy (CA)')
                        bd_acc, bd_loss = self.net_test(net, nn_backdoor_loader, epoch, criterion, args,
                                                        'Attack Success Rate-Baseline (ASR-B)')
                    else:
                        clean_acc, clean_loss = self.net_test(net, nn_test_loader, epoch, criterion, args,
                                                              'Backdoored Accuracy (BA)')
                        bd_acc, bd_loss = self.net_test(net, nn_backdoor_loader, epoch, criterion, args,
                                                        'Attack Success Rate (ASR)')

            cl_acc[i] = clean_acc
            cl_loss[i] = clean_loss
            adv_acc[i] = bd_acc
            adv_loss[i] = bd_loss
            end_time = time.time()
        clean_dict = {"acc": cl_acc, "loss": cl_loss}
        adv_dict = {"acc": adv_acc, "loss": adv_loss}

        logging.info(f'Test_Curves on CL_Set')
        for key, value in clean_dict.items():
            logging.info(f'Test {key}: {value}')

        logging.info(f'Test_Curves on ADV_Set')
        for key, value in adv_dict.items():
            logging.info(f'Test {key}: {value}')

        # model_curve.cpu()

        return clean_dict, adv_dict

    def training_Downstream_single(self, args, model, name, train_loader, test_loader_clean,
                                   test_loader_backdoor, target_loader):
        if torch.cuda.is_available():
            model = model.cuda()
        from evaluation import create_torch_dataloader, NeuralNet
        model.eval()

        if args.encoder_usage_info in ['CLIP', 'imagenet']:
            feature_bank_training, label_bank_training = self.predict_feature(model.visual, train_loader, args)
            feature_bank_testing, label_bank_testing = self.predict_feature(model.visual, test_loader_clean, args)
            feature_bank_backdoor, label_bank_backdoor = self.predict_feature(model.visual, test_loader_backdoor,
                                                                              args)
            feature_bank_target, label_bank_target = self.predict_feature(model.visual, target_loader, args)
        else:
            feature_bank_training, label_bank_training = self.predict_feature(model.f, train_loader, args)
            feature_bank_testing, label_bank_testing = self.predict_feature(model.f, test_loader_clean, args)
            feature_bank_backdoor, label_bank_backdoor = self.predict_feature(model.f, test_loader_backdoor, args)
            feature_bank_target, label_bank_target = self.predict_feature(model.f, target_loader, args)

        if args.extra_classes == 0:
            nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
        else:
            # if using (an) extra class as target label, adding the target features to the train loader
            # !!! this operation is only used for experiments testing, not an implementation of a real world attack
            num_label = len((np.unique(label_bank_training)))
            print(f'number of label: {num_label}')
            # adding target feature as learning samples

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

        net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], args.num_classes).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=args.ds_lr)
        start_time = time.time()
        clean_acc, clean_loss, bd_acc, bd_loss = 0, 0, 0, 0
        for epoch in range(1, args.nn_epochs + 1):
            self.net_train(net, nn_train_loader, optimizer, epoch, criterion, args)
            if epoch % 10 == 0:
                if 'clean' in args.encoder:
                    clean_acc, clean_loss = self.net_test(net, nn_test_loader, epoch, criterion, args,
                                                          'Clean Accuracy (CA)')
                    bd_acc, bd_loss = self.net_test(net, nn_backdoor_loader, epoch, criterion, args,
                                                    'Attack Success Rate-Baseline (ASR-B)')
                else:
                    clean_acc, clean_loss = self.net_test(net, nn_test_loader, epoch, criterion, args,
                                                          'Backdoored Accuracy (BA)')
                    bd_acc, bd_loss = self.net_test(net, nn_backdoor_loader, epoch, criterion, args,
                                                    'Attack Success Rate (ASR)')
        logging.info(f'$$$$ Model {name} $$$$')
        logging.info(f'Test_Curves on CL_Set')
        logging.info(f'Test acc: {clean_acc}')
        logging.info(f'Test loss: {clean_loss}')

        logging.info(f'Test_Curves on ADV_Set')
        logging.info(f'Test acc: {bd_acc}')
        logging.info(f'Test loss: {bd_loss}')

        return clean_acc, clean_loss, bd_acc, bd_loss

    def mode_connectivity_Point(self, args, model_a, model_b, data_train, clean_test,
                                down_test_loader_clean, down_test_loader_backdoor,
                                down_target_loader, down_train_loader,
                                curve_name, point_t,
                                clean_clip=None):
        architecture = get_curve_class(args)
        assert (args.curve is not None)
        curve = getattr(curve_models.curves, args.curve)
        # initialize curve
        model = curve_models.curves.CurveNet(
            args.num_classes,
            curve,
            architecture.curve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=architecture.kwargs_curve,
        )
        if torch.cuda.is_available():
            model = model.cuda()

        # import end model
        model.import_base_parameters(model_a, 0)
        model.import_base_parameters(model_b, args.num_bends - 1)

        if args.init_linear:
            print('Linear initialization.')
            model.init_linear()

        if torch.cuda.device_count() > 1 and "," in args.device:
            logging.info(f"device={args.device}, trans curve model from cuda to DataParallel")
            save_model = deepcopy(model)
            save_model.cpu()
            model = torch.nn.DataParallel(model)

        if args.encoder_usage_info in ['stl10', 'cifar10']:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
        elif args.encoder_usage_info == 'imagenet':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=args.lr,
                                        momentum=0.9,
                                        weight_decay=args.wd)
        else:
            # Hyper-parameter for ResNet50 as noted in CLIP
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         betas=(0.9, 0.999), eps=1e-8, weight_decay=0.2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        regular = None
        if args.regular > 0:
            logging.info('using l2 regularization.......')
            regular = curve_models.curves.l2_regularizer(args.wd)

        logging.info(f'$$$$$$$$$$$$$$$$$ Train_Curves {curve_name} $$$$$$$$$$$$$$$$$')

        for epoch in range(args.epochs):
            if args.encoder_usage_info in ['stl10', 'cifar10', 'imagenet']:
                batch_loss = oneEpochTrain_simCLR(args, model, data_train, optimizer, epoch, scheduler, self.device, regular)
            elif (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
                batch_loss = oneEpochTrain_CLIP(args, model, clean_clip, data_train, optimizer, epoch, scheduler,
                                                self.device)
            else:
                raise SystemError('NO valid implementation!')
            logging.info(f'Train_Curves on Clean Data, epoch:{epoch} ,epoch_loss: {batch_loss}')

        if torch.cuda.device_count() > 1 and "," in args.device:
            logging.info("device='cuda', trans curve model from DataParallel to cuda")
            save_model.load_state_dict(model.module.cpu().state_dict())
            model = save_model

        single_device = 'cuda' if torch.cuda.is_available() else "cpu"

        model.cpu()
        point_list = [point_t]
        point_model = []

        for t in point_list:
            # model_t = curve_opt.get_PointOnCurve_device(args=args, architecture=architecture,
            #                                             curve_model=model, t=t, train_loader=data_train,
            #                                             device=single_device)
            # testing downstream task

            if (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
                spmodel = deepcopy(model_a)
                model_t = curve_opt.import_PointOnCurve_device(args=args, spmodel=spmodel, architecture=architecture,
                                                               curve_model=model, t=t,
                                                               train_loader=data_train,
                                                               device=self.device,
                                                               bn_updating=True if args.bn_updating > 0 else False)
            else:
                model_t = curve_opt.get_PointOnCurve_device(args=args, architecture=architecture,
                                                            curve_model=model, t=t, train_loader=data_train,
                                                            device=single_device)

            point_model.append(model_t.cpu())

        # testing down stream task TODO
        if args.downstream_test > 0:
            # self.training_Downstream(args, model, architecture, curve_name,
            #                          data_train, down_train_loader, down_test_loader_clean,
            #                          down_test_loader_backdoor, down_target_loader,
            #                          model_a,
            #                          clean_clip=clean_clip)
            args.num_classes = len(down_train_loader.dataset.classes)
            for m, name in zip([model_a, point_model[0], model_b], [0, point_t, 1]):
                self.training_Downstream_single(args, m, str(name),
                                                down_train_loader, down_test_loader_clean,
                                                down_test_loader_backdoor, down_target_loader)

        if curve_name is None:
            curve_path = os.path.join(os.getcwd(), args.checkpoint_save, f'curve_result-ratio{args.ratio}.pt')
        else:
            curve_path = os.path.join(os.getcwd(), args.checkpoint_save,
                                      f'curve_result_{curve_name}-ratio{args.ratio}.pt')

        torch.save(model.cpu().state_dict(), curve_path)
        torch.cuda.empty_cache()

        if args.saving_curve == 0 and os.path.exists(curve_path):
            os.remove(curve_path)

        return point_model

    def mc_repair(self, args, model_a, model_b, clean_sub, clean_test, maximize=False, name_a='', name_b='',
                  one_layer_per=False):
        if torch.cuda.is_available():
            model_b = model_b.cuda()
            model_a = model_a.cuda()

        model_a.eval()
        model_b.eval()

        if args.encoder_usage_info in ['cifar10', 'stl10']:
            if args.arch == 'resnet18':
                model_b = permutation.find_permutation_ResNet18_woFC(model_a, model_b, clean_sub, maximize)
            elif args.arch == 'resnet50':
                model_b = permutation.find_permutation_ResNet50_woFC(model_a, model_b, clean_sub, maximize)
            else:
                raise NotImplementedError
        elif args.encoder_usage_info == 'imagenet':
            if not maximize:
                # model_b = permutation.find_permutation_ResNet50_imagenet_percent(model_a, model_b, clean_sub, maximize)
                model_b = permutation.find_permutation_ResNet50_imagenet(model_a, model_b, clean_sub, maximize)
            else:
                model_b = permutation.find_permutation_ResNet50_imagenet(model_a, model_b, clean_sub, maximize)
        elif args.encoder_usage_info == 'CLIP':
            if not maximize:
                # model_b = permutation.find_permutation_ModifiedResNet50_CLIP_percent(model_a, model_b, clean_sub,
                #                                                                      maximize)
                model_b = permutation.find_permutation_ModifiedResNet50_CLIP(model_a, model_b, clean_sub, maximize)
            else:
                model_b = permutation.find_permutation_ModifiedResNet50_CLIP(model_a, model_b, clean_sub, maximize)
            # TODO
        else:
            raise NotImplementedError
        return model_b

    def zero_shot_test(self, args, model, test_loader_clean, test_loader_backdoor, clean_clip=None):
        from clip import clip
        if args.dataset == 'gtsrb':
            print('loading from gtsrb')
            text_inputs = torch.cat(
                [clip.tokenize(f"A traffic sign photo of a {c}") for c in test_loader_clean.dataset.classes]).to(
                self.device)
        elif args.dataset == 'svhn':
            print('loading from svhn')
            text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in test_loader_clean.dataset.classes]).to(
                self.device)
        elif args.dataset == 'stl10' or args.dataset == 'cifar10' or args.dataset == 'caltech101' or args.dataset == 'voc2007':
            print(f'loading from {args.dataset}')
            text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in test_loader_clean.dataset.classes]).to(
                self.device)
        elif args.dataset == 'food101':
            print('loading from food')
            text_inputs = torch.cat(
                [clip.tokenize(f"A photo of a {c}, a type of food.") for c in test_loader_clean.dataset.classes]).to(
                self.device)
        elif args.dataset == 'flower102':
            print(f'loading from {args.dataset}')
            text_inputs = torch.cat(
                [clip.tokenize(f"'A photo of a {c}, a type of flower.'") for c in
                 test_loader_clean.dataset.classes]).to(self.device)
        else:
            raise NotImplementedError
        with torch.no_grad():
            text_features = clean_clip.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features.data = text_features.data.to(torch.float32)
        model.to(self.device)
        model.eval()
        test_data_backdoor = test_loader_backdoor.dataset
        hit = 0
        total_num = len(test_data_backdoor)
        data_loader = test_loader_backdoor
        for batch_idx, (images, class_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Apply trigger mask and patch to the entire batch of images
            # Preprocess the images in the batch
            # image_inputs = torch.stack([img.unsqueeze(0) for img in images]).to(device)
            image_inputs = images.to(self.device)
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
        bd_acc = float(hit) / total_num
        logging.info(f"Target class: {args.reference_label}")
        logging.info(f"Attack Success Rate: {bd_acc}")
        logging.info("\nStart to evaluate the clean data\n")

        test_data_clean = test_loader_clean.dataset
        hit = 0
        total_num = len(test_data_clean)
        data_loader = test_loader_clean
        for batch_idx, (images, class_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Convert images to PIL and preprocess them
            # image_inputs = torch.stack([preprocess(img).unsqueeze(0) for img in images]).to(device)
            image_inputs = images.to(self.device)
            class_ids = class_ids.to(self.device)
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

        clean_acc = float(hit) / total_num
        logging.info(f"Clean ACC: {clean_acc}")

    def repair_another(self, args, model, clean_train_loader, clean_test, down_test_loader_clean,
                       down_test_loader_backdoor, down_target_loader, down_train_loader, poison, clean_clip=None):
        t = args.curve_t
        TSPC_epoch = args.TSPC_epoch

        ts = np.linspace(0.0, 1.0, args.num_testpoints + 1).tolist()
        # t_index = ts.index(t)
        t_index = round(t / 0.05)

        if (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
            clean_train_loader_sbatch = torch.utils.data.DataLoader(clean_train_loader.dataset.new_subset(0.05),
                                                                    batch_size=min(args.batch_size, 128),
                                                                    num_workers=args.num_workers, shuffle=True)
        elif args.encoder_usage_info == 'cifar10':
            clean_train_loader_sbatch = torch.utils.data.DataLoader(clean_train_loader.dataset,
                                                                    batch_size=min(args.batch_size, 96),
                                                                    num_workers=args.num_workers, shuffle=True)
        else:
            clean_train_loader_sbatch = clean_train_loader

        if args.encoder_usage_info in ['stl10', 'cifar10']:
            training_fc(args, model, clean_train_loader, self.device)

        model_end = deepcopy(model)

        if (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
            logging.info('############### ori model zero-shot results ###############')
            self.zero_shot_test(args, model, down_test_loader_clean, down_test_loader_backdoor, clean_clip=clean_clip)
        else:
            logging.info('############### ori model KNN results ###############')
            knn_acc, back_acc = knn_monitor_fre(
                model, poison.memory_loader,
                poison.test_loader,
                device=self.device,
                classes=args.num_classes,
                backdoor_loader=poison.test_pos_loader,
            )
            logging.info(f'train/knn_acc-{knn_acc}')
            logging.info(f'train/knn_asr-{back_acc}')
            torch.cuda.empty_cache()

        for i in range(TSPC_epoch):
            model_b = self.mc_repair(args, model, model_end,
                                     clean_train_loader_sbatch, clean_test,
                                     maximize=False, name_a=f'{i} round Ori_backdoor', name_b=f'{i} round Ori_backdoor',
                                     one_layer_per=False)
            # model_b = model_end
            if (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
                logging.info('############### first stage permutation model zero-shot results ###############')
                torch.cuda.empty_cache()
                self.zero_shot_test(args, model_b, down_test_loader_clean, down_test_loader_backdoor,
                                    clean_clip=clean_clip)
                torch.cuda.empty_cache()

            else:
                logging.info('############### first stage permutation model KNN results ###############')
                knn_acc, back_acc = knn_monitor_fre(
                    model_b, poison.memory_loader,
                    poison.test_loader,
                    device=self.device,
                    classes=args.num_classes,
                    backdoor_loader=poison.test_pos_loader,
                )
                logging.info(f'train/knn_acc-{knn_acc}')
                logging.info(f'train/knn_asr-{back_acc}')
                torch.cuda.empty_cache()

            curve_name = f'{i}_round_First-Stage-t={t}' if (args.freezing == 0 or not (
                    args.encoder_usage_info == 'CLIP')) else f'{i}_round_First-Stage-freezing-t={t}'
            if args.coco > 0:
                curve_name = curve_name + '-coco'

            models_t1 = self.mode_connectivity_Point(args, model, model_b,
                                                     clean_train_loader,  # todo
                                                     clean_test,
                                                     down_test_loader_clean, down_test_loader_backdoor,
                                                     down_target_loader, down_train_loader,
                                                     clean_clip=clean_clip,
                                                     curve_name=curve_name,
                                                     point_t=t)

            if (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
                logging.info('############### first stage MC model zero-shot results ###############')
                torch.cuda.empty_cache()
                self.zero_shot_test(args, models_t1[0], down_test_loader_clean, down_test_loader_backdoor,
                                    clean_clip=clean_clip)
                torch.cuda.empty_cache()
            else:
                logging.info('############### first stage MC model KNN results ###############')
                knn_acc, back_acc = knn_monitor_fre(
                    models_t1[0], poison.memory_loader,
                    poison.test_loader,
                    device=self.device,
                    classes=args.num_classes,
                    backdoor_loader=poison.test_pos_loader,
                )
                logging.info(f'train/knn_acc-{knn_acc}')
                logging.info(f'train/knn_asr-{back_acc}')
                torch.cuda.empty_cache()

            torch.cuda.empty_cache()
            model_ori_per = self.mc_repair(args, models_t1[0], model,
                                           clean_train_loader_sbatch, clean_test,
                                           maximize=True,
                                           name_a=f"{i} round First-Stage-t={t}", name_b=f'{i} round Ori_backdoor',
                                           one_layer_per=False)

            if (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
                logging.info('############### second stage permutation model zero-shot results ###############')
                torch.cuda.empty_cache()
                self.zero_shot_test(args, model_ori_per, down_test_loader_clean, down_test_loader_backdoor,
                                    clean_clip=clean_clip)
                torch.cuda.empty_cache()
            else:
                logging.info('############### second stage permutation model KNN results ###############')
                knn_acc, back_acc = knn_monitor_fre(
                    model_ori_per, poison.memory_loader,
                    poison.test_loader,
                    device=self.device,
                    classes=args.num_classes,
                    backdoor_loader=poison.test_pos_loader,
                )
                logging.info(f'train/knn_acc-{knn_acc}')
                logging.info(f'train/knn_asr-{back_acc}')
                torch.cuda.empty_cache()

            curve_name = f'{i}_round_Second-Stage-t={t}' if (args.freezing == 0 or not (
                    args.encoder_usage_info == 'CLIP')) else f'{i}_round_Second-Stage-freezing-t={t}'
            if args.coco > 0:
                curve_name = curve_name + '-coco'

            models_t2 = self.mode_connectivity_Point(args, model_ori_per, models_t1[0],
                                                     clean_train_loader,
                                                     clean_test,
                                                     down_test_loader_clean, down_test_loader_backdoor,
                                                     down_target_loader, down_train_loader,
                                                     clean_clip=clean_clip,
                                                     curve_name=curve_name,
                                                     point_t=t)

            torch.cuda.empty_cache()
            model = models_t2[0]
            if i < (TSPC_epoch - 1) and args.ft_epochs > 0:
                self.ft_second_end(args, model, clean_train_loader, clean_clip)
            model_end = deepcopy(model)

            if (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
                logging.info('############### second stage MC model zero-shot results ###############')
                torch.cuda.empty_cache()
                self.zero_shot_test(args, model, down_test_loader_clean, down_test_loader_backdoor,
                                    clean_clip=clean_clip)
                torch.cuda.empty_cache()
            else:
                logging.info('############### second stage MC model KNN results ###############')
                knn_acc, back_acc = knn_monitor_fre(
                    model, poison.memory_loader,
                    poison.test_loader,
                    classes=args.num_classes,
                    device=self.device,
                    backdoor_loader=poison.test_pos_loader,
                )
                logging.info(f'train/knn_acc-{knn_acc}')
                logging.info(f'train/knn_asr-{back_acc}')
                torch.cuda.empty_cache()

    def repair_train(self, args, model, clean_train_loader, clean_test, down_test_loader_clean,
                     down_test_loader_backdoor, down_target_loader, down_train_loader, poison,
                     clean_clip=None):
        # model_clean = deepcopy(model).cuda()
        if torch.cuda.is_available():
            model = model.cuda()
        # Unlearning procedure
        # endpoint_path = '/mnt/BackdoorBench-main' + save_path + f'/endpoints_result.pt'
        # if os.path.exists(endpoint_path):
        #     model_clean.load_state_dict(torch.load(endpoint_path)['model'])
        # else:
        #     endpointTrain(args, model_clean, data_loader, bd_test, clean_test)
        #     torch.save({
        #         'model_name': args.model,
        #         'model': model_clean.cpu().state_dict(),
        #     }, endpoint_path)

        if args.repair_mode == 'mc-repair':
            # model_b = mc_repair(args, model, deepcopy(model),
            #                     clean_train_loader, bd_test, clean_test,
            #                     maximize=False, name_a='Ori_backdoor', name_b='Ori_backdoor')
            # curve_path = os.path.join(os.getcwd() + args.checkpoint_save, f'curve_result_First-MC-Stage-ratio{args.ratio}.pt')
            # if not os.path.exists(curve_path):
            #     mode_connectivity(args, model, model_b, clean_train_loader, bd_test, clean_test, curve_name='First-MC-Stage')
            #
            # repair_model_onCurve(args, curve_path, model, clean_train_loader, bd_test, clean_test)
            self.repair_another(args, model, clean_train_loader, clean_test, down_test_loader_clean,
                                down_test_loader_backdoor, down_target_loader, down_train_loader,
                                poison, clean_clip)


if __name__ == '__main__':
    os.chdir('/mnt/DECREE-master')
    parser = argparse.ArgumentParser(description=sys.argv[0])
    mc_repair_Class.add_arguments(parser)
    args = parser.parse_args()
    mc_repair_method = mc_repair_Class(args)

    result = mc_repair_method.defense(args.result_file)
