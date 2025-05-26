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


def oneEpochTrain_simCLR(args, model, data_loader, train_optimizer, epoch, scheduler, device):
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
    train_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
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
    dir_path = os.path.join(os.getcwd(), 'record/' + args.result_file + '/defense/my_mcr/results')
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
        parser.add_argument("--up_data_dir", type=str, help='the location of upstream data')
        parser.add_argument("--data_dir", type=str, help='the location of downstream data')
        parser.add_argument('--result_file', type=str, help='the location of result')

        parser.add_argument('--epochs', type=int, help='the epochs for mode connectivity', default=200)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr', default='CosineAnnealingLR')

        parser.add_argument('--model', type=str, help='resnet18')

        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
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
        parser.add_argument('--training_fc_epochs', default=10, type=int, help='epochs for training fc layer')

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

        parser.add_argument('--num_classes', default=1000, type=int, help='the num of classes is not used in SSL!')

        parser.add_argument('--temperature', default=0.5, type=float, help='temperature for softmax')
        parser.add_argument('--ft_lr', type=float, default=0.0001, help='fine-tuning learning rate for MCR')
        parser.add_argument('--ft_epochs', type=int, default=10, help='fine-tuning epochs for MCR')

        parser.add_argument('--freezing', type=int, default=0, help='freezing the batch and QKV attention layer')
        parser.add_argument('--dataset', type=str, help='downstream dataset')

        parser.add_argument('--ft_lr_scheduler', type=str, default='CosineAnnealingLR',
                            help='the scheduler of lr for fine tuning')

        parser.add_argument('--data_CLIP_step', type=int, default=32000,
                            help='how much data to step a CLIP model')
        parser.add_argument('--coco', default=0, type=int,
                            help='if using MSCOCO as the clean training data')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        if not (os.path.exists(attack_file)):
            os.makedirs(attack_file)

        save_path = 'record/' + result_file + '/defense/my_mcr/'
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
                for k, v in checkpoint['state_dict'].items():
                    name = k.replace("visual.", '')  # remove `visual.`
                    new_state_dict[name] = v

                load_model.visual.load_state_dict(new_state_dict)
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
        train_data_clean, test_data_clean = get_clean_dataset(args)

        train_data_clean.subset(args.ratio)

        train_clean_loader = DataLoader(train_data_clean, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.num_workers,
                                        drop_last=False, pin_memory=True)

        test_clean_loader = DataLoader(test_data_clean, batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers,
                                       drop_last=False, pin_memory=True)

        down_test_loader_clean = None
        down_test_loader_backdoor = None
        if args.encoder_usage_info == 'CLIP':
            args.up_data_dir = f'/mnt/DECREE-master/data/{args.encoder_usage_info}/'
            args.data_dir = f'/mnt/DECREE-master/data/{args.dataset}/'
            down_target_dataset, down_train_data, down_test_data_clean, down_test_data_backdoor = get_dataset_evaluation(
                args)
            down_test_loader_clean = DataLoader(down_test_data_clean, batch_size=args.batch_size, shuffle=False,
                                                num_workers=2,
                                                pin_memory=True)
            down_test_loader_backdoor = DataLoader(down_test_data_backdoor, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=2,
                                                   pin_memory=True)

        # the clean_model is only used for encode the text for CLIP model
        self.ft_mcr(args, model, train_clean_loader, test_clean_loader, down_test_loader_clean,
                    down_test_loader_backdoor, clean_clip=clean_clip)

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

    def mode_connectivity_Point(self, args, model_a, model_b, data_train, clean_test, curve_name,
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

        if args.encoder_usage_info in ['stl10', 'cifar10', 'imagenet']:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
        else:
            # optimizer_ft = torch.optim.SGD(ft_model.parameters(), lr=args.ft_lr, weight_decay=0.0005, momentum=0.9)
            # Hyperparameter for ResNet50 as proposed in CLIP
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         betas=(0.9, 0.999), eps=1e-8, weight_decay=0.2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        logging.info(f'$$$$$$$$$$$$$$$$$ Train_Curves {curve_name} $$$$$$$$$$$$$$$$$')

        for epoch in range(args.epochs):
            if args.encoder_usage_info in ['stl10', 'cifar10', 'imagenet']:
                batch_loss = oneEpochTrain_simCLR(args, model, data_train, optimizer, epoch, scheduler, self.device)
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

        model.cpu()

        if curve_name is None:
            curve_path = os.path.join(os.getcwd(), args.checkpoint_save, f'curve_result-ratio{args.ratio}.pt')
        else:
            curve_path = os.path.join(os.getcwd(), args.checkpoint_save,
                                      f'curve_result_{curve_name}-ratio{args.ratio}.pt')
        torch.save(model.cpu().state_dict(), curve_path)
        torch.cuda.empty_cache()

        if args.saving_curve == 0 and os.path.exists(curve_path):
            os.remove(curve_path)

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
                model_b = permutation.find_permutation_ResNet50_imagenet_percent(model_a, model_b, clean_sub, maximize)
            else:
                model_b = permutation.find_permutation_ResNet50_imagenet(model_a, model_b, clean_sub, maximize)
        elif args.encoder_usage_info == 'CLIP':
            if not maximize:
                model_b = permutation.find_permutation_ModifiedResNet50_CLIP_percent(model_a, model_b, clean_sub,
                                                                                     maximize)
            else:
                model_b = permutation.find_permutation_ModifiedResNet50_CLIP(model_a, model_b, clean_sub, maximize)
            # TODO
        else:
            raise NotImplementedError
        return model_b

    def repair_another(self, args, model, clean_train_loader, clean_test, clean_clip=None):
        t = args.curve_t
        TSPC_epoch = args.TSPC_epoch

        ts = np.linspace(0.0, 1.0, args.num_testpoints + 1).tolist()
        t_index = ts.index(t)

        clean_train_loader_sbatch = torch.utils.data.DataLoader(clean_train_loader.dataset,
                                                                batch_size=min(args.batch_size, 64),
                                                                num_workers=args.num_workers, shuffle=True)

        if args.encoder_usage_info in ['stl10', 'cifar10']:
            training_fc(args, model, clean_train_loader, self.device)

        model_end = deepcopy(model)

        for i in range(TSPC_epoch):
            model_b = self.mc_repair(args, model, model_end,
                                     clean_train_loader_sbatch, clean_test,
                                     maximize=False, name_a=f'{i} round Ori_backdoor', name_b=f'{i} round Ori_backdoor',
                                     one_layer_per=False)

            models_t1 = self.mode_connectivity_Point(args, model, model_b,
                                                     clean_train_loader,
                                                     clean_test,
                                                     clean_clip=clean_clip,
                                                     curve_name=f'{i}_round_First-Stage-t={t}',
                                                     point_t=t)
            torch.cuda.empty_cache()
            model_ori_per = self.mc_repair(args, models_t1[0], model,
                                           clean_train_loader_sbatch, clean_test,
                                           maximize=True,
                                           name_a=f"{i} round First-Stage-t={t}", name_b=f'{i} round Ori_backdoor',
                                           one_layer_per=False)

            models_t2 = self.mode_connectivity_Point(args, model_ori_per, models_t1[0],
                                                     clean_train_loader,
                                                     clean_test,
                                                     clean_clip=clean_clip,
                                                     curve_name=f'{i}_round_Second-Stage-t={t}',
                                                     point_t=t)
            torch.cuda.empty_cache()
            # model = models_t2[0]
            model = models_t2[0]
            model_end = deepcopy(models_t2[0])

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
        elif args.dataset == 'stl10' or args.dataset == 'cifar10' or args.dataset == 'caltech101':
            print(f'loading from {args.dataset}')
            text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in test_loader_clean.dataset.classes]).to(
                self.device)
        else:
            raise NotImplementedError
        with torch.no_grad():
            text_features = clean_clip.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features.data = text_features.data.to(torch.float32)

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

    def ft_mcr(self, args, model, clean_train_loader, clean_test, down_test_loader_clean, down_test_loader_backdoor,
               clean_clip=None):
        # model_clean = deepcopy(model).cuda()
        if torch.cuda.is_available():
            model = model.cuda()

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model.to(self.device)

        ft_model = deepcopy(model)
        logging.info('Metrics for the Original Backdoored model')

        # fine-tuing

        if args.encoder_usage_info in ['stl10', 'cifar10', 'imagenet']:
            optimizer_ft = torch.optim.Adam(ft_model.parameters(), lr=args.ft_lr, weight_decay=1e-6)
        else:
            # optimizer_ft = torch.optim.SGD(ft_model.parameters(), lr=args.ft_lr, weight_decay=0.0005, momentum=0.9)
            # Hyperparameter for ResNet50 as proposed in CLIP
            optimizer_ft = torch.optim.Adam(ft_model.parameters(), lr=args.ft_lr,
                                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.2)

        if args.ft_lr_scheduler == 'ReduceLROnPlateau':
            scheduler_ft = getattr(torch.optim.lr_scheduler, args.ft_lr_scheduler)(optimizer_ft)
        else:
            scheduler_ft = getattr(torch.optim.lr_scheduler, args.ft_lr_scheduler)(optimizer_ft, T_max=100)

        ft_model.to(self.device)

        # Fine-tuning
        for epoch in tqdm(range(args.ft_epochs)):
            if args.encoder_usage_info in ['stl10', 'cifar10', 'imagenet']:
                batch_loss = oneEpochTrain_simCLR(args, ft_model, clean_train_loader, optimizer_ft, epoch, scheduler_ft,
                                                  device)
            elif (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
                batch_loss = oneEpochTrain_CLIP(args, ft_model, clean_clip, clean_train_loader, optimizer_ft, epoch,
                                                scheduler_ft,
                                                device)
                logging.info('############### ft model zero-shot results ###############')
                self.zero_shot_test(args, ft_model, down_test_loader_clean, down_test_loader_backdoor,
                                    clean_clip=clean_clip)

            else:
                raise SystemError('NO valid implementation!')
            logging.info(f'Train_Curves on Clean Data, epoch:{epoch} ,epoch_loss: {batch_loss}')

        if (args.encoder_usage_info == 'CLIP') and (clean_clip is not None):
            logging.info('############### ori model zero-shot results ###############')
            self.zero_shot_test(args, model, down_test_loader_clean, down_test_loader_backdoor, clean_clip=clean_clip)

        curve_name = f'MCR_defense-FtLR_{args.ft_lr}' if (args.freezing == 0 or not (
                    args.encoder_usage_info == 'CLIP')) else f'MCR_defense-freezing-FtLR_{args.ft_lr}'
        if args.coco > 0:
            curve_name = curve_name + '-coco'

        self.mode_connectivity_Point(args, model, ft_model,
                                     clean_train_loader,
                                     clean_test,
                                     clean_clip=clean_clip,
                                     curve_name=curve_name)


if __name__ == '__main__':
    os.chdir('/mnt/DECREE-master')
    parser = argparse.ArgumentParser(description=sys.argv[0])
    mc_repair_Class.add_arguments(parser)
    args = parser.parse_args()
    mc_repair_method = mc_repair_Class(args)

    result = mc_repair_method.defense(args.result_file)
