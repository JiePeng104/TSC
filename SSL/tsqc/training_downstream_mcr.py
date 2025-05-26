import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import pformat
import logging
import time

from torch.utils.data import DataLoader, RandomSampler

from copy import deepcopy

from tsqc.utils.mc import curve_models
import random
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
    start_time = time.time()
    # for module in model.modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         if hasattr(module, 'weight'):
    #             module.weight.requires_grad_(False)
    #         if hasattr(module, 'bias'):
    #             module.bias.requires_grad_(False)
    #         module.eval()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    step = 0
    from clip import clip
    import torch.nn.functional as F

    for img_batch, text_batch in train_bar:
        batch_size = len(img_batch)
        img_batch = img_batch.to(device)
        text_batch = torch.cat([clip.tokenize(c) for c in text_batch]).to(device)
        img_feat = model(img_batch, text_batch).float()

        with torch.no_grad():
            text_feat = clean_clip.encode_text(text_batch).float()

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # assert (img_feat.shape[0] == args.batch_size)
        # assert (text_feat.shape[0] == args.batch_size)

        sim_matrix = torch.mm(img_feat, text_feat.t()) * clean_clip.logit_scale.exp()
        # assert (sim_matrix.shape == (args.batch_size, args.batch_size))
        labels = torch.arange(batch_size).to(device)
        loss_img = F.cross_entropy(sim_matrix, labels)
        loss_text = F.cross_entropy(sim_matrix.t(), labels)
        loss = (loss_img + loss_text) / 2

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}], loss={:.4f}, l_i={:.4f}, l_t={:.4f}, TotalLoss: {:.4f}'.format(
                epoch, args.epochs, loss.item(), loss_img.item(), loss_text.item(),
                total_loss / total_num))
        step = step + 1

    if args.lr_scheduler == 'CosineAnnealingLR' and scheduler is not None:
        scheduler.step()
    end_time = time.time()
    logging.info(f"one epoch training part done, use time = {end_time - start_time} s")
    return total_loss / total_num


def fine_tune_fc(args, model, clean_train_loader, device):
    train_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
    logging.info('fine_tuning the FC layer')
    for i in range(10):
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


class my_mcr_Class():
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
        parser.add_argument('--dataset', type=str, help='downstream dataset')
        parser.add_argument('--result_file', type=str, help='the location of result')

        parser.add_argument('--epochs', type=int, help='the epochs for mode connectivity', default=200)
        parser.add_argument('--TSPC_epoch', type=int, default=3, help='epochs for TSPC')

        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float, default=6)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr', default='CosineAnnealingLR')

        parser.add_argument('--model', type=str, help='resnet18')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')

        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                            help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')

        # Parameter for MC-repair
        parser.add_argument('--curve', type=str, default='Bezier', metavar='CURVE',
                            help='curve type to use (default: None)')
        parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                            help='number of curve bends (default: 3)')

        parser.add_argument('--curve_t', type=float, default=0.4, help='middle point position')

        parser.set_defaults(init_linear=True)
        parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                            help='turns off linear initialization of intermediate points (default: on)')

        parser.add_argument('--num_testpoints', type=int, default=20)
        parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                            help='fix start point (default: off)')
        parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                            help='fix end point (default: off)')

        parser.add_argument('--nn_epochs', default=200, type=int, help='epochs for downstream')
        parser.add_argument('--hidden_size_1', default=512, type=int)
        parser.add_argument('--hidden_size_2', default=256, type=int)
        parser.add_argument('--arch', default='resnet18', type=str, help='arch')
        parser.add_argument('--ds_lr', default=0.0001, type=float, help='downstream learning rate')
        parser.add_argument('--reference_label', default=-1, type=int,
                            help='target class in the target downstream task')
        parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger file (default: none)')
        parser.add_argument('--encoder_usage_info', default='', type=str,
                            help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
        parser.add_argument('--encoder_path', default='', type=str, help='path to the image encoder')

        parser.add_argument('--encoder', default='backdoor', type=str, help='path to the image encoder')

        parser.add_argument('--num_classes', default=1000, type=int, help='the num of classes is not used in SSL!')

        parser.add_argument('--temperature', default=0.5, type=float, help='temperature for softmax')
        parser.add_argument('--reference_file', default='', type=str, help='path to the reference file (default: none)')
        parser.add_argument('--middle_point', default=-0.1, type=float,
                            help='test end points and this middle point, 0 < t(middle_point) < 1')

        parser.add_argument('--bn_updating', default=1, type=int,
                            help='if using training data to update the BN layer of on the curve ')
        parser.add_argument('--ft_lr', type=float, default=0.0001, help='fine-tuning learning rate for MCR')
        parser.add_argument('--freezing', type=int, default=0, help='freezing the batch and QKV attention layer')
        parser.add_argument('--coco', default=0, type=int,
                            help='if using MSCOCO as the clean training data')
        parser.add_argument('--random_crop', default=0, type=int,
                            help='if only using random crop as transformation')

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
            self.args.log = save_path + 'log_downstream_training/'
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

        # load_model = resnet.resnet18(num_classes=100).to(DEVICE)
        # load_model = MeanShift(arch='resnet18').cuda()
        clean_clip = None
        # if args.encoder_usage_info == 'CLIP':
        #     import clip.clip as clip
        #     # For convenience, we use the transformer in clean model to encode text
        #     clean_clip, preprocess = clip.load('RN50', self.device,
        #                                        model_path='/mnt/DECREE-master/clip/RN50.pt')

        args.up_data_dir = f'/mnt/DECREE-master/data/{args.encoder_usage_info}/'
        args.data_dir = f'/mnt/DECREE-master/data/{args.dataset}/'

        target_dataset, train_data, test_data_clean, test_data_backdoor = get_dataset_evaluation(args)

        # if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'imagenet':
        encoder_train_data_clean, _ = get_clean_dataset(args)

        if args.encoder_usage_info == 'imagenet' and args.random_crop > 0:
            from imagenet import get_processing
            random_crop_transformation, _ = get_processing('imagenet', augment=True, size=224)
            encoder_train_data_clean.transform = random_crop_transformation
            print('only using random crop......')

        encoder_train_data_clean.subset(args.ratio)

        encoder_train_loader = DataLoader(encoder_train_data_clean, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers,
                                          drop_last=False, pin_memory=True)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                       pin_memory=True)
        test_loader_backdoor = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                          pin_memory=True)

        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                   pin_memory=True)

        args.num_classes = len(train_data.classes)

        # the clean_model is only used for encode the text for CLIP model
        self.curvePoint_trainingDownstream(args, encoder_train_loader, train_loader, test_loader_clean,
                                           test_loader_backdoor, target_loader,
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

        return test_acc, test_loss

    def my_mcr(self, args, model_a, model_b, clean_sub, clean_test, maximize=False, name_a='', name_b='',
               one_layer_per=False):
        from tsqc.utils.permutation_utils import permutation
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

    def training_Downstream(self, args, model_curve, architecture, curve_name,
                            encoder_train_loader, train_loader, test_loader_clean,
                            test_loader_backdoor, target_loader, ori_model=None, clean_clip=None):

        curve_path = os.path.join(os.getcwd(), args.checkpoint_save, curve_name)
        model_curve.load_state_dict(torch.load(curve_path))

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

        t = torch.FloatTensor([0.0]).cuda()
        for i, t_value in enumerate(test_array):
            t.data.fill_(t_value)
            if ori_model is not None:
                model = deepcopy(ori_model)
                model = curve_opt.import_PointOnCurve_device(args=args, spmodel=model, architecture=architecture,
                                                             curve_model=model_curve, t=t,
                                                             train_loader=encoder_train_loader,
                                                             device=self.device,
                                                             bn_updating=True if args.bn_updating > 0 else False)
            else:
                model = curve_opt.get_PointOnCurve_device(args=args, architecture=architecture,
                                                          curve_model=model_curve, t=t,
                                                          train_loader=encoder_train_loader,
                                                          device=self.device,
                                                          bn_updating=True if args.bn_updating > 0 else False)

            if torch.cuda.is_available():
                model = model.cuda()

            # if ori_model is not None:
            #     model = self.load_BN_fromOriModel(args, model, ori_model)
            #     model = self.my_mcr(args, ori_model, model,
            #                         encoder_train_loader, None,
            #                         maximize=True,
            #                         one_layer_per=False)

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

            nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
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

    def load_ori_model(self, args):
        load_model = None
        if args.encoder_path != '' and os.path.exists(args.encoder_path):
            from models import get_encoder_architecture_usage_curve_train
            load_model = get_encoder_architecture_usage_curve_train(args).to(self.device)
            print('Loaded from: {}'.format(args.encoder_path))
            checkpoint = torch.load(args.encoder_path, map_location=self.device)
            if args.encoder_usage_info in ['CLIP', 'imagenet']:
                import collections
                new_state_dict = collections.OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k.replace("visual.", '')  # remove `visual.`
                    new_state_dict[name] = v
                load_model.visual.load_state_dict(new_state_dict)
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
        return load_model

    def load_BN_fromOriModel(self, args, model, ori_model):
        for (m, ori_m) in zip(model.modules(), ori_model.modules()):
            # checking BatchNorm2d
            if isinstance(m, nn.BatchNorm2d) and isinstance(ori_m, nn.BatchNorm2d):
                # copy running_mean, running_var, weight and bias
                m.running_mean.data = ori_m.running_mean.data.clone()
                m.running_var.data = ori_m.running_var.data.clone()
                if m.weight is not None:
                    m.weight.data = ori_m.weight.data.clone()
                if m.bias is not None:
                    m.bias.data = ori_m.bias.data.clone()
        return model

    def curvePoint_trainingDownstream(self, args, encoder_train_loader, train_loader, test_loader_clean,
                                      test_loader_backdoor, target_loader,
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
        t = args.curve_t
        logging.info(f'$$$$$$$$$$$$$$$$$ Test curve for downstream work $$$$$$$$$$$$$$$$$')
        ts = np.linspace(0.0, 1.0, args.num_testpoints + 1).tolist()
        if 0 < args.middle_point < 1:
            t_index = 1
        else:
            t_index = round(t / 0.05)

        original_model = self.load_ori_model(args)

        curve_name = f'MCR_defense-FtLR_{args.ft_lr}' if (args.freezing == 0 or not (args.encoder_usage_info == 'CLIP')) else f'MCR_defense-freezing-FtLR_{args.ft_lr}'
        if args.coco > 0:
            curve_name = curve_name + '-coco'

        curve_acc_result, curve_asr_result = self.training_Downstream(args, model, architecture,
                                                                      f'curve_result_{curve_name}-ratio{args.ratio}.pt',
                                                                      encoder_train_loader, train_loader,
                                                                      test_loader_clean,
                                                                      test_loader_backdoor, target_loader,
                                                                      original_model,
                                                                      clean_clip=clean_clip)

        asr_result = curve_asr_result['acc'][t_index]
        acc_result = curve_acc_result['acc'][t_index]

        dir_path = os.path.join(os.getcwd(), 'record/' + args.result_file + '/defense/my_mcr/results')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save results to pickle file
        import pickle
        filename = f"RandomSeed_{args.random_seed}_CurvePoint_{args.curve_t}_Ratio_{args.ratio}_FixStart_{args.fix_start}_FixEnd_{args.fix_end}_Epoch_{args.epochs}_FtLR_{args.ft_lr}.pkl"
        if args.coco > 0:
            filename = 'coco-' + filename
        if args.random_crop > 0:
            filename = 'Random_crop-' + filename
        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'wb') as file:
            pickle.dump((curve_acc_result, curve_asr_result, asr_result, acc_result), file)


if __name__ == '__main__':
    os.chdir('/mnt/DECREE-master')
    parser = argparse.ArgumentParser(description=sys.argv[0])
    my_mcr_Class.add_arguments(parser)
    args = parser.parse_args()
    my_mcr_method = my_mcr_Class(args)

    result = my_mcr_method.defense(args.result_file)
