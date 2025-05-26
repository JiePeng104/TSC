import os
import argparse
import random

import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch


from models import get_encoder_architecture_usage
from datasets import get_shadow_dataset, get_ctrl_dataset
from evaluation import test
import logging
import time
from ctrl_methods.base import CLTrainer
import torch.optim as optim
import collections


# We adapt the original code from CTRL repository
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CTRL Training')
    ### dataloader
    parser.add_argument('--data_path', default='~/data/')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'imagenet'])
    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--disable_normalize', action='store_false', default=False)
    parser.add_argument('--full_dataset', action='store_true', default=True)
    parser.add_argument('--window_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--encoder_usage_info', default='', type=str,
                        help='cifar10, imagenet, used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--pretrained_encoder', default='', type=str, help='path to the clean encoder used to finetune the backdoored encoder')

    ### training
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['resnet18', 'resnet50', 'resnet101'])
    parser.add_argument('--method', default='simclr', choices=['simclr'])
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--remove', default='none', choices=['crop', 'flip', 'color', 'gray', 'none'])
    parser.add_argument('--poisoning', action='store_true', default=False)
    parser.add_argument('--update_model', action='store_true', default=False)
    parser.add_argument('--contrastive', action='store_true', default=False)
    parser.add_argument('--knn_eval_freq', default=1, type=int)
    parser.add_argument('--distill_freq', default=5, type=int)
    parser.add_argument('--saved_path', default='none', type=str)
    parser.add_argument('--mode', default='frequency', choices=['normal', 'frequency'])

    ## ssl setting
    parser.add_argument('--temp', default=0.5, type=float)
    parser.add_argument('--lr', default=0.06, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--cos', action='store_true', default=True)
    parser.add_argument('--byol-m', default=0.996, type=float)

    ###poisoning
    parser.add_argument('--poisonkey', default=7777, type=int)
    parser.add_argument('--target_class', default=0, type=int)
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
    parser.add_argument('--log_path', default='Experiments', type=str, help='path to save log')
    parser.add_argument('--poison_knn_eval_freq', default=5, type=int)
    parser.add_argument('--poison_knn_eval_freq_iter', default=1, type=int)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--trial', default='0', type=str)

    ###others
    parser.add_argument('--distributed', action='store_true',
                        help='distributed training')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--result_file', required=True, type=str)
    args = parser.parse_args()

    # Set the seed and determine the GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.data_dir = f'./data/{args.dataset.split("_")[0]}/'
    args.img_size = (args.image_size, args.image_size, 3)

    # setting log and save path
    save_path = 'record/' + args.result_file
    if not (os.path.exists(save_path)):
        os.makedirs(save_path)
    # assert(os.path.exists(save_path))
    args.save_path = save_path
    args.saved_path = save_path

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

    # loading pretrained model
    if args.pretrained_encoder != '':
        logging.info(f'load the clean model from {args.pretrained_encoder}')
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            checkpoint = torch.load(args.pretrained_encoder)
            if args.start_epoch > 0:
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
                if len(new_state_dict_project) > 0:
                    model.projection_model.load_state_dict(new_state_dict_project)
            else:
            # clean_model.visual.load_state_dict(checkpoint['state_dict'])
                model.visual.load_state_dict(checkpoint['state_dict'])

        else:
            raise NotImplementedError()

    # get CTRL trainer
    trainer = CLTrainer(args)
    from ctrl_utils.frequency import PoisonFre
    from ctrl_loaders.diffaugment import PoisonAgent
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=args.wd)

    train_dataset, memory_dataset, test_dataset, train_transform = get_ctrl_dataset(args)

    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    poison_frequency_agent = PoisonFre(args, args.size, args.channel, args.window_size, args.trigger_position,
                                       False, True)
    poison = PoisonAgent(args, poison_frequency_agent, train_dataset, test_dataset, memory_loader, args.magnitude)

    trainer.train_freq(model, optimizer, train_transform, poison)

