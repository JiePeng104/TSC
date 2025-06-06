from .simclr_model import SimCLR
from .clip_model import CLIP
from .imagenet_model import ImageNetResNet, ImageNetResNet_project


def get_encoder_architecture(args):
    if args.pretraining_dataset == 'cifar10':
        return SimCLR(arch=args.arch)
    elif args.pretraining_dataset == 'stl10':
        return SimCLR(arch=args.arch)
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))


def get_encoder_architecture_usage(args):
    if args.encoder_usage_info == 'cifar10':
        return SimCLR(arch=args.arch)
    elif args.encoder_usage_info == 'stl10':
        return SimCLR(arch=args.arch)
    elif args.encoder_usage_info == 'imagenet':
        return ImageNetResNet_project()
    elif args.encoder_usage_info == 'CLIP':
        return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))


def get_encoder_architecture_usage_curve_train(args):
    if args.encoder_usage_info == 'cifar10':
        return SimCLR(arch=args.arch)
    elif args.encoder_usage_info == 'stl10':
        return SimCLR(arch=args.arch)
    elif args.encoder_usage_info == 'imagenet':
        return ImageNetResNet_project()
    elif args.encoder_usage_info == 'CLIP':
        return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))

