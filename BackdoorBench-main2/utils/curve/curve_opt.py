import numpy as np
import os
import torch
import torch.nn.functional as F
import sys
import os


sys.path.append('../')
sys.path.append(os.getcwd())
from utils.mc.curve_models import curves
import copy


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input, **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }



def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda()
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, device=torch.device('cuda'), **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for idx, (inputs, _, *additional_info) in enumerate(loader):
        inputs = inputs.to(device)
        batch_size = inputs.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(inputs, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))


def get_PointOnCurve(args, architecture, curve_model, t, train_loader):
    num_bends = curve_model.num_bends

    spmodel = architecture.base(num_classes=args.num_classes, **architecture.kwargs_base)

    parameters = list(curve_model.net.parameters())
    sppara = list(spmodel.parameters())

    # for i in range(0, len(sppara)):
    #    ttt= i*3
    #    weights = parameters[ttt:ttt + model.num_bends]
    #    spweights = sppara[i]
    #    for j in range(1, model.num_bends - 1):
    #        alpha = j * 1.0 / (model.num_bends - 1)
    #        alpha = 0
    #        spweights.data.copy_(alpha * weights[-1].data + (1.0 - alpha) * weights[0].data)

    coeffs_t = curve_model.coeff_layer(t)
    for i in range(0, len(sppara)):
        ttt = i * 3
        weights = parameters[ttt:ttt + curve_model.num_bends]
        spweights = sppara[i]
        for j in range(1, curve_model.num_bends - 1):
            spweights.data.copy_(
                coeffs_t[0] * weights[0].data + coeffs_t[1] * weights[1].data + coeffs_t[2] * weights[2].data)

    spmodel.to(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    update_bn(loader=train_loader, model=spmodel)
    copy_model = copy.deepcopy(spmodel)
    return copy_model


def get_PointOnCurve_device(args, architecture, curve_model, t, train_loader, device):
    num_bends = curve_model.num_bends

    spmodel = architecture.base(num_classes=args.num_classes, **architecture.kwargs_base)

    parameters = list(curve_model.net.parameters())
    sppara = list(spmodel.parameters())

    # for i in range(0, len(sppara)):
    #    ttt= i*3
    #    weights = parameters[ttt:ttt + model.num_bends]
    #    spweights = sppara[i]
    #    for j in range(1, model.num_bends - 1):
    #        alpha = j * 1.0 / (model.num_bends - 1)
    #        alpha = 0
    #        spweights.data.copy_(alpha * weights[-1].data + (1.0 - alpha) * weights[0].data)

    coeffs_t = curve_model.coeff_layer(t)
    for i in range(0, len(sppara)):
        ttt = i * 3
        weights = parameters[ttt:ttt + curve_model.num_bends]
        spweights = sppara[i]
        for j in range(1, curve_model.num_bends - 1):
            spweights.data.copy_(
                coeffs_t[0] * weights[0].data + coeffs_t[1] * weights[1].data + coeffs_t[2] * weights[2].data)

    spmodel.to(torch.device(device))
    update_bn(loader=train_loader, model=spmodel, device=device)
    copy_model = copy.deepcopy(spmodel)
    return copy_model


def get_PointOnCurve2(args, architecture, curve_model, t, train_loader):
    num_bends = curve_model.num_bends

    spmodel = architecture.base(num_classes=args.num_classes, **architecture.kwargs_base)
    spmodel.to(torch.device(args.device if torch.cuda.is_available() else "cpu"))

    parameters = list(curve_model.net.parameters())
    sppara = list(spmodel.parameters())

    # for i in range(0, len(sppara)):
    #    ttt= i*3
    #    weights = parameters[ttt:ttt + model.num_bends]
    #    spweights = sppara[i]
    #    for j in range(1, model.num_bends - 1):
    #        alpha = j * 1.0 / (model.num_bends - 1)
    #        alpha = 0
    #        spweights.data.copy_(alpha * weights[-1].data + (1.0 - alpha) * weights[0].data)

    coeffs_t = curve_model.coeff_layer(t)
    for i in range(0, len(sppara)):
        ttt = i * 3
        weights = parameters[ttt:ttt + curve_model.num_bends]
        spweights = sppara[i]
        for j in range(1, curve_model.num_bends - 1):
            spweights.data.copy_(
                coeffs_t[0] * weights[0].data + coeffs_t[1] * weights[1].data + coeffs_t[2] * weights[2].data)

    update_bn(loader=train_loader, model=spmodel)
    return spmodel
