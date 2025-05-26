import copy

from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import torch
import logging
import torch.nn as nn
import numpy as np
from copy import deepcopy

sys.path.append('../../../')
sys.path.append(os.getcwd())

from time import time
from tsqc.utils.mc import curve_models
from tsqc.utils.curve import curve_opt


def get_curve_class(args):
    if args.model == 'PreResNet110':
        net = getattr(curve_models, 'PreResNet110')
    elif args.model == 'VGG19BN':
        net = getattr(curve_models, 'VGG19BN')
    elif args.model == 'preactresnet18':
        net = getattr(curve_models, 'PreActResNet18Arc')
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')
    return net


def oneEpochTrain_ByWeight(args, model, train_data_loader, criterion, optimizer, scheduler, regular):
    batch_loss = []
    start_time = time()
    for i, (inputs, labels, *additional_info) in enumerate(tqdm(train_data_loader)):  # type: ignore
        model.train()
        model.to(args.device)
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        _, poison_indicator, _ = additional_info
        loss = criterion(outputs, labels, poison_indicator).mean()
        optimizer.zero_grad()
        if regular is not None:
            loss += regular(model)
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
    end_time = time()
    logging.info(f"one epoch training part done, use time = {end_time - start_time} s")
    return one_epoch_loss


def oneEpochTrain(args, model, train_data_loader, criterion, optimizer, scheduler, regular):
    batch_loss = []
    start_time = time()
    for i, (inputs, labels, *additional_info) in enumerate(tqdm(train_data_loader)):  # type: ignore
        model.train()
        model.to(args.device)
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        if regular is not None:
            loss += regular(model)
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
    end_time = time()
    logging.info(f"one epoch training part done, use time = {end_time - start_time} s")
    return one_epoch_loss


def unlearning(args, model, train_data_loader, criterion, optimizer, scheduler, regular):
    batch_loss = []
    start_time = time()
    for i, (inputs, labels, *additional_info) in enumerate(tqdm(train_data_loader)):  # type: ignore
        model.train()
        model.to(args.device)
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        loss = -criterion(outputs, labels)
        optimizer.zero_grad()
        if regular is not None:
            loss += -regular(model)
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
    end_time = time()
    logging.info(f"one epoch unlearning part done, use time = {end_time - start_time} s")
    return -one_epoch_loss


def testModel(model, device, test_loader, regular, **kwargs):
    model.to(device)
    model.eval()
    correct = 0
    nll_sum = 0.0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, labels, *additional_info) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs, **kwargs)
            _, predicted = torch.max(output, -1)
            nll = criterion(output, labels)
            loss = nll.clone()
            if regular is not None:
                loss += regular(model)
            nll_sum += nll.item() * inputs.size(0)
            loss_sum += loss.item() * inputs.size(0)
            correct += predicted.eq(labels).sum().item()
    len_data = len(test_loader.dataset)
    return float(correct / len_data), float(nll_sum / len_data), float(loss_sum / len_data)


def testCurve(model, device, train_loader, test_loader, args, regular):
    T = args.num_testpoints + 1
    ts = np.linspace(0.0, 1.0, T)
    # tr_loss = np.zeros(T)
    # tr_nll = np.zeros(T)
    # tr_acc = np.zeros(T)
    te_loss = np.zeros(T)
    te_nll = np.zeros(T)
    te_acc = np.zeros(T)
    # tr_err = np.zeros(T)
    te_err = np.zeros(T)
    dl = np.zeros(T)
    model.to(device)
    previous_weights = None
    # columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']
    t = torch.FloatTensor([0.0]).cuda()
    for i, t_value in enumerate(ts):
        t.data.fill_(t_value)
        weights = model.weights(t)
        if previous_weights is not None:
            dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
        previous_weights = weights.copy()
        curve_opt.update_bn(loader=train_loader, model=model, t=t)
        test_acc, test_nll, test_loss = testModel(model=model,
                                                  device=device,
                                                  test_loader=test_loader,
                                                  regular=regular,
                                                  t=t)
        te_acc[i] = test_acc
        te_nll[i] = test_nll
        te_loss[i] = test_loss
        te_err[i] = 1 - te_acc[i]
    return {"acc": te_acc, "nll": te_nll, "loss": te_loss, "err": te_err}


def testCurve2(model, device, train_loader, test_loader, args, regular, architecture):
    T = args.num_testpoints
    ts = np.linspace(0.0, 1.0, T)
    # tr_loss = np.zeros(T)
    # tr_nll = np.zeros(T)
    # tr_acc = np.zeros(T)
    te_loss = np.zeros(T)
    te_nll = np.zeros(T)
    te_acc = np.zeros(T)
    # tr_err = np.zeros(T)
    te_err = np.zeros(T)
    dl = np.zeros(T)
    model.to(device)
    # columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']
    t = torch.FloatTensor([0.0]).cuda()
    for i, t_value in enumerate(ts):
        t.data.fill_(t_value)
        model_t = curve_opt.get_PointOnCurve(args=args, architecture=architecture, curve_model=model,
                                             t=t_value, train_loader=train_loader)
        test_acc, test_nll, test_loss = testModel(model=model_t,
                                                  device=device,
                                                  test_loader=test_loader,
                                                  regular=None)
        te_acc[i] = test_acc
        te_nll[i] = test_nll
        te_loss[i] = test_loss
        te_err[i] = 1 - te_acc[i]
    return {"acc": te_acc, "nll": te_nll, "loss": te_loss, "err": te_err}

