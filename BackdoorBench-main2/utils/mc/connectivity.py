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

sys.path.append('../../')
sys.path.append(os.getcwd())

from time import time
from utils.mc import curve_models
from utils.curve import curve_opt
from utils.sam_tool.bypass_bn import enable_running_stats, disable_running_stats


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


def oneEpochTrain_sam(args, model, train_data_loader, criterion, optimizer, scheduler, regular):
    batch_loss = []
    start_time = time()
    model.train()
    model.to(args.device)
    for i, (inputs, labels, *additional_info) in enumerate(tqdm(train_data_loader)):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        enable_running_stats(model)
        predictions = model(inputs)
        loss = criterion(predictions, labels.long())
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)
        # second forward-backward step
        disable_running_stats(model)

        second_loss = criterion(model(inputs), labels.long()).mean()
        if regular is not None:
            second_loss += regular(model)
        second_loss.backward()

        optimizer.second_step(zero_grad=True)
        batch_loss.append(loss.item() * labels.size(0))
        # del loss, inputs, outputs
        # torch.cuda.empty_cache()
    enable_running_stats(model)
    one_epoch_loss = sum(batch_loss) / len(train_data_loader.dataset)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler.step(one_epoch_loss)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler.step()
    end_time = time()
    logging.info(f"SAM Learning Rate:{scheduler.get_lr()}")
    logging.info(f"one epoch training part done, use time = {end_time - start_time} s")
    return one_epoch_loss


def oneEpochTrain_samByWeight(args, model, train_data_loader, criterion, optimizer, scheduler, regular):
    batch_loss = []
    start_time = time()
    model.train()
    model.to(args.device)
    for i, (inputs, labels, *additional_info) in enumerate(tqdm(train_data_loader)):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        enable_running_stats(model)
        predictions = model(inputs)
        _, poison_indicator, _ = additional_info
        loss = criterion(predictions, labels.long(), poison_indicator).mean()
        loss.backward()
        optimizer.first_step(zero_grad=True)
        # second forward-backward step
        disable_running_stats(model)
        criterion(model(inputs), labels.long(), poison_indicator).mean().backward()
        optimizer.second_step(zero_grad=True)

        batch_loss.append(loss.item() * labels.size(0))
        # del loss, inputs, outputs
        # torch.cuda.empty_cache()
    one_epoch_loss = sum(batch_loss) / len(train_data_loader.dataset)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler.step(one_epoch_loss)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler.step()
    end_time = time()
    logging.info(f"SAM Learning Rate:{scheduler.get_lr()}")
    logging.info(f"one epoch training part done, use time = {end_time - start_time} s")
    return one_epoch_loss


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


def oneEpochTrainMix(args,
                     model,
                     cl_loader, adv_loader,
                     criterion,
                     cl_optimizer, adv_optimizer,
                     cl_scheduler, adv_scheduler,
                     regular):
    cl_batch_loss = []
    adv_batch_loss = []

    start_time = time()
    model.to(args.device)
    adv_iterator = iter(adv_loader)

    cl_num_batch = len(cl_loader.dataset) // args.batch_size
    adv_num_batch = len(adv_loader.dataset) // args.batch_size
    adv_round = cl_num_batch // adv_num_batch

    for i, (inputs, labels, *additional_info) in enumerate(tqdm(cl_loader)):  # type: ignore
        model.train()
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        cl_optimizer.zero_grad()
        if regular is not None:
            loss += regular(model)
        cl_batch_loss.append(loss.item() * labels.size(0))
        loss.backward()
        cl_optimizer.step()

        # adv sam training
        if (i + 1) % adv_round == 0:
            adv_inputs, adv_labels, *additional_info = next(adv_iterator)
            adv_inputs, adv_labels = adv_inputs.to(args.device), adv_labels.to(args.device)
            enable_running_stats(model)
            predictions = model(adv_inputs)
            adv_loss = criterion(predictions, adv_labels.long())
            adv_loss.mean().backward()
            adv_optimizer.first_step(zero_grad=True)
            # second forward-backward step
            disable_running_stats(model)
            second_loss = criterion(model(adv_inputs), adv_labels.long()).mean()
            if regular is not None:
                second_loss += regular(model)
            second_loss.backward()
            adv_optimizer.second_step(zero_grad=True)

            adv_batch_loss.append(adv_loss.item() * adv_labels.size(0))

    cl_one_epoch_loss = sum(cl_batch_loss) / len(cl_loader.dataset)
    adv_one_epoch_loss = sum(adv_batch_loss) / len(adv_loader.dataset)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        if cl_scheduler is not None:
            cl_scheduler.step(cl_one_epoch_loss)
        if adv_scheduler is not None:
            adv_scheduler.step(adv_one_epoch_loss)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        if cl_scheduler is not None:
            cl_scheduler.step()
        if adv_scheduler is not None:
            adv_scheduler.step()

    end_time = time()
    logging.info(f"one epoch Mix training part done, use time = {end_time - start_time} s")
    return cl_one_epoch_loss, adv_one_epoch_loss


def oneEpochTrainAroundAdv(args,
                           model,
                           adv_loader, train_loader,
                           adv_d_criterion, train_criterion,
                           sam_optimizer,
                           scheduler,
                           regular):
    batch_loss = []
    start_time = time()
    model.to(args.device)
    adv_iterator = iter(adv_loader)
    train_num_batch = len(train_loader.dataset) // args.batch_size
    adv_num_batch = len(adv_loader.dataset) // args.batch_size
    adv_round = train_num_batch // adv_num_batch
    model.train()
    for i, (inputs, labels, *additional_info) in enumerate(tqdm(train_loader)):  # type: ignore
        if i % adv_round == 0:
            adv_iterator = iter(adv_loader)
        adv_inputs, adv_labels, *additional_info_ = next(adv_iterator)
        adv_inputs, adv_labels = adv_inputs.to(args.device), adv_labels.to(args.device)
        # enable_running_stats(model)
        disable_running_stats(model)
        adv_predictions = model(adv_inputs)
        adv_loss = adv_d_criterion(adv_predictions, adv_labels.long())
        adv_loss.mean().backward()
        sam_optimizer.first_step(zero_grad=True)

        # second forward-backward step
        # disable_running_stats(model)
        enable_running_stats(model)
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        if args.bd_sam:
            _, poison_indicator, _ = additional_info
            second_loss = train_criterion(model(inputs), labels.long(), poison_indicator).mean()
        else:
            second_loss = train_criterion(model(inputs), labels.long()).mean()

        if regular is not None:
            second_loss += regular(model)
        second_loss.backward()
        sam_optimizer.second_step(zero_grad=True)
        batch_loss.append(second_loss.item() * labels.size(0))

    one_epoch_loss = sum(batch_loss) / len(train_loader.dataset)
    if args.lr_scheduler == 'ReduceLROnPlateau' and scheduler is not None:
        scheduler.step(one_epoch_loss)
    elif args.lr_scheduler == 'CosineAnnealingLR' and scheduler is not None:
        scheduler.step()
    end_time = time()
    logging.info(f"one epoch Mix training part done, use time = {end_time - start_time} s")
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
        torch.cuda.empty_cache()
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

