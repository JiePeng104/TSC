import argparse
import logging
import os
import random
import sys
import scipy.optimize as opt

sys.path.append('../')
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def run_corr_matrix(net0, net1, val_loader):
    n = len(val_loader)
    mean0 = mean1 = std0 = std1 = None
    with torch.no_grad():
        net0.eval()
        net1.eval()
        for i, (images, target, *additional_info) in enumerate(tqdm(val_loader)):
            images = images.to('cuda')

            out0 = net0(images.float())
            out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
            out0 = out0.reshape(-1, out0.shape[2]).float()
            out1 = net1(images.float())
            out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
            out1 = out1.reshape(-1, out1.shape[2]).float()
            mean0_b = out0.mean(dim=0)
            mean1_b = out1.mean(dim=0)
            std0_b = out0.std(dim=0)
            std1_b = out1.std(dim=0)
            outer_b = (out0.T @ out1) / out0.shape[0]
            if i == 0:
                mean0 = torch.zeros_like(mean0_b)
                mean1 = torch.zeros_like(mean1_b)
                std0 = torch.zeros_like(std0_b)
                std1 = torch.zeros_like(std1_b)
                outer = torch.zeros_like(outer_b)
            mean0 += mean0_b / n
            mean1 += mean1_b / n
            std0 += std0_b / n
            std1 += std1_b / n
            outer += outer_b / n
    cov = outer - torch.outer(mean0, mean1)
    corr = cov / (torch.outer(std0, std1) + 1e-4)
    return corr


def compute_perm_map(corr_mtx):
    # sort the (i, j) channel pairs by correlation
    nchan = corr_mtx.shape[0]
    triples = [(i, j, corr_mtx[i, j].item()) for i in range(nchan) for j in range(nchan)]
    triples = sorted(triples, key=lambda p: -p[2])
    # greedily find a matching
    perm_d = {}
    for i, j, c in triples:
        if not (i in perm_d.keys() or j in perm_d.values()):
            perm_d[i] = j
    perm_map = torch.tensor([perm_d[i] for i in range(nchan)])
    # qual_map will be a permutation of the indices in the order
    # of the quality / degree of correlation between the neurons found in the permutation.
    # this just for visualization purposes.
    qual_l = [corr_mtx[i, perm_map[i]].item() for i in range(nchan)]
    qual_map = torch.tensor(sorted(range(nchan), key=lambda i: -qual_l[i]))
    return perm_map, qual_map


# returns the channel-permutation to make layer1's activations most closely
# match layer0's.
def get_layer_perm(net0, net1, val_loader, method='max_weight', maximize=False, vizz=False):
    corr_mtx = run_corr_matrix(net0, net1, val_loader)

    if method == 'greedy':
        perm_map, qual_map = compute_perm_map(corr_mtx)
        # if vizz:
        #     corr_mtx_viz = (corr_mtx[qual_map].T[perm_map[qual_map]]).T
        #     viz(corr_mtx_viz)
    elif method == 'max_weight':
        corr_mtx_a = corr_mtx.cpu().numpy()
        logging.info(f"Computing permutation for corr matrix: {corr_mtx_a.shape}")
        # Here, we maximize the corr between model A and model B to find the WORST permutation
        row_ind, col_ind = opt.linear_sum_assignment(corr_mtx_a, maximize=maximize)
        assert (row_ind == np.arange(len(corr_mtx_a))).all()
        perm_map = torch.tensor(col_ind).long()
    else:
        raise Exception('Unknown method: %s' % method)
    return perm_map


def permute_output(perm_map, conv, bn):
    pre_weights = [
        conv.weight,
        bn.weight,
        bn.bias,
        bn.running_mean,
        bn.running_var,
    ]
    for w in pre_weights:
        w.data = w[perm_map]


def permute_conv_output(perm_map, conv):
    pre_weights = [
        conv.weight,
    ]
    for w in pre_weights:
        w.data = w[perm_map]


# modifies the weight matrix of a convolution layer for a given
# permutation of the input channels
def permute_input(perm_map, after_convs):
    if not isinstance(after_convs, list):
        after_convs = [after_convs]
    post_weights = [c.weight for c in after_convs]
    for w in post_weights:
        w.data = w[:, perm_map, :, :]


# Here, the code of permutation is written 'linearly'. You know, the ResNet residual.....
def find_permutation_PreAct_ResNet18(model0, model1, val_loader, maximize=False):

    # First Conv
    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.conv1, model1.layer1[0].bn1)
    # make permutation for RESIDUAL Block
    permute_output(perm_map, model1.layer1[0].conv2, model1.layer1[1].bn1)
    permute_output(perm_map, model1.layer1[1].conv2, model1.layer2[0].bn1)
    permute_input(perm_map, [model1.layer1[0].conv1, model1.layer1[1].conv1,
                             model1.layer2[0].conv1, model1.layer2[0].shortcut[0]])

    # First Layer
    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.layer1[0].bn1(self.conv1(x)))
            x = self.layer1[0].conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer1[0].conv1, model1.layer1[0].bn2)
    permute_input(perm_map, model1.layer1[0].conv2)


    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            x = self.layer1[0](x)
            x = F.relu(self.layer1[1].bn1(x))
            x = self.layer1[1].conv1(x)
            return x
    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer1[1].conv1, model1.layer1[1].bn2)
    permute_input(perm_map, model1.layer1[1].conv2)

    # Layer2
    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            x = self.layer1(x)
            x = F.relu(self.layer2[0].bn1(x))
            x = self.layer2[0].conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer2[0].conv1, model1.layer2[0].bn2)
    permute_input(perm_map, model1.layer2[0].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            x = self.layer1(x)
            block = self.layer2[0]
            x = block.conv1(F.relu(block.bn1(x)))
            x = block.conv2(F.relu(block.bn2(x)))
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer2[0].conv2, model1.layer2[1].bn1)
    permute_output(perm_map, model1.layer2[1].conv2, model1.layer3[0].bn1)
    permute_conv_output(perm_map, model1.layer2[0].shortcut[0])
    permute_input(perm_map, [model1.layer2[1].conv1,
                             model1.layer3[0].conv1, model1.layer3[0].shortcut[0]])

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2[0](x)
            block = self.layer2[1]
            x = block.conv1(F.relu(block.bn1(x)))

            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer2[1].conv1, model1.layer2[1].bn2)
    permute_input(perm_map, model1.layer2[1].conv2)


    # Layer3
    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = F.relu(self.layer3[0].bn1(x))
            x = self.layer3[0].conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer3[0].conv1, model1.layer3[0].bn2)
    permute_input(perm_map, model1.layer3[0].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            block = self.layer3[0]
            x = block.conv1(F.relu(block.bn1(x)))
            x = block.conv2(F.relu(block.bn2(x)))
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer3[0].conv2, model1.layer3[1].bn1)
    permute_output(perm_map, model1.layer3[1].conv2, model1.layer4[0].bn1)
    permute_conv_output(perm_map, model1.layer3[0].shortcut[0])
    permute_input(perm_map, [model1.layer3[1].conv1,
                             model1.layer4[0].conv1, model1.layer4[0].shortcut[0]])

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3[0](x)
            block = self.layer3[1]
            x = block.conv1(F.relu(block.bn1(x)))

            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer3[1].conv1, model1.layer3[1].bn2)
    permute_input(perm_map, model1.layer3[1].conv2)

    # Layer4
    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = F.relu(self.layer4[0].bn1(x))
            x = self.layer4[0].conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer4[0].conv1, model1.layer4[0].bn2)
    permute_input(perm_map, model1.layer4[0].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            block = self.layer4[0]
            x = block.conv1(F.relu(block.bn1(x)))
            x = block.conv2(F.relu(block.bn2(x)))
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer4[0].conv2, model1.layer4[1].bn1)
    permute_conv_output(perm_map, model1.layer4[1].conv2)
    permute_conv_output(perm_map, model1.layer4[0].shortcut[0])
    permute_input(perm_map, model1.layer4[1].conv1)
    # Make permutation for Linear Layer
    model1.linear.weight.data = model1.linear.weight[:, perm_map]

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4[0](x)
            block = self.layer4[1]
            x = block.conv1(F.relu(block.bn1(x)))

            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer4[1].conv1, model1.layer4[1].bn2)
    permute_input(perm_map, model1.layer4[1].conv2)

    return model1


def find_permutation_PreAct_ResNet18_one_layer(model0, model1, val_loader, maximize=False):
    layer_permutation = random.randint(0, 3)
    logging.info(f"Randomly permutation one layer, the choosen layer:{layer_permutation}")
    # First Conv
    if layer_permutation == 0:
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.conv1, model1.layer1[0].bn1)
        # make permutation for RESIDUAL Block
        permute_output(perm_map, model1.layer1[0].conv2, model1.layer1[1].bn1)
        permute_output(perm_map, model1.layer1[1].conv2, model1.layer2[0].bn1)
        permute_input(perm_map, [model1.layer1[0].conv1, model1.layer1[1].conv1,
                                 model1.layer2[0].conv1, model1.layer2[0].shortcut[0]])

        # First Layer
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.layer1[0].bn1(self.conv1(x)))
                x = self.layer1[0].conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer1[0].conv1, model1.layer1[0].bn2)
        permute_input(perm_map, model1.layer1[0].conv2)


        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                x = self.layer1[0](x)
                x = F.relu(self.layer1[1].bn1(x))
                x = self.layer1[1].conv1(x)
                return x
        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer1[1].conv1, model1.layer1[1].bn2)
        permute_input(perm_map, model1.layer1[1].conv2)

    # Layer2
    if layer_permutation == 1:
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                x = self.layer1(x)
                x = F.relu(self.layer2[0].bn1(x))
                x = self.layer2[0].conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer2[0].conv1, model1.layer2[0].bn2)
        permute_input(perm_map, model1.layer2[0].conv2)

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                x = self.layer1(x)
                block = self.layer2[0]
                x = block.conv1(F.relu(block.bn1(x)))
                x = block.conv2(F.relu(block.bn2(x)))
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer2[0].conv2, model1.layer2[1].bn1)
        permute_output(perm_map, model1.layer2[1].conv2, model1.layer3[0].bn1)
        permute_conv_output(perm_map, model1.layer2[0].shortcut[0])
        permute_input(perm_map, [model1.layer2[1].conv1,
                                 model1.layer3[0].conv1, model1.layer3[0].shortcut[0]])

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2[0](x)
                block = self.layer2[1]
                x = block.conv1(F.relu(block.bn1(x)))

                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer2[1].conv1, model1.layer2[1].bn2)
        permute_input(perm_map, model1.layer2[1].conv2)


    # Layer3
    if layer_permutation == 2:
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = F.relu(self.layer3[0].bn1(x))
                x = self.layer3[0].conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer3[0].conv1, model1.layer3[0].bn2)
        permute_input(perm_map, model1.layer3[0].conv2)

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                block = self.layer3[0]
                x = block.conv1(F.relu(block.bn1(x)))
                x = block.conv2(F.relu(block.bn2(x)))
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer3[0].conv2, model1.layer3[1].bn1)
        permute_output(perm_map, model1.layer3[1].conv2, model1.layer4[0].bn1)
        permute_conv_output(perm_map, model1.layer3[0].shortcut[0])
        permute_input(perm_map, [model1.layer3[1].conv1,
                                 model1.layer4[0].conv1, model1.layer4[0].shortcut[0]])

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3[0](x)
                block = self.layer3[1]
                x = block.conv1(F.relu(block.bn1(x)))

                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer3[1].conv1, model1.layer3[1].bn2)
        permute_input(perm_map, model1.layer3[1].conv2)

    # Layer4
    if layer_permutation == 3:
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = F.relu(self.layer4[0].bn1(x))
                x = self.layer4[0].conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer4[0].conv1, model1.layer4[0].bn2)
        permute_input(perm_map, model1.layer4[0].conv2)

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                block = self.layer4[0]
                x = block.conv1(F.relu(block.bn1(x)))
                x = block.conv2(F.relu(block.bn2(x)))
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer4[0].conv2, model1.layer4[1].bn1)
        permute_conv_output(perm_map, model1.layer4[1].conv2)
        permute_conv_output(perm_map, model1.layer4[0].shortcut[0])
        permute_input(perm_map, model1.layer4[1].conv1)
        # Make permutation for Linear Layer
        model1.linear.weight.data = model1.linear.weight[:, perm_map]

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4[0](x)
                block = self.layer4[1]
                x = block.conv1(F.relu(block.bn1(x)))

                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer4[1].conv1, model1.layer4[1].bn2)
        permute_input(perm_map, model1.layer4[1].conv2)

    return model1


def find_permutation_ResNet18(model0, model1, val_loader, maximize=False):
    # First Conv
    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.conv1, model1.bn1)
    # make permutation for RESIDUAL Block
    permute_output(perm_map, model1.layer1[0].conv2, model1.layer1[0].bn2)
    permute_output(perm_map, model1.layer1[1].conv2, model1.layer1[1].bn2)
    permute_input(perm_map, [model1.layer1[0].conv1, model1.layer1[1].conv1,
                             model1.layer2[0].conv1, model1.layer2[0].downsample[0]])

    # The First layer - Conv1
    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1[0].conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer1[0].conv1, model1.layer1[0].bn1)
    permute_input(perm_map, model1.layer1[0].conv2)

    # The First layer - Conv2
    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1[0](x)
            x = self.layer1[1].conv1(x)
            return x
    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer1[1].conv1, model1.layer1[1].bn1)
    permute_input(perm_map, model1.layer1[1].conv2)

    # Layer2
    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2[0].conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer2[0].conv1, model1.layer2[0].bn1)
    permute_input(perm_map, model1.layer2[0].conv2)


    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            block = self.layer2[0]
            x = F.relu(block.bn1(block.conv1(x)))
            x = block.conv2(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer2[0].conv2, model1.layer2[0].bn2)
    permute_output(perm_map, model1.layer2[1].conv2, model1.layer2[1].bn2)
    permute_output(perm_map, model1.layer2[0].downsample[0], model1.layer2[0].downsample[1])
    permute_input(perm_map, [model1.layer2[1].conv1,
                             model1.layer3[0].conv1, model1.layer3[0].downsample[0]])


    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2[0](x)
            x = self.layer2[1].conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer2[1].conv1, model1.layer2[1].bn1)
    permute_input(perm_map, model1.layer2[1].conv2)

    # Layer3
    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3[0].conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer3[0].conv1, model1.layer3[0].bn1)
    permute_input(perm_map, model1.layer3[0].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            block = self.layer3[0]
            x = F.relu(block.bn1(block.conv1(x)))
            x = block.conv2(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer3[0].conv2, model1.layer3[0].bn2)
    permute_output(perm_map, model1.layer3[1].conv2, model1.layer3[1].bn2)
    permute_output(perm_map, model1.layer3[0].downsample[0], model1.layer3[0].downsample[1])
    permute_input(perm_map, [model1.layer3[1].conv1,
                             model1.layer4[0].conv1, model1.layer4[0].downsample[0]])

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3[0](x)
            x = self.layer3[1].conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer3[1].conv1, model1.layer3[1].bn1)
    permute_input(perm_map, model1.layer3[1].conv2)

    # Layer4
    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4[0].conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer4[0].conv1, model1.layer4[0].bn1)
    permute_input(perm_map, model1.layer4[0].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            block = self.layer4[0]
            x = F.relu(block.bn1(block.conv1(x)))
            x = block.conv2(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer4[0].conv2, model1.layer4[0].bn2)
    permute_output(perm_map, model1.layer4[1].conv2, model1.layer4[1].bn2)
    permute_output(perm_map, model1.layer4[0].downsample[0], model1.layer4[0].downsample[1])
    permute_input(perm_map, model1.layer4[1].conv1)
    # Make permutation for Linear Layer
    model1.linear.weight.data = model1.linear.weight[:, perm_map]

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4[0](x)
            x = self.layer4[1].conv1(x)
            return x

    perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
    permute_output(perm_map, model1.layer4[1].conv1, model1.layer4[1].bn1)
    permute_input(perm_map, model1.layer4[1].conv2)

    return model1


def find_permutation_ResNet18_one_layer(model0, model1, val_loader, maximize=False):
    layer_permutation = random.randint(0, 4)
    logging.info(f"Randomly permutation one layer, the choosen layer:{layer_permutation}")
    # First Conv
    if layer_permutation == 0:
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = self.conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.conv1, model1.bn1)
        # make permutation for RESIDUAL Block
        permute_output(perm_map, model1.layer1[0].conv2, model1.layer1[0].bn2)
        permute_output(perm_map, model1.layer1[1].conv2, model1.layer1[1].bn2)
        permute_input(perm_map, [model1.layer1[0].conv1, model1.layer1[1].conv1,
                                 model1.layer2[0].conv1, model1.layer2[0].downsample[0]])

    # The First layer - Conv1

    if layer_permutation == 1:
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1[0].conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer1[0].conv1, model1.layer1[0].bn1)
        permute_input(perm_map, model1.layer1[0].conv2)

    # The First layer - Conv2
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1[0](x)
                x = self.layer1[1].conv1(x)
                return x
        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer1[1].conv1, model1.layer1[1].bn1)
        permute_input(perm_map, model1.layer1[1].conv2)

    # Layer2
    if layer_permutation == 2:
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2[0].conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer2[0].conv1, model1.layer2[0].bn1)
        permute_input(perm_map, model1.layer2[0].conv2)

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                block = self.layer2[0]
                x = F.relu(block.bn1(block.conv1(x)))
                x = block.conv2(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer2[0].conv2, model1.layer2[0].bn2)
        permute_output(perm_map, model1.layer2[1].conv2, model1.layer2[1].bn2)
        permute_output(perm_map, model1.layer2[0].downsample[0], model1.layer2[0].downsample[1])
        permute_input(perm_map, [model1.layer2[1].conv1,
                                 model1.layer3[0].conv1, model1.layer3[0].downsample[0]])

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2[0](x)
                x = self.layer2[1].conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer2[1].conv1, model1.layer2[1].bn1)
        permute_input(perm_map, model1.layer2[1].conv2)

    # Layer3
    if layer_permutation == 3:
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3[0].conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer3[0].conv1, model1.layer3[0].bn1)
        permute_input(perm_map, model1.layer3[0].conv2)

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2(x)
                block = self.layer3[0]
                x = F.relu(block.bn1(block.conv1(x)))
                x = block.conv2(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer3[0].conv2, model1.layer3[0].bn2)
        permute_output(perm_map, model1.layer3[1].conv2, model1.layer3[1].bn2)
        permute_output(perm_map, model1.layer3[0].downsample[0], model1.layer3[0].downsample[1])
        permute_input(perm_map, [model1.layer3[1].conv1,
                                 model1.layer4[0].conv1, model1.layer4[0].downsample[0]])

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3[0](x)
                x = self.layer3[1].conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer3[1].conv1, model1.layer3[1].bn1)
        permute_input(perm_map, model1.layer3[1].conv2)

    # Layer4
    if layer_permutation == 4:
        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4[0].conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer4[0].conv1, model1.layer4[0].bn1)
        permute_input(perm_map, model1.layer4[0].conv2)

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                block = self.layer4[0]
                x = F.relu(block.bn1(block.conv1(x)))
                x = block.conv2(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer4[0].conv2, model1.layer4[0].bn2)
        permute_output(perm_map, model1.layer4[1].conv2, model1.layer4[1].bn2)
        permute_output(perm_map, model1.layer4[0].downsample[0], model1.layer4[0].downsample[1])
        permute_input(perm_map, model1.layer4[1].conv1)
        # Make permutation for Linear Layer
        model1.linear.weight.data = model1.linear.weight[:, perm_map]

        class Subnet(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                self = self.model
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4[0](x)
                x = self.layer4[1].conv1(x)
                return x

        perm_map = get_layer_perm(Subnet(model0), Subnet(model1), val_loader, maximize=maximize)
        permute_output(perm_map, model1.layer4[1].conv1, model1.layer4[1].bn1)
        permute_input(perm_map, model1.layer4[1].conv2)

    return model1


def find_permutation_ResNet50(model0, model1, val_loader, maximize=False):
    def get_model_resnet50_block(model):
        return nn.Sequential(nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
                             *model.layer1, *model.layer2, *model.layer3, *model.layer4)

    # the model_block: first layer of ResNet, all the blocks for the rest...
    # len(model_block) = 17
    model0_block = get_model_resnet50_block(model0)

    model1_block = get_model_resnet50_block(model1)

    # make permutation for each intra-block
    # The block of ResNet50 consists of three conv-bn-relu layers
    for k in range(1, len(model0_block)):
        block0 = model0_block[k]
        block1 = model1_block[k]
        subnet0 = nn.Sequential(model0_block[:k],
                                block0.conv1, block0.bn1, block0.relu)
        subnet1 = nn.Sequential(model1_block[:k],
                                block1.conv1, block1.bn1, block1.relu)
        perm_map = get_layer_perm(subnet0, subnet1, val_loader, maximize=maximize)
        permute_output(perm_map, block1.conv1, block1.bn1)
        permute_input(perm_map, block1.conv2)

        subnet0 = nn.Sequential(model0_block[:k],
                                block0.conv1, block0.bn1, block0.relu,
                                block0.conv2, block0.bn2, block0.relu)
        subnet1 = nn.Sequential(model1_block[:k],
                                block1.conv1, block1.bn1, block1.relu,
                                block1.conv2, block1.bn2, block1.relu)
        perm_map = get_layer_perm(subnet0, subnet1, val_loader, maximize=maximize)
        permute_output(perm_map, block1.conv2, block1.bn2)
        permute_input(perm_map, block1.conv3)

    # make permutation for block by block

    # kk index for each block
    last_kk = None
    perm_map = None
    kk_list = sum([[0], 3*[3], 4*[7], 6*[13], 3*[16]], [])
    for k in range(len(model0_block)):
        kk = kk_list[k]

        if kk != last_kk:
            perm_map = get_layer_perm(model0_block[:kk + 1], model1_block[:kk + 1], val_loader, maximize=maximize)
            last_kk = kk

        if k > 0:
            permute_output(perm_map, model1_block[k].conv3, model1_block[k].bn3)
            shortcut = model1_block[k].downsample
            if shortcut:
                permute_output(perm_map, shortcut[0], shortcut[1])
        else:
            permute_output(perm_map, model1.conv1, model1.bn1)

        if k + 1 < len(model1_block):
            permute_input(perm_map, model1_block[k + 1].conv1)
            shortcut = model1_block[k + 1].downsample
            if shortcut:
                permute_input(perm_map, shortcut[0])
        else:
            model1.fc.weight.data = model1.fc.weight[:, perm_map]

    return model1


def find_permutation_VGG16BN(model0, model1, val_loader, maximize=False):
    method = 'max_weight'

    def subnet_features(model, n_layers):
        return model.features[:n_layers]

    def subnet_classifier(model, n_layers):
        class flatten_subnet(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = torch.flatten

            def forward(self, x):
                return self.flatten(x, 1)

        return nn.Sequential(model.features, model.avgpool, flatten_subnet(), model.classifier[:n_layers])
        # class Subnet(nn.Module):
        #     def __init__(self, model, n_layers):
        #         super().__init__()
        #         self.model = model
        #         self.n_layers = n_layers
        #
        #     def forward(self, x):
        #         n_layer = self.n_layers
        #         self = self.model
        #         x = self.features(x)
        #         x = self.avgpool(x)
        #         x = torch.flatten(x, 1)
        #         for k in range(n_layer):
        #             x = self.classifier[k](x)
        #         return x
        #
        # return Subnet(model, n_layers)

    feats1 = model1.features
    classifier1 = model1.classifier
    n = len(feats1)
    for i in range(n):
        layer = feats1[i]
        if isinstance(layer, nn.Conv2d):
            # get permutation and permute output of conv and maybe bn
            if isinstance(feats1[i + 1], nn.BatchNorm2d):
                assert isinstance(feats1[i + 2], nn.ReLU)
                perm_map = get_layer_perm(subnet_features(model0, i + 2 + 1), subnet_features(model1, i + 2 + 1), val_loader, method=method,
                                          maximize=maximize)
                permute_output(perm_map, feats1[i], feats1[i + 1])
            else:
                assert isinstance(feats1[i + 1], nn.ReLU)
                perm_map = get_layer_perm(subnet_features(model0, i + 1 + 1), subnet_features(model1, i + 1 + 1), val_loader, method=method,
                                          maximize=maximize)
                permute_conv_output(perm_map, feats1[i])
            next_layer = None
            for j in range(i + 1, n):
                if isinstance(feats1[j], nn.Conv2d):
                    next_layer = feats1[j]
                    break
            # look for succeeding layer to permute input
            if next_layer is None:
                next_layer = model1.classifier

            permute_input(perm_map, next_layer)

    # # permutation of first linear in classifier
    # torch.cuda.empty_cache()
    # logging.info('first classifier')
    # perm_map = get_layer_perm(subnet_classifier(model0, 3), subnet_classifier(model1, 3), val_loader, method=method,
    #                           maximize=maximize)
    # linear1 = model1.classifier[0]
    # w_list = [linear1.weight, linear1.bias]
    # for w in w_list:
    #     w.data = w[perm_map]
    # w = model1.classifier[3].weight
    # w.data = w.data[:, perm_map]
    #
    # # permutation of second linear in classifier
    # torch.cuda.empty_cache()
    # perm_map = get_layer_perm(subnet_classifier(model0, 6), subnet_classifier(model1, 6), val_loader, method=method,
    #                           maximize=maximize)
    # linear2 = model1.classifier[3]
    # w_list = [linear2.weight, linear2.bias]
    # for w in w_list:
    #     w.data = w[perm_map]
    # w = model1.classifier[6].weight
    # w.data = w.data[:, perm_map]

    return model1



def find_permutation_VGG19BN(model0, model1, val_loader, maximize=False):
    method = 'max_weight'

    def subnet_features(model, n_layers):
        return model.features[:n_layers]

    def subnet_classifier(model, n_layers):
        class flatten_subnet(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = torch.flatten

            def forward(self, x):
                return self.flatten(x, 1)

        return nn.Sequential(model.features, model.avgpool, flatten_subnet(), model.classifier[:n_layers])
        # class Subnet(nn.Module):
        #     def __init__(self, model, n_layers):
        #         super().__init__()
        #         self.model = model
        #         self.n_layers = n_layers
        #
        #     def forward(self, x):
        #         n_layer = self.n_layers
        #         self = self.model
        #         x = self.features(x)
        #         x = self.avgpool(x)
        #         x = torch.flatten(x, 1)
        #         for k in range(n_layer):
        #             x = self.classifier[k](x)
        #         return x
        #
        # return Subnet(model, n_layers)

    feats1 = model1.features
    classifier1 = model1.classifier
    n = len(feats1)
    for i in range(n):
        layer = feats1[i]
        if isinstance(layer, nn.Conv2d):
            next_layer = None
            for j in range(i + 1, n):
                if isinstance(feats1[j], nn.Conv2d):
                    next_layer = feats1[j]
                    break
            if next_layer is None:
                break
            # get permutation and permute output of conv and maybe bn
            if isinstance(feats1[i + 1], nn.BatchNorm2d):
                assert isinstance(feats1[i + 2], nn.ReLU)
                perm_map = get_layer_perm(subnet_features(model0, i + 2 + 1), subnet_features(model1, i + 2 + 1), val_loader, method=method,
                                          maximize=maximize)
                permute_output(perm_map, feats1[i], feats1[i + 1])
            else:
                assert isinstance(feats1[i + 1], nn.ReLU)
                perm_map = get_layer_perm(subnet_features(model0, i + 1 + 1), subnet_features(model1, i + 1 + 1), val_loader, method=method,
                                          maximize=maximize)
                permute_conv_output(perm_map, feats1[i])
            # look for succeeding layer to permute input
            # if next_layer is None:
            #     logging.info('last features')
            #     # next_layer = model1.classifier[0]
            #     torch.cuda.empty_cache()
            #     perm_map = get_layer_perm(subnet_classifier(model0, 0), subnet_classifier(model1, 0), val_loader, method='greedy',
            #                               maximize=maximize)
            #     w = model1.classifier[0].weight
            #     w.data = w.data[:, perm_map]
            #     break
            permute_input(perm_map, next_layer)

    # # permutation of first linear in classifier
    # torch.cuda.empty_cache()
    # logging.info('first classifier')
    # perm_map = get_layer_perm(subnet_classifier(model0, 3), subnet_classifier(model1, 3), val_loader, method=method,
    #                           maximize=maximize)
    # linear1 = model1.classifier[0]
    # w_list = [linear1.weight, linear1.bias]
    # for w in w_list:
    #     w.data = w[perm_map]
    # w = model1.classifier[3].weight
    # w.data = w.data[:, perm_map]
    #
    # # permutation of second linear in classifier
    # torch.cuda.empty_cache()
    # perm_map = get_layer_perm(subnet_classifier(model0, 6), subnet_classifier(model1, 6), val_loader, method=method,
    #                           maximize=maximize)
    # linear2 = model1.classifier[3]
    # w_list = [linear2.weight, linear2.bias]
    # for w in w_list:
    #     w.data = w[perm_map]
    # w = model1.classifier[6].weight
    # w.data = w.data[:, perm_map]
    torch.cuda.empty_cache()

    return model1
