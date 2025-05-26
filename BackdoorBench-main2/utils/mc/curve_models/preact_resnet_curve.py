import torch.nn as nn
import torch.nn.functional as F
from utils.mc.curve_models import curves
from models.preact_resnet import PreActResNet, PreActBlock


class PreActBlockCurve(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, fix_points, planes, stride=1):
        super(PreActBlockCurve, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = curves.BatchNorm2d(in_planes, fix_points=fix_points)
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = curves.Conv2d(in_planes, planes, kernel_size=3, fix_points=fix_points, stride=stride, padding=1,
                                   bias=False)

        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)

        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = curves.Conv2d(planes, planes, kernel_size=3, fix_points=fix_points, stride=1, padding=1,
                                   bias=False)

        self.ind = None

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = curves.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                                          stride=stride, bias=False, fix_points=fix_points)

    def forward(self, x, coeffs_t):
        out = self.bn1(x, coeffs_t)
        out = F.relu(out)
        shortcut = self.shortcut(out, coeffs_t) if hasattr(self, "shortcut") else x
        out = self.conv1(out, coeffs_t)
        out = self.bn2(out, coeffs_t)
        out = F.relu(out)
        out = self.conv2(out, coeffs_t)
        if self.ind is not None:
            out += shortcut[:, self.ind, :, :]
        else:
            out += shortcut
        return out


class PreActResNetCurve(nn.Module):
    def __init__(self, block, num_blocks, fix_points, num_classes=10):
        super(PreActResNetCurve, self).__init__()
        self.in_planes = 64
        self.conv1 = curves.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, fix_points=fix_points)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], fix_points, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], fix_points, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], fix_points, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], fix_points, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = curves.Linear(512 * block.expansion, num_classes, fix_points=fix_points)

    def _make_layer(self, block, planes, num_blocks, fix_points, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, fix_points=fix_points, stride=stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x, coeffs_t):
        out = self.conv1(x, coeffs_t)
        for block in self.layer1:
            out = block(out, coeffs_t)
        for block in self.layer2:
            out = block(out, coeffs_t)
        for block in self.layer3:
            out = block(out, coeffs_t)
        for block in self.layer4:
            out = block(out, coeffs_t)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out, coeffs_t)
        return out


class PreActResNet18Arc:
    base = PreActResNet
    curve = PreActResNetCurve
    kwargs_base = {'block': PreActBlock, 'num_blocks': [2, 2, 2, 2]}
    kwargs_curve = {'block': PreActBlockCurve, 'num_blocks': [2, 2, 2, 2]}
