from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from utils.mc.curve_models import curves
from models.resnet import ResNet, BasicBlock, Bottleneck

from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.utils import _log_api_usage_once

# from torchvision.models import resnet
# from torchvision.models import


def conv3x3(fix_points, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return curves.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        fix_points=fix_points,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(fix_points, in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return curves.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, fix_points=fix_points)


class BasicBlockCurve(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        fix_points,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = curves.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(fix_points, inplanes, planes, stride)
        self.bn1 = norm_layer(planes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(fix_points, planes, planes)
        self.bn2 = norm_layer(planes, fix_points=fix_points)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, coeffs_t) -> Tensor:
        identity = x

        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = self.relu(out)

        out = self.conv2(out, coeffs_t)
        out = self.bn2(out, coeffs_t)

        if self.downsample is not None:
            for s_module in self.downsample:
                identity = s_module(identity, coeffs_t)

        out += identity
        out = self.relu(out)

        return out


class BottleneckCurve(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        fix_points,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = curves.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(fix_points, inplanes, width)
        self.bn1 = norm_layer(width, fix_points=fix_points)
        self.conv2 = conv3x3(fix_points, width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, fix_points=fix_points)
        self.conv3 = conv1x1(fix_points, width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, coeffs_t) -> Tensor:
        identity = x

        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = self.relu(out)

        out = self.conv2(out, coeffs_t)
        out = self.bn2(out, coeffs_t)
        out = self.relu(out)

        out = self.conv3(out, coeffs_t)
        out = self.bn3(out, coeffs_t)

        if self.downsample is not None:
            for s_module in self.downsample:
                identity = s_module(identity, coeffs_t)

        out += identity
        out = self.relu(out)

        return out


class ResNetCurve(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlockCurve, BottleneckCurve]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        fix_points=None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = curves.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = curves.Conv2d(3, self.inplanes, kernel_size=7, fix_points=fix_points, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(fix_points, block, 64, layers[0])
        self.layer2 = self._make_layer(fix_points, block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(fix_points, block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(fix_points, block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = curves.Linear(512 * block.expansion, num_classes, fix_points=fix_points)

    def _make_layer(
        self,
        fix_points,
        block: Type[Union[BasicBlockCurve, BottleneckCurve]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(fix_points, self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, fix_points=fix_points),
            )

        layers = []
        layers.append(
            block(
                fix_points, self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    fix_points,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, coeffs_t) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x, coeffs_t)
        x = self.bn1(x, coeffs_t)
        x = self.relu(x)
        x = self.maxpool(x)

        for block in self.layer1:
            x = block(x, coeffs_t)
        for block in self.layer2:
            x = block(x, coeffs_t)
        for block in self.layer3:
            x = block(x, coeffs_t)
        for block in self.layer4:
            x = block(x, coeffs_t)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x, coeffs_t)

        return x

    def forward(self, x: Tensor, coeffs_t) -> Tensor:
        return self._forward_impl(x, coeffs_t)

    def retain_grad_penultimate(self):
        return self.penultimate.retain_grad()

    def get_grad_of_pen_representation(self):
        return self.penultimate.grad


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


class ResNet18Arc:
    base = ResNet
    curve = ResNetCurve
    kwargs_base = {'block': BasicBlock, 'layers': [2, 2, 2, 2]}
    kwargs_curve = {'block': BasicBlockCurve, 'layers': [2, 2, 2, 2]}


class ResNet50Arc:
    base = ResNet
    curve = ResNetCurve
    kwargs_base = {'block': Bottleneck, 'layers': [3, 4, 6, 3]}
    kwargs_curve = {'block': BottleneckCurve, 'layers': [3, 4, 6, 3]}
