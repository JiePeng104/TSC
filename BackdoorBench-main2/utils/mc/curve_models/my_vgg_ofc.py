from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn

from models.my_vgg_ofc import VGG, make_layers
from utils.mc.curve_models import curves

__all__ = ['VGG16BN']


def make_layers_curve(cfg: List[Union[str, int]], batch_norm: bool = False, fix_points=None) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = curves.Conv2d(in_channels, v, kernel_size=3, padding=1, fix_points=fix_points)
            if batch_norm:
                layers += [conv2d, curves.BatchNorm2d(v, fix_points=fix_points), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGCurve(nn.Module):
    def __init__(
        self, cfg, batch_norm, fix_points, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.features = make_layers_curve(cfgs[cfg], batch_norm=batch_norm, fix_points=fix_points)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = curves.Linear(512, num_classes, fix_points=fix_points)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, coeffs_t) -> torch.Tensor:
        for block in self.features:
            if isinstance(block, curves.Conv2d) or isinstance(block, curves.BatchNorm2d):
                x = block(x, coeffs_t)
            else:
                x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x, coeffs_t)
        return x



cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG16BN:
    base = VGG
    curve = VGGCurve
    cfg = "D"
    kwargs_base = {
        "features": make_layers(cfgs[cfg], batch_norm=True),
        "init_weights": False
    }
    kwargs_curve = {
        "cfg": cfg,
        "batch_norm": True,
        "init_weights": False
    }
