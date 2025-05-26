import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple, Union

from tsqc.utils.mc.curve_models import curves
from models.clip_model import CLIP, AttentionPool2d


# from torchvision.models import resnet
# from torchvision.models import

class BottleneckCurve(nn.Module):
    expansion = 4

    def __init__(self, fix_points, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = curves.Conv2d(inplanes, planes, 1, bias=False, fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(planes, fix_points=fix_points)

        self.conv2 = curves.Conv2d(planes, planes, 3, padding=1, bias=False, fix_points=fix_points)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = curves.Conv2d(planes, planes * self.expansion, 1, bias=False, fix_points=fix_points)
        self.bn3 = curves.BatchNorm2d(planes * self.expansion, fix_points=fix_points)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * BottleneckCurve.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", curves.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False, fix_points=fix_points)),
                ("1", curves.BatchNorm2d(planes * self.expansion, fix_points=fix_points))
            ]))

    def forward(self, x: torch.Tensor, coeffs_t):
        identity = x

        out = self.relu(self.bn1(self.conv1(x, coeffs_t), coeffs_t))
        out = self.relu(self.bn2(self.conv2(out, coeffs_t), coeffs_t))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out, coeffs_t), coeffs_t)

        if self.downsample is not None:
            for s_module in self.downsample:
                if isinstance(s_module, curves.Conv2d) or isinstance(s_module, curves.BatchNorm2d):
                    identity = s_module(identity, coeffs_t)
                else:
                    identity = s_module(identity)
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2dCurve(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, fix_points=None):
        super().__init__()
        self.positional_embedding = curves.PositionalEmbedding(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5,
            fix_points=fix_points)
        self.k_proj = curves.Linear(embed_dim, embed_dim, fix_points=fix_points)
        self.q_proj = curves.Linear(embed_dim, embed_dim, fix_points=fix_points)
        self.v_proj = curves.Linear(embed_dim, embed_dim, fix_points=fix_points)
        self.c_proj = curves.Linear(embed_dim, output_dim or embed_dim, fix_points=fix_points)
        self.num_heads = num_heads

    def forward(self, x, coeffs_t):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x = self.positional_embedding(x, coeffs_t)

        k_proj_weight_t, k_proj_bias_t = self.k_proj.compute_weights_t(coeffs_t)
        q_proj_weight_t, q_proj_bias_t = self.q_proj.compute_weights_t(coeffs_t)
        v_proj_weight_t, v_proj_bias_t = self.v_proj.compute_weights_t(coeffs_t)
        c_proj_weight_t, c_proj_bias_t = self.c_proj.compute_weights_t(coeffs_t)

        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=q_proj_weight_t,
            k_proj_weight=k_proj_weight_t,
            v_proj_weight=v_proj_weight_t,
            in_proj_weight=None,
            in_proj_bias=torch.cat([q_proj_bias_t, k_proj_bias_t, v_proj_bias_t]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=c_proj_weight_t,
            out_proj_bias=c_proj_bias_t,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]


class ModifiedResNetCurve(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, fix_points=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = curves.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False, fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(width // 2, fix_points=fix_points)
        self.conv2 = curves.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False, fix_points=fix_points)
        self.bn2 = curves.BatchNorm2d(width // 2, fix_points=fix_points)
        self.conv3 = curves.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False, fix_points=fix_points)
        self.bn3 = curves.BatchNorm2d(width, fix_points=fix_points)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(fix_points, width, layers[0])
        self.layer2 = self._make_layer(fix_points, width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(fix_points, width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(fix_points, width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2dCurve(input_resolution // 32, embed_dim, heads, output_dim,
                                             fix_points=fix_points)
        # self.attnpool = self.AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, fix_points, planes, blocks, stride=1):
        layers = [BottleneckCurve(fix_points, self._inplanes, planes, stride)]

        self._inplanes = planes * BottleneckCurve.expansion
        for _ in range(1, blocks):
            layers.append(BottleneckCurve(fix_points, self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, coeffs_t):
        def stem(x, coeffs_t):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x, coeffs_t), coeffs_t))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight_0.dtype)
        x = stem(x, coeffs_t)

        for block in self.layer1:
            x = block(x, coeffs_t)
        for block in self.layer2:
            x = block(x, coeffs_t)
        for block in self.layer3:
            x = block(x, coeffs_t)
        for block in self.layer4:
            x = block(x, coeffs_t)
        # x = self.attnpool(x)
        x = self.attnpool(x, coeffs_t)

        return x


class CLIP_Curve(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 fix_points=None,
                 ):
        super().__init__()

        vision_heads = vision_width * 32 // 64
        self.visual = ModifiedResNetCurve(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width,
            fix_points=fix_points
        )

    @property
    def dtype(self):
        return self.visual.conv1.weight_0.dtype

    def encode_image(self, image, coeffs_t):
        return self.visual(image.type(self.dtype), coeffs_t)

    def forward(self, image, coeffs_t):
        image_features = self.encode_image(image, coeffs_t)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features


class ResNet50Arc_CLIP:
    base = CLIP
    curve = CLIP_Curve
    kwargs_base = {'embed_dim': 1024, 'image_resolution': 224, 'vision_layers': (3, 4, 6, 3), 'vision_width': 64}
    kwargs_curve = {'embed_dim': 1024, 'image_resolution': 224, 'vision_layers': (3, 4, 6, 3), 'vision_width': 64}
