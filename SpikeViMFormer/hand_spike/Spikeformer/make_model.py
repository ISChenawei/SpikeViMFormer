import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_convnext import convnext_tiny, LayerNorm
from .backbones.spike_transformer import metaspikformer_8_512
from .backbones.resnet import Resnet
from timm.layers import DropPath
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter
from SpikeViMFormer.Utils import init
from spikingjelly.activation_based import neuron, functional
from .Qtrick import Multispike_norm
# class MlpSNN(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., reduction=4):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#
#         self.fc1 = nn.Conv1d(in_features, hidden_features,kernel_size=1, stride=1, bias=False)
#         self.spike1 = Multispike_norm()
#         self.spike2 = Multispike_norm()
#         self.spike3 = Multispike_norm()
#         self.fc2 = nn.Conv1d(hidden_features,out_features, kernel_size=1, stride=1, bias=False)
#
#         self.se_fc1 = nn.Conv1d(out_features, out_features // reduction, kernel_size=1, stride=1, bias=False)
#         self.se_fc2 = nn.Conv1d(out_features // reduction, out_features, kernel_size=1, stride=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(hidden_features)
#         self.bn2 = nn.BatchNorm1d(out_features)
#         self.bn_se = nn.BatchNorm1d( out_features // reduction)
#         self.bn_se = nn.BatchNorm1d(out_features)
#         self.se_spike = Multispike_norm()
#         self.se_sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.spike1(x)
#         x = self.fc1(x.permute(0,2,1))
#         x = self.bn1(x)
#         x = self.spike2(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         se = x.permute(0,2,1).mean(dim=1, keepdim=True)
#         se = self.se_spike(se)
#         se = self.se_fc1(se.permute(0,2,1))
#         se = self.se_fc2(se)
#         se = self.se_sigmoid(se)
#
#         return x * se
#
# class MambaLayerSNN(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#         self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
#         self.ssm = nn.Linear(dim, dim)
#         self.gate_spike = Multispike_norm()
#         self.norm = nn.LayerNorm(dim)
#         self.spike = Multispike_norm()
#     def forward(self, x):
#         x = self.spike(x)  # (B, L, C) -> (B, C, L)
#         local_x = self.conv(x)
#         global_x = self.ssm(x.permute(0, 2, 1))  # (B, C, L) -> (B, L, C)
#         gate = self.gate_spike(global_x)
#         x = local_x.permute(0, 2, 1) * gate
#         x = self.norm(x)
#         return x
# class MLLABlockSNN(nn.Module):
#     def __init__(self, dim, input_resolution, num_heads=4, mlp_ratio=4., drop=0., drop_path=0.,
#                  norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#
#         self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
#         self.norm1 = norm_layer(dim)
#         # self.se_fc1 = nn.Conv1d(out_features, out_features // reduction, kernel_size=1, stride=1, bias=False)
#         # self.in_proj = nn.Linear(dim, dim)
#         self.in_proj = nn.Conv1d(dim, dim,kernel_size=1, stride=1, bias=False)
#         self.act_proj = nn.Linear(dim, dim)
#         self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
#         self.spike1 = Multispike_norm()
#         self.spike2 = Multispike_norm()
#         self.spike3 = Multispike_norm()
#         self.spike4 = Multispike_norm()
#         self.mamba = MambaLayerSNN(dim)
#
#         self.out_proj = nn.Linear(dim, dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#         self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
#         self.norm2 = norm_layer(dim)
#         self.mlp = MlpSNN(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
#
#     def forward(self, x):
#
#         B, C, L, = x.shape
#         H = self.input_resolution
#         # import pdb;
#         # pdb.set_trace()
#         assert L == H, "Input feature has wrong size"
#
#         x = x + self.cpe1(x)
#         shortcut = x
#         # import pdb;
#         # pdb.set_trace()
#         x = self.norm1(x.permute(0, 2, 1))
#         act_res = self.spike1(self.act_proj(x))
#
#         x = self.in_proj(x.permute(0, 2, 1))
#         # import pdb;
#         # pdb.set_trace()
#         x = self.spike2(self.dwc(x))
#         x = self.mamba(x)
#         x0 = x
#         x = self.spike3(x)
#         x = self.out_proj(x * act_res + x0)
#         x = self.drop_path(x)
#         x = shortcut + x.permute(0, 2, 1)
#         x = x + self.cpe2(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 1))))
#
#         return x
#
#
# class MS_MLP(nn.Module):
#     def __init__(
#             self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0,
#     ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
#         self.fc1_bn = nn.BatchNorm1d(hidden_features)
#         self.fc1_spike = Multispike_norm()
#
#         self.fc2_conv = nn.Conv1d(
#             hidden_features, out_features, kernel_size=1, stride=1
#         )
#         self.fc2_bn = nn.BatchNorm1d(out_features)
#         self.fc2_spike = Multispike_norm()
#
#         self.c_hidden = hidden_features
#         self.c_output = out_features
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         N = H * W
#         x = x.flatten(2)
#         x = self.fc1_spike(x)
#         x = self.fc1_conv(x)
#         x = self.fc1_bn(x).reshape(B, self.c_hidden, N).contiguous()
#         x = self.fc2_spike(x)
#         x = self.fc2_conv(x)
#         x = self.fc2_bn(x).reshape(B, C, H, W).contiguous()
#
#         return x
#
# class Block(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
#
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
#         self.norm = LayerNorm(dim, eps=1e-6)
#         # self.norm1 = LayerNorm(dim, eps=1e-6)
#
#         self.pwconv1 = nn.Linear(dim, 2 * dim)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(2 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.spike = Multispike_norm()
#         self.spike1 = Multispike_norm()
#         self.spike2 = Multispike_norm()
#     def forward(self, x):
#         input = x
#         x = self.spike(x)
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.spike1(x)
#         x = self.pwconv1(x)
#         # x = self.act(x)
#         x = self.spike2(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#
#         x = input + self.drop_path(x)
#         return x
#
#
# class Mix(nn.Module):
#     def __init__(self, m=-0.80):
#         super(Mix, self).__init__()
#         self.w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
#         self.mix_block = nn.Sigmoid()
#
#     def forward(self, fea1, fea2):
#         mix_factor = self.mix_block(self.w)
#         out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
#         return out
#
#
# class Attention(nn.Module):
#     def __init__(self, channel, b=1, gamma=2):
#         super(Attention, self).__init__()
#
#         # 全局平均池化
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         # 计算卷积核大小
#         t = int(abs((math.log(channel, 2) + b) / gamma))
#         k = t if t % 2 else t + 1
#
#         # 1D卷积操作
#         self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
#
#         # 1x1卷积操作
#         self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
#         self.spike = Multispike_norm()
#         self.spike1 = Multispike_norm()
#         # 激活函数
#         self.sigmoid = nn.Sigmoid()
#
#         # Mix模块
#         self.mix = Mix()
#
#     def forward(self, input):
#         # 通过全局平均池化得到通道描述
#         x = self.avg_pool(input)
#         x = self.spike(x)
#         # 计算1D卷积的特征
#         x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
#
#         # 计算1x1卷积的特征
#         x2 = self.fc(x).squeeze(-1).transpose(-1, -2)
#
#         # 使用矩阵乘法对特征进行融合
#         out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
#
#         # Sigmoid 激活
#         out1 = self.sigmoid(out1)
#
#         # 再次使用矩阵乘法
#         out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
#
#         # Sigmoid 激活
#         out2 = self.sigmoid(out2)
#
#         # 使用Mix模块进行特征融合
#         out = self.mix(out1, out2)
#         out = self.spike1(out)
#         # 通过卷积进行特征转换
#         out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#
#         # 最后一个Sigmoid激活
#         out = self.sigmoid(out)
#
#         return input * out

class build_spike(nn.Module):
    def __init__(self, num_classes, block=4, return_f=False, resnet=False):
        super(build_spike, self).__init__()
        self.return_f = return_f
        if resnet:
            convnext_name = "resnet101"
            print('using model_type: {} as a backbone'.format(convnext_name))
            self.in_planes = 2048
            self.convnext = Resnet(pretrained=True)
        else:
            convnext_name = "metaspikformer_8_512"
            print('using model_type: {} as a backbone'.format(convnext_name))
            if '1024' in convnext_name:
                self.in_planes = 1024
            elif '1536' in convnext_name:
                self.in_planes = 1536
            elif '2048' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 360
            self.convnext = create_model('metaspikformer_8_512',pretrained=False)

            # self.mlla_block = MLLABlockSNN(dim=360, input_resolution=576)
            #self.block = Block(dim = 360)

    def forward(self, x):
        # -- backbone feature extractor
        gap_feature, part_features,_ = self.convnext(x)
        gap_feature = gap_feature.mean([-2, -1])
        # pfeat = part_features.flatten(2)  # (bs, c, h*w)
        # # pfeat_align = self.mlla_block(pfeat)

        # gap_feature = part_features.mean([-2, -1])
        # import pdb;
        # pdb.set_trace()
        # print(gap_feature.shape)
        # print(part_features.shape)
        # gap_feature = self.convnext(x)
        # -- Training
        if self.training:

            return gap_feature, part_features

        else:

            # y = torch.cat([y, ffeature], dim=2)
            pass
        return gap_feature, part_features

def make_spike_model(num_class, block=4, return_f=False, resnet=False):
    print('===========building convnext===========')
    model = build_spike(num_class, block=block, return_f=return_f, resnet=resnet)
    return model
