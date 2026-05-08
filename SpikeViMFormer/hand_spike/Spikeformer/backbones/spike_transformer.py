# # # from visualizer import get_local
from collections import OrderedDict
# #
import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from timm.models.layers import DropPath
from timm.models.registry import register_model
import torch.nn.functional as F
from functools import partial
from spikingjelly.clock_driven import layer
from SpikeViMFormer.hand_convnext.ConvNext.backbones.model_convnext import LayerNorm
from .Qtrick import Multispike_norm
from mmengine.logging import print_log
from mmengine.runner import CheckpointLoader
class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class RepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            expansion_ratio=2,
            act2_layer=nn.Identity,
            bias=False,
            kernel_size=7,
            padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = Multispike_norm()
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.spike2 = Multispike_norm()
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.spike1(x)

        x = self.bn1(self.pwconv1(x))

        x = self.spike2(x)

        x = self.dwconv(x)
        x = self.bn2(self.pwconv2(x))
        return x


class SepConv_Spike(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            expansion_ratio=2,
            act2_layer=nn.Identity,
            bias=False,
            kernel_size=7,
            padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = Multispike_norm()
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike2 = Multispike_norm()
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels,
                      bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = Multispike_norm()
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = self.spike1(x)

        x = self.pwconv1(x)

        x = self.spike2(x)

        x = self.dwconv(x)

        x = self.spike3(x)

        x = self.pwconv2(x)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv(dim=dim)

        self.mlp_ratio = mlp_ratio

        self.spike1 = Multispike_norm()
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = Multispike_norm()
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x)).reshape(B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)
        x = self.bn2(self.conv2(x)).reshape(B, C, H, W)
        x = x_feat + x

        return x


class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim)

        self.mlp_ratio = mlp_ratio

        self.spike1 = Multispike_norm()
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = Multispike_norm()
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x)).reshape(B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)
        x = self.bn2(self.conv2(x)).reshape(B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = Multispike_norm()

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = Multispike_norm()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2)
        x = self.fc1_spike(x)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).reshape(B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(B, C, H, W).contiguous()

        return x


class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.head_spike = Multispike_norm()

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = Multispike_norm()

        self.k_spike = Multispike_norm()

        self.v_spike = Multispike_norm()

        self.attn_spike = Multispike_norm(Vth=0.5)

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )

        # self.proj_conv = nn.Sequential(
        #     nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        # )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        x = self.head_spike(x)

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = self.q_spike(q)
        q = q.flatten(2)
        q = (
            q.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        k = self.k_spike(k)
        k = k.flatten(2)
        k = (
            k.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        v = self.v_spike(v)
        v = v.flatten(2)
        v = (
            v.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(2, 3).reshape(B, C, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(B, C, H, W)
        x = self.proj_conv(x).reshape(B, C, H, W)

        return x


class MS_Attention_linear(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
            lamda_ratio=1,
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio

        self.head_spike = Multispike_norm()

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = Multispike_norm()

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.k_spike = Multispike_norm()

        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim * lamda_ratio), 1, 1, bias=False),
                                    nn.BatchNorm2d(int(dim * lamda_ratio)))

        self.v_spike = Multispike_norm()

        self.attn_spike = Multispike_norm()

        # self.proj_conv = nn.Sequential(
        #     RepConv(dim*lamda_ratio, dim, bias=False), nn.BatchNorm2d(dim)
        # )

        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim * lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        C_v = int(C * self.lamda_ratio)

        x = self.head_spike(x)

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = self.q_spike(q)
        q = q.flatten(2)
        q = (
            q.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        k = self.k_spike(k)
        k = k.flatten(2)
        k = (
            k.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        v = self.v_spike(v)
        v = v.flatten(2)
        v = (
            v.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C_v // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        # import pdb;
        # pdb.set_trace()
        x = q @ k.transpose(-2, -1)
        x = (x @ v) * (self.scale * 2)

        x = x.transpose(2, 3).reshape(B, C_v, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(B, C_v, H, W)
        x = self.proj_conv(x).reshape(B, C, H, W)

        return x


class MS_Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class MS_Block_Spike_SepConv(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
            init_values=1e-6
    ):
        super().__init__()

        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)

        self.attn = MS_Attention_linear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            lamda_ratio=4,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # self.layer_scale1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        # self.layer_scale2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        # self.layer_scale3 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = x + self.conv(x)
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class MS_DownSampling(nn.Module):
    def __init__(
            self,
            in_channels=2,
            embed_dims=256,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=True,
            T=None,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = Multispike_norm()

    def forward(self, x):

        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x)
        x = self.encode_bn(x)

        return x

class Spiking_vit_MetaFormer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dim=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        pretrained=False,
        init_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),

        T=4,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        self.init_cfg = init_cfg
        # embed_dim = [64, 128, 256, 512]
        self.freeze_bn_ = not norm_cfg['requires_grad']
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,

        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,

                )
                for j in range(6)
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,

        )

        self.block4 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,

                )
                for j in range(2)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            for name, param in m.named_parameters():
                param.requires_grad = True
    def init_weights(self):
        # logger = MMlogger.get_current_instance()
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                # 使用mmseg保存的checkpoint中包含backbone, neck, decode_head三个部分
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            # import pdb; pdb.set_trace()
            # for K in state_dict.keys():
            #     print(K)
            info = self.load_state_dict(state_dict, strict=False)

            print_log(info)
            print_log("--------------Successfully load checkpoint for BACKNONE------------")
            print_log("Time step: {:}".format(self.T))
    def forward_features(self, x):
        # import pdb;
        # pdb.set_trace()
        B, TC, H, W = x.shape
        x = x.reshape(B, self.T, TC // self.T, H, W)
        x = self.downsample1_1(x.flatten(0, 1))  # [Conv + BN]
        for blk in self.ConvBlock1_1:
            x = blk(x)

        x1 = x
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x2 = x
        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)

        for blk in self.ConvBlock2_2:
            x = blk(x)

        x3 = x
        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        x4 = x
        # print("x1", x1.shape) # torch.Size([12, 32, 256, 256])
        # print("x2", x2.shape) # torch.Size([12, 64, 128, 128])
        # print("x3", x3.shape) # torch.Size([12, 128, 64, 64])
        # print("x4", x4.shape) # torch.Size([12, 360, 32, 32])

        # if self.decode_mode == 'QTrick':
        #     return [x1, x2, x3, x4]
        return x, [x1, x2, x3, x4]# T,B,C,N

    def forward(self, x):
        x, feat = self.forward_features(x)  # B,C,H,W
        # print(x.shape)
        # import pdb; pdb.set_trace()
        # return (x.mean([-2, -1])), x,feat
        return x, x,feat


# @register_model
# def metaspikformer_8_512(**kwargs):
#     model = Spiking_vit_MetaFormer(
#         img_size_h=384,
#         img_size_w=384,
#         patch_size=16,
#         embed_dim=[48, 96, 192, 240],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=8,
#         sr_ratios=1,
#         T=1,
#         init_cfg = dict(
#             type = 'Pretrained',
#             checkpoint = '/home/hk/PAPER/SNN-main/SNN-main/V3_10.0M_1x4.pth'
#         ),
#         **kwargs,
#     )
#     model.init_weights()
#     return model

@register_model
def metaspikformer_8_512(**kwargs):
    model = Spiking_vit_MetaFormer(
        img_size_h=384,
        img_size_w=384,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        T=1,
        init_cfg = dict(
            type = 'Pretrained',
            checkpoint = '/data1/CZW/SNN-main/SNN-main/V3_19.0M_1x4.pth'
        ),
        **kwargs,
    )
    model.init_weights()
    return model
#
# #
# #
# # from timm.models import create_model
# # if __name__ == '__main__':
# #     model = create_model('metaspikformer_8_512',pretrained=False,num_classes=1000)
# #     # checkpoint = torch.load("55M_kd.pth", map_location="cpu")
# #     # model.load_state_dict(checkpoint)
# # #
# import math
# from collections import OrderedDict
# from functools import partial
# import torch
# import torch.nn as nn
# import torch
# import torch.nn as nn
# from mmengine import print_log
# from mmengine.runner import CheckpointLoader
# from spikingjelly.clock_driven import layer
# from timm.models.layers import to_2tuple, trunc_normal_, DropPath
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
# from einops.layers.torch import Rearrange
# import torch.nn.functional as F
# from timm.models.vision_transformer import PatchEmbed, Block
#

# import copy
# from torchvision import transforms
# import matplotlib.pyplot as plt
# import torch.nn as nn
#
# # timestep 1x4
# T = 8
#
#
# class multispike(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, lens=T):
#         ctx.save_for_backward(input)
#         ctx.lens = lens
#         return torch.floor(torch.clamp(input, 0, lens) + 0.5)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         temp1 = 0 < input
#         temp2 = input < ctx.lens
#         return grad_input * temp1.float() * temp2.float(), None
#
#
# class Multispike(nn.Module):
#     def __init__(self, spike=multispike, norm=T):
#         super().__init__()
#         self.lens = norm
#         self.spike = spike
#         self.norm = norm
#
#     def forward(self, inputs):
#         return self.spike.apply(inputs) / self.norm
#
#
# def MS_conv_unit(in_channels, out_channels, kernel_size=1, padding=0, groups=1):
#     return nn.Sequential(
#         layer.SeqToANNContainer(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=True),
#             nn.BatchNorm2d(out_channels)
#         )
#     )
#
#
# class MS_ConvBlock(nn.Module):
#     def __init__(self, dim,
#                  mlp_ratio=4.0):
#         super().__init__()
#
#         self.neuron1 = Multispike()
#         self.conv1 = MS_conv_unit(dim, dim * mlp_ratio, 3, 1)
#
#         self.neuron2 = Multispike()
#         self.conv2 = MS_conv_unit(dim * mlp_ratio, dim, 3, 1)
#
#     def forward(self, x, mask=None):
#         short_cut = x
#         x = self.neuron1(x)
#         x = self.conv1(x)
#         x = self.neuron2(x)
#         x = self.conv2(x)
#         x = x + short_cut
#         return x
#
#
# class MS_MLP(nn.Module):
#     def __init__(
#             self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
#     ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
#         self.fc1_bn = nn.BatchNorm1d(hidden_features)
#         self.fc1_lif = Multispike()
#
#         self.fc2_conv = nn.Conv1d(
#             hidden_features, out_features, kernel_size=1, stride=1
#         )
#         self.fc2_bn = nn.BatchNorm1d(out_features)
#         self.fc2_lif = Multispike()
#
#         self.c_hidden = hidden_features
#         self.c_output = out_features
#
#     def forward(self, x):
#         T, B, C, N = x.shape
#
#         x = self.fc1_lif(x)
#         x = self.fc1_conv(x.flatten(0, 1))
#         x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
#
#         x = self.fc2_lif(x)
#         x = self.fc2_conv(x.flatten(0, 1))
#         x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()
#
#         return x
#
#
# class RepConv(nn.Module):
#     def __init__(
#             self,
#             in_channel,
#             out_channel,
#             bias=False,
#     ):
#         super().__init__()
#         # TODO in_channel-> 2*in_channel->in_channel
#         self.conv1 = nn.Sequential(nn.Conv1d(in_channel, int(in_channel * 1.5), kernel_size=1, stride=1, bias=False),
#                                    nn.BatchNorm1d(int(in_channel * 1.5)))
#         self.conv2 = nn.Sequential(nn.Conv1d(int(in_channel * 1.5), out_channel, kernel_size=1, stride=1, bias=False),
#                                    nn.BatchNorm1d(out_channel))
#
#     def forward(self, x):
#         return self.conv2(self.conv1(x))
#
#
# class RepConv2(nn.Module):
#     def __init__(
#             self,
#             in_channel,
#             out_channel,
#             bias=False,
#     ):
#         super().__init__()
#         # TODO in_channel-> 2*in_channel->in_channel
#         self.conv1 = nn.Sequential(nn.Conv1d(in_channel, int(in_channel), kernel_size=1, stride=1, bias=False),
#                                    nn.BatchNorm1d(int(in_channel)))
#         self.conv2 = nn.Sequential(nn.Conv1d(int(in_channel), out_channel, kernel_size=1, stride=1, bias=False),
#                                    nn.BatchNorm1d(out_channel))
#
#     def forward(self, x):
#         return self.conv2(self.conv1(x))
#
#
# class MS_Attention_Conv_qkv_id(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#         self.dim = dim
#         self.num_heads = num_heads
#         self.scale = 0.125
#         self.sr_ratio = sr_ratio
#
#         self.head_lif = Multispike()
#
#         # track 1: split convs
#         self.q_conv = nn.Sequential(RepConv(dim, dim), nn.BatchNorm1d(dim))
#         self.k_conv = nn.Sequential(RepConv(dim, dim), nn.BatchNorm1d(dim))
#         self.v_conv = nn.Sequential(RepConv(dim, dim * sr_ratio), nn.BatchNorm1d(dim * sr_ratio))
#
#         # track 2: merge (prefer) NOTE: need `chunk` in forward
#         # self.qkv_conv = nn.Sequential(RepConv(dim,dim * 3), nn.BatchNorm2d(dim * 3))
#
#         self.q_lif = Multispike()
#
#         self.k_lif = Multispike()
#
#         self.v_lif = Multispike()
#
#         self.attn_lif = Multispike()
#
#         self.proj_conv = nn.Sequential(RepConv(sr_ratio * dim, dim), nn.BatchNorm1d(dim))
#
#     def forward(self, x):
#         T, B, C, N = x.shape
#
#         x = self.head_lif(x)
#
#         x_for_qkv = x.flatten(0, 1)
#         q_conv_out = self.q_conv(x_for_qkv).reshape(T, B, C, N)
#
#         q_conv_out = self.q_lif(q_conv_out)
#
#         q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
#                                                                                                        4)
#
#         k_conv_out = self.k_conv(x_for_qkv).reshape(T, B, C, N)
#
#         k_conv_out = self.k_lif(k_conv_out)
#
#         k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
#                                                                                                        4)
#
#         v_conv_out = self.v_conv(x_for_qkv).reshape(T, B, self.sr_ratio * C, N)
#
#         v_conv_out = self.v_lif(v_conv_out)
#
#         v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, self.sr_ratio * C // self.num_heads).permute(
#             0, 1, 3, 2,
#             4)
#
#         x = k.transpose(-2, -1) @ v
#         x = (q @ x) * self.scale
#         x = x.transpose(3, 4).reshape(T, B, self.sr_ratio * C, N)
#         x = self.attn_lif(x)
#
#         x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, N)
#         return x
#
#
# class MS_Block(nn.Module):
#     def __init__(
#             self,
#             dim,
#             choice,
#             num_heads,
#             mlp_ratio=4.0,
#             qkv_bias=False,
#             qk_scale=None,
#             drop=0.0,
#             attn_drop=0.0,
#             drop_path=0.0,
#             norm_layer=nn.LayerNorm,
#             sr_ratio=1, init_values=1e-6, finetune=False,
#     ):
#         super().__init__()
#         self.model = choice
#         if self.model == "base":
#             self.rep_conv = RepConv2(dim, dim)  # if have param==83M
#         self.lif = Multispike()
#         self.attn = MS_Attention_Conv_qkv_id(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#             sr_ratio=sr_ratio,
#         )
#         self.finetune = finetune
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
#
#         if self.finetune:
#             self.layer_scale1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
#             self.layer_scale2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
#
#     def forward(self, x):
#         T, B, C, N = x.shape
#         if self.model == "base":
#             x = x + self.rep_conv(self.lif(x).flatten(0, 1)).reshape(T, B, C, N)
#         # TODO: need channel-wise layer scale, init as 1e-6
#         if self.finetune:
#             x = x + self.drop_path(self.attn(x) * self.layer_scale1.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
#             x = x + self.drop_path(self.mlp(x) * self.layer_scale2.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
#         else:
#             x = x + self.attn(x)
#             x = x + self.mlp(x)
#         return x
#
#
# class MS_DownSampling(nn.Module):
#     def __init__(
#             self,
#             in_channels=2,
#             embed_dims=256,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             first_layer=True,
#     ):
#         super().__init__()
#
#         self.encode_conv = nn.Conv2d(
#             in_channels,
#             embed_dims,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#         )
#
#         self.encode_bn = nn.BatchNorm2d(embed_dims)
#         if not first_layer:
#             self.encode_lif = Multispike()
#
#     def forward(self, x):
#         T, B, _, _, _ = x.shape
#         if hasattr(self, "encode_lif"):
#             x = self.encode_lif(x)
#         x = self.encode_conv(x.flatten(0, 1))
#         _, _, H, W = x.shape
#         x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
#         return x
#
#
# class Spikformer(nn.Module):
#     def __init__(self, T=1,
#                  choice=None,
#                  img_size_h=224,
#                  img_size_w=224,
#                  patch_size=16,
#                  embed_dim=[128, 256, 512, 640],
#                  num_heads=8,
#                  mlp_ratios=4,
#                  in_channels=3,
#                  qk_scale=None,
#                  drop_rate=0.0,
#                  attn_drop_rate=0.0,
#                  drop_path_rate=0.1,
#                  num_classes=1000,
#                  qkv_bias=False,
#                  norm_layer=partial(nn.LayerNorm, eps=1e-6),  # norm_layer=nn.LayerNorm shaokun
#                  depths=8,
#                  sr_ratios=1,
#                  mlp_ratio=4.,
#                  nb_classes=1000,
#                  kd=True,
#                  init_cfg=None,
#                  **kwargs
#                  ):
#         super().__init__()
#         self.init_cfg = init_cfg
#         ### MAE encoder spikformer
#         self.T = T
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
#         dpr = [
#             x.item() for x in torch.linspace(0, drop_path_rate, depths)
#         ]  # stochastic depth decay rule
#         self.downsample1_1 = MS_DownSampling(
#             in_channels=in_channels,
#             embed_dims=embed_dim[0] // 2,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             first_layer=True,
#         )
#
#         self.ConvBlock1_1 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
#         )
#
#         self.downsample1_2 = MS_DownSampling(
#             in_channels=embed_dim[0] // 2,
#             embed_dims=embed_dim[0],
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             first_layer=False,
#         )
#
#         self.ConvBlock1_2 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
#         )
#
#         self.downsample2 = MS_DownSampling(
#             in_channels=embed_dim[0],
#             embed_dims=embed_dim[1],
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             first_layer=False,
#         )
#
#         self.ConvBlock2_1 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
#         )
#
#         self.ConvBlock2_2 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
#         )
#
#         self.downsample3 = MS_DownSampling(
#             in_channels=embed_dim[1],
#             embed_dims=embed_dim[2],
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             first_layer=False,
#         )
#
#         self.block3 = nn.ModuleList(
#             [
#                 MS_Block(
#                     dim=embed_dim[2],
#                     choice=choice,
#                     num_heads=num_heads,
#                     mlp_ratio=mlp_ratios,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop_rate,
#                     attn_drop=attn_drop_rate,
#                     drop_path=dpr[j],
#                     norm_layer=norm_layer,
#                     sr_ratio=sr_ratios,
#                     finetune=True,
#                 )
#                 for j in range(depths)
#             ]
#         )
#         self.head = nn.Linear(embed_dim[2], nb_classes)
#         self.lif = Multispike(norm=1)
#         self.kd = kd
#         if self.kd:
#             self.head_kd = (
#                 nn.Linear(embed_dim[-1], num_classes)
#                 if num_classes > 0
#                 else nn.Identity()
#             )
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def init_weights(self):
#         # logger = MMlogger.get_current_instance()
#         if self.init_cfg is None:
#             print_log(f'No pre-trained weights for '
#                       f'{self.__class__.__name__}, '
#                       f'training start from scratch')
#         else:
#             assert 'checkpoint' in self.init_cfg, f'Only support ' \
#                                                   f'specify `Pretrained` in ' \
#                                                   f'`init_cfg` in ' \
#                                                   f'{self.__class__.__name__} '
#             ckpt = CheckpointLoader.load_checkpoint(
#                 self.init_cfg['checkpoint'], logger=None, map_location='cpu')
#             if 'state_dict' in ckpt:
#                 _state_dict = ckpt['state_dict']
#             elif 'model' in ckpt:
#                 _state_dict = ckpt['model']
#             else:
#                 _state_dict = ckpt
#             state_dict = OrderedDict()
#             for k, v in _state_dict.items():
#                 # 使用mmseg保存的checkpoint中包含backbone, neck, decode_head三个部分
#                 if k.startswith('backbone.'):
#                     state_dict[k[9:]] = v
#                 else:
#                     state_dict[k] = v
#             # import pdb; pdb.set_trace()
#             # for K in state_dict.keys():
#             #     print(K)
#             info = self.load_state_dict(state_dict, strict=False)
#
#             print_log(info)
#             print_log("--------------Successfully load checkpoint for BACKNONE------------")
#             print_log("Time step: {:}".format(self.T))
#
#     def forward_encoder(self, x):
#         x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
#
#         x = self.downsample1_1(x)
#
#         for blk in self.ConvBlock1_1:
#             x = blk(x)
#
#         x = self.downsample1_2(x)
#         for blk in self.ConvBlock1_2:
#             x = blk(x)
#
#         x = self.downsample2(x)
#
#         for blk in self.ConvBlock2_1:
#             x = blk(x)
#
#         for blk in self.ConvBlock2_2:
#             x = blk(x)
#         x = self.downsample3(x)
#         x = x.flatten(3)  # T,B,C,N
#
#         for blk in self.block3:
#             x = blk(x)
#         x = x.mean(0)
#         return x
#
#     def forward(self, imgs):
#         x = self.forward_encoder(imgs)
#         # print(x.shape)
#         B,C,_ = x.shape
#         # H = math.sqrt(WH)
#         x = x.view(B,C,24,24)
#         return x,x,x
#
#
# def spikformer12_512(**kwargs):
#     model = Spikformer(
#         T=1,
#         choice="base",
#         img_size_h=32,
#         img_size_w=32,
#         patch_size=16,
#         embed_dim=[128, 256, 512],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=12,
#         **kwargs)
#     return model
#
# @register_model
# def metaspikformer_8_512(**kwargs):
#     model = Spikformer(
#         T=1,
#         choice="large",
#         img_size_h=384,
#         img_size_w=384,
#         patch_size=16,
#         embed_dim=[196, 384, 768],
#         # embed_dim=[128, 256, 512],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=12,
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='/home/hk/PAPER/SNN-main/SNN-main/171M-1x8_86_2.pth'
#         ),
#         **kwargs,
#     )
#     model.init_weights()
#     return model
#
#
# # from timm.models import create_model
# # if __name__ == '__main__':
# #     model = create_model('metaspikformer_8_768',pretrained=False,num_classes=1000)
# #     checkpoint = torch.load("55M_kd.pth", map_location="cpu")
# #     model.load_state_dict(checkpoint)
# #
# # from visualizer import get_local
# from collections import OrderedDict
# #
# import torch
# import torchinfo
# import torch.nn as nn
# from mmengine.runner import CheckpointLoader
# from spikingjelly.clock_driven.neuron import (
#     MultiStepParametricLIFNode,
#     MultiStepLIFNode,
# )
# from .Qtrick import Multispike_norm
# from mmengine.logging import print_log
# from spikingjelly.clock_driven import layer
# from timm.models.layers import to_2tuple, trunc_normal_, DropPath
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
# from einops.layers.torch import Rearrange
# import torch.nn.functional as F
# from functools import partial
# from .Qtrick_architecture.clock_driven.neuron import Q_IFNode
# from .Qtrick_architecture.clock_driven.surrogate import Quant,Quant4
#
# class BNAndPadLayer(nn.Module):
#     def __init__(
#         self,
#         pad_pixels,
#         num_features,
#         eps=1e-5,
#         momentum=0.1,
#         affine=True,
#         track_running_stats=True,
#     ):
#         super(BNAndPadLayer, self).__init__()
#         self.bn = nn.BatchNorm2d(
#             num_features, eps, momentum, affine, track_running_stats
#         )
#         self.pad_pixels = pad_pixels
#
#     def forward(self, input):
#         output = self.bn(input)
#         if self.pad_pixels > 0:
#             if self.bn.affine:
#                 pad_values = (
#                     self.bn.bias.detach()
#                     - self.bn.running_mean
#                     * self.bn.weight.detach()
#                     / torch.sqrt(self.bn.running_var + self.bn.eps)
#                 )
#             else:
#                 pad_values = -self.bn.running_mean / torch.sqrt(
#                     self.bn.running_var + self.bn.eps
#                 )
#             output = F.pad(output, [self.pad_pixels] * 4)
#             pad_values = pad_values.view(1, -1, 1, 1)
#             output[:, :, 0 : self.pad_pixels, :] = pad_values
#             output[:, :, -self.pad_pixels :, :] = pad_values
#             output[:, :, :, 0 : self.pad_pixels] = pad_values
#             output[:, :, :, -self.pad_pixels :] = pad_values
#         return output
#
#     @property
#     def weight(self):
#         return self.bn.weight
#
#     @property
#     def bias(self):
#         return self.bn.bias
#
#     @property
#     def running_mean(self):
#         return self.bn.running_mean
#
#     @property
#     def running_var(self):
#         return self.bn.running_var
#
#     @property
#     def eps(self):
#         return self.bn.eps
#
#
# class RepConv(nn.Module):
#     def __init__(
#         self,
#         in_channel,
#         out_channel,
#         bias=False,
#     ):
#         super().__init__()
#         # hidden_channel = in_channel
#         conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
#         bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
#         conv3x3 = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
#             nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
#             nn.BatchNorm2d(out_channel),
#         )
#
#         self.body = nn.Sequential(conv1x1, bn, conv3x3)
#
#     def forward(self, x):
#         return self.body(x)
#
#
# class SepConv(nn.Module):
#     r"""
#     Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
#     """
#
#     def __init__(
#         self,
#         dim,
#         expansion_ratio=2,
#         act2_layer=nn.Identity,
#         bias=False,
#         kernel_size=7,
#         padding=3,
#     ):
#         super().__init__()
#         med_channels = int(expansion_ratio * dim)
#         self.lif1 = Multispike_norm()
#         self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
#         self.bn1 = nn.BatchNorm2d(med_channels)
#         self.lif2 = Multispike_norm()
#         self.dwconv = nn.Conv2d(
#             med_channels,
#             med_channels,
#             kernel_size=kernel_size,
#             padding=padding,
#             groups=med_channels,
#             bias=bias,
#         )  # depthwise conv
#         self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
#         self.bn2 = nn.BatchNorm2d(dim)
#
#     def forward(self, x):
#         T, B, C, H, W = x.shape
#         x = self.lif1(x)
#         x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
#         x = self.lif2(x)
#         x = self.dwconv(x.flatten(0, 1))
#         x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)
#         return x
#
#
# class MS_ConvBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         mlp_ratio=4.0,
#     ):
#         super().__init__()
#
#         self.Conv = SepConv(dim=dim)
#         # self.Conv = MHMC(dim=dim)
#
#         self.lif1 = Multispike_norm()
#         self.conv1 = nn.Conv2d(
#             dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
#         )
#         # self.conv1 = RepConv(dim, dim*mlp_ratio)
#         self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
#         self.lif2 = Multispike_norm()
#         self.conv2 = nn.Conv2d(
#             dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
#         )
#         # self.conv2 = RepConv(dim*mlp_ratio, dim)
#         self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进
#
#     def forward(self, x):
#         T, B, C, H, W = x.shape
#
#         x = self.Conv(x) + x
#         x_feat = x
#         x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, 4 * C, H, W)
#         x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
#         x = x_feat + x
#
#         return x
#
#
# class MS_MLP(nn.Module):
#     def __init__(
#         self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
#     ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         # self.fc1 = linear_unit(in_features, hidden_features)
#         self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
#         self.fc1_bn = nn.BatchNorm1d(hidden_features)
#         self.fc1_lif = Multispike_norm()
#
#         # self.fc2 = linear_unit(hidden_features, out_features)
#         self.fc2_conv = nn.Conv1d(
#             hidden_features, out_features, kernel_size=1, stride=1
#         )
#         self.fc2_bn = nn.BatchNorm1d(out_features)
#         self.fc2_lif = Multispike_norm()
#         # self.drop = nn.Dropout(0.1)
#
#         self.c_hidden = hidden_features
#         self.c_output = out_features
#
#     def forward(self, x):
#         T, B, C, H, W = x.shape
#         N = H * W
#         x = x.flatten(3)
#         x = self.fc1_lif(x)
#         x = self.fc1_conv(x.flatten(0, 1))
#         x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
#
#         x = self.fc2_lif(x)
#         x = self.fc2_conv(x.flatten(0, 1))
#         x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
#
#         return x
#
#
# class MS_Attention_RepConv_qkv_id(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads=8,
#         qkv_bias=False,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         sr_ratio=1,
#     ):
#         super().__init__()
#         assert (
#             dim % num_heads == 0
#         ), f"dim {dim} should be divided by num_heads {num_heads}."
#         self.dim = dim
#         self.num_heads = num_heads
#         self.scale = 0.125
#
#         self.head_lif = Multispike_norm()
#
#         self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
#
#         self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
#
#         self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
#
#         self.q_lif = Multispike_norm()
#
#         self.k_lif = Multispike_norm()
#
#         self.v_lif = Multispike_norm()
#
#         self.attn_lif = Multispike_norm()
#
#         self.proj_conv = nn.Sequential(
#             RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
#         )
#
#     def forward(self, x):
#         T, B, C, H, W = x.shape
#         N = H * W
#
#         x = self.head_lif(x)
#
#         q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
#         k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
#         v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
#
#         q = self.q_lif(q).flatten(3)
#         q = (
#             q.transpose(-1, -2)
#             .reshape(T, B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 1, 3, 2, 4)
#             .contiguous()
#         )
#
#         k = self.k_lif(k).flatten(3)
#         k = (
#             k.transpose(-1, -2)
#             .reshape(T, B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 1, 3, 2, 4)
#             .contiguous()
#         )
#
#         v = self.v_lif(v).flatten(3)
#         v = (
#             v.transpose(-1, -2)
#             .reshape(T, B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 1, 3, 2, 4)
#             .contiguous()
#         )
#
#         x = k.transpose(-2, -1) @ v
#         x = (q @ x) * self.scale
#
#         x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
#         x = self.attn_lif(x).reshape(T, B, C, H, W)
#         x = x.reshape(T, B, C, H, W)
#         x = x.flatten(0, 1)
#         x = self.proj_conv(x).reshape(T, B, C, H, W)
#
#         return x
#
#
# class MS_Block(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=False,
#         qk_scale=None,
#         drop=0.0,
#         attn_drop=0.0,
#         drop_path=0.0,
#         norm_layer=nn.LayerNorm,
#         sr_ratio=1,
#     ):
#         super().__init__()
#
#         self.attn = MS_Attention_RepConv_qkv_id(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#             sr_ratio=sr_ratio,
#         )
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
#
#     def forward(self, x):
#         x = x + self.attn(x)
#         x = x + self.mlp(x)
#
#         return x
#
#
# class MS_DownSampling(nn.Module):
#     def __init__(
#         self,
#         in_channels=2,
#         embed_dims=256,
#         kernel_size=3,
#         stride=2,
#         padding=1,
#         first_layer=True,
#     ):
#         super().__init__()
#
#         self.encode_conv = nn.Conv2d(
#             in_channels,
#             embed_dims,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#         )
#
#         self.encode_bn = nn.BatchNorm2d(embed_dims)
#         if not first_layer:
#             self.encode_lif = Multispike_norm()
#
#     def forward(self, x):
#         T, B, _, _, _ = x.shape
#
#         if hasattr(self, "encode_lif"):
#             x = self.encode_lif(x)
#         x = self.encode_conv(x.flatten(0, 1))
#         _, _, H, W = x.shape
#         x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
#
#         return x
#
#
# class Spiking_vit_MetaFormer(nn.Module):
#     def __init__(
#         self,
#         img_size_h=128,
#         img_size_w=128,
#         patch_size=16,
#         in_channels=2,
#         num_classes=11,
#         embed_dim=[64, 128, 256],
#         num_heads=[1, 2, 4],
#         mlp_ratios=[4, 4, 4],
#         qkv_bias=False,
#         qk_scale=None,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.0,
#         norm_layer=nn.LayerNorm,
#         depths=[6, 8, 6],
#         sr_ratios=[8, 4, 2],
#         kd=False,
#         init_cfg=None,
#         **kwargs
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.depths = depths
#         self.T = 2
#         self.init_cfg = init_cfg
#         # embed_dim = [64, 128, 256, 512]
#
#         dpr = [
#             x.item() for x in torch.linspace(0, drop_path_rate, depths)
#         ]  # stochastic depth decay rule
#
#         self.downsample1_1 = MS_DownSampling(
#             in_channels=in_channels,
#             embed_dims=embed_dim[0] // 2,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             first_layer=True,
#         )
#
#         self.ConvBlock1_1 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
#         )
#
#         self.downsample1_2 = MS_DownSampling(
#             in_channels=embed_dim[0] // 2,
#             embed_dims=embed_dim[0],
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             first_layer=False,
#         )
#
#         self.ConvBlock1_2 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
#         )
#
#         self.downsample2 = MS_DownSampling(
#             in_channels=embed_dim[0],
#             embed_dims=embed_dim[1],
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             first_layer=False,
#         )
#
#         self.ConvBlock2_1 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
#         )
#
#         self.ConvBlock2_2 = nn.ModuleList(
#             [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
#         )
#
#         self.downsample3 = MS_DownSampling(
#             in_channels=embed_dim[1],
#             embed_dims=embed_dim[2],
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             first_layer=False,
#         )
#
#         self.block3 = nn.ModuleList(
#             [
#                 MS_Block(
#                     dim=embed_dim[2],
#                     num_heads=num_heads,
#                     mlp_ratio=mlp_ratios,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop_rate,
#                     attn_drop=attn_drop_rate,
#                     drop_path=dpr[j],
#                     norm_layer=norm_layer,
#                     sr_ratio=sr_ratios,
#                 )
#                 for j in range(6)
#             ]
#         )
#
#         self.downsample4 = MS_DownSampling(
#             in_channels=embed_dim[2],
#             embed_dims=embed_dim[3],
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             first_layer=False,
#         )
#
#         self.block4 = nn.ModuleList(
#             [
#                 MS_Block(
#                     dim=embed_dim[3],
#                     num_heads=num_heads,
#                     mlp_ratio=mlp_ratios,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop_rate,
#                     attn_drop=attn_drop_rate,
#                     drop_path=dpr[j],
#                     norm_layer=norm_layer,
#                     sr_ratio=sr_ratios,
#                 )
#                 for j in range(2)
#             ]
#         )
#
#         self.lif = Multispike_norm()
#         self.head = (
#             nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
#         )
#
#         self.kd = kd
#         if self.kd:
#             self.head_kd = (
#                 nn.Linear(embed_dim[3], num_classes)
#                 if num_classes > 0
#                 else nn.Identity()
#             )
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def init_weights(self):
#         # logger = MMlogger.get_current_instance()
#         if self.init_cfg is None:
#             print_log(f'No pre-trained weights for '
#                       f'{self.__class__.__name__}, '
#                       f'training start from scratch')
#         else:
#             assert 'checkpoint' in self.init_cfg, f'Only support ' \
#                                                   f'specify `Pretrained` in ' \
#                                                   f'`init_cfg` in ' \
#                                                   f'{self.__class__.__name__} '
#             ckpt = CheckpointLoader.load_checkpoint(
#                 self.init_cfg['checkpoint'], logger=None, map_location='cpu')
#             if 'state_dict' in ckpt:
#                 _state_dict = ckpt['state_dict']
#             elif 'model' in ckpt:
#                 _state_dict = ckpt['model']
#             else:
#                 _state_dict = ckpt
#             state_dict = OrderedDict()
#             for k, v in _state_dict.items():
#                 # 使用mmseg保存的checkpoint中包含backbone, neck, decode_head三个部分
#                 if k.startswith('backbone.'):
#                     state_dict[k[9:]] = v
#                 else:
#                     state_dict[k] = v
#             # import pdb; pdb.set_trace()
#             # for K in state_dict.keys():
#             #     print(K)
#             info = self.load_state_dict(state_dict, strict=False)
#
#             print_log(info)
#             print_log("--------------Successfully load checkpoint for BACKNONE------------")
#             print_log("Time step: {:}".format(self.T))
#
#     def forward_features(self, x):
#         x = self.downsample1_1(x)
#         for blk in self.ConvBlock1_1:
#             x = blk(x)
#         x = self.downsample1_2(x)
#         for blk in self.ConvBlock1_2:
#             x = blk(x)
#
#         x = self.downsample2(x)
#         for blk in self.ConvBlock2_1:
#             x = blk(x)
#         for blk in self.ConvBlock2_2:
#             x = blk(x)
#
#         x = self.downsample3(x)
#         for blk in self.block3:
#             x = blk(x)
#
#         x = self.downsample4(x)
#         for blk in self.block4:
#             x = blk(x)
#         return x  # T,B,C,N
#
#     def forward(self, x):
#         x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
#         x = self.forward_features(x)
#         x = x.mean(0)
#         # x = x.flatten(3).mean(3)
#         # x_lif = self.lif(x)
#         # x = self.head(x_lif).mean(0)
#         # if self.kd:
#         #     x_kd = self.head_kd(x_lif).mean(0)
#         #     if self.training:
#         #         return x, x_kd
#         #     else:
#         #         return (x + x_kd) / 2
#         return x,x,x
#
#
#
# # def metaspikformer_8_384(**kwargs):
# #     model = Spiking_vit_MetaFormer(
# #         img_size_h=224,
# #         img_size_w=224,
# #         patch_size=16,
# #         embed_dim=[96, 192, 384, 480],
# #         num_heads=8,
# #         mlp_ratios=4,
# #         in_channels=3,
# #         num_classes=1000,
# #         qkv_bias=False,
# #         norm_layer=partial(nn.LayerNorm, eps=1e-6),
# #         depths=8,
# #         sr_ratios=1,
# #         **kwargs,
# #     )
# #     return model
# #
# #
# # def metaspikformer_8_512(**kwargs):
# #     model = Spiking_vit_MetaFormer(
# #         img_size_h=224,
# #         img_size_w=224,
# #         patch_size=16,
# #         embed_dim=[128, 256, 512, 640],
# #         num_heads=8,
# #         mlp_ratios=4,
# #         in_channels=3,
# #         num_classes=1000,
# #         qkv_bias=False,
# #         norm_layer=partial(nn.LayerNorm, eps=1e-6),
# #         depths=8,
# #         sr_ratios=1,
# #         **kwargs,
# #     )
# #     return model
#
# @register_model
# def metaspikformer_8_512(**kwargs):
#     model = Spiking_vit_MetaFormer(
#         img_size_h=384,
#         img_size_w=384,
#         patch_size=16,
#         embed_dim=[128, 256, 512, 640],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=8,
#         sr_ratios=1,
#         init_cfg = dict(
#             type = 'Pretrained',
#             checkpoint = '/home/hk/PAPER/SNN-main/SNN-main/55M_kd.pth'
#         ),
#         **kwargs,
#     )
#     model.init_weights()
#     return model
# #
# from timm.models import create_model
# if __name__ == '__main__':
#     model = create_model('metaspikformer_8_768',pretrained=False,num_classes=1000)
#     checkpoint = torch.load("55M_kd.pth", map_location="cpu")
#     model.load_state_dict(checkpoint)

from timm.models import create_model

