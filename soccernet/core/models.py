from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    padding = (int((kernel_size - 1) / 2), int((kernel_size - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu': return nn.ReLU(inplace)
    elif act_type == 'lrelu': return nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu': return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    raise NotImplementedError(f'Activation {act_type} not found')

class ESA(nn.Module):
    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = F.interpolate(self.conv3(v_max), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        c4 = self.conv4(c3 + self.conv_f(c1_))
        return x * self.sigmoid(c4)

class RLFB(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, esa_channels=16):
        super(RLFB, self).__init__()
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels
        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)
        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)
        self.act = activation('lrelu')

    def forward(self, x):
        out = self.act(self.c1_r(x))
        out = self.act(self.c2_r(out))
        out = self.act(self.c3_r(out))
        return self.esa(self.c5(out + x))

class RLFN_S(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_channels=48, upscale=4):
        super(RLFN_S, self).__init__()
        self.conv_1 = conv_layer(in_channels, feature_channels, 3)
        self.blocks = nn.Sequential(*[RLFB(feature_channels) for _ in range(6)])
        self.conv_2 = conv_layer(feature_channels, feature_channels, 3)
        self.upsampler = nn.Sequential(
            nn.Conv2d(feature_channels, out_channels * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        feat = self.conv_1(x)
        out = self.conv_2(self.blocks(feat)) + feat
        return self.upsampler(out)