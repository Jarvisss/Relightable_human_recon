""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import antialiased_cnns

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, inplace=False, norm_type='group'):
        super().__init__()
        
        activation_layer = nn.LeakyReLU(inplace)
        if norm_type == 'group':
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, out_channels),
                activation_layer,
            )
        elif norm_type == 'batch':
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                activation_layer,
            )
        elif norm_type == 'none':
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                activation_layer,
            )
    def forward(self, x):
        return self.single_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, inplace=False, norm_type='group', padding_mode='zeros'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        n_group = 32
        print(n_group, mid_channels, out_channels)
        activation_layer = nn.LeakyReLU(inplace)
        if norm_type == 'group':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
                nn.GroupNorm(n_group, mid_channels),
                activation_layer,
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
                nn.GroupNorm(n_group, out_channels),
                activation_layer
            )
        elif norm_type == 'batch':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
                nn.BatchNorm2d(mid_channels),
                activation_layer,
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
                nn.BatchNorm2d(out_channels),
                activation_layer
            )
        elif norm_type == 'none':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
                activation_layer,
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
                activation_layer
            )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, pool_type='blur', inplace=False, norm_type='group', conv_type='double', padding_mode='zeros'):
        super().__init__()
        pool_layer = nn.MaxPool2d(2) if pool_type == 'max' else antialiased_cnns.BlurPool(in_channels, stride=2)
        
        if conv_type == 'double':
            self.conv = nn.Sequential(
                pool_layer,
                DoubleConv(in_channels, out_channels, inplace=inplace, norm_type=norm_type, padding_mode=padding_mode)
            )
        else:
            self.conv = nn.Sequential(
                pool_layer,
                SingleConv(in_channels, out_channels, inplace=inplace, norm_type=norm_type)
            )
        

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, inplace=False, norm_type='group', conv_type='double', padding_mode='zeros'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            if conv_type=='double':
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, inplace=inplace, norm_type=norm_type, padding_mode=padding_mode)
            else:
                self.conv = SingleConv(in_channels, out_channels, inplace=inplace, norm_type=norm_type)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if conv_type=='double':
                self.conv = DoubleConv(in_channels, out_channels, inplace=inplace, norm_type=norm_type, padding_mode=padding_mode)
            else:
                self.conv = SingleConv(in_channels, out_channels, inplace=inplace, norm_type=norm_type)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, last_op=None):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.last_op = last_op
    def forward(self, x):
        out = self.conv(x)
        if self.last_op is not None:
            out2 = self.last_op(out)
            return out2
        return out
