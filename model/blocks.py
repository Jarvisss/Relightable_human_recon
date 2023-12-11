import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import functools


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False
        


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, 32)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer

def make_norm_layer(norm, channels):
    if norm=='bn':
        return nn.BatchNorm2d(channels, affine=True)
    elif norm=='in':
        return nn.InstanceNorm2d(channels, affine=True)
    else:
        return None


def warp_flow(source, flow, align_corners=True, mode='bilinear', mask=None, mask_value=-1):
    '''
    Warp a image x according to the given flow
    Input:
        x: (b, c, H, W)
        flow: (b, 2, H, W) # range [-w/2, w/2] [-h/2, h/2]
        mask: (b, 1, H, W)
    Ouput:
        y: (b, c, H, W)
    '''
    [b, c, h, w] = source.shape
    # mesh grid
    x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
    y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
    grid = torch.stack([x,y], dim=0)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

    grid = 2*grid - 1

    flow = 2* flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
    
    grid = (grid+flow).permute(0, 2, 3, 1)
    
    '''grid = grid + flow # in this way flow is -1 to 1
    '''
    # to (b, h, w, c) for F.grid_sample
    output = F.grid_sample(source, grid, mode=mode, padding_mode='zeros', align_corners=align_corners)

    if mask is not None:
        output = torch.where(mask>0.5, output, output.new_ones(1).mul_(mask_value))
    return output


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False, padding_mode='zeros'):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias, padding_mode=padding_mode)


class Siren(nn.Module):
    pass



class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch', inplace=False, padding_mode='zeros'):
        super(ConvBlock, self).__init__()
        self.inplace = inplace
        self.conv1 = conv3x3(in_planes, int(out_planes / 2), padding_mode=padding_mode)
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4), padding_mode=padding_mode)
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4), padding_mode=padding_mode)

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(self.inplace),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, self.inplace)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, self.inplace)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, self.inplace)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, inplace='False'):
        super(up_conv, self).__init__()
        self.inplace = inplace
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=self.inplace )
        )

    def forward(self, x):
        x = self.up(x)
        return x


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module

class ResEncoderBlock(nn.Module):
    """
    Define a encoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResEncoderBlock, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1), use_spect)
        bypass = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, 
                                       norm_layer(hidden_nc), nonlinearity, conv2,)
        
        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out

class ResDecoderBlock(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False):
        super(ResDecoderBlock, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        bypass = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, norm_layer(hidden_nc), nonlinearity, conv2,)

        self.shortcut = nn.Sequential(bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out

class ResBlock(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, output_nc=None, hidden_nc=None, kernel_size=3, padding=1, norm_type='bn', use_spectral_norm=False, learnable_shortcut=False):
        super(ResBlock, self).__init__()
        hidden_nc = in_features if hidden_nc is None else hidden_nc
        output_nc = in_features if output_nc is None else output_nc

        self.learnable_shortcut = True if in_features != output_nc else learnable_shortcut

        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_features, out_channels=hidden_nc, kernel_size=kernel_size,
                                padding=padding))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=hidden_nc, out_channels=output_nc, kernel_size=kernel_size,
                                padding=padding))
        else:
            self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=hidden_nc, kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(in_channels=hidden_nc, out_channels=output_nc, kernel_size=kernel_size, padding=padding)

        if self.learnable_shortcut:
            bypass = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_features, out_channels=output_nc, kernel_size=kernel_size,
                                padding=padding))
            self.shortcut = nn.Sequential(bypass,)
        self.norm1 = make_norm_layer(norm_type, in_features)
        self.norm2 = make_norm_layer(norm_type, in_features)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.learnable_shortcut:
            out += self.shortcut(x)
        else:
            out += x
        return out
