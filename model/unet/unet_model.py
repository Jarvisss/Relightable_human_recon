""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from ..MLP import MLP

class LightFilter(nn.Module):
    def __init__(self, n_down=5, in_channels=512, out_channels=512, pool_type='max'):
        super().__init__()
        self.n_down = n_down
        self.pool_layer = nn.MaxPool2d(2) if pool_type == 'max' else antialiased_cnns.BlurPool(in_channels, stride=2)
        for i in range(n_down):
            self.add_module('conv_' + str(i), nn.Conv2d(in_channels, out_channels, 1, 1, 0))
    
    def forward(self, x):
        # [32,32,512]
        out = x
        for i in range(self.n_down):
            out = self.pool_layer(out)
            out = self._modules['conv_' + str(i)](out)

        # []
        return out

class UNet_s_min(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers=6, bilinear=True, inplace=False, norm_type='group'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 8, inplace=inplace, norm_type=norm_type) # [512,512,64]    1,2-th 3-64-64
        self.down1 = Down(8, 16, inplace=inplace, norm_type=norm_type) # [256,256,128]              3,4-th 64-128-128
        self.down2 = Down(16, 32, inplace=inplace, norm_type=norm_type) # [128,128,256]             5,6-th 128-256-256
        self.down2 = Down(32, 64, inplace=inplace, norm_type=norm_type) # [128,128,256]             5,6-th 128-256-256
        # self.down5 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [16,16,512]               11,12-th
        # self.down6 = Down(512, 512) # [8,8,512]                 6-th
        # self.up1 = Up(1024, 512, bilinear) #  up[8,8,512] + [16,16,512] -> [16,16,1024] -> [16,16,512]          7-th
        # self.up2 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type)  # up[16,16,512] + [32,32,512] -> [32,32,1024] -> [32,32,512]        8-th
        self.up4 = Up(96, 64, bilinear, inplace=inplace, norm_type=norm_type) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
        self.up5 = Up(80, 64, bilinear, inplace=inplace, norm_type=norm_type) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
        self.up6 = Up(72, 64, bilinear, inplace=inplace, norm_type=norm_type) # up[256,256,256] + [512,512,64] -> [512,512,320] -> [512,512,256]    12-th
        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x7 = self.down6(x6)
        # x = self.up1(x7, x6)
        x = self.up5(x3, x2) # 256+256 -> 256
        x = self.up6(x, x1) # 128+256 -> 256
        logits = self.outc(x) # 256 -> 256

        mid_feat = x3
        return logits, mid_feat


class UNet_s_mid(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers=6, bilinear=True, inplace=False, norm_type='group'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 16, inplace=inplace, norm_type=norm_type) # [512,512,64]    1,2-th 3-64-64
        self.down1 = Down(16, 32, inplace=inplace, norm_type=norm_type) # [256,256,128]              3,4-th 64-128-128
        self.down2 = Down(32, 64, inplace=inplace, norm_type=norm_type) # [128,128,256]             5,6-th 128-256-256
        self.down3 = Down(64, 128, inplace=inplace, norm_type=norm_type) # [128,128,256]             5,6-th 128-256-256
        # self.down5 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [16,16,512]               11,12-th
        # self.down6 = Down(512, 512) # [8,8,512]                 6-th
        # self.up1 = Up(1024, 512, bilinear) #  up[8,8,512] + [16,16,512] -> [16,16,1024] -> [16,16,512]          7-th
        # self.up2 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type)  # up[16,16,512] + [32,32,512] -> [32,32,1024] -> [32,32,512]        8-th
        self.up4 = Up(192, 64, bilinear, inplace=inplace, norm_type=norm_type) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
        self.up5 = Up(96, 64, bilinear, inplace=inplace, norm_type=norm_type) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
        self.up6 = Up(80, 32, bilinear, inplace=inplace, norm_type=norm_type) # up[256,256,256] + [512,512,64] -> [512,512,320] -> [512,512,256]    12-th
        self.outc = OutConv(32, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # x7 = self.down6(x6)
        # x = self.up1(x7, x6)
        x = self.up4(x4, x3) # 256+256 -> 256
        x = self.up5(x3, x2) # 256+256 -> 256
        x = self.up6(x, x1) # 128+256 -> 256
        logits = self.outc(x) # 256 -> 256

        mid_feat = x3
        return logits, mid_feat

class UNet_s_deep(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers=6, bilinear=True, inplace=False, norm_type='group'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64, inplace=inplace, norm_type=norm_type) # [512,512,64]    1,2-th 3-64-64
        self.down1 = Down(64, 64, inplace=inplace, norm_type=norm_type) # [256,256,128]              3,4-th 64-128-128
        self.down2 = Down(64, 128, inplace=inplace, norm_type=norm_type) # [128,128,256]             5,6-th 128-256-256
        self.down3 = Down(128, 256, inplace=inplace, norm_type=norm_type) # [64,64,512]               7,8-th 256-512-512
        # self.down5 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [16,16,512]               11,12-th
        # self.down6 = Down(512, 512) # [8,8,512]                 6-th
        # self.up1 = Up(1024, 512, bilinear) #  up[8,8,512] + [16,16,512] -> [16,16,1024] -> [16,16,512]          7-th
        # self.up2 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type)  # up[16,16,512] + [32,32,512] -> [32,32,1024] -> [32,32,512]        8-th
        self.up4 = Up(384, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
        self.up5 = Up(320, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
        self.up6 = Up(320, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[256,256,256] + [512,512,64] -> [512,512,320] -> [512,512,256]    12-th
        self.outc = OutConv(256, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # 512
        
        # x7 = self.down6(x6)
        # x = self.up1(x7, x6)
        x = self.up4(x4, x3) # 256+128 -> 256
        x = self.up5(x, x2) # 256+64 -> 320
        x = self.up6(x, x1) # 256+32 -> 288
        logits = self.outc(x) # 256 -> 256

        mid_feat = x4
        return logits, mid_feat

class UNet_s_deeper_128(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers=6, bilinear=True, inplace=False, norm_type='group', conv_type='double', padding_mode='zeros'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        if conv_type=='double':
            self.inc = DoubleConv(n_channels, 64, inplace=inplace, norm_type=norm_type, padding_mode=padding_mode) # [512,512,64]    1,2-th 3-64-64
        else:
            self.inc = SingleConv(n_channels, 64, inplace=inplace, norm_type=norm_type)
        self.down1 = Down(64, 64, inplace=inplace, norm_type=norm_type, conv_type=conv_type, padding_mode=padding_mode) # [256,256,128]              3,4-th 64-128-128
        self.down2 = Down(64, 128, inplace=inplace, norm_type=norm_type, conv_type=conv_type, padding_mode=padding_mode) # [128,128,256]             5,6-th 128-256-256
        self.down3 = Down(128, 128, inplace=inplace, norm_type=norm_type, conv_type=conv_type, padding_mode=padding_mode) # [64,64,512]               7,8-th 256-512-512
        self.down4 = Down(128, 256, inplace=inplace, norm_type=norm_type, conv_type=conv_type, padding_mode=padding_mode) # [32,32,512]               9,10-th
        # self.down5 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [16,16,512]               11,12-th
        # self.down6 = Down(512, 512) # [8,8,512]                 6-th
        # self.up1 = Up(1024, 512, bilinear) #  up[8,8,512] + [16,16,512] -> [16,16,1024] -> [16,16,512]          7-th
        # self.up2 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type)  # up[16,16,512] + [32,32,512] -> [32,32,1024] -> [32,32,512]        8-th
        self.up3 = Up(256+128, 256, bilinear, inplace=inplace, norm_type=norm_type, conv_type=conv_type, padding_mode=padding_mode) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
        self.up4 = Up(256+128, 256, bilinear, inplace=inplace, norm_type=norm_type, conv_type=conv_type, padding_mode=padding_mode) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
        self.up5 = Up(256+64, 128, bilinear, inplace=inplace, norm_type=norm_type, conv_type=conv_type, padding_mode=padding_mode) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
        self.up6 = Up(128+64, 128, bilinear, inplace=inplace, norm_type=norm_type, conv_type=conv_type, padding_mode=padding_mode) # up[256,256,256] + [512,512,64] -> [512,512,320] -> [512,512,256]    12-th
        self.outc = OutConv(128, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 512
        
        # x7 = self.down6(x6)
        # x = self.up1(x7, x6)
        x = self.up3(x5, x4)
        x = self.up4(x, x3) # 256+128 -> 256
        x = self.up5(x, x2) # 256+64 -> 320
        x = self.up6(x, x1) # 256+32 -> 288
        logits = self.outc(x) # 256 -> 256

        mid_feat = x5
        return logits, mid_feat

class UNet_s_deeper_64(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers=6, bilinear=True, inplace=False, norm_type='group', conv_type='double'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        if conv_type=='double':
            self.inc = DoubleConv(n_channels, 32, inplace=inplace, norm_type=norm_type) # [512,512,64]    1,2-th 3-64-64
        else:
            self.inc = SingleConv(n_channels, 32, inplace=inplace, norm_type=norm_type)
        self.down1 = Down(32, 32, inplace=inplace, norm_type=norm_type, conv_type=conv_type) # [256,256,128]              3,4-th 64-128-128
        self.down2 = Down(32, 64, inplace=inplace, norm_type=norm_type, conv_type=conv_type) # [128,128,256]             5,6-th 128-256-256
        self.down3 = Down(64, 64, inplace=inplace, norm_type=norm_type, conv_type=conv_type) # [64,64,512]               7,8-th 256-512-512
        self.down4 = Down(64, 128, inplace=inplace, norm_type=norm_type, conv_type=conv_type) # [32,32,512]               9,10-th
        # self.down5 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [16,16,512]               11,12-th
        # self.down6 = Down(512, 512) # [8,8,512]                 6-th
        # self.up1 = Up(1024, 512, bilinear) #  up[8,8,512] + [16,16,512] -> [16,16,1024] -> [16,16,512]          7-th
        # self.up2 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type)  # up[16,16,512] + [32,32,512] -> [32,32,1024] -> [32,32,512]        8-th
        self.up3 = Up(128+64, 128, bilinear, inplace=inplace, norm_type=norm_type, conv_type=conv_type) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
        self.up4 = Up(128+64, 128, bilinear, inplace=inplace, norm_type=norm_type, conv_type=conv_type) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
        self.up5 = Up(128+32, 64, bilinear, inplace=inplace, norm_type=norm_type, conv_type=conv_type) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
        self.up6 = Up(64+32, 64, bilinear, inplace=inplace, norm_type=norm_type, conv_type=conv_type) # up[256,256,256] + [512,512,64] -> [512,512,320] -> [512,512,256]    12-th
        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 512
        
        # x7 = self.down6(x6)
        # x = self.up1(x7, x6)
        x = self.up3(x5, x4)
        x = self.up4(x, x3) # 256+128 -> 256
        x = self.up5(x, x2) # 256+64 -> 320
        x = self.up6(x, x1) # 256+32 -> 288
        logits = self.outc(x) # 256 -> 256

        mid_feat = x5
        return logits, mid_feat



# class UNet_s_deeper_64(nn.Module):
#     def __init__(self, n_channels, n_classes, n_layers=6, bilinear=True, inplace=False, norm_type='group'):
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
        
#         self.inc = DoubleConv(n_channels, 64, inplace=inplace, norm_type=norm_type) # [512,512,64]    1,2-th 3-64-64
#         self.down1 = Down(64, 64, inplace=inplace, norm_type=norm_type) # [256,256,128]              3,4-th 64-128-128
#         self.down2 = Down(64, 128, inplace=inplace, norm_type=norm_type) # [128,128,256]             5,6-th 128-256-256
#         self.down3 = Down(128, 128, inplace=inplace, norm_type=norm_type) # [64,64,512]               7,8-th 256-512-512
#         self.down4 = Down(128, 256, inplace=inplace, norm_type=norm_type) # [32,32,512]               9,10-th
#         # self.down5 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [16,16,512]               11,12-th
#         # self.down6 = Down(512, 512) # [8,8,512]                 6-th
#         # self.up1 = Up(1024, 512, bilinear) #  up[8,8,512] + [16,16,512] -> [16,16,1024] -> [16,16,512]          7-th
#         # self.up2 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type)  # up[16,16,512] + [32,32,512] -> [32,32,1024] -> [32,32,512]        8-th
#         self.up3 = Up(384, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
#         self.up4 = Up(384, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
#         self.up5 = Up(320, 128, bilinear, inplace=inplace, norm_type=norm_type) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
#         self.up6 = Up(192, 64, bilinear, inplace=inplace, norm_type=norm_type) # up[256,256,256] + [512,512,64] -> [512,512,320] -> [512,512,256]    12-th
#         self.outc = OutConv(64, n_classes)


#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3) # 512
#         x5 = self.down4(x4) # 512
        
#         # x7 = self.down6(x6)
#         # x = self.up1(x7, x6)
#         x = self.up3(x5, x4)
#         x = self.up4(x, x3) # 256+128 -> 256
#         x = self.up5(x, x2) # 256+64 -> 320
#         x = self.up6(x, x1) # 256+32 -> 288
#         logits = self.outc(x) # 256 -> 256

#         mid_feat = x5
#         return logits, mid_feat



class UNet_s_deeper(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers=6, bilinear=True, inplace=False, norm_type='group'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64, inplace=inplace, norm_type=norm_type) # [512,512,64]    1,2-th 3-64-64
        self.down1 = Down(64, 64, inplace=inplace, norm_type=norm_type) # [256,256,128]              3,4-th 64-128-128
        self.down2 = Down(64, 128, inplace=inplace, norm_type=norm_type) # [128,128,256]             5,6-th 128-256-256
        self.down3 = Down(128, 128, inplace=inplace, norm_type=norm_type) # [64,64,512]               7,8-th 256-512-512
        self.down4 = Down(128, 256, inplace=inplace, norm_type=norm_type) # [32,32,512]               9,10-th
        # self.down5 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [16,16,512]               11,12-th
        # self.down6 = Down(512, 512) # [8,8,512]                 6-th
        # self.up1 = Up(1024, 512, bilinear) #  up[8,8,512] + [16,16,512] -> [16,16,1024] -> [16,16,512]          7-th
        # self.up2 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type)  # up[16,16,512] + [32,32,512] -> [32,32,1024] -> [32,32,512]        8-th
        self.up3 = Up(384, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
        self.up4 = Up(384, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
        self.up5 = Up(320, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
        self.up6 = Up(320, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[256,256,256] + [512,512,64] -> [512,512,320] -> [512,512,256]    12-th
        self.outc = OutConv(256, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 512
        
        # x7 = self.down6(x6)
        # x = self.up1(x7, x6)
        x6 = self.up3(x5, x4)
        x7 = self.up4(x6, x3) # 256+128 -> 256
        x8 = self.up5(x7, x2) # 256+64 -> 320
        x9 = self.up6(x8, x1) # 256+32 -> 288
        logits = self.outc(x9) # 256 -> 256

        mid_feat = x5
        return logits, mid_feat


class UNet_s_deepest(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers=6, bilinear=True, inplace=False, norm_type='group'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64, inplace=inplace, norm_type=norm_type) # [512,512,64]    1,2-th 3-64-64
        self.down0 = Down(64, 64, inplace=inplace, norm_type=norm_type) # [256,256,128]              3,4-th 64-128-128
        self.down1 = Down(64, 64, inplace=inplace, norm_type=norm_type) # [256,256,128]              3,4-th 64-128-128
        self.down2 = Down(64, 128, inplace=inplace, norm_type=norm_type) # [128,128,256]             5,6-th 128-256-256
        self.down3 = Down(128, 128, inplace=inplace, norm_type=norm_type) # [64,64,512]               7,8-th 256-512-512
        self.down4 = Down(128, 256, inplace=inplace, norm_type=norm_type) # [32,32,512]               9,10-th
        # self.down5 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [16,16,512]               11,12-th
        # self.down6 = Down(512, 512) # [8,8,512]                 6-th
        # self.up1 = Up(1024, 512, bilinear) #  up[8,8,512] + [16,16,512] -> [16,16,1024] -> [16,16,512]          7-th
        # self.up2 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type)  # up[16,16,512] + [32,32,512] -> [32,32,1024] -> [32,32,512]        8-th
        self.up3 = Up(384, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
        self.up4 = Up(384, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
        self.up5 = Up(320, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
        self.up6 = Up(320, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[256,256,256] + [512,512,64] -> [512,512,320] -> [512,512,256]    12-th
        self.up7 = Up(320, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[256,256,256] + [512,512,64] -> [512,512,320] -> [512,512,256]    12-th
        self.outc = OutConv(256, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down0(x1)
        x3 = self.down1(x2)
        x4 = self.down2(x3)
        x5 = self.down3(x4) # 512
        x6 = self.down4(x5) # 512
        
        # x7 = self.down6(x6)
        # x = self.up1(x7, x6)
        x = self.up3(x6, x5)
        x = self.up4(x, x4) # 256+128 -> 256
        x = self.up5(x, x3) # 256+64 -> 320
        x = self.up6(x, x2) # 256+32 -> 288
        x = self.up7(x, x1) # 256+32 -> 288
        logits = self.outc(x) # 256 -> 256

        mid_feat = x6
        return logits, mid_feat


class UNet_indirect(nn.Module):
    def __init__(self, n_channels, n_classes=3, n_layers=6, bilinear=True, inplace=False, norm_type='group', last_op=nn.Tanh()):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64, inplace=inplace, norm_type=norm_type) # [512,512,64]    1,2-th 3-64-64
        self.down1 = Down(64, 128, inplace=inplace, norm_type=norm_type) # [256,256,128]              3,4-th 64-128-128
        self.down2 = Down(128, 256, inplace=inplace, norm_type=norm_type) # [128,128,256]             5,6-th 128-256-256
        self.down3 = Down(256, 512, inplace=inplace, norm_type=norm_type) # [64,64,512]               7,8-th 256-512-512
        self.down4 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [32,32,512]               9,10-th
        # self.down5 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [16,16,512]               11,12-th
        # self.down6 = Down(512, 512) # [8,8,512]                 6-th
        # self.up1 = Up(1024, 512, bilinear) #  up[8,8,512] + [16,16,512] -> [16,16,1024] -> [16,16,512]          7-th
        # self.up2 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type)  # up[16,16,512] + [32,32,512] -> [32,32,1024] -> [32,32,512]        8-th
        self.up3 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type) # up[32,32,512] + [64,64,512] -> [64,64,1024] -> [64,64,512]         9-th
        self.up4 = Up(768, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
        self.up5 = Up(384, 128, bilinear, inplace=inplace, norm_type=norm_type) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
        self.up6 = Up(192, 64, bilinear, inplace=inplace, norm_type=norm_type) # up[256,256,128] + [512,512,64] -> [512,512,320] -> [512,512,256]    12-th
        self.outc = OutConv(64, n_classes, last_op=last_op)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 512
        # x7 = self.down6(x6)
        # x = self.up1(x7, x6)
        x = self.up3(x5, x4) # 512+512 -> 512
        x = self.up4(x, x3) # 512+512 -> 256
        x = self.up5(x, x2) # 256+256 -> 256
        x = self.up6(x, x1) # 128+256 -> 256
        logits = self.outc(x) # 256 -> 256

        
        return logits

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers=6, bilinear=True, inplace=False, norm_type='group'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64, inplace=inplace, norm_type=norm_type) # [512,512,64]    1,2-th 3-64-64
        self.down1 = Down(64, 128, inplace=inplace, norm_type=norm_type) # [256,256,128]              3,4-th 64-128-128
        self.down2 = Down(128, 256, inplace=inplace, norm_type=norm_type) # [128,128,256]             5,6-th 128-256-256
        self.down3 = Down(256, 512, inplace=inplace, norm_type=norm_type) # [64,64,512]               7,8-th 256-512-512
        self.down4 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [32,32,512]               9,10-th
        # self.down5 = Down(512, 512, inplace=inplace, norm_type=norm_type) # [16,16,512]               11,12-th
        # self.down6 = Down(512, 512) # [8,8,512]                 6-th
        # self.up1 = Up(1024, 512, bilinear) #  up[8,8,512] + [16,16,512] -> [16,16,1024] -> [16,16,512]          7-th
        # self.up2 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type)  # up[16,16,512] + [32,32,512] -> [32,32,1024] -> [32,32,512]        8-th
        self.up3 = Up(1024, 512, bilinear, inplace=inplace, norm_type=norm_type) # up[32,32,512] + [64,64,512] -> [64,64,1024] -> [64,64,512]         9-th
        self.up4 = Up(768, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[64,64,512] + [128,128,256] -> [128,128,768] -> [128,128,256]     10-th
        self.up5 = Up(384, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[128,128,256] + [256,256,128] -> [256,256,384] -> [256,256,256]   11-th
        self.up6 = Up(320, 256, bilinear, inplace=inplace, norm_type=norm_type) # up[256,256,256] + [512,512,64] -> [512,512,320] -> [512,512,256]    12-th
        self.outc = OutConv(256, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 512
        # x7 = self.down6(x6)
        # x = self.up1(x7, x6)
        x = self.up3(x5, x4) # 512+512 -> 512
        x = self.up4(x, x3) # 512+512 -> 256
        x = self.up5(x, x2) # 256+256 -> 256
        x = self.up6(x, x1) # 128+256 -> 256
        logits = self.outc(x) # 256 -> 256

        mid_feat = x5
        return logits, mid_feat