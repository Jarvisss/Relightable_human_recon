import torch
from torch.functional import norm
import torch.nn as nn
from model.blocks import get_nonlinearity_layer, get_norm_layer

class Norm2Corr(nn.Module):
    def __init__(self):
        super(Norm2Corr, self).__init__()
        self._make_layers()


    def _make_layers(self,):
        pass

    def forward(self, normal_map):
        
        pass

class GeoNet(nn.Module):
    '''
    Take from Pix2PixHD, GlobalGenerator
    '''
    def __init__(self, input_nc, output_nc, ngf=64, n_down=4, n_blocks=9, norm_type='bn',activation='LeakyReLU',padding_type='reflect'):
        super(GeoNet, self).__init__()
        # norm_layer = get_norm_layer(norm_type=norm_type)
        norm_layer = nn.BatchNorm2d
        nonlinear_layer = get_nonlinearity_layer(activation_type=activation)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nonlinear_layer]    
        ### downsample
        for i in range(n_down):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nonlinear_layer]
        
        # ### resnet blocks
        mult = 2**n_down
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=nonlinear_layer, norm_layer=norm_layer)]
        
        # ### upsample         
        for i in range(n_down):
            mult = 2**(n_down - i)
            # model += [
            #     nn.Upsample(scale_factor = 2, mode='bilinear'),
            #     nn.ReflectionPad2d(1),
            #     nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0),
            #            norm_layer(int(ngf * mult / 2)), nonlinear_layer]
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nonlinear_layer]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
    def forward(self, lit):
        return self.model(lit)


class UVNet(nn.Module):
    '''
    Take from Pix2PixHD, GlobalGenerator
    '''
    def __init__(self, input_nc, output_nc, ngf=32, n_down=4, n_blocks=1, norm_type='bn',activation='LeakyReLU',padding_type='reflect'):
        super(UVNet, self).__init__()
        # norm_layer = get_norm_layer(norm_type=norm_type)
        norm_layer = nn.BatchNorm2d
        nonlinear_layer = get_nonlinearity_layer(activation_type=activation)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nonlinear_layer]    
        ### downsample
        for i in range(n_down):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nonlinear_layer]
        
        # ### resnet blocks
        mult = 2**n_down
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=nonlinear_layer, norm_layer=norm_layer)]
        
        # ### upsample         
        for i in range(n_down):
            mult = 2**(n_down - i)
            # model += [
            #     nn.Upsample(scale_factor = 2, mode='bilinear'),
            #     nn.ReflectionPad2d(1),
            #     nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0),
            #            norm_layer(int(ngf * mult / 2)), nonlinear_layer]
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nonlinear_layer]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
    def forward(self, lit, normal):
        return self.model(torch.cat((lit,normal), dim=1))

class Tex2Cloth(nn.Module):
    '''
    Take from Pix2PixHD, GlobalGenerator
    '''
    def __init__(self, input_nc, output_nc, ngf, n_down=4, n_blocks=1, norm_type='bn',activation='LeakyReLU',padding_type='reflect'):
        super(Tex2Cloth, self).__init__()
        # norm_layer = get_norm_layer(norm_type=norm_type)
        norm_layer = nn.BatchNorm2d
        nonlinear_layer = get_nonlinearity_layer(activation_type=activation)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nonlinear_layer]    
        ### downsample
        for i in range(n_down):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nonlinear_layer]
        
        # ### resnet blocks
        mult = 2**n_down
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=nonlinear_layer, norm_layer=norm_layer)]
        
        # ### upsample         
        for i in range(n_down):
            mult = 2**(n_down - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nonlinear_layer]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
    def forward(self, unlit, normal, lit):
        inp = torch.cat((unlit, normal, lit), dim=1) # N * 3C * H * W
        return self.model(inp)  
        

class Tex2ClothDecompose(nn.Module):
    '''
    Take from Pix2PixHD, GlobalGenerator
    '''
    def __init__(self, input_nc, output_nc, ngf, n_down=3, n_blocks=9, norm_type='bn',activation='LeakyReLU',padding_type='reflect'):
        super(Tex2ClothDecompose, self).__init__()
        # norm_layer = get_norm_layer(norm_type=norm_type)
        norm_layer = nn.BatchNorm2d
        nonlinear_layer = get_nonlinearity_layer(activation_type=activation)

        encoder_model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nonlinear_layer]    
        ### downsample
        for i in range(n_down):
            mult = 2**i
            encoder_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nonlinear_layer]
        
        # ### resnet blocks
        mult = 2**n_down
        for i in range(n_blocks):
            if i == 0:
                albedo_model = [ResnetBlock(ngf * mult, padding_type=padding_type, activation=nonlinear_layer, norm_layer=norm_layer)]
                shading_model = [ResnetBlock(ngf * mult, padding_type=padding_type, activation=nonlinear_layer, norm_layer=norm_layer)]
            else:
                albedo_model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=nonlinear_layer, norm_layer=norm_layer)]
                shading_model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=nonlinear_layer, norm_layer=norm_layer)]
        
        # ### upsample         
        for i in range(n_down):
            mult = 2**(n_down - i)
            albedo_model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nonlinear_layer]
            shading_model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nonlinear_layer]
        albedo_model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        shading_model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]


        self.encoder_model = nn.Sequential(*encoder_model)
        self.albedo_model = nn.Sequential(*albedo_model)
        self.shading_model = nn.Sequential(*shading_model)
    def forward(self, normal, lit):
        inp = torch.cat((normal, lit), dim=1) # N * 3C * H * W
        latent_code = self.encoder_model(inp)
        albedo = self.albedo_model(latent_code)
        shading = self.shading_model(latent_code)
        return albedo, shading
        

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, activation_type='LeakyReLU',
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        # activation = nn.ReLU(True)        
        activation = get_nonlinearity_layer(activation_type)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 
                                         kernel_size=3, stride=2, 
                                         padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), 
                       activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return  torch.clamp(self.model(input), 0., 1.)  




