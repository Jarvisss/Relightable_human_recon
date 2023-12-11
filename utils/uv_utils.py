from logging import root
from random import sample
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.vis_utils import tensor2im
from PIL import Image
import os.path as osp
import os

def uv_sampler(texture, uv_img, mask, padding_mode='zeros', align_corners=False):
    '''
    @ Input:
        texture: [N, C, H, W] image 
        uv_img: [N, 2, H, W] with 0 ~ 1 float value indicates UV location in texture space
        mask: [N, 1, H, W] value 0|1
    @ Return:
        sampled_img: [N, C, H, W] image 
    '''
    u  = uv_img[:, 0, :, :]
    v  = uv_img[:, 1, :, :]
    v_1 = torch.ones_like(v)
    ## our uv [0,0] is at left-bottom
    ## while grid_sample's origin(-1,-1) is at left-top
    ## https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html 
    
    uv_normalize = (torch.stack((u,v_1-v), dim=1) - 0.5)/0.5 # [0, 1] -> [-1, 1]
    grid = uv_normalize.permute(0,2,3,1)
    # grid should be [-1, 1]
    uv_sampled = F.grid_sample(texture, grid, padding_mode=padding_mode, align_corners=align_corners)
    # [N, 3, H, W]
    # [N, 3, H, W] * [N, 1, H, W]
    uv_smapled_masked = uv_sampled * mask

    return uv_smapled_masked 

def sampler_unit_test(texture, uv_img, mask, ground_truth, output_path):

    if not osp.exists(osp.dirname(output_path)):
        os.makedirs(osp.dirname(output_path))
    sampled_img = uv_sampler(texture, uv_img, mask)
    sampled_im = tensor2im(sampled_img, out_size=(512,512))
    Image.fromarray(sampled_im.type(torch.uint8).numpy()).save(output_path)
    loss = (torch.flatten(sampled_img) - torch.flatten(ground_truth)) ** 2.0 * torch.flatten(mask)
    result = torch.sum(loss) / torch.sum(mask)
    print(result)
    pass



def grid_sample(image, optical):
    '''
    equals to F.grid_sample(image, optical, padding_mode='border', align_corners=True),
    enables gradient
    https://github.com/pytorch/pytorch/issues/34704#issuecomment-878940122
    '''
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape
    if torch.any(torch.isnan(optical)):
        print('optical has nan')
    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def recompose(albedo, shading):
    '''
    recompose image from albedo and shading
    '''
    return albedo * shading


# def convert2atlas(input_texture):

#     from UVTextureConverter import Normal2Atlas
#     from PIL import Image
#     import numpy as np

#     converter = Normal2Atlas(normal_size=256, atlas_size=200)

'''
Unit test
'''

if __name__ == '__main__':
    from utils.common_utils import load_img_input
    root_path = '/home/lujiawei/workspace/dataset/rendered_cloth'

    render_dir = 'cloth_prt_512_train_linear_wo_mipmap'
    # render_dir = 'cloth_prt_512_train'

    texture_path = root_path + '/fabric_256/00372.jpg'
    uv_path = root_path + '/{0}/UV/gar_5/0_0.png'.format(render_dir)
    uv_npz_path = root_path + '/{0}/UV/gar_5/0_0.npz'.format(render_dir)
    mask_path = root_path + '/{0}/MASK/gar_5/0_0.png'.format(render_dir)
    albedo_path = root_path + '/{0}/ALBEDO/gar_5/00372_0_0_.png'.format(render_dir)
    output_path = root_path + '/unit_test/uv_sampler/gar_5/00372_0_0_.png'
    load_size = 512

    texture = load_img_input(texture_path, load_size, normalize=False).unsqueeze(0)
    # uv = load_img_input(uv_path, load_size, normalize=False).unsqueeze(0)
    uv = torch.from_numpy( np.load(uv_npz_path)['T']).permute(2,0,1).unsqueeze(0) # [HWC]->[CHW]
    mask = load_img_input(mask_path, load_size, normalize=False).unsqueeze(0)
    ground_truth = load_img_input(albedo_path, load_size, normalize=False).unsqueeze(0)


    print(texture.shape)
    ## do unit test on the sampler
    sampler_unit_test(texture, uv, mask, ground_truth, output_path)

