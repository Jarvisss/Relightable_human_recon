import os
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0")
cpu = torch.device("cpu")

import logging
import sys
from model.UNet_unified import UNet_unified

from model.DiffRenderer_unified import DiffRenderer_unified
from utils.common_utils import *
from utils.camera import MVP_from_P, get_quat_from_world_mat_np, get_calib_extri_from_pose
import math
import torchvision.transforms as T
import torchvision.transforms.functional as ttf
from torchvision.transforms.functional import gaussian_blur
from utils.geo_utils import *
from utils.render_utils import *
from options.options import get_options
from tqdm import tqdm
from PIL import Image, ImageDraw
import cv2
import random
from tensorboardX import SummaryWriter
from utils.sdf_utils import save_samples_truncted_prob, save_samples_color
import time
from model.blocks import _freeze, _unfreeze
import lpips
import kornia

loss_fn_vgg = lpips.LPIPS(net='vgg')
loss_fn_alex = lpips.LPIPS(net='alex')

def save_generator(epoch, lossesG, netG, i_batch,path_to_chkpt_G,parallel=False, opt_pose=False, pose_vec=None, intensity=None):
    netG_state_dict = netG.state_dict() if not parallel else netG.module.state_dict()
    saved_dict = {
        'epoch': epoch,
        'lossesG': lossesG,
        'G_state_dict': netG_state_dict,
        'i_batch': i_batch
    }
    if opt_pose:
        saved_dict.update({'pose': pose_vec})
    if intensity is not None:
        saved_dict.update({'intensity': intensity})
    
    torch.save(saved_dict, path_to_chkpt_G)



def precompute_and_save_feats(opt, batch_data,input_view, backbone, save_dir):
    imgs_reshape = batch_data['img'].float().to(device)[input_view]
    if opt.load_size != 512:
        imgs_reshape = F.interpolate(imgs_reshape, (512, 512), mode='bicubic', align_corners=True)

    im_feat_list = []

    for s in range(imgs_reshape.shape[0]):
        with torch.no_grad():
            im_feat_pred = backbone.filter(imgs_reshape[s*step:s*step+step])

        im_feat_np = im_feat_pred.detach().cpu().numpy()

        im_feat_list += [im_feat_np]
    
    im_feat_np = np.concatenate(im_feat_list, axis=0)
    print('im feat shape', im_feat_np.shape)
    np.save(os.path.join(save_dir, 'im_feat.npy'), im_feat_np)


def load_feat_from_disk(save_dir):
    im_feat_np = np.load(os.path.join(save_dir, 'im_feat.npy'))
    light_feat_np = np.load(os.path.join(save_dir, 'light_feat.npy'))
    im_feat = torch.from_numpy(im_feat_np).to(device)
    light_feat = torch.from_numpy(light_feat_np).to(device)

    return im_feat, light_feat

def get_real_data(opt, data_dir='/mnt/data1/lujiawei/real_data/yh/yh_1019', n_sample_space=200000, num_target=12):
    IW, IH = opt.load_size[0], opt.load_size[1]
    b_min = torch.tensor([-1, -1, -1])
    b_max = torch.tensor([1, 1, 1])
    intensity = torch.tensor(opt.init_intensity).float()
    name = data_dir.split('/')[-1]
    with open(os.path.join(data_dir, 'test_ids_%d.txt' % (num_target)), 'r') as f:
        test_ids = f.readlines()
    # mask_dir = os.path.join(data_dir, 'mask_512')
    in_img_mask_path = os.path.join(data_dir, 'in_ori.png')
    mask_dir = os.path.join(data_dir, 'mask_%d_%d' % (IW, IH))
    img_dir = os.path.join(data_dir, 'image_%d_%d' % (IW, IH))
    cam_dir = os.path.join(data_dir, 'cams')
    calib_path = os.path.join(data_dir, 'cameras_%d_%d.npz' % (IW, IH))
    cam_model = np.load(calib_path)
    test_data = {}
    mask_list = []
    mask_dilate_list = []
    img_list = []
    calib_list = []
    extri_list = []
    # extri_inv_list = []
    quat_list = []
    intri_list = []
    c2w_list =[]
    w2c_list =[]
    norm_list = []
    view_id_list = []
    test_id_list = []
    # for i in range(opt.num_views):
    print(test_ids)
    for i in range(len(test_ids)):
        if test_ids[i].strip() == '':
            break
        view_id, test_id = test_ids[i].strip().split()
        # mask_path = os.path.join(mask_dir, test_id + '_alpha.png')
        mask_path = os.path.join(mask_dir, test_id + '.png')
        # img_path = os.path.join(img_dir, test_id + '.jpg')
        # mask_path = os.path.join(mask_dir, test_id + '.png')
        img_path = os.path.join(img_dir, test_id + '.png')

        mask = Image.open(mask_path).convert('L')
        img = Image.open(img_path).convert('RGBA')
        dilate_kernel_size = opt.dilate_size
        dilate_kernel = np.ones((dilate_kernel_size,dilate_kernel_size), np.uint8)
    
        mask_dilate = cv2.dilate(np.array(mask), dilate_kernel, iterations=1)
    
        world_mat = cam_model['world_mat_%s' % view_id]
        scale_mat = cam_model['scale_mat_%s' % view_id]
        c2w = cam_model['c2w_%s' % view_id]
        w2c = cam_model['w2c_%s' % view_id]
        normed_mat = world_mat @ scale_mat
        # if opt.use_CV_perspective:
        K, R, t = KRT_from_P(normed_mat[:3,:])
        extrinsic_cv, intrinsic_cv = np.eye(4), np.eye(4)
        intrinsic_cv[:3,:3] = K
        extrinsic_cv[:3,:3] = R
        extrinsic_cv[:3,3:] = t

        quat_cv = get_quat_from_world_mat_np(extrinsic_cv) #[7]
        calib_cv = torch.Tensor(np.matmul(intrinsic_cv, extrinsic_cv)).float()
        proj_func = perspective_opencv
        pt = torch.Tensor([0,0,0])
        xyz_cv = proj_func(pt[None,:,None], calib_cv[None,:,:], size=opt.load_size)
        # else:
            
        proj, model_view, quat_gl = MVP_from_P(normed_mat[:3, :4], IW, IH, opt.near, opt.far)
        gl_2_cv_matrix = np.eye(4)
        gl_2_cv_matrix[1,1] = -1
        proj = gl_2_cv_matrix @ proj
        # axis_adj = np.eye(4)
        # axis_adj[1,1]= -1
        # axis_adj[2,2]= -1
        # model_view = axis_adj @ model_view
        extrinsic_gl = model_view
        intrinsic_gl = proj
    
        calib_gl = torch.Tensor(np.matmul(intrinsic_gl, extrinsic_gl)).float()
        proj_func = perspective
        pt = torch.Tensor([0,0,0])
        xyz_gl = proj_func(pt[None,:,None], calib_gl[None,:,:], size=opt.load_size)

        if opt.use_CV_perspective:
            calib = calib_cv
            extrinsic = extrinsic_cv
            intrinsic = intrinsic_cv
            quat = quat_cv
        else:
            calib = calib_gl
            extrinsic = extrinsic_gl
            intrinsic = intrinsic_gl
            quat = quat_gl

        mask = T.ToTensor()(mask)
        mask_dilate = T.ToTensor()(mask_dilate)
        img = T.ToTensor()(img)[:3,:,:]
        # green = torch.zeros_like(img)
        # green[1] = 1.0
        # img = (mask.expand_as(img) > 0) * img + (mask.expand_as(img) <= 0) * green
        img = (mask.expand_as(img) > 0) * img

        img_list.append(img)
        mask_list.append(mask)
        mask_dilate_list.append(mask_dilate)
        calib_list.append(calib)
        c2w_list.append(torch.Tensor(c2w).float())
        w2c_list.append(torch.Tensor(w2c).float())
        quat_list.append(torch.Tensor(quat).float())
        # extri_list.append( torch.Tensor(model_view).float())
        extri_list.append( torch.Tensor(extrinsic).float())
        # extri_inv_list.append( torch.Tensor(np.linalg.inv(model_view)).float())
        # intri_list.append( torch.Tensor(proj).float())
        intri_list.append( torch.Tensor(intrinsic).float())
        norm_list.append(torch.Tensor(np.eye(4)).float())
        # norm_list.append(torch.Tensor(scale_mat).float())
        view_id_list.append(view_id)
        test_id_list.append(test_id)


    length = b_max - b_min
    space_points = (torch.rand(3, n_sample_space) * length.unsqueeze(-1) + b_min.unsqueeze(-1)).float().unsqueeze(0)
    
    test_data['intensity'] = intensity
    test_data['b_min'] = b_min
    test_data['b_max'] = b_max
    test_data['z_center'] = torch.zeros(1, 3, 1) if opt.normalize_z else None
    test_data['view_ids'] = view_id_list
    test_data['calib'] = torch.stack(calib_list, dim=0)
    test_data['pose'] = torch.stack(quat_list, dim=0)
    test_data['c2w'] = torch.stack(c2w_list, dim=0)
    test_data['w2c'] = torch.stack(w2c_list, dim=0)

    test_data['extrinsic'] = torch.stack(extri_list, dim=0)
    # test_data['extrinsic_inv'] = torch.stack(extri_inv_list, dim=0)
    test_data['intrinsic'] = torch.stack(intri_list, dim=0)
    test_data['normal_matrices'] = torch.stack(norm_list, dim=0)
    test_data['img'] = torch.stack(img_list, dim=0)
    test_data['mask'] = torch.stack(mask_list, dim=0)
    test_data['in_ori_img'] = T.ToTensor()(Image.open(in_img_mask_path).convert('L'))
    test_data['mask_dilate'] = torch.stack(mask_dilate_list, dim=0)
    test_data['name'] = name
    test_data['samples'] = space_points
    return test_data

def load_data(opt, n_samp, subject='0526', real_data_dir = '', angle_step=30, num_target=12, test=False):
    train_data = get_real_data(opt, real_data_dir, n_sample_space=n_samp, num_target=num_target)
    
    strr = 'test' if test else 'train'
    print('load {0} data {1}, total images {2}'.format(strr, subject, len(train_data['img'])))
    
    return train_data

def render_imgs(backbone, renderer: DiffRenderer_unified, data_input, data_target, intensity, IH, IW, render_dir,debug=False,sample_intersect=True, cal_diff_normal=False, sample_batch=6000, use_TSDF=False, TSDF_thres=0.1,epoch=-1, with_alpha=True):
    '''
    render sdf to image using ray tracing
    '''
    print('render imgs to %s'%render_dir)
    print('intensity:', intensity.item())
    os.makedirs(render_dir, exist_ok=True)
    
    debug_flag = debug

    spec_scale = 1
    rough_scale = 1

    backbone.eval()
    renderer.eval()
    renderer.debug=False
    renderer.ray_tracer.debug=False
    imgs_input = data_input['imgs_input']
    masks_input = data_input['masks_input']
    

    target_views = data_target['target_views']
    imgs_target = data_target['imgs_target']
    masks_target = data_target['masks_target']
    masks_dilate_target = data_target['masks_dilate_target']
    ray_os = data_target['ray_os']
    ray_ds = data_target['ray_ds']
    light_dirs = data_target['light_dirs']
    masks_intersect = data_target['masks_intersect_target']
    front_back_intersections = data_target['front_back_intersections_target']

    with torch.no_grad():
        input_tensor = imgs_input
        if opt.feed_mask:
            input_tensor = torch.cat((input_tensor, masks_input), dim=1)
        
        im_feat_pred = backbone.filter(input_tensor)
    ## as we use align_corners=True in all experiments, we turn ray_coord to the center of pixel by adding 0.5
    print('render views: ')
    # for k in tqdm(range(0, imgs_target.shape[0])):
    for k in range(0, imgs_target.shape[0]):
        # k = (1+i_batch_current) % K
        img_k = imgs_target[k:k+1].view(3, -1).T
        mask_k = masks_target[k:k+1].view(1, -1).T
        dilate_mask_k = masks_dilate_target[k:k+1].view(1, 1, -1).permute(0,2,1)
        if sample_intersect:
            mask_id = torch.where(masks_intersect[k].view(-1)>0.5)[0]
        else:
            mask_id = torch.where(dilate_mask_k.view(-1) > 0.5 )[0]
        n_rays = len(mask_id)
        ray_os_k = ray_os[k][mask_id] ## [n_rays, 3]
        ray_ds_k = ray_ds[k][mask_id] ## [n_rays, 3]
        light_dirs_k = light_dirs[k][mask_id]
        ray_cs_k = img_k[mask_id] ## [n_rays, 3]
        ray_ms_k = mask_k[mask_id] ## [n_rays, 1]
        if False:
            init_points = front_back_intersections[k,mask_id,0:1] * ray_ds_k + ray_os_k
            debug_path = path_to_vis
            mask_pts = init_points.detach()
            init_points_back = front_back_intersections[k,mask_id,1:2] * ray_ds_k + ray_os_k
            mask_pts_back = init_points_back.detach()
            mask_pts_color = torch.ones_like(mask_pts) * torch.rand_like(mask_pts[0:1,:])
            mask_pts_color_back = torch.ones_like(mask_pts_back)  * torch.rand_like(mask_pts[0:1,:])
            save_samples_color(debug_path + '/debug/view%d_back.ply' % k, mask_pts_back.cpu(), mask_pts_color_back.cpu()*255)
            save_samples_color(debug_path + '/debug/view%d_front.ply' % k, mask_pts.cpu(), mask_pts_color.cpu()*255)
        front_back_intersections_k = front_back_intersections[k, mask_id, :].to(device)
        mask_intersect_k = masks_intersect[k, mask_id, :].to(device)

        view_id = k
        sample_num = sample_batch
        if n_rays % sample_num == 0:
            length = n_rays // sample_num
        else:
            length = n_rays // sample_num + 1
        out_albedo_k = torch.zeros_like(img_k) #[N, 3]
        out_error_k = torch.zeros_like(img_k) #[N, 3]
        out_depth_k = torch.zeros_like(mask_k) #[N, 1]
        out_mask_k = torch.zeros_like(mask_k) #[N, 1]
        out_shading_k = torch.zeros_like(img_k)
        out_cosine_k = torch.zeros_like(img_k)
        out_spec_shading_k = torch.zeros_like(img_k)
        out_spec_albedo_k = torch.zeros_like(img_k)
        out_spec_roughness_k = torch.zeros_like(mask_k)
        out_rendered_k = torch.zeros_like(img_k)
        out_diffuse_color_k = torch.zeros_like(img_k)
        out_specular_color_k = torch.zeros_like(img_k)
        out_normal_k = torch.zeros_like(img_k)
        out_gt_k = torch.zeros_like(img_k)
        # for i in tqdm(range(length)):
        for i in range(length):
            left = i * sample_num
            right = min((i+1) * sample_num, n_rays)
            ray_os_slice = ray_os_k[left:right]
            ray_ds_slice = ray_ds_k[left:right]
            light_dirs_slice = light_dirs_k[left:right]
            ray_cs_slice = ray_cs_k[left:right]
            ray_ms_slice = ray_ms_k[left:right].flatten()

            sampling_id_slice = mask_id[left:right]
            front_back_intersections_slice = front_back_intersections_k[left:right, :]
            mask_intersect_slice = mask_intersect_k[left:right, :].flatten()

            old_thres = TSDF_thres
            old_iters = renderer.ray_tracer.sphere_tracing_iters
            TSDF_thres = 0.03
            renderer.ray_tracer.sphere_tracing_iters = 40
            output = renderer.forward(data_input=data_input, src_img_feats=im_feat_pred, 
                tgt_views=target_views, ray_os=ray_os_slice, ray_ds=ray_ds_slice, ray_cs=ray_cs_slice, ray_ms=ray_ms_slice, light_dirs=light_dirs_slice,
                front_back_dist_init=front_back_intersections_slice, ray_mask_init=mask_intersect_slice, cal_diff_normal=cal_diff_normal, no_grad=True,intensity=intensity, jittor_std=opt.eik_std,use_poe=opt.use_positional_encoding,
                use_TSDF=use_TSDF, TSDF_thres=TSDF_thres, epoch=epoch, save_intermediate=True
            )
            # sdf_func, albedo_func, gradient_func, spec_func, rough_func = renderer.get_sdf_albedo_gradient_funcs(im_feat_pred, imgs_input, calibs_input, masks_input, z_center, feed_extrin, opt.cal_diff_normal,use_poe=opt.use_positional_encoding, joints_3d=joints_3d)

            # alpha = 200
            # loss, loss_dict, loss_str = renderer.cal_loss(output, sdf_func, gradient_func,albedo_func,spec_func, rough_func, \
            #     ray_weights=None,
            #     ray_in_original_img=None, 
            #     epoch=epoch,
            #     include_smooth_loss=False,
            #     precomputed_indirect=opt.precomputed_indirect, 
            #     alpha=alpha,
            #     lambda_reg=opt.lambda_reg, 
            #     lambda_mask=opt.lambda_mask, 
            #     lambda_align=opt.lambda_align, writer=writer)

            TSDF_thres = old_thres
            renderer.ray_tracer.sphere_tracing_iters = old_iters
            out_gt_k[sampling_id_slice, :] = ray_cs_slice

            if output is not None:
                
                out_albedo_k[sampling_id_slice, :] = shading_l2g(output['diff_albedos'].detach())
                out_shading_k[sampling_id_slice, :] = torch.clamp(shading_l2g(output['diffuse_shadings'].detach()), 0, 1)
                out_cosine_k[sampling_id_slice, :] = torch.clamp(output['cosine_term_light'].detach(), 0, 1)
                out_spec_shading_k[sampling_id_slice, :] = torch.clamp(shading_l2g(output['specular_shadings'].detach()), 0, 1)
                out_spec_albedo_k[sampling_id_slice, :] = shading_l2g(output['specular_albedos'].detach())
                out_spec_roughness_k[sampling_id_slice, :] = output['roughness'].detach()
                out_rendered_k[sampling_id_slice, :] = shading_l2g(output['preds'].detach())
                out_diffuse_color_k[sampling_id_slice, :] = shading_l2g(output['preds_diffuse_color'].detach())
                out_specular_color_k[sampling_id_slice, :] = torch.clamp(out_rendered_k[sampling_id_slice, :] - out_diffuse_color_k[sampling_id_slice, :], min=0.0)
                out_mask_k[sampling_id_slice, :] = output['network_object_mask'].detach().unsqueeze(-1).float()  # [N, 1]
                out_normal_k[sampling_id_slice, :] = output['normals'].detach()
            else:
                print('output is none for this slice')
            

            
        # pdb.set_trace()
        # print('spec max %.3f, spec mean %.3f, spec min %.3f' % (out_spec_albedo_k[mask_id].max(), out_spec_albedo_k[mask_id].mean(), out_spec_albedo_k[mask_id][out_spec_albedo_k[mask_id]>0].min()))
        # print('rough max %.3f, rough mean %.3f, rough min %.3f' % (out_spec_roughness_k[mask_id].max(), out_spec_roughness_k[mask_id].mean(), out_spec_roughness_k[mask_id][out_spec_roughness_k[mask_id]>0].min()))
        out_error_k = torch.clamp(torch.sum(torch.abs(out_rendered_k - out_gt_k), dim=1), 0, 1)
        save_albedo_img = (out_albedo_k.view(IH, IW, 3).cpu().numpy())
        save_shading_img = (out_shading_k.view(IH, IW, 3).cpu().numpy())
        save_spec_shading_img = (out_spec_shading_k.view(IH, IW, 3).cpu().numpy())
        save_spec_albedo_img = (out_spec_albedo_k.view(IH, IW, 3).cpu().numpy()) / spec_scale
        save_spec_roughness_img = (np.tile(out_spec_roughness_k.view(IH, IW, 1).cpu().numpy(), (1,1,3))) / rough_scale
        save_diffuse_color_img = (out_diffuse_color_k.view(IH, IW, 3).cpu().numpy())
        save_specular_color_img = (out_specular_color_k.view(IH, IW, 3).cpu().numpy())

        save_normal_img = ((out_normal_k.view(IH, IW, 3).cpu().numpy()) + 1) / 2
        save_cosine_img = ((out_cosine_k.view(IH, IW, 3).cpu().numpy()))
        save_cosine_img = ((save_cosine_img - 0.5)*2 - 0.5) * 2
        save_render_img = (out_rendered_k.view(IH, IW, 3).cpu().numpy())
        save_error_img = (matplotlib.colormaps['jet'](out_error_k.view(IH, IW).cpu().numpy()))
        save_error_img = save_error_img[:,:,:3]
        save_depth_np = (out_depth_k.view(IH,IW,1).cpu().numpy())
        save_depth_img = (out_depth_k.view(IH,IW,1).repeat(1,1,3).cpu().numpy())
        save_mask_img = (out_mask_k.view(IH,IW,1).repeat(1,1,3).cpu().numpy())
        save_gt_img = (out_gt_k.view(IH, IW, 3).cpu().numpy())
        save_1 = np.concatenate((save_gt_img, save_render_img, save_mask_img, save_error_img, ), axis=1) ## gt, pred, pred_mask, error
        save_2 = np.concatenate((save_normal_img, save_shading_img, save_albedo_img, save_diffuse_color_img, ), axis=1) # normal, shading, albedo, diffuse
        save_3 = np.concatenate((save_spec_shading_img, save_cosine_img, save_spec_albedo_img, save_spec_roughness_img ), axis=1)
        save_all = np.concatenate((save_1, save_2, save_3), axis=0)
        
        # del output

        save_gt_path = '%s/eval/gt_view_%d.png'% (render_dir, view_id)
        save_render_path = '%s/eval/render_pred_view_%d.png'% (render_dir, view_id)
        save_all_path = '%s/eval/all_view_%d.png'% (render_dir, view_id)
        save_albedo_path = '%s/brdf/diff_albedo_pred_view_%d.png'% (render_dir, view_id)
        save_spec_shading_path = '%s/brdf/spec_shading_pred_view_%d.png'% (render_dir, view_id)
        save_spec_albedo_path = '%s/brdf/spec_albedo_pred_view_%d.png'% (render_dir, view_id)
        save_spec_roughness_path = '%s/brdf/spec_roughness_pred_view_%d.png'% (render_dir, view_id)
        save_diffuse_color_path = '%s/brdf/diffuse_color_view_%d.png'% (render_dir, view_id)
        save_specular_color_path = '%s/brdf/specular_color_view_%d.png'% (render_dir, view_id)
        save_error_path = '%s/error_map/view_%d.png'% (render_dir, view_id)
        save_normal_path = '%s/geo/normal_pred_view_%d.png'% (render_dir, view_id)
        save_shading_path = '%s/geo/shading_pred_view_%d.png'% (render_dir, view_id)
        save_depth_np_path = '%s/geo/depth_pred_view_%d.npy'% (render_dir, view_id)
        save_depth_img_path = '%s/geo/depth_pred_view_%d.png'% (render_dir, view_id)
        save_mask_img_path = '%s/geo/mask_pred_view_%d.png'% (render_dir, view_id)

        os.makedirs('%s/error_map'%render_dir, exist_ok=True)
        os.makedirs('%s/eval'%render_dir, exist_ok=True)
        os.makedirs('%s/geo'%render_dir, exist_ok=True)
        os.makedirs('%s/brdf'%render_dir, exist_ok=True)

        Image.fromarray(to8b(save_gt_img)).save(save_gt_path)
        Image.fromarray(to8b(save_albedo_img)).save(save_albedo_path)
        Image.fromarray(to8b(save_shading_img)).save(save_shading_path)
        Image.fromarray(to8b(save_spec_shading_img)).save(save_spec_shading_path)
        Image.fromarray(to8b(save_spec_roughness_img)).save(save_spec_roughness_path)
        Image.fromarray(to8b(save_spec_albedo_img)).save(save_spec_albedo_path)
        Image.fromarray(to8b(save_diffuse_color_img)).save(save_diffuse_color_path)
        Image.fromarray(to8b(save_specular_color_img)).save(save_specular_color_path)
        Image.fromarray(to8b(save_render_img)).save(save_render_path)
        Image.fromarray(to8b(save_error_img)).save(save_error_path)
        Image.fromarray(to8b(save_normal_img)).save(save_normal_path)
        Image.fromarray(to8b(save_mask_img)).save(save_mask_img_path)

        Image.fromarray(to8b(save_all)).save(save_all_path)


        if with_alpha:
            render_dir_alpha = os.path.join(render_dir, 'alpha')
            os.makedirs(render_dir_alpha, exist_ok=True)
            mask_pred = Image.fromarray(to8b(save_mask_img)[:,:,0]).convert('L')
            mask_gt = Image.fromarray(to8b(masks_target[k].permute(1,2,0).cpu().numpy())[:,:,0]).convert('L')

            out_error_k = torch.clamp(torch.sum(torch.abs(out_rendered_k - out_gt_k), dim=1), 0, 1)
            save_albedo_img = (out_albedo_k.view(IH, IW, 3).cpu().numpy())
            save_shading_img = (out_shading_k.view(IH, IW, 3).cpu().numpy())
            save_spec_shading_img = (out_spec_shading_k.view(IH, IW, 3).cpu().numpy())
            save_spec_albedo_img = (out_spec_albedo_k.view(IH, IW, 3).cpu().numpy()) / spec_scale
            save_spec_roughness_img = (np.tile(out_spec_roughness_k.view(IH, IW, 1).cpu().numpy(), (1,1,3))) / rough_scale
            save_diffuse_color_img = (out_diffuse_color_k.view(IH, IW, 3).cpu().numpy())
            save_specular_color_img = (out_specular_color_k.view(IH, IW, 3).cpu().numpy())

            save_normal_img = ((out_normal_k.view(IH, IW, 3).cpu().numpy()) + 1) / 2
            save_cosine_img = ((out_cosine_k.view(IH, IW, 3).cpu().numpy()))
            save_cosine_img = ((save_cosine_img - 0.5)*2 - 0.5) * 2
            save_render_img = (out_rendered_k.view(IH, IW, 3).cpu().numpy())
            save_error_img = (matplotlib.colormaps['jet'](out_error_k.view(IH, IW).cpu().numpy()))
            save_error_img = save_error_img[:,:,:3]
            save_depth_np = (out_depth_k.view(IH,IW,1).cpu().numpy())
            save_depth_img = (out_depth_k.view(IH,IW,1).repeat(1,1,3).cpu().numpy())
            save_mask_img = (out_mask_k.view(IH,IW,1).repeat(1,1,3).cpu().numpy())
            save_gt_img = (out_gt_k.view(IH, IW, 3).cpu().numpy())
            save_1 = np.concatenate((save_gt_img, save_render_img, save_mask_img, save_error_img, ), axis=1) ## gt, pred, pred_mask, error
            save_2 = np.concatenate((save_normal_img, save_shading_img, save_albedo_img, save_diffuse_color_img, ), axis=1) # normal, shading, albedo, diffuse
            save_3 = np.concatenate((save_spec_shading_img, save_cosine_img, save_spec_albedo_img, save_spec_roughness_img ), axis=1)
            save_all = np.concatenate((save_1, save_2, save_3), axis=0)
            
            # del output

            save_gt_path = '%s/eval/gt_view_%d.png'% (render_dir_alpha, view_id)
            save_render_path = '%s/eval/render_pred_view_%d.png'% (render_dir_alpha, view_id)
            save_all_path = '%s/eval/all_view_%d.png'% (render_dir_alpha, view_id)
            save_albedo_path = '%s/brdf/diff_albedo_pred_view_%d.png'% (render_dir_alpha, view_id)
            save_spec_shading_path = '%s/brdf/spec_shading_pred_view_%d.png'% (render_dir_alpha, view_id)
            save_spec_albedo_path = '%s/brdf/spec_albedo_pred_view_%d.png'% (render_dir_alpha, view_id)
            save_spec_roughness_path = '%s/brdf/spec_roughness_pred_view_%d.png'% (render_dir_alpha, view_id)
            save_diffuse_color_path = '%s/brdf/diffuse_color_view_%d.png'% (render_dir_alpha, view_id)
            save_specular_color_path = '%s/brdf/specular_color_view_%d.png'% (render_dir_alpha, view_id)
            save_error_path = '%s/error_map/view_%d.png'% (render_dir_alpha, view_id)
            save_normal_path = '%s/geo/normal_pred_view_%d.png'% (render_dir_alpha, view_id)
            save_shading_path = '%s/geo/shading_pred_view_%d.png'% (render_dir_alpha, view_id)
            save_depth_np_path = '%s/geo/depth_pred_view_%d.npy'% (render_dir_alpha, view_id)
            save_depth_img_path = '%s/geo/depth_pred_view_%d.png'% (render_dir_alpha, view_id)
            save_mask_img_path = '%s/geo/mask_pred_view_%d.png'% (render_dir_alpha, view_id)

            os.makedirs('%s/error_map'%render_dir_alpha, exist_ok=True)
            os.makedirs('%s/eval'%render_dir_alpha, exist_ok=True)
            os.makedirs('%s/geo'%render_dir_alpha, exist_ok=True)
            os.makedirs('%s/brdf'%render_dir_alpha, exist_ok=True)

            aa = Image.fromarray(to8b(save_gt_img)); aa.putalpha(mask_gt); aa.save(save_gt_path)
            aa = Image.fromarray(to8b(save_albedo_img)); aa.putalpha(mask_pred);aa.save(save_albedo_path)
            aa = Image.fromarray(to8b(save_shading_img)); aa.putalpha(mask_pred);aa.save(save_shading_path)
            aa = Image.fromarray(to8b(save_spec_shading_img)); aa.putalpha(mask_pred);aa.save(save_spec_shading_path)
            aa = Image.fromarray(to8b(save_spec_roughness_img)); aa.putalpha(mask_pred);aa.save(save_spec_roughness_path)
            aa = Image.fromarray(to8b(save_spec_albedo_img)); aa.putalpha(mask_pred);aa.save(save_spec_albedo_path)
            aa = Image.fromarray(to8b(save_diffuse_color_img)); aa.putalpha(mask_pred);aa.save(save_diffuse_color_path)
            aa = Image.fromarray(to8b(save_specular_color_img)); aa.putalpha(mask_pred);aa.save(save_specular_color_path)
            aa = Image.fromarray(to8b(save_render_img)); aa.putalpha(mask_pred);aa.save(save_render_path)
            aa = Image.fromarray(to8b(save_error_img)); aa.putalpha(mask_pred);aa.save(save_error_path)
            aa = Image.fromarray(to8b(save_normal_img)); aa.putalpha(mask_pred);aa.save(save_normal_path)
            aa = Image.fromarray(to8b(save_mask_img)); aa.putalpha(mask_pred);aa.save(save_mask_img_path)

            aa = Image.fromarray(to8b(save_all)); aa.save(save_all_path)

        # np.save(save_depth_np_path, save_depth_np)

    renderer.debug=debug_flag
    renderer.ray_tracer.debug=debug_flag 
    torch.cuda.empty_cache()

def generate_mesh(opt, backbone, train_data, data_input, save_path_geo):
    backbone.eval()
    imgs_input = data_input['imgs_input']
    masks_input = data_input['masks_input']
    calibs_input = data_input['calibs_input']
    z_center = data_input['z_center']
    feed_extrin = data_input['feed_extrin']
    joints_3d = data_input['joints_3d'] if backbone.use_spatial else None
    if not os.path.exists(os.path.dirname(save_path_geo)) :
        os.makedirs(os.path.dirname(save_path_geo))
        pass

    with torch.no_grad():
        input_tensor = imgs_input
        if opt.feed_mask:
            input_tensor = torch.cat((input_tensor, masks_input), dim=1)
        
        im_feat_pred = backbone.filter(input_tensor)
    
    
    with torch.no_grad():
        base_sdf_func =  lambda x ,im_feats=im_feat_pred, calibs=calibs_input, masks=masks_input, z_center_world_space=z_center, extris=feed_extrin , use_poe=opt.use_positional_encoding, joints_3d=joints_3d\
            :backbone.query_sdf(im_feats, x, calibs, z_center_world_space, masks, extrinsic_reshape=extris, use_positional_encoding=use_poe, dilation_size=opt.dilate_size, joints_3d=joints_3d)
    
        base_albedo_func = lambda x ,im_feats=im_feat_pred, calibs=calibs_input, masks=masks_input, z_center_world_space=z_center, extris=feed_extrin , use_poe=opt.use_positional_encoding, joints_3d=joints_3d\
            :backbone.query_albedo(im_feats, x, calibs, z_center_world_space, masks, extrinsic_reshape=extris, use_positional_encoding=use_poe, joints_3d=joints_3d)
        
        base_rough_func = lambda x, im_feats=im_feat_pred, calibs=calibs_input, masks=masks_input, z_center_world_space=z_center, extris=feed_extrin , use_poe=opt.use_positional_encoding, joints_3d=joints_3d\
            :backbone.query_roughness(im_feats, x, calibs, z_center_world_space, masks, extrinsic_reshape=extris, use_positional_encoding=use_poe, joints_3d=joints_3d)
        
        base_spec_func = lambda x, im_feats=im_feat_pred, calibs=calibs_input, masks=masks_input, z_center_world_space=z_center, extris=feed_extrin , use_poe=opt.use_positional_encoding, joints_3d=joints_3d\
            :backbone.query_spec_albedo(im_feats, x, calibs, z_center_world_space, masks, extrinsic_reshape=extris, use_positional_encoding=use_poe, joints_3d=joints_3d)
        
        # base_rough_func = None
        # base_spec_func = None
        
        print('generate mesh color, using %d views' % im_feat_pred.shape[0])

        test_mesh_color_unified(opt, base_sdf_func, base_albedo_func, device, train_data, save_path_geo,\
            use_octree=opt.use_octree, levels=opt.output_levels,\
            has_gt_albedo=opt.tune_datatype=='syn', vis_error=opt.tune_datatype=='syn', export_uv=opt.export_uv, output_spec_roughness=False, rough_func=base_rough_func, spec_func=base_spec_func)

    torch.cuda.empty_cache()
    pass

def save_masks(masks_reshape, save_gt_img_path):
    
    save_img_list = []
    for v in range(masks_reshape.shape[0]):
        save_img = (np.transpose(masks_reshape[v].detach().cpu().numpy(), (1, 2, 0)) )
        save_img_list.append(save_img)
    save_gt_img = np.concatenate(save_img_list, axis=1)
    Image.fromarray(to8b(save_gt_img)).save(save_gt_img_path) 
    pass

def save_input_target_imgs(imgs_reshape, imgs_input, imgs_target, path_to_vis):
    
    save_img_list = []
    save_input_img_list = []
    save_target_img_list = []
    save_gt_img_path = '%s/imgs_gt.png'% (path_to_vis)
    save_input_img_path = '%s/imgs_input.png'% (path_to_vis)
    save_target_img_path = '%s/imgs_target.png'% (path_to_vis)
    for v in range(imgs_reshape.shape[0]):
        save_img = (np.transpose(imgs_reshape[v].detach().cpu().numpy(), (1, 2, 0)) )
        save_img_list.append(save_img)
    for v in range(imgs_input.shape[0]):
        save_img = (np.transpose(imgs_input[v].detach().cpu().numpy(), (1, 2, 0)) )
        save_input_img_list.append(save_img)
    for v in range(imgs_target.shape[0]):
        save_img = (np.transpose(imgs_target[v].detach().cpu().numpy(), (1, 2, 0)) )
        save_target_img_list.append(save_img)
        
    save_gt_img = np.concatenate(save_img_list, axis=1)
    save_input_img = np.concatenate(save_input_img_list, axis=1)
    save_target_img = np.concatenate(save_target_img_list, axis=1)
    Image.fromarray(to8b(save_gt_img)).save(save_gt_img_path) 
    Image.fromarray(to8b(save_input_img)).save(save_input_img_path) 
    Image.fromarray(to8b(save_target_img)).save(save_target_img_path) 

    pass

def random_shift(ids:torch.Tensor, H:int, W:int):
    '''
    random shift [+1, -1] pixel at x or y direction
    :param ids: tensor of ids
    :param ori_shape: tuple
    
    :return shifted ids: tensor of ids
    '''
    
    ### get shifted ids of rays 
    def _get_ihw_of_ids(ids, H, W):
        i = torch.div(ids, (H*W), rounding_mode='trunc')
        xy_in_i = ids % (H*W)
        h = torch.div(xy_in_i, W, rounding_mode='trunc')
        w = torch.div(xy_in_i, H, rounding_mode='trunc')
        return i,h,w
    
    def _get_id_from_ixy(i, h, w, H, W):
        id = i * H * W + h * W + w
        return id
    
    def _random_shift(i,h,w,H,W):
        offset_h = 2*(torch.rand(len(i)).cuda() > 0.5)-1
        offset_w = 2*(torch.rand(len(i)).cuda() > 0.5)-1
        h_shift = torch.clamp(h + offset_h, 0, H-1)
        w_shift = torch.clamp(w + offset_w, 0, W-1)
        assert torch.sum( h_shift >= H) + torch.sum( h_shift <0 )  == 0
        assert torch.sum( w_shift >= W) + torch.sum( w_shift <0 )  == 0
        return i,h_shift,w_shift

    i,h,w = _get_ihw_of_ids(ids, H, W)
    i,x_shift,y_shift = _random_shift(i,h,w, H,W)
    id_shift = _get_id_from_ixy(i, x_shift, y_shift, H, W)
    return id_shift

# def teaching_MLP_F_by_G(opt, path_to_F, path_to_vis, pretrained_ckpt_path=None, logger=None):
#     path_to_latest_G = path_to_ckpt + 'model_weights.tar'
#     save_param_to_json(opt, path_to_ckpt)
#     projection_mode = 'perspective' if opt.use_perspective else 'orthogonal'
#     backbone = UNet_unified(opt, base_views=opt.num_views, projection_mode=projection_mode).to(device)

#     surface_classifier = MLP()

def get_input_data_by_view(input_view, no_grad_view, imgs, masks, normal_mats, calibs, extris, z_center, joints_3d, scale_factor, linear_z=True):
    imgs_input = F.interpolate(imgs[input_view, ...].clone(), scale_factor=scale_factor, mode='bicubic', align_corners=True)
    masks_input =  F.interpolate(masks[input_view, ...].clone(), scale_factor=scale_factor, mode='bicubic', align_corners=True) 
    # imgs_input = imgs_input * 2
    # imgs_input = ttf.rgb_to_grayscale(imgs_input, 3) # gray
    # imgs_input = ttf.rgb_to_grayscale(imgs_input, 3) # gray
    # imgs_input = shading_reshape[input_view,...].clone()
    # imgs_input = albedo_reshape[input_view,...].clone() 
    # imgs_input = torch.ones_like(imgs_input) * masks_input  # 0/1
    # imgs_input = ttf.gaussian_blur(imgs_input, kernel_size=23)  # blur
    normal_mat_input = normal_mats[input_view, ...]
    calibs_input = calibs[input_view, ...]
    extrinsic_input = extris[input_view, ...]

    if linear_z:
        feed_extrin = extrinsic_input
    else:
        feed_extrin = None
    
    data_input = {}
    data_input['input_views'] = input_view
    data_input['no_grad_view'] = no_grad_view
    data_input['imgs_input'] = imgs_input
    data_input['masks_input'] = masks_input
    data_input['normal_mat_input'] = normal_mat_input
    data_input['calibs_input'] = calibs_input
    data_input['feed_extrin'] = feed_extrin
    data_input['z_center'] = z_center
    data_input['joints_3d'] = joints_3d
    
    return data_input

def get_target_data_by_view(target_view, view_weight, imgs, masks, masks_dilated, masks_intersect, front_back_intersections, ray_os, ray_ds, light_dirs, in_ori_mask):
    data_target = {}
    data_target['target_views'] = target_view
    data_target['target_view_weight'] = view_weight[target_view, ...]
    data_target['imgs_target'] = imgs[target_view, ...]
    data_target['masks_target'] = masks[target_view, ...]
    data_target['masks_dilate_target'] = masks_dilated[target_view, ...]
    data_target['masks_intersect_target'] = masks_intersect[target_view, ...]
    # data_target['masks_intersect_target'] = mask_intersect_erode[target_view, ...]
    data_target['front_back_intersections_target'] = front_back_intersections[target_view, ...]
    data_target['ray_os'] = ray_os[target_view, ...]
    data_target['ray_ds'] = ray_ds[target_view, ...]
    data_target['light_dirs'] = light_dirs[target_view, ...]
    data_target['in_ori_mask_target'] = in_ori_mask[target_view, ...]
    return data_target

def get_sample_rays_by_target_data(data_target, sample_num, IH, IW):
    dilated_mask_ids = torch.where(data_target['masks_intersect_target'].view(-1)>0.5)[0]
    # dilated_mask_ids = torch.where(mask_intersect_erode[target_view].view(-1)>0.5)[0]
    
    total_pixels = dilated_mask_ids.shape[0]
    # np.random.choice()
    sampling_idx = torch.randperm(total_pixels)[: sample_num]
    dilated_sampling_idx = dilated_mask_ids[sampling_idx]
    
    ray_os_sampled = data_target['ray_os'].contiguous().view(-1,3)[dilated_sampling_idx]           #  [BK,N,3] --> [BKN, 3] --> [N', 3]
    ray_ds_sampled = data_target['ray_ds'].contiguous().view(-1,3)[dilated_sampling_idx]           #  [BK,N,3] --> [BKN, 3] --> [N', 3]
    light_dirs_sampled = data_target['light_dirs'].contiguous().view(-1,3)[dilated_sampling_idx]           #  [BK,N,3] --> [BKN, 3] --> [N', 3]
    ray_weights_sampled = data_target['target_view_weight'].contiguous().view(-1,1)[dilated_sampling_idx] # [N']
    # target_mask_intersect_erode = mask_intersect_erode[target_view]
    ray_cs_sampled = data_target['imgs_target'].contiguous().view(-1,3,IH*IW).permute(0,2,1).contiguous().view(-1,3)[dilated_sampling_idx]     #  [BK,N,3] --> [BKN, 3] --> [N', 2]
    ray_ms_sampled = data_target['masks_target'].contiguous().view(-1,1,IH*IW).permute(0,2,1).contiguous().view(-1)[dilated_sampling_idx]    #  [BK,N,2] --> [BKN, 2] --> [N', 2]
    ray_in_ori_sampled = data_target['in_ori_mask_target'].contiguous().view(-1,1,IH*IW).permute(0,2,1).contiguous().view(-1)[dilated_sampling_idx] 
    front_back_dist_init_sampled = data_target['front_back_intersections_target'].contiguous().view(-1,2)[dilated_sampling_idx]     #  [BK,N,2] --> [BKN, 2] --> [N', 2]
    ray_mask_init_sampled = data_target['masks_intersect_target'].contiguous().view(-1)[dilated_sampling_idx]            #  [BK,N,1] --> [BKN] --> [N']

    sampled_rays = {}
    sampled_rays['ray_os_sampled'] = ray_os_sampled
    sampled_rays['ray_ds_sampled'] = ray_ds_sampled
    sampled_rays['light_dirs_sampled'] = light_dirs_sampled
    sampled_rays['ray_cs_sampled'] = ray_cs_sampled
    sampled_rays['ray_ms_sampled'] = ray_ms_sampled
    sampled_rays['ray_in_ori_sampled'] = ray_in_ori_sampled
    sampled_rays['ray_weights_sampled'] = ray_weights_sampled
    sampled_rays['front_back_dist_init_sampled'] = front_back_dist_init_sampled
    sampled_rays['ray_mask_init_sampled'] = ray_mask_init_sampled
    return sampled_rays

def fine_tuning(opt, path_to_ckpt, path_to_vis, pretrained_ckpt_path=None, logger=None):
    path_to_latest_G = path_to_ckpt + 'model_weights.tar'
    save_param_to_json(opt, path_to_ckpt)
    if opt.use_perspective:
        projection_mode = 'perspective'
        projection_method = perspective
    elif opt.use_CV_perspective:
        projection_mode = 'perspective_cv'
        projection_method = perspective_opencv
    else:
        projection_mode = 'orthogonal'
        projection_method = orthogonal
    backbone = UNet_unified(opt, base_views=opt.num_views, projection_mode=projection_mode).to(device)
    debug=False
    dr = DiffRenderer_unified(opt, backbone,  dr_num_views=1, use_indirect=opt.pred_indirect, device=device, debug=debug, path_to_vis=path_to_vis).to(device)
    
    lr_init_net = opt.lr_G
    lr_init_color = opt.lr_color
    lr_init_pose = opt.lr_pose
    lr_init_intensity = opt.lr_intensity
    lr_init_k = opt.lr_k
    n_sample_space = 200000

    train_data = load_data(opt, n_samp=n_sample_space, subject=opt.subject, real_data_dir=opt.finetune_real_data_train_dir, num_target=opt.num_target)

    ## calculate image feats
    imgs_reshape = train_data['img'].to(device)

    albedo_reshape = train_data['albedo'].to(device) if opt.tune_datatype == 'syn' else None
    shading_reshape = train_data['shading'].to(device) if opt.tune_datatype == 'syn' else None
    masks_reshape = train_data['mask'].to(device)
    in_ori_mask = train_data['in_ori_img'].to(device)
    in_ori_mask = in_ori_mask.unsqueeze(0).expand_as(masks_reshape)
    masks_dilate_reshape = train_data['mask_dilate'].to(device)
    z_center = train_data['z_center'].to(device)
    
    joints_3d = train_data['smpl_joints'].to(device) if opt.use_spatial else None
    init_pose = train_data['pose'].to(device) ## [K, 7]
    init_intensity = train_data['intensity'].to(device)
    '''
    optimize the image feature volumn and mlps
    '''
    if pretrained_ckpt_path is None:
        # initiate checkpoint if inexist
        print('No specified checkpoint...')
        if not os.path.isfile(path_to_latest_G):
            print('No continue train checkpoint...')
            print('Initiating new checkpoint...')
            pose_vec = nn.Parameter(init_pose.clone(), requires_grad=True) if not opt.noisy_pose else nn.Parameter(init_pose.clone()+torch.rand_like(init_pose) * 0.005, requires_grad=True) 
            intensity = nn.Parameter(init_intensity.clone(), requires_grad=True)
            save_generator( 0, [],backbone, 0, path_to_latest_G, opt_pose=opt.optimize_pose, pose_vec=pose_vec, intensity=intensity)
            print('Done...')
        else:
            print('use old checkpoint ... %s' % path_to_latest_G)
        print('Load checkpoint from ... %s' % path_to_latest_G)
        checkpoint_G = torch.load(path_to_latest_G, map_location=device)
    else:
        print('Load checkpoint from ... %s' % pretrained_ckpt_path)
        checkpoint_G = torch.load(pretrained_ckpt_path, map_location=device)

    current_model_dict = backbone.state_dict()
    new_state_dict={k:v if v.size()==current_model_dict[k].size() else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), checkpoint_G['G_state_dict'].values())}
    # pdb.set_trace()
    backbone.load_state_dict(new_state_dict, strict=True)
    # backbone.load_state_dict(checkpoint_G['G_state_dict'], strict=False)
    
    if 'intensity' in checkpoint_G:
        intensity = checkpoint_G['intensity']
    else:
        intensity = nn.Parameter(init_intensity.clone(), requires_grad=True)
    param_list = [
        {'params': list(backbone.surface_classifier.parameters()) + list(backbone.geo_transformer_encoder.parameters()) + list(backbone.image_filter.parameters()) , 'lr':lr_init_net},
        {'params': list(backbone.tex_transformer_encoder.parameters())+ list(backbone.albedo_predictor.parameters()) + list(backbone.spec_albedo_predictor.parameters()) + list(backbone.roughness_predictor.parameters()) , 'lr': lr_init_color},
        {'params': backbone.k, 'lr': lr_init_k},
        {'params': intensity, 'lr':lr_init_intensity},
        # {'params': backbone.spec_albedo_predictor.parameters(), 'lr': lr_init_color},
        # {'params': backbone.roughness_predictor.parameters(), 'lr': lr_init_color},
    ]
    if opt.optimize_pose:
        if 'pose' in checkpoint_G:
            ## no inhret from checkpoint
            pose_vec = checkpoint_G['pose']
            # pose_vec = nn.Parameter(init_pose.clone(), requires_grad=True) if not opt.noisy_pose else nn.Parameter(init_pose.clone()+torch.rand_like(init_pose) * 0.005, requires_grad=True) 
            pass
        else:
            pose_vec = nn.Parameter(init_pose.clone(), requires_grad=True) if not opt.noisy_pose else nn.Parameter(init_pose.clone()+torch.rand_like(init_pose) * 0.005, requires_grad=True) 
        
        # pose_vec = nn.Parameter(init_pose.clone(), requires_grad=True) if not opt.noisy_pose else nn.Parameter(init_pose.clone()+torch.rand_like(init_pose) * 0.005, requires_grad=True) 
        
        param_list+=[{'params': pose_vec, 'lr':lr_init_pose}]
    else:
        pose_vec = init_pose
    optimizerG = optim.Adam(param_list, amsgrad=False)
    epochCurrent = checkpoint_G['epoch']
    
    # optimizerG = optim.SGD(param_list, momentum=0.9)
    # pdb.set_trace()
    # extrinsic_inv = pose

    intrinsic = train_data['intrinsic'].to(device)
    # extrinsic = train_data['extrinsic'].to(device)
    calib = train_data['calib'].to(device)
    # c2w = train_data['c2w'].to(device)
    # w2c = train_data['w2c'].to(device)

    samples = train_data['samples'].float().to(device) 
    normal_matrices_reshape = train_data['normal_matrices'].to(device)
    view_ids = train_data['view_ids']
    
    B = 1
    K = imgs_reshape.shape[0]
    IW,IH = opt.load_size[0], opt.load_size[1]
    BK,_,iH,iW = imgs_reshape.shape
    assert(iH==IH)
    assert(iW==IW)
    assert(BK==B*K)
    # light_pos = train_data['light_pos'].to(device)
    # if opt.num_views > 4:
        # input_view = np.random.choice(input_view, size=4, replace=False)
    # target_view = [i for i in range(opt.num_views, imgs_reshape.shape[0])]
    target_view = [i for i in range(0, imgs_reshape.shape[0])]
    target_view_weight = [1.0 for _ in range(imgs_reshape.shape[0])]
    K_all = imgs_reshape.shape[0]
    
    # uvs = uv.expand(K_all, -1, -1) ## [K, HW, 2]
    uvs = get_uvs(opt, IW, IH).expand(K_all, -1, -1).to(device)

    ## cpu function to save gpu memory
    pose_vec.requires_grad_(False)
    calibs, extris, extris_inv = get_calib_extri_from_pose(pose_vec, intrinsic) ## get learnable calibs, extris, extris_inv
    # calibs, extris, extris_inv = calib, extrinsic, torch.inverse(extrinsic) ## get learnable calibs, extris, extris_inv
    ray_ds, cam_locs, light_dirs = get_camera_params_in_model_space(uvs.cpu(), extris_inv.cpu(), intrinsic.cpu(), neg_z=opt.use_perspective)
    # ray_ds, cam_locs = get_camera_params_in_model_space(uvs.cpu(), c2w.cpu(), intrinsic.cpu(), neg_z=opt.use_perspective)
    debug_dir = path_to_vis + '/debug2'
    for i in range(len(ray_ds)):
        fn = os.path.join(debug_dir, '%02d.ply'%i)
        pts = (ray_ds[i] * 0.1+ cam_locs[i][None,:])
        pts_color = torch.rand_like(pts[0:1]) * torch.ones_like(pts)* 255.0
        save_samples_color(fn, pts.cpu().numpy(), pts_color.cpu().numpy())


    if opt.image_proj_save_freq > 0:
        save_img_list = []
        save_img_path = '%s/debug/proj_%d.png'% (
                        path_to_vis, epochCurrent)
        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
        # xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, GRID_SIZE), np.linspace(-0.5, 0.5, GRID_SIZE),
        #                         np.linspace(-0.5, 0.5, GRID_SIZE))
        # points_grid = torch.from_numpy(np.stack((xx.flatten(), yy.flatten(), zz.flatten()))).float().unsqueeze(0)
        # points_grid =  samples[:,:,np.random.randint(n_sample_space, size=opt.num_sample_inout)]  if opt.num_sample_inout else None
        # points_to_proj = torch.tensor([0.0373989, -0.636125, 0.242863]).reshape(1,3,1)
        # points_to_proj = torch.tensor([-0.0263908, -0.7, 0.254911]).reshape(1,3,1)
        # points_to_proj = torch.tensor([-0.089, -0.4974, 0.09717]).reshape(1,3,1)
        points_to_proj_start1 = torch.tensor([0.04016, -0.41624, 0.16283]).reshape(1,3,1)
        points_to_proj_end1 = torch.tensor([-0.1794, -0.4703, 0.0586]).reshape(1,3,1)
        points_to_proj1 = [torch.lerp(points_to_proj_start1, points_to_proj_end1, i / 10) for i in range(11) ]
        points_to_proj1 = torch.cat(points_to_proj1, dim=-1)

        points_to_proj_start2 = torch.tensor([0.08175, -0.088, 0.126]).reshape(1,3,1)
        points_to_proj_end2 = torch.tensor([0.08175, 0.04325, 0.15671]).reshape(1,3,1)
        points_to_proj2 = [torch.lerp(points_to_proj_start2, points_to_proj_end2, i / 10) for i in range(11) ]
        points_to_proj2 = torch.cat(points_to_proj2, dim=-1)

        IW,IH = opt.load_size[0], opt.load_size[1]
        for v in range(imgs_reshape.shape[0]):
            save_img = (np.transpose(imgs_reshape[v].detach().cpu().numpy(), (1, 2, 0)) )
            calib_k = calibs[v:v+1,...]
            proj_func = perspective if opt.use_perspective else orthogonal
            projs_space1 = proj_func(points_to_proj1.to(device), calib_k)[0].detach().cpu().numpy()
            projs_space2 = proj_func(points_to_proj2.to(device), calib_k)[0].detach().cpu().numpy()
            space_xy1 = (np.array([IW,IH]).reshape(2,1)*(projs_space1[:2, :] * 0.5 + 0.5)) 
            space_xy2 = (np.array([IW,IH]).reshape(2,1)*(projs_space2[:2, :] * 0.5 + 0.5)) 
            ## point xy in wh order
            pil_save_img = Image.fromarray(to8b(save_img))
            draw = ImageDraw.Draw(pil_save_img)
            
            for n in range(projs_space1.shape[-1]):
                draw.point(tuple(space_xy1[:,n]), fill=(0,255,0))
            for n in range(projs_space2.shape[-1]):
                draw.point(tuple(space_xy2[:,n]), fill=(0,0,255))
                # draw.point(np.array([IW//2,IH//2]), fill=(0,255,0))
                # draw.point(tuple(np.array([IW//2,IH//2])), fill=(0,255,0))
                # draw.point((144.0,256.0), fill=(0,255,0))
            save_img_list.append(np.array(pil_save_img))
            
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(save_img).save(save_img_path) 

    ray_ds = ray_ds.to(device)
    light_dirs = light_dirs.to(device)
    cam_locs = cam_locs.to(device)
    N_rays = ray_ds.shape[1]
    ray_os = cam_locs.unsqueeze(1).expand(-1,N_rays,-1)
    light_dirs = light_dirs.expand(-1,N_rays,-1)

    print(view_ids)
    print('view_weights:', target_view_weight)
    # target_view_weight[1] = 0.0
    # target_view_weight[8] = 0.1
    # target_view_weight[9] = 10.0
    target_view_weight = torch.tensor(target_view_weight).float().to(device).unsqueeze(-1).expand(-1, IW * IH).contiguous()
    
    fb_intersection_path = os.path.join(path_to_ckpt, 'front_back_intersections.npy')
    mask_intersection_path = os.path.join(path_to_ckpt, 'mask_intersect.npy')
    mask_gt_path = os.path.join(path_to_ckpt, 'mask_gt.png')
    erode_mask_intersection_path = os.path.join(path_to_ckpt, 'mask_intersect_erode.npy')
    if opt.cal_vhull_intersection_online:
        front_back_intersections, mask_intersect = get_ray_visual_hull_intersection(
            cam_locs=cam_locs, ray_ds=ray_ds, calibs=calibs, 
            masks_dilated=masks_dilate_reshape, projection_method=projection_method,
            radius=opt.object_bounding_sphere, n_sample_per_ray=512, debug_vis_path=path_to_vis, sphere_intersection=False, ori_aspect=1)
        
        erode_mask = kornia.morphology.erosion(mask_intersect.reshape(BK, IH, IW, 1).permute(0,3,1,2).float(), torch.ones(3, 3).float().to(mask_intersect.device)).bool()
        erode_mask_save = erode_mask.reshape(BK, -1, 1)

        np.save(fb_intersection_path, front_back_intersections.cpu())
        np.save(mask_intersection_path, mask_intersect.cpu())
        np.save(erode_mask_intersection_path, erode_mask_save.cpu())
        
        ## for debug
        if True:
            for k in range(K_all):
                init_points = front_back_intersections[k,:,0:1] * ray_ds[k] + ray_os[k]
                mask_k = mask_intersect[k]
                debug_path = path_to_vis
                mask_pts = init_points[mask_k.squeeze(-1)]
                init_points_back = front_back_intersections[k,:,1:2] * ray_ds[k] + ray_os[k]
                mask_pts_back = init_points_back[mask_k.squeeze(-1)]
                mask_pts_color = torch.ones_like(mask_pts) * torch.rand_like(mask_pts[0:1,:])
                mask_pts_color_back = torch.ones_like(mask_pts_back)  * torch.rand_like(mask_pts[0:1,:])
                save_samples_color(debug_path + '/debug/view%d_back.ply' % k, mask_pts_back.cpu(), mask_pts_color_back.cpu()*255)
                save_samples_color(debug_path + '/debug/view%d_front.ply' % k, mask_pts.cpu(), mask_pts_color.cpu()*255)

        mask_all_intesect = mask_intersect.detach().cpu().numpy().reshape(BK, IH, IW, 1)
        ma_list = []
        for i in range(K_all):
            ma_list.append(mask_all_intesect[i])
        ma_img = np.concatenate(ma_list, axis=1)
        Image.fromarray(to8b(np.tile(ma_img, (1,1,3)))).save(mask_intersection_path.replace('.npy', '.png'))
        erode_mask = kornia.morphology.erosion(mask_intersect.reshape(BK, IH, IW, 1).permute(0,3,1,2).float(), torch.ones(3, 3).float().to(mask_intersect.device))
        erode_mask_all = erode_mask.detach().cpu().numpy().reshape(BK, IH, IW, 1)
        ma_list = []
        for i in range(K_all):
            ma_list.append(erode_mask_all[i])
        ma_img = np.concatenate(ma_list, axis=1)
        Image.fromarray(to8b(np.tile(ma_img, (1,1,3)))).save(mask_intersection_path.replace('.npy', '_erode.png'))
        save_masks(masks_reshape, mask_gt_path)
        pass
    
    front_back_intersections = torch.from_numpy(np.load(fb_intersection_path)).float().to(device)
    mask_intersect = torch.from_numpy(np.load(mask_intersection_path)).to(device)
    mask_intersect_erode = torch.from_numpy(np.load(erode_mask_intersection_path)).to(device) ## only used for depth smooth loss

    input_view = [i for i in range(opt.num_views)]
    scale_factor = 512 / max(IH,IW)
    data_input = get_input_data_by_view(input_view, [], imgs_reshape, masks_reshape, normal_matrices_reshape, calibs, extris, z_center, joints_3d, scale_factor, opt.use_linear_z)
    data_target = get_target_data_by_view(target_view, target_view_weight, imgs_reshape, masks_reshape, masks_dilate_reshape, mask_intersect, front_back_intersections, ray_os, ray_ds, light_dirs, in_ori_mask)
    save_input_target_imgs(imgs_reshape, imgs_reshape[input_view, ...], imgs_reshape[target_view, ...], path_to_vis)

    if opt.blur_target:
        kernel_size = 9
        sigma = 5
        target_masks_tensor = gaussian_blur(masks_reshape[target_view], kernel_size, sigma)
        target_imgs_tensor = gaussian_blur(imgs_reshape[target_view], kernel_size, sigma)
        save_img_list = []
        save_mask_list= []
        for v in range(target_imgs_tensor.shape[0]):
            save_img = (np.transpose(target_imgs_tensor[v,:3,...].expand(3,-1,-1).detach().cpu().numpy(), (1, 2, 0)))
            save_mask = (np.transpose(target_masks_tensor[v,:3,...].expand(3,-1,-1).detach().cpu().numpy(), (1, 2, 0)))
            save_img_list.append(save_img)
            save_mask_list.append(save_mask)
        save_img = np.concatenate(save_img_list, axis=1)
        save_mask = np.concatenate(save_mask_list, axis=1)
        Image.fromarray(to8b(save_img)).save(path_to_vis + 'im_blur.png')
        Image.fromarray(to8b(save_mask)).save(path_to_vis + 'ma_blur.png')
    if False:
        for k in range(K_all):
            init_points = front_back_intersections[k,:,0:1] * ray_ds[k] + ray_os[k]
            mask_k = mask_intersect[k]
            debug_path = path_to_vis
            mask_pts = init_points[mask_k.squeeze(-1)].detach()
            init_points_back = front_back_intersections[k,:,1:2] * ray_ds[k] + ray_os[k]
            mask_pts_back = init_points_back[mask_k.squeeze(-1)].detach()
            mask_pts_color = torch.ones_like(mask_pts) * torch.rand_like(mask_pts[0:1,:])
            mask_pts_color_back = torch.ones_like(mask_pts_back)  * torch.rand_like(mask_pts[0:1,:])
            save_samples_color(debug_path + '/debug/view%d_back.ply' % k, mask_pts_back.cpu(), mask_pts_color_back.cpu()*255)
            save_samples_color(debug_path + '/debug/view%d_front.ply' % k, mask_pts.cpu(), mask_pts_color.cpu()*255)
    # pdb.set_trace()
    # with torch.no_grad():
    #     backbone.eval()
    #     precompute_and_save_feats(opt, train_data, input_view, backbone, path_to_ckpt)


    if opt.precompute_img_feat:
        im_feat_pred = load_feat_from_disk(path_to_ckpt)
    else:
        im_feat_pred = None
    # print('load im feat from file, with shape:', im_feat_pred.shape)
    # print(im_feat_pred.min())
    
    if opt.freeze_img_filter:
        _freeze(backbone.image_filter)
        backbone.k.requires_grad = False
    
    if opt.freeze_albedo:
        _freeze(backbone.albedo_predictor)
        if opt.use_transformer:
            _freeze(backbone.tex_transformer_encoder)
    
    if opt.freeze_sdf:
        _freeze(backbone.surface_classifier)
        if opt.use_transformer:
            _freeze(backbone.geo_transformer_encoder)
    
    if opt.freeze_indirect and opt.pred_indirect:
        _freeze(backbone.indirect_predictor)

    if opt.no_grad:
        _freeze(backbone.image_filter)
        _freeze(backbone.albedo_predictor)
        _freeze(backbone.surface_classifier)
        if opt.pred_indirect:
            _freeze(backbone.indirect_predictor)
        backbone.k.requires_grad = False



    pytorch_total_params = sum(p.numel() for p in backbone.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    if opt.lambda_reg_finetune > 0:
        fixed_backbone = type(backbone)(opt, base_views=opt.num_views, projection_mode=projection_mode).to(device)
        fixed_backbone.load_state_dict(backbone.state_dict())
        fixed_backbone.eval()
    
    if debug:
        logger.info('image filter:')
        logger.info(backbone.image_filter)
        
        logger.info('albedo_classifier:')
        logger.info(backbone.albedo_predictor)

        logger.info('light regressor:')
        logger.info(backbone.light_filter)
        logger.info('Model params: %d, trainable params: %d' %(pytorch_total_params, pytorch_total_trainable_params) )

    
    


    if opt.render_init_imgs:
        # with torch.no_grad():
        render_dir = os.path.join(path_to_vis, 'rendered_imgs/epoch%d_init/'% (epochCurrent))
        render_imgs(backbone, dr, data_input, data_target, intensity, IH, IW, render_dir, cal_diff_normal=False, use_TSDF=opt.use_TSDF, TSDF_thres=opt.TSDF_thres)

    if opt.gen_init_mesh:
        save_path_geo = '%s/geometry/test_eval_gt_%s_geo.obj' % (
            path_to_vis, train_data['name'])
        generate_mesh(opt, backbone, train_data, data_input, save_path_geo)
    

    n_iters = 0
    loss_mean_num = 10
    epoch_loss_G = np.zeros(loss_mean_num)
    torch.autograd.set_detect_anomaly(True)
    precomputed_indirect = opt.precomputed_indirect
    ## prepare sampled rays from target views

    lambda_reg = opt.lambda_reg
    lambda_mask = opt.lambda_mask
    lambda_align = opt.lambda_align
    lambda_reg_alpha = opt.lambda_reg_alpha
    lambda_mask_alpha = opt.lambda_mask_alpha
    lambda_reg_milestone = opt.lambda_reg_milestone
    lambda_mask_milestone = opt.lambda_mask_milestone

    unfreeze_epoch = 20001

    for i in range(epochCurrent):
        if i in lambda_reg_milestone:
            lambda_reg = lambda_reg * lambda_reg_alpha
        if i in lambda_mask_milestone:
            lambda_mask = lambda_mask * lambda_mask_alpha
    
    ## for mask loss alpha
    alpha = 200
    alpha_factor = 2
    alpha_milestones = [500, 1000, 1500, 2000]
    for acc in alpha_milestones:
        if epochCurrent > acc:
            alpha = alpha * alpha_factor
    
    render_img_freq = 50
    
    # _freeze(backbone)
    # _unfreeze(backbone.albedo_predictor)
    # _unfreeze(backbone.tex_transformer_encoder)
    for epoch in tqdm(range(epochCurrent, epochCurrent + opt.epochs)):
        # if epoch - epochCurrent > 49:
        #     _unfreeze(backbone.surface_classifier)
        #     _unfreeze(backbone.geo_transformer_encoder)
        # if epoch - epochCurrent > 99:
        #     _unfreeze(backbone)
        
        
        if opt.progressive_train:
            if epoch <= 200:
                _freeze(backbone.image_filter)
                _unfreeze(backbone.image_filter.down4)
                _unfreeze(backbone.image_filter.up3)
            elif epoch < 500:
                _freeze(backbone.image_filter)
                _unfreeze(backbone.image_filter.down3)
                _unfreeze(backbone.image_filter.up4)
            elif epoch < 1000:
                _freeze(backbone.image_filter)
                _unfreeze(backbone.image_filter.down2)
                _unfreeze(backbone.image_filter.up5)
            elif epoch < 1500:
                _freeze(backbone.image_filter)
                _unfreeze(backbone.image_filter.down1)  
                _unfreeze(backbone.image_filter.up6)
            elif epoch < 2000:
                _freeze(backbone.image_filter)
                _unfreeze(backbone.image_filter.inc)
                _unfreeze(backbone.image_filter.outc)
            pass
        # if epoch==868:
        #     pdb.set_trace()
        """ Training start """
        if epoch in alpha_milestones:
            alpha = alpha * alpha_factor
        if epoch in lambda_mask_milestone:
            lambda_mask = lambda_mask * lambda_mask_alpha
        if epoch in lambda_reg_milestone:
            lambda_reg = lambda_reg * lambda_reg_alpha


        if opt.optimize_pose and epoch-epochCurrent > opt.opt_pose_epoch:
            pose_vec.requires_grad_(True)
            ## only if we optimize pose should we re-calculate the camera matrix from a different pose_vec
            calibs, extris, extris_inv = get_calib_extri_from_pose(pose_vec, intrinsic) ## get learnable calibs, extris, extris_inv
            ray_ds, cam_locs, light_dirs = get_camera_params_in_model_space(uvs, extris_inv, intrinsic, neg_z=opt.use_perspective)
            N_rays = ray_ds.shape[1]
            ray_os = cam_locs.unsqueeze(1).expand(-1,N_rays,-1)
            light_dirs = light_dirs.expand(-1,N_rays,-1)
            # front_back_intersections, mask_intersect = get_ray_visual_hull_intersection(
            #     cam_locs=cam_locs, ray_ds=ray_ds, calibs=calibs, 
            #     masks_dilated=masks_dilate_reshape,
            #     radius=opt.object_bounding_sphere, n_sample_per_ray=512, debug_vis_path=path_to_vis, sphere_intersection=True)
        
        N_all = len(imgs_reshape)
        all_view = list(range(N_all))

        # input_view = np.random.choice(all_view, size=opt.num_views, replace=False)
        # target_view = list(set(all_view) - set(input_view))

        input_view = list(range(opt.num_views))
        target_view = all_view

        # print('input view:', input_view, 'target view:', target_view)

        k_no_grad = []
        if opt.selected_train:
            MAX_GRAD_VIEW = 4
            if opt.num_views > MAX_GRAD_VIEW:
                k_no_grad = np.random.choice(list(range(opt.num_views)), size=opt.num_views-MAX_GRAD_VIEW, replace=False)
            
        data_input = get_input_data_by_view(input_view, k_no_grad, imgs_reshape, masks_reshape, normal_matrices_reshape, calibs, extris, z_center, joints_3d, scale_factor, opt.use_linear_z)
        data_target = get_target_data_by_view(target_view,target_view_weight, imgs_reshape, masks_reshape, masks_dilate_reshape, mask_intersect, front_back_intersections, ray_os, ray_ds, light_dirs, in_ori_mask)
        
        n_masks = 4
        srcs_input = imgs_reshape[input_view, ...].clone()
        # mask_indices = input_view[-n_masks:]
        # mask_indices = random.sample(input_view[2:], n_masks)
        # srcs_input[mask_indices] = torch.zeros_like(imgs_input[mask_indices])

        if not opt.precompute_img_feat:
            im_feat_pred = backbone.filter(srcs_input, k_no_grad)

        if opt.no_grad:
            backbone.eval()
            dr.eval()
        else:
            backbone.train()
            dr.train()
        
        # decay_rate = 1
        decay_rate_geo = 0.1
        decay_rate = 0.1
        # decay_step = 500
        decay_step = 2000
        decay_val = max(decay_rate, decay_rate ** (epoch / decay_step))
        decay_val_geo = max(decay_rate_geo, decay_rate_geo ** (epoch / decay_step))
        # new_lr_net = lr_init_net * decay_val
        new_lr_net = lr_init_net * decay_val_geo
        set_learning_rate(optimizerG, 0, new_lr_net)
        new_lr_color = lr_init_color * decay_val
        set_learning_rate(optimizerG, 1, new_lr_color)
        new_lr_intensity = lr_init_intensity * decay_val
        set_learning_rate(optimizerG, 3, new_lr_intensity)
        if opt.optimize_pose:
            new_lr_pose = lr_init_pose * decay_val
            set_learning_rate(optimizerG, 4, new_lr_pose)

        if epoch-epochCurrent > 200:
            use_mask_sample = True
        
        


        error_albedo = torch.zeros(1).to(device)
        loss_total = torch.zeros(1).to(device)

        sample_num = opt.num_sample_dr
        sample_idx = None
        
        if not opt.sample_patch:
            sampled_rays = get_sample_rays_by_target_data(data_target, sample_num, IH, IW)

            ray_os_sampled = sampled_rays['ray_os_sampled']
            ray_ds_sampled = sampled_rays['ray_ds_sampled']
            light_dirs_sampled = sampled_rays['light_dirs_sampled']
            ray_cs_sampled = sampled_rays['ray_cs_sampled']
            ray_ms_sampled = sampled_rays['ray_ms_sampled']
            ray_in_ori_sampled = sampled_rays['ray_in_ori_sampled']
            ray_weights_sampled = sampled_rays['ray_weights_sampled']
            front_back_dist_init_sampled = sampled_rays['front_back_dist_init_sampled']
            ray_mask_init_sampled = sampled_rays['ray_mask_init_sampled']


            # dilated_mask_ids = torch.where(mask_intersect[target_view].view(-1)>0.5)[0]
            # # dilated_mask_ids = torch.where(mask_intersect_erode[target_view].view(-1)>0.5)[0]
            
            # total_pixels = dilated_mask_ids.shape[0]
            # # np.random.choice()
            # sampling_idx = torch.randperm(total_pixels)[: sample_num]
            # dilated_sampling_idx = dilated_mask_ids[sampling_idx]
            
            # ray_os_sampled = ray_os[target_view].contiguous().view(-1,3)[dilated_sampling_idx]           #  [BK,N,3] --> [BKN, 3] --> [N', 3]
            # ray_ds_sampled = ray_ds[target_view].contiguous().view(-1,3)[dilated_sampling_idx]           #  [BK,N,3] --> [BKN, 3] --> [N', 3]
            # ray_weights = target_view_weight[target_view].contiguous().view(-1,1)[dilated_sampling_idx] # [N']
            # ray_in_ori_tensor = in_ori_mask[target_view]
            # target_masks_tensor = masks_reshape[target_view]
            # target_imgs_tensor = imgs_reshape[target_view]
            # target_fb_intersect = front_back_intersections[target_view]
            # target_mask_intersect = mask_intersect[target_view]
            
            # # target_mask_intersect_erode = mask_intersect_erode[target_view]
            # ray_cs_sampled = target_imgs_tensor.contiguous().view(-1,3,IH*IW).permute(0,2,1).contiguous().view(-1,3)[dilated_sampling_idx]     #  [BK,N,3] --> [BKN, 3] --> [N', 2]
            # ray_ms_sampled = target_masks_tensor.contiguous().view(-1,1,IH*IW).permute(0,2,1).contiguous().view(-1)[dilated_sampling_idx]    #  [BK,N,2] --> [BKN, 2] --> [N', 2]
            # ray_in_ori_sampled = ray_in_ori_tensor.contiguous().view(-1,1,IH*IW).permute(0,2,1).contiguous().view(-1)[dilated_sampling_idx] 
            # front_back_dist_init_sampled = target_fb_intersect.contiguous().view(-1,2)[dilated_sampling_idx]     #  [BK,N,2] --> [BKN, 2] --> [N', 2]
            # ray_mask_init_sampled = target_mask_intersect.contiguous().view(-1)[dilated_sampling_idx]            #  [BK,N,1] --> [BKN] --> [N']

            if opt.use_depth_smooth_loss:
                dilated_sampling_idx_shift = random_shift(dilated_sampling_idx, IH, IW)
                ray_os_sampled_shift = ray_os[target_view].contiguous().view(-1,3)[dilated_sampling_idx_shift]           #  [BK,N,3] --> [BKN, 3] --> [N', 3]
                ray_ds_sampled_shift = ray_ds[target_view].contiguous().view(-1,3)[dilated_sampling_idx_shift]           #  [BK,N,3] --> [BKN, 3] --> [N', 3]
                ray_weights_shift = target_view_weight[target_view].contiguous().view(-1,1)[dilated_sampling_idx_shift] # [N']
                ray_cs_sampled_shift = target_imgs_tensor.contiguous().view(-1,3,IH*IW).permute(0,2,1).contiguous().view(-1,3)[dilated_sampling_idx_shift]     #  [BK,N,3] --> [BKN, 3] --> [N', 2]
                ray_ms_sampled_shift = target_masks_tensor.contiguous().view(-1,1,IH*IW).permute(0,2,1).contiguous().view(-1)[dilated_sampling_idx_shift]    #  [BK,N,2] --> [BKN, 2] --> [N', 2]
                ray_in_ori_sampled_shift = ray_in_ori_tensor.contiguous().view(-1,1,IH*IW).permute(0,2,1).contiguous().view(-1)[dilated_sampling_idx_shift] 
                front_back_dist_init_sampled_shift = target_fb_intersect.contiguous().view(-1,2)[dilated_sampling_idx_shift]     #  [BK,N,2] --> [BKN, 2] --> [N', 2]
                ray_mask_init_sampled_shift = target_mask_intersect.contiguous().view(-1)[dilated_sampling_idx_shift]            #  [BK,N,1] --> [BKN] --> [N']

                ray_os_sampled = torch.cat((ray_os_sampled, ray_os_sampled_shift), dim=0)
                ray_ds_sampled = torch.cat((ray_ds_sampled, ray_ds_sampled_shift), dim=0)
                ray_weights_sampled = torch.cat((ray_weights_sampled, ray_weights_shift), dim=0)
                ray_cs_sampled = torch.cat((ray_cs_sampled, ray_cs_sampled_shift), dim=0)
                ray_ms_sampled = torch.cat((ray_ms_sampled, ray_ms_sampled_shift), dim=0)
                ray_in_ori_sampled = torch.cat((ray_in_ori_sampled, ray_in_ori_sampled_shift), dim=0)
                front_back_dist_init_sampled = torch.cat((front_back_dist_init_sampled, front_back_dist_init_sampled_shift), dim=0)
                ray_mask_init_sampled = torch.cat((ray_mask_init_sampled, ray_mask_init_sampled_shift), dim=0)
        else:
            # crop a patch by center random drawn from gt mask
            w0, h0 = int(np.sqrt(opt.num_sample_dr)), int(np.sqrt(opt.num_sample_dr))
            
            view0 = random.choice(target_view)
            cropped_idx, [cx,cy], [tt,bb,ll,rr] = get_random_crop(masks_reshape[view0][0], w0, h0)
            ray_os_sampled = ray_os[view0:view0+1].contiguous().view(-1,3)[cropped_idx]           #  [BK,N,3] --> [BKN, 3] --> [N', 3]
            ray_ds_sampled = ray_ds[view0:view0+1].contiguous().view(-1,3)[cropped_idx]           #  [BK,N,3] --> [BKN, 3] --> [N', 3]
            ray_weights_sampled = target_view_weight[view0:view0+1].contiguous().view(-1,1)[cropped_idx] # [N']

            ray_in_ori_tensor = in_ori_mask[view0:view0+1]
            target_masks_tensor = masks_reshape[view0:view0+1]
            target_imgs_tensor = imgs_reshape[view0:view0+1]
            target_mask_intersect = mask_intersect[view0:view0+1]
            target_fb_intersect = front_back_intersections[view0:view0+1]
        
            ray_cs_sampled = target_imgs_tensor.contiguous().view(-1,3,IH*IW).permute(0,2,1).contiguous().view(-1,3)[cropped_idx]     #  [BK,N,3] --> [BKN, 3] --> [N', 2]
            ray_ms_sampled = target_masks_tensor.contiguous().view(-1,1,IH*IW).permute(0,2,1).contiguous().view(-1)[cropped_idx]    #  [BK,N,2] --> [BKN, 2] --> [N', 2]
            ray_in_ori_sampled = ray_in_ori_tensor.contiguous().view(-1,1,IH*IW).permute(0,2,1).contiguous().view(-1)[cropped_idx] 
            front_back_dist_init_sampled = target_fb_intersect.contiguous().view(-1,2)[cropped_idx]     #  [BK,N,2] --> [BKN, 2] --> [N', 2]
            ray_mask_init_sampled = target_mask_intersect.contiguous().view(-1)[cropped_idx]            #  [BK,N,1] --> [BKN] --> [N']
        t1 = time.time()
        ## stage 1

        ## update the calibs by quat
        
        save_intermediate = True
        output = dr.forward(data_input=data_input, src_img_feats=im_feat_pred, 
            tgt_views=target_view, ray_os=ray_os_sampled, ray_ds=ray_ds_sampled, ray_cs=ray_cs_sampled, ray_ms=ray_ms_sampled, light_dirs=light_dirs_sampled,
            front_back_dist_init=front_back_dist_init_sampled, ray_mask_init=ray_mask_init_sampled, cal_diff_normal=opt.cal_diff_normal,no_grad=opt.no_grad, intensity=intensity,
            jittor_std=opt.eik_std, use_poe=opt.use_positional_encoding, use_TSDF=opt.use_TSDF, TSDF_thres=opt.TSDF_thres, epoch=epoch, save_intermediate=save_intermediate
            )

        sdf_func, albedo_func, gradient_func, spec_func, rough_func = dr.get_sdf_albedo_gradient_funcs(im_feat_pred, data_input, opt.cal_diff_normal,use_poe=opt.use_positional_encoding)
        
        t2 = time.time()

        if epoch - epochCurrent > 200:
            include_smooth_loss = True
        else:
            include_smooth_loss = False
        
        
        if output is None:
            loss, loss_str = torch.zeros_like(error_albedo), ''
        else:
            dfp = output['diff_points'].detach()
            dfs = output['diffuse_shadings'].detach()


            loss, loss_dict, loss_str = dr.cal_loss(output, sdf_func, gradient_func,albedo_func,spec_func, rough_func, \
                ray_weights=ray_weights_sampled,
                ray_in_original_img=ray_in_ori_sampled, 
                epoch=epoch,
                include_smooth_loss=include_smooth_loss,
                precomputed_indirect=precomputed_indirect, 
                alpha=alpha,
                lambda_reg=lambda_reg, 
                lambda_mask=lambda_mask, 
                lambda_align=lambda_align, writer=writer, save_intermediate=save_intermediate)

            
            if epoch > 1000 and loss_dict['mask_loss'] > 0.2:
                render_dir = os.path.join(path_to_vis, 'rendered_imgs/epoch%d/'% (epoch))
                render_imgs(backbone, dr, data_input, data_target, intensity, IH, IW, render_dir,cal_diff_normal=False, use_TSDF=opt.use_TSDF, TSDF_thres=opt.TSDF_thres)
                save_path_geo = '%s/geometry/epoch_%d/epoch_%d.obj' % (path_to_vis, epoch, epoch)
                generate_mesh(opt, backbone, train_data, data_input, save_path_geo)
                
        # writer.add_scalar('Net/Filter/layer1.weight.mean', backbone.image_filter.inc.)
        loss_total = loss_total + loss
        # if opt.lambda_reg_finetune > 0:
        #     surface_pts = output['diff_points'].detach().view(B, -1 , 3).transpose(1, 2)
        #     samples_space =  samples[:,:,np.random.randint(n_sample_space, size=opt.num_sample_inout)]  if opt.num_sample_inout else None
            
        #     xyz = perspective(samples_space.expand(opt.num_views,-1,-1), calibs_input)
        #     in_hull = in_visual_hull(xyz, masks_input, opt.num_views)
        #     samples_space = samples_space.permute(0,2,1).reshape(-1,3)[in_hull.reshape(-1)].T.unsqueeze(0)
        #     with torch.no_grad():
        #         im_feat, _ = fixed_backbone.filter(imgs_input)
        #         space_sdf_psedo_gt, _ = fixed_backbone.query_sdf(im_feat, samples_space, calibs_input,z_center_world_space=z_center, masks=masks_input, extrinsic_reshape=feed_extrin)
        #         # surface_sdf_psedo_gt, _ = fixed_backbone.query_sdf(im_feat, surface_pts, calibs_input,z_center_world_space=z_center, masks=masks_input, extrinsic_reshape=feed_extrin)
        #     space_sdf_pred, _ = backbone.query_sdf(im_feat_pred, samples_space, calibs_input,z_center_world_space=z_center, masks= masks_input, extrinsic_reshape=feed_extrin)
        #     # surface_sdf_pred, _ = backbone.query_sdf(im_feat_pred, surface_pts, calibs_input,z_center_world_space=z_center, masks= masks_input, extrinsic_reshape=feed_extrin)
        #     criterion_reg = nn.MSELoss()
        #     regular_loss = criterion_reg(space_sdf_pred, space_sdf_psedo_gt)  * opt.lambda_reg_finetune
        #                     # criterion_reg(surface_sdf_pred, surface_sdf_psedo_gt)  * opt.lambda_reg_finetune 
        #     loss_total = loss_total + regular_loss
        
        
        criterion_pose = nn.L1Loss()
        pose_reg_loss = criterion_pose(torch.cat((F.normalize(pose_vec[:,:4]), pose_vec[:,4:]), dim=1), init_pose)
        if opt.lambda_pose>0:
            loss_total = loss_total + pose_reg_loss * opt.lambda_pose
        # error_geo, geo_error_dict = backbone.forward(imgs_reshape, samples, calibs_reshape, z_center_world_space=z_center, \
        #     im_feat=im_feat_pred, smpl_feat=samples_smpl_sdf, light_feat=light_feat_pred, masks=masks_reshape, labels_space=labels, \
        # points_surf=samples_surf, labels_surf=samples_surf_normal, name=name, vids=view_ids)
        
        # loss_total += error_geo
        epoch_loss_G[epoch % loss_mean_num] = loss_total.item()
        epoch_loss_G_moving = epoch_loss_G.mean()

        if opt.optimize_pose:
            post_fix_str = 'Ep%d,lr_n=%.3f*10-4,lr_i=%.3f*10-3,lr_p=%.3f*10-3,lo_mov=%.4f,lo=%.4f,k=%.4f' %(epoch, new_lr_net*10000, new_lr_intensity*1000, new_lr_pose*1000, epoch_loss_G_moving,loss.item(), backbone.k.item())
        else:
            post_fix_str = 'Ep%d,lr_n=%.3f*10-4,lr_i=%.3f*10-3,lo_mov=%.4f,lo=%.4f,k=%.4f' %(epoch, new_lr_net*10000, new_lr_intensity*1000, epoch_loss_G_moving,loss.item(), backbone.k.item())


        # if opt.lambda_reg_finetune > 0:
        #     post_fix_str += ',reg=%.6f' % ( regular_loss.item() )
        
        post_fix_str += ',' + loss_str
        post_fix_str += ',inten: %.3f' % (intensity)

        # post_fix_str += ',lc:[%.2f,%.2f,%.2f]' % (dr.light_color.data[0],dr.light_color.data[1], dr.light_color.data[2])

        if opt.optimize_pose:
            pose_err = nn.L1Loss(reduction='none')(torch.cat((F.normalize(pose_vec[:,:4]), pose_vec[:,4:]), dim=1).detach(), init_pose)
            # pose_err_str = 'pose err:' + str(pose_err)
            # logger.info(pose_err_str)

            post_fix_str += ',pose_reg_loss: %.3f*10-3' % (pose_reg_loss.item() * 1000)
        # post_fix_str += ',pose_opt: ', (pose_vec.data[0])
        # post_fix_str += ' l%.3f, %.3f, %.3f' % (dr.light_offset.data[0],dr.light_offset.data[1],dr.light_offset.data[2])
        # post_fix_str = 'L:mov=%.4f, im=%.4f, albedo=%.4f, geo=%.4f, ssdf=%.4f,snor=%.4f,bce=%.4f,reg=%.4f, lr=%.5f, k:%.3f'\
        #     %(epoch_loss_G_moving, image_loss.item(), error_albedo.item(),\
        # error_geo.item(), geo_error_dict['surface_sdf_error'].item(),geo_error_dict['surface_normal_error'].item(),\
        # geo_error_dict['bce_error'].item(), geo_error_dict['reg_error'].item(), lr, backbone.k)

        # if len(output) > 0:
        # pdb.set_trace()
        if not opt.no_grad:
            optimizerG.zero_grad()
            loss_total.backward(retain_graph=False)
            optimizerG.step()

        logger.info(post_fix_str)
        
        if epoch-epochCurrent in [100, 200, 400, 1000, 2000]:
            render_img_freq = min((epoch-epochCurrent)//2, opt.render_img_freq)

        if opt.render_img_freq > 0 and epoch % render_img_freq == 0:
            '''
            render sdf to image using ray tracing
            '''
            # with torch.no_grad():
            # input_view = all_view
            target_view = all_view
            data_input = get_input_data_by_view(input_view, [], imgs_reshape, masks_reshape, normal_matrices_reshape, calibs, extris, z_center, joints_3d, scale_factor, opt.use_linear_z)
            data_target = get_target_data_by_view(target_view,target_view_weight, imgs_reshape, masks_reshape, masks_dilate_reshape, mask_intersect, front_back_intersections, ray_os, ray_ds, light_dirs, in_ori_mask)
        
            render_dir = os.path.join(path_to_vis, 'rendered_imgs/epoch%d/'% (epoch))
            render_imgs(backbone, dr, data_input, data_target, intensity, IH, IW, render_dir,cal_diff_normal=False, use_TSDF=opt.use_TSDF, TSDF_thres=opt.TSDF_thres)
            pass

        if opt.gen_feat_freq > 0 and epoch % opt.gen_feat_freq==0:
            # backbone.eval()
            # with torch.no_grad():
            #     im_feat_pred = backbone.filter(imgs_input)
            save_path_feat = '%s/feat/epoch_%d/feat.png' % (path_to_vis, epoch)
            os.makedirs(os.path.dirname(save_path_feat), exist_ok=True)
            save_img_list = []

            for v in range(im_feat_pred.shape[0]):
                save_path_feat = '%s/feat/epoch_%d/feat_%d.png' % (path_to_vis, epoch, v)
                save_img = (np.transpose(im_feat_pred[v,:3,...].clamp(-1,1).detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)
                Image.fromarray(to8b(save_img)).save(save_path_feat)
                save_img_list.append(save_img)
            save_img = np.concatenate(save_img_list, axis=1)
            save_path_feat = '%s/feat/epoch_%d/feat.png' % (path_to_vis, epoch)
            Image.fromarray(to8b(save_img)).save(save_path_feat)


        if opt.gen_mesh_freq > 0  and epoch % opt.gen_mesh_freq == 0:
            # input_view = all_view
            data_input = get_input_data_by_view(input_view, [], imgs_reshape, masks_reshape, normal_matrices_reshape, calibs, extris, z_center, joints_3d, scale_factor, opt.use_linear_z)
        
            save_path_geo = '%s/geometry/epoch_%d/epoch_%d.obj' % (
                    path_to_vis, epoch, epoch)
            generate_mesh(opt, backbone, train_data, data_input, save_path_geo)
        
                
        if opt.model_save_freq > 0 and epoch % opt.model_save_freq == 0:
            path_to_save_G = path_to_ckpt + 'epoch_{}_G.tar'.format(epoch)
            save_generator( epoch, [], backbone, 0, path_to_save_G, opt_pose=opt.optimize_pose, pose_vec=pose_vec, intensity=intensity)
            save_generator( epoch, [], backbone, 0, path_to_latest_G, opt_pose=opt.optimize_pose, pose_vec=pose_vec, intensity=intensity)

def render_views(opt, train_data, test_data, output_dir, test_ckpt_path):
    
    ckpt_name = os.path.basename(test_ckpt_path).split('.')[0]
    output_dir = os.path.join(output_dir, ckpt_name)
    save_path_geo = os.path.join(output_dir, ckpt_name, 'pred.obj')
    os.makedirs(output_dir, exist_ok=True)
    if opt.use_perspective:
        projection_mode = 'perspective'
        projection_method = perspective
    elif opt.use_CV_perspective:
        projection_mode = 'perspective_cv'
        projection_method = perspective_opencv
    else:
        projection_mode = 'orthogonal'
        projection_method = orthogonal
    backbone = UNet_unified(opt, base_views=opt.num_views, projection_mode=projection_mode).to(device)
    dr = DiffRenderer_unified(opt, backbone,  dr_num_views=1, use_indirect=False, device=device, debug=False, path_to_vis=output_dir).to(device)
    
    ckpt = torch.load(test_ckpt_path, map_location=device)

    # current_model_dict = backbone.state_dict()
    # new_state_dict={k:v if v.size()==current_model_dict[k].size() else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), ckpt['G_state_dict'].values())}
    # pdb.set_trace()
    # backbone.load_state_dict(new_state_dict, strict=True)
    backbone.load_state_dict(ckpt['G_state_dict'], strict=True)
    
    IW,IH = opt.load_size[0], opt.load_size[1]
    scale_factor = 512 / max(IH,IW)
    
    input_view = [i for i in range(opt.num_views)]
    imgs_train = train_data['img'].to(device)
    masks_train = train_data['mask'].to(device)
    in_ori_mask = train_data['in_ori_img'].to(device)
    init_pose_train = train_data['pose'].to(device) ## [K, 7]
    intrinsic_train = train_data['intrinsic'].to(device)
    normal_matrices_train = train_data['normal_matrices'].to(device)
    z_center_train = train_data['z_center'].to(device)
    joints_3d = data_input['joints_3d'] if backbone.use_spatial else None


    imgs_test = test_data['img'].to(device)
    masks_test = test_data['mask'].to(device)

    masks_dilate_test = test_data['mask_dilate'].to(device)
    z_center_test = test_data['z_center'].to(device)
    init_pose_test = test_data['pose'].to(device) ## [K, 7]
    intrinsic_test = test_data['intrinsic'].to(device)
    init_intensity_test = test_data['intensity'].to(device)
    in_ori_mask = in_ori_mask.unsqueeze(0).expand_as(masks_test)


    if 'intensity' in ckpt:
        intensity = ckpt['intensity']
    else:
        intensity = nn.Parameter(init_intensity_test.clone(), requires_grad=False)

    if 'pose' in ckpt:
        ## no inhret from checkpoint
        pose_vec_train = ckpt['pose']
    else:
        pose_vec_train = nn.Parameter(init_pose_train.clone(), requires_grad=False)
    
    # calibs_input, extrinsic_input, extris_inv_train = get_calib_extri_from_pose(pose_vec_train[input_view, ...], intrinsic_train[input_view, ...]) ## get learnable calibs, extris, extris_inv
    calibs_train, extris_train, extris_inv = get_calib_extri_from_pose(pose_vec_train, intrinsic_train) ## get learnable calibs, extris, extris_inv
    # imgs_input = F.interpolate(imgs_train[input_view, ...].clone(), scale_factor=scale_factor, mode='bicubic', align_corners=True) 
    # masks_input =  F.interpolate(masks_train[input_view, ...].clone(), scale_factor=scale_factor, mode='bicubic', align_corners=True) 
    # normal_mat_input = normal_matrices_train[input_view, ...]

    data_input = get_input_data_by_view(input_view, [], imgs_train, masks_train, normal_matrices_train, calibs_train, extris_train, z_center_train, joints_3d, scale_factor, opt.use_linear_z)
    
    target_view_weight = [1.0 for _ in range(imgs_test.shape[0])]
    target_view_weight = torch.tensor(target_view_weight).float().to(device).unsqueeze(-1).expand(-1, IW * IH).contiguous()
    target_view = [i for i in range(imgs_test.shape[0])]

    N_test = len(target_view)
    uvs = get_uvs(opt, IW, IH).expand(N_test, -1, -1).to(device)
    calibs_test, extris_test, extris_inv_test = get_calib_extri_from_pose(init_pose_test, intrinsic_test) ## get learnable calibs, extris, extris_inv
    ray_ds, cam_locs, light_dirs = get_camera_params_in_model_space(uvs, extris_inv_test, intrinsic_test, neg_z=opt.use_perspective)
    N_rays = ray_ds.shape[1]
    ray_os = cam_locs.unsqueeze(1).expand(-1,N_rays,-1)
    light_dirs = light_dirs.expand(-1,N_rays,-1)
    fb_intersection_path = os.path.join(output_dir, 'front_back_intersect.npy')
    mask_intersection_path = os.path.join(output_dir, 'mask_intersect.npy')
    # if False:
    if os.path.exists(fb_intersection_path) and os.path.exists(mask_intersection_path):
        front_back_intersections = torch.from_numpy(np.load(fb_intersection_path)).float().to(device)
        mask_intersect = torch.from_numpy(np.load(mask_intersection_path)).to(device)
    else:
        front_back_intersections, mask_intersect = get_ray_visual_hull_intersection(
                cam_locs=cam_locs, ray_ds=ray_ds, calibs=calibs_test, 
                masks_dilated=masks_dilate_test, projection_method=projection_method,
                radius=opt.object_bounding_sphere, n_sample_per_ray=512, debug_vis_path=output_dir, sphere_intersection=False, max_unobservations=0, ori_aspect=1)
        np.save(fb_intersection_path, front_back_intersections.cpu())
        np.save(mask_intersection_path, mask_intersect.cpu())
        mask_all_intesect = mask_intersect.detach().cpu().numpy().reshape(N_test, IH, IW, 1)
        ma_list = []
        for i in range(N_test):
            ma_list.append(mask_all_intesect[i])
        ma_img = np.concatenate(ma_list, axis=1)
        Image.fromarray(to8b(np.tile(ma_img, (1,1,3)))).save(mask_intersection_path.replace('.npy', '.png'))
    data_target = get_target_data_by_view(target_view,target_view_weight, imgs_test, masks_test, masks_dilate_test, mask_intersect, front_back_intersections, ray_os, ray_ds, light_dirs, in_ori_mask)

    generate_mesh(opt, backbone, train_data, data_input, save_path_geo=save_path_geo)
    render_imgs(backbone, dr, data_input, data_target, intensity, IH, IW, output_dir, cal_diff_normal=False, sample_batch=10000, use_TSDF=opt.use_TSDF, TSDF_thres=opt.TSDF_thres)
    pass
    
if __name__ == "__main__":
    from datetime import datetime
    opt = get_options()
    # for k,v in sorted(vars(opt).items()):
    #     print(k,':',v)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    if opt.phase == 'train':
        today = datetime.today().strftime("%Y%m%d")
        today = '20231022'

        set_random_seed(opt.seed)
        print(opt.id)
        geo_ckpt_path = opt.load_pretrained_path

        path_to_ckpt, path_to_vis = make_training_dirs(opt, opt.id, today)

        writer = SummaryWriter(path_to_ckpt)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(os.path.join(path_to_ckpt, 'loss.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)
        if opt.no_grad:
            with torch.no_grad():
                fine_tuning(opt, path_to_ckpt, path_to_vis, pretrained_ckpt_path=geo_ckpt_path, logger=logger)
        else:
            fine_tuning(opt, path_to_ckpt, path_to_vis, pretrained_ckpt_path=geo_ckpt_path, logger=logger)
    else:
        ## test
        from utils.common_utils import make_testing_dirs
        date = '20231022'
        input_data_dir = opt.finetune_real_data_train_dir
        test_data_dir = opt.finetune_real_data_test_dir
        test_ckpt_path = opt.load_pretrained_path
        test_output_dir = make_testing_dirs(opt, opt.id, date)

        n_sample_space = 200000
        train_data = load_data(opt, n_samp=n_sample_space, subject=opt.subject, real_data_dir=input_data_dir, test=False)
        test_data = load_data(opt, n_samp=n_sample_space, subject=opt.subject, real_data_dir=test_data_dir, test=True, num_target=opt.num_target) ## load test synthetic data by ids

        render_views(opt,train_data, train_data, test_output_dir, test_ckpt_path=test_ckpt_path)
    pass


