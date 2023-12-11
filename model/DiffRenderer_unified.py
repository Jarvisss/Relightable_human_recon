import torch
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np
from model.SampleNetwork import SampleNetwork
from .UNet_unified import UNet_unified
from .RayTracer import RayTracing
import utils.render_utils as render_utils
from PIL import Image, ImageDraw
from utils.sdf_utils import *
from utils.common_utils import *
from utils.geo_utils import index, perspective
import pdb
import time
from .vgg import VGGLoss
from .loss import PyramidL2Loss, ssim_loss_fn
from typing import List
from .ggx_renderer import eval_collocate_ggx
from .disney_renderer import eval_collocate_burley
import lpips
from tensorboardX import SummaryWriter

class DiffRenderer_unified(nn.Module):
    def __init__(self, opt, backbone:UNet_unified, dr_num_views:int, use_indirect=False, use_indirect_tracing=False, debug=False, verbose=False,device='cuda', path_to_vis=''):
        super(DiffRenderer_unified, self).__init__()
        self.backbone = backbone
        self.intensity = nn.Parameter(torch.tensor(opt.init_intensity).float())
        self.light_offset = nn.Parameter(torch.zeros(3).float())
        self.light_color = nn.Parameter(torch.ones(3).float())

        if opt.sample_patch:
            self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() ## same as traditional perceptual loss
            self.loss_fn_pyramid = PyramidL2Loss()
            self.ssim_fn = ssim_loss_fn
        self.debug_vis_path = path_to_vis + '/debug'
        self.ray_tracer = RayTracing(opt, dr_num_views, debug=debug, verbose=verbose, device=device, debug_vis_path=self.debug_vis_path) ## tracing only one view in one forward pass
        self.sample_network = SampleNetwork()
        self.opt = opt
        self.use_indirect = use_indirect
        self.use_ggx = opt.use_ggx
        self.use_indirect_tracing = use_indirect_tracing
        self.n_samp_indirect = opt.samp_indirect
        self.dr_num_views = dr_num_views
        self.debug = debug
        self.verbose = verbose
        self.device = device
        self.path_to_vis = path_to_vis
    
    def get_sdf_albedo_gradient_funcs(self, im_feats, data_input, cal_diff_normal, use_poe):
        '''
        a
        '''
        src_imgs = data_input['imgs_input']
        src_masks = data_input['masks_input']
        src_calibs = data_input['calibs_input']
        src_extris = data_input['feed_extrin']
        src_z_center = data_input['z_center']
        src_joints_3d = data_input['joints_3d']
        
        sdf_func = lambda x ,im_feats=im_feats, calibs=src_calibs, masks=src_masks, z_center_world_space=src_z_center, extris=src_extris, use_poe=use_poe,joints_3d=src_joints_3d \
            :self.backbone.query_sdf(im_feats, x, calibs, z_center_world_space, masks, extrinsic_reshape=extris, use_positional_encoding=use_poe,joints_3d=joints_3d)[0]
        
        color_func = lambda x,im_feats=im_feats, calibs=src_calibs, masks=src_masks, z_center_world_space=src_z_center, use_poe=use_poe,feed_original_img=self.opt.feed_original_img, imgs=src_imgs, extris=src_extris,joints_3d=src_joints_3d\
            :self.backbone.query_albedo(im_feats, x, calibs, z_center_world_space, masks, use_positional_encoding=use_poe, feed_original_img=feed_original_img, imgs=imgs, extrinsic_reshape=extris,joints_3d=joints_3d)
        
        spec_alb_func = lambda x,im_feats=im_feats, calibs=src_calibs, masks=src_masks, z_center_world_space=src_z_center, use_poe=use_poe,feed_original_img=self.opt.feed_original_img, imgs=src_imgs, extris=src_extris,joints_3d=src_joints_3d\
            :self.backbone.query_spec_albedo(im_feats, x, calibs, z_center_world_space, masks, use_positional_encoding=use_poe, feed_original_img=feed_original_img, imgs=imgs, extrinsic_reshape=extris,joints_3d=joints_3d)
        
        roughness_func = lambda x,im_feats=im_feats, calibs=src_calibs, masks=src_masks, z_center_world_space=src_z_center, use_poe=use_poe,feed_original_img=self.opt.feed_original_img, imgs=src_imgs, extris=src_extris,joints_3d=src_joints_3d\
            :self.backbone.query_roughness(im_feats, x, calibs, z_center_world_space, masks, use_positional_encoding=use_poe, feed_original_img=feed_original_img, imgs=imgs, extrinsic_reshape=extris,joints_3d=joints_3d)
        

        if cal_diff_normal:
            gradient_func = lambda x ,images=src_imgs, im_feats=im_feats, calibs=src_calibs, masks=src_masks, z_center_world_space=src_z_center, extris=src_extris,use_poe=use_poe,joints_3d=src_joints_3d\
            :self.backbone.get_sdf_normal_by_diff(images, x, calibs, extrinsic_reshape=extris, z_center_world_space=z_center_world_space, masks=masks, im_feat=im_feats, use_positional_encoding=use_poe,joints_3d=joints_3d)
        else:
            gradient_func = lambda x ,images=src_imgs, im_feats=im_feats, calibs=src_calibs, masks=src_masks, z_center_world_space=src_z_center, extris=src_extris,use_poe=use_poe,joints_3d=src_joints_3d \
            :self.backbone.get_sdf_normal(images, x, calibs, extrinsic_reshape=extris, z_center_world_space=z_center_world_space, masks=masks, im_feat=im_feats, use_positional_encoding=use_poe,joints_3d=joints_3d)

        return sdf_func, color_func, gradient_func, spec_alb_func, roughness_func

    def smithG1(cosTheta, alpha):
        '''
        [B,1,N], [B,3,N]
        '''
        sinTheta = torch.sqrt(1.0 - cosTheta * cosTheta)
        tanTheta = sinTheta / (cosTheta + 1e-10)
        root = alpha * tanTheta
        return 2.0 / (1.0 + torch.hypot(root, torch.ones_like(root)))


    def forward_single(self, 
    images, masks, calibs, front_back_visual_hull_intersections, visual_hull_mask_intersect, source_view_id: list,
    target_view_id: int, k_no_grad: list, sample_id_k: list, uv_k, im_k, full_im_k, mask_k, full_mask_k, calib_k, extri_inv_mat_k, intri_mat_k, normal_mat_k, intensity_k, 
    im_feats_in=None, light_feats_in=None, light_pos_k=None, albedo_k=None, full_depth_k=None, indirect_k = None,
    shading_k=None, z_center=None, epoch=0, cal_indirect=False, cal_diff_normal=False, no_grad=False, extris=None):
        '''
        :param images [BK, 3, H, W] source images                                         --- for the pifu input
        :param masks [BK, 1, H, W] source image masks                                     --- for the pifu input
        :param calibs     [BK, 4, 4]
        :param front_back_intersections [B,N,2]:   intersections of this view ray and visual hull
        :param mask_intersect [B,N,1]:             intersect mask of this view ray and visual hull
        :param source_view_id        int, view id for the source views                --- for the rendering loss of current view
        :param target_view_id        int, view id for the target views                --- for the rendering loss of current view
        :param sample_id_k    [int], sampling ray pixel ids                           --- for the rendering loss of current view
        :param uv_k [B, N, 2] sampled pixels uv coordinates of the current view       --- for the rendering loss of current view
        :param im_k [B, N, 3] sampled pixels color values of the current view         --- for the rendering loss of current view
        :param full_im_k [B, 3, H, W] source image for the target view                --- for the rendering loss of current view
        :param full_depth_k [B, 1, H, W] source image for the target view             --- for the rendering loss of current view
        :param full_indirect_k [B, 3, H, W] source image for the target view          --- for the rendering loss of current view
        :param mask_k [B, N, 1] sampled pixel mask valules of the current view        --- for the rendering loss of current view
        :param calib_mat_k [B, 4, 4]
        :param extri_inv_mat_k [B, 4, 4]
        :param intri_mat_k [B, 4, 4]
        :param normal_mat_k [B, 4, 4]

        '''
        if target_view_id in source_view_id:
            images = images.clone()

            # idx = list.index(source_view_id, target_view_id)
            idx = target_view_id
            images[idx] = 0
            if im_feats_in is not None:
                im_feats_in = im_feats_in.clone()
                im_feats_in[idx] = 0
            if light_feats_in is not None:
                light_feats_in = light_feats_in.clone()
                light_feats_in[idx] = 0

        c2w_k = extri_inv_mat_k
        calibs_to_model_k = calib_k # [B, 4, 4]
        ray_dirs, cam_loc = render_utils.get_camera_params_in_model_space(uv_k, c2w_k, intri_mat_k)
        B, num_pixels, C_dir = ray_dirs.shape
        B1,C_loc = cam_loc.shape
        K = images.shape[0]
        assert(B == B1)
        assert(C_loc == 3)
        assert(C_dir == 3)
        
        ## get sphere tracinng result in eval mode
        self.backbone.eval()

        t1 = time.time()
        with torch.no_grad():
            if im_feats_in is None:
                im_feats, _ = self.backbone.filter(images, k_no_grad)
            else:
                im_feats = im_feats_in.detach()
            
            ret = self.ray_tracer.forward(
                sdf=lambda x ,im_feats=im_feats, calibs=calibs, z_center_world_space=z_center , masks=masks, extris=extris :self.backbone.query_sdf(im_feats, x, calibs,z_center_world_space, masks, extrinsic_reshape=extris)[0], 
                front_back_intersections = front_back_visual_hull_intersections,
                mask_intersect = visual_hull_mask_intersect,
                cam_loc=cam_loc,
                view_ids=[target_view_id],
                sample_id_k = sample_id_k,
                object_mask=mask_k,
                # calibs=calibs, ## not used if use sphere intersection
                # masks=masks, ## not used if use sphere intersection
                ray_directions=ray_dirs
            )
            if ret == None:
                print('no object intersected in sphere tracing')
                return []
            else:
                points, net_mask, net_dists = ret
                # [K, BN, 3]
                # [K, BN]

                network_object_mask = net_mask[0]
                dists = net_dists[0]
        
        t2 = time.time()
        '''
        Here we do not train geo filter
        '''
        if not no_grad:
            self.backbone.train()
        # im_feats = self.geo_mlp.filter(images)
        # im_feats, light_feats_mlp =self.backbone.filter(images)
        # pdb.set_trace()
        if no_grad:
            with torch.no_grad():
                if im_feats_in is None:
                    im_feats, light_feats_mlp = self.backbone.filter(images, k_no_grad)
                else:
                    im_feats, light_feats_mlp = im_feats_in, light_feats_in
        else:
            if im_feats_in is None:
                im_feats, light_feats_mlp = self.backbone.filter(images, k_no_grad)
            else:
                im_feats, light_feats_mlp = im_feats_in, light_feats_in
        

        if self.use_indirect:
            im_feat_k, light_feat_k = self.backbone.filter(full_im_k)
            indirect_func = lambda x, normal, im_feats=im_feat_k, calibs=calibs_to_model_k, masks=full_mask_k, z_center_world_space=z_center, use_positional_encoding=self.opt.use_positional_encoding,feed_original_img=self.opt.feed_original_img, imgs=images  \
                :self.backbone.query_indirect_lighting(im_feats, x, calibs,z_center_world_space, masks, normal=normal)
        
        light = self.backbone.light_MLP(light_feats_mlp)
        intensity = intensity_k * torch.ones_like(light[:,0:1]) if self.opt.use_gt_intensity else  torch.exp(light[:,0:1])
        # intensity = intensity_k * torch.ones_like(light[:,0:1]) if self.opt.use_gt_intensity else   1. + 100 * nn.functional.relu(light[:,0:1])
        # intensity = self.backbone.intensity

        sdf_func_proj_model_to_im = lambda x ,im_feats=im_feats, calibs=calibs, masks=masks, z_center_world_space=z_center, extris=extris \
            :self.backbone.query_sdf(im_feats, x, calibs, z_center_world_space, masks, extrinsic_reshape=extris)[0]
        
        color_func = lambda x,im_feats=im_feats, calibs=calibs, masks=masks, z_center_world_space=z_center, use_positional_encoding=self.opt.use_positional_encoding,feed_original_img=self.opt.feed_original_img, imgs=images, extris=extris  \
            : self.backbone.query_albedo(im_feats, x, calibs,z_center_world_space, masks, use_positional_encoding=use_positional_encoding, feed_original_img=feed_original_img, imgs=imgs, extrinsic_reshape=extris)
        
        if cal_diff_normal:
            gradient_func = lambda x ,images=images, im_feats=im_feats, calibs=calibs, masks=masks, z_center_world_space=z_center, extris=extris  :self.backbone.get_sdf_normal_by_diff(images, x, calibs, extrinsic_reshape=extris, z_center_world_space=z_center_world_space, masks=masks, im_feat=im_feats)
        else:
            gradient_func = lambda x ,images=images, im_feats=im_feats, calibs=calibs, masks=masks, z_center_world_space=z_center, extris=extris  :self.backbone.get_sdf_normal(images, x, calibs, extrinsic_reshape=extris, z_center_world_space=z_center_world_space, masks=masks, im_feat=im_feats)
        albedo_grad_func = lambda x ,images=images, im_feats=im_feats, calibs=calibs, masks=masks, z_center_world_space=z_center, use_positional_encoding=self.opt.use_positional_encoding,feed_original_img=self.opt.feed_original_img:self.backbone.get_albedo_gradient(images, x, calibs, z_center_world_space, masks, im_feat=im_feats, use_positional_encoding=use_positional_encoding, feed_original_img=feed_original_img)

        # res_shading_func = lambda n, l: self.backbone.res_shading_predictor(n, l)

        output_all = []

        cam_locs_k = cam_loc
        ray_dirs_k = ray_dirs
        object_mask_k = mask_k > 0.5
        # shading_k = shading_patch[k*batch_size:(k+1)*batch_size, :,:,:] if shading_patch is not None else None
        object_mask_k = object_mask_k.view(-1)
        # shading_k = shading_k.permute(0, 2, 3 ,1).contiguous().view(-1, 3)

        points_k = (cam_locs_k.unsqueeze(1) + dists.reshape(B, num_pixels, 1) * ray_dirs_k)# [B, N, 3]
        sdf_output = sdf_func_proj_model_to_im(points_k.transpose(1,2)).flatten() # [BN]
        points_k = points_k.view(-1,3) # [BN, 3]
        ray_dirs_k = ray_dirs_k.view(-1,3) # [BN, 3]

        t3 = time.time()
        
        if not no_grad:
            # 只train mask内的部分是不对的?
            # surface_mask = network_object_mask & object_mask_k # [BN']
            surface_mask = network_object_mask # [BN']
            if surface_mask.sum() == 0:
                print('no object intersected of (gt & pred) ')
                return None, intensity
            surface_points = points_k[surface_mask] # [BN', 3]
            surface_dists = dists[surface_mask].unsqueeze(-1) #[BN', 1]
            surface_ray_dirs = ray_dirs_k[surface_mask]
            surface_cam_locs = cam_locs_k.unsqueeze(1).repeat(1, num_pixels, 1).view(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            
            # Sample points for the eikonal loss
            eik_bounding_box = self.opt.object_bounding_sphere
            n_eik_points = B * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points_k.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            surface_sdf_values = sdf_func_proj_model_to_im(
                surface_points.view(B,-1,3).transpose(1,2)
                ).flatten().detach() # [BN']
            
            
            surface_gradients = gradient_func(
                surface_points.view(B,-1,3).transpose(1,2)
                ).transpose(1,2).view(-1,3).clone().detach() # [BN', 3]
            
            space_gradients = gradient_func(
                eikonal_points.view(B,-1,3).transpose(1,2)
                ).transpose(1,2).view(-1,3)# [BN', 3]

            t4 = time.time()
            differentiable_surface_points = self.sample_network(
                surface_output, # not detached
                surface_gradients, # detached
                surface_dists,
                surface_cam_locs,
                surface_ray_dirs
            )
            t5 = time.time()

        else:
            surface_mask = network_object_mask
            if surface_mask.sum() == 0:
                print('no object for this area ')
                return None, intensity
            differentiable_surface_points = points_k[surface_mask]
            space_gradients = None
        


        albedos = torch.zeros_like(points_k).float().to(self.device) # [BN, 3]
        albedo_grads = torch.zeros_like(points_k).float().to(self.device)
        diffuse_shadings = torch.zeros_like(points_k).float().to(self.device)
        indirect_shadings = torch.zeros_like(points_k).float().to(self.device)
        normals = torch.zeros_like(points_k).float().to(self.device)
        preds = torch.zeros_like(points_k).float().to(self.device)

        indirect_traced = torch.zeros_like(points_k).float().to(self.device)
        # shadings_all_views = torch.zeros_like(shadings).unsqueeze(0).repeat(self.num_views, 1, 1)
        # visibilities = torch.zeros_like(shadings_all_views[:,:,0]).bool()
        # gts_all_views = torch.zeros_like(shadings_all_views)
        if differentiable_surface_points.shape[0] > 0:
            # print('dr_points_shape:',differentiable_surface_points.shape)
            # [BN, 3]

            if torch.any(torch.isnan(differentiable_surface_points)):
                print('diff point has nan')
            
            
            normal = gradient_func(differentiable_surface_points.view(B, -1 , 3).transpose(1, 2)) #[B,3,N]
            
            '''
            sample rays for each point
            '''

            if self.use_indirect:
                indirect_lighting = indirect_func(differentiable_surface_points.view(B, -1 , 3).transpose(1, 2), normal)
                indirect_shadings[surface_mask] = torch.clamp(indirect_lighting.transpose(1,2).contiguous().view(-1, 3), 0, 1)
                print(indirect_lighting)
            
            if cal_indirect:
                
                count = 64
                for j in tqdm(range(count)):
                    indirect_cal_k, sec_hit_points_k, sec_mask, hit_points_color = self.trace_indirect(
                        points_k, surface_mask, normal.transpose(1,2).contiguous().view(-1,3), img_k=full_im_k, depth_k=full_depth_k, calib_k=calibs_to_model_k, 
                        im_feats=im_feats, calibs=calibs, z_center=z_center, masks=masks, nsamp=self.n_samp_indirect)
                ## return 
                    indirect_traced[surface_mask] = indirect_traced[surface_mask] + indirect_cal_k
                indirect_traced[surface_mask] /= count

            # albedo_grad = albedo_grad_func(differentiable_surface_points.view(B, -1 , 3).transpose(1, 2))
            # []
            rot = normal_mat_k[:, :3, :3] #[B,3,3]
            trans = normal_mat_k[:, :3, 3:4] # [B,3,1]
            world_position = torch.baddbmm(trans, rot, differentiable_surface_points.view(B, -1 , 3).transpose(1, 2)) # [B, 3, N]
            world_cam_loc = torch.baddbmm(trans, rot, cam_locs_k.view(B, -1 , 3).transpose(1, 2)) # [B, 3, N]
            
            if light_pos_k is not None:
                light_pos = torch.baddbmm(trans, rot, light_pos_k.view(B, -1 , 3).transpose(1, 2))
            else:
                light_pos = world_cam_loc
            
            light_pos = light_pos
            # light_pos = world_cam_loc * np.sin(np.radians(42/2)) * 1
            # cal light in world space
            light_vector = (world_position - light_pos) # [B, 3, N]
            incoming_light_dir = safe_l2norm( light_vector ) # light pos [B, 3, N]
            world_normal = safe_l2norm(normal)  # [B, 3, N]
            distance = light_vector.norm(dim=1, keepdim=True) #[B, 1, N]
            # attenuation = 1 / (1 + 0.09 * distance + 0.032 * distance * distance)
            attenuation = 1 / (distance * distance + 1e-8)
            cosine_term = torch.clamp( torch.sum(world_normal * -incoming_light_dir, dim=1), 0, 1) # [B, N]
            flash_shading = cosine_term[:,None,:] * attenuation * intensity # [B, 1, N] * [B, 1] -> [B, 1, N]
            albedos[surface_mask] = color_func(differentiable_surface_points.view(B, -1 , 3).transpose(1, 2)).transpose(1,2).contiguous().view(-1, 3)
            # albedo_grads[surface_mask] = albedo_grad.transpose(1,2).contiguous().view(-1, 3)
            diffuse_shadings[surface_mask] = flash_shading.expand(-1,3,-1).transpose(1,2).contiguous().view(-1, 3)

            normals[surface_mask] = world_normal.transpose(1,2).contiguous().view(-1, 3)
            shad = diffuse_shadings[surface_mask]
            if self.use_indirect:
                shad = shad+indirect_shadings[surface_mask]

            preds[surface_mask] = torch.clamp( shading_l2g(shad * albedos[surface_mask]), 0, 1)

        if self.opt.image_patch_save_freq > 0 and epoch % self.opt.image_patch_save_freq == 0:
            
            start_points_unfinished = differentiable_surface_points.detach() #[BN, 3]
            save_dir = self.debug_vis_path + '/tracing_pt/epoch%d_img/' % (epoch)
            save_samples_color(save_dir + 'albedo_%d.ply' % (target_view_id), start_points_unfinished.cpu(), img_l2g(albedos[surface_mask]).detach().cpu() * 255)
            save_samples_color(save_dir + 'normal_%d.ply' % (target_view_id), start_points_unfinished.cpu(), (safe_l2norm(normal.transpose(1,2).contiguous().view(-1, 3))*0.5+0.5).detach().cpu() * 255)
            save_samples_color(save_dir + 'direct_shading_%d.ply' % (target_view_id), start_points_unfinished.cpu(), (torch.clamp( shading_l2g(diffuse_shadings[surface_mask]), 0, 1)).detach().cpu() * 255)
            save_samples_color(save_dir + 'pred_%d.ply' % (target_view_id), start_points_unfinished.cpu(), (torch.clamp( shading_l2g(diffuse_shadings[surface_mask] * albedos[surface_mask]), 0, 1)).detach().cpu() * 255)
            save_samples_color(save_dir + 'gt_%d.ply' % (target_view_id), start_points_unfinished.cpu(), im_k.view(-1, 3)[surface_mask].detach().cpu() * 255)
            # pdb.set_trace()
            if albedo_k is not None:
                save_samples_color(save_dir + 'net_shading_gt_albedo%d.ply' % (target_view_id), start_points_unfinished.cpu(), (torch.clamp( shading_l2g(diffuse_shadings[surface_mask] * albedo_k.permute(0, 2, 3 ,1).contiguous().view(-1, 3)[surface_mask]), 0, 1)).detach().cpu() * 255)
            if cal_indirect:
                # hit_points_k = sec_hit_points_k[sec_mask]
                # save_samples_color(save_dir + 'indirect_hit_%d.ply' % (target_view_id), hit_points_k.cpu(), hit_points_color.transpose(1,2).contiguous().view(-1, 3).detach().cpu() * 255)
                pass
                # n_sec_cam = start_points_unfinished.shape[0]
                # sample_dirs, cosines = self.sample_half_sphere( normal.transpose(1,2).contiguous().view(-1,3), 128)

                # for j in range(n_sec_cam):
                #     start_point_j = start_points_unfinished[j:j+1].cpu().detach()
                #     sample_dir_j = sample_dirs[j:j+1].cpu().detach()
                #     start_point_color_j = torch.ones_like(start_point_j) * 255
                #     start_point_color_j[:,1:] = 0

                                        
                #     # hit_points_j = sec_hit_points_k.reshape(n_sec_cam, self.n_samp_indirect, 3)[j][sec_mask.reshape(n_sec_cam, self.n_samp_indirect)[j]].detach().cpu()
                     
                #     n_line_points = 10
                #     step = 0.05
                #     hit_points_j = torch.zeros(n_line_points, 128 ,3) ## []
                #     for k in range(n_line_points):
                #         dist = k * step
                #         hit_points_j[k, :, :] = start_point_j.repeat(128, 1) + sample_dir_j * dist
                        
                #     hit_points_color_j = torch.ones_like(hit_points_j) * 255
                #     hit_points_color_j[:,:2] = 0
                    
                #     save_samples_color(save_dir + 'indirect_hit_%d_%d.ply' % (target_view_id, j), hit_points_j.view(-1,3), hit_points_color_j.view(-1,3))
                #     save_samples_color(save_dir + 'indirect_start_%d_%d.ply' % (target_view_id, j), start_point_j, start_point_color_j)

            # save_path1 = save_dir + 'img_patch_view_%d.png' %(target_view_id)
            # img = (img_k[0]).permute(1,2,0).cpu().numpy()* 255.0
            # Image.fromarray(np.uint8(img)).save(save_path1)

        
        output_k = {
            'imgs': im_k.view(-1, 3), #[BN, 3]
            'albedos_gt': albedo_k.view(-1, 3) if albedo_k is not None else None, #[BN, 3]
            'diff_points': differentiable_surface_points,
            # 'shadings_gt': shading_k,
            'object_mask': object_mask_k, #[BN]
            'space_gradients':space_gradients,
            'points': points_k, # [BN, 3]
            'depth': self.backbone.projection(points_k.transpose(0,1).contiguous().view(3, -1).unsqueeze(0), calibs_to_model_k)[:,2:,:].transpose(1,2).contiguous().view(-1,1),
            'diffuse_shadings': diffuse_shadings, # [BN, 3]
            'indirect_shadings': indirect_shadings,
            'indirect_traced':indirect_traced,
            'indirect_input':indirect_k.view(-1, 3) if indirect_k is not None else None,
            'normals':normals, # [BN, 3]
            # 'shadings_all': shadings_all_views, #[K, BN, 3]
            # 'visibilities_all': visibilities,
            # 'gts_all': gts_all_views, #[K, BN, 3]
            'albedos': albedos, # [BN, 3]
            'preds':preds,
            # 'albedos_grads' : albedo_grads,
            'sdf_output': sdf_output, #[BN]
            'intensity': intensity,
            'network_object_mask': network_object_mask, #[BN]
        }
        output_all.append(output_k)
        
        t6 = time.time()

        if self.verbose:
            print('ray tracing time:', t2-t1)
            print('sdf time:', t3-t2)
            print('gradient time:', t4-t3)
            print('sample network time:', t5-t4)
            print('other time:', t6-t5)

        return output_all, intensity

    def forward(self, data_input,
        src_img_feats: torch.Tensor, 
        tgt_views: List[int],  ray_os: torch.Tensor, ray_ds: torch.Tensor, ray_cs: torch.Tensor, ray_ms: torch.Tensor, light_dirs:torch.Tensor,
        front_back_dist_init: torch.Tensor, ray_mask_init:torch.Tensor, cal_diff_normal: bool,  no_grad:bool, intensity:torch.Tensor,
        use_poe:bool, jittor_std: float=0.01, is_metal=False, use_TSDF=False, TSDF_thres=0.1, epoch=0, save_intermediate=False
    ):
        '''
        :param src_views:     List[int]
        :param src_no_grad:   List[int]                                   --- no grad list of src views
        :param src_imgs:      [BK, 3, H_L, W_L] source images                         --- for the pifu input low res
        :param src_masks:     [BK, 1, H_L, W_L] source image masks                    --- for the pifu input low res
        :param src_z_center:  [1, 3, 1] source z center                               --- for the pifu input
        :param src_joints_3d:  [1, K, 3] source joints                                --- (optional) for the pifu input
        :param src_calibs:    [BK, 4, 4]                                              --- for the pifu input
        :param src_extris:    [BK, 4, 4]                                              --- for the pifu input
        :param src_img_feats:     [BK, C, H_L, W_L]                                   --- for the pifu input(optional)
        :param src_light_feats:     [BK, C, H_L, W_L]                                 --- for the pifu input(optional)
        :param tgt_views:     List[int]
        :param scale_mat:     [1, 4, 4] scale mat convert scene from unit sphere back to world space
        :param ray_os:        [N, 3] sampled rays origins from all target views
        :param ray_os_world:  [N, 3] sampled rays origins from all target views in world space for light
        :param ray_ds:        [N, 3] sampled rays directions from all target views
        :param ray_cs:        [N, 3] sampled rays colors from all target views
        :param ray_ms:        [N] sampled rays in or out gt mask
        :param light_dirs:        [N, 3] sampled rays in or out gt mask
        :param front_back_dist_init: [N, 2] sampled rays in or out gt mask
        :param ray_mask_init:        [N] sampled rays in or out gt mask
        :param cal_diff_normal:   bool 
        :param no_grad:   bool 
        :param jittor_std:     float

        '''

        src_no_grad = data_input['no_grad_view']
        src_views = data_input['input_views']
        src_imgs = data_input['imgs_input']
        src_masks = data_input['masks_input']
        scale_mat = data_input['normal_mat_input'][:1]
        src_calibs = data_input['calibs_input']
        src_extris = data_input['feed_extrin']
        src_z_center = data_input['z_center']
        src_joints_3d = data_input['joints_3d']

        N_rays, _ = ray_ds.shape
        t1 = time.time()
        self.backbone.eval()
        ## ray tracing without gradient
        with torch.no_grad():
            if src_img_feats is None:
                im_feats_no_grad, _ = self.backbone.filter(src_imgs, src_no_grad)
            else:
                im_feats_no_grad = src_img_feats.detach()
            
            ret = self.ray_tracer.forward_ray_batch(
                sdf=lambda x ,im_feats=im_feats_no_grad, calibs=src_calibs, z_center_world_space=src_z_center , masks=src_masks, extris=src_extris,use_poe=use_poe\
                    :self.backbone.query_sdf(im_feats, x, calibs,z_center_world_space, masks, extrinsic_reshape=extris, use_positional_encoding=use_poe, joints_3d=src_joints_3d)[0], 
                front_back_dist_init= front_back_dist_init,
                mask_init = ray_mask_init,
                gt_mask = ray_ms,
                ray_os = ray_os,
                ray_ds = ray_ds,
                use_TSDF=use_TSDF,
                TSDF_thres=TSDF_thres,
                epoch = epoch,
                save_intermediate=save_intermediate
            )
            if ret == None:
                print('no object intersected in sphere tracing')
                return None
            else:
                _, trace_mask, dists = ret ## [[N,3], [N], [N]]
                 
        ## get source feat related functions
        t2 = time.time()
        if not no_grad:
            self.backbone.train()
        im_feats = src_img_feats
        if im_feats is None:
            if not no_grad:
                im_feats = self.backbone.filter(src_imgs, src_no_grad)
            else:
                with torch.no_grad():
                    im_feats = self.backbone.filter(src_imgs, src_no_grad)
        
        # light = self.backbone.light_MLP(light_feats_mlp)
        # intensity = torch.exp(light[:,0:1])
        # intensity = 2.1 * torch.ones_like(light[:,0:1])
        sdf_func, albedo_func, gradient_func, spec_albedo_func, roughness_func = self.get_sdf_albedo_gradient_funcs(im_feats, data_input, cal_diff_normal, use_poe=use_poe)
        
        ## reparam intersect points
        t3 = time.time()
        points = (ray_os + dists.unsqueeze(-1) * ray_ds)# [N_ray, 3]
        points_jittored = (ray_os + (dists.unsqueeze(-1) + (torch.rand_like(dists.unsqueeze(-1)) * 2  - 1) *  jittor_std ) * ray_ds)
        in_gt_mask = ray_ms > 0.5
        sdf_output = sdf_func(points.T.unsqueeze(0)).flatten()
        # surface_mask = trace_mask # [N_ray]
        if not no_grad:
            surface_mask = trace_mask & in_gt_mask # [N_ray]
            # surface_mask = in_gt_mask # [N_ray]
            if surface_mask.sum() == 0: 
                print('no object intersected of (gt & pred) ')
                return None

            surface_points = points[surface_mask] # [N', 3]
            surface_dists = dists[surface_mask].unsqueeze(-1) # [N', 1]
            surface_ray_dirs = ray_ds[surface_mask] # [N', 3]
            surface_cam_locs = ray_os[surface_mask] # [N', 3]
            surface_light_dirs = light_dirs[surface_mask]
            
            surface_output = sdf_output[surface_mask] # [N']
            # Sample points for the eikonal loss
            # eik_bounding_box = self.opt.object_bounding_sphere
            # eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            n_eik_points = N_rays
            # n_eik_points = N_rays//2
            eik_bounding_box = self.opt.object_bounding_sphere

            # b_min = np.array([-0.4,-0.8,-0.4])
            # b_max = np.array([0.4,1, 0.4])
            
            b_min = np.array([-eik_bounding_box, -eik_bounding_box, -eik_bounding_box])
            b_max = np.array([eik_bounding_box,  eik_bounding_box,  eik_bounding_box])

            length = b_max - b_min
            eikonal_points = torch.from_numpy(np.random.rand(n_eik_points, 3) * length + b_min).float().cuda()

            eikonal_pixel_points = points.detach()
            eikonal_pixel_points_jittored = points_jittored.clone().detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points, eikonal_pixel_points_jittored], 0)
            # eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)
            # print(eikonal_points.shape)
            eikonal_points = eikonal_points[torch.randperm(eikonal_points.shape[0])[:N_rays]]

            
            surface_gradients = gradient_func(
                surface_points.T.unsqueeze(0)
                ).transpose(1,2).view(-1,3).clone().detach() # [N', 3]
            
            space_gradients = gradient_func(
                eikonal_points.T.unsqueeze(0)
                ).transpose(1,2).view(-1,3)# [N', 3]
            
            # pdb.set_trace()
            # save_samples_truncted_prob('/home/lujiawei/workspace/tex2cloth/visualize_result/20230622/0309_n_4_z_linear_m_12_n_in_m_transformer_cross_img_sample_rays_m_0.5/geometry/points_.ply', points.cpu().numpy(), sdf_output.detach().cpu().numpy(), 0)
            # save_samples_truncted_prob('/home/lujiawei/workspace/tex2cloth/visualize_result/20230622/0309_n_4_z_linear_m_12_n_in_m_transformer_cross_img_sample_rays_m_0.5/geometry/points_surface.ply', points[surface_mask].cpu().numpy(), sdf_output[surface_mask].detach().cpu().numpy(), 0)
            # save_samples_color('/home/lujiawei/workspace/tex2cloth/visualize_result/20230622/0309_n_4_z_linear_m_12_n_in_m_transformer_cross_img_sample_rays_m_0.5/geometry/eik.ply', eikonal_points.cpu().numpy()[N_rays // 2:], (safe_l2norm(space_gradients)[N_rays // 2:].detach().cpu().numpy()*0.5+0.5)*255)
            # save_samples_color('/home/lujiawei/workspace/tex2cloth/visualize_result/20230622/0309_n_4_z_linear_m_12_n_in_m_transformer_cross_img_sample_rays_m_0.5/geometry/surface_normal.ply', surface_points.cpu().numpy(), (safe_l2norm(surface_gradients).detach().cpu().numpy()*0.5+0.5)*255)

            t4 = time.time()
            differentiable_surface_points = self.sample_network.forward(
                surface_output, # not detached [N']
                surface_gradients, # detached [N', 3]
                surface_dists, # no grad[N', 1]
                surface_cam_locs, # leaf node const[N', 3]
                surface_ray_dirs, # leaf node const[N', 3]
            )
            t5 = time.time()
        else:
            surface_mask = trace_mask
            surface_cam_locs = ray_os[surface_mask]
            surface_light_dirs = light_dirs[surface_mask]
            if surface_mask.sum() == 0:
                print('no object for this area ')
                return None
            differentiable_surface_points = points[surface_mask]
            space_gradients = None
            eikonal_points= None

        differentiable_surface_dists = (differentiable_surface_points - surface_cam_locs).norm(dim=1, keepdim=True)
        
        ## calculate shading and final color
        pred_diff_albedos = torch.zeros_like(points) #[N_ray, 3]
        pred_points = torch.zeros_like(points)
        pred_spec_albedos = torch.zeros_like(points) #[N_ray, 3]
        pred_alphas = torch.zeros_like(points[:,:1]) #[N_ray, 3]
        pred_depths = torch.zeros_like(points[:,:1]) #[N_ray, 1]
        # pred_albedos_delta = torch.zeros_like(points) #[N_ray, 3]
        cosine_shadings = torch.zeros_like(points)
        diffuse_shadings = torch.zeros_like(points) #[N_ray, 3]
        specular_shadings = torch.zeros_like(points) #[N_ray, 3]
        indirect_shadings = torch.zeros_like(points) #[N_ray, 3]
        pred_normals = torch.zeros_like(points) #[N_ray, 3]
        pred_diffuse_colors = torch.zeros_like(points) #[N_ray, 3]
        pred_spec_colors = torch.zeros_like(points) #[N_ray, 3]
        pred_final_colors = torch.zeros_like(points) #[N_ray, 3]

        if differentiable_surface_points.shape[0] > 0:
            if torch.any(torch.isnan(differentiable_surface_points)):
                print('diff point has nan')
            
            normal = gradient_func(differentiable_surface_points.T.unsqueeze(0)) #[B,3,N]
            rot = scale_mat[:, :3, :3] #[B,3,3]
            trans = scale_mat[:, :3, 3:4] # [B,3,1]
            # Y = t + R@X, transform back to world space when calculate lighting
            surface_position_world = torch.baddbmm(trans, rot, differentiable_surface_points.T.unsqueeze(0)) # [B, 3, N]
            cam_loc_world = torch.baddbmm(trans, rot, surface_cam_locs.T.unsqueeze(0)) # [B, 3, N']
            light_pos_world = cam_loc_world
            # light_pos_world = light_pos_world + self.light_offset[None,:,None]

            light_vector = surface_position_world - light_pos_world
            incoming_light_dir = safe_l2norm( light_vector ) # ldir_in [B, 3, N]
            normal_world = safe_l2norm(normal)  # [B, 3, N]
            distance = light_vector.norm(dim=1) #[B, 1, N]
            attenuation = 1 / (distance * distance + 1e-8) # [B,N]

            back_normal_penalty = torch.clamp(torch.sum(normal_world * incoming_light_dir, dim=1), 0)
            cosine_term = torch.clamp( torch.sum(normal_world * -incoming_light_dir, dim=1), 0.00001, 0.99999) # [B, N]
            cosine_term_light = torch.clamp( torch.sum(surface_light_dirs.T.unsqueeze(0) * incoming_light_dir, dim=1), 0.00001, 0.99999) # [B,N]
            # cosine_term = torch.abs( torch.sum(normal_world * -incoming_light_dir, dim=1)) # [B, N]
            # light_term = torch.exp(intensity) * attenuation
            light_term = intensity * attenuation * cosine_term_light # [B,N]
            # light_term = intensity * attenuation # [B,N]
            pred_diff_albedos[surface_mask] = albedo_func(differentiable_surface_points.T.unsqueeze(0)).transpose(1,2).contiguous().view(-1, 3)
            pred_points[surface_mask] = differentiable_surface_points
            pred_depths[surface_mask] = differentiable_surface_dists
            
            cosine_shadings[surface_mask] = cosine_term_light.T.expand(-1,3)
            if self.use_ggx:
                

                if self.opt.use_disney:
                    diffuse_color , specular_color = eval_collocate_burley(cosine_term.T, pred_alphas[surface_mask], pred_diff_albedos[surface_mask], pred_spec_albedos[surface_mask][:,:1], pred_spec_albedos[surface_mask][:,1:2])
                    final_color = diffuse_color + specular_color
                    pred_spec_colors[surface_mask] = specular_color
                else:
                    pred_alphas[surface_mask] = roughness_func(differentiable_surface_points.T.unsqueeze(0)).transpose(1,2).contiguous().view(-1, 1).abs() + 0.01
                    spec_albedo = spec_albedo_func(differentiable_surface_points.T.unsqueeze(0)).transpose(1,2).contiguous().view(-1, 3).abs()
                    if not is_metal:
                        spec_albedo = torch.mean(spec_albedo, dim=-1, keepdim=True).expand_as(spec_albedo)
                    pred_spec_albedos[surface_mask] = spec_albedo

                    ## disney
                    # diffuse_shading = torch.pow( 1 + (2*torch.sqrt(pred_alphas[surface_mask]) - 0.5)*torch.pow(1-cosine_term.flatten().unsqueeze(-1), 5), 2) \
                    #     * light_term * cosine_term.flatten().unsqueeze(-1)

                    ## lambertian
                    diffuse_shading = cosine_term * light_term #[N, 1]
                    
                    ## oren-nayar
                    ## TODO

                    diffuse_shadings[surface_mask] = diffuse_shading.T.expand(-1,3)
                    diffuse_color = diffuse_shadings[surface_mask] * pred_diff_albedos[surface_mask]
                    
                    spec_component= eval_collocate_ggx(normal_world.transpose(1,2).contiguous().view(-1, 3), -incoming_light_dir.transpose(1,2).contiguous().view(-1, 3), cosine_term.flatten().unsqueeze(-1), pred_alphas[surface_mask])
                    # spec_brdf = spec_component * pred_spec_albedos[surface_mask]
                    spec_brdf = spec_component
                    specular_color = spec_brdf * light_term.T
                    final_color = diffuse_color + specular_color
                    pred_spec_colors[surface_mask] = specular_color

            else:
                diffuse_shading = cosine_term * light_term #[B, N]
                diffuse_shadings[surface_mask] = diffuse_shading.T.expand(-1,3)
                diffuse_color = diffuse_shadings[surface_mask] * pred_diff_albedos[surface_mask] # [N,3]
                final_color = diffuse_color
            
            pred_normals[surface_mask] = normal_world.transpose(1,2).contiguous().view(-1, 3)
            pred_diffuse_colors[surface_mask] = diffuse_color
            pred_final_colors[surface_mask] = final_color

        
        output = {
            'imgs': ray_cs, #[N, 3]
            'diff_points': pred_points, #[N, 3]
            'diff_dists': pred_depths, #[N, 3]
            'object_mask': in_gt_mask, #[N]
            'back_normal_penalty': back_normal_penalty, #[N]
            'surface_mask': surface_mask, #[N]
            'network_object_mask': trace_mask, #[N]
            'points': points, # [N, 3]
            'ray_os': ray_os,
            'ray_ds': ray_ds,
            'ray_ms': ray_ms,
            'diff_albedos': pred_diff_albedos, # [N, 3]
            'roughness': pred_alphas,
            'normals':pred_normals, # [BN, 3]
            'diffuse_shadings': diffuse_shadings, # [N, 3]
            'cosine_term_light': cosine_shadings, # [N, 3]
            'specular_shadings' : pred_spec_colors,
            'specular_albedos' : pred_spec_albedos,
            'indirect_shadings': indirect_shadings,
            'preds':pred_final_colors,
            'preds_diffuse_color':pred_diffuse_colors,
            'preds_spec_color':pred_spec_colors,
            'sdf_output': sdf_output, #[BN]
            'space_gradients': space_gradients, #[N', 3]
            'space_pts':eikonal_points, #[N', 3]
        }
        t6 = time.time()

        if self.verbose:
            print('ray tracing time:', t2-t1)
            print('sdf time:', t3-t2)
            print('gradient time:', t4-t3)
            print('sample network time:', t5-t4)
            print('other time:', t6-t5)

        return output

        

    def sample_dirs(self, normal, r_theta, r_phi):
        '''
        normals: [BN, 3]
        r_theta: [BN, nsamp]
        r_phi: [BN, nsamp]
        return: [BN, nsmap, 3]
        '''
        normal = safe_l2norm(normal)
        normals = normal.unsqueeze(1)
        z_axis = torch.zeros_like(normals).to(self.device)
        z_axis[:, :, 0] = 1

        
        U = safe_l2norm(torch.cross(z_axis, normals), dim=-1)
        V = safe_l2norm(torch.cross(normals, U), dim=-1)

        r_theta = r_theta.unsqueeze(-1).expand(-1, -1, 3)
        r_phi = r_phi.unsqueeze(-1).expand(-1, -1, 3)
        sample_raydirs = U * torch.cos(r_theta) * torch.sin(r_phi) \
                        + V * torch.sin(r_theta) * torch.sin(r_phi) \
                        + normals * torch.cos(r_phi) # [num_cam, num_samples, 3]
        
        return sample_raydirs

    def sample_half_sphere(self, normal, nsamp):
        '''
        normals: [BN, 3]
        nsamp: int
        @ return: [BN, nsmap, 3]
        '''
        n_points = normal.shape[0]
        r_thetas = torch.rand(n_points, nsamp).to(self.device) * 2 * np.pi
        # r_theta = torch.zeros(sec_cam_loc.shape[0], nsamp).to(self.device) * 2 * np.pi
        rand_z = torch.rand(n_points, nsamp).to(self.device) 
        # rand_z = torch.zeros(sec_cam_loc.shape[0], nsamp).to(self.device) 
        r_phis = torch.asin(rand_z)
        NdotL = torch.cos(r_phis) #[BN, nsamp]
        return self.sample_dirs(normal, r_thetas, r_phis), NdotL

    def trace_indirect(self, points, net_mask, normals, img_k, depth_k, calib_k, nsamp=1, im_feats=None, calibs=None, view_id=None, sample_id_k=None, z_center=None, masks=None,calibs_to_model_k=None):
        '''
        @ points: [BN, 3]
        @ img_k: [B,3,H,W]
        @ net_mask: [BN]
        @ normals: [BN, 3]

        @ return: [N_cam, 3]
        '''
        sec_cam_loc = points[net_mask].clone() # [N_hit, 3]
        n_cam = sec_cam_loc.shape[0]
        if sec_cam_loc.shape[0] > 0:
            dirs, cosines = self.sample_half_sphere(normals, nsamp)  ## sample on the half sphere
            sec_object_mask = torch.ones(sec_cam_loc.shape[0] * nsamp).bool().to(self.device)
            # pdb.set_trace()
            with torch.no_grad():
                sec_points, sec_net_object_mask, sec_dist = self.ray_tracer(
                    sdf=lambda x ,im_feats=im_feats, calibs=calibs, z_center_world_space=z_center , masks=masks :self.backbone.query_sdf(im_feats, x, calibs,z_center_world_space, masks)[0], 
                    cam_loc=sec_cam_loc, ##[N_first_hit, 3]
                    view_ids=[view_id],
                    sample_id_k = sample_id_k,
                    object_mask=sec_object_mask,
                    # calibs=calibs, ## not used if use sphere intersection
                    ray_directions=dirs  ####[N_first_hit,nsamp, 3]
                )
                # sec_points = sec_points.reshape(n_cam, nsamp, 3)
                # sec_net_object_mask = sec_net_object_mask.reshape(n_cam, nsamp)
                hit_points = sec_points[sec_net_object_mask] # [N_sec_hit, 3]
                mask_radiance = torch.zeros_like(sec_points).to(self.device) # [N_total, 3]
                mask_cosine = torch.zeros_like(sec_points[:,0:1]).to(self.device) # [N_total, 1]
                
                mask_cosine = torch.relu(mask_cosine)
                xyz = self.backbone.projection(hit_points.transpose(0,1).contiguous().view(3, -1).unsqueeze(0), calib_k)
                xy = xyz[:,:2,:]
                z = xyz[:,2:,:]
                a = index(img_k, xy, padding_mode='zeros')
                d = index(depth_k, xy, mode='nearest')

                a[:,:,(z>d)[0,0]] = 0 ## hit point not visible, then set zero

                mask_radiance[sec_net_object_mask] = a.transpose(1,2)

                mask_radiance = mask_radiance.reshape(n_cam, nsamp, 3)

                mask_radiance = mask_radiance * cosines.unsqueeze(-1)
                sec_radiance = mask_radiance.sum(dim=1) / nsamp ## average for random sampling rays
                return sec_radiance, sec_points, sec_net_object_mask, a

        
        return 
    
    # def get_albedo_smooth_loss(self, grad_albedo):
    #     if grad_albedo is None:
    #         return torch.tensor(0.0).cuda().float()
    #     eikonal_loss = ((grad_albedo.norm(2, dim=1)) ** 2).mean()
    #     return eikonal_loss

    def get_eikonal_loss(self, space_gradients):
        if space_gradients is None:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((space_gradients.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_albedo_smooth_loss(self, albedo_func, pts, pts_albedo, pts_sdf, std=1e-2, decay=10, feature_dim=3):
        '''
        Params
        : albedo_func
        : pts [N,3]
        : pts_albedo [N,3]
        : pts_sdf [N,1]
        : std, random sample variance
        '''  
        def compute_relative_smoothness_loss(values, values_jittor):
            base = torch.maximum(values, values_jittor).clip(min=1e-6)
            difference = torch.sum(((values - values_jittor) / base)**2, dim=-1, keepdim=True)  # [..., 1]

            return difference
        pts_jittor = pts + torch.rand_like(pts) * std
        pts_jittor_albedo = albedo_func(pts_jittor.detach().T.unsqueeze(0)).transpose(1,2).contiguous().view(-1, feature_dim)

        albedo_smt_cost = compute_relative_smoothness_loss(pts_albedo.detach(), pts_jittor_albedo)
        # beta = torch.exp(-decay * torch.abs(pts_sdf))
        
        # albedo_smt_loss = albedo_smt_cost * beta
        albedo_smt_loss = albedo_smt_cost
        # albedo_smt_loss = 1/(pts_sdf.detach()+1e-6) * albedo_smt_cost
        return torch.mean(albedo_smt_loss)

    def get_coarea_loss(self, ):
        '''
        min area loss
        '''

        pass

    def get_curvature_loss(self, gradient_func, points, sdf_gradients, eps=1e-4):
        '''
        Params
        : gradient_func
        : pts [N,3]
        : sdf_gradients [N,3]
        : eps, random sample variance
        '''
        #get the curvature along a certain random direction for each point
        #does it by computing the normal at a shifted point on the tangent plant and then computing a dot produt
        #to the original positions, add also a tiny eps 
        rand_directions=torch.randn_like(points)
        rand_directions=nnF.normalize(rand_directions,dim=-1)

        #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
        normals=nnF.normalize(sdf_gradients,dim=-1)
        # normals=normals.detach()
        tangent=torch.cross(normals, rand_directions)
        points_shifted=points + tangent * eps
        
        #get the gradient at the shifted point
        sdf_gradients_shifted=gradient_func(points_shifted.detach().T.unsqueeze(0)).transpose(1,2).view(-1,3)

        normals_shifted=nnF.normalize(sdf_gradients_shifted,dim=-1)
        curvature_constraint = 1 - nnF.cosine_similarity(normals, normals_shifted, dim=-1)
        return torch.mean(curvature_constraint)

        # dot=(normals*normals_shifted).sum(dim=-1, keepdim=True)
        # # the dot would assign low weight importance to normals that are almost the same, and increasing error the more they deviate. 
        # # So it's something like and L2 loss. But we want a L1 loss so we get the angle, and then we map it to range [0,1]
        # angle=torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6)) #goes to range 0 when the angle is the same and pi when is opposite


        # curvature=angle/math.pi #map to [0,1 range]

        # return sdf_shifted, curvature

    def get_alignment_loss(self, sdf_func, gradient_func, pts, pts_grad=None, decay=10):
        '''
        Params:
        @ pts: [N, 3]

        Return:
        loss tensor
        '''
        pts_sdf = sdf_func(pts.T.unsqueeze(0)).flatten() #[N]
        pts_grad = gradient_func(pts.T.unsqueeze(0)).transpose(1,2).view(-1,3) if pts_grad is None else pts_grad  #[N, 3]
        pts_grad_norm = nnF.normalize(pts_grad, dim=-1)
        pts_moved = pts - pts_grad_norm * pts_sdf.unsqueeze(1)
        # pts_moved_sdf = sdf_func(pts_moved)
        pts_moved_grad = gradient_func(pts_moved.detach().T.unsqueeze(0)).transpose(1,2).view(-1,3)
        pts_moved_grad_norm = nnF.normalize(pts_moved_grad, dim=-1)
        align_constraint = 1 - nnF.cosine_similarity(pts_grad_norm, pts_moved_grad_norm, dim=-1)
        beta = torch.exp(-decay * torch.abs(pts_sdf))
        # beta = 1
        return torch.mean(beta * align_constraint)


    def get_mask_loss(self, sdf_output, network_object_mask, object_mask, in_original_img, ray_weights, alpha):
        ### We only calculate mask loss in original img
        ### In case of non-square mask padding
        in_gt_mask = object_mask >0.5
        mask = ~(network_object_mask & in_gt_mask) & (in_original_img > 0.5)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

        alpha = torch.exp(self.backbone.k)
        # sdf_pred = -torch.exp(self.backbone.k) * sdf_output[mask]
        sdf_pred = -alpha * sdf_output[mask]
        # sdf_pred = sdf_output[mask]
        mask_ray_weights = ray_weights.flatten()[mask]
        # sdf_pred = -50 * sdf_output[mask]
        gt = object_mask[mask].float()

        # mask_loss = nn.functional.binary_cross_entropy_with_logits(sdf_pred[gt<0.5], gt[gt<0.5], reduction='sum') / float(object_mask.shape[0])  / min(torch.exp(self.backbone.k), 1600)
        # if alpha <= 50:
        mask_loss = (1 / alpha) * (nn.functional.binary_cross_entropy_with_logits(sdf_pred, gt, reduction='none') * mask_ray_weights).sum()  / float(object_mask.shape[0])
        # mask_loss = (1 / alpha) * (nn.functional.binary_cross_entropy_with_logits(sdf_pred[gt<0.5], gt[gt<0.5], reduction='none') * mask_ray_weights[gt<0.5]).sum()  / float(object_mask.shape[0])
        # else:
            # mask_loss = (1 / alpha) * (nn.functional.binary_cross_entropy_with_logits(sdf_pred, gt, reduction='none') * mask_ray_weights).sum()  / float(object_mask.shape[0])
        # mask_loss = nn.functional.binary_cross_entropy_with_logits(sdf_pred, gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss
    
    def get_vgg_loss(self, pred, gt, normalize=True, use_lpips=True):
        '''
        :param pred [BCHW] 
        :param gt [BCHW]
        :param normalize bool
        :return loss_value Scalar
        '''
        if use_lpips:
            return self.loss_fn_vgg(pred, gt, normalize=normalize)
        else:
            vgg_loss = VGGLoss()
    # def get_vgg_loss(self, pred, gt):
    #     vgg_loss = VGGLoss()
    #     loss_vgg = vgg_loss(pred, gt)
    #     return loss_vgg

    def get_albedo_sparse_loss(self, pred_albedo, bins=15):
        albedo_entropy = 0
        for i in range(3):
            channel = pred_albedo[..., i]
            hist = GaussianHistogram(bins, 0., 1., sigma=torch.var(channel))
            h = hist(channel)
            if h.sum() > 1e-6:
                h = h.div(h.sum()) + 1e-6
            else:
                h = torch.ones_like(h).to(h)
            albedo_entropy += torch.sum(-h*torch.log(h))
        return albedo_entropy

    def cal_loss(self, output, sdf_func, gradient_func, albedo_func, spec_func, rough_func, ray_weights=None, ray_in_original_img=None, epoch=0,
            include_smooth_loss=False, alpha=50.0, precomputed_indirect=False, lambda_reg=1, lambda_mask=1, lambda_align=0.01, patch_size=64, writer:SummaryWriter=None, save_intermediate=False):
        # loss_func = nn.L1Loss(reduction='sum')
        # img_loss_func = nn.L1Loss()
        l2_loss = nn.MSELoss()  
        l1_loss = nn.L1Loss(reduction='sum')
        loss = torch.tensor(0.0).to(self.device).float()
        loss_img = torch.tensor(0.0).to(self.device).float()
        loss_mask = torch.tensor(0.0).to(self.device).float()
        loss_ek = torch.tensor(0.0).to(self.device).float()
        loss_spec_smt = torch.tensor(0.0).to(self.device).float()
        loss_rough_smt = torch.tensor(0.0).to(self.device).float()
        loss_albedo_smt = torch.tensor(0.0).to(self.device).float()
        loss_albedo_sparse = torch.tensor(0.0).to(self.device).float()
        loss_curvature = torch.tensor(0.0).to(self.device).float()
        loss_backnormal= torch.tensor(0.0).to(self.device).float()
        loss_align = torch.tensor(0.0).to(self.device).float()
        loss_depth_smooth = torch.tensor(0.0).to(self.device).float()
        loss_vgg = torch.tensor(0.0).to(self.device).float()
        loss_ssim = torch.tensor(0.0).to(self.device).float()
        loss_dict = {}
        network_object_mask = output['network_object_mask'] 
        # object_mask = output['object_mask']
        object_mask = output['ray_ms'] > 0.5
        surface_mask = output['surface_mask']
        sdf_output = output['sdf_output']
        if (network_object_mask).sum() == 0:
            return loss, loss_dict, "no intersect in this iter"
        gt = output['imgs'][surface_mask & object_mask]
        # pdb.set_trace()

        ray_weights = ray_weights if ray_weights is not None else torch.ones_like(output['imgs'][:,:1])
        ray_in_original_img = ray_in_original_img if ray_in_original_img is not None else torch.ones_like(output['imgs'][:,0])

        # intersect_ray_weights = ray_weights[surface_mask & object_mask]
        intersect_ray_weights = ray_weights[surface_mask]
        # shading_gt = output['shadings_gt'][network_object_mask]
        pred_diffuse_shading = output['diffuse_shadings'][surface_mask]
        pred_roughness = output['roughness'][surface_mask]
        pred_pts = output['diff_points']
        pred_depth = output['diff_dists']
        pred_normal = output['normals'][surface_mask]
        space_gradients = output['space_gradients']
        space_pts = output['space_pts']
        points = output['points']
        ray_os = output['ray_os']
        ray_ds = output['ray_ds']

        pred_diff_albedo = output['diff_albedos'][surface_mask]
        pred_spec_albedo = output['specular_albedos'][surface_mask]
        pred = output['preds'][surface_mask & object_mask]

        pred_total_shading = pred_diffuse_shading
        if self.use_indirect:
            pred_indirect = output['indirect_shadings'][surface_mask]
            pred_total_shading = pred_total_shading + pred_indirect

        if precomputed_indirect:
            precompute_indirect = output['indirect_input'][surface_mask]
            pred_total_shading = pred_total_shading + precompute_indirect

        loss_img = l1_loss(img_g2l(gt) * intersect_ray_weights, pred * intersect_ray_weights) / float(object_mask.shape[0])
        mask_loss = self.get_mask_loss(sdf_output, network_object_mask, object_mask, ray_in_original_img, ray_weights, alpha)
        if save_intermediate:
            save_dir = os.path.join(self.debug_vis_path, 'trace_pts/epoch_%d/'%epoch)
            os.makedirs(save_dir, exist_ok=True)
            in_gt_mask = object_mask >0.5
            mask = ~(network_object_mask & in_gt_mask) & (ray_in_original_img > 0.5)
            # alpha = torch.exp(self.backbone.k)
            # sdf_pred = -torch.exp(self.backbone.k) * sdf_output[mask]
            sdf_pred = -alpha * sdf_output[mask] ## negative for out mask values, as in gt is 0 for out mask
            
            pts_cal_mask = points[mask]
            sdf_color = torch.zeros_like(pts_cal_mask)
            sdf_color[sdf_pred>=0, 0] = 1
            sdf_color[sdf_pred<0, 1] = 1
            
            # sdf_pred = sdf_output[mask]
            mask_ray_weights = ray_weights.flatten()[mask]
            # sdf_pred = -50 * sdf_output[mask]
            gt = object_mask[mask].float()
            save_samples_color(save_dir + 'mask_pts.ply', pts_cal_mask.detach().cpu().numpy(), torch.ones_like(pts_cal_mask).detach().cpu().numpy() * sdf_color.detach().cpu().numpy() * 255)

        loss_mask = loss_mask + mask_loss * lambda_mask
        ek_loss = self.get_eikonal_loss(space_gradients)
        loss_ek = loss_ek + ek_loss * lambda_reg

        writer.add_scalar('Loss/l1', loss_img.item(), epoch)
        writer.add_scalar('Loss/mask', loss_mask.item(), epoch)
        writer.add_scalar('Loss/reg', loss_ek.item(), epoch)
        if self.opt.sample_patch:
            gt = img_g2l(output['imgs']).view(1, patch_size, patch_size, 3).permute(0,3,1,2)
            pred = output['preds'].view(1, patch_size, patch_size, 3).permute(0,3,1,2)
            pred_mask = output['network_object_mask'].view(1, patch_size, patch_size, 1).permute(0,3,1,2)
            # vgg_loss = self.get_vgg_loss(gt, pred, normalize=True)
            # vgg_loss = vgg_loss.squeeze()

            vgg_loss = self.loss_fn_pyramid.forward(pred, gt)
            ssim_loss = self.ssim_fn(pred, gt, pred_mask)
            from PIL import Image
            gt_debug_path = os.path.join(self.debug_vis_path, 'patch/', 'ep%d/'%epoch, 'gt_patch.png')
            pred_debug_path = os.path.join(self.debug_vis_path, 'patch/', 'ep%d/'%epoch,'pred_patch.png')
            pred_mask_debug_path = os.path.join(self.debug_vis_path, 'patch/', 'ep%d/'%epoch,'pred_mask_patch.png')
            pred_mask_loss_path = os.path.join(self.debug_vis_path, 'patch/', 'ep%d/'%epoch,'pred_mask_loss.png')
            os.makedirs(os.path.dirname(gt_debug_path), exist_ok=True)
            Image.fromarray(to8b(output['imgs'].view(patch_size, patch_size, 3).detach().cpu().numpy())).save(gt_debug_path)
            Image.fromarray(to8b(img_l2g(output['preds']).view(patch_size, patch_size, 3).detach().cpu().numpy())).save(pred_debug_path)
            Image.fromarray(to8b(output['network_object_mask'].view(patch_size, patch_size, 1).expand(-1,-1,3).detach().cpu().numpy())).save(pred_mask_debug_path)
            Image.fromarray(to8b( ~(output['network_object_mask'] & (output['ray_ms']>0.5)).view(patch_size, patch_size, 1).expand(-1,-1,3).detach().cpu().numpy())).save(pred_mask_loss_path)
            loss_vgg = loss_vgg + vgg_loss * self.opt.lambda_vgg
            loss_ssim = loss_ssim + ssim_loss * self.opt.lambda_ssim

        if self.opt.use_depth_smooth_loss:
            len_mask = len(surface_mask)//2
            valid_mask = surface_mask[:len_mask] & surface_mask[len_mask:]
            smooth_loss = nn.MSELoss()(pred_depth[:len_mask][valid_mask], pred_depth[len_mask:][valid_mask])
            loss_depth_smooth = loss_depth_smooth + smooth_loss * self.opt.lambda_smooth_depth
            writer.add_scalar('Loss/depth_smooth', loss_depth_smooth.item(), epoch)
        if self.opt.use_align_loss:
            align_loss = self.get_alignment_loss(sdf_func, gradient_func, space_pts, space_gradients)
            loss_align = loss_align + align_loss * lambda_align
            writer.add_scalar('Loss/align', loss_align.item(), epoch)

        if self.opt.use_spec_smooth_loss:
            spec_smooth_loss = self.get_albedo_smooth_loss(spec_func, pred_pts[surface_mask], pred_spec_albedo, sdf_output[surface_mask], std=0.01)
            loss_spec_smt = loss_spec_smt + spec_smooth_loss * self.opt.lambda_smooth_spec
            writer.add_scalar('Loss/spec_smooth', loss_spec_smt.item(), epoch)
        
        if self.opt.use_rough_smooth_loss:
            rough_smooth_loss = self.get_albedo_smooth_loss(rough_func, pred_pts[surface_mask], pred_roughness, sdf_output[surface_mask], std=0.01, feature_dim=1)
            loss_rough_smt = loss_rough_smt + rough_smooth_loss * self.opt.lambda_smooth_rough
            writer.add_scalar('Loss/rough_smooth', loss_rough_smt.item(), epoch)

        if self.opt.use_alb_smooth_loss:
            albedo_smooth_loss = self.get_albedo_smooth_loss(albedo_func, pred_pts[surface_mask], pred_diff_albedo, sdf_output[surface_mask], std=0.01)
            loss_albedo_smt = loss_albedo_smt + albedo_smooth_loss * self.opt.lambda_smooth_albedo
            writer.add_scalar('Loss/albedo_smooth', loss_albedo_smt.item(), epoch)
        
        if self.opt.use_alb_sparse_loss:
            albedo_sparse_loss = self.get_albedo_sparse_loss(pred_diff_albedo)
            loss_albedo_sparse = loss_albedo_sparse + albedo_sparse_loss * self.opt.lambda_sparse_albedo
            writer.add_scalar('Loss/albedo_sparse', loss_albedo_sparse.item(), epoch)

        if self.opt.use_curvature_loss:
            # curvature_loss = self.get_curvature_loss(gradient_func, pred_pts[surface_mask], pred_normal, eps=1e-4)
            curvature_loss = self.get_curvature_loss(gradient_func, space_pts, space_gradients, eps=1e-4)
            loss_curvature = loss_curvature + curvature_loss * self.opt.lambda_curvature
            writer.add_scalar('Loss/curvature', loss_curvature.item(), epoch)
        
        if self.opt.use_backnormal_loss:
            loss_backnormal = torch.mean(output['back_normal_penalty']) * self.opt.lambda_backnormal
            writer.add_scalar('Loss/back_normal', loss_backnormal.item(), epoch)
        
        # if self.opt.use_roughness_loss:
        #     loss_roughness = (pred_roughness[pred_roughness>0.5] - 0.5).mean() * self.opt.lambda_roughrange

        # pdb.set_trace()
        
        

        if self.verbose:
            print('gt',  gt)
            print('gt_g2l',  img_g2l(gt))
            print('pred', pred)
        
        loss_dict['img_loss'] = loss_img
        loss_dict['mask_loss'] = loss_mask
        loss_dict['ek_loss'] = loss_ek
        loss_dict['align_loss'] = loss_align
        loss_dict['abd_smt_loss'] = loss_albedo_smt
        loss_dict['vgg_loss'] = loss_vgg
        loss_dict['ssim_loss'] = loss_ssim
        # loss = loss_mask + loss_ek + loss_abd_smt
        # loss = loss_img * 0.01 + loss_mask + loss_ek
        loss = loss_img + loss_mask + loss_ek + loss_align + loss_curvature + loss_backnormal + loss_vgg + loss_ssim + loss_albedo_sparse

        if include_smooth_loss:
            loss = loss + loss_albedo_smt + loss_spec_smt + loss_rough_smt + loss_depth_smooth

        loss_str = 'lo_im: %.3f,lmd_ma:%.3f,ma:%.3f,lmd_ek:%.3f,ek:%.3f'\
             %(loss_img, lambda_mask, loss_mask, lambda_reg, loss_ek)
        if self.opt.sample_patch:
            loss_str = loss_str + ',vgg:%.4f' % loss_vgg 
            loss_str = loss_str + ',ssim:%.4f' % loss_ssim
        
        if self.opt.use_align_loss:
            loss_str = loss_str + ',algn:%.4f' % loss_align
        if self.opt.use_spec_smooth_loss:
            loss_str = loss_str + ',spec_smt:%.4f' % loss_spec_smt
        if self.opt.use_alb_smooth_loss:
            loss_str = loss_str + ',alb_smt:%.4f' % loss_albedo_smt
        if self.opt.use_alb_sparse_loss:
            loss_str = loss_str + ',alb_sps:%.4f' % loss_albedo_sparse
        if self.opt.use_rough_smooth_loss:
            loss_str = loss_str + ',rough_smt:%.4f' % loss_rough_smt
        if self.opt.use_depth_smooth_loss:
            loss_str = loss_str + ',dep_smt:%.4f' % loss_depth_smooth
        if self.opt.use_curvature_loss:
            loss_str = loss_str + ',curv:%.4f' % loss_curvature
        if self.opt.use_backnormal_loss:
            loss_str = loss_str + ',bnor:%.4f' % loss_backnormal
           
        return loss, loss_dict, loss_str


class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=torch.device('cuda:0')).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1) # [K,N]
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x #[K]


# if __name__ == '__main__':
#     from options.options import get_options
#     from dataset.Thuman2_pifu_dataset_sdf import make_dataset
#     from utils.common_utils import set_random_seed
#     import numpy as np 
#     opt = get_options()
#     opt.path_to_dataset = '/home/lujiawei/workspace/dataset/thuman2_rescaled_prt_512_single_light_w_flash_no_env_persp_fov_60_unit_cube'
#     opt.path_to_obj = '/home/lujiawei/workspace/dataset/thuman2_rescaled'
#     opt.path_to_sample_pts = '/home/lujiawei/workspace/dataset/SAMPLE_PT_SCALE'
#     opt.load_pretrained_path = '/home/lujiawei/workspace/tex2cloth/checkpoints/20220801/train_equal_angle_no_decay_3views_w_reg_w_normal_w_g1_w_bce_use_confidence_swish_HG_filter_flash_perspective_no_visual_hull_unit_cube/model_weights_G.tar'
#     opt.load_pretrained_path_color = '/home/lujiawei/workspace/tex2cloth/checkpoints/20220818/train_light_w_flash_no_env_pred_intensity_perspective_no_visuall_hull_unit_cube/model_weights_G.tar'
#     opt.num_views = 3
#     opt.line_step_iters = 3
#     opt.line_search_step = 0.5
#     opt.sphere_tracing_iters = 10
#     opt.n_secant_steps = 8
#     opt.n_steps = 100
#     opt.object_bounding_sphere = 0.7
#     opt.sdf_threshold = 5e-5
#     opt.mlp_activation_type = 'silu'
#     opt.norm = 'group'
#     opt.field_type = 'sdf'

#     opt.use_perspective=True
#     opt.use_feature_confidence = True
#     opt.offline_sample = True
#     opt.align_corner = True
#     opt.random_scale = True
#     opt.random_trans = True
#     opt.random_aug_offset = 0
#     test_dataset = make_dataset(opt, 'Train')
#     test_dataset.is_train = True

#     set_random_seed(opt.seed)

#     device = torch.device("cuda:0")
#     cpu = torch.device("cpu")

#     id = 0
#     data11 = test_dataset[id]
    
#     calib = data11['calib']
#     mask = data11['mask']
#     name = data11['name']
#     sid = data11['sid']
#     yid = data11['yid']
#     view_ids = data11['view_ids']
#     samples = data11['samples']
#     surface_samples = data11['surface_samples']
#     labels = data11['labels']
#     extrinsic = data11['extrinsic']
#     intrinsic = data11['intrinsic']
#     cam_extrinsics = data11['cam_extrinsics']
#     normal_matrices = data11['normal_matrices']
#     cam_centers = data11['cam_centers']
#     cam_directions = data11['cam_directions']
#     img = data11['img']

#     print(name)

#     device = torch.device("cuda:0")
#     cpu = torch.device("cpu")

#     img = img.to(device)
#     mask = mask.to(device)
#     w2i = torch.bmm(intrinsic, cam_extrinsics).to(device)
#     w2c = cam_extrinsics.to(device)
#     m2c = extrinsic.to(device)
#     c_in = intrinsic.to(device)
#     normal_mat = normal_matrices.to(device)

#     path_to_ckpt_Geo = opt.load_pretrained_path
#     path_to_ckpt_Color = opt.load_pretrained_path_color
#     dr = DiffRenderer_unified(opt, opt.num_views, debug=True, device=device).to(device)
#     checkpoint_G = torch.load(path_to_ckpt_Geo, map_location=device)
#     checkpoint_C = torch.load(path_to_ckpt_Color, map_location=device)
#     dr.geo_mlp.load_state_dict(checkpoint_G['G_state_dict'], strict=True)
#     dr.albedo_mlp.load_state_dict(checkpoint_C['G_state_dict'], strict=True)
#     dr.train()

#     psize = 64

#     Np = 512 // psize

#     for px in range(Np):
#         for py in range(Np):
#             uv = (np.mgrid[px*psize : (px+1)*psize, py*psize : (py+1) * psize].astype(np.int32) - 256 + 0.5) / 256
#             mask_batch = mask[:, :, px*psize : (px+1)*psize, py*psize : (py+1) * psize]
#             if mask_batch.sum()==0:
#                 print('no object in patch [%d, %d]'%(px, py))
#                 continue
#             # uv = np.mgrid[0:512, 0:512].astype(np.int32) 
#             uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float() # flip to get correct x,y instead of y,x
#             uv = uv.view(2, -1).transpose(1, 0).unsqueeze(0).repeat(opt.num_views, 1, 1)
#             uv = -uv # here we flip the uv as the z axis is positive and the camera is toward -z
#             uv = uv.to(device)

#             output = dr.forward(uv, img, m2c, c_in, normal_mat, mask, px, py, psize=psize)
#             if len(output) == 0:
#                 print('output is none, continue')
#                 continue
#             image_loss = dr.cal_loss(output)
#             print(image_loss.item())
#             image_loss.backward()
#             print(dr.geo_mlp.filters[0].weight.grad)
#             print(dr.geo_mlp.filters[0].weight.grad.max())
#             print(dr.geo_mlp.filters[0].weight.grad.min())
            
#             print(dr.albedo_mlp.albedo_classifier.filters[0].weight.grad)
#             print(dr.albedo_mlp.albedo_classifier.filters[0].weight.grad.max())
#             print(dr.albedo_mlp.albedo_classifier.filters[0].weight.grad.min())


