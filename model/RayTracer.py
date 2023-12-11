import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from utils.sdf_utils import *
import pdb
import time
from tqdm import tqdm
import os
from matplotlib import cm

class RayTracing(nn.Module):
    def __init__(self, opt, num_views, debug=True, verbose=True, device='cuda', debug_vis_path=''):
        super().__init__()
        self.opt = opt
        self.device = device
        self.num_views = num_views
        self.line_step_iters = self.opt.line_step_iters # how many steps back if goes across the surface
        self.line_search_step = self.opt.line_search_step # if this step goes into the object, step back a little and re-tracing
        self.sphere_tracing_iters = self.opt.sphere_tracing_iters # num of steps for sphere tracing
        self.n_secant_steps = self.opt.n_secant_steps # num of steps for secant method 
        self.n_steps = self.opt.n_steps # num of steps for non-converge rays
        self.object_bounding_sphere = self.opt.object_bounding_sphere
        self.sdf_threshold = self.opt.sdf_threshold
        self.debug = debug
        self.debug_vis_path = debug_vis_path
        self.chunk_size = self.opt.num_sample_dr * 4
        self.trunc_threshold = 0.1

        self.verbose =verbose
        self.eps = 1e-10

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf):
        ''' Runs the secant method for interval [z_low, z_high] for n_secant_steps '''

        z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low + self.eps) + z_low
        for i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid.view(1,-1,3).transpose(1,2)).flatten()
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]

            z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low + self.eps) + z_low
            
        return z_pred
    def ray_sampler(self, sdf, ray_os, ray_ds, object_mask, sampler_min_max, sampler_work_mask):
        ''' 
        Handle the rays which not converge in sphere tracing
        run secant on rays which have sign transition 

        :param sdf  func                            sdf_func, given world point to sdf
        :param ray_os [N_rays, 3]                   ray_os, in world space
        :param ray_ds [N_rays, 3]:                  ray_ds, in model space, normalized
        :param object_mask [N_rays]                 gt masks
        :param sampler_min_max [N_rays,2]:          initial intersection t_front and t_back from sphere tracing
        :param sampler_work_mask [N_rays]:               unfinished mask start

        :return sampler_pts, [N_rays, 3]            the points converge at sphere tracing
        :return sampler_convergent_mask,[N_rays]       not converged points at this step
        :return sampler_dists,      [N_rays]        points start distance after sphere tracing
        '''
        N_rays, _ = ray_ds.shape
        sampler_pts = torch.zeros_like(ray_ds)
        sampler_dists = torch.zeros_like(object_mask, dtype=ray_os.dtype)

        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).to(self.device).view(1, -1)

        ## [N_rays, 1] * [1, n_step] = [N_rays, n_step]
        pts_intervals = sampler_min_max[:, 0].unsqueeze(-1) + intervals_dist * (sampler_min_max[:, 1] - sampler_min_max[:, 0]).unsqueeze(-1)
        """pts_intervals: [N_rays, n_step]"""
        ## [N_rays, n_step, 1] * [N_rays, 1, 3] = [N_rays, n_step, 3] + ray_os [N_rays, 1, 3]
        points = ray_os.unsqueeze(1) +  ray_ds.unsqueeze(1) * pts_intervals.unsqueeze(-1)
        """points: [N_rays, n_step, 3]"""

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(sampler_work_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_work_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_work_mask]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), self.chunk_size, dim=0):
            sdf_chunk = sdf(pnts.view(1,-1,3).transpose(1,2)).flatten()
            sdf_val_all.append(sdf_chunk)
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps) #[N_rays, 100]

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).to(self.device).float().reshape((1, self.n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1)
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]

        true_surface_pts = object_mask[sampler_work_mask]
        net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)

        # take points with minimal SDF value for P_out pixels
        p_out_mask = ~(true_surface_pts & net_surface_pts)
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][torch.arange(n_p_out), out_pts_idx, :]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        sampler_convergent_mask = sampler_work_mask.clone()
        sampler_convergent_mask[mask_intersect_idx[~net_surface_pts]] = False

        # Run Secant method
        secant_pts = net_surface_pts & true_surface_pts if self.training else net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]

            ## awful bug
            # cam_loc_secant = ray_os.unsqueeze(1).repeat(1, N_rays, 1).reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            cam_loc_secant = ray_os[mask_intersect_idx[secant_pts]]
            ray_directions_secant = ray_ds.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)

            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        return sampler_pts, sampler_convergent_mask, sampler_dists

    def sphere_tracing(self, sdf, ray_os, ray_ds, mask_init, fb_dist_init, use_TSDF=False, TSDF_thres=0.1):
        ''' 
        Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection 
        
        :param sdf  func                        sdf_func, given world point to sdf
        :param ray_os [N_rays, 3]                   ray_os, in world space
        :param ray_ds [N_rays, 3]           ray_ds, in world space, normalized
        :param mask_init [N_rays]:             initial intersection mask with the (unit sphere)/visual hull
        :param fb_dist_init [N_rays, 2]:    initial intersection t_front and t_back with visual hull

        :return curr_start_points, [N_rays, 3]      the points converge at sphere tracing from front to back
        :return curr_end_points, [N_rays, 3]        the points converge at sphere tracing from back to front
        :return unfinished_mask_start,[N_rays]      not converged points by sphere tracing but in mask
        :return unfinished_mask_end,[N_rays]        not converged points by sphere tracing but in mask
        :return acc_start_dis,      [N_rays]        t_start after sphere tracing
        :return acc_end_dis,        [N_rays]        t_end   after sphere tracing
        :return min_dis,                        points min distance without sphere tracing
        :return max_dis                         points max distance without sphere tracing
        '''
        cmap = cm.get_cmap('rainbow')
        N_rays, _ = ray_ds.shape # has order
        # [N_rays, 1, 3] + [N_rays, 1, 3] * [N_rays, 2, 1]
        ''' [N_ray, 2, 3]'''
        init_intersect_points = ray_os.unsqueeze(1) +  ray_ds.unsqueeze(1) * fb_dist_init.unsqueeze(-1)
        unfinished_mask_start = mask_init.clone()
        unfinished_mask_end = mask_init.clone()
        # Initialize start current points
        curr_start_points = init_intersect_points[:,0,:]
        # curr_start_points = torch.zeros_like(ray_ds)
        # curr_start_points[unfinished_mask_start] = init_intersect_points[:,0,:][unfinished_mask_start]
        # Initialize end current points
        # curr_end_points = torch.zeros_like(ray_ds)
        curr_end_points = init_intersect_points[:,1,:]
        # curr_end_points[unfinished_mask_end] = init_intersect_points[:,1,:][unfinished_mask_end]

        acc_start_dis = fb_dist_init[:, 0]
        acc_end_dis = fb_dist_init[:, 1]
        # acc_start_dis = torch.zeros_like(mask_init, dtype=ray_os.dtype)
        # acc_start_dis[unfinished_mask_start] = fb_dist_init[unfinished_mask_start,0]
        # acc_end_dis = torch.zeros_like(mask_init, dtype=ray_os.dtype)
        # acc_end_dis[unfinished_mask_end] = fb_dist_init[unfinished_mask_end, 1]
        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0
        next_sdf_start = torch.zeros_like(acc_start_dis)
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start].view(1,-1,3).transpose(1,2)).flatten() 
        next_sdf_end = torch.zeros_like(acc_end_dis)
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end].view(1,-1,3).transpose(1,2)).flatten()
        is_debug=False
        # save_dir = '/home/lujiawei/workspace/tex2cloth/visualize_result/20230723/0723_sit1_n4_m44_from_align_fscratch_7_use_align_loss_eik_in_unit_box_eik_jittored_0.01_opt_light_debug4_sample_intersect/debug2/'
        save_dir = os.path.join(self.debug_vis_path, 'debug/')
        if is_debug:
            ## debug
            os.makedirs(save_dir, exist_ok=True)
            save_samples_color(save_dir + 'start_pt_iter_%d.ply' % (iters), curr_start_points.cpu(), torch.rand_like(curr_start_points[0:1,:]).expand_as(curr_start_points).detach().cpu() * 255)
            save_samples_color(save_dir + 'end_pt_iter_%d.ply' % (iters), curr_end_points.cpu(), torch.rand_like(curr_end_points[0:1,:]).expand_as(curr_end_points).detach().cpu() * 255)

        while True:

            # Update sdf
            curr_sdf_start = torch.zeros_like(acc_start_dis)
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]

            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = torch.zeros_like(acc_end_dis)
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            ## for TSDF, not exceed the surface
            if use_TSDF:
                curr_sdf_start = torch.clamp(curr_sdf_start, max=TSDF_thres)
                curr_sdf_end = torch.clamp(curr_sdf_end, max=TSDF_thres)

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold)

            if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                if self.verbose:
                    if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0):
                        print('all finished')
                    if iters==self.sphere_tracing_iters:
                        print('max iter')
                break
            iters += 1

            # Make step
            # Update distance
            # here our z axis is toward z+
            
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end
            # Update points
            curr_start_points = ray_os + acc_start_dis.reshape(N_rays, 1) * ray_ds
            curr_end_points = ray_os + acc_end_dis.reshape(N_rays, 1) * ray_ds
            
            if is_debug:
                ## debug
                os.makedirs(save_dir, exist_ok=True)
                color_id = iters / self.sphere_tracing_iters
                color = cmap(color_id)
                save_samples_color(save_dir + 'start_pt_iter_%d.ply' % (iters), curr_start_points.cpu(), torch.ones_like(curr_start_points).detach().cpu().numpy()*color[:3] * 255)
                save_samples_color(save_dir + 'end_pt_iter_%d.ply' % (iters), curr_end_points.cpu(), torch.ones_like(curr_start_points).detach().cpu().numpy()*color[:3] * 255)

            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis)
            if unfinished_mask_start.sum() > 0:
                next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start].view(1, -1, 3).transpose(1,2)).flatten()

            next_sdf_end = torch.zeros_like(acc_end_dis)
            if unfinished_mask_end.sum() > 0:
                next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end].view(1, -1, 3).transpose(1,2)).flatten()

            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (not_projected_start.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] = acc_start_dis[not_projected_start] - ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (ray_os + acc_start_dis.reshape(N_rays, 1) * ray_ds)[not_projected_start]
                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start].view(1, -1, 3).transpose(1,2)).flatten()
                # Update mask
                not_projected_start = next_sdf_start < 0
                not_proj_iters += 1
            
            not_proj_iters = 0
            while (not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_end_dis[not_projected_end] = acc_end_dis[not_projected_end] + ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (ray_os + acc_end_dis.reshape(N_rays, 1) * ray_ds)[not_projected_end]
                # Calc sdf
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end].view(1, -1, 3).transpose(1,2)).flatten()
                # Update mask
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1
            
            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

            
            # if is_debug:
            #     ## debug
            #     os.makedirs(save_dir, exist_ok=True)
            #     save_samples_color(save_dir + 'start_pt_iter_%d_step_back.ply' % (iters), curr_start_points.cpu(), torch.rand_like(curr_start_points[0:1,:]).expand_as(curr_start_points).detach().cpu() * 255)
            #     save_samples_color(save_dir + 'end_pt_iter_%d_step_back.ply' % (iters), curr_end_points.cpu(), torch.rand_like(curr_end_points[0:1,:]).expand_as(curr_end_points).detach().cpu() * 255)


        return curr_start_points,curr_end_points, unfinished_mask_start,unfinished_mask_end, acc_start_dis, acc_end_dis, min_dis, max_dis

    def forward(self,sdf,
                front_back_intersections,
                mask_intersect,
                cam_loc,
                view_ids,
                sample_id_k,
                object_mask,
                # calibs,
                # masks,
                ray_directions,
                size=(512,512)):
        '''
        Given sdf evaluation function, camera location, object mask, and ray directions, 
        predict the intersection point of rays and sdf, and the intersection mask.

        :param sdf:                         function to predict sdf at a given point in model space
        :param front_back_intersections [B,N,2]:   intersections of this view ray and visual hull
        :param mask_intersect [B,N,1]:             intersect mask of this view ray and visual hull
        :param cam_loc [B, 3]:             camera locations in model space
        :param object_mask [B,1,H,W]:      mask of object in each view
        :param calibs [BK, 4, 4]:           camera calibrations, from model space to image space
        :param ray_directions [B, N, 3]:   sampled rays to calculate intersection

        :return curr_start_points [1, BN, 3]
        :return network_object_mask [1, BN]
        :return acc_start_dis [1, BN]

        '''
        n_cams, num_pixels, _  = ray_directions.shape
        batch_size = 1

        # _,_,H,W = object_mask.shape
        H,W = size
        # get intersections in model space

        ## can be precomputed
        # front_back_intersections, mask_intersect = get_ray_visual_hull_intersection(cam_loc, ray_directions, calibs=calibs, masks=masks, radius=self.object_bounding_sphere)
        # front_back_intersections, mask_intersect = get_sphere_intersection(cam_loc=cam_loc, ray_directions=ray_directions, r=self.object_bounding_sphere, min=0.001)
        # sample the dilated mask to get all possible ray directions

        curr_start_points_all = []
        network_object_masks_all = []
        acc_start_dis_all = []

        for k in range(self.num_views):
            this_view_id = view_ids[k]
            this_view_mask_intersect = mask_intersect
            if this_view_mask_intersect.sum() == 0:
                curr_start_points_all.append(torch.zeros_like(this_view_mask_intersect).expand(-1,-1,3).view(-1,3))
                network_object_masks_all.append(torch.zeros_like(this_view_mask_intersect).view(-1))
                acc_start_dis_all.append(torch.zeros_like(this_view_mask_intersect).view(-1))
                continue
            this_view_front_back_intersections = front_back_intersections
            this_view_ray_dirs = ray_directions
            this_view_cam_locs = cam_loc
            this_view_object_mask = object_mask > 0.5

            this_view_mask_intersect = this_view_mask_intersect.view(-1)
            this_view_object_mask = this_view_object_mask.view(-1)
            if self.debug:
                os.makedirs(self.debug_vis_path, exist_ok=True)
                save_path = self.debug_vis_path + '/front_visual_hull_intersection_view_%d.png' % this_view_id
                mask_img = torch.zeros(H*W, dtype=bool).view(-1).to(self.device)
                mask_img[sample_id_k] = this_view_mask_intersect
                img = mask_img.view(batch_size, 1, H, W)[0].permute(1,2,0).expand(-1,-1,3).cpu().numpy()* 255.0
                Image.fromarray(np.uint8(img)).save(save_path)
                start_points_unfinished = this_view_cam_locs[0].unsqueeze(0) + (this_view_ray_dirs[0] * this_view_front_back_intersections[0,:,0:1])[this_view_mask_intersect.view(batch_size, -1)[0]]
                # print(start_points_unfinished.shape)
                save_samples_color(self.debug_vis_path + '/visual_hull_intersection_front_view_%d.ply' % this_view_id, start_points_unfinished.cpu(), torch.ones_like(start_points_unfinished).cpu() * 255)
                end_points_unfinished = this_view_cam_locs[0].unsqueeze(0) + (this_view_ray_dirs[0] * this_view_front_back_intersections[0,:,1:2])[this_view_mask_intersect.view(batch_size, -1)[0]]
                # print(end_points_unfinished.shape)
                save_samples_color(self.debug_vis_path + '/visual_hull_intersection_back_view_%d.ply' % this_view_id, end_points_unfinished.cpu(), torch.ones_like(end_points_unfinished).cpu() * 255)
                
            # as sdf func takes single view as input, here we do tracing at each view
            t2 = time.time()

            # may pass through some points
            curr_start_points, curr_end_points, unfinished_mask_start, unfinished_mask_end, acc_start_dis, acc_end_dis, min_dis, max_dis = \
                self.sphere_tracing(
                    sdf, 
                    this_view_cam_locs, 
                    this_view_ray_dirs, 
                    this_view_mask_intersect, 
                    this_view_front_back_intersections)
            
            if torch.any(torch.isnan(acc_start_dis)):
                print('sphere tracing has nan')
            t3 = time.time()
            
            if self.debug:
                save_path = self.debug_vis_path + '/front_sphere_tracing_view_%d.png' % this_view_id
                mask_img = torch.zeros(H*W, dtype=bool).view(-1).to(self.device)
                mask_img[sample_id_k] = this_view_mask_intersect

                img = mask_img.view(batch_size, 1, H, W)[0].permute(1,2,0).expand(-1,-1,3).cpu().numpy()* 255.0
                Image.fromarray(np.uint8(img)).save(save_path)
                start_points_unfinished = curr_start_points[this_view_mask_intersect]
                save_samples_color(self.debug_vis_path + '/sphere_tracing_intersection_front_view_%d.ply' % this_view_id, start_points_unfinished.cpu(), torch.ones_like(start_points_unfinished).cpu() * 255)
                

            network_object_mask = (acc_start_dis < acc_end_dis).view(-1)
            # network_object_mask = (acc_start_dis < acc_end_dis).view(-1)
            # pdb.set_trace()
            # The non convergent rays should be handled by the sampler
            sampler_work_mask = unfinished_mask_start # [BN]
            sampler_convergent_mask = torch.zeros_like(sampler_work_mask).bool().to(self.device)

            if sampler_work_mask.sum() > 0:
                sampler_min_max = torch.zeros((n_cams, num_pixels, 2)).to(self.device)
                sampler_min_max.view(-1, 2)[sampler_work_mask, 0] = acc_start_dis[sampler_work_mask]
                sampler_min_max.view(-1, 2)[sampler_work_mask, 1] = acc_end_dis[sampler_work_mask]

                sampler_pts, sampler_convergent_mask, sampler_dists = self.ray_sampler(sdf,
                                                                                    this_view_cam_locs,
                                                                                    this_view_object_mask.view(-1),
                                                                                    this_view_ray_dirs,
                                                                                    sampler_min_max,
                                                                                    sampler_work_mask
                                                                                    )
                
                curr_start_points[sampler_work_mask] = sampler_pts[sampler_work_mask]
                acc_start_dis[sampler_work_mask] = sampler_dists[sampler_work_mask]
                network_object_mask[sampler_work_mask] = sampler_convergent_mask[sampler_work_mask]
            t4 = time.time()

            if self.verbose:
                print('sphere tracing time: ', t3-t2)
                print('ray sampler time: ', t4-t3)

                print('----------------------------------------------------------------')
                print('RayTracing: object = {0}/{1}, secant on {2}/{3}.'
                    .format(network_object_mask.sum(), len(network_object_mask), sampler_convergent_mask.sum(), sampler_work_mask.sum()))
                print('----------------------------------------------------------------')
                
            if self.debug:
                save_path = self.debug_vis_path + '/tracing_with_sampler_%d.png' % this_view_id
                mask_img = torch.zeros(H*W, dtype=bool).view(-1).to(self.device)
                mask_img[sample_id_k] = network_object_mask
                img = mask_img.view(batch_size, 1, H, W)[0].permute(1,2,0).expand(-1,-1,3).cpu().numpy()* 255.0
                Image.fromarray(np.uint8(img)).save(save_path)
                secant_points = curr_start_points[sampler_convergent_mask]
                save_samples_color(self.debug_vis_path + '/secant_points_%d.ply' % this_view_id, secant_points.cpu(), torch.ones_like(secant_points).cpu() * 255)
                start_points_unfinished = curr_start_points[network_object_mask]
                save_samples_color(self.debug_vis_path + '/sphere_tracing_intersection_front_view_sampler_%d.ply' % this_view_id, start_points_unfinished.cpu(), torch.ones_like(start_points_unfinished).cpu() * 255)



            
            if not self.training:
                curr_start_points_all.append(curr_start_points)
                network_object_masks_all.append(network_object_mask)
                acc_start_dis_all.append(acc_start_dis)
                continue

            '''
            network_object_mask: traced points with dist_start < dist_end, means the mask of the object predicted by the network 
            this_view_object_mask: ground truth mask of this view of object
            sampler_work_mask: unfinished in sphere tracing step
            
            in_mask: in the gt mask but no intersections
            out_mask: out the gt mask and 

            '''
            in_mask = ~network_object_mask & this_view_object_mask & ~sampler_work_mask
            out_mask = ~this_view_object_mask & ~sampler_work_mask
            mask_left_out = (in_mask | out_mask) & ~this_view_mask_intersect

            # if self.debug:
                # save_path1 = self.debug_vis_path + '/in_mask_%d.png' % k
                # save_path2 = self.debug_vis_path + '/out_mask_%d.png' % k
                # save_path3 = self.debug_vis_path + '/mask_left_out_%d.png' % k
                # save_path4 = self.debug_vis_path + '/sampler_mask_%d.png' % k
                # save_path5 = self.debug_vis_path + '/gt_object_mask_%d.png' % k
                # img = in_mask.view(batch_size, 1, H, W)[0].permute(1,2,0).expand(-1,-1,3).cpu().numpy()* 255.0
                # Image.fromarray(np.uint8(img)).save(save_path1)
                # img = out_mask.view(batch_size, 1, H, W)[0].permute(1,2,0).expand(-1,-1,3).cpu().numpy()* 255.0
                # Image.fromarray(np.uint8(img)).save(save_path2)
                # img = mask_left_out.view(batch_size, 1, H, W)[0].permute(1,2,0).expand(-1,-1,3).cpu().numpy()* 255.0
                # Image.fromarray(np.uint8(img)).save(save_path3)
                # img = sampler_work_mask.view(batch_size, 1, H, W)[0].permute(1,2,0).expand(-1,-1,3).cpu().numpy()* 255.0
                # Image.fromarray(np.uint8(img)).save(save_path4)
                # img = this_view_object_mask.view(batch_size, 1, H, W)[0].permute(1,2,0).expand(-1,-1,3).cpu().numpy()* 255.0
                # Image.fromarray(np.uint8(img)).save(save_path5)

            if mask_left_out.sum() > 0:  # project the origin to the not intersect points on the sphere
                cam_left_out = this_view_cam_locs.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask_left_out]
                rays_left_out = this_view_ray_dirs.view(-1,3)[mask_left_out]
                acc_start_dis[mask_left_out] = -torch.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze()
                curr_start_points[mask_left_out] = cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out

            mask = (in_mask | out_mask) & this_view_mask_intersect

            if mask.sum() > 0:
                min_dis[network_object_mask & out_mask] = acc_start_dis[network_object_mask & out_mask]
                
                min_mask_points, min_mask_dist = self.minimal_sdf_points(num_pixels, sdf, this_view_cam_locs, this_view_ray_dirs, mask, min_dis, max_dis)
                
                curr_start_points[mask] = min_mask_points
                acc_start_dis[mask] = min_mask_dist


            curr_start_points_all.append(curr_start_points)
            network_object_masks_all.append(network_object_mask)
            acc_start_dis_all.append(acc_start_dis)
        
        curr_start_points_all = torch.stack(curr_start_points_all) # [K, BN, 3]
        network_object_masks_all = torch.stack(network_object_masks_all) # [K, BN]
        acc_start_dis_all = torch.stack(acc_start_dis_all) # [K, BN]
        if torch.any(torch.isnan(acc_start_dis_all)):
            print('tracing dist has nan')
        if torch.any(torch.isnan(curr_start_points_all)):
            print('tracing points has nan')
        return curr_start_points_all, \
               network_object_masks_all, \
               acc_start_dis_all
    
    def forward_ray_batch(self, sdf,
        front_back_dist_init,
        mask_init,
        gt_mask,
        ray_os,
        ray_ds,
        use_TSDF=False,
        TSDF_thres=0.1,
        epoch=0,
        save_intermediate=False):
        '''
        Parameters:
        sdf (func):     sdf func give x output sdf
        front_back_dist_init (Tensor):  [N, 2] init dist of rays (intersect with sphere or visual hull)
        mask_init(Tensor):      [N] init mask of rays (intersect with sphere or visual hull)
        gt_mask(Tensor):    [N] gt mask of rays
        ray_os(Tensor):     [N, 3] ray origins
        ray_ds(Tensor):     [N, 3] ray dirs
        
        Returns:
        start_points(Tensor):   [N, 3]
        converged_rays(Tensor): [N]
        start_dists(Tensor):    [N]
        '''
        is_debug=save_intermediate
        if is_debug:
            save_dir = os.path.join(self.debug_vis_path, 'trace_pts/epoch_%d/'%epoch)
            os.makedirs(save_dir, exist_ok=True)
        N_rays, _ = ray_os.shape
        if mask_init.sum() == 0:
            return None
        object_mask_gt = gt_mask > 0.5

        # as sdf func takes single view as input, here we do tracing at each view
        t2 = time.time()
        init_intersect_points = ray_os +  ray_ds * front_back_dist_init[:,0:1]
        if is_debug:
            ## debug
            color = cm.get_cmap('rainbow')(0)
            save_samples_color(save_dir + 'pts_init.ply', init_intersect_points.cpu(), torch.ones_like(init_intersect_points).detach().cpu().numpy()*color[:3] * 255)


        # trace from front and back
        curr_start_points, _, unfinished_mask_start, _, acc_start_dis, acc_end_dis, min_dis, max_dis = \
            self.sphere_tracing(
                sdf, 
                ray_os, 
                ray_ds, 
                mask_init, 
                front_back_dist_init,
                use_TSDF,
                TSDF_thres)
        
        after_intersect_points = ray_os +  ray_ds * acc_start_dis.unsqueeze(-1)
        network_object_mask = acc_start_dis < acc_end_dis

        if is_debug:
            ## debug
            cmap = cm.get_cmap('rainbow')
            color = cmap(0)
            save_samples_color(save_dir + 'pts_init_after_sptracing.ply', after_intersect_points.cpu(), torch.ones_like(after_intersect_points).detach().cpu().numpy()*color[:3] * 255)


        if torch.any(torch.isnan(acc_start_dis)):
            print('sphere tracing has nan')
        t3 = time.time()
        if is_debug:
            ## debug
            cmap = cm.get_cmap('rainbow')
            color = cmap(0.2)
            save_samples_color(save_dir + 'pts_after_sp_tracing.ply', curr_start_points.cpu(), torch.ones_like(curr_start_points).detach().cpu().numpy()*color[:3] * 255)
            save_samples_color(save_dir + 'hit_pts.ply', curr_start_points[network_object_mask].cpu(), torch.ones_like(curr_start_points[network_object_mask]).detach().cpu().numpy()*color[:3] * 255)


        # pdb.set_trace()
        # The non convergent rays should be handled by the sampler
        sampler_work_mask = unfinished_mask_start # [BN]
        sampler_convergent_mask = torch.zeros_like(sampler_work_mask).bool().to(self.device)

        if sampler_work_mask.sum() > 0:
            sampler_min_max = torch.stack((acc_start_dis.clone(), acc_end_dis.clone()),dim=-1)

            # sampler_min_max = torch.zeros((N_rays, 2)).to(self.device)
            # sampler_min_max[sampler_work_mask, 0] = acc_start_dis[sampler_work_mask]
            # sampler_min_max[sampler_work_mask, 1] = acc_end_dis[sampler_work_mask]

            sampler_pts, sampler_convergent_mask, sampler_dists = self.ray_sampler(sdf,
                                                                                ray_os,
                                                                                ray_ds,
                                                                                object_mask_gt.view(-1),
                                                                                sampler_min_max,
                                                                                sampler_work_mask
                                                                                )
            
            curr_start_points[sampler_work_mask] = sampler_pts[sampler_work_mask]
            acc_start_dis[sampler_work_mask] = sampler_dists[sampler_work_mask]
            network_object_mask[sampler_work_mask] = sampler_convergent_mask[sampler_work_mask]
        t4 = time.time()
        if is_debug:
            ## debug
            cmap = cm.get_cmap('rainbow')
            color = cmap(0.4)
            save_samples_color(save_dir + 'pts_after_sampler.ply', curr_start_points.cpu(), torch.ones_like(curr_start_points).detach().cpu().numpy()*color[:3] * 255)
            save_samples_color(save_dir + 'pts_after_sampler_changed.ply', curr_start_points[sampler_work_mask].cpu(), torch.ones_like(curr_start_points[sampler_work_mask]).detach().cpu().numpy()*color[:3] * 255)


        if self.verbose:
            print('sphere tracing time: ', t3-t2)
            print('ray sampler time: ', t4-t3)

            print('----------------------------------------------------------------')
            print('RayTracing: object = {0}/{1}, secant on {2}/{3}.'
                .format(network_object_mask.sum(), len(network_object_mask), sampler_convergent_mask.sum(), sampler_work_mask.sum()))
            print('----------------------------------------------------------------')
        
        if not self.training:
            return curr_start_points, \
                   network_object_mask, \
                   acc_start_dis
        '''
        network_object_mask: traced points with dist_start < dist_end, means the mask of the object predicted by the network 
        this_view_object_mask: ground truth mask of this view of object
        sampler_work_mask: unfinished in sphere tracing step
        
        in_mask: in the gt mask but not converged rays
        out_mask: out of gt mask but not converged rays

        '''
        in_mask = ~network_object_mask & object_mask_gt 
        out_mask = ~object_mask_gt
        mask_left_out = (in_mask | out_mask) & ~mask_init & ~sampler_work_mask
        mask_left_in = (in_mask | out_mask) & mask_init & ~sampler_work_mask

        if mask_left_out.sum() > 0:  # project the origin to the not intersect points on the sphere
            print('mask_left_out_sum = %d' % mask_left_out.sum())
            cam_left_out = ray_os[mask_left_out]
            rays_left_out = ray_ds[mask_left_out]
            acc_start_dis[mask_left_out] = -torch.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze()
            curr_start_points[mask_left_out] = cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out
        
        if is_debug:
            ## debug
            cmap = cm.get_cmap('rainbow')
            color = cmap(0.6)
            save_samples_color(save_dir + 'pts_after_handle_left_out.ply', curr_start_points.cpu(), torch.ones_like(curr_start_points).detach().cpu().numpy()*color[:3] * 255)

        


        if mask_left_in.sum() > 0:
            min_dis[network_object_mask & out_mask] = acc_start_dis[network_object_mask & out_mask]
            
            min_mask_points, min_mask_dist = self.minimal_sdf_points(N_rays, sdf, ray_os, ray_ds, mask_left_in, min_dis, max_dis)
            
            curr_start_points[mask_left_in] = min_mask_points
            acc_start_dis[mask_left_in] = min_mask_dist

        if is_debug:
            ## debug
            cmap = cm.get_cmap('rainbow')
            color = cmap(0.8)
            if mask_left_in.sum()>0:
                save_samples_color(save_dir + 'pts_after_handle_left_in_mask.ply', min_mask_points.cpu(), torch.ones_like(min_mask_points).detach().cpu().numpy()*color[:3] * 255)
            save_samples_color(save_dir + 'pts_after_handle_left_in.ply', curr_start_points.cpu(), torch.ones_like(curr_start_points).detach().cpu().numpy()*color[:3] * 255)


        return curr_start_points, \
               network_object_mask, \
               acc_start_dis


    def minimal_sdf_points(self, num_pixels, sdf, ray_os, ray_ds, mask, min_dis, max_dis):
        ''' Find points with minimal SDF value on rays for P_out pixels '''
        num_pixels, _ = ray_ds.shape
        batch_size=1
        n_mask_points = mask.sum()

        n = self.n_steps
        # steps = torch.linspace(0.0, 1.0,n).to(self.device)
        steps = torch.empty(n).uniform_(0.0, 1.0).to(self.device)
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        mask_points = ray_os[mask]
        mask_rays = ray_ds[mask]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []

        for pnts in torch.split(points, self.chunk_size, dim=0):
            sdf_chunk = sdf(pnts.view(batch_size,-1,3).transpose(1,2)).flatten()
            mask_sdf_all.append(sdf_chunk)

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]

        return min_mask_points, min_mask_dist