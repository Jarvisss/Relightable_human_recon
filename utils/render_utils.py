from turtle import back
import matplotlib
from pip import main
import torch
import torch.nn.functional as F
from .sdf_utils import *
from skimage import measure
from PIL import Image
from utils.vis_utils import vis_sdf_x_plane,vis_sdf_y_plane,vis_sdf_z_plane
from utils.uv_utils import grid_sample
# from utils.grid_sample_gradfix import grid_sample
from utils.common_utils import img_l2g, shading_l2g
from utils.camera import KRT_from_P, quat_to_rot
import cv2
import pdb
import random


view_colors = torch.rand(50,3)

def get_random_color():
    r = random.random()
    g = random.random()
    b = random.random()
    return [r,g,b]

def load_K_Rt_from_P(P):
    print(P)
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    print(K)
    print(R)
    print(t)
    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def lift(x, y, z, intrinsics, z_dir=-1):
    '''
    
    '''
    # parse intrinsics
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z)), dim=-1)

def get_camera_params(uv, cam2worlds, intrinsics):
    '''
    :param UV [BK, N_ray, 2]
    :param cam2worlds [BK, 4, 4]
    :param intrinsics [BK, 4, 4]
    Return:
    @ ray_dirs: [B, N_ray, 3]
    @ cam_locs: [B, 3]
    '''
    cam_locs = cam2worlds[:, :3, 3]
    p = cam2worlds
    # cam_loc = camera_to_world[:, :3, 3]
    # p = camera_to_world
    print('c2w matrix',p)

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).to(uv.device)
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1) * -1

    # a virtual image plane in camera space at z = 1
    # [B, N, 4]
    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    # [B, N, 4] to [B, 4, N]
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)
    print('lifted points', pixel_points_cam)
    print('lifted points 0: ', pixel_points_cam[0,:,0])
    # camera space to world space [B,4,4] @ [B,4,N] -> [B,4,N] -> [B,3,N]
    world_coords = torch.bmm(p, pixel_points_cam)[:, :3, :]
    print('lifted to world', world_coords)
    ray_dirs = world_coords.permute(0,2,1) - cam_locs[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=-1)
    print(ray_dirs)
    print(cam_locs)

    return ray_dirs, cam_locs




def get_uvs(opt, width, height):
    '''
    :param width ,int
    :param height ,int
    :param n_imgs ,int
    :return uvs ,[K, HW, 2]
    '''
    if opt.use_CV_perspective:
        uv = np.mgrid[0 : height, 0 : width].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv.reshape(1, 2, -1), axis=1).copy()).float().permute(0,2,1) 
    else:
        half_HW = np.array([height//2,width//2])[:,None,None]
        uv = (np.mgrid[0 : height, 0 : width].astype(np.int32) + 0.5 - half_HW) / (half_HW)
        uv = torch.from_numpy(np.flip(uv.reshape(1, 2, -1), axis=1).copy()).float().permute(0,2,1) # flip to get correct x,y instead of y,x  [1, HW, 2]
        uv = -uv
    return uv


def get_camera_params_in_model_space(uv, pose, intrinsics, neg_z=False):
    '''
    Params
    @ UV [BK, N_ray, 2] 
    @ pose [BK, 4, 4] or [BK, 7]
    @ intrinsics [BK, 4, 4] 
    
    Return
    @ ray_dirs [BK, N_ray, 3]
    @ cam_loc [BK, 3]
    @ light_dir [BK, 1, 3]
    '''
    N_test = pose.shape[0]
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_locs_in_model_space = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_locs_in_model_space
    else:
        cam_locs_in_model_space = pose[:, :3, 3]
        p = pose
    
    # pdb.set_trace()
    # cam_loc = camera_to_world[:, :3, 3]
    # p = camera_to_world

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).to(uv.device)
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)
    light_d = torch.Tensor([[0,0,1,1]])
    
    if neg_z:
        z_cam = -z_cam
        light_d = torch.Tensor([[0,0,-1,1]])
    light_d = light_d.expand(p.shape[0], -1).unsqueeze(-1).to(uv.device)
    # a virtual image plane in camera space at z = 1
    # [B, N, 4]
    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    # [B, N, 4] to [B, 4, N]
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)
    # camera space to world space [B,4,4] @ [B,4,N] -> [B,4,N] -> [B,3,N]
    model_space_coords = torch.bmm(p, pixel_points_cam)[:, :3, :]
    model_space_lightd = torch.bmm(p, light_d)[:, :3, :]
    ray_dirs = model_space_coords.permute(0,2,1) - cam_locs_in_model_space[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=-1)

    light_dir = model_space_lightd.permute(0,2,1) - cam_locs_in_model_space[:, None, :]

    return ray_dirs, cam_locs_in_model_space, light_dir



def torch_index(feat, uv, padding='zeros', mode='nearest'):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    # samples = F.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    samples = F.grid_sample(feat, uv,mode=mode, padding_mode=padding, align_corners=True)
    return samples[:, :, :, 0]  # [B, C, N]

# def index(feat, uv):
#     '''
#     :param feat: [B, C, H, W] image features
#     :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
#     :return: [B, C, N] image features at the uv coordinates
#     '''
#     uv = uv.transpose(1, 2)  # [B, N, 2]
#     uv = uv.unsqueeze(2)  # [B, N, 1, 2]
#     # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
#     # for old versions, simply remove the aligned_corners argument.
#     # samples = F.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
#     samples = grid_sample(feat, uv)
#     return samples[:, :, :, 0]  # [B, C, N]


def get_sphere_intersection(cam_loc, ray_directions, r = 1.0, min=0.001):
    '''
    Input: B x 3 ; B x n_rays x 3
    Output: B * n_rays x 2 (close t and far t) ; B * n_rays
    '''
    # import pdb
    # pdb.set_trace()
    device = cam_loc.device
    n_imgs, n_pix, _ = ray_directions.shape

    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze(-1)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)  ## if point is inside the sphere, then under_sqrt is greater than 0

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0

    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).to(device).float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).to(device).float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(min)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix, 1)


    return sphere_intersections, mask_intersect

# def get_ray_visual_hull_intersection(num_views, cam_loc, ray_directions, calibs, masks, radius, n_sample_per_ray=100, dilation_size=7):
#     '''
#     :param numviews
#     :param cam_loc [B, 3]
#     :param ray_directions [B,N,3] normalized
#     :param calibs [B,4,4] calibs without model matrix, transfer world to image
#     :param masks [B,1,H,W] image masks
    
#     :return front_back_intersection[B,N,2]: the t_start and t_end for each ray's intersect with visual hull from front and back
#     :return mask_intersect [B,N,1]: if ray has intersection with visual hull
#     '''
#     batch_size, n_samples, _ = ray_directions.shape
#     batch_size, _, H, W = masks.shape
#     device = cam_loc.device
#     front_unintersect_mask = torch.ones((batch_size, n_samples, 1), dtype=torch.bool).to(device)
#     back_unintersect_mask = torch.ones((batch_size, n_samples, 1), dtype=torch.bool).to(device)
#     front_first_intersection = torch.zeros((batch_size, n_samples, 1)).to(device)
#     back_first_intersection = torch.zeros((batch_size, n_samples, 1)).to(device)

#     # [B,N,2], [B,N]
#     t_start_end, ray_intersect = get_sphere_intersection(cam_loc, ray_directions, r=radius)
#     if ray_intersect.sum() ==0:
#         return None
#     min_t_start = torch.min(t_start_end[ray_intersect][:,0]) # single value
#     max_t_end = torch.max(t_start_end[ray_intersect][:,1]) # single value
#     t_samples = torch.linspace(min_t_start, max_t_end, n_sample_per_ray)
#     # ray_marching
#     for i in range(len(t_samples)):
#         t = t_samples[i]
#         # [B,N,3], marched pts at step t
#         query_pts_t = cam_loc[:, None, :] + t * ray_directions
#         pts_in_vhull = in_visual_hull(num_views, query_pts_t.permute(0,2,1), calibs, masks, dilation_size).permute(0,2,1) # [B, N, 1]
#         front_first_intersection[front_unintersect_mask & pts_in_vhull] = t
#         front_unintersect_mask = front_unintersect_mask & ~pts_in_vhull # update the unintersected mask
    
#     for i in range(len(t_samples)-1, -1, -1):
#         t = t_samples[i]
#         # [B,N,3], marched pts at step t
#         query_pts_t = cam_loc[:, None, :] + t * ray_directions
                
#         pts_in_vhull = in_visual_hull(num_views, query_pts_t.permute(0,2,1), calibs, masks, dilation_size).permute(0,2,1) # [B, N, 1]
#         back_first_intersection[back_unintersect_mask & pts_in_vhull] = t
#         back_unintersect_mask = back_unintersect_mask & ~pts_in_vhull # update the unintersected mask

#     assert( front_unintersect_mask.equal(back_unintersect_mask) )

#     return torch.cat((front_first_intersection, back_first_intersection), dim=-1), ~front_unintersect_mask


def sample_rays(uvs, c2ws, intris, calibs, masks_dilated):
    '''
    We have K views and K masks, we want to get the intersections of each view's rays with the visual hull formed by the masks

    Params:
    uvs [BK, N_rays, 2] input uvs
    c2ws [BK,4,4] inverse extrinsic
    intris [BK,4,4] intrinsic
    calibs [BK,4,4] calibs without model matrix, transfer world to image
    masks [BK,1,H,W] image masks
    
    Return:
    ray_os [BK,N,2]: the t_start and t_end for each ray's intersect with visual hull from front and back
    ray_ds [BK,N,1]: if ray has intersection with visual hull
    ray_cs
    ray_ms 
    '''




def get_ray_visual_hull_intersection(
    cam_locs, ray_ds, ## views 
    calibs, masks_dilated, radius, projection_method, ## visual hull
    n_sample_per_ray=100, debug_vis_path='', sphere_intersection=False, depth_min=0.1, depth_max=3, min_observations=5, max_unobservations=0, ori_aspect=1):
    '''
    We have K views and K masks, we want to get the intersections of each view's rays with the visual hull formed by the masks

    :param cam_locs [BK, 3] all ray origins
    :param ray_ds [BK, N_rays, 3] all ray directions

    :param calibs [BK, 4, 4] calibs without model matrix, transfer world to image
    :param masks [BK,1,H,W] image masks
    
    :return front_back_intersection[BK,N,2]: the t_start and t_end for each ray's intersect with visual hull from front and back
    :return mask_intersect [BK,N,1]: if ray has intersection with visual hull
    '''
    # uvs = uvs.cpu()
    # c2ws = c2ws.cpu()
    # intris = intris.cpu()
    # calibs = calibs.cpu()
    # masks_dilated = masks_dilated.cpu()
    ray_directions, cam_loc = ray_ds, cam_locs
    _,_,H,W=masks_dilated.shape
    K_all = calibs.shape[0]
    K_observe = cam_loc.shape[0]
    batch_size, n_samples, _ = ray_directions.shape
    assert(K_all==batch_size)
    device = cam_loc.device
    front_unintersect_mask = torch.ones((batch_size, n_samples, 1), dtype=torch.bool).to(device)
    back_unintersect_mask = torch.ones((batch_size, n_samples, 1), dtype=torch.bool).to(device)
    t_start_end, ray_intersect = get_sphere_intersection(cam_loc, ray_directions, r=radius)
    if sphere_intersection:
        return t_start_end, ray_intersect
    front_first_intersection = t_start_end[:,:,0:1].clone()
    back_first_intersection = t_start_end[:,:,1:2].clone()
    # front_unintersect_mask = front_unintersect_mask & (~ray_intersect)
    # back_unintersect_mask = back_unintersect_mask & (~ray_intersect)
    

    query_pts_t_start = cam_loc[:, None, :] + t_start_end[:,:,0:1] * ray_directions
    query_pts_t_end = cam_loc[:, None, :] + t_start_end[:,:,1:2] * ray_directions
    
    # for k in range(K_all):
    #     color = torch.ones_like(query_pts_t_start[0]).cpu()
    #     r,g,b = get_random_color()
    #     color[:,0] *= r
    #     color[:,1] *= g
    #     color[:,2] *= b
    #     save_samples_color(debug_vis_path + '/view_%d_t_start.ply' % (k), query_pts_t_start[k].cpu(), color * 255)
    #     save_samples_color(debug_vis_path + '/view_%d_t_end.ply' % (k), query_pts_t_end[k].cpu(), color * 255)
    
    # [B,N,2], [B,N]
    if ray_intersect.sum() ==0:
        return None
    # ray_intersect  = ray_intersect[:,:,0]
    # pdb.set_trace()
    min_t_start = torch.min(t_start_end[ray_intersect.squeeze(-1)][:,0]) # single value
    max_t_end = torch.max(t_start_end[ray_intersect.squeeze(-1)][:,1]) # single value
    t_samples = torch.linspace(min_t_start, max_t_end, n_sample_per_ray)
    # ray_marching
    for i in tqdm(range(0, len(t_samples))):
        t = t_samples[i]
        # [B,N,3], marched pts at step t
        query_pts_t = cam_loc[:, None, :] + t * ray_directions # [BN3]

        pts_in_vhull_observe = []
        for k in range(K_observe):
            points_stack = query_pts_t[k:k+1].expand(K_all, -1, -1)
            xyz = projection_method(points_stack.permute(0,2,1), calibs, size=(W,H)) # this projection is differentiable
            # pts_in_vhull = in_visual_hull(xyz, masks_dilated, K_all, dilation_kernel_size=0, min_observations=K_all//3*2, max_unobservations=max_unobservations, ori_aspect=ori_aspect).permute(0,2,1) # [B, N, 1]
            pts_in_vhull = in_visual_hull(xyz, masks_dilated, K_all, dilation_kernel_size=0, min_observations=K_all, max_unobservations=max_unobservations, ori_aspect=ori_aspect).permute(0,2,1) # [B, N, 1]
            color = torch.ones_like(query_pts_t[0]) * view_colors[k].unsqueeze(0).to(device)
            color[:,0] = pts_in_vhull[0,:,0]
            
            in_hull_pts = query_pts_t[k][pts_in_vhull[0,:,0]]
            color = color[pts_in_vhull[0,:,0]]
            # if in_hull_pts.shape[0]>0:
                # save_samples_color(debug_vis_path + '/debug/view_%d/marching_step%d_t_%.3f.ply' % (k, i, t), in_hull_pts.cpu(), color.cpu() * 255)
            pts_in_vhull_observe.append(pts_in_vhull)
        
        pts_in_vhull = torch.cat(pts_in_vhull_observe, dim=0)
        front_first_intersection[front_unintersect_mask & pts_in_vhull] =  t
        front_unintersect_mask = front_unintersect_mask & ~pts_in_vhull # update the unintersected mask
    
    for i in tqdm(range(len(t_samples)-1, -1, -1)):
        t = t_samples[i]
        # [B,N,3], marched pts at step t
        query_pts_t = cam_loc[:, None, :] + t * ray_directions
        pts_in_vhull_observe = []
        for k in range(K_observe):
            points_stack = query_pts_t[k:k+1].expand(K_all, -1, -1)
            xyz = projection_method(points_stack.permute(0,2,1), calibs, size=(W,H)) # this projection is differentiable
            # pts_in_vhull = in_visual_hull(xyz, masks_dilated, K_all, dilation_kernel_size=0, min_observations=K_all//3*2, max_unobservations=max_unobservations, ori_aspect=ori_aspect).permute(0,2,1) # [B, N, 1]
            pts_in_vhull = in_visual_hull(xyz, masks_dilated, K_all, dilation_kernel_size=0, min_observations=K_all, max_unobservations=max_unobservations, ori_aspect=ori_aspect).permute(0,2,1) # [B, N, 1]
            color = torch.ones_like(query_pts_t[0]) * view_colors[k].unsqueeze(0).to(device)
            color[:,0] = pts_in_vhull[0,:,0]
            
            in_hull_pts = query_pts_t[k][pts_in_vhull[0,:,0]]
            color = color[pts_in_vhull[0,:,0]]
            # if in_hull_pts.shape[0]>0:
            #     save_samples_color(debug_vis_path + '/view_%d/marching_step%d_t_%.3f.ply' % (k, i, t), in_hull_pts.cpu(), color.cpu() * 255)
            pts_in_vhull_observe.append(pts_in_vhull)
        
        pts_in_vhull = torch.cat(pts_in_vhull_observe, dim=0)
        back_first_intersection[back_unintersect_mask & pts_in_vhull] = t
        back_unintersect_mask = back_unintersect_mask & ~pts_in_vhull # update the unintersected mask

    if not front_unintersect_mask.equal(back_unintersect_mask):
        pdb.set_trace()
    assert( front_unintersect_mask.equal(back_unintersect_mask) )

    return torch.cat((front_first_intersection, back_first_intersection), dim=-1), ~front_unintersect_mask
    # return torch.cat((front_first_intersection, back_first_intersection), dim=-1), masks_dilated.view(K_all, 1, -1).permute(0,2,1).contiguous() > 0.5


def not_in_visul_hull(projected_points, masks, num_views, dilation_kernel_size=3, min_observations=0, max_unobservations=0, ori_aspect=1):
    xy = projected_points[:, :2, :] # [BK, 2, N]
    K = num_views
    N = xy.shape[-1]
    in_img = (xy[:, 0] >= -ori_aspect) & (xy[:, 0] <= ori_aspect) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
    in_img.unsqueeze_(1) # [BK, 1, N]

    if dilation_kernel_size>0:
        dilation_kernel = torch.ones((1,1,dilation_kernel_size,dilation_kernel_size)).to(masks.device)
        masks_dilated = torch.clamp(F.conv2d(masks, dilation_kernel, padding=dilation_kernel_size//2), 0, 1)
        mask_val = torch_index(masks_dilated, xy, 'zeros', 'nearest')
    else:
        mask_val = torch_index(masks, xy, 'zeros', 'nearest')

    mask_val = torch_index(masks, xy, 'zeros', 'nearest')
    out_hull = in_img & (mask_val<0.5) # in img and not in mask
    out_hull = out_hull.view(-1, K, 1 , N)
    out_hull = out_hull.sum(dim=1) ## any view in img and not in mask, then, not in visual hull

    return out_hull > 0

def in_visual_hull(projected_points, masks, num_views,dilation_kernel_size=3, min_observations=0, max_unobservations=0, ori_aspect=1):
    xy = projected_points[:, :2, :] # [BK, 2, N]
    K = num_views
    N = xy.shape[-1]
    _,_,H,W = masks.shape

    if dilation_kernel_size>0:
        dilation_kernel = torch.ones((1,1,dilation_kernel_size,dilation_kernel_size)).to(masks.device)
        masks_dilated = torch.clamp(F.conv2d(masks, dilation_kernel, padding=dilation_kernel_size//2), 0, 1)
        mask_val = torch_index(masks_dilated, xy, 'zeros', 'nearest')
    else:
        mask_val = torch_index(masks, xy, 'zeros', 'nearest')

    in_img = (xy[:, 0] >= -ori_aspect) & (xy[:, 0] < ori_aspect) & (xy[:, 1] >= -1.0) & (xy[:, 1] < 1.0)
    in_img.unsqueeze_(1) # [BK, 1, N]

    all_not_in_img = in_img.view(-1, K, 1 , N).sum(dim=1) == 0
    
    in_mask = mask_val > 0.5
    in_img_not_in_mask = in_img & (~in_mask)
    in_img_not_in_mask_all = in_img_not_in_mask.view(-1, K, 1 , N).sum(1) > 0

    must_be_outside = all_not_in_img | in_img_not_in_mask_all
    inside = ~must_be_outside
    return inside

# def in_visual_hull(projected_points, masks, num_views, dilation_kernel_size=3, min_observations=0, max_unobservations=0, ori_aspect=1):
#     '''
#     Given N 3D points, K calibs and K image masks,
#     output in/out visual hull 
#     NOTE: out of image should not be taken into consider
#     :param projected_points: [BK, 3, N] 
#     :param masks: [BK, 1, H, W]
#     :param min_observations: int (in img, e.g. should in 2/3 img)
#     :param max_unobservations: int (not in mask, but in img, e.g. should in 2/3 img)

#     :return: [B, 1, N] if points in visual hull
#     '''
#     xy = projected_points[:, :2, :] # [BK, 2, N]
#     K = num_views
#     N = xy.shape[-1]
#     _,_,H,W = masks.shape

#     if dilation_kernel_size>0:
#         dilation_kernel = torch.ones((1,1,dilation_kernel_size,dilation_kernel_size)).to(masks.device)
#         masks_dilated = torch.clamp(F.conv2d(masks, dilation_kernel, padding=dilation_kernel_size//2), 0, 1)
#         mask_val = torch_index(masks_dilated, xy, 'zeros', 'nearest')
#     else:
#         mask_val = torch_index(masks, xy, 'zeros', 'nearest')


#     in_img = (xy[:, 0] >= -ori_aspect) & (xy[:, 0] <= ori_aspect) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
#     in_img.unsqueeze_(1) # [BK, 1, N]
#     in_hull = (mask_val>0.5)
    
#     '''
#     Here we have all possible points, with inside image masks, and outside image range 
#     If a point is inside one image but not in mask, it is not in geometry
#     This step we get all points that:
#     1. All in mask
#     or 
#     2. All either cannot be viewed in img or in mask   
#     '''
#     in_hull = (~ in_img) | (mask_val>0.5)
#     # in_hull = (mask_val>0.5)

#     in_hull = in_hull.view(-1, K, 1 , N)
#     in_hull = in_hull.sum(dim=1) 

    
#     '''if a point is in all img, then out of mask ENSURE no geometry exists'''
#     in_vis_hull = in_hull > (K-max_unobservations-0.5)  # allow for samples at boundary

#     '''  What if the point is outside some of the images, but exists in others and in mask?  '''
#     all_not_in_img = in_img.view(-1, K, 1 , N).sum(dim=1) == 0
#     at_least_in_k_imgs = in_img.view(-1, K, 1 , N).sum(dim=1) >= min_observations

#     '''
#     This step we remove points those are only viewed in less than 3 images
#     '''
#     # return in_vis_hull & ( ~all_not_in_img)
#     return in_vis_hull & at_least_in_k_imgs



if __name__ == '__main__':

    from options.options import get_options
    from dataset.Thuman2_pifu_dataset_sdf import make_dataset
    from utils.common_utils import set_random_seed
    opt = get_options()

    opt.path_to_dataset = '/home/lujiawei/workspace/dataset/thuman2_rescaled_prt_512_single_light_no_lighting_change_w_flash_no_env_persp'
    opt.path_to_obj = '/home/lujiawei/workspace/dataset/thuman2_rescaled'
    opt.path_to_sample_pts = '/home/lujiawei/workspace/dataset/SAMPLE_PT_SCALE'
    opt.num_views = 3
    opt.use_perspective=True
    opt.offline_sample = True
    opt.random_scale = True
    opt.random_trans = True
    opt.random_flip = True
    opt.random_aug_offset = 0
    test_dataset = make_dataset(opt, 'Train')
    test_dataset.is_train = True
    set_random_seed(opt.seed)

    device = torch.device("cuda:0")
    cpu = torch.device("cpu")

    id = 0
    data11 = test_dataset[id]
    
    calib = data11['calib']
    mask = data11['mask']
    name = data11['name']
    sid = data11['sid']
    yid = data11['yid']
    view_ids = data11['view_ids']
    samples = data11['samples']
    surface_samples = data11['surface_samples']
    labels = data11['labels']
    extrinsic = data11['extrinsic']
    intrinsic = data11['intrinsic']
    cam_extrinsics = data11['cam_extrinsics']

    cam_centers = data11['cam_centers']
    cam_directions = data11['cam_directions']
    img = data11['img']

    print(name)
    uv = (np.mgrid[0:512, 0:512].astype(np.int32) - 256 + 0.5) / 256
    # uv = np.mgrid[0:512, 0:512].astype(np.int32) 
    uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
    uv = uv.view(2, -1).transpose(1, 0).unsqueeze(0).repeat(opt.num_views, 1, 1)
    uv[1,:] = -uv[1,:]

    with_out_normalize = torch.bmm(intrinsic, cam_extrinsics)
    cam2world =  torch.linalg.inv(cam_extrinsics)

    ray_dirs, cam_locs = get_camera_params(uv.to(device), cam2world.to(device), intrinsics=intrinsic.to(device))
    front_back_intersections, intersect_mask = get_ray_visual_hull_intersection(
        opt.num_views, 
        cam_locs, 
        ray_dirs, 
        with_out_normalize.to(device), 
        mask.to(device), 
        radius=100,
        n_sample_per_ray=100, 
        dilation_size=7 
        )
    pass


