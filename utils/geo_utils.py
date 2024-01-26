import matplotlib
from plyfile import PlyData, PlyElement
import torch
import torch.nn.functional as F
from trimesh import Trimesh
from .sdf_utils import *
from scipy.spatial.ckdtree import cKDTree
from skimage import measure
from PIL import Image
from utils.vis_utils import vis_sdf_x_plane,vis_sdf_y_plane,vis_sdf_z_plane
from utils.uv_utils import grid_sample ## written by alb
from utils.render_utils import get_uvs, get_camera_params_in_model_space
from utils.camera import get_calib_extri_from_pose
# from utils.render_utils import in_visual_hull
# from utils.cuda_gridsample import grid_sample_2d ## written  by nvidia team
# import cu_grid_sample as cu
# from utils.grid_sample_gradfix import grid_sample
from utils.common_utils import img_l2g, shading_l2g
from utils.export_materials import export_materials

from matplotlib import cm
import matplotlib.pyplot as plt
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.ops.mesh import check_sign
# from model.UNet_unified import UNet_unified
import pdb
import os


name = 'jet' # error color map   

def index(feat, uv, mode='bilinear', padding_mode='border', size=None):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]

    if size!=None:
        uv = (uv - size/2) /(size/2)
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    # samples = F.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    if mode=='bilinear' and padding_mode=='border':
        samples = grid_sample(feat, uv)
        # samples = grid_sample_2d(feat, uv, padding_mode=padding_mode)
        # samples = grid_sample_2d(feat, uv, padding_mode=padding_mode, align_corners=True)
    else:
        samples = grid_sample(feat, uv)
        # samples = F.grid_sample(feat, uv, mode=mode,padding_mode=padding_mode)
        # samples = grid_sample_2d(feat, uv, padding_mode=padding_mode, align_corners=True)
    # samples = cu.grid_sample_2d(feat, uv, 'zeros', align_corners=True)
    return samples[:, :, :, 0]  # [B, C, N]

def orthogonal(points, calibrations, transforms=None, size=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    # Rot * points + Trans
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N] 
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
        
    return pts

def unproject_orthogonal(pts, calibrations):
    '''
    Compute the orthogonal unprojections of 3D points from the image plane to the world space by given projection matrix
    :param xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :return: points: [B, 3, N] Tensor of 3D points
    '''

    rot = calibrations[:, :3, :3] # [B,3,3]
    trans = calibrations[:, :3, 3:4] #[B,3,1]

    rot_inv = torch.linalg.inv(rot)
    trans_inv = -trans
    # points - 
    unproj_pts = torch.bmm(rot_inv, pts+trans_inv)
    # unproj_pts = torch.baddbmm(trans_inv, rot_inv, pts)
    return unproj_pts

# def orthogonal_unprojection(points_xy, calibrations, transforms=None):
#     '''
#     Compute the orthogonal projections of 3D points into the image plane by given projection matrix
#     :param points: [B, 3, N] Tensor of 3D points
#     :param calibrations: [B, 4, 4] Tensor of projection matrix
#     :param transforms: [B, 2, 3] Tensor of image transform matrix
#     :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
#     '''
#     rot = calibrations[:, :3, :3]
#     trans = calibrations[:, :3, 3:4]
#     # Rot * points + Trans
#     pts = torch.baddbmm(trans, rot, points)  # [B, 3, N] 
#     if transforms is not None:
#         scale = transforms[:2, :2]
#         shift = transforms[:2, 2:3]
#         pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
#     return pts



# def perspective(points, calibrations, transforms=None):
#     '''
#     Compute the perspective projections of 3D points into the image plane by given projection matrix
#     :param points: [Bx3xN] Tensor of 3D points
#     :param calibrations: [Bx4x4] Tensor of projection matrix
#     :param transforms: [Bx2x3] Tensor of image transform matrix
#     :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
#     '''
#     rot = calibrations[:, :3, :3]
#     trans = calibrations[:, :3, 3:4]
#     homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
#     xy = homo[:, :2, :] / homo[:, 2:3, :]
#     if transforms is not None:
#         scale = transforms[:2, :2]
#         shift = transforms[:2, 2:3]
#         xy = torch.baddbmm(shift, scale, xy)

#     xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
#     print('xmax:', xyz[:,0,:].max())
#     print('ymax:', xyz[:,1,:].max())
#     print('zmax:', xyz[:,2,:].max())

#     print('xmin:', xyz[:,0,:].min())
#     print('ymin:', xyz[:,1,:].min())
#     print('zmin:', xyz[:,2,:].min())
#     return xyz



def perspective_opencv(points, calib, transforms=None, size=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [BKx3xN] Tensor of 3D points
    :param calibrations: [BKx3x4] Tensor of projection matrix(K @ [R|t])
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :param size: (W,H)
    :return: xyz: [Bx3x(N_cam*N_samp)] Tensor of xy coordinates in the image plane
    '''

    w = torch.ones_like(points)[:,:1,:]
    if torch.any(torch.isnan(points)):
        print('pt has nan')
    points_w  = torch.cat((points,w),dim=1)
    homo = torch.bmm(calib, points_w) # [BK,3,N]
    if torch.any(torch.isnan(calib)):
        print('calib has nan')
    if torch.any(torch.isnan(points_w)):
        print('pt_w has nan')
    if torch.any(torch.isnan(homo)):
        print('homo has nan')
        print(torch.where(torch.isnan(homo)))
        print('ptw min', points_w.min())
        print('ptw max', points_w.max())

        print(homo)
        print(calib)
        print(points_w)
    
    xy = homo[:,:2,:] / (homo[:,2:3,:] + 1e-8) ## []
    # if torch.any(torch.isnan(xyz)):
    #     print('proj_points_hasnan')
    if size is not None:
        if type(size) == int:
            xy = (xy - size/2) / (size/2)
        else:
            W,H = size
            xy[:,0:1,:] = (xy[:,0:1,:] - W/2) / (W/2)
            xy[:,1:2,:] = (xy[:,1:2,:] - H/2) / (H/2)
    return torch.cat((xy, homo[:,2:,:]), dim=1)


def perspective(points, calibrations, transforms=None, size=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [BKx3xN] Tensor of 3D points
    :param calibrations: [BKx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xyz: [Bx3x(N_cam*N_samp)] Tensor of xy coordinates in the image plane
    '''
    w = torch.ones_like(points)[:,:1,:]
    if torch.any(torch.isnan(points)):
        print('pt has nan')
    points_w  = torch.cat((points,w),dim=1)
    homo = torch.bmm(calibrations, points_w)
    if torch.any(torch.isnan(calibrations)):
        print('calib has nan')
    if torch.any(torch.isnan(points_w)):
        print('pt_w has nan')
    if torch.any(torch.isnan(homo)):
        print('homo has nan')
        print(torch.where(torch.isnan(homo)))
        print('ptw min', points_w.min())
        print('ptw max', points_w.max())

        print(homo)
        print(calibrations)
        print(points_w)
    
    xyz = homo[:,:3,:] / (homo[:,3:,:] + 1e-8)
    if torch.any(torch.isnan(xyz)):
        print('proj_points_hasnan')

    # print('xmax:', xyz[:,0,:].max())
    # print('ymax:', xyz[:,1,:].max())
    # print('zmax:', xyz[:,2,:].max())

    # print('xmin:', xyz[:,0,:].min())
    # print('ymin:', xyz[:,1,:].min())
    # print('zmin:', xyz[:,2,:].min())
    # rot = calibrations[:, :3, :3]
    # trans = calibrations[:, :3, 3:4]
    # homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    # xy = homo[:, :2, :] / homo[:, 2:3, :]
    # if transforms is not None:
    #     scale = transforms[:2, :2]
    #     shift = transforms[:2, 2:3]
    #     xy = torch.baddbmm(shift, scale, xy)

    # xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz

def save_mesh_ply(filename, verts, faces , colors=None):
    print(verts.shape)
    vertex = np.core.records.fromarrays(verts.transpose(), names='x, y, z', formats='f4, f4, f4')
    n = len(vertex)
    desc = vertex.dtype.descr

    if colors is not None:
        vertex_color = np.core.records.fromarrays(colors.transpose() * 255, names='red, green, blue',
                                                  formats='u1, u1, u1')
        assert len(vertex_color) == n
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(n, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    faces_building = []
    for i in range(0, faces.shape[0]):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = PlyElement.describe(vertex_all, "vertex")
    el_faces = PlyElement.describe(faces_tuple, "face")

    ply_data = PlyData([el_verts, el_faces])
    if not os.path.exists(os.path.dirname(filename)):
       os.makedirs(os.path.dirname(filename))
    ply_data.write(filename)

def save_obj_mesh_with_color(mesh_path, verts, faces,normals, colors):
    meshexport = Trimesh(verts, faces, normals, vertex_colors=colors)
    meshexport.export(mesh_path, 'obj')

def get_rendered(shading_path, albedo_path, out_render_path):
    import trimesh
    shading_mesh = trimesh.load(shading_path, force='mesh')
    albedo_mesh = trimesh.load(albedo_path, force='mesh')
    shadings = shading_mesh.visual.vertex_colors
    print(shadings.max())
    albedos = albedo_mesh.visual.vertex_colors
    colors = np.clip(albedos / 255 * np.clip(shadings/255, 0, 1), 0, 1) * 255
    verts = albedo_mesh.vertices
    faces = albedo_mesh.faces
    mesh = Trimesh(verts, faces, vertex_colors=colors)
    mesh.export(out_render_path)
    diff_mesh = Trimesh(verts, faces, vertex_colors=colors)
    # save_obj_mesh_with_color(out_render_path, verts,faces, albedos )
    
    pass

def reconstruction_visuall_hull(net, cuda, calib_tensor, mask_tensor, 
    resolution, b_min, b_max, num_views, transform=None, level=0.5):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor [K,H,W]
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param num_views: how many input views in the generation process
    :return: marching cubes results.
    '''
    from utils.render_utils import in_visual_hull

    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        samples = samples.repeat(num_views,1,1)
        xyz = net.projection(samples, calib_tensor,transform)
        pred = ~ in_visual_hull(xyz,mask_tensor,num_views) # [BK,1,N], [BK,1,N]
        return pred # 

    sdf = eval_grid_multi(coords, eval_func=eval_func)
    print(sdf.max())
    print(sdf.min())

    
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf.detach().cpu().numpy(), level)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        return verts, faces, normals, values
    except Exception as e:
        print(e)
        print('error cannot marching cubes')
        return -1            


def reconstruction_mlp(sdf_func, coarse_sdf_func, cuda, resolution, b_min, b_max, num_samples=100000, transform=None):
    resx, resy, resz = get_true_resolution(b_min, b_max, resolution)
    coords, mat = create_grid(resx, resy, resz,
                              b_min, b_max, transform=transform)

    def eval_func(points):
        pshape = points.shape
        npts = pshape[-1]
        a = torch.rand(npts,128).to(cuda)
        
        points = np.expand_dims(points, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        pred = sdf_func(samples, coarse_sdf_func(samples))

        # pred = net(samples.transpose(1,2).view(-1,3),a ).view(1,-1,1).transpose(1,2)
        if isinstance(pred, tuple):
            pred = pred[0]
        return pred #              

    sdf = eval_grid_multi(coords, eval_func, num_samples=num_samples)
    print(sdf.max())
    print(sdf.min())
    
    return sdf.detach().cpu().numpy(), mat


def get_true_resolution(b_min, b_max, resolution):
    shortest_axis = np.argmin(b_max-b_min)
    if shortest_axis==0:
        resx = resolution
        resy = (b_max[1] - b_min[1])/(b_max[0]-b_min[0]+1e-6) * resolution
        resz = (b_max[2] - b_min[2])/(b_max[0]-b_min[0]+1e-6) * resolution
    elif shortest_axis ==1:
        resx = (b_max[0] - b_min[0])/(b_max[1]-b_min[1]+1e-6) * resolution
        resy = resolution
        resz = (b_max[2] - b_min[2])/(b_max[1]-b_min[1]+1e-6) * resolution
    elif shortest_axis ==2:
        resx = (b_max[0] - b_min[0])/(b_max[2]-b_min[2]+1e-6) * resolution
        resy = (b_max[1] - b_min[1])/(b_max[2]-b_min[2]+1e-6) * resolution
        resz = resolution
    
    resx = int(resx+0.5)
    resy = int(resy+0.5)
    resz = int(resz+0.5)
    print(resx, resy, resz)
    return resx, resy, resz

def reconstruction_multi_2(net, cuda, calib_tensor, mask_tensor, normal_tensor, im_feat,
        resolution, b_min, b_max, num_views, z_center=None, use_positional_encoding=False,
        use_smpl=False, smpl_faces=None, smpl_verts=None, triangles=None,
        use_octree=False, num_samples=10000, transform=None, extrinsic_reshape=None, joints_3d=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor [K,H,W]
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    resx, resy, resz = get_true_resolution(b_min, b_max, resolution)
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resx, resy, resz,
                              b_min, b_max, transform=transform)
    # Then we define the lambda function for cell evaluation

    def eval_func(points):
        
        points = np.expand_dims(points, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        if not use_smpl:
            smpl_feat=None
        else:
            # import pdb
            # pdb.set_trace()
            residues, pts_ind, _ = point_to_mesh_distance( 
                samples.permute(0,2,1).contiguous(),  # [B,N,3]
                torch.from_numpy(triangles).float().to('cuda').unsqueeze(0))
            pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3)) # [B,N]
            pts_signs = 2.0 * (
                check_sign(
                    torch.from_numpy(smpl_verts).float().to('cuda').unsqueeze(0), 
                    torch.from_numpy(smpl_faces).long().to('cuda'), 
                    samples.permute(0,2,1).contiguous()
                    ).float() - 0.5) # inside points is 1
            pts_sdf = (pts_dist * -pts_signs) # we have sdf -1 inside
            smpl_feat = pts_sdf.unsqueeze(1) # [B, N]
        
        pred, in_img = net.query_sdf(im_feat, samples, calib_tensor,z_center,mask_tensor, smpl_feat, joints_3d=joints_3d,
            use_positional_encoding=use_positional_encoding, extrinsic_reshape=extrinsic_reshape, input_normal=normal_tensor) # [BK,1,N], [BK,1,N]
        if isinstance(pred, tuple):
            pred = pred[0]
        in_img = in_img.sum(dim=0).unsqueeze(0)
        in_vis_hull = in_img >=1

        # pred[in_vis_hull == False] = 10
        return pred # 
    
    # Then we evaluate the grid
    sdf = eval_grid_multi(coords, eval_func, num_samples=num_samples)
    print(sdf.max())
    print(sdf.min())
    
    return sdf.detach().cpu().numpy(), mat

def reconstruction_multi(sdf_func, cuda, # calibs, masks, ori_aspect,
        resolution, b_min, b_max, use_positional_encoding=False,
        use_smpl=False, smpl_faces=None, smpl_verts=None, triangles=None,
        use_octree=False, num_samples=100000, transform=None, sdf_func_extra=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor [K,H,W]
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    resx, resy, resz = get_true_resolution(b_min, b_max, resolution)
    coords, mat = create_grid(resx, resy, resz,
                              b_min, b_max, transform=transform)
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        if not use_smpl:
            smpl_feat=None
        else:
            # import pdb
            # pdb.set_trace()
            residues, pts_ind, _ = point_to_mesh_distance( 
                samples.permute(0,2,1).contiguous(),  # [B,N,3]
                torch.from_numpy(triangles).float().to('cuda').unsqueeze(0))
            pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3)) # [B,N]
            pts_signs = 2.0 * (
                check_sign(
                    torch.from_numpy(smpl_verts).float().to('cuda').unsqueeze(0), 
                    torch.from_numpy(smpl_faces).long().to('cuda'), 
                    samples.permute(0,2,1).contiguous()
                    ).float() - 0.5) # inside points is 1
            pts_sdf = (pts_dist * -pts_signs) # we have sdf -1 inside
            smpl_feat = pts_sdf.unsqueeze(1) # [B, N]
        
        if sdf_func_extra is not None:
            pred, in_img = sdf_func_extra(samples)
            # print(pred.min())
            # print(pred.max())
        else:
            pred, in_img = sdf_func(samples)
            # print(pred_extra.min())
            # print(pred_extra.max())

        # pred, in_img = net.query_sdf(im_feat, samples, calib_tensor,z_center,mask_tensor, smpl_feat, use_positional_encoding=use_positional_encoding) # [BK,1,N], [BK,1,N]
        if isinstance(pred, tuple):
            pred = pred[0]
        in_img = in_img.sum(dim=0).unsqueeze(0)
        in_vis_hull = in_img >= 1

        # pred[in_vis_hull == False] = 0.2
        # pred[in_vis_hull == True] = -0.2
        return pred # 

    # Then we evaluate the grid
    sdf = eval_grid_multi(coords, eval_func, num_samples=num_samples)
    # resolution = coords.shape[1:4]
    # coords = coords.reshape([1, 3, -1])
    # xyz = perspective(coords, calibs)
    # pts_in_vhull = in_visual_hull(xyz, masks, calibs.shape[0], dilation_kernel_size=0, min_observations=calibs.shape[0], ori_aspect=ori_aspect).permute(0,2,1)
    # pdb.set_trace()
    print(sdf.max())
    print(sdf.min())
    
    return sdf.detach().cpu().numpy(), mat
         

def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, transform=None, level=0.5):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor [1,K,H,W]
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        pred, _ = net.query_sdf(samples, calib_tensor)[0][0]
        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    print(sdf.max())
    print(sdf.min())
    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf, level)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        return verts, faces, normals, values
    except Exception as e:
        print(e)
        print('error cannot marching cubes')
        return -1

def gen_vis_hull(opt, net, cuda, data,save_path, num_views,level=0, ):
    level= 0.5
    image_tensor = data['img'][0].to(device=cuda) # [K,C,H,W]
    print(image_tensor.shape)
    calib_tensor = data['calib'][0].to(device=cuda) # [K,H,W]
    mask_tensor = data['mask'][0].to(device=cuda) # [K,H,W]    
    b_min = data['b_min'][0]
    b_max = data['b_max'][0]
    ret_visual_hull = reconstruction_visuall_hull(net, cuda, calib_tensor, mask_tensor, opt.resolution, b_min, b_max, num_views, level=level)
    if ret_visual_hull == -1:
        pass
    else:
        verts_hull, faces_hull, _, _ = ret_visual_hull
    

    verts_hull_tensor = torch.from_numpy(verts_hull.T).unsqueeze(0).to(device=cuda).float()
    xyz_hull_tensor = net.projection(verts_hull_tensor, calib_tensor[:1])
    uv_hull = xyz_hull_tensor[:, :2, :]
    color_hull = index(image_tensor[:1], uv_hull).detach().cpu().numpy()[0].T
    color_hull = color_hull * 0.5 + 0.5
    save_obj_mesh_with_color(save_path, verts_hull, faces_hull, color_hull)


def test_mesh_mlp(opt, sdf_func, coarse_sdf_func, cuda, data, save_path, save_ply=True):
    b_min = data['b_min']
    b_max = data['b_max']

    
    with torch.no_grad():
        sdf_low, mat_low  = reconstruction_mlp(sdf_func, coarse_sdf_func, cuda, 128, b_min, b_max, num_samples=opt.mc_batch_pts)
        try:
            verts, faces, normals, values = measure.marching_cubes(sdf_low, 0)
            # transform verts into world coordinate system
            verts = np.matmul(mat_low[:3, :3], verts.T) + mat_low[:3, 3:4]
            verts = verts.T
        except Exception as e:
            print(e)
            print('error cannot marching cubes for level %.2f' % 0)
        eps = 0.05
        # save_mesh_ply(save_albedo_path, verts, faces, albedo)

        b_min = verts.min(axis=0) - eps
        b_max = verts.max(axis=0) + eps

    with torch.no_grad():
        sdf, mat  = reconstruction_mlp(sdf_func, coarse_sdf_func, cuda, opt.resolution, b_min, b_max, num_samples=opt.mc_batch_pts)

    level=0
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf, level)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        
    except Exception as e:
        print(e)
        print('error cannot marching cubes for level %.2f' % level)

    # Now Getting colors
    verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
    albedo = np.ones(verts.shape)


    if save_ply:
        save_albedo_path = save_path[:-4]+'_albedo_%.2f.ply' % level
        save_mesh_ply(save_albedo_path, verts, faces, albedo)
    else:
        save_albedo_path = save_path[:-4]+'_albedo_%.2f.obj' % level
        save_obj_mesh_with_color(save_albedo_path, verts, faces, albedo)



def gen_mesh_color_unified(opt, unified_backbone, cuda, data, save_path, num_views, 
    use_octree=True, levels=[0], vid=None, save_ply=True):
    print('generate mesh color, using %d views' % num_views)
    image_tensor = data['img'].to(device=cuda) # [K,C,H,W]
    albedo_tensor = data['albedo'].to(device=cuda) # [K,C,H,W]
    calib_tensor = data['calib'].to(device=cuda) # [K,H,W]
    mask_tensor = data['mask'].to(device=cuda) # [K,H,W]
    # dist_tensor = data['dist'].to(device=cuda) # [K,H,W]
    extrinsic_tensor = data['extrinsic'].to(device=cuda)
    intrinsic_tensor = data['intrinsic'].to(device=cuda)
    pose_tensor = data['quat'].to(device=cuda)
    # labels = data['surface_color'].to(device=cuda)
    view_ids = data['view_ids']
    z_center = data['z_center'].to(device=cuda).unsqueeze(0)

    joints_3d = data['smpl_joints'].to(device=cuda).unsqueeze(0) if opt.use_spatial else None

    smpl_faces = data['smpl_faces'] if opt.use_smpl else None
    smpl_verts = data['smpl_verts'] if opt.use_smpl else None
    triangles = data['triangles'] if opt.use_smpl else None
    normal_tensor = data['normal'].float().to(device=cuda) if opt.input_normal else None


    
    if vid is not None: # specific view generation
        image_tensor = image_tensor[vid:vid+1,...]
        albedo_tensor = albedo_tensor[vid:vid+1,...]
        calib_tensor = calib_tensor[vid:vid+1,...]
        mask_tensor = mask_tensor[vid:vid+1,...]
        # dist_tensor = dist_tensor[vid:vid+1,...]
        extrinsic_tensor = extrinsic_tensor[vid:vid+1,...]
        if opt.input_normal:
            normal_tensor = normal_tensor[vid:vid+1,...]
    
    K,_,IH,IW = image_tensor.shape
    # with torch.no_grad():
    # im_feat = unified_backbone.filter(image_tensor)
    input_tensor = image_tensor
    if opt.feed_dir:
        uvs = get_uvs(IW, IH).expand(K, -1, -1).to(cuda)
        calibs, extris, extris_inv = get_calib_extri_from_pose(pose_tensor, intrinsic_tensor)

        ray_ds, cam_locs = get_camera_params_in_model_space(uvs.cpu(), extris_inv.cpu(), intrinsic_tensor.cpu(), neg_z=opt.use_perspective)
        ray_ds = ray_ds.transpose(1,2).reshape(K, 3, IW, IH).to(cuda)
        input_tensor = torch.cat((input_tensor, ray_ds), dim=1)

    if opt.feed_mask:
        input_tensor = torch.cat((input_tensor, mask_tensor), dim=1)
    if opt.feed_bound:
        input_tensor = torch.cat((input_tensor, mask_tensor), dim=1)
    # if opt.feed_dist:
    #     input_tensor = torch.cat((input_tensor, dist_tensor), dim=1)
    

    im_feat = unified_backbone.filter(input_tensor)
    

    save_img_path = save_path[:-4] + '_feat.png'
    save_img_list = []
    for v in range(im_feat.shape[0]):
        save_img = (np.transpose(im_feat[v,:3,...].clamp(-1,1).detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    Image.fromarray(np.uint8(save_img)).save(save_img_path)
    b_min = data['b_min']
    b_max = data['b_max']
    # try:
    save_img_path = save_path[:-4] + '.png'
    save_albedo_path = save_path[:-4] + '_albedo.png'
    save_img_list = []
    save_albedo_list = []
    save_img_tensor = image_tensor.clone()
    save_albedo_tensor = albedo_tensor.clone()

    if save_img_tensor.min()<0:
        save_img_tensor = save_img_tensor 
    if save_albedo_tensor.min()<0:
        save_albedo_tensor = save_albedo_tensor 
    
    if opt.use_linear_z:
        feed_extrin = extrinsic_tensor
    else:
        feed_extrin = None
    for v in range(save_img_tensor.shape[0]):
        save_img = (np.transpose(save_img_tensor[v].detach().cpu().numpy(), (1, 2, 0))) * 255.0
        save_img_list.append(save_img)
        save_albedo = (np.transpose(save_albedo_tensor[v].detach().cpu().numpy(), (1, 2, 0))) * 255.0
        save_albedo_list.append(save_albedo)

    save_img = np.concatenate(save_img_list, axis=1)
    save_albedo_img = np.concatenate(save_albedo_list, axis=1)
    Image.fromarray(np.uint8(save_img)).save(save_img_path)
    Image.fromarray(np.uint8(save_albedo_img)).save(save_albedo_path)

    # with torch.no_grad():
    #     sdf_low, mat_low  = reconstruction_multi_2(
    #     unified_backbone, cuda, calib_tensor, mask_tensor, normal_tensor,
    #     im_feat, 128, b_min, b_max, num_views,
    #     z_center, opt.use_positional_encoding, opt.use_smpl, smpl_faces, smpl_verts, triangles, use_octree=use_octree, num_samples=opt.mc_batch_pts, extrinsic_reshape=feed_extrin)
    #     try:
    #         verts, faces, normals, values = measure.marching_cubes(sdf_low, 0)
    #         # transform verts into world coordinate system
    #         verts = np.matmul(mat_low[:3, :3], verts.T) + mat_low[:3, 3:4]
    #         verts = verts.T
    #     except Exception as e:
    #         print(e)
    #         print('error cannot marching cubes for level %.2f' % 0)
    #         return
    #     eps = 0.05
    #     # save_mesh_ply(save_albedo_path, verts, faces, albedo)

    #     b_min = verts.min(axis=0) - eps
    #     b_max = verts.max(axis=0) + eps

    ## TODO: add view dir training 
    with torch.no_grad():
        sdf, mat  = reconstruction_multi_2(
        unified_backbone, cuda, calib_tensor, mask_tensor, normal_tensor,
        im_feat, opt.resolution, b_min, b_max, num_views,
        z_center, opt.use_positional_encoding, opt.use_smpl, smpl_faces, smpl_verts, triangles, use_octree=use_octree, num_samples=opt.mc_batch_pts, extrinsic_reshape=feed_extrin,joints_3d=joints_3d)
    
    for level in levels:
        try:
            verts, faces, normals, values = measure.marching_cubes(sdf, level, allow_degenerate=False)
            # transform verts into world coordinate system
            verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
            verts = verts.T
            
        except Exception as e:
            print(e)
            print('error cannot marching cubes')
            continue
        
        # Now Getting colors
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()

        color = np.zeros(verts.shape)
        albedo = np.zeros(verts.shape)
        normal = np.zeros(verts.shape)
        shading = np.zeros((num_views, verts.shape[0], verts.shape[1]))
        recon = np.zeros((num_views, verts.shape[0], verts.shape[1]))
        interval = 1000

        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            
            with torch.no_grad():
                pred_albedo = unified_backbone.query_albedo(im_feat, verts_tensor[:, :, left:right], calib_tensor, z_center_world_space=z_center, masks=mask_tensor, joints_3d=joints_3d,
                    use_positional_encoding=opt.use_positional_encoding, feed_original_img=opt.feed_original_img, imgs=image_tensor, extrinsic_reshape=feed_extrin, input_normal=normal_tensor) # [BK,1,N], [BK,1,N]
            # geo_normals = unified_backbone.get_sdf_normal(image_tensor, verts_tensor[:, :, left:right], calib_tensor, z_center, mask_tensor, im_feat=im_feat)

            # pred_shading = unified_backbone.query_shading(geo_normals, verts_tensor[:, :, left:right], extrinsic_tensor )
            albedo[left:right] = img_l2g(pred_albedo[0]).T.detach().cpu().numpy()
            
            if opt.input_normal:
                with torch.no_grad():
                    pred_normal = unified_backbone.query_normal(im_feat, verts_tensor[:, :, left:right], calib_tensor, z_center_world_space=z_center, masks=mask_tensor, use_positional_encoding=opt.use_positional_encoding, feed_original_img=opt.feed_original_img, imgs=image_tensor, extrinsic_reshape=feed_extrin, input_normal=normal_tensor)
                normal[left:right] = pred_normal[0].T.detach().cpu().numpy()
                
            # for k in range(num_views):
            #     pred_shading_k =  torch.clamp(pred_shading, 0, 1)[k]
            #     pred_recon = pred_shading_k * pred_albedo[0]
            #     shading[k, left:right] = shading_l2g(pred_shading_k).T.detach().cpu().numpy()
            #     recon[k, left:right] = img_l2g(pred_recon).T.detach().cpu().numpy()

        if save_ply:
            save_albedo_path = save_path[:-4]+'_albedo_%.2f.ply' % level
            save_mesh_ply(save_albedo_path, verts, faces, albedo)
            if opt.input_normal:
                save_normal_path = save_path[:-4]+'_normal_%.2f.ply' % level
                save_mesh_ply(save_normal_path, verts, faces, normal * .5 + .5)
        else:
            save_albedo_path = save_path[:-4]+'_albedo_%.2f.obj' % level
            save_obj_mesh_with_color(save_albedo_path, verts, faces, albedo)
            if opt.input_normal:
                save_normal_path = save_path[:-4]+'_normal_%.2f.obj' % level
                save_obj_mesh_with_color(save_normal_path, verts, faces, normal * .5 + .5)

        

    # for k in range(num_views):
    #     save_shading_path = save_path[:-4]+'_shading_%02d.obj' % k
    #     save_recon_path = save_path[:-4]+'_recon_%02d.obj' % k
    #     save_obj_mesh_with_color(save_shading_path, verts, faces, shading[k])
    #     save_obj_mesh_with_color(save_recon_path, verts, faces, recon[k])

    if opt.vis_level_set:
        resx, resy, resz = get_true_resolution(b_min, b_max, opt.resolution)
        
        for z in range(0,resz,4):
            # vis_sdf_x_plane(sdf, save_dir=save_path[:-4], depth=z, res=512)
            # vis_sdf_y_plane(sdf, save_dir=save_path[:-4], depth=z, res=512)
            vis_sdf_z_plane(sdf, save_dir=save_path[:-4],bmin=b_min, bmax=b_max, depth=z, resx=resx, resy=resy)
        for y in range(0,resy,4):
            # vis_sdf_x_plane(sdf, save_dir=save_path[:-4], depth=z, res=512)
            # vis_sdf_y_plane(sdf, save_dir=save_path[:-4], depth=z, res=512)
            vis_sdf_y_plane(sdf, save_dir=save_path[:-4],bmin=b_min, bmax=b_max, depth=y, resx=resx, resz=resz)

        import glob
        import cv2
        img_array_z = []
        zs = list(range(0, resz, 4))
        save_dir = save_path[:-4] + '_sdf_z'
        for z in zs:
            img = cv2.imread(os.path.join(save_dir, '%d.png'%z))
            height, width, layers = img.shape
            size = (width,height)
            img_array_z.append(img)
        out = cv2.VideoWriter(os.path.join(save_dir,'project.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
        
        for i in range(len(img_array_z)):
            out.write(img_array_z[i])
        out.release()

        img_array_y = []
        ys = list(range(0, resy, 4))
        save_dir = save_path[:-4] + '_sdf_y'
        for y in ys:
            img = cv2.imread(os.path.join(save_dir, '%d.png'%y))
            height, width, layers = img.shape
            size = (width,height)
            img_array_y.append(img)
        out = cv2.VideoWriter(os.path.join(save_dir,'project.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
        
        for i in range(len(img_array_y)):
            out.write(img_array_y[i])
        out.release()

# def test_mesh_color_unified(opt, unified_backbone, cuda, data, save_path, num_views, use_octree=True, level=0, vid=None, save_ply=True):
def test_mesh_color_unified(opt, sdf_func, color_func, cuda, data, save_path, 
sdf_func_extra=None, albedo_func_extra=None, use_octree=True, levels=[0],
save_ply=False, has_gt_albedo=True, vis_error=False, export_uv=False, output_spec_roughness=False, rough_func=None, spec_func=None):
    if export_uv and output_spec_roughness:
        assert(rough_func is not None)
        assert(spec_func is not None)
    else:
        rough_func = None
        spec_func = None
    
    b_min = data['b_min']
    b_max = data['b_max']
    
    with torch.no_grad():
        sdf_low, mat_low  = reconstruction_multi(
        sdf_func, cuda, 128, b_min, b_max, use_octree=use_octree, num_samples=opt.mc_batch_pts, sdf_func_extra=sdf_func_extra)
        try:
            verts, faces, normals, values = measure.marching_cubes(sdf_low, 0)
            # transform verts into world coordinate system
            verts = np.matmul(mat_low[:3, :3], verts.T) + mat_low[:3, 3:4]
            verts = verts.T
        except Exception as e:
            print(e)
            print('error cannot marching cubes for level %.2f' % 0)
            return
        margin = 0.05
        # save_mesh_ply(save_albedo_path, verts, faces, albedo)
        mesh_low_res = Trimesh(verts, faces, normals)
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float32)
        mesh_low_res = components[areas.argmax()]
        b_min = mesh_low_res.vertices.min(axis=0) - margin
        b_max = mesh_low_res.vertices.max(axis=0) + margin

    # b_min = np.array([-0.33, -1,-0.25])
    # b_max = np.array([0.25, 1.01, 0.25])

    with torch.no_grad():
        sdf, mat  = reconstruction_multi(
        sdf_func, cuda, opt.resolution, b_min, b_max, use_octree=use_octree, num_samples=opt.mc_batch_pts, sdf_func_extra=sdf_func_extra)
    

    for level in levels:
        try:
            verts, faces, normals, values = measure.marching_cubes(sdf, level)
            # transform verts into world coordinate system
            verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
            verts = verts.T
            
        except Exception as e:
            print(e)
            print('error cannot marching cubes for level %.2f' % level)
            return
        mesh_high_res = Trimesh(verts, faces, vertex_normals=normals)
        # components = mesh_high_res.split(only_watertight=False)
        # areas = np.array([c.area for c in components], dtype=np.float32)
        # mesh_high_res = components[areas.argmax()]
        verts, faces, normals = mesh_high_res.vertices, mesh_high_res.faces, mesh_high_res.vertex_normals

        # Now Getting colors
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()

        color = np.zeros(verts.shape)
        albedo = np.zeros(verts.shape)
        interval = 1000

        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            
            with torch.no_grad():
                if albedo_func_extra is None:
                    pred_albedo = color_func(verts_tensor[:, :, left:right]) # [BK,1,N], [BK,1,N]
                else:
                    pred_albedo = albedo_func_extra(verts_tensor[:,:,left:right]) # [BK,1,N], [BK,1,N]

            
            albedo[left:right] = img_l2g(pred_albedo[0]).T.detach().cpu().numpy()
            
        if save_ply and not export_uv:
            save_albedo_path = save_path[:-4]+'_albedo.ply'
            save_mesh_ply(save_albedo_path, verts, faces, albedo)
        else:
            save_albedo_path = save_path[:-4]+'_albedo.obj'
            save_obj_mesh_with_color(save_albedo_path, verts, faces, normals, albedo)
            if export_uv:
                blender_fpath = "/mnt/data1/lujiawei/blender-3.1.0-linux-x64/blender"
                if not os.path.isfile(blender_fpath):
                    os.system(
                        "wget https://mirrors.tuna.tsinghua.edu.cn/blender/release/Blender3.1/blender-3.1.0-linux-x64.tar.xz && \
                            tar -xvf blender-3.1.0-linux-x64.tar.xz"
                    )
                export_out_dir = os.path.dirname(save_albedo_path)
                mesh_name = os.path.basename(save_albedo_path)
                export_tex_dir = os.path.join(export_out_dir, 'tex')
                os.makedirs(export_tex_dir, exist_ok=True)
                os.system(
                f"{blender_fpath} --background --python utils/export_uv.py {os.path.join(export_out_dir, mesh_name)} {os.path.join(export_tex_dir, 'mesh.obj')}"
                )

                albedo_fn = color_func
                rough_fn = rough_func
                spec_fn = spec_func
                export_materials(os.path.join(export_tex_dir, "mesh.obj"), albedo_fn, rough_fn=rough_fn, spec_fn=spec_fn, out_dir=export_tex_dir, max_num_pts=320000, texture_H=4096, texture_W=4096, n_samples=5*10**6)



        if vis_error and level==0:
            # mesh_verts = data['mesh_verts']
            mesh = data['mesh']
            # surface_samples = data['surface_samples']
            save_error_vis_path = save_path[:-4]+'_error.ply'
            save_error_val_path = save_path[:-4]+'_error.txt'
            # dist = get_point_set_distance(verts, surface_samples[0].cpu().numpy().T)
            # dist = get_point_to_mesh_distance_kd_tree(verts, mesh_verts) # [N] array
            dist = get_point_to_mesh_distance_kaolin(verts, mesh) # [N,1] tensor
            import matplotlib as mpl
            import matplotlib.cm as cm

            norm = mpl.colors.Normalize(vmin=0, vmax=0.05, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('jet'))
            rgba = mapper.to_rgba(dist)
            save_mesh_ply(save_error_vis_path, verts, faces, rgba[:, :3])
            np.savetxt(save_error_val_path, [dist.min(), dist.max()])

    
    if opt.vis_level_set:
        # b_min = data['b_min']
        # b_max = data['b_max']
        resx, resy, resz = get_true_resolution(b_min, b_max, opt.resolution)
        
        for z in range(0,resz,4):
            vis_sdf_z_plane(sdf, save_dir=save_path[:-4],bmin=b_min, bmax=b_max, depth=z, resx=resx, resy=resy)
        for y in range(0,resy,4):
            vis_sdf_y_plane(sdf, save_dir=save_path[:-4],bmin=b_min, bmax=b_max, depth=y, resx=resx, resz=resz)

        # import glob
        # import cv2
        # img_array_z = []
        # zs = list(range(0, resz, 4))
        # save_dir = save_path[:-4] + '_sdf_z'
        # for z in zs:
        #     img = cv2.imread(os.path.join(save_dir, '%d.png'%z))
        #     height, width, layers = img.shape
        #     size = (width,height)
        #     img_array_z.append(img)
        # out = cv2.VideoWriter(os.path.join(save_dir,'project.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
        
        # for i in range(len(img_array_z)):
        #     out.write(img_array_z[i])
        # out.release()

        # img_array_y = []
        # ys = list(range(0, resy, 4))
        # save_dir = save_path[:-4] + '_sdf_y'
        # for y in ys:
        #     img = cv2.imread(os.path.join(save_dir, '%d.png'%y))
        #     height, width, layers = img.shape
        #     size = (width,height)
        #     img_array_y.append(img)
        # out = cv2.VideoWriter(os.path.join(save_dir,'project.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
        
        # for i in range(len(img_array_y)):
        #     out.write(img_array_y[i])
        # out.release()
        # generate sdf image at a slice


def get_point_set_distance(sp1, sp2, slice=10000):
    '''
    get the distance from source point cloud [sp1] to the nearest point in [sp2]
    :@ sp1 [N1, 3]
    :@ sp2 [N2, 3]

    :@ return
        min_dist_from_sp1_to_sp2: [N1]
    '''

    import scipy.spatial.distance as spdist

    N1, c = sp1.shape
    N2, c = sp2.shape
    n_slice = N1 // slice

    dist_matrix = np.empty((N1, N2), dtype=np.float32)
    for i in range(n_slice):
        dist_matrix[i*slice: i*slice+slice] = spdist.cdist(XA=sp1[i*slice: i*slice+slice], XB=sp2) # [N1, N2] array

    dist_matrix[(n_slice-1) * slice:] = spdist.cdist(XA=sp1[(n_slice-1) * slice:], XB=sp2)
    min_dist_from_sp1_to_sp2 = dist_matrix.min(axis=1)
    assert min_dist_from_sp1_to_sp2.shape[0] == N1
    return min_dist_from_sp1_to_sp2

def get_point_to_mesh_distance_kaolin(points:np.array, mesh):
    point_tensor = torch.from_numpy(points).float().unsqueeze_(0).to('cuda') # BN3
    verts_tensor = torch.from_numpy(mesh.vertices).float().unsqueeze_(0).to('cuda') # BN3
    faces_tensor = torch.from_numpy(mesh.faces).to('cuda') # N3
    face_vertices = index_vertices_by_faces(verts_tensor, faces_tensor) # BN33
    distance, index, dist_type = point_to_mesh_distance(point_tensor, face_vertices)
    distance = torch.sqrt(distance).cpu().numpy()[0] # [N,1]
    return distance

def get_point_to_mesh_distance_kd_tree(point_cloud:np.array, vertice_points:np.array):
    '''
    get point to vertice distance
    '''
    # make efficient search tree
    tree = cKDTree(vertice_points)

    # get indices of closest three points to use as vetice to calculate distance to
    d, idx_of_point_in_mesh = tree.query(point_cloud, 3)

    # anchor point to span a plane from
    anchor_points = vertice_points[idx_of_point_in_mesh[:,0],:]

    # use next two nearest points to span a plane with two vectors
    # from anchor point
    plane_points = vertice_points[idx_of_point_in_mesh[:,1:],:]
    plane_vecs = np.array(plane_points)
    plane_vecs[:,0,:] -= anchor_points
    plane_vecs[:,1,:] -= anchor_points

    # calculate normal vectors of the planes
    normals = np.cross(plane_vecs[:,0,:], plane_vecs[:,1,:], axis=1)
    # normals_norm = normals / (np.linalg.norm(normals, ord=2, axis=1)[:, None])

    # distance from each point to its anchor point for spanning a plane
    PQ = anchor_points - point_cloud

    # distance is dot product between normal and PQ vector
    # since normals and PQ are arrays of vectors 
    # use einsum to calc dot product along first dimension
    # dists = np.einsum('ij,ij->i', PQ, normals_norm)
    dist_vert = np.linalg.norm(PQ, axis=1)
    return dist_vert
    # return np.abs(dists)



def gen_mesh_color(opt, netG, netC, cuda, data, save_path, num_views, use_octree=True, level=0, vid=None):
    print('generate mesh color, using %d views' % num_views)
    image_tensor = data['img'].to(device=cuda) # [K,C,H,W]
    albedo_tensor = data['albedo'].to(device=cuda) # [K,C,H,W]
    calib_tensor = data['calib'].to(device=cuda) # [K,H,W]
    mask_tensor = data['mask'].to(device=cuda) # [K,H,W]
    extrinsic_tensor = data['extrinsic'].to(device=cuda).unsqueeze(0)
    cam_centers = data['cam_centers'].to(device=cuda) # [K, 3]
    normal_matrices = data['normal_matrices'].to(device=cuda)
    labels = data['surface_color'].to(device=cuda)
    view_ids = data['view_ids']

    if vid is not None: # specific view generation
        image_tensor = image_tensor[vid:vid+1,...]
        albedo_tensor = albedo_tensor[vid:vid+1,...]
        calib_tensor = calib_tensor[vid:vid+1,...]
        mask_tensor = mask_tensor[vid:vid+1,...]
        cam_centers = cam_centers[vid:vid+1,...]
        normal_matrices = normal_matrices[vid:vid+1,...]

    with torch.no_grad():
        geo_im_feat = netG.filter(image_tensor)
        netC.filter(image_tensor)
        if not opt.use_netG_feat:
            netC.attach(geo_im_feat)

    b_min = data['b_min']
    b_max = data['b_max']
    z_center = data['z_center']
    # try:
    save_img_path = save_path[:-4] + '.png'
    save_albedo_path = save_path[:-4] + '_albedo.png'
    save_img_list = []
    save_albedo_list = []
    save_img_tensor = image_tensor.clone()
    save_albedo_tensor = albedo_tensor.clone()
    if save_img_tensor.min()<0:
        save_img_tensor = save_img_tensor *0.5 +0.5 
    if save_albedo_tensor.min()<0:
        save_albedo_tensor = save_albedo_tensor * 0.5 + 0.5

    for v in range(save_img_tensor.shape[0]):
        save_img = (np.transpose(save_img_tensor[v].detach().cpu().numpy(), (1, 2, 0))) * 255.0
        save_img_list.append(save_img)
        save_albedo = (np.transpose(save_albedo_tensor[v].detach().cpu().numpy(), (1, 2, 0))) * 255.0
        save_albedo_list.append(save_albedo)

    save_img = np.concatenate(save_img_list, axis=1)
    save_albedo_img = np.concatenate(save_albedo_list, axis=1)
    Image.fromarray(np.uint8(save_img)).save(save_img_path)
    Image.fromarray(np.uint8(save_albedo_img)).save(save_albedo_path)

    with torch.no_grad():
        ret  = reconstruction_multi(
        netG, cuda, calib_tensor, mask_tensor, geo_im_feat, opt.resolution, b_min, b_max, num_views, z_center=z_center, use_octree=use_octree, num_samples=opt.mc_batch_pts)
    if ret == -1:
        return -1
    else:
        verts, faces, gt_normals, _, sdf = ret

    
    # Now Getting colors
    verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()

    

    color = np.zeros(verts.shape)
    albedo = np.zeros(verts.shape)
    shading = np.zeros((num_views, verts.shape[0], verts.shape[1]))
    recon = np.zeros((num_views, verts.shape[0], verts.shape[1]))
    interval = 1000

    for i in range(len(color) // interval):
        left = i * interval
        right = i * interval + interval
        if i == len(color) // interval - 1:
            right = -1
        geo_normals = netG.get_sdf_normal(image_tensor, verts_tensor[:, :, left:right], calib_tensor, mask_tensor)
        
        with torch.no_grad():
            if opt.use_netG_feat:
                geo_mlp_feat = netG.geo_feat
                netC.query(verts_tensor[:, :, left:right], calib_tensor, labels=labels, netG_normal=geo_normals.detach(), Model_mat=normal_matrices, light_pos=cam_centers.unsqueeze(-1), netG_feat=geo_mlp_feat)
            else:
                netC.query(verts_tensor[:, :, left:right], calib_tensor, labels=labels, netG_normal=geo_normals.detach(), Model_mat=normal_matrices, light_pos=cam_centers.unsqueeze(-1))
        
        pred_albedo = netC.get_preds()['pred_albedo'][0]
        albedo[left:right] = img_l2g(pred_albedo).T.detach().cpu().numpy()

        for k in range(num_views):
            pred_shading =  torch.clamp(netC.shading, 0, 1)[k]
            pred_recon = netC.get_preds()['pred_color'][k]

            shading[k, left:right] = shading_l2g(pred_shading).T.detach().cpu().numpy()
            recon[k, left:right] = img_l2g(pred_recon).T.detach().cpu().numpy()

        
    
    save_albedo_path = save_path[:-4]+'_albedo.obj'
    
    save_obj_mesh_with_color(save_albedo_path, verts, faces, albedo)
    

    for i in range(num_views):

        xyz_tensor = netG.projection(verts_tensor, calib_tensor[i:i+1])
        uv = xyz_tensor[:, :2, :]
        gt_albedo = index(albedo_tensor[i:i+1], uv).detach().cpu().numpy()[0].T
        gt_color = index(image_tensor[i:i+1], uv).detach().cpu().numpy()[0].T
        gt_albedo = gt_albedo * 0.5 + 0.5
        gt_color = gt_color * 0.5 + 0.5
        L1_loss = np.mean(np.abs(albedo-gt_albedo), axis=1)
        error_rgba = cm.get_cmap(plt.get_cmap(name))(L1_loss)

        save_color_gt_path = save_path[:-4]+'_color_gt_view_%d_%d.obj' % (i, view_ids[i])
        save_albedo_gt_path = save_path[:-4]+'_albedo_gt_view_%d_%d.obj' % (i, view_ids[i])
        save_albedo_error_path = save_path[:-4]+'_albedo_diff_view_%d_%d.obj' % (i, view_ids[i])
        save_shading_path = save_path[:-4]+'_shading_view_%d_%d.obj' % (i, view_ids[i])
        save_recon_path = save_path[:-4]+'_recon_view_%d_%d.obj' % (i, view_ids[i])
        save_obj_mesh_with_color(save_shading_path, verts, faces, shading[i])
        save_obj_mesh_with_color(save_recon_path, verts, faces, recon[i])
        save_obj_mesh_with_color(save_color_gt_path, verts, faces, gt_color)
        save_obj_mesh_with_color(save_albedo_gt_path, verts, faces, gt_albedo)
        save_obj_mesh_with_color(save_albedo_error_path, verts, faces, error_rgba[...,:3])


    # except Exception as e:
    #     print(e)
    #     print('Can not create marching cubes at this time.')


if __name__ == '__main__':
    pdir = '/home/lujiawei/workspace/tex2cloth/visualize_result/20220902/debug'
    sp = pdir + '/train_eval_epoch5_batch11000_0286_geo_shading.obj'
    ap = pdir + '/train_eval_epoch5_batch11000_0286_geo_albedo_gt_view_0_270.obj'
    op = pdir + '/train_eval_epoch5_batch11000_0286_color_gt_view_0_270.obj'
    get_rendered(sp, ap, op)
    
    pass




