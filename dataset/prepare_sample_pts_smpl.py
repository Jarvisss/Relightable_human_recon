from matplotlib.pyplot import yscale
import trimesh
import numpy as np
import random
from utils.common_utils import set_random_seed
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
import torch
import os
import os.path as osp
# from mesh_to_sdf import get_surface_point_cloud
from .smplx_fit.smplx_fit import *

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_sample_points(mesh, scale, n_sample_points, b_min, b_max):
    '''
    sample points from mesh surface (in model space) and the bounding box
    separate into inside/outside
    '''
    
    # 16:1 sampling rate for surface sampling and space uniform sampling
    
    surface_points, _ = trimesh.sample.sample_surface(mesh, n_sample_points)
    
    surface_points_with_pertubation = surface_points + np.random.normal(scale=scale, size=surface_points.shape)

    length = b_max - b_min
    space_points = np.random.rand(n_sample_points//8, 3) * length + b_min
    # random_points_x = np.random.uniform(self.b_min[0], self.b_max[0], (n_sample_points // 4, 1))
    # random_points_y = np.random.uniform(self.b_min[1], self.b_max[1], (n_sample_points // 4, 1))
    # random_points_z = np.random.uniform(self.b_min[2], self.b_max[2], (n_sample_points // 4, 1))
    # space_points = np.concatenate((random_points_x, random_points_y, random_points_z), axis=1)

    sample_points = np.concatenate((surface_points_with_pertubation, space_points), axis=0 )
    # Multi-dimensional arrays are only shuffled along the first axis:
    np.random.shuffle(sample_points)
    # inside = mesh.contains(sample_points)
    
    ## the functions from kaolin takes gpu tensors as input/output
    verts_tensor = torch.from_numpy(mesh.vertices).float().unsqueeze_(0).to('cuda') # BN3
    faces_tensor = torch.from_numpy(mesh.faces).to('cuda') # N3

    pts_tensor = torch.from_numpy(sample_points).float().unsqueeze_(0).to('cuda') # BN3
    #### we compute this costly function on GPU and then send back to cpu
    inside_tensor = check_sign(verts_tensor, faces_tensor, pts_tensor)
    inside = inside_tensor[0].to('cpu').numpy()

    '''GPU end'''
    inside_points = sample_points[inside]
    outside_points = sample_points[np.logical_not(inside)]

    nin = inside_points.shape[0]
    inside_points = inside_points[: n_sample_points//2] if nin>n_sample_points//2 else inside_points
    outside_points = outside_points[: n_sample_points//2] if nin>n_sample_points //2 else outside_points[:(n_sample_points - nin)]

    samples = np.concatenate((inside_points, outside_points), axis=0) # [N, 3]
    labels =  np.concatenate( (  np.zeros((inside_points.shape[0], 1)), np.ones((outside_points.shape[0], 1)) ), axis=0) #[N, 1]

    
    return samples, labels  
    

def get_surface_sample_points(mesh, n_sample_points):
    '''
    sample points from mesh surface (in model space) and the bounding box
    separate into inside/outside
    '''
    # cloud = get_surface_point_cloud(mesh, 
    #     surface_point_method='scan',
    #     scan_count=100,
    #     scan_resolution=400,
    #     sample_point_count = n_sample_points,
    #     calculate_normals=True)
    # # 16:1 sampling rate for surface sampling and space uniform sampling
    # indices = np.random.choice(cloud.points.shape[0], n_sample_points, replace=False)
    # face_samples = cloud.points[indices]
    # face_normals = cloud.normals[indices]
    # face_normals = face_normals / np.sqrt((face_normals**2).sum(axis=1))[:, None]
    

    face_samples, face_indices = trimesh.sample.sample_surface(mesh, n_sample_points)
    face_normals = mesh.face_normals[face_indices]
    return face_samples, face_normals

def load_trimesh(subjects, path_obj, sigma=5.0):
    from tqdm import tqdm
    meshes = {}
    sigma_scale = {}
    for sub_name in tqdm(subjects):
        print(sub_name)
        # meshes[sub_name] = trimesh.load(os.path.join(path_obj, sub_name, '%s_100k.obj'%sub_name))
        meshes[sub_name] = trimesh.load(os.path.join(path_obj, sub_name, '%s.obj'%sub_name))
        # meshes[sub_name] = trimesh.load(os.path.join(path_obj, sub_name, '%s.obj'%'model_normalized'))
        sigma_scale[sub_name] = (meshes[sub_name].vertices.max(0)-meshes[sub_name].vertices.min(0))[1] /180.0 * sigma
    return meshes, sigma_scale

def face_vertices(v, f):
    nv = v.shape[0]
    nf = f.shape[0]
    return v[f]



import argparse
def get_parser_args():
    parser = argparse.ArgumentParser()

    '''Common options'''
    parser.add_argument('--n_sample_space', '-n1',  type=int,default=10000,  help='train|test')
    parser.add_argument('--n_sample_surface', '-n2', type=int, default=10000, help = 'experiment ID. the experiment dir will be set as "./checkpoint/id/"')

    args = parser.parse_args()
    return args

# path_root = '/home/lujiawei/workspace/dataset/render_people_normalized_analytic_512'
# path_root = '/home/lujiawei/workspace/dataset/thuman2_rescaled'
path_root = '/home/lujiawei/workspace/dataset/thuman2'
# path_data = '/home/lujiawei/workspace/dataset'
path_data = '/mnt/data1/lujiawei/thuman2_rescaled_unit_sphere_2'
sample_path = osp.join(path_data, 'SAMPLE_PT_SCALE_UNIT_SPHERE')

y_scale_path = sample_path
os.makedirs(sample_path, exist_ok=True)
os.makedirs(y_scale_path, exist_ok=True)
# obj_path = osp.join(path_root,  'GEO', 'OBJ')
obj_path = path_root
# subjects = sorted(os.listdir(obj_path))[0:100]
# subjects = sorted(os.listdir(obj_path))[100:200]
# subjects = sorted(os.listdir(obj_path))[200:300]
# subjects = sorted(os.listdir(obj_path))[300:400]
# subjects = sorted(os.listdir(obj_path))[400:500]
subjects = sorted(os.listdir(obj_path))[500:]

meshes, sigma_scales = load_trimesh(subjects, obj_path)

b_min = np.array([-1, -1, -1])
b_max = np.array([1, 1, 1])

# b_min = np.array([-128, -28, -128]) / 180
# b_max = np.array([128, 228, 128])   / 180
args = get_parser_args()
n_sample_space = args.n_sample_space
n_sample_surface = args.n_sample_surface
y_target = 0.95

for i in range(len(subjects)):
    subject = subjects[i]
    print(subject)
    os.makedirs(osp.join(sample_path, subject), exist_ok=True)
    os.makedirs(osp.join(y_scale_path, subject), exist_ok=True)
    mesh = meshes[subject]
    sigma = sigma_scales[subject]
    
    vertices = mesh.vertices
    up_axis=1
    print('vmax: [%.2f %.2f %.2f], vmin: [%.2f %.2f %.2f]' %(vertices.max(0)[0],vertices.max(0)[1],vertices.max(0)[2], vertices.min(0)[0], vertices.min(0)[1], vertices.min(0)[2]))
    vmax = vertices.max(0)
    vmin = vertices.min(0)
    vmed = np.median(vertices, 0)
    vmed[up_axis] = (vmax[up_axis] + vmin[up_axis])/2

    # y_scale = y_target / (vmax[up_axis]-vmin[up_axis])
    y_scale = y_target / np.linalg.norm((vertices - vmed), axis=1).max()


    # space_pts, space_inout = get_sample_points(mesh, sigma, n_sample_points=n_sample_space, b_min=b_min, b_max=b_max)
    # surface_pts, surface_normals = get_surface_sample_points(mesh, n_sample_points=n_sample_surface)
    
    space_pts = np.load(osp.join(sample_path, subject, 'space.npy'))[:,:3]
    surface_pts = np.load(osp.join(sample_path, subject, 'surface.npy'))[:,:3]

    fit_file = f'{path_root}/{subject}/smplx_param.pkl'
    rescale_fitted_body, joints = load_fit_body(fit_file,
                                            y_scale,
                                            -vmed,
                                            smpl_type='smplx',
                                            smpl_gender='male')
    
    # rescale_fitted_body.apply_translation(-vmed)
    np.save(osp.join(y_scale_path, subject, 'scale.npy'),np.array(y_scale))
    np.save(osp.join(y_scale_path, subject, 'vmed.npy'),vmed)


    smpl_vertices = rescale_fitted_body.vertices
    smpl_faces = rescale_fitted_body.faces

    smpl_faces = smpl_faces[~SMPLX().smplx_eyeball_fid] # remove eyeball
    mouth_faces = (SMPLX().smplx_mouth_fid) # cover mouth
    smpl_faces = np.concatenate((smpl_faces, mouth_faces), axis=0)
    # rescale_fitted_body.vertices = smpl_vertices
    # rescale_fitted_body.faces = smpl_faces
    # rescale_fitted_body.export('/home/lujiawei/workspace/dataset/smpl_related/smpl_data/test.obj')

    triangles = face_vertices(smpl_vertices, smpl_faces)

    residues, pts_ind, _ = point_to_mesh_distance( torch.from_numpy(space_pts).float().to('cuda').unsqueeze(0), torch.from_numpy(triangles).float().to('cuda').unsqueeze(0))
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))
    pts_signs = 2.0 * (
                check_sign(
                    torch.from_numpy(smpl_vertices).float().to('cuda').unsqueeze(0), 
                    torch.from_numpy(smpl_faces).long().to('cuda'), 
                    torch.from_numpy(space_pts).float().to('cuda').unsqueeze(0)
                    ).float() - 0.5)
    
    space_pts_sdf = (pts_dist * -pts_signs)

    residues, pts_ind, _ = point_to_mesh_distance( torch.from_numpy(surface_pts).float().to('cuda').unsqueeze(0), torch.from_numpy(triangles).float().to('cuda').unsqueeze(0))
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))
    pts_signs = 2.0 * (
                check_sign(
                    torch.from_numpy(smpl_vertices).float().to('cuda').unsqueeze(0), 
                    torch.from_numpy(smpl_faces).long().to('cuda'), 
                    torch.from_numpy(surface_pts).float().to('cuda').unsqueeze(0)
                    ).float() - 0.5)
    
    surf_pts_sdf = (pts_dist * -pts_signs)


    save_sdf_space = space_pts_sdf.cpu().detach().numpy().T
    save_sdf_surf = surf_pts_sdf.cpu().detach().numpy().T

    np.save(osp.join(sample_path, subject, 'space_smpl_sdf.npy'),save_sdf_space)
    np.save(osp.join(sample_path, subject, 'surf_smpl_sdf.npy'),save_sdf_surf)

    visualize=True
    if visualize:
        save_sdf_space = (save_sdf_space - save_sdf_space.min() ) / (save_sdf_space.max() - save_sdf_space.min())

        to_save = np.concatenate([space_pts, (save_sdf_space)*255,(save_sdf_space*0.5+0.5)*0,(save_sdf_space*0.5+0.5)*0], axis=-1)
        fname = osp.join(sample_path, subject, 'space_sdf.ply')
        np.savetxt(fname,
                        to_save,
                        fmt='%.6f %.6f %.6f %d %d %d',
                        comments='',
                        header=(
                            'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                            surface_pts.shape[0])
                        )
        
        print(surf_pts_sdf.min())
        print(surf_pts_sdf.max())
        # save_sdf_surf = (save_sdf_surf - save_sdf_surf.min() ) / (save_sdf_surf.max() - save_sdf_surf.min())
        save_sdf_surf[save_sdf_surf <=0] = 0
        save_sdf_surf[save_sdf_surf >0] = 1

        to_save = np.concatenate([surface_pts, (save_sdf_surf)*255,(save_sdf_surf*0.5+0.5)*0,(save_sdf_surf*0.5+0.5)*0], axis=-1)
        fname = osp.join(sample_path, subject, 'surface_sdf.ply')
        np.savetxt(fname,
                        to_save,
                        fmt='%.6f %.6f %.6f %d %d %d',
                        comments='',
                        header=(
                            'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                            surface_pts.shape[0])
                        )
    import pdb
    pdb.set_trace()


    

    

