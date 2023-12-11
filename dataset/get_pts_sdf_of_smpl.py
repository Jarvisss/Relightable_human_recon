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
import pdb

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
# path_root = '/home/lujiawei/workspace/dataset/thuman2'
# path_data = '/home/lujiawei/workspace/dataset'
path_smpl = '/mnt/data1/lujiawei/thuman2_rescaled_unit_sphere_2'
path_smpl_param = '/home/lujiawei/workspace/dataset/thuman2'
path_existing_pts = '/mnt/data1/lujiawei/SAMPLE_PT_SCALE_UNIT_SPHERE' 

subjects = sorted(os.listdir(path_existing_pts))

for i in range(len(subjects)):
    subject = subjects[i]
    print(subject)

    scale, x, y, z = np.load(os.path.join(path_smpl, subject, '%s_scale_trans.npy'%subject))

    space_pts = np.load(osp.join(path_existing_pts, subject, 'space.npy'))[:,:3]
    surface_pts = np.load(osp.join(path_existing_pts, subject, 'surface.npy'))[:,:3]

    # fit_file = f'{path_smpl}/{subject}/{subject}_smplx.obj'
    fit_file = f'{path_smpl_param}/{subject}/smplx_param.pkl'
    # rescale_fitted_body = trimesh.load(fit_file)
    rescale_fitted_body, joints = load_fit_body(fit_file,
                                            scale,
                                            -np.array([x,y,z]),
                                            smpl_type='smplx',
                                            smpl_gender='male')

    smpl_vertices = rescale_fitted_body.vertices
    smpl_faces = rescale_fitted_body.faces

    smpl_faces = smpl_faces[~SMPLX().smplx_eyeball_fid] # remove eyeball
    mouth_faces = (SMPLX().smplx_mouth_fid) # cover mouth
    smpl_faces = np.concatenate((smpl_faces, mouth_faces), axis=0)
    rescale_fitted_body.vertices = smpl_vertices
    rescale_fitted_body.faces = smpl_faces
    rescale_fitted_body.export(f'{path_smpl}/{subject}/{subject}_watertight.obj')

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

    np.save(osp.join(path_existing_pts, subject, 'space_smpl_sdf.npy'),save_sdf_space)
    np.save(osp.join(path_existing_pts, subject, 'surf_smpl_sdf.npy'),save_sdf_surf)

    visualize=True
    if visualize:
        # save_sdf_space = (save_sdf_space - save_sdf_space.min() ) / (save_sdf_space.max() - save_sdf_space.min())
        save_sdf_space[save_sdf_space>0] = 1 
        save_sdf_space[save_sdf_space<=0] = 0 

        to_save = np.concatenate([space_pts, (save_sdf_space)*255,save_sdf_space*0,(1-save_sdf_space)*255], axis=-1)
        fname = osp.join(path_existing_pts, subject, 'space_sdf.ply')
        np.savetxt(fname,
                        to_save,
                        fmt='%.6f %.6f %.6f %d %d %d',
                        comments='',
                        header=(
                            'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                            surface_pts.shape[0])
                        )
        
        # print(surf_pts_sdf.min())
        # print(surf_pts_sdf.max())
        # save_sdf_surf = (save_sdf_surf - save_sdf_surf.min() ) / (save_sdf_surf.max() - save_sdf_surf.min())
        save_sdf_surf[save_sdf_surf <=0] = 0
        save_sdf_surf[save_sdf_surf >0] = 1

        to_save = np.concatenate([surface_pts, (save_sdf_surf)*255,(save_sdf_surf*0.5+0.5)*0,(save_sdf_surf*0.5+0.5)*0], axis=-1)
        fname = osp.join(path_existing_pts, subject, 'surface_sdf.ply')
        np.savetxt(fname,
                        to_save,
                        fmt='%.6f %.6f %.6f %d %d %d',
                        comments='',
                        header=(
                            'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                            surface_pts.shape[0])
                        )
    # pdb.set_trace()


    

    

