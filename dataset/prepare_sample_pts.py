from matplotlib.pyplot import yscale
import trimesh
import numpy as np
import random
from utils.common_utils import set_random_seed
from kaolin.ops.mesh import check_sign
import torch
import os
import os.path as osp
# from mesh_to_sdf import get_surface_point_cloud

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
    y_scale = {}
    for sub_name in tqdm(subjects):
        # meshes[sub_name] = trimesh.load(os.path.join(path_obj, sub_name, '%s_100k.obj'%sub_name))
        meshes[sub_name] = trimesh.load_mesh(os.path.join(path_obj, sub_name, '%s.ply'%sub_name))
        # meshes[sub_name] = trimesh.load(os.path.join(path_obj, sub_name, '%s.obj'%'model_normalized'))
        y_scale[sub_name] = (meshes[sub_name].vertices.max(0)-meshes[sub_name].vertices.min(0))[1] /180.0 * sigma
    return meshes, y_scale




import argparse
def get_parser_args():
    parser = argparse.ArgumentParser()

    '''Common options'''
    parser.add_argument('--n_sample_space', '-n1',  type=int,default=200000,  help='train|test')
    parser.add_argument('--n_sample_surface', '-n2', type=int, default=200000, help = 'experiment ID. the experiment dir will be set as "./checkpoint/id/"')

    args = parser.parse_args()
    return args

# path_root = '/home/lujiawei/workspace/dataset/render_people_normalized_analytic_512'
# path_root = '/home/lujiawei/workspace/dataset/thuman2_rescaled'
obj_path = '/home/lujiawei/workspace/dataset/thuman2_rescaled_unit_sphere'
path_data = '/home/lujiawei/workspace/dataset'
# path_data = '/mnt/data1/lujiawei'
sample_path = osp.join(path_data, 'SAMPLE_PT_SCALE_UNIT_SPHERE')
os.makedirs(sample_path, exist_ok=True)
# obj_path = osp.join(path_root,  'GEO', 'OBJ')
subjects = sorted(os.listdir(obj_path))

# subjects = subjects[:100]
# subjects = subjects[100:200]
# subjects = subjects[200:300]
# subjects = subjects[300:400]
# subjects = subjects[400:500]
# subjects = subjects[500:]
subjects = subjects[-1:]

meshes, y_scales = load_trimesh(subjects, obj_path)

b_min = np.array([-1, -1, -1])
b_max = np.array([1, 1, 1])
# b_min = np.array([-1.5, -1.5, -1.5])
# b_max = np.array([1.5, 1.5, 1.5])

args = get_parser_args()
n_sample_space = args.n_sample_space
n_sample_surface = args.n_sample_surface


for i in range(len(subjects)):
    subject = subjects[i]
    print(subject)
    os.makedirs(osp.join(sample_path, subject), exist_ok=True)
    mesh = meshes[subject]
    y_scale = y_scales[subject]
    space_pts, space_inout = get_sample_points(mesh, y_scale, n_sample_points=n_sample_space, b_min=b_min, b_max=b_max)
    surface_pts, surface_normals = get_surface_sample_points(mesh, n_sample_points=n_sample_surface)
    

    level_set = 0.5
    r = (space_inout < level_set).reshape([-1, 1]) * 255
    g = (space_inout > level_set).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([space_pts, r, g, b], axis=-1)
    fname = osp.join(sample_path, subject, 'space.ply')
    np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          space_pts.shape[0])
                      )


    to_save = np.concatenate([surface_pts, (surface_normals*0.5+0.5)*255], axis=-1)
    fname = osp.join(sample_path, subject, 'surface.ply')
    np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          surface_pts.shape[0])
                      )


    
    np.save(osp.join(sample_path, subject, 'space.npy'),np.concatenate((space_pts, space_inout), axis=1))
    np.save(osp.join(sample_path, subject, 'surface.npy'),np.concatenate((surface_pts, surface_normals), axis=1))

    

