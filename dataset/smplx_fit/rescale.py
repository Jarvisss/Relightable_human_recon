
import numpy as np
import os
import os.path as osp
import trimesh
from smplx_fit import *
from argparse import ArgumentParser
from tqdm import tqdm

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--iseg', '-i', type=int, default=0)
    parser.add_argument('--nseg', type=int, default=5)
    args = parser.parse_args()
    return args


args = get_parser()

N_SEG = args.nseg
I_SEG = args.iseg

mesh_paths = sorted(os.listdir('./OBJ'))
# y_target = 1
y_target = 0.95

SEG = len(mesh_paths) // N_SEG
start = I_SEG * SEG
end = (I_SEG+1) * SEG if I_SEG < N_SEG-1 else len(mesh_paths)


for i in  tqdm(range(start, end)):
    mp = mesh_paths[i]
    print(mp)
    mesh_path = osp.join('./OBJ', mp, '%s.obj'%mp)
    fit_file = f'./smplx_fit/{mp}/smplx_param.pkl'
    
    mesh = trimesh.load(mesh_path, maintain_order=True, skip_materials=True)
    vertices = mesh.vertices
    up_axis=1
    print('vmax: [%.2f %.2f %.2f], vmin: [%.2f %.2f %.2f]' %(vertices.max(0)[0],vertices.max(0)[1],vertices.max(0)[2], vertices.min(0)[0], vertices.min(0)[1], vertices.min(0)[2]))
    vmax = vertices.max(0)
    vmin = vertices.min(0)
    vmed = np.median(vertices, 0)
    vmed[up_axis] = (vmax[up_axis] + vmin[up_axis])/2

    # y_scale = y_target / (vmax[up_axis]-vmin[up_axis])
    y_scale = y_target / np.linalg.norm((vertices - vmed), axis=1).max()



    (tex_data, tex_name, mat_name) = mesh.visual.material.to_obj()

    
    # vmin = vertices.min(0)
    # vmax = vertices.max(0)
    # up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
    # y_scale = 1/(vmax[up_axis] - vmin[up_axis])
    rescale_fitted_body, joints = load_fit_body(fit_file,
                                            y_scale,
                                            smpl_type='smplx',
                                            smpl_gender='male')

    # vmed[up_axis] = 
    # vmed = joints[0]
    
    mesh.apply_translation(-vmed)
    mesh.apply_scale(y_scale)
    vertices = mesh.vertices
    print('scaled vmax: [%.2f %.2f %.2f], scaled vmin: [%.2f %.2f %.2f]' %(vertices.max(0)[0],vertices.max(0)[1],vertices.max(0)[2], vertices.min(0)[0], vertices.min(0)[1], vertices.min(0)[2]))

    rescale_fitted_body.apply_translation(-vmed)


    save_dir = osp.join('./thuman2_rescaled_unit_sphere', mp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, '%s.obj'%mp)
    save_mtl_path = osp.join(save_dir, 'material0.mtl')
    save_tex_path = osp.join(save_dir, 'material0.png')
    save_smpl_path = osp.join(save_dir, '%s_smplx.obj'%mp)
    
    mesh.export(save_path)
    rescale_fitted_body.export(save_smpl_path)

    os.remove(save_mtl_path)
    os.remove(save_tex_path)

