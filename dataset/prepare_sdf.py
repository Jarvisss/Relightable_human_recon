import os
import sys

from absl import flags
from absl import app
import imageio
import numpy as np
from py import process
import trimesh
from mesh_to_sdf import mesh_to_sdf
from mesh_to_sdf import sample_sdf_near_surface

os.environ['PYOPENGL_PLATFORM'] = 'egl'
FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'The path to the input mesh')
flags.DEFINE_string('output_dir', None, 'The path to the input mesh')
flags.DEFINE_string('input_mesh', None, 'The path to the input mesh')
flags.DEFINE_string('output_npz', None, 'The path to the output .npz')
flags.DEFINE_string('output_ply_uniform_out', None, 'The path to the output .npz')
flags.DEFINE_string('output_ply_uniform_in', None, 'The path to the output .npz')
flags.DEFINE_string('output_ply_surface_out', None, 'The path to the output .npz')
flags.DEFINE_string('output_ply_surface_in', None, 'The path to the output .npz')
flags.DEFINE_integer('n_samples', 200000, 'Number of surface points')
flags.DEFINE_float('points_uniform_ratio', 0.2, 'Ratio of points to sample uniformly')
flags.DEFINE_float('bbox_padding', 0.2, 'Padding for bounding box')
flags.DEFINE_float('max_dist', 0.03, 'Max distance for points near surface')

def sample_nonsurface(mesh, count, uniform_ratio, box_padding, max_dist):
    n_points_uniform = int(count * uniform_ratio)
    n_points_surface = count - n_points_uniform

    boxsize = 1 + box_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)

    points_surface, _ = trimesh.sample.sample_surface(mesh, n_points_surface)
    points_surface += max_dist * (np.random.rand(n_points_surface, 3)*2 - 1.0)
    points = np.concatenate([points_uniform, points_surface], axis=0)

    return points, n_points_uniform

def save_samples_truncted_prob(fname, points, prob, level_set=0.5):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    
    r = (prob > level_set).reshape([-1, 1]) * 255
    g = (prob < level_set).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


def save_samples_color(fname, points, points_colors):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''

    to_save = np.concatenate([points, points_colors], axis=-1)
    np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

    return 


def process_one_mesh(input_dir, input_name, output_dir, n_samples, surface_uniform_ratio, bbox_padding, max_dist):
    input_mesh = os.path.join(input_dir, input_name, '%s.obj'%input_name)
    print(f"Processing {input_mesh}")
    mesh = trimesh.load(input_mesh)

    # mesh_lst = mesh.split(only_watertight=False)
    # print()

    # comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
    # print(comp_num)
    # mesh = mesh_lst[comp_num.index(max(comp_num))]
    # print(mesh.is_watertight)
    points, n_points_uniform = sample_nonsurface(mesh, n_samples, surface_uniform_ratio, bbox_padding, max_dist)
    # sdfs = mesh_to_sdf(mesh, points, 
    #     surface_point_method='sample',
    #     sign_method='normal',
    #     bounding_radius=None,
    #     sample_point_count=10000000,
    #     normal_sample_count=11)
    sdfs = mesh_to_sdf(mesh, points, 
        surface_point_method='scan',
        scan_count=100,
        scan_resolution=400,
        sign_method='depth',
        bounding_radius=None,
        sample_point_count=10000000,
        normal_sample_count=11)


    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    os.makedirs(os.path.join(output_dir, input_name), exist_ok=True)
    uniform_points = points[:n_points_uniform]
    uniform_sdfs = sdfs[:n_points_uniform]
    surface_points = points[n_points_uniform:]
    surface_sdfs = sdfs[n_points_uniform:]

    output_npz = os.path.join(output_dir, input_name,'%s_sdf.npz'%input_name)
    output_ply_uniform_out = os.path.join(output_dir, input_name, '%s_unif_out.ply'%input_name)
    output_ply_uniform_in = os.path.join(output_dir, input_name, '%s_unif_in.ply'%input_name)
    output_ply_surface_out = os.path.join(output_dir, input_name, '%s_surf_out.ply'%input_name)
    output_ply_surface_in = os.path.join(output_dir, input_name, '%s_surf_in.ply'%input_name)
    np.savez(output_npz, uniform_points=uniform_points, uniform_sdfs=uniform_sdfs, surface_points=surface_points, surface_sdfs=surface_sdfs)

    outside = uniform_sdfs >= 0
    inside = uniform_sdfs < 0
    save_samples_truncted_prob(output_ply_uniform_out, uniform_points[outside], uniform_sdfs[outside], level_set=0)
    save_samples_truncted_prob(output_ply_uniform_in, uniform_points[inside], uniform_sdfs[inside], level_set=0)
    outside = surface_sdfs >= 0
    inside = surface_sdfs < 0
    save_samples_truncted_prob(output_ply_surface_out, surface_points[outside], surface_sdfs[outside], level_set=0)
    save_samples_truncted_prob(output_ply_surface_in, surface_points[inside], surface_sdfs[inside], level_set=0)


def main(argv):
    from tqdm import tqdm
    # input_names = sorted(os.listdir(FLAGS.input_dir))[:130]
    # input_names = sorted(os.listdir(FLAGS.input_dir))[130:260]
    # input_names = sorted(os.listdir(FLAGS.input_dir))[260:390]
    input_names = sorted(os.listdir(FLAGS.input_dir))[390:]
    for input_name in tqdm(input_names):
        process_one_mesh(
            FLAGS.input_dir, 
            input_name, 
            FLAGS.output_dir, 
            FLAGS.n_samples, 
            FLAGS.points_uniform_ratio, 
            FLAGS.bbox_padding, 
            FLAGS.max_dist
            )
    

if __name__ == "__main__":
    app.run(main)
