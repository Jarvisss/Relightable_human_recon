import argparse
import os
import numpy as np
import sys
sys.path.append("./")
from src.utils.data_export import save_ply

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_dir', type=str, default="./samples")
parser.add_argument('--class_id', type=str, default="04256520")
parser.add_argument('--obj_id', type=str, default="1a04dcce7027357ab540cc4083acfa57")
parser.add_argument('--output_root_dir', type=str, default="./samples")
args = parser.parse_args()

def load_data(data_root_dir, class_id, obj_id):
    surface_path = os.path.join(data_root_dir, "nonsurface", class_id, obj_id+".npz")
    npz_data = np.load(surface_path)

    return npz_data["uniform_points"], npz_data["uniform_sdfs"], npz_data["surface_points"], npz_data["surface_sdfs"]

def surface2pc(points, sdfs):
    colors = np.zeros([points.shape[0], 3])
    colors[sdfs <= 0.0] = np.array([0.2, 0.2, 0.8])
    colors[sdfs > 0.0] = np.array([0.8, 0.2, 0.2])
    pts = points

    return pts, colors

if __name__ == "__main__":
    uniform_points, uniform_sdfs, surface_points, surface_sdfs = load_data(args.data_root_dir, args.class_id, args.obj_id)
    uniform_pts, uniform_colors = surface2pc(uniform_points, uniform_sdfs)
    surface_pts, surface_colors = surface2pc(surface_points, surface_sdfs)
    outfile_uniform = os.path.join(args.output_root_dir, args.class_id, args.obj_id+"_uniform.ply")
    outfile_surface = os.path.join(args.output_root_dir, args.class_id, args.obj_id+"_nsurface.ply")
    save_ply(uniform_pts, outfile_uniform, uniform_colors)
    save_ply(surface_pts, outfile_surface, surface_colors)
