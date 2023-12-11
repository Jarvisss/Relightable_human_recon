from multiprocessing.spawn import get_preparation_data
from operator import sub
import os
import os.path as osp
from urllib.request import ProxyBasicAuthHandler
import numpy as np
from utils.camera import Camera, MVP_from_P, get_quat_from_world_mat_np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as F
from tqdm import tqdm

import random
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
from .smplx_fit.smplx_fit import *
import cv2
from utils.common_utils import crop_padding, set_random_seed, find_border
from utils.sdf_utils import save_samples_truncted_prob
# from kaolin.metrics.trianglemesh import point_to_mesh_distance
# from kaolin.ops.mesh import index_vertices_by_faces, check_sign
import trimesh
import logging
import pdb

log = logging.getLogger('trimesh')
log.setLevel(40)

def make_dataset(opt, phase):
    """Create dataset"""
    # root_synthetic = opt.path_to_dataset # /home/lujiawei/workspace/dataset/rendered_cloth

    # data_dir = 'thuman2_render_prt-new-normal_{0}'.format(opt.size)
    # path_root = osp.join(root_synthetic, data_dir)
    # the Thuman 2.0 dataset is normalized to [-0.5, 0.5]
    # we add 0.1 for safe padding
    path_root = opt.path_to_dataset
    path_root_taichi = opt.path_to_taichi_render
    path_obj = opt.path_to_obj
    path_sdf = '/mnt/data/lujiawei/thuman2_rescaled_unit_sphere_sdf'
    path_pt = opt.path_to_sample_pts
    path_SMPL = opt.path_to_SMPL
    path_SMPL_joints = opt.path_to_SMPL_joints
    if opt.dataset=='Thuman2':
        b_min = np.array([-1, -1, -1])
        b_max = np.array([1, 1, 1])
    elif opt.dataset == 'RP_normalized':
        b_min = np.array([-128, -28, -128]) / 180
        b_max = np.array([128, 228, 128])  / 180
    elif opt.dataset == 'RP':
        b_min = np.array([-128, -28, -128])
        b_max = np.array([128, 228, 128])
    else:
        print('Undefined dataset: %s' % opt.dataset)
        return -1
    dataset = Thuman2_taichi3_dataset(
        opt = opt,
        phase = phase,
        path_root = path_root,
        path_albedo = osp.join(path_root_taichi,  'albedo') ,
        path_mask = osp.join(path_root_taichi, 'mask') ,
        path_lit = osp.join(path_root_taichi,  'img') ,
        path_normal = osp.join(path_root_taichi, 'normal') ,
        path_param = osp.join(path_root_taichi, 'parameter') ,
        # path_obj = osp.join(path_root,  'GEO', 'OBJ') ,
        path_obj = path_obj, # save for disk space
        path_pts = path_pt,
        path_sdf = path_sdf,
        path_smpl= path_SMPL,
        path_smpl_joints= path_SMPL_joints,
        path_shading = osp.join(path_root, 'SHADING') ,
        path_uv_lit = osp.join(path_root,  'UV_RENDER') ,
        path_uv_mask = osp.join(path_root,  'UV_MASK') ,
        path_uv_pos = osp.join(path_root,  'UV_POS') ,
        path_uv_normal = osp.join(path_root,  'UV_NORMAL') ,
        path_uv_albedo = osp.join(path_root,  'UV_ALBEDO') ,
        path_uv_shading = osp.join(path_root,  'UV_SHADING') ,
        B_Min = b_min,
        B_Max = b_max, 
        padding_rate = opt.pr,
        size=opt.size, # [512, 512]
        random_trans = opt.random_trans,
        random_flip = opt.random_flip,
        random_scale = opt.random_scale,
        aug_blur = opt.aug_blur
        )
    
    return dataset

def make_dataloader(opt, dataset, phase):
    is_train = phase == 'train'
    batch_size = 1 if not is_train else opt.batch_size
    shuffle = is_train # 如果在训练阶段则打乱顺序
    drop_last = is_train # 如果在训练阶段则丢掉不完整的batch
    num_workers = opt.num_workers
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=num_workers, drop_last=drop_last)
    return dataloader

class ColorAug:
    def __init__(self, bri=0.5, con=0.5, sat=0.5, hue=0.2):
        self.color_jit = ColorJitter(bri, con, sat, hue)
    
    def generate(self):
        self.trans = self.color_jit.get_params(self.color_jit.brightness, self.color_jit.contrast, self.color_jit.saturation, self.color_jit.hue)

    def apply(self, img):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.trans
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        return img




class Thuman2_taichi3_dataset(data.Dataset):
    def __init__(
        self, opt, phase, path_root, path_albedo,path_normal, path_param,
        path_mask, path_lit, path_obj,path_pts, path_sdf, path_smpl, path_smpl_joints,path_shading,
        path_uv_lit, path_uv_mask, path_uv_pos, path_uv_normal, path_uv_albedo, path_uv_shading,
        B_Min, B_Max, padding_rate, size,
        random_trans=True, random_flip=True, random_scale=True, aug_blur=0,verbose=False
        ):
        self.is_train = True if phase == 'train' else False
        self.opt = opt
        self.use_smpl = opt.use_smpl
        self.path_root = path_root
        self.path_albedo = path_albedo
        self.path_lit = path_lit
        self.path_obj = path_obj
        self.path_pts = path_pts
        self.path_sdf = path_sdf
        self.path_smpl = path_smpl
        self.path_smpl_joints = path_smpl_joints
        self.path_normal = path_normal
        self.path_param = path_param
        self.path_mask = path_mask
        self.path_shading = path_shading
        self.path_uv_lit = path_uv_lit
        self.path_uv_mask = path_uv_mask
        self.path_uv_pos = path_uv_pos
        self.path_uv_normal = path_uv_normal
        self.path_uv_albedo = path_uv_albedo
        self.path_uv_shading = path_uv_shading

        self.load_size = size
        self.b_min = B_Min
        self.b_max = B_Max
        self.padding_rate = padding_rate
        self.verbose = verbose
        
        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_surface = self.opt.num_sample_surface
        self.num_sample_color = self.opt.num_sample_color
        self.random_trans = random_trans
        self.random_flip = random_flip
        self.random_scale = random_scale
        self.aug_blur = aug_blur
        self.yaw_angle_step = opt.yaw_angle_step
        self.yaw_list = list(range(0, 360, self.yaw_angle_step))
        # self.yaw_list = [0]
        # self.yaw_list = [222]
        self.pitch_list = [0]   
        self.subjects = self.get_subjects()

        if self.opt.load_mesh_ram and not self.opt.offline_sample:
            self.meshes, self.y_scales = self.load_trimesh()
        if self.opt.offline_sample:
            if self.opt.use_gt_sdf:
                ## use sdf as label
                self.space_sdfs = self.load_sdf()
            ## use sign as label
            ## space: [xyz, sdf] 4d
            ## surface: [xyz, normal] 6d
            self.space_pts, self.surface_pts, self.space_sdf, self.surface_sdf = self.load_pts()
            assert(self.space_pts[self.subjects[0]].shape[1] == 4)
            assert(self.surface_pts[self.subjects[0]].shape[1] == 6)

        if self.opt.use_smpl:
            self.smpl_faces, self.smpl_verts = self.load_SMPLs()
        
        if self.opt.use_spatial:
            self.smpl_joints = self.load_SMPL_joints()
        # normalize to [-1, 1] 
        self.rgb2tensor = transforms.Compose([
            transforms.Resize(self.opt.load_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # do not normalize for mask
        self.mask2tensor = transforms.Compose([
            transforms.Resize(self.opt.load_size),
            transforms.ToTensor(),
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

    def get_subjects(self):
        train_subjects = np.loadtxt(osp.join(self.path_root, 'train.txt'), dtype=str)
        val_subjects = np.loadtxt(osp.join(self.path_root, 'val.txt'), dtype=str)

        if self.is_train:
            return sorted(list(train_subjects))
        else:
            return sorted(list(val_subjects))

    def load_trimesh(self):
        from tqdm import tqdm
        if self.opt.dataset=='Thuman2':
            post_fix = ''
        elif self.opt.dataset == 'RP' or self.opt.dataset == 'RP_normalized':
            post_fix = '_100k'
        else:
            print('Undefined dataset: %s' % self.opt.dataset)
            return -1
        meshes = {}
        y_scale = {}
        for sub_name in tqdm(self.subjects):
            meshes[sub_name] = trimesh.load(os.path.join(self.path_obj, sub_name, '%s%s.obj'%(sub_name, post_fix)))
            # meshes[sub_name] = trimesh.load(os.path.join(self.path_obj, sub_name, '%s_100k_meter_180.obj'%sub_name))
            y_scale[sub_name] = (meshes[sub_name].vertices.max(0)-meshes[sub_name].vertices.min(0))[1] /180.0 * self.opt.sigma
        return meshes, y_scale

    def load_SMPL_joints(self):
        smpl_joints_all = {}
        for sub_name in tqdm(self.subjects):
            joint_file = f'{self.path_smpl_joints}/{sub_name}/{sub_name}_smplx_joints.npy'
            smpl_joints_all.update({sub_name: np.load(joint_file)})
        
        return smpl_joints_all
    def load_SMPLs(self):
        
        smpl_faces_all= {}
        smpl_vertices_all = {}
        
        for sub_name in tqdm(self.subjects):
            fit_file = f'{self.path_smpl}/{sub_name}/smplx_param.pkl'
            fy_scale = f'{self.path_smpl}/{sub_name}/scale.npy'
            fvmed = f'{self.path_smpl}/{sub_name}/vmed.npy'
            y_scale = np.load(fy_scale)
            vmed = np.load(fvmed)
            rescale_fitted_body, joints = load_fit_body(fit_file,
                                            y_scale,
                                            smpl_type='smplx',
                                            smpl_gender='male')
            rescale_fitted_body.apply_translation(-vmed)
            smpl_vertices = rescale_fitted_body.vertices
            smpl_faces = rescale_fitted_body.faces

            smpl_faces = smpl_faces[~SMPLX().smplx_eyeball_fid] # remove eyeball
            mouth_faces = (SMPLX().smplx_mouth_fid) # cover mouth
            smpl_faces = np.concatenate((smpl_faces, mouth_faces), axis=0)
            
            smpl_faces_all[sub_name] = np.array(smpl_faces)
            smpl_vertices_all[sub_name] = np.array(smpl_vertices)
            
        return smpl_faces_all, smpl_vertices_all

    def load_pts(self):
        from tqdm import tqdm
        space_pts = {}
        surface_pts = {}
        space_smpl_sdf = {}
        surface_smpl_sdf = {}
        for sub_name in tqdm(self.subjects):
            space_pts[sub_name] = np.load(os.path.join(self.path_pts, sub_name, 'space.npy'))
            surface_pts[sub_name] = np.load(os.path.join(self.path_pts, sub_name, 'surface.npy'))
            if self.use_smpl:
                space_smpl_sdf[sub_name] = np.load(os.path.join(self.path_pts, sub_name, 'space_smpl_sdf.npy'))
                surface_smpl_sdf[sub_name] = np.load(os.path.join(self.path_pts, sub_name, 'surf_smpl_sdf.npy'))

        return space_pts, surface_pts, space_smpl_sdf, surface_smpl_sdf

    def load_sdf(self):
        from tqdm import tqdm
        space_sdf = {}
        for sub_name in tqdm(self.subjects):
            sdfs = np.load(os.path.join(self.path_sdf, sub_name, '%s_sdf.npz'%sub_name))
            uniform_sdfs = sdfs['uniform_sdfs']
            space_sdf[sub_name] = uniform_sdfs
        return space_sdf

        
    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) 
        # return n*36

    def __getitem__(self, index):
        '''
        
        '''
        # x = random.randint(1, 10)
        # print(x)
        
        # subject_id = index // ((len(self.yaw_list) * len(self.pitch_list)))
        # yid = index // len(self.pitch_list) % len(self.yaw_list)
        # pid = index % len(self.pitch_list)
        subject_id = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)
        pid = tmp // len(self.yaw_list)
        subject_name = self.subjects[subject_id]
        # print(subject_name)
        # print(np.random.randint(1, 10, 5 ))  
        data_item = {
            'name': subject_name,
            'mesh_path': os.path.join(self.path_obj, subject_name,  '%s.obj'%subject_name),
            'sid': subject_id,
            'yid': yid,
            'pid': pid,
            'b_min': self.b_min,
            'b_max': self.b_max,
        }


        if self.opt.color_aug:
            color_aug = ColorAug(self.opt.aug_bri, self.opt.aug_con, self.opt.aug_sat, self.opt.aug_hue)
            color_aug.generate()
        else:
            color_aug = None


        # data_item = self.get_data(subject_id, yid, pid)
        sub_name = self.subjects[subject_id]
        import time
        t0 = time.time()
        num_views= self.opt.num_views
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset)  % len(self.yaw_list)] for offset in range(num_views)]

        if self.is_train:
            # view_ids = np.random.choice(self.yaw_list, num_views, replace=False)
            if self.opt.random_multiview:
                # np.random.seed(int(time.time()))
                view_ids = [np.random.choice(range(i*360//num_views, (i+1)*360//num_views), 1)[0] for i in range(num_views)]
                
                # view_step = 360 // num_views
                # view_ids = [self.yaw_list[(yid + view_step * i + random.choice(list(range(-15, 15+1)))) % len(self.yaw_list)] for i in range(num_views)]
                
                
                np.random.shuffle(view_ids)

        render_item = self.get_render(subject_id, view_ids, color_aug=color_aug, no_correct=True)
        data_item.update(render_item)

        if self.opt.use_smpl:
            smpl_faces, smpl_verts = self.smpl_faces[sub_name], self.smpl_verts[sub_name]
            triangles = smpl_verts[smpl_faces]
            smpl_item = {
                'smpl_faces' : smpl_faces,
                'smpl_verts' : smpl_verts,
                'triangles' : triangles,
            }
            data_item.update(smpl_item)

        if self.opt.use_spatial:
            smpl_joints = self.smpl_joints[sub_name]
            data_item.update({
                'smpl_joints' : torch.Tensor(smpl_joints).float(),
            })

        t1 = time.time()
        if not self.opt.offline_sample:
            if self.opt.load_mesh_ram:
                mesh = self.meshes[sub_name]
                scale = self.y_scales[sub_name]
            else:
                mesh = trimesh.load(os.path.join(self.path_obj, sub_name, '%s.obj'%sub_name))
                vmax = mesh.vertices.max(0) 
                vmin = mesh.vertices.min(0)
                y_scale = (vmax - vmin)[1]
                scale = y_scale / 180.0 * self.opt.sigma  # set gaussian radius to be rate of y_scale of model, default: sigma = 5cm vs modelsize = 180cm
            
            if self.verbose:
                print('render_item: %.3f' % (t1-t0))
            # data_item.update(render)
            if self.num_sample_inout:
                sample_pt = self.get_sample_points(mesh,scale, self.num_sample_inout)
                t = time.time()
                if self.verbose:
                    print('sample_pt: %.3f' % (t-t1))
                t1 = t
                data_item.update(sample_pt)
            if self.num_sample_surface:
                sample_surf_pts = self.get_surface_sample_points(mesh, self.num_sample_surface)
                t = time.time()
                if self.verbose:
                    print('sample_surf_pt: %.3f' % (t-t1))
                t1 = t
                data_item.update(sample_surf_pts)
            if self.num_sample_color:
                sample_color = self.get_sample_colors(subject_id, self.num_sample_color, render_item['view_ids'], pid)
                t = time.time()
                if self.verbose:
                    print('sample_color: %.3f' % (t-t1))
                data_item.update(sample_color)
            del mesh
        else:
            ## offline sample and read
            base_rot = np.eye(3)
            if os.path.exists(os.path.join(self.path_param, sub_name, 'base_rot.npy')):
                base_rot = np.load(os.path.join(self.path_param, sub_name, 'base_rot.npy'))
            if self.num_sample_inout:
                this_space_pt = self.space_pts[sub_name]
                random_index = np.random.randint(this_space_pt.shape[0], size=self.num_sample_inout)
                sample_pt = {
                    'samples': torch.Tensor(base_rot @ this_space_pt[random_index, :3].T).float(), # N3
                    'labels': torch.Tensor(this_space_pt[random_index, 3:].T).float() if self.opt.field_type=='sdf' else 1-torch.Tensor(this_space_pt[random_index, 3:].T).float(), # N1
                }

                if self.opt.use_gt_sdf:
                    sample_pt.update({'samples_sdf' : torch.Tensor(self.space_sdfs[sub_name][random_index]).float()})
                if self.use_smpl:
                    sample_pt.update({'samples_smpl_sdf' : torch.tensor(self.space_sdf[sub_name][random_index].T)})

                t = time.time()
                if self.verbose:
                    print('sample_pt: %.3f' % (t-t1))
                t1 = t
                data_item.update(sample_pt)
            
            if self.num_sample_surface:
                this_surface_pt = self.surface_pts[sub_name]
                random_index = np.random.randint(this_surface_pt.shape[0], size=self.num_sample_surface)
                surface_pt = {
                    'surface_samples': torch.Tensor(base_rot @ this_surface_pt[random_index, :3].T).float(), # N3 -> 3N
                    'surface_normals': torch.Tensor(base_rot @ this_surface_pt[random_index, 3:].T).float() # N3 [-1, 1] -> 3N
                }
                if self.use_smpl:
                    surface_pt.update({'surface_smpl_sdf' : torch.tensor(self.surface_sdf[sub_name][random_index].T)})
                t = time.time()
                if self.verbose:
                    print('sample_pt_surface: %.3f' % (t-t1))
                t1 = t
                data_item.update(surface_pt)

            if self.num_sample_color:
                sample_color = self.get_sample_colors(subject_id, base_rot, self.num_sample_color, render_item['view_ids'], pid)
                t = time.time()
                if self.verbose:
                    print('sample_color: %.3f' % (t-t1))
                data_item.update(sample_color)
            

        # print(data_item['samples'].sum())
        # print(data_item['labels'].sum())


            
        return data_item
        
    # spatial sampling of points in the bounding box and on the surface
    def get_sample_points(self, mesh,scale, n_sample_points):
        '''
        sample points from mesh surface (in model space) and the bounding box
        separate into inside/outside
        '''
        if not self.is_train: 
            set_random_seed(self.opt.seed)
        
        # 16:1 sampling rate for surface sampling and space uniform sampling
        
        surface_points, _ = trimesh.sample.sample_surface(mesh, n_sample_points)
        
        surface_points_with_pertubation = surface_points + np.random.normal(scale=scale, size=surface_points.shape)

        length = self.b_max - self.b_min
        space_points = np.random.rand(n_sample_points//8, 3) * length + self.b_min
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

        samples = np.concatenate((inside_points, outside_points), axis=0).T # [3, n_sample_points]

        if self.opt.field_type == 'sdf':
            labels =  np.concatenate((np.zeros((1, inside_points.shape[0])), np.ones((1, outside_points.shape[0]))), axis=1) #[1, n_sample_points]
        elif self.opt.field_type == 'occupancy':
            labels =  np.concatenate((np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))), axis=1) #[1, n_sample_points]


        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()
        return {
            'samples': samples,
            'labels': labels
        }

    def get_surface_sample_points(self, mesh, n_sample_points):
        '''
        sample points from mesh surface (in model space) and the bounding box
        separate into inside/outside
        '''
        if not self.is_train: 
            set_random_seed(self.opt.seed)
        
        # 16:1 sampling rate for surface sampling and space uniform sampling
        
        surface_points, face_ids = trimesh.sample.sample_surface(mesh, n_sample_points)
        surface_normals = mesh.face_normals[face_ids]
        pts_tensor = torch.from_numpy(surface_points).float().T # N3
        norms_tensor = torch.from_numpy(surface_normals).float().T # N3
        
        return {
            'surface_samples': pts_tensor,
            'surface_normals': norms_tensor
        }


    def get_sample_colors(self, subject_id, base_rot, n_sample_color, view_ids, pid=0):
        subject = self.subjects[subject_id]
        
        uv_mask_path = os.path.join(self.path_uv_mask, subject, '%02d.png' % (0))
        uv_pos_path = os.path.join(self.path_uv_pos, subject, '%02d.exr' % (0))
        uv_normal_path = os.path.join(self.path_uv_normal, subject, '%02d.png' % (0))
        uv_albedo_path = os.path.join(self.path_uv_albedo, subject, '%02d.jpg' % (0))
        uv_mask = cv2.imread(uv_mask_path)
        # [H, W] bool
        uv_mask = uv_mask[:,:,0] != 0
        # [H, W, 3] 0 ~ 1 float rgb
        uv_albedo = cv2.imread(uv_albedo_path)
        uv_albedo = cv2.cvtColor(uv_albedo, cv2.COLOR_BGR2RGB) / 255.0
        
        # [H, W, 3] -1 ~ 1 float
        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = uv_normal * 2.0 - 1.0 


        # [H, W, 3] Position render. each pixel is the xyz coordinates of the point
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]

        uv_mask = uv_mask.reshape((-1))
        uv_pos = uv_pos.reshape((-1, 3))
        uv_albedo = uv_albedo.reshape((-1, 3))
        uv_normal = uv_normal.reshape((-1, 3))

        surface_points = uv_pos[uv_mask]
        surface_albedos = uv_albedo[uv_mask]
        surface_normal = uv_normal[uv_mask]
        sample_list = random.sample(range(0, surface_points.shape[0] - 1), n_sample_color)
        surface_points = surface_points[sample_list].T
        surface_normal = surface_normal[sample_list].T
        surface_albedos = surface_albedos[sample_list].T
        rgbs_albedo_color = torch.Tensor(surface_albedos).float()

        normal = torch.Tensor(surface_normal).float()
        samples = torch.Tensor(surface_points).float() \
                    + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma / 180).expand_as(normal) * normal
        return {
            'color_samples': torch.tensor(base_rot).float() @ samples,
            # 'surface_normal': normal,
            # 'surface_shading': render_shadings,
            # 'surface_color': render_colors,
            'surface_albedo': rgbs_albedo_color
        }


        

    # get rendered image and calibration info, with optional augmentation
    def get_render(self, subject_id, view_ids, color_aug=None, no_correct=False):
        '''
        Return the render data
        :param subject: subject name
        :param yid: the first view_id. If None, select a random one.
        :param pid: the first view_id. If None, select a random one.
        :return:
            'img': [C, W, H] images
            'calib': [4, 4] calibration matrix
            'extrinsic': [4, 4] extrinsic matrix
            'mask': [1, W, H] masks'''
        subject = self.subjects[subject_id]

        calib_list = []
        render_list = []
        albedo_list = []
        normal_list = []
        mask_list = []
        dist_list = []
        quat_list = []
        extrinsic_list = []
        intrinsic_list = []
        for vid in view_ids:
            extr_path = os.path.join(self.path_param, subject, '%d_extrinsic.npy' % (vid))
            intr_path = os.path.join(self.path_param, subject, '%d_intrinsic.npy' % (vid))
            render_path = os.path.join(self.path_lit, subject, '%d.png' % (vid))
            albedo_path = os.path.join(self.path_albedo, subject, '%d.png' % (vid))
            normal_path = os.path.join(self.path_normal, subject, '%d.png' % (vid))
            mask_path = os.path.join(self.path_mask, subject, '%d.png' % (vid))
            # loading calibration data
            extrinsic = np.eye(4)
            intrinsic = np.eye(4)
            
            extrinsic[:3,:4] = np.load(extr_path)
            intrinsic[:3,:3] = np.load(intr_path)
            quat = get_quat_from_world_mat_np(extrinsic) #[7]


            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')
            albedo = Image.open(albedo_path).convert('RGB')
            normal = Image.open(normal_path).convert('RGB')

            dist = cv2.distanceTransform(255 - np.array(mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            dist = cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)
            dist = Image.fromarray(dist * 255)
            w, h = render.size
            scale_w = self.load_size / w
            scale_h = self.load_size / h

            S = self.load_size
            if (not no_correct):
                fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
                x_min, x_max, y_min, y_max = find_border(mask)
                y_min -= 20
                y_max += 20
                y_len = y_max - y_min
                x_min = (x_max + x_min - y_len) // 2
                x_max = x_min + y_len
                scale = S / y_len

                fx = fx * scale
                fy = fy * scale
                cx = scale * (cx - x_min)
                cy = scale * (cy - y_min)
                intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2] = fx, fy, cx, cy


            # intrinsic[0, 0] *= scale_w
            # intrinsic[1, 1] *= scale_h
            # intrinsic[0, 2] *= scale_w
            # intrinsic[1, 2] *= scale_h

            flip_y = True
            if flip_y:
                intrinsic[1, :] *= -1
                intrinsic[1, 2] += S

            
            if (not no_correct):
                imgs_list = [render, mask, normal, albedo, dist]
                for i, img in enumerate(imgs_list):
                    img = ImageOps.expand(img, S // 2, fill=0)
                    imgs_list[i] = (img.crop((x_min + S // 2, y_min + S // 2, x_max + S // 2, y_max + S // 2))).resize((S, S), Image.BILINEAR)
                render, mask, normal, albedo, dist = imgs_list
                # if self.opt.input_4k:
                #     RS = self.opt.ori_resolution
                #     ratio = RS // S
                #     hr_img = hr_img.resize((RS, RS), Image.BILINEAR)
                #     hr_img = ImageOps.expand(hr_img, RS // 2, fill=0)
                #     hr_img = (hr_img.crop((ratio*(x_min + S // 2),
                #             ratio*(y_min + S // 2), ratio*(x_max + S // 2), ratio*(y_max + S // 2)))).resize((S, S), Image.BILINEAR)


            if color_aug is not None:
                render = color_aug.apply(render)
            if self.aug_blur > 0.00001:
                blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                render = render.filter(blur)
            

            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()

            intrinsic = torch.Tensor(intrinsic).float()
            extrinsic = torch.Tensor(extrinsic).float()

            quat = torch.Tensor(quat).float()
            mask = self.mask2tensor(mask)  # [1, H, W]
            dist = self.mask2tensor(dist)  # [1, H, W]
            
            render = self.rgb2tensor(render) # [3, H, W]
            # albedo = F.adjust_brightness(albedo, 0.5)
            albedo = self.rgb2tensor(albedo) # [3, H, W]
            normal = self.rgb2tensor(normal)
            render = mask.expand_as(render) * render
            albedo = mask.expand_as(albedo) * albedo
            normal = mask.expand_as(normal) * normal

            render_list.append(render)
            albedo_list.append(albedo)
            normal_list.append(normal)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)
            intrinsic_list.append(intrinsic)
            quat_list.append(quat)
            mask_list.append(mask)
            dist_list.append(dist)


        imgs = torch.stack(render_list, dim=0)
        albedos = torch.stack(albedo_list, dim=0)
        normals = torch.stack(normal_list, dim=0)
        calibs = torch.stack(calib_list, dim=0)
        extrinsics = torch.stack(extrinsic_list, dim=0)
        quats = torch.stack(quat_list, dim=0)
        intrinsics = torch.stack(intrinsic_list, dim=0)
        masks = torch.stack(mask_list, dim=0)
        dists = torch.stack(dist_list, dim=0)
        z_center = torch.zeros(3, 1)
        return {
            'view_ids': view_ids,
            'z_center': z_center,
            # 'img': albedos,
            'img': imgs,
            'albedo': albedos,
            'normal':normals,
            'calib': calibs,
            'extrinsic': extrinsics,
            'quat': quats,
            'intrinsic': intrinsics,
            'mask': masks,
            'dist': dists
        }


if __name__ == '__main__':
    from options.options import get_options
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
    print(sid)
    print(yid)
    print(calib)
    print('mask: ', mask)
    print(mask.dtype)
    print(mask.shape)

    print(extrinsic)

    print('samples: ')
    print(samples)
    print(samples.shape)
    print(labels)
    print('imgs: ')
    print(img.shape)
    
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('agg')

    K = img.shape[0]
    plt.figure()
    for k in range(K):
        calib_k = calib[k]
        img_k = np.uint8((np.transpose(img[k].numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.0)
        rot = calib_k[:3, :3] # [3,3]
        trans = calib_k[:3, 3:4] # [3,1]\
        print('aaa')

        surface_samples_w = torch.cat((surface_samples, torch.ones((1,5000))), dim=0)
        # pts = rot @ surface_samples + trans 
        pts = calib_k @ surface_samples_w
        pts[:3,:] = pts[:3,:] / pts[3:,:]
        print(pts.T)
        print(pts.shape)
        xy = pts[:2,:] * 0.5 + 0.5  # [2, N]
        
        plt.imshow(img_k)
        plt.scatter(xy[0, :]*512, xy[1, :]*512,s=1, marker='*')
        plt.show()
        plt.savefig( '/home/lujiawei/workspace/dataset/my_%s_%d.jpg'%(name, view_ids[k]))
        plt.clf()


    # save_img = (np.transpose(img[0].numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.0
    # save_path = '/home/lujiawei/workspace/dataset/my_%s_%d.jpg'%(name, yid)
    # Image.fromarray(np.uint8(save_img)).save(save_path)

