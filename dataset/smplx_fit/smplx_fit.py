import numpy as np
import trimesh
import torch
import smplx
import os.path as osp
from scipy.spatial import cKDTree


class SMPLX():
    def __init__(self):
        
        self.current_dir = '/mnt/data1/lujiawei/smpl_related'

        self.smpl_verts_path = osp.join(self.current_dir,
                                        "smpl_data/smpl_verts.npy")
        self.smplx_verts_path = osp.join(self.current_dir,
                                         "smpl_data/smplx_verts.npy")
        self.faces_path = osp.join(self.current_dir,
                                   "smpl_data/smplx_faces.npy")
        self.cmap_vert_path = osp.join(self.current_dir,
                                       "smpl_data/smplx_cmap.npy")

        self.smplx_eyeball_fid = osp.join(self.current_dir,
                                          "smpl_data/eyeball_fid.npy")
        self.smplx_fill_mouth_fid = osp.join(self.current_dir,
                                             "smpl_data/fill_mouth_fid.npy")

        self.faces = np.load(self.faces_path)
        self.verts = np.load(self.smplx_verts_path)
        self.smpl_verts = np.load(self.smpl_verts_path)

        self.smplx_eyeball_fid = np.load(self.smplx_eyeball_fid)
        self.smplx_mouth_fid = np.load(self.smplx_fill_mouth_fid)

        self.model_dir = osp.join(self.current_dir, "models")
        self.tedra_dir = osp.join(self.current_dir, "../tedra_data")

    def get_smpl_mat(self, vert_ids):

        mat = torch.as_tensor(np.load(self.cmap_vert_path)).float()
        return mat[vert_ids, :]

    def smpl2smplx(self, vert_ids=None):
        """convert vert_ids in smpl to vert_ids in smplx
        Args:
            vert_ids ([int.array]): [n, knn_num]
        """
        smplx_tree = cKDTree(self.verts, leafsize=1)
        _, ind = smplx_tree.query(self.smpl_verts, k=1)  # ind: [smpl_num, 1]

        if vert_ids is not None:
            smplx_vert_ids = ind[vert_ids]
        else:
            smplx_vert_ids = ind

        return smplx_vert_ids

    def smplx2smpl(self, vert_ids=None):
        """convert vert_ids in smplx to vert_ids in smpl
        Args:
            vert_ids ([int.array]): [n, knn_num]
        """
        smpl_tree = cKDTree(self.smpl_verts, leafsize=1)
        _, ind = smpl_tree.query(self.verts, k=1)  # ind: [smplx_num, 1]
        if vert_ids is not None:
            smpl_vert_ids = ind[vert_ids]
        else:
            smpl_vert_ids = ind

        return smpl_vert_ids

model_init_params = dict(
    gender='male',
    model_type='smplx',
    model_path=SMPLX().model_dir,
    create_global_orient=False,
    create_body_pose=False,
    create_betas=False,
    create_left_hand_pose=False,
    create_right_hand_pose=False,
    create_expression=False,
    create_jaw_pose=False,
    create_leye_pose=False,
    create_reye_pose=False,
    create_transl=False,
    num_pca_comps=12)


def get_smpl_model(model_type, gender): return smplx.create(
    **model_init_params)

def load_fit_body(fitted_path, scale, trans, smpl_type='smplx', smpl_gender='neutral', noise_dict=None):

    param = np.load(fitted_path, allow_pickle=True)
    for key in param.keys():
        param[key] = torch.as_tensor(param[key])

    smpl_model = get_smpl_model(smpl_type, smpl_gender)
    model_forward_params = dict(betas=param['betas'],
                                global_orient=param['global_orient'],
                                body_pose=param['body_pose'],
                                left_hand_pose=param['left_hand_pose'],
                                right_hand_pose=param['right_hand_pose'],
                                jaw_pose=param['jaw_pose'],
                                leye_pose=param['leye_pose'],
                                reye_pose=param['reye_pose'],
                                expression=param['expression'],
                                return_verts=True)

    if noise_dict is not None:
        model_forward_params.update(noise_dict)

    smpl_out = smpl_model(**model_forward_params)

    smpl_verts = (
        (smpl_out.vertices[0] * param['scale'] + param['translation'] + trans) * scale).detach()
    smpl_joints = (
        (smpl_out.joints[0] * param['scale'] + param['translation'] + trans) * scale).detach()
    smpl_mesh = trimesh.Trimesh(smpl_verts,
                                smpl_model.faces,
                                process=False, maintain_order=True)

    return smpl_mesh, smpl_joints

