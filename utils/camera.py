import cv2
import numpy as np

from .glm import ortho
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as SPR
import pdb

class Camera:
    def __init__(self, width=1600, height=1200, fov=60, near=1, far=500):
        # Focal Length
        # equivalent 50mm
        fov = np.radians(fov)
        self.focal_x = width /  (np.tan(fov/2) * 2)
        self.focal_y = height /  (np.tan(fov/2) * 2)
        # Principal Point Offset
        self.principal_x = width / 2
        self.principal_y = height / 2
        # Axis Skew
        self.skew = 0
        # Image Size
        self.width = width
        self.height = height

        self.near = near
        self.far = far

        # Camera Center
        self.center = np.array([0, 0, 250])
        self.direction = np.array([0, 0, -1])
        self.right = np.array([1, 0, 0])
        self.up = np.array([0, 1, 0])

        self.ortho_ratio = None

    def sanity_check(self):
        self.center = self.center.reshape([-1])
        self.direction = self.direction.reshape([-1])
        self.right = self.right.reshape([-1])
        self.up = self.up.reshape([-1])

        assert len(self.center) == 3
        assert len(self.direction) == 3
        assert len(self.right) == 3
        assert len(self.up) == 3

    @staticmethod
    def normalize_vector(v):
        v_norm = np.linalg.norm(v)
        return v if v_norm == 0 else v / v_norm

    def get_real_z_value(self, z):
        z_near = self.near
        z_far = self.far
        z_n = 2.0 * z - 1.0
        z_e = 2.0 * z_near * z_far / (z_far + z_near - z_n * (z_far - z_near))
        return z_e

    def get_rotation_matrix(self):
        rot_mat = np.eye(3)
        s = self.right
        s = self.normalize_vector(s)
        rot_mat[0, :] = s
        u = self.up
        u = self.normalize_vector(u)
        rot_mat[1, :] = -u
        rot_mat[2, :] = self.normalize_vector(self.direction)

        return rot_mat

    def get_translation_vector(self):
        rot_mat = self.get_rotation_matrix()
        trans = -np.dot(rot_mat, self.center)
        return trans

    def get_intrinsic_matrix(self):
        int_mat = np.eye(3)

        int_mat[0, 0] = self.focal_x
        int_mat[1, 1] = self.focal_y
        int_mat[0, 1] = self.skew
        int_mat[0, 2] = self.principal_x
        int_mat[1, 2] = self.principal_y

        return int_mat

    def get_projection_matrix(self):
        ext_mat = self.get_extrinsic_matrix()
        int_mat = self.get_intrinsic_matrix()

        return np.matmul(int_mat, ext_mat)

    def get_extrinsic_matrix(self):
        rot_mat = self.get_rotation_matrix()
        trans = self.get_translation_vector()

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rot_mat
        extrinsic[:3, 3] = trans

        return extrinsic[:3, :]

    def set_rotation_matrix(self, rot_mat):
        self.direction = rot_mat[2, :]
        self.up = -rot_mat[1, :]
        self.right = rot_mat[0, :]

    def set_intrinsic_matrix(self, int_mat):
        self.focal_x = int_mat[0, 0]
        self.focal_y = int_mat[1, 1]
        self.skew = int_mat[0, 1]
        self.principal_x = int_mat[0, 2]
        self.principal_y = int_mat[1, 2]

    def set_projection_matrix(self, proj_mat):
        res = cv2.decomposeProjectionMatrix(proj_mat)
        int_mat, rot_mat, camera_center_homo = res[0], res[1], res[2]
        camera_center = camera_center_homo[0:3] / camera_center_homo[3]
        camera_center = camera_center.reshape(-1)
        int_mat = int_mat / int_mat[2][2]

        self.set_intrinsic_matrix(int_mat)
        self.set_rotation_matrix(rot_mat)
        self.center = camera_center

        self.sanity_check()

    def get_gl_matrix(self):
        z_near = self.near
        z_far = self.far
        rot_mat = self.get_rotation_matrix()
        int_mat = self.get_intrinsic_matrix()
        trans = self.get_translation_vector()

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rot_mat
        extrinsic[:3, 3] = trans
        axis_adj = np.eye(4)
        axis_adj[2, 2] = -1
        axis_adj[1, 1] = -1
        model_view = np.matmul(axis_adj, extrinsic)

        projective = np.zeros([4, 4])
        projective[:2, :2] = int_mat[:2, :2]
        projective[:2, 2:3] = -int_mat[:2, 2:3]
        projective[3, 2] = -1
        projective[2, 2] = (z_near + z_far)
        projective[2, 3] = (z_near * z_far)

        if self.ortho_ratio is None:
            ndc = ortho(0, self.width, 0, self.height, z_near, z_far)
            perspective = np.matmul(ndc, projective)
        else:
            perspective = ortho(-self.width * self.ortho_ratio / 2, self.width * self.ortho_ratio / 2,
                                -self.height * self.ortho_ratio / 2, self.height * self.ortho_ratio / 2,
                                z_near, z_far)

        return perspective, model_view


def quat_to_rot(q):
    ## cal in double
    ## q in [x,y,z,w] order
    # r = SPR.from_quat(q)
    # return r.as_matrix()
    device = q.device
    batch_size, _ = q.shape
    # q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).to(device)

    for k in range(q.shape[0]):
        qx = q[k, 0]
        qy = q[k, 1]
        qz = q[k, 2]
        qw = q[k, 3]
        R[k, 0, 0] = 2 * (qw * qw + qx * qx) - 1
        R[k, 0, 1] = 2 * (qx * qy - qw * qz)
        R[k, 0, 2] = 2 * (qx * qz + qw * qy)
        R[k, 1, 0] = 2 * (qx * qy + qw * qz)
        R[k, 1, 1] = 2 * (qw * qw + qy * qy) - 1
        R[k, 1, 2] = 2 * (qy * qz - qw * qx)
        R[k, 2, 0] = 2 * (qx * qz - qw * qy)
        R[k, 2, 1] = 2 * (qy * qz + qw * qx)
        R[k, 2, 2] = 2 * (qw * qw + qz * qz) - 1
    return R


def get_calib_extri_from_pose(pose, intrinsics):
    '''
    Torch func
    Params
    :@ pose [BK, 7], float64
    :@ intrinsics [BK, 4, 4], float64 
    
    Return
    :@ calibs [BK, 4,4]
    :@ extris [BK, 4,4]
    '''
    device = pose.device
    cam_locs_in_model_space = pose[:, 4:]
    R = quat_to_rot(pose[:,:4])
    extri_inv = torch.eye(4).repeat(pose.shape[0],1,1).float().to(device)
    extri_inv[:, :3, :3] = R
    extri_inv[:, :3, 3] = cam_locs_in_model_space

    extri = torch.eye(4).repeat(pose.shape[0],1,1).float().to(device)
    extri[:, :3, :3] = R.transpose(1,2)
    extri[:, :3, 3:] = torch.bmm(-R.transpose(1,2) ,cam_locs_in_model_space.unsqueeze(-1))

    calibs = torch.eye(4).repeat(pose.shape[0],1,1).float().to(device)
    calibs = torch.bmm(intrinsics, extri)
    return calibs.float(), extri.float(), extri_inv.float()

def get_quat_from_world_mat_np(w2c_mat):
    '''
    !! input matrix should be a rotation matrix!!
    input the world matrice, output c2w quat
    '''
    R = w2c_mat[:3,:3]
    t = w2c_mat[:3, 3]
    
    R_T = R.transpose()
    t_T = -R_T@t
    c2w = np.eye(4)
    c2w[:3, :3] = R_T
    c2w[:3, 3] = t_T
    
    import pdb
    # init_quat = rot_to_quat_np(c2w)
    init_quat = rot_to_quat_np_stable(c2w)
    quat = np.concatenate([init_quat, c2w[:3,3]], 0)
    return quat


def rot_to_quat_np_stable(R):
    # r = SPR.from_matrix(R)
    # return r.as_quat()
    _,_ = R.shape
    q = np.ones(4)
    R00 = R[0, 0]
    R01 = R[0, 1]
    R02 = R[0, 2]
    R10 = R[1, 0]
    R11 = R[1, 1]
    R12 = R[1, 2]
    R20 = R[2, 0]
    R21 = R[2, 1]
    R22 = R[2, 2]
    trace = R00+R11+R22

    if trace > 0:
        t = np.sqrt(1.0+trace)
        q[3]=0.5 * t
        t = 0.5 / t
        q[0]=(R21-R12) * t
        q[1] = (R02 - R20) * t
        q[2] = (R10 - R01) * t
    else:
        i = 0
        if (R11 > R00):
            i = 1
        if (R22 > R11):
            i = 2
        j = (i+1) % 3
        k = (j+1) % 3

        t = np.sqrt(R[i,i] - R[j,j] - R[k,k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (R[k,j] - R[j,k]) * t
        q[j] = (R[j,i] + R[i,j]) * t
        q[k] = (R[k,i] + R[i,k]) * t
    # print(q[0])
    # print(q[1])
    # print(q[2])
    # print(q[3])
    
    return q

# def rot_to_quat_2(m):
#     t = np.matrix.trace(m)
#     q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

#     if(t > 0):
#         t = np.sqrt(t + 1)
#         q[0] = 0.5 * t
#         t = 0.5/t
#         q[1] = (m[2,1] - m[1,2]) * t
#         q[2] = (m[0,2] - m[2,0]) * t
#         q[3] = (m[1,0] - m[0,1]) * t

#     else:
#         i = 0
#         if (m[1,1] > m[0,0]):
#             i = 1
#         if (m[2,2] > m[i,i]):
#             i = 2
#         j = (i+1)%3
#         k = (j+1)%3

#         t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
#         q[i] = 0.5 * t
#         t = 0.5 / t
#         q[0] = (m[k,j] - m[j,k]) * t
#         q[j] = (m[j,i] + m[i,j]) * t
#         q[k] = (m[k,i] + m[i,k]) * t

#     return q



def get_matrices_by_pose(pose, intrinsic):
    


    return calibs, extrinsic


def KRT_from_P(proj_mat, normalize_K=True):
    res = cv2.decomposeProjectionMatrix(proj_mat)
    K, Rot, camera_center_homog = res[0], res[1], res[2]
    camera_center = camera_center_homog[0:3] / camera_center_homog[3]
    trans = -Rot.dot(camera_center)
    if normalize_K:
        K = K / K[2][2]
    return K, Rot, trans

def MVP_from_P_colmap(proj_mat, width, height, near=0.1, far=10000):
    '''
    Convert OpenCV camera calibration matrix to OpenGL projection and model view matrix
    :param proj_mat: OpenCV camera projeciton matrix
    :param width: Image width
    :param height: Image height
    :param near: Z near value
    :param far: Z far value
    :return: OpenGL projection matrix and model view matrix
    '''
    res = cv2.decomposeProjectionMatrix(proj_mat)
    K, Rot, camera_center_homog = res[0], res[1], res[2]
    camera_center = camera_center_homog[0:3] / camera_center_homog[3]
    print('camera_center', camera_center)
    print('camera_dist', np.linalg.norm(camera_center))
    trans = -Rot.dot(camera_center)

    K = K / K[2][2]

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rot
    extrinsic[:3, 3:4] = trans
    axis_adj = np.eye(4)
    axis_adj[2, 2] = -1
    axis_adj[1, 1] = -1
    model_view = np.matmul(axis_adj, extrinsic)


    zFar = far
    zNear = near
    projective = np.zeros([4, 4])
    projective[:2, :2] = K[:2, :2]
    projective[:2, 2:3] = -K[:2, 2:3]
    projective[3, 2] = -1
    projective[2, 2] = (zNear + zFar)
    projective[2, 3] = (zNear * zFar)

    ndc = ortho(0, width, 0, height, zNear, zFar)

    perspective = np.matmul(ndc, projective)

    return perspective, model_view

def MVP_from_P(proj_mat, width, height, near=0.1, far=10000):
    '''
    Convert OpenCV camera calibration matrix to OpenGL projection and model view matrix
    :param proj_mat: OpenCV camera projeciton matrix
    :param width: Image width
    :param height: Image height
    :param near: Z near value
    :param far: Z far value
    :return: OpenGL projection matrix and model view matrix
    '''
    res = cv2.decomposeProjectionMatrix(proj_mat)
    K, Rot, camera_center_homog = res[0], res[1], res[2]
    camera_center = camera_center_homog[0:3] / camera_center_homog[3]
    # print('camera_center', camera_center)
    print('camera_dist', np.linalg.norm(camera_center))
    trans = -Rot.dot(camera_center)
    K = K / K[2][2]

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rot
    extrinsic[:3, 3:4] = trans
    axis_adj = np.eye(4)
    axis_adj[2, 2] = -1 ## opencv to opengl convert [view mat, revert y and z axis]
    axis_adj[1, 1] = -1
    model_view = np.matmul(axis_adj, extrinsic)

    quat = get_quat_from_world_mat_np(model_view)

    zFar = far
    zNear = near
    perspective = np.zeros([4, 4])
    perspective[:2, :2] = K[:2, :2]
    perspective[:2, 2:3] = -K[:2, 2:3]
    perspective[3, 2] = -1
    perspective[2, 2] = (zNear + zFar)
    perspective[2, 3] = (zNear * zFar)

    ndc = ortho(0, width, 0, height, zNear, zFar)

    projective = np.matmul(ndc, perspective)


    return projective, model_view, quat
