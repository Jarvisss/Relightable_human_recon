from turtle import back
import matplotlib
from pip import main
import torch
import torch.nn.functional as F
from .sdf_utils import *
from skimage import measure
from PIL import Image
from utils.vis_utils import vis_sdf_x_plane,vis_sdf_y_plane,vis_sdf_z_plane
from utils.geo_utils import perspective
from utils.uv_utils import grid_sample
# from utils.grid_sample_gradfix import grid_sample
from utils.common_utils import img_l2g, shading_l2g
from utils.camera import KRT_from_P
import cv2


def SchlickFresnel(u):
    m = torch.clamp(1-u, 0, 1)
    m2 = m * m
    return m2 * m2 * m


# as V=L=H, this can be simplified a lot
def diffuse_fresnel_brdf(N, V, roughness):
    '''
    N: [B,3,N]
    V: [B,3,N]
    roughness: [B,1,N]
    '''
    NdotL = torch.clamp( torch.sum(N * V, dim=1), 0, 1) 
    HdotL = torch.clamp( torch.sum(V * V, dim=1), 0, 1) 
    NdotV = NdotL
    FL = SchlickFresnel(NdotL)
    FV = SchlickFresnel(NdotV)
    Fd90 = 0.5 + 2 * HdotL * HdotL * roughness
    # fd = 
    
    
    pass


def specular_brdf():
    pass