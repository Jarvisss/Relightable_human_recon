import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


R0 = 0.04
# R0 = 1
EPS = 1e-4


###################
#
#  Helper Functions
#
###################

def FresnelApproximate(R0,cosTheta):
    a = 1. - torch.clamp(cosTheta, 0., 1.)
    a2 = a * a
    return (a2 * a2 * a) * (1-R0) + R0


def smithG1(cosTheta, alpha):
    sinTheta = torch.sqrt(1.0 - cosTheta * cosTheta)
    tanTheta = sinTheta / (cosTheta + 1e-10)
    root = alpha * tanTheta
    return 2.0 / (1.0 + torch.hypot(root, torch.ones_like(root)))


def G1_(w,alphax,alphay):
    '''
    w is normalized
    '''
    vv = torch.stack([
        w[...,0] * alphax,
        w[...,1] * alphay,
        w[...,2]],dim=-1 )
    ## NOTICE !! potential numerical problem
    vv_mag = torch.sqrt(torch.sum(vv*vv, dim=-1))
    return (2. * w[...,2]) / (w[...,2] + vv_mag)

def G1(wi,wo,alphax,alphay):
    return G1_(wi,alphax,alphay) * G1_(wo,alphax,alphay)

def TrowbridgeReitzDistribution(wh,alphax,alphay):
    ## NOTICE !! potential numerical problem
    vv = torch.stack([
        wh[...,0] / alphax,
        wh[...,1] / alphay,
        wh[...,2]],dim=-1 )
    len2 = torch.sum(vv*vv, dim=-1)
    return 1.0/(np.pi * alphax * alphay * len2 * len2)


def _apply_shading_burley(points, normals, view_dirs, light_dirs, irradiance, brdf_params, brdf_config):
    normals = F.normalize(normals, dim=-1)
    light_dirs_ = F.normalize(light_dirs, dim=-1)
    view_dirs = F.normalize(view_dirs, dim=-1)


    falloff = F.relu(-(normals * light_dirs_).sum(-1)) # (...)
    forward_facing = dot(normals, view_dirs) < 0
    visible_mask = ((falloff > 0) & forward_facing) # (...) boolean
    falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (...) cosine falloff, 0 if not visible
    irradiance = torch.unsqueeze(falloff, dim=-1) * irradiance  # (..., 3) or (..., 1)

    diffuse, non_diffuse = _burley_shading(normals, -light_dirs_, -view_dirs, brdf_params, brdf_config)
    return diffuse*irradiance, non_diffuse*irradiance

# https://github.com/za-cheng/WildLight/blob/main/models/physicalshader.py#L111
def _diffuse(dot, roughness):
    F_D90 = 2*roughness + 0.5
    base_diffuse = (1 + (F_D90 - 1)*(1-dot)**5)**2 / np.pi

    return base_diffuse

def _GGX_smith(dot, roughness, epsilon=1e-10):
    hz_sq = dot**2
    roughness_sq = roughness**2
    D = roughness_sq / np.pi / (hz_sq * (roughness_sq-1) + 1 + epsilon)**2 # GGX
    G = 2 / ( torch.sqrt(1 + roughness_sq * (1/hz_sq - 1)) + 1)

    return D, G

def eval_collocate_burley(dot, roughness, base_color, metallic, specular):
    '''
    for collocate light setting, we have wi == wo == wh
    :param dot: [N, 1]
    :param roughness: [N, 1]
    :param base_color: [N, 3]
    :param metallic: [N, 1]
    :param specular: [N, 1]

    '''
    # wo = wi
    # wh = wi
    # dot = torch.sum(wi * wn, dim=-1, keepdims=True)
    # dot = torch.clamp(dot, min=0.00001, max=0.99999)  # must be very precise; cannot be 0.999
    alpha = 0.0001 + (roughness**2) * (1-0.0002)
    alpha = torch.clamp(alpha, min=0.0001)
    dot2 = dot * dot
    print(specular.max())
    print(metallic.max())
    D_metal, G_metal = _GGX_smith(dot, alpha) #(..., 1)
    F_metal = (1-metallic)*specular*0.08 + metallic*base_color # (..., 3)

    r_diffuse = _diffuse(dot, roughness) * base_color
    print(max(_diffuse(dot, roughness)))
    r_specular = D_metal * G_metal * F_metal / (4 * dot2) # (..., 3)

    return  (1-metallic)*r_diffuse, r_specular
    

