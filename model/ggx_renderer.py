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


def eval_collocate_ggx(wn, wi,dot, alpha):
    '''
    for collocate light setting, we have wi == wo == wh
    :param wn: [N, 3]
    :param wi: [N, 3]
    :param dot: [N, 1]
    :param alpha: [N, 1]

    '''
    # wo = wi
    # wh = wi
    # dot = torch.sum(wi * wn, dim=-1, keepdims=True)
    # dot = torch.clamp(dot, min=0.00001, max=0.99999)  # must be very precise; cannot be 0.999
    alpha = torch.clamp(alpha, min=0.0001)
    cosTheta2 = dot * dot
    root = cosTheta2 + (1.0 - cosTheta2) / (alpha * alpha + 1e-10)

    D = 1.0 / (np.pi * alpha * alpha * root * root + 1e-10)
    # F = FresnelApproximate(R0, cosTheta=dot) ## cosTheta==1 as dot(wo, wh)=1 
    F = R0 ## cosTheta==1 as dot(wo, wh)=1 
    G = smithG1(dot, alpha) ** 2

    # D = TrowbridgeReitzDistribution(wh, alpha,alpha)
    # G = G1(wi, wi, alpha, alpha)

    # spec_component = (F * G * D) / (4. * dot + 1e-10)
    spec_component = (F * G * D) / (4. * cosTheta2 + 1e-10)
    return spec_component
    

