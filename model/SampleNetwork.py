
import torch.nn as nn
import torch
import pdb

class SampleNetwork(nn.Module):
    '''
    Represent the intersection (sample) point as differentiable function of the implicit geometry and camera parameters.
    See equation 3 in the paper for more details.
    '''

    def forward(self, surface_output, surface_points_grad, surface_dists, surface_cam_loc, surface_ray_dirs):
        '''
        Params:
            surface_output, # not detached [N']
            surface_points_grad, # detached [N', 3]
            surface_dists, # no grad[N', 1]
            surface_cam_loc, # leaf node const[N', 3]
            surface_ray_dirs, # leaf node const[N', 3]
        
        Return:
            surface_points_theta_c_v, # differentiable of surface_output [N', 3]
        '''
        # t -> t(theta)
        surface_ray_dirs_0 = surface_ray_dirs.detach()
        surface_points_dot = torch.bmm(surface_points_grad.view(-1, 1, 3),
                                       surface_ray_dirs_0.view(-1, 3, 1)).squeeze(-1)
        if torch.any(torch.isnan(surface_dists)):
            print('surface_dists cause nan')
        if torch.any(torch.isnan(surface_points_dot)):
            print('surface_points_dot cause nan')
        
        # [N', 1]
        surface_dists_theta = surface_dists - (surface_output - surface_output.detach()).unsqueeze(-1) / (surface_points_dot)
        if torch.any(torch.isnan(surface_dists_theta)):
            print('dot product division cause nan')
        # t(theta) -> x(theta,c,v)
        
        surface_points_theta_c_v = surface_cam_loc + surface_dists_theta * surface_ray_dirs

        return surface_points_theta_c_v
