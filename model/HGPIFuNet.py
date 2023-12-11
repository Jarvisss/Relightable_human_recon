import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from .BasePIFuNet import BasePIFuNet
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from .MLP import MLP

from utils.common_utils import init_net


class HGPIFuNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 num_views,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'hgpifu'

        self.opt = opt
        self.num_views = num_views
        self.image_filter = HGFilter(opt)

        last_op = nn.Sigmoid() if opt.field_type=='occupancy' else None

        self.surface_classifier = MLP(
            filter_channels=self.opt.mlp_dim, # [257, 1024, 512, 256, 128, 1]
            res_layers=self.opt.mlp_res_layers,
            mean_layer=self.opt.mlp_mean_layer,
            norm = self.opt.mlp_norm_type,
            activation=self.opt.mlp_activation_type,
            last_op=last_op,
            num_views=self.num_views,
            use_confidence= self.opt.use_confidence,
            use_feature_confidence = self.opt.use_feature_confidence
        )
            

        self.normalizer = DepthNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.weights_space = None

        self.intermediate_preds_list = []
        init_net(self)
        # init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            im_feat_list = [im_feat_list[-1]]

        return im_feat_list

    def query(self, im_feat_list, points, calibs, transforms=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 4, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''

        xyz = self.projection(points, calibs, transforms) # this projection is differentiable
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        B = points.shape[0] // self.num_views
        # [B, 1, N]
        z_feat = self.normalizer(z) # normalize z to [-1, 1] by dividing a scalar number
        
        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_occupancy_list = []

        for im_feat in im_feat_list: # we have totally num_stacks' feature tensor
            # [B, C_feat + 1, N]
            # [B, C_feat, N] cat [B, 1, N]
            point_local_feat_list = [self.index(im_feat, xy), z_feat]

            if self.opt.skip_hourglass:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)
            
            occupancy_field, _ = self.surface_classifier(point_local_feat)
            # out of image plane is always set to 0
            pred_occupancy = in_img[:B,None].float() * occupancy_field
            self.intermediate_preds_occupancy_list.append(pred_occupancy)

            

        preds = self.intermediate_preds_occupancy_list[-1]
        return preds, None
        

        
    def cal_sdf_grad(self, points, calibs, transforms=None):
        points.requires_grad = True
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 4, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        

        xyz = self.projection(points, calibs, transforms) # this projection is differentiable
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        # [B, 1, N]
        z_feat = self.normalizer(z) # normalize z to [-1, 1] by dividing a scalar number
        
        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)


        im_feat = self.im_feat_list[-1]
        point_local_feat_list = [self.index(im_feat, xy), z_feat]

        if self.opt.skip_hourglass:
            point_local_feat_list.append(tmpx_local_feature)

        point_local_feat = torch.cat(point_local_feat_list, 1)

        occupancy_field = self.surface_classifier(point_local_feat)
        
        # out of image plane is always set to 0

        pred_occupancy = in_img[:,None].float() * occupancy_field

        
        # scalar = self.error_term(occupancy_field, torch.zeros_like(occupancy_field))
        # # scalar =  occupancy_field.sum()
        # scalar.backward(retain_graph=True)
        # print(z_feat.grad)
        normal = -autograd.grad(
            occupancy_field,
            points,
            torch.ones_like(occupancy_field, requires_grad=False, device=occupancy_field.device),
            retain_graph=True,
            create_graph=True
        )[0]

        # print(normal.shape)
        # [B 3 N]
        normal = (normal + 1e-12) / torch.norm(normal + 1e-12, dim=1, keepdim=True)
        # print('mean normal:',normal.mean())
        # print('max normal:',normal.max())
        # print('min normal:',normal.min())
        # normal = self.estimate_normals(points, occupancy_field)
        self.normals = normal
        return normal

    
    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_preds(self):
        '''
        Get the predictions from the last query
        :return: [B, Res, N] network prediction for the last query
        :return the normal calculated from occupancy if required
        '''
        preds = {
            'pred_occupancy': self.preds 
        }
        
        return preds
    
    def get_occupancy_error(self, labels=None):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        if labels is None:
            return error
        for preds in self.intermediate_preds_occupancy_list:
            error += self.error_term(preds, labels)
        error /= len(self.intermediate_preds_occupancy_list)
        
        return error

    # only for surface points
    def get_normal_error(self, labels_normal=None):
        # print(self.normals)
        # print(self.labels_normal)
        if labels_normal is None:
            return 0
        if self.opt.use_cosine_loss:
            error = 1 - F.cosine_similarity(self.normals, labels_normal).mean()
        else:
            error = F.l1_loss(self.normals, labels_normal)
        return error




    
    def get_error(self, labels=None, labels_normal=None):
        '''
        we DO NOT have the ground truth normal of the unsurface points
        '''
        
        # for surface points
        # self.normal_error = self.get_normal_error(labels_normal)

        # for all points
        self.occupancy_error = self.get_occupancy_error(labels)

        return self.occupancy_error 
        # return self.occupancy_error
        # return self.normal_error

    def get_error_dict(self):
        return {
            'occu_error': self.occupancy_error,
        }

    def forward(self, images, points, calibs, transforms=None, labels_space=None, points_surf=None, labels_surf=None):
        # Get image feature
        K = self.num_views
        points_surf_stack = points_surf.repeat(K, 1, 1) # [BK, 3, N]
        points_space_stack = points.repeat(K, 1, 1) # [BK, 3, N]
        im_feat_list = self.filter(images)

        # Phase 2: point query
        preds,_ = self.query(im_feat_list=im_feat_list, points=points_space_stack, calibs=calibs, transforms=transforms)
        # get the error
        error = self.get_error(labels=labels_space, labels_normal=labels_surf)
        error_dict = self.get_error_dict()

        return error, error_dict