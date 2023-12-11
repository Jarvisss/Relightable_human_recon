import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as nnF

import math
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .attention import Encoder
from .DepthNormalizer import DepthNormalizer
from .unet import *

from .HGPIFuNet import HGFilter
# from .MVS_featureNet import FeatureNet
from .embedder import get_embedder
from utils.common_utils import init_net, gradient, safe_l2norm, linspace
from utils.render_utils import in_visual_hull
from .vgg import VGG19
from .spatial import SpatialEncoder

class UNet_unified(BasePIFuNet):
    '''
    UNet PIFu network uses UNet as the image filter.
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
                 base_views,
                 finetune=False,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(UNet_unified, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'unet_unified'

        self.opt = opt
        self.load_size = opt.load_size
        self.finetune = finetune
        self.eps = 1e-10
        self.sep_geo_brdf=False
        self.use_spatial=self.opt.use_spatial
        self.use_world_coord = self.opt.use_world_coord
        self.feed_mask = self.opt.feed_mask
        self.feed_bound = self.opt.feed_bound
        self.feed_dist = self.opt.feed_dist
        self.feed_dir = self.opt.feed_dir
        # self.sep_geo_brdf=True
        self.base_views = base_views
        self.use_poe = self.opt.use_positional_encoding
        self.is_train = opt.phase == 'train'

        self.in_channels = 3
        if self.feed_mask:
            self.in_channels+=1
        if self.feed_dist:
            self.in_channels+=1
        if self.feed_bound:
            self.in_channels+=1
        if self.feed_dir:
            self.in_channels+=3


        # if opt.filter_type == 'FPN':
        #     self.image_filter = FeatureNet()

        if opt.filter_type == 'HHG':
            # do not down sample
            self.image_filter = HGFilter(
                opt,
                highres=True
            )
            n_down = 7

        if opt.filter_type == 'HG':
            self.image_filter = HGFilter(
                    opt,
                    padding_mode=opt.hg_padding_mode
                )
            n_down = 7
            
        elif opt.filter_type == 'UNet':
            self.image_filter = UNet(
                n_channels=self.in_channels, 
                n_classes=256, 
                n_layers=6, 
                bilinear=True,
                inplace=self.opt.inplace,
                norm_type=self.opt.norm)
            n_down = 5
        
        elif opt.filter_type == 'UNet_s_min':
            self.image_filter = UNet_s_min(
                n_channels=self.in_channels, 
                n_classes=64, 
                n_layers=3, 
                bilinear=True,
                inplace=self.opt.inplace,
                norm_type=self.opt.norm)
            n_down = 7
        
        elif opt.filter_type == 'UNet_s_mid':
            self.image_filter = UNet_s_mid(
                n_channels=self.in_channels, 
                n_classes=32, 
                n_layers=3, 
                bilinear=True,
                inplace=self.opt.inplace,
                norm_type=self.opt.norm)
            n_down = 7

        elif opt.filter_type == 'UNet_s_deep':
            self.image_filter = UNet_s_deep(
                n_channels=self.in_channels, 
                n_classes=256, 
                n_layers=3, 
                bilinear=True,
                inplace=self.opt.inplace,
                norm_type=self.opt.norm)
            n_down = 6
            
        elif opt.filter_type == 'UNet_s_deeper':
            self.image_filter = UNet_s_deeper(
                n_channels=self.in_channels, 
                n_classes=256, 
                n_layers=3, 
                bilinear=True,
                inplace=self.opt.inplace,
                norm_type=self.opt.norm)
            n_down = 5
        elif opt.filter_type == 'UNet_s_deeper_128':
            self.image_filter = UNet_s_deeper_128(
                n_channels=self.in_channels, 
                n_classes=128, 
                n_layers=3, 
                bilinear=True,
                inplace=self.opt.inplace,
                norm_type=self.opt.norm,
                # padding_mode='zeros',
                padding_mode=opt.padding_mode
                )
            n_down = 5
        elif opt.filter_type == 'UNet_s_deeper_64':
            self.image_filter = UNet_s_deeper_64(
                n_channels=self.in_channels, 
                n_classes=64, 
                n_layers=3, 
                bilinear=True,
                inplace=self.opt.inplace,
                norm_type=self.opt.norm,
                conv_type=self.opt.conv_type)
            n_down = 5
        elif opt.filter_type == 'UNet_s_deepest':
            self.image_filter = UNet_s_deepest(
                n_channels=self.in_channels, 
                n_classes=256, 
                n_layers=3, 
                bilinear=True,
                inplace=self.opt.inplace,
                norm_type=self.opt.norm)
            n_down = 4
        elif opt.filter_type=='VGG19':
            self.image_filter = VGG19()

        
        ## from 32 dim to 1 dim
        in_channels = opt.mlp_dim_light[0]
        # self.light_filter = LightFilter(
        #     n_down = n_down,
        #     in_channels=in_channels,
        #     out_channels=in_channels
        # )

        # self.light_MLP = MLP(
        #     filter_channels=self.opt.mlp_dim_light,
        #     res_layers=self.opt.mlp_res_layers_light,
        #     mean_layer=self.opt.mlp_mean_layer_light,
        #     norm = self.opt.mlp_norm_type_light,
        #     activation=self.opt.mlp_activation_type_light,
        #     last_op=None,
        #     num_views=self.base_views,
        #     use_feature_confidence = self.opt.use_feature_confidence,
        #     use_mean_var=False,
        # )

        ## predict indirect incoming light, as (N dot L) + indiret
        if opt.pred_indirect:
            self.indirect_predictor = MLP(
                filter_channels=self.opt.mlp_dim_indirect,
                res_layers=self.opt.mlp_res_layers_indirect,
                mean_layer=self.opt.mlp_mean_layer_indirect,
                norm = self.opt.mlp_norm_type_indirect,
                activation=self.opt.mlp_activation_type_indirect,
                last_op=None,
                num_views=1,
                use_feature_confidence = self.use_feature_confidence,
                use_mean_var=False,
            )
        
        if opt.use_spatial:
            self.nkpt=24
            self.spatial = SpatialEncoder(sp_level=3, n_kpt=self.nkpt)
        mlp_dim_geo = self.opt.mlp_dim
        mlp_dim_alb = self.opt.mlp_dim_albedo        
        mlp_dim_spec_alb = self.opt.mlp_dim_spec_albedo
        mlp_dim_roughness = self.opt.mlp_dim_roughness
        print(mlp_dim_spec_alb)
        print(mlp_dim_roughness)

        if self.opt.feed_original_img:
            img_dim = 3
            mlp_dim_alb[0] += img_dim * (self.opt.multires * 2 + 1)
            mlp_dim_spec_alb[0] += img_dim * (self.opt.multires * 2 + 1)
            mlp_dim_roughness[0] += img_dim * (self.opt.multires * 2 + 1)

        if self.use_poe: # xyz positional encoding
            poe_feat_dim = 3
            mlp_dim_geo[0] += poe_feat_dim * (self.opt.multires * 2 + 1)
            mlp_dim_alb[0] += poe_feat_dim * (self.opt.multires * 2 + 1)
            mlp_dim_spec_alb[0] += poe_feat_dim * (self.opt.multires * 2 + 1)
            mlp_dim_roughness[0] += poe_feat_dim * (self.opt.multires * 2 + 1)
        
        elif self.use_spatial: # spatial encoding
            feat_dim = self.spatial.get_dim()
            mlp_dim_geo[0] += feat_dim
            mlp_dim_alb[0] += feat_dim
            mlp_dim_spec_alb[0] += feat_dim 
            mlp_dim_roughness[0] += feat_dim
        else: # z
            if not self.use_world_coord:
                feat_dim = 1
            else:
                feat_dim = 3
            if self.feed_mask:
                feat_dim+=1
            if self.feed_bound:
                feat_dim+=1
            mlp_dim_geo[0] += feat_dim
            mlp_dim_alb[0] += feat_dim
            mlp_dim_spec_alb[0] += feat_dim
            mlp_dim_roughness[0] += feat_dim
            

        mlp_input_views = self.base_views
        mlp_use_feat_confidence = self.opt.use_feature_confidence
        if opt.use_transformer:
            self.geo_transformer_encoder = Encoder(
                n_layers=self.opt.transformer_geo_n_layers,
                n_head=self.opt.transformer_geo_n_head,
                d_k=self.opt.transformer_geo_d_k,
                d_v=self.opt.transformer_geo_d_v,
                d_model=mlp_dim_geo[0],
                d_inner=self.opt.transformer_geo_d_inner,
                dropout=self.opt.transformer_geo_dropout
            )

            self.tex_transformer_encoder = Encoder(
                n_layers=self.opt.transformer_tex_n_layers,
                n_head=self.opt.transformer_tex_n_head,
                d_k=self.opt.transformer_tex_d_k,
                d_v=self.opt.transformer_tex_d_v,
                d_model=mlp_dim_alb[0],
                d_inner=self.opt.transformer_tex_d_inner,
                dropout=self.opt.transformer_tex_dropout
            )

            mlp_input_views = 1
            mlp_use_feat_confidence = False

        
        self.surface_classifier = MLP(
            filter_channels=mlp_dim_geo, # [257, 1024, 512, 256, 128, 1]
            res_layers=self.opt.mlp_res_layers,
            mean_layer=self.opt.mlp_mean_layer,
            norm = self.opt.mlp_norm_type,
            activation=self.opt.mlp_activation_type,
            last_op=None,
            num_views=mlp_input_views,
            use_feature_confidence = mlp_use_feat_confidence,
            use_mean_var=self.opt.use_mean_var,
            )

        self.albedo_predictor = MLP(
            filter_channels=mlp_dim_alb, # [257, 1024, 512, 256, 128, 3]
            res_layers=self.opt.mlp_res_layers_albedo,
            mean_layer=self.opt.mlp_mean_layer_albedo,
            norm = self.opt.mlp_norm_type_albedo,
            activation=self.opt.mlp_activation_type_albedo,
            # last_op=nn.Tanh(),
            last_op=nn.Sigmoid() if opt.albedo_out_fn=='sigmoid' else nn.Tanh(),
            num_views=mlp_input_views,
            use_feature_confidence = mlp_use_feat_confidence,
            use_mean_var=self.opt.use_mean_var,
            )
        
        self.spec_albedo_predictor = MLP(
            filter_channels=mlp_dim_spec_alb, # [257, 1024, 512, 256, 128, 3]
            res_layers=self.opt.mlp_res_layers_albedo,
            mean_layer=self.opt.mlp_mean_layer_albedo,
            norm = self.opt.mlp_norm_type_albedo,
            activation=self.opt.mlp_activation_type_albedo,
            last_op=None,
            output_bias=0.4,
            output_scale=0.1,
            num_views=mlp_input_views,
            use_feature_confidence = mlp_use_feat_confidence,
            use_mean_var=self.opt.use_mean_var,
            )
        
        self.roughness_predictor = MLP( ## output alpha = roughness * roughness
            filter_channels=mlp_dim_roughness, # [257, 1024, 512, 256, 128, 1]
            res_layers=self.opt.mlp_res_layers_albedo,
            mean_layer=self.opt.mlp_mean_layer_albedo,
            norm = self.opt.mlp_norm_type_albedo,
            activation=self.opt.mlp_activation_type_albedo,
            last_op=None,
            output_bias=0.2,
            output_scale=0.1,
            num_views=mlp_input_views,
            use_feature_confidence = mlp_use_feat_confidence,
            use_mean_var=self.opt.use_mean_var,
            )

        # self.normal_predictor = MLP(
        #     filter_channels=self.opt.mlp_dim_albedo, # [257, 1024, 512, 256, 128, 3]
        #     res_layers=self.opt.mlp_res_layers_albedo,
        #     mean_layer=self.opt.mlp_mean_layer_albedo,
        #     norm = self.opt.mlp_norm_type_albedo,
        #     activation=self.opt.mlp_activation_type_albedo,
        #     last_op=nn.Tanh(),
        #     num_views=mlp_input_views,
        #     use_feature_confidence = mlp_use_feat_confidence,
        #     use_mean_var=self.opt.use_mean_var,
        #     )

        self.normalizer = DepthNormalizer(opt)
        self.k = nn.Parameter(torch.tensor(opt.init_k))
        # self.intensity = nn.Parameter(torch.tensor(opt.intensity))
        self.offset = nn.Parameter(torch.zeros(1, 3, 1), requires_grad=False)

        init_net(self, init_type='normal')
        # init_net(self)


    def filter(self, images, k_no_grad=[], load_size=512):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [BK, C, H, W] input images
        :return im_feat: [BK, C_F, H', W']
        :return light_feat_mlp: [BK, C_L, 1]
        '''
        if len(k_no_grad)>0:
            im_feat_list = []
            light_feat_list = []
            for i in range(images.shape[0]):
                if i in k_no_grad:
                    with torch.no_grad():
                        im_feat_i, light_feat_i = self.image_filter(images[i:i+1])
                        # light_mlp_i = self.light_filter(light_feat_i).squeeze(-1)
                else:
                    im_feat_i, light_feat_i = self.image_filter(images[i:i+1])
                    # light_mlp_i = self.light_filter(light_feat_i).squeeze(-1)
                
                im_feat_list.append(im_feat_i)
                # light_feat_list.append(light_mlp_i)
            
            im_feat = torch.cat(im_feat_list, dim=0)
            # light_feat_mlp = torch.cat(light_feat_list, dim=0)
        else:
            if self.opt.filter_type == 'HG' or self.opt.filter_type=='HHG':
                im_feat, _, _ = self.image_filter(images) # [BK, c, H,W]

                # im feat of size [H/4, W/4, 256]
                im_feat = im_feat[-1]
                # light_feat_mlp = self.light_filter(im_feat).squeeze(-1)

            elif  'UNet' in self.opt.filter_type:
                im_feat, light_feat = self.image_filter(images) # [BK, c, H,W]
                # light_feat_mlp = self.light_filter(light_feat).squeeze(-1)

            elif self.opt.filter_type=='VGG19':
                output = self.image_filter(images)
                _,_,H,W= output['relu1_2'].shape
                import torch.nn.functional as F

                im_feat = torch.cat( (
                    output['relu1_2'], 
                    F.upsample(output['relu2_2'], scale_factor=2), 
                    F.upsample(output['relu3_4'], scale_factor=4), 
                    F.upsample(output['relu4_4'], scale_factor=8), 
                    F.upsample(output['relu5_4'], scale_factor=16)
                    ), dim=1 )
                light_feat= output['relu4_4']
                # light_feat_mlp = self.light_filter(light_feat).squeeze(-1)

        # return im_feat, light_feat_mlp
        return im_feat


    def basic_positional_encoding(self, x): 
        ## x: [B, C, H, W ...]
        ## cat at channel level
        x_proj = (2.*math.pi* x ) 
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)

    def in_mask(self, projected_points, masks):
        xy = projected_points[:, :2, :] # [B, 2, N]
        mask_val = self.index(masks, xy) 
        in_mask = mask_val >= 1
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        
        return in_mask * in_img



    def infer_res_shading(self, normal, light_dir):
        res_shading = self.res_shading_predictor(torch.cat((normal, light_dir), dim=1))
        pass

    def query_sdf(self, im_feat, points, calibs, z_center_world_space=None, masks=None, smpl_feat=None, coarse_sdf_feat=None, input_normal=None, transforms=None,
        use_positional_encoding=False, feed_original_img=False, imgs=None,extrinsic_reshape=None, dilation_size=5, joints_3d=None):
        K= calibs.shape[0]
        im_feat_dim = im_feat.shape[1]
        if self.sep_geo_brdf:
            sdf_feat, sdf_in_vhull = self.query(im_feat[:,:im_feat_dim//2,...], points, calibs, z_center_world_space, masks, smpl_feat,  transforms, 
            use_positional_encoding, input_normal=input_normal, extrinsic_reshape=extrinsic_reshape, dilation_size=dilation_size, joints_3d=joints_3d)
        else:
            sdf_feat, sdf_in_vhull = self.query(im_feat, points, calibs, z_center_world_space, masks, smpl_feat,  transforms, 
            use_positional_encoding, input_normal=input_normal, extrinsic_reshape=extrinsic_reshape, dilation_size=dilation_size, joints_3d=joints_3d)

        _,C,N = sdf_feat.shape

        if coarse_sdf_feat is not None:
            # sdf_feat = torch.cat((sdf_feat[:,:im_feat_dim,:], coarse_sdf_feat.expand(K, -1, -1)),dim=1)
            sdf_feat = torch.cat((sdf_feat[:,1:,:], coarse_sdf_feat.expand(K, -1, -1)),dim=1)
        # output_sdf_feat dim: [BK,C,N]
        if self.opt.use_transformer:
            ## input feat dim: [BN, K, C]
            feat_fused = self.geo_transformer_encoder(sdf_feat.view(-1, K, C, N).permute(0,3,1,2).view(-1, K, C))
            if self.use_world_coord:
                feat_fused = torch.cat((feat_fused[:,:,:-3], points.permute((2,0,1)).contiguous().expand(-1,K,-1)), dim=-1)

            sdf = self.surface_classifier(feat_fused.view(-1, N, K, C)[:,:,:1,:].permute(0,2,3,1).view(-1, C, N))
        else:
            sdf = self.surface_classifier(sdf_feat)

        return sdf, sdf_in_vhull

    def query_albedo(self, im_feat, points, calibs, z_center_world_space=None,  masks=None, smpl_feat=None,coarse_sdf_feat=None, input_normal=None, transforms=None, 
        use_positional_encoding=False, feed_original_img=False, imgs=None, extrinsic_reshape=None, dilation_size=5, joints_3d=None):
        K= calibs.shape[0]
        im_feat_dim = im_feat.shape[1]
        if self.sep_geo_brdf:
            albedo_feat, _ = self.query(im_feat[:,im_feat_dim//2:,...], points, calibs, z_center_world_space, masks, smpl_feat, transforms, 
            use_positional_encoding, feed_original_img, imgs=imgs,input_normal=input_normal, extrinsic_reshape=extrinsic_reshape, dilation_size=dilation_size, joints_3d=joints_3d)
        else:
            albedo_feat, _ = self.query(im_feat, points, calibs, z_center_world_space, masks, smpl_feat, transforms, 
            use_positional_encoding, feed_original_img, imgs=imgs,input_normal=input_normal, extrinsic_reshape=extrinsic_reshape, dilation_size=dilation_size, joints_3d=joints_3d)

        if coarse_sdf_feat is not None:
            # albedo_feat = torch.cat((albedo_feat[:,:im_feat_dim,:], coarse_sdf_feat.expand(K, -1, -1)),dim=1)
            albedo_feat = torch.cat((albedo_feat[:,1:,:], coarse_sdf_feat.expand(K, -1, -1)),dim=1)
        
        _,C,N = albedo_feat.shape
        if self.opt.use_transformer:
            ## input feat dim: [BN, K, C]
            feat_fused = self.tex_transformer_encoder(albedo_feat.view(-1, K, C, N).permute(0,3,1,2).view(-1, K, C))
            if self.use_world_coord:
                feat_fused = torch.cat((feat_fused[:,:,:-3], points.permute((2,0,1)).contiguous().expand(-1,K,-1)), dim=-1)

            # albedo = self.albedo_predictor(feat_fused.view(-1, N, K, C)[:,:,:1,:].permute(0,2,3,1).view(-1, C, N)) * 0.5 + 0.5
            albedo = self.albedo_predictor(feat_fused.view(-1, N, K, C)[:,:,:1,:].permute(0,2,3,1).view(-1, C, N))
            if self.opt.albedo_out_fn == 'tanh':
                albedo = albedo * 0.5 +0.5
        else:
            # albedo = self.albedo_predictor(albedo_feat) * 0.5 + 0.5
            albedo = self.albedo_predictor(albedo_feat)
            if self.opt.albedo_out_fn == 'tanh':
                albedo = albedo * 0.5 +0.5
        
        return albedo

    def query_spec_albedo(self, im_feat, points, calibs, z_center_world_space=None,  masks=None, smpl_feat=None,coarse_sdf_feat=None, input_normal=None, transforms=None, 
        use_positional_encoding=False, feed_original_img=False, imgs=None, extrinsic_reshape=None, dilation_size=5, joints_3d=None):
        K= calibs.shape[0]
        im_feat_dim = im_feat.shape[1]
        if self.sep_geo_brdf:
            albedo_feat, _ = self.query(im_feat[:,im_feat_dim//2:,...], points, calibs, z_center_world_space, masks, smpl_feat, transforms, 
            use_positional_encoding, feed_original_img, imgs=imgs,input_normal=input_normal, extrinsic_reshape=extrinsic_reshape, dilation_size=dilation_size, joints_3d=joints_3d)
        else:
            albedo_feat, _ = self.query(im_feat, points, calibs, z_center_world_space, masks, smpl_feat, transforms, 
            use_positional_encoding, feed_original_img, imgs=imgs,input_normal=input_normal, extrinsic_reshape=extrinsic_reshape, dilation_size=dilation_size, joints_3d=joints_3d)

        if coarse_sdf_feat is not None:
            # albedo_feat = torch.cat((albedo_feat[:,:im_feat_dim,:], coarse_sdf_feat.expand(K, -1, -1)),dim=1)
            albedo_feat = torch.cat((albedo_feat[:,1:,:], coarse_sdf_feat.expand(K, -1, -1)),dim=1)
        
        _,C,N = albedo_feat.shape
        if self.opt.use_transformer:
            ## input feat dim: [BN, K, C]
            feat_fused = self.tex_transformer_encoder(albedo_feat.view(-1, K, C, N).permute(0,3,1,2).view(-1, K, C))
            albedo = self.spec_albedo_predictor(feat_fused.view(-1, N, K, C)[:,:,:1,:].permute(0,2,3,1).view(-1, C, N))
        else:
            albedo = self.spec_albedo_predictor(albedo_feat)
        
        return albedo
    
    def query_roughness(self, im_feat, points, calibs, z_center_world_space=None,  masks=None, smpl_feat=None,coarse_sdf_feat=None, input_normal=None, transforms=None, 
        use_positional_encoding=False, feed_original_img=False, imgs=None, extrinsic_reshape=None, dilation_size=5, joints_3d=None):
        K= calibs.shape[0]
        im_feat_dim = im_feat.shape[1]
        if self.sep_geo_brdf:
            albedo_feat, _ = self.query(im_feat[:,im_feat_dim//2:,...], points, calibs, z_center_world_space, masks, smpl_feat, transforms, 
            use_positional_encoding, feed_original_img, imgs=imgs,input_normal=input_normal, extrinsic_reshape=extrinsic_reshape, dilation_size=dilation_size, joints_3d=joints_3d)
        else:
            albedo_feat, _ = self.query(im_feat, points, calibs, z_center_world_space, masks, smpl_feat, transforms, 
            use_positional_encoding, feed_original_img, imgs=imgs,input_normal=input_normal, extrinsic_reshape=extrinsic_reshape, dilation_size=dilation_size, joints_3d=joints_3d)
        if coarse_sdf_feat is not None:
            # albedo_feat = torch.cat((albedo_feat[:,:im_feat_dim,:], coarse_sdf_feat.expand(K, -1, -1)),dim=1)
            albedo_feat = torch.cat((albedo_feat[:,1:,:], coarse_sdf_feat.expand(K, -1, -1)),dim=1)
        
        _,C,N = albedo_feat.shape
        if self.opt.use_transformer:
            ## input feat dim: [BN, K, C]
            feat_fused = self.tex_transformer_encoder(albedo_feat.view(-1, K, C, N).permute(0,3,1,2).view(-1, K, C))
            roughness = self.roughness_predictor(feat_fused.view(-1, N, K, C)[:,:,:1,:].permute(0,2,3,1).view(-1, C, N))
        else:
            roughness = self.roughness_predictor(albedo_feat)
        
        return roughness

    def query_normal(self, im_feat, points, calibs, z_center_world_space=None,  masks=None, smpl_feat=None,coarse_sdf_feat=None, input_normal=None, transforms=None, use_positional_encoding=False, feed_original_img=False, imgs=None, extrinsic_reshape=None):
        '''
        query normal from image feature directly
        '''
        K= calibs.shape[0]
        feat, _ = self.query(im_feat, points, calibs, z_center_world_space, masks, smpl_feat, transforms, use_positional_encoding, feed_original_img, imgs=imgs,input_normal=input_normal, extrinsic_reshape=extrinsic_reshape)
        im_feat_dim = im_feat.shape[1]
        if coarse_sdf_feat is not None:
            # albedo_feat = torch.cat((albedo_feat[:,:im_feat_dim,:], coarse_sdf_feat.expand(K, -1, -1)),dim=1)
            feat = torch.cat((feat[:,1:,:], coarse_sdf_feat.expand(K, -1, -1)),dim=1)
        
        _,C,N = feat.shape
        if self.opt.use_transformer:
            ## input feat dim: [BN, K, C]
            feat_fused = self.tex_transformer_encoder(feat.view(-1, K, C, N).permute(0,3,1,2).view(-1, K, C))
            normal = self.normal_predictor(feat_fused.view(-1, N, K, C)[:,:,:1,:].permute(0,2,3,1).view(-1, C, N))
        else:
            normal = self.normal_predictor(feat)
        
        return normal

    def query_indirect_lighting(self, im_feat, points, calibs, z_center_world_space=None, masks=None, smpl_feat=None, transforms=None, use_positional_encoding=False, feed_original_img=False, imgs=None, normal=None, extrinsic_reshape=None):
        '''
        query the indirect lighting of the query point from a local neighborhood

        ## new feat: normal: B,3,N
        '''
        feat, _ = self.query(im_feat, points, calibs, z_center_world_space, masks, smpl_feat, transforms, use_positional_encoding, extrinsic_reshape=extrinsic_reshape)
        indirect_feat = torch.cat((feat, normal), dim=1) # [256+1+3]
        indirect_lighting = self.indirect_predictor(indirect_feat)
        
        return indirect_lighting

    def query_shading(self, normals, points, extrinsics):
        # [BK,3,N]
        B = points.shape[0]
        K = self.num_views
        points_list = []
        normals_list = []
        for b in range(B):
            points_list.append(points[b,...].expand(K, -1, -1))
            normals_list.append(normals[b,...].expand(K, -1, -1))
        points_stack = torch.cat(points_list, dim=0)
        normals_stack = torch.cat(normals_list, dim=0)
        cam2models = torch.linalg.inv(extrinsics)
        # [BK, 3]
        cam_locs = cam2models[:, :3, 3]
        light_pos = cam_locs * math.sin(math.radians(42/2))
        light_vector = points_stack - light_pos[:,:,None] #[B,3,N]
        inc_light_dir = safe_l2norm(light_vector) #[B,3,N]
        world_normal = safe_l2norm(normals_stack) ##[B,3,N]
        distance = light_vector.norm(dim=1, keepdim=True)  #[B,1,N]
        att = 1/ (distance **2 + 1e-8) #[B,1,N]
        cos = torch.clamp( torch.sum(world_normal * -inc_light_dir, dim=1), 0, 1)
        shading = cos[:,None,:] * att * self.opt.intensity ##[B,1,N]

        return shading.repeat(1,3,1)
        

    def query(self, im_feat, points, calibs, z_center_world_space=None, masks=None, smpl_feat=None, transforms=None, use_positional_encoding=False, 
        feed_original_img=False, imgs=None, input_normal=None, extrinsic_reshape=None, dilation_size=5, joints_3d=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param im_feat: [BK, c_feat, H, W] image feature
        :param points: [n_cam, 3, N] model space coordinates of points
        :param calibs: [BK, 4, 4] calibration matrices for each image
        :param z_center: [1, 3, 1] world center of z
        :param masks: [BK, 1, H, W] mask feature
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param points_mask: [BK, N, 1] for points in calculation
        :return: [BK, C, N] predictions for each point
        :return: [BK, 1, N] confidences for each point
        '''
        # print(points.shape) # [BK, 3, N]
        # print(calibs.shape) # [BK, 4, 4]
        # points = torch.ones_like(points) * 0.2
        B = points.shape[0]
        K = calibs.shape[0]
        n_samp = points.shape[-1]

        if torch.any(torch.isnan(points)):
            print('point has nan')
        
        
        if self.finetune:
            self.offset.requires_grad_()

        points = points.permute((1,2,0)).contiguous().view(3, -1).unsqueeze(0) # [1,3,N]

        # points_list = []
        # for b in range(B):
        #     points_list.append(points[b,...].expand(K, -1, -1))
        points_stack = points.expand(K, -1, -1)
        
        xyz = self.projection(points_stack + self.offset, calibs, transforms, self.load_size) # this projection is differentiable
        xy = xyz[:, :2, :] # [BK, 2, N]
        if extrinsic_reshape is not None:
            w = torch.ones_like(points_stack)[:,:1,:]
            points_w  = torch.cat((points_stack,w),dim=1)
            xyz_camera_space = torch.bmm(extrinsic_reshape, points_w)
            z = xyz_camera_space[:, 2:3, :]
        else:
            z = xyz[:, 2:3, :] # [BK, 1, N]

        # if not self.opt.use_perspective:
        if not self.opt.use_perspective and (not self.opt.use_CV_perspective):
            z_feat = self.normalizer(z)
        else:
            if self.use_spatial:
                # pdb.set_trace()
                kptv = joints_3d[:,:self.nkpt,:].permute(0,2,1).expand(K, -1, -1) # [B,3,Nkp]->[BV,3,Nkp]
                cxyz = xyz_camera_space.permute(0,2,1)[:, :, :3] # [BV,3,N]->[BV,N,3]
                w = torch.ones_like(kptv)[:,:1,:]
                kptvw  = torch.cat((kptv,w),dim=1)
                # pdb.set_trace()
                kptvxyz = torch.bmm(extrinsic_reshape, kptvw).permute(0,2,1)[:, :, :3] # [BV,4,4] dot [BV,4,Nkp] ->[BV,4,Nkp]
                z_feat = self.spatial.forward(cxyz, kptvxyz)
            else:
                if self.use_world_coord:
                    z_feat = points_stack
                else:
                    z_feat = z
                    if z_center_world_space is not None:
                        z_centers_stack = z_center_world_space.expand(K, -1, B*n_samp)
                        if extrinsic_reshape is not None:
                            w = torch.ones_like(z_centers_stack)[:,:1,:]
                            points_w  = torch.cat((z_centers_stack,w),dim=1)
                            xyz_camera_space = torch.bmm(extrinsic_reshape, points_w)
                            proj_z_center = xyz_camera_space[:, 2:3, :]
                        else:
                            proj_z_center = self.projection(z_centers_stack, calibs, transforms, self.load_size)[:, 2:3, :] # this projection is differentiable
                        
                        z_feat = z - proj_z_center
        
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        in_img.unsqueeze_(1) # [B, 1, N]

        pts_in_visual_hull = in_visual_hull(xyz, masks, K, dilation_kernel_size=dilation_size)  # [B, 1, N]

        # [B, 1, N]
        if use_positional_encoding:
            embed_fn, input_ch = get_embedder(self.opt.multires)

            xy_norm_z = torch.cat((xy, z_feat), dim=1) # embed the projected xy and normalized_z
            
            feat_poe = embed_fn(points_stack) # embed the original points
            z_feat = torch.cat((z_feat, feat_poe), dim=1)
            # z_feat_p = embed_fn(xy_norm_z)
            # zf_list = []
            # for b in range(B):
            #     zf_list.append(z_feat_p[b,...].expand(K, -1, -1))
            # z_feat = torch.cat(zf_list, dim=0)

        # 
        # [BK, C_feat, N] cat [BK, 1, N]
        # [BK, C_feat + 1, N]
        # point_local_feat_list = [self.index(im_feat, xy), z_feat]
        point_local_feat_list = [self.index(im_feat, xy), z_feat] # + in_mask and in_img
        
        if self.feed_bound:
            point_local_feat_list += [in_img]

        if self.feed_mask:
            point_local_feat_list += [F.grid_sample(masks, xy.transpose(1,2).unsqueeze(2))[..., 0].detach()]
        if input_normal is not None:
            point_local_feat_list += [self.index(input_normal, xy).detach()]

        if self.opt.use_visual_hull and masks is not None:
            visual_hull_feature = in_visual_hull(xyz, masks, K, dilation_kernel_size=dilation_size) # [B, 1, N]
            visual_hull_feature_list = []
            for b in range(visual_hull_feature.shape[0]):
                visual_hull_feature_list.append(visual_hull_feature[b,...].expand(self.base_views, -1, -1))
            visual_hull_feature_stack = torch.cat(visual_hull_feature_list, dim=0)
            point_local_feat_list += [visual_hull_feature_stack]
        
        if smpl_feat is not None:
            point_local_feat_list += [smpl_feat.expand(self.base_views, -1, -1)]

        if feed_original_img:
            embed_fn, input_ch = get_embedder(self.opt.multires)
            point_local_feat_list += [embed_fn(self.index(imgs, xy).detach())]
        


        point_local_feat = torch.cat(point_local_feat_list, 1)
        if feed_original_img:
            point_local_feat = point_local_feat.detach()
        # sdf = self.surface_classifier(point_local_feat)
        # albedo = self.albedo_predictor(point_local_feat)
        # sdf = sdf * pts_in_visual_hull
        # sdf = sdf + (~pts_in_visual_hull).float() * 1000
        return point_local_feat, pts_in_visual_hull
        # return point_local_feat, in_img
        # out of image plane is always set to 0
        # pred_occupancy = in_img[:,None].float() * occupancy_field
        # self.intermediate_preds_occupancy_list.append(pred_occupancy)
        # pred_sdf = in_img[:,None].float() * sdf
        

    def get_bce_error(self, preds, labels, mask=None):
        '''
        Given K {pred, label, }, calculate the bce error
        :param preds: [B, 1, N] 
        :param labels: [B, 1, N] the ground truth label of the points
        :return: Const,  the bce error tensor
        '''
        if self.opt.field_type == 'sdf':
            criterion = nn.BCEWithLogitsLoss(reduction='none')
            B, C, N = preds.shape
            # error = criterion(preds * torch.exp(self.k), labels) # BCE(phi(kd)-L)
            if self.opt.k_type == 'raw':
                error = criterion(preds * self.k, labels) # BCE(phi(kd), L)
            elif self.opt.k_type == 'exp':
                error = criterion(preds * torch.exp(self.k), labels)   # BCE(phi(kd), L)
            else:
                ### undefined 
                print('Undefined BCE error type')
                exit(-1)
        else:
            # criterion = nn.BCEWithLogitsLoss(reduction='none')
            criterion = nn.BCELoss(reduction='none')
            B, K, N = preds.shape
            error = criterion(preds, labels.repeat(1,K,1))
        if mask is not None:
            return torch.sum(error * mask) / (torch.sum(mask) + self.eps)
        
        return error.mean()
        # return error.sum(dim=1).mean()

    def get_regularization_2_error(self, sdf_normal_grad):
        B, C, N = sdf_normal_grad.shape
        assert C == 3
        criterion = nn.MSELoss(reduction='none')
        norm_grad2 = torch.norm(sdf_normal_grad, dim=1)
        error = criterion(norm_grad2, torch.zeros_like(norm_grad2, requires_grad=False))
        # [BK,N] or [B,N]
        return error.mean()

    def get_regularization_error(self, sdf_normal, mask=None):
        '''
        :param sdf_normal: [BK, 3, N] if no numerical diff else [B,3,N]
        :return: Const,  the bce error tensor
        '''
        B, C, N = sdf_normal.shape
        assert C == 3
        criterion = nn.MSELoss(reduction='none')
        norm1 = torch.norm(sdf_normal, dim=1)
        # print('norm1: ', norm1)
        error = criterion(norm1, torch.ones_like(norm1, requires_grad=False))
        if mask is not None:
            return torch.sum(error * mask) / (torch.sum(mask) + self.eps)
        
        return torch.mean(error)
        # return error.mean()

    # only for surface points
    def get_normal_error(self, surf_pts_gradient, labels_normal, mask=None):
        '''
        Given K {sdf_normal, label_normal, confidences}, calculate the surface geometry error
        :param surf_pts_gradient: [B, 3, N] the predicted sdf value of the points of K views
        :param labels_normal: [B, 3, N] the ground truth label of the points
        :return: scalar,  the bce error tensor
        '''
        # print(self.normals)
        # print(self.labels_normal)

        B, C, N = surf_pts_gradient.shape
        assert C == 3
        assert len(surf_pts_gradient.shape) == 3
        # error = torch.norm(safe_l2norm(surf_pts_gradient.view(B, 3, K, N).contiguous()) - labels_normal.repeat(1, K ,1).view(B, 3, K, N).contiguous(), p=2, dim=1)
        error = torch.norm(surf_pts_gradient - labels_normal, p=2, dim=1)
        
        if mask is not None:
            return torch.sum(error * mask)  / (torch.sum(mask) + self.eps)
        return error.mean() 


    # only for surface points
    def get_surface_error(self, preds, mask=None):
        error = preds.abs()
        if mask is not None:
            return torch.sum(error * mask)  / (torch.sum(mask) + self.eps)
        return error.mean()
    
    def get_error_reg(self, gradient_space, mask=None):
        ## [B, K * 3, N]
        if self.opt.lambda_reg==0:
            self.reg_error = torch.zeros(1, device=self.k.device)
        else:
            self.reg_error = self.get_regularization_error(gradient_space, mask) * self.opt.lambda_reg

        return self.reg_error

    def get_alignment_loss(self, sdf_func, gradient_func, pts, pts_grad=None, decay=10):
        '''
        Params:
        @ sdf_func: [B,3,N] -> B1N
        @ gradient_func: [B,3,N] -> B3N
        @ pts: [B,3,N]
        @ pts_grad: [B,3,N]

        Return:
        loss tensor
        '''
        pts_sdf = sdf_func(pts) #[B1N]
        pts_grad = gradient_func(pts) if pts_grad is None else pts_grad  #[B3N]
        pts_grad_norm = nnF.normalize(pts_grad, dim=-1)
        pts_moved = pts - pts_grad_norm * pts_sdf
        # pts_moved_sdf = sdf_func(pts_moved)
        pts_moved_grad = gradient_func(pts_moved.detach()) # B3N
        pts_moved_grad_norm = nnF.normalize(pts_moved_grad, dim=-1) # B3N
        align_constraint = 1 - nnF.cosine_similarity(pts_grad_norm, pts_moved_grad_norm, dim=-1) # BN
        beta = torch.exp(-decay * torch.abs(pts_sdf))
        return torch.mean(beta * align_constraint.unsqueeze(-1))

    def get_error_g(self, surf_pts_gradient, preds,  labels_normal, mask=None):
        '''
        Given K {sdf_normal, label_normal, sdf}, calculate the surface geometry error
        :param surf_pts_gradient: [BK, 3, N] the predicted sdf value of the points
        :param labels_normal: [BK, 3, N] the ground truth label of the points
        :param preds:[B, K, N] the sdf value of the predictions
        :param confidences:[BK, N] the confidence of the predictions
        :return: Const,  the bce error tensor
        '''
        if self.opt.lambda_g2==0:
            self.normal_error = torch.zeros(1, device=self.k.device)
        else:
            self.normal_error = self.get_normal_error(surf_pts_gradient, labels_normal, mask) * self.opt.lambda_g2

        if self.opt.lambda_g1==0:
            self.surface_error = torch.zeros(1, device=self.k.device)
        else:
            self.surface_error = self.get_surface_error(preds, mask) * self.opt.lambda_g1
        return self.surface_error + self.normal_error

    def get_error_sdf(self, preds, labels, mask=None):
        '''
            preds: [B,1,N]
            labels: [B,1,N]
        '''

        criterion = nn.L1Loss(reduction='none')
        # criterion = nn.MSELoss(reduction='none')

        B, K, N = preds.shape
        # error = criterion(preds * torch.exp(self.k), labels) # BCE(phi(kd)-L)
        if not self.opt.truncate_sdf:
            error = criterion(preds, labels.repeat(1,K,1))   # BCE(phi(kd), L)
        else:
            error = criterion(torch.clamp(preds, max=0.1), torch.clamp(labels.repeat(1,K,1), max=0.1))

        if mask is not None:
            return torch.sum(error * mask)  / (torch.sum(mask) + self.eps)
        return error.sum(dim=1).mean()



    # def get_error_align(self, sdf_normal, mask=None):

    
    def get_error_reg_2(self, sdf_normal_grad):
        if self.opt.lambda_reg_2==0 or sdf_normal_grad==None:
            self.reg_error_2 = torch.zeros(1, device=self.k.device)
        else:
            self.reg_error_2 = self.get_regularization_2_error(sdf_normal_grad) * self.opt.lambda_reg_2

        return self.reg_error_2



    def get_sdf_normal(self, images, points, calibs,extrinsic_reshape=None, z_center_world_space=None, input_normal=None, masks=None, smpl_feat=None,  transforms=None, im_feat=None, 
        use_positional_encoding=False, joints_3d=None):
        points.requires_grad_()

        if im_feat is None:
            im_feat, _ = self.filter(images)  # get im_feat with shape [B*K, c_feat, H, W]
        pred_sdf, in_vishull = self.query_sdf(im_feat=im_feat, points=points, calibs=calibs, z_center_world_space=z_center_world_space,
            smpl_feat=smpl_feat, masks=masks, input_normal=input_normal, transforms=transforms,use_positional_encoding=use_positional_encoding, extrinsic_reshape=extrinsic_reshape, joints_3d=joints_3d) 
        grad = gradient(points, pred_sdf)
        return grad

    def get_sdf_normal_by_diff(self, images, points, calibs,extrinsic_reshape=None, z_center_world_space=None,input_normal=None, masks=None, smpl_feat=None,  transforms=None, im_feat=None, 
        use_positional_encoding=False, delta=1e-4, joints_3d=None):
        ### no need to store grad
        '''
        Given points and input images of K views, output the finite difference at the points
        :param images: [BK, 3, H, W] the predicted sdf value of the points
        :param calibs: [BK, 4, 4] the predicted sdf value of the points
        :param points: [B, 3, N] the predicted sdf value of the points
        :return: normal (calculated by finite difference) [B,3,N]
        '''
        dx = torch.zeros_like(points)
        dy = torch.zeros_like(points)
        dz = torch.zeros_like(points)

        dx[:,0,:] = delta
        dy[:,1,:] = delta
        dz[:,2,:] = delta
        
        nx = points - dx
        ny = points - dy
        nz = points - dz
        px = points + dx
        py = points + dy
        pz = points + dz

        if im_feat is None:
            im_feat, _ = self.filter(images)  # get im_feat with shape [B*K, c_feat, H, W]


        nx_sdf, _ = self.query_sdf(im_feat=im_feat, points=nx, calibs=calibs, z_center_world_space=z_center_world_space, smpl_feat=smpl_feat, masks=masks,
            input_normal=input_normal, transforms=transforms,use_positional_encoding=use_positional_encoding, extrinsic_reshape=extrinsic_reshape, joints_3d=joints_3d)
        ny_sdf, _ = self.query_sdf(im_feat=im_feat, points=ny, calibs=calibs, z_center_world_space=z_center_world_space, smpl_feat=smpl_feat, masks=masks,
            input_normal=input_normal, transforms=transforms,use_positional_encoding=use_positional_encoding, extrinsic_reshape=extrinsic_reshape, joints_3d=joints_3d)
        nz_sdf, _ = self.query_sdf(im_feat=im_feat, points=nz, calibs=calibs, z_center_world_space=z_center_world_space, smpl_feat=smpl_feat, masks=masks, 
            input_normal=input_normal, transforms=transforms,use_positional_encoding=use_positional_encoding, extrinsic_reshape=extrinsic_reshape, joints_3d=joints_3d)
        px_sdf, _ = self.query_sdf(im_feat=im_feat, points=px, calibs=calibs, z_center_world_space=z_center_world_space, smpl_feat=smpl_feat, masks=masks, 
            input_normal=input_normal, transforms=transforms,use_positional_encoding=use_positional_encoding, extrinsic_reshape=extrinsic_reshape, joints_3d=joints_3d)
        py_sdf, _ = self.query_sdf(im_feat=im_feat, points=py, calibs=calibs, z_center_world_space=z_center_world_space, smpl_feat=smpl_feat, masks=masks, 
            input_normal=input_normal, transforms=transforms,use_positional_encoding=use_positional_encoding, extrinsic_reshape=extrinsic_reshape, joints_3d=joints_3d)
        pz_sdf, _ = self.query_sdf(im_feat=im_feat, points=pz, calibs=calibs, z_center_world_space=z_center_world_space, smpl_feat=smpl_feat, masks=masks, 
            input_normal=input_normal, transforms=transforms,use_positional_encoding=use_positional_encoding, extrinsic_reshape=extrinsic_reshape, joints_3d=joints_3d)

        ## [B,1,N] for output sdf

        dx = (px_sdf - nx_sdf)/(2*delta)
        dy = (py_sdf - ny_sdf)/(2*delta)
        dz = (pz_sdf - nz_sdf)/(2*delta)

        normal = torch.cat((dx,dy,dz),dim=1)
        return normal
    
    def get_albedo_gradient(self, images, points, calibs, z_center_world_space=None, masks=None, smpl_feat=None,  transforms=None, im_feat=None, use_positional_encoding=False, feed_original_img=False):
        points.requires_grad_()
        if im_feat is None:
            im_feat, _ = self.filter(images)  # get im_feat with shape [B*K, c_feat, H, W]
        pred_albedo = self.query_albedo(im_feat=im_feat, points=points, calibs=calibs, z_center_world_space=z_center_world_space,smpl_feat=smpl_feat, masks=masks, transforms=transforms,use_positional_encoding=use_positional_encoding, feed_original_img=feed_original_img, imgs=images) 
        grad = gradient(points, pred_albedo)
        return grad

    def forward(self, images, points_space, calibs,\
         z_center_world_space=None, im_feat=None,smpl_feat_space=None,smpl_feat_surface=None,\
            light_feat=None, masks=None, input_normal=None, transforms=None,\
            labels_space=None, samples_sdf=None, points_surf=None, labels_surf=None,\
                 name=None, vids=None,extrinsic_reshape=None, joints_3d=None):
        '''
        Given K {images, points, calibs, labels, points_surf, labels_surf}, calculate the error of the predicted sdf
        :param images:[B*K, 3, H, W] the predicted sdf value of the points
        :param points: Tensor, [B, 3, N] the world space coord of spatial points
        :param calibs: List, [B*K , 4, 4] the MVP matrix of the image
        :param z_center_world_space [optional]: Tensor, [B, 3, N] the world space coord of body center
        :param im_feat [optional]: Tensor, [B, C, N] the world space coord of body center
        :param smpl_feat [optional]: Tensor, [B, 1, N] the sdf of points to smpl
        :param labels: Tensor, [B, 1, N] the sign of the ground-truth sdf
        :param points_surf: Tensor, [B, 3, N] the world space coord of surface points
        :param labels_surf: Tensor, [B, 3, N] the world space coord normal of surface points
        :return: error, the scalar value
        :return: error_dict, the dict contains different errors
        '''

        ## confidence based normal loss

        K = self.base_views
        B = points_surf.shape[0]
        ## [0|1] for in visual hull points

        # points_surf_stack = points_surf.repeat(K, 1, 1) # [BK, 3, N]
        # points_space_stack = points_space.repeat(K, 1, 1) # [BK, 3, N]

        
        if im_feat is None:
            self.im_feat = self.filter(images)  # get im_feat with shape [B*K, c_feat, H, W]
        else:
            self.im_feat = im_feat

        feat_dim = self.im_feat.shape[1]
        ## light_feat: [BK, C_L, 1] -> [B, 28]
        # self.light = self.light_MLP(self.light_feat_mlp).squeeze()
        points_surf.requires_grad_()
        pred_sdf_surf, surf_in_vishull = self.query_sdf(im_feat=self.im_feat, points=points_surf, calibs=calibs, z_center_world_space= z_center_world_space,
            smpl_feat=smpl_feat_surface, masks=masks,input_normal=input_normal, transforms=transforms, use_positional_encoding=self.use_poe, extrinsic_reshape=extrinsic_reshape, joints_3d=joints_3d)
        self.surf_pts_gradient = gradient(points_surf, pred_sdf_surf) # [B, 3, N]
        
        points_space.requires_grad_()
        pred_sdf_space, space_in_vishull = self.query_sdf(im_feat=self.im_feat, points=points_space, calibs=calibs,z_center_world_space= z_center_world_space,
            smpl_feat=smpl_feat_space, masks=masks,input_normal=input_normal, transforms=transforms, use_positional_encoding=self.use_poe, extrinsic_reshape=extrinsic_reshape, joints_3d=joints_3d)
        # preds_space: [BK,1,N] or [B,1,N] for confidence or mean
        self.output_pred_space = pred_sdf_space
        self.space_pts_gradient = gradient(points_space, pred_sdf_space) # [B, 3, N]

        # surface sdf value & normal
        # error_g = self.get_error_g(surf_pts_gradient.view(-1, self.num_views * 3, N_Surf).contiguous(), preds_surf, labels_surf, confidences=None)
        error = torch.zeros(1, device=self.k.device)
        
        space_in_vishull = None
        error_g = self.get_error_g(self.surf_pts_gradient, pred_sdf_surf, labels_surf, mask=surf_in_vishull)
        error = error + error_g
        error_bce = self.get_bce_error(pred_sdf_space, labels_space, mask=space_in_vishull) * self.opt.lambda_sign if self.opt.lambda_sign!=0 else torch.zeros(1, device=self.k.device)
        error = error + error_bce
        
        if self.opt.use_gt_sdf:
            error_sdf = self.get_error_sdf(pred_sdf_space, samples_sdf, mask=space_in_vishull) * self.opt.lambda_sdf if self.opt.lambda_sdf!=0 else torch.zeros(1, device=self.k.device)
            error = error + error_sdf
        
        error_reg = self.get_error_reg(self.space_pts_gradient, mask=space_in_vishull)

        
        error = error + error_reg
        error_align = torch.zeros_like(error_reg)
        
        if self.opt.use_align_loss:
            sdf_func = lambda x ,im_feats=self.im_feat, calibs=calibs, masks=masks, z_center_world_space=z_center_world_space, extris=extrinsic_reshape \
            :self.query_sdf(im_feats, x, calibs, z_center_world_space, masks, use_positional_encoding=self.use_poe, extrinsic_reshape=extris, joints_3d=joints_3d)[0]
            gradient_func = lambda x ,images=images, im_feats=self.im_feat, calibs=calibs, masks=masks, z_center_world_space=z_center_world_space, extris=extrinsic_reshape \
            :self.get_sdf_normal(images, x, calibs, extrinsic_reshape=extris, use_positional_encoding=self.use_poe, z_center_world_space=z_center_world_space, masks=masks, im_feat=im_feats, joints_3d=joints_3d)

            error_align = self.get_alignment_loss(sdf_func, gradient_func, points_space, self.space_pts_gradient)
            error = error + error_align
        
        error_dict = {}
        error_dict['surface_sdf_error'] = self.surface_error
        error_dict['surface_normal_error'] = self.normal_error
        error_dict['bce_error'] = error_bce
        error_dict['reg_error'] = error_reg
            
        error_dict['align_error'] = error_align
        if self.opt.use_gt_sdf:
            error_dict['sdf_error'] = error_sdf
        # error = error_l
        return error, error_dict

    

