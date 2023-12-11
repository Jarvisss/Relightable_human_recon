from cProfile import label
from turtle import back
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from model.UNet_unified import UNet_unified
from dataset.Thuman2_pifu_dataset_sdf import make_dataset, make_dataloader
# from dataset.Thuman2_taichi3_dataset import make_dataset, make_dataloader
import pdb

from model.DiffRenderer_unified import DiffRenderer_unified
from utils.common_utils import *
import torchvision.transforms as transforms
from utils.geo_utils import *
from utils.render_utils import get_camera_params_in_model_space, get_uvs
from utils.camera import get_calib_extri_from_pose, get_quat_from_world_mat_np
from utils.vis_utils import vis_pt_normal_arrow
from options.options import get_options
from tqdm import tqdm
import cv2
import torchvision.transforms as T
from PIL import Image, ImageDraw
import random
from tensorboardX import SummaryWriter
from utils.sdf_utils import save_samples_truncted_prob, save_samples_color
import time
import matplotlib.pyplot as plt
from model.blocks import _freeze, _unfreeze

device = torch.device("cuda:0")
cpu = torch.device("cpu")
# device = cpu

def save_generator(epoch, lossesG, netG, i_batch, optimizerG,path_to_chkpt_G,parallel=False):
    netG_state_dict = netG.state_dict() if not parallel else netG.module.state_dict()
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'G_state_dict': netG_state_dict,
        'i_batch': i_batch,
        'optimizer_state_dict': optimizerG.state_dict(),
    }, path_to_chkpt_G)

def save_param_to_json(opt, path_to_ckpt):
    import json
    param_file = 'param.json'
    param_path = os.path.join(path_to_ckpt, param_file)
    with open(param_path, 'w') as f:
        json.dump(vars(opt), f, indent=4)



def train_unified(opt, exp_name, pretrained_ckpt_path, date):
    path_to_ckpt, path_to_vis = make_training_dirs(opt, exp_name, date)
    path_to_latest_G = path_to_ckpt + 'model_weights.tar'
    writer = SummaryWriter(logdir=path_to_ckpt)
    save_param_to_json(opt, path_to_ckpt)
    print('-------Creating Model--------')
    if opt.use_perspective:
        projection_mode = 'perspective'
    elif opt.use_CV_perspective:
        projection_mode = 'perspective_cv'
    else:
        projection_mode = 'orthogonal'
    # projection_mode = 'perspective' if opt.use_perspective else 'orthogonal'
    backbone = UNet_unified(opt, base_views=opt.num_views, projection_mode=projection_mode).to(device)
    # dr = DiffRenderer_unified(opt, backbone, opt.num_views, device=device, debug=False).to(device)
    
    backbone.eval()

    lr_init_net = opt.lr_G

    param_list = [
        {'params': list(backbone.surface_classifier.parameters()) + list(backbone.geo_transformer_encoder.parameters()) + list(backbone.image_filter.parameters()) , 'lr':lr_init_net},
        {'params': list(backbone.tex_transformer_encoder.parameters())+ list(backbone.albedo_predictor.parameters()) + list(backbone.spec_albedo_predictor.parameters()) + list(backbone.roughness_predictor.parameters()) , 'lr': lr_init_net},
        {'params': backbone.k, 'lr': lr_init_net},
        # {'params': backbone.spec_albedo_predictor.parameters(), 'lr': lr_init_color},
        # {'params': backbone.roughness_predictor.parameters(), 'lr': lr_init_color},
    ]
    # optimizerG = optim.Adam(backbone.parameters(), lr=lr_init, amsgrad=False)
    optimizerG = optim.Adam(param_list, amsgrad=False)
    # optimizerG = optim.RMSprop(backbone.parameters(), lr=lr_init, momentum=0, weight_decay=0)
    if pretrained_ckpt_path is None:
        # initiate checkpoint if inexist
        print('No specified checkpoint...')
        if not os.path.isfile(path_to_latest_G):
            print('No continue train checkpoint...')

            print('Initiating new checkpoint...')
            save_generator( 0, [],backbone, 0, optimizerG, path_to_latest_G)
            print('Done...')
        print('Load checkpoint from ... %s' % path_to_latest_G)
        checkpoint_G = torch.load(path_to_latest_G, map_location=device)
        # backbone.load_state_dict(checkpoint_G['G_state_dict'], strict=False)

    else:
        print('Load checkpoint from ... %s' % pretrained_ckpt_path)
        checkpoint_G = torch.load(pretrained_ckpt_path, map_location=device)

    '''
    allow for weight shape mismatch loading
    '''
    # current_model_dict = backbone.state_dict()
    # new_state_dict={k:v if v.size()==current_model_dict[k].size() else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), checkpoint_G['G_state_dict'].values())}
    # backbone.load_state_dict( new_state_dict, strict=False)
    backbone.load_state_dict(checkpoint_G['G_state_dict'], strict=False)

    # backbone.load_state_dict( checkpoint_G['G_state_dict'], strict=False)
    if opt.freeze_img_filter:
        _freeze(backbone.image_filter)
    # init_net(backbone.albedo_predictor)
    epochCurrent = checkpoint_G['epoch']
    i_batch_current = checkpoint_G['i_batch']

    pytorch_total_params = sum(p.numel() for p in backbone.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    if opt.lambda_reg_finetune > 0:
        fixed_backbone = type(backbone)(opt, projection_mode=projection_mode).to(device)
        fixed_backbone.load_state_dict(backbone.state_dict(), strict=False)
        fixed_backbone.eval()
    # init_net(backbone.albedo_predictor)

    debug=True
    if debug:
        print('image filter:')
        print(backbone.image_filter)
        
        print('albedo_classifier:')
        print(backbone.albedo_predictor)

        print('geo_predictor:')
        print(backbone.surface_classifier)
        # print('light regressor:')
        # print(backbone.light_filter)
        print('Model params: %d, trainable params: %d' %(pytorch_total_params, pytorch_total_trainable_params) )

    print('-------Loading Dataset--------')
    train_dataset = make_dataset(opt, 'train')
    train_dataloader = make_dataloader(opt, train_dataset, phase='train')
    test_dataset = make_dataset(opt, 'test')
    test_dataloader = make_dataloader(opt, test_dataset, phase='test')
    print('train data size: ', len(train_dataloader))
    print('test data size: ', len(test_dataloader))
    print('-------Training Start--------')

    # if opt.lambda_reg_finetune > 0:
    if opt.gen_init_mesh:
        print('generate mesh (gt) ...')
        test_data = random.choice(test_dataset)
        save_path_geo_gt = '%s/geo/epoch_%d_batch%d_gt/test_%s.obj' % (path_to_vis, epochCurrent, i_batch_current, test_data['name'])
        os.makedirs(os.path.dirname(save_path_geo_gt), exist_ok=True)
        with torch.no_grad():
            for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                gen_mesh_color_unified(opt, backbone, device, test_data, save_path_geo_gt, num_views=opt.num_views, use_octree=opt.use_octree, levels=opt.output_levels)

    
    i_batch_total = epochCurrent * train_dataloader.__len__() // opt.batch_size + i_batch_current


    for epoch in range(epochCurrent, epochCurrent + opt.epochs):
        """ Training start """
        if epoch > epochCurrent:
            i_batch_current = 0
        epoch_loss_G = 0
        pbar = tqdm(train_dataloader, leave=True, initial=0)
        pbar.set_description('[{0:>4}/{1:>4}]'.format(epoch,epochCurrent + opt.epochs))
        # lr = adjust_learning_rate(optimizerG, epoch-epochCurrent, lr, opt.schedule, opt.gamma)
        opt.lambda_g1 = adjust_lambda_g1(epoch, opt.lambda_g1_init, opt.lambda_g1_end)
        
        for i_batch, batch_data in enumerate(pbar, start=0):
            t0 = time.time()
            z_center = batch_data['z_center'].float().to(device) if opt.normalize_z else None
            samples = batch_data['samples'].float().to(device) if opt.num_sample_inout else None
            samples_surf = batch_data['surface_samples'].float().to(device) if opt.num_sample_surface else None
            labels = batch_data['labels'].to(device) if opt.num_sample_inout else None
            samples_sdf = batch_data['samples_sdf'].float().to(device) if opt.use_gt_sdf else None
            samples_surf_normal = batch_data['surface_normals'].to(device) if opt.num_sample_surface else None
            samples_color_pts = batch_data['color_samples'].to(device) if opt.num_sample_color else None
            samples_albedo = batch_data['surface_albedo'].to(device) if opt.num_sample_color else None

            samples_smpl_sdf = batch_data['samples_smpl_sdf'].float().to(device) if opt.use_smpl else None
            surface_smpl_sdf = batch_data['surface_smpl_sdf'].float().to(device) if opt.use_smpl else None

            smpl_joints = batch_data['smpl_joints'].float().to(device) if opt.use_spatial else None

            imgs = batch_data['img'].float().to(device) # [B, K, C, H, W ]
            albedos = batch_data['albedo'].float().to(device)
            normals = batch_data['normal'].float().to(device)
            calibs = batch_data['calib'].float().to(device)
            extrinsic = batch_data['extrinsic'].float().to(device)
            quat = batch_data['quat'].float().to(device)
            intrinsic = batch_data['intrinsic'].float().to(device)
            mask = batch_data['mask'].float().to(device)
            dist = batch_data['dist'].float().to(device) if opt.feed_dist else None
            name = batch_data['name']
            # labels_albedo = batch_data['surface_albedo'].to(device)
            # labels_shading = batch_data['surface_shading'].to(device)
            view_ids = batch_data['view_ids']

            quat = quat.view(quat.shape[0] * quat.shape[1], quat.shape[2]).contiguous()
            
            
            masks_reshape = mask.view(mask.shape[0] * mask.shape[1],
                mask.shape[2],
                mask.shape[3],
                mask.shape[4]).contiguous()
            normals_reshape = normals.view(normals.shape[0] * normals.shape[1],
                normals.shape[2],
                normals.shape[3],
                normals.shape[4]).contiguous()
            intrinsic_reshape = intrinsic.view(
                intrinsic.shape[0] * intrinsic.shape[1],
                intrinsic.shape[2],
                intrinsic.shape[3]
            ).contiguous()

            extrinsic_reshape = extrinsic.view(
                extrinsic.shape[0] * extrinsic.shape[1],
                extrinsic.shape[2],
                extrinsic.shape[3]
            ).contiguous()

            # pdb.set_trace()
            # calibs, extris, extris_inv = get_calib_extri_from_pose(quat, intrinsic_reshape)
            # pdb.set_trace()
            # calibs = calibs.unsqueeze(0)
            imgs_reshape, calibs_reshape = reshape_multiview_images(imgs, calibs) 
            B, _, surface_pt_num = samples_surf.shape
            K = opt.num_views
            BK,_,IH,IW = imgs_reshape.shape
            assert(B*K==BK)
            if opt.use_linear_z:
                feed_extrin = extrinsic_reshape
            else:
                feed_extrin = None

            
            normal_input = normals_reshape if opt.input_normal else None
            # [B, K, C, H, W ] --> [B * K, C, H, W]
            # [B, K, 4, 4] --> [B * K, 4, 4]
            backbone.train()
            
            
            criterion = nn.L1Loss(reduction='mean')
            
            t1 = time.time()
            n_input = opt.num_views
            input_view = [i for i in range(n_input)]
            if opt.selected_train:
                if n_input > 4:
                    a = np.random.choice(input_view, size=4, replace=False)
                else:
                    a = input_view
                k_no_grad = list(set(input_view) - set(a))
            else:
                k_no_grad = []
            # im_feat_pred, light_feat_pred = backbone.filter(imgs_reshape, k_no_grad=k_no_grad)
            # im_feat_pred = backbone.filter(imgs_reshape, k_no_grad=k_no_grad)
            input_tensor = imgs_reshape
            if opt.feed_mask:
                input_tensor = torch.cat((input_tensor, masks_reshape), dim=1)
            if opt.feed_dist:
                dist_reshape = dist.view(dist.shape[0] * dist.shape[1],
                    dist.shape[2],
                    dist.shape[3],
                    dist.shape[4]).contiguous()
                input_tensor = torch.cat((input_tensor, dist_reshape), dim=1)
            if opt.feed_dir:
                uvs = get_uvs(IW, IH).expand(K, -1, -1).to(device)
                ray_ds, cam_locs = get_camera_params_in_model_space(uvs.cpu(), extris_inv.cpu(), intrinsic_reshape.cpu(), neg_z=opt.use_perspective)
                ray_ds = ray_ds.transpose(1,2).reshape(K, 3, IW, IH).to(device)
                input_tensor = torch.cat((input_tensor, ray_ds), dim=1)
            
            
            im_feat_pred = backbone.filter(input_tensor,k_no_grad=k_no_grad)
            light_feat_pred = None
            t2 = time.time()
            # print('img filter forward time: %.3f' %(t2-t1))
            if opt.lambda_albedo > 0:
                albedo = backbone.query_albedo(im_feat_pred, samples_color_pts, calibs_reshape,z_center_world_space=z_center, masks=masks_reshape, use_positional_encoding=opt.use_positional_encoding, 
                feed_original_img=opt.feed_original_img, imgs=imgs_reshape, extrinsic_reshape=feed_extrin, input_normal=normal_input, joints_3d=smpl_joints)
                error_albedo = criterion(albedo, img_g2l(samples_albedo)) * opt.lambda_albedo
            else:
                error_albedo = torch.zeros(1).to(device)

            if opt.lambda_normal>0:
                normal = backbone.query_normal(im_feat_pred, samples_surf, calibs_reshape,z_center_world_space=z_center, masks=masks_reshape, use_positional_encoding=opt.use_positional_encoding,
                feed_original_img=opt.feed_original_img, imgs=imgs_reshape, extrinsic_reshape=feed_extrin, input_normal=normal_input, joints_3d=smpl_joints)
                error_normal = criterion(normal, samples_surf_normal) * opt.lambda_normal
            else:
                error_normal = torch.zeros(1).to(device)

            loss_total = torch.zeros(1).to(device)

            lr = lr_init_net


            error_geo, geo_error_dict = backbone.forward(imgs_reshape, samples, calibs_reshape, z_center_world_space=z_center, \
                im_feat=im_feat_pred, smpl_feat_space=samples_smpl_sdf, smpl_feat_surface=surface_smpl_sdf, light_feat=light_feat_pred, masks=masks_reshape, labels_space=labels, samples_sdf=samples_sdf, \
            points_surf=samples_surf, labels_surf=samples_surf_normal, name=name, vids=view_ids, extrinsic_reshape=feed_extrin, input_normal=normal_input, joints_3d=smpl_joints)
            writer.add_scalar('Loss_geo', error_geo.item() ,i_batch_total)
            t3 = time.time()
            
            loss_total = error_geo + error_albedo + error_normal

            # loss_total = error_albedo
            epoch_loss_G += loss_total.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
            post_fix_str = 'lr=%.5f, k:%.3f:{mov=%.4f, geo=%.4f, albedo=%.4f, aln=%.4f, ssdf=%.4f,snor=%.4f,bce=%.4f,reg=%.4f'\
                %(lr, backbone.k, epoch_loss_G_moving, error_geo.item(), error_albedo.item(), geo_error_dict['align_error'].item(), geo_error_dict['surface_sdf_error'].item(),geo_error_dict['surface_normal_error'].item(),\
            geo_error_dict['bce_error'].item(), geo_error_dict['reg_error'].item())

            if opt.use_gt_sdf:
                post_fix_str += ',sdf=%.4f' % geo_error_dict['sdf_error']

            optimizerG.zero_grad()
            loss_total.backward()
            # print(backbone.albedo_predictor.filters[0].weight.grad)
            optimizerG.step()

            t4 = time.time()
            post_fix_str += '},lod: %.3f'%(t1-t0)
            post_fix_str += ',flt: %.3f'%(t2-t1)
            post_fix_str += ',fwd: %.3f'%(t3-t2)
            post_fix_str += ',bkw: %.3f'%(t4-t3)
            pbar.set_postfix_str(post_fix_str)
                        
            if opt.model_save_freq > 0 and i_batch_total % opt.model_save_freq == 0 and epoch-epochCurrent+i_batch > 0:
                path_to_save_G = path_to_ckpt + 'epoch_{}_batch_{}_G.tar'.format(epoch, i_batch)
                save_generator( epoch, [], backbone, i_batch, optimizerG, path_to_save_G)
                save_generator( epoch, [], backbone, i_batch, optimizerG, path_to_latest_G)

            if opt.gen_mesh_freq > 0  and  i_batch_total% opt.gen_mesh_freq == 0 and epoch-epochCurrent+i_batch > 0:
                level = 0.5 if opt.field_type=='occupancy' else 0
                backbone.eval()
                test_data = random.choice(test_dataset)
                train_data = random.choice(train_dataset)
                save_path_geo_test = '%s/geo/epoch_%d_batch%d/test_%s.obj' % (path_to_vis, epoch,i_batch, test_data['name'])
                os.makedirs(os.path.dirname(save_path_geo_test), exist_ok=True)
                save_path_geo_train = '%s/geo/epoch_%d_batch%d/train_%s.obj' % (path_to_vis, epoch,i_batch, train_data['name'])

                np.savetxt(save_path_geo_test.replace('obj', 'txt'), test_data['view_ids'], '%d')
                np.savetxt(save_path_geo_train.replace('obj', 'txt'), train_data['view_ids'], '%d')

                with torch.no_grad():
                    print('generate mesh (test) ...')
                    for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                        gen_mesh_color_unified(opt, backbone, device, test_data, save_path_geo_test, num_views=opt.num_views, use_octree=opt.use_octree, levels=opt.output_levels)
                    print('generate mesh (train) ...')
                    for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                        gen_mesh_color_unified(opt, backbone, device, train_data, save_path_geo_train, num_views=opt.num_views, use_octree=opt.use_octree, levels=opt.output_levels)
                    
                # '''
                # print('generate mesh (gt) ...')
                # for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                #     # test_data = random.choice(test_dataset)
                #     save_path_geo = '%s/test_eval_gt_epoch%d_batch%d_%s_geo.obj' % (
                #         path_to_vis, epoch,i_batch, test_data['name'])
                #     # with torch.no_grad():
                #     gen_mesh_color_unified(opt, fixed_backbone, device, test_data, save_path_geo, num_views=opt.num_views, use_octree=opt.use_octree, level=level)
                # '''
                
                # backbone.train()

            if opt.image_proj_save_freq > 0 and i_batch_total % opt.image_proj_save_freq == 0 and epoch-epochCurrent+i_batch > 0:
                save_img_list = []
                save_img_path = '%s/train_eval_epoch%d_batch%d_%s_imgs_gt.png'% (
                                path_to_vis, epoch, i_batch, name[0])
                GRID_SIZE = 20
                xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, GRID_SIZE), np.linspace(-0.5, 0.5, GRID_SIZE),
                                        np.linspace(-0.5, 0.5, GRID_SIZE))
                points_grid = torch.from_numpy(np.stack((xx.flatten(), yy.flatten(), zz.flatten()))).float().unsqueeze(0)

                for v in range(imgs[0].shape[0]):

                    calib_k = calibs[:,v,...]
                    save_img = (np.transpose(imgs[0][v].detach().cpu().numpy(), (1, 2, 0)) )[:, :, ::-1] * 255.0
                    proj_func = backbone.projection
                    projs = proj_func(samples_surf, calib_k, size=opt.size)[0].detach().cpu().numpy()
                    projs_space = proj_func(points_grid.to(device), calib_k,  size=opt.size)[0].detach().cpu().numpy()
                    xy = (np.array(opt.load_size)[:, np.newaxis]*(projs[:2,:] * 0.5 + 0.5))   # [2, N]
                    space_xy = (np.array(opt.load_size)[:, np.newaxis]*(projs_space[:2,:] * 0.5 + 0.5)) 
                    pil_save_img = Image.fromarray(np.uint8(save_img))
                    draw = ImageDraw.Draw(pil_save_img)
                    for n in range(projs.shape[-1]):
                        draw.point(xy[:,n], fill=(255,0,0))
                    
                    # for n in range(projs_space.shape[-1]):
                    #     # if labels[0,0, n].cpu() > 0:
                    #         # draw.point(space_xy[:,n], fill=(0,0,255))
                    #     # else:
                    #     draw.point(space_xy[:,n], fill=(0,255,0))

                    save_img_list.append(np.array(pil_save_img))
                    
                save_img = np.concatenate(save_img_list, axis=1)
                Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path) 
            
            i_batch_total += 1
            

            

def test_one_object(opt,netG, date, test_data, test_result_dir, vid=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    experiment_name = opt.test_id
    
    yids = test_data['view_ids']
    if vid is not None:
        view_ids = yids[vid:vid+1]
        num_views = 1
    else:
        view_ids = yids
        num_views = opt.num_views
    
    post_fix = '%d' % view_ids[0]
    for i in range(1, len(view_ids)):
        post_fix += ' %d' % view_ids[i]
    save_path = '%s/%s/test_%s.obj' % (
            test_result_dir, test_data['name'], post_fix)
    
    level = 0.5 if opt.field_type=='occupancy' else 0

    gen_mesh_color_unified(opt, netG, device, test_data, save_path,num_views=num_views, use_octree=opt.use_octree, levels=opt.output_levels, vid=vid)
    # gen_mesh(opt, netG, device, test_data, save_path)
    pass



def get_syn_data_with_dr(opt, subject='0526', n_sample_space=200000, angle_step=18, num_target=None, test=False):
    test_data = {}

    calib_list = []
    mask_list = []
    mask_dilate_list = []
    img_list = []
    albedo_list = []
    shading_list = []
    extri_list = []
    # extri_inv_list = []
    intri_list = []
    norm_list = []
    quat_list = []

    # b_min = torch.tensor([-0.5, -0, -0.5])
    # b_max = torch.tensor([0.5, 1, 0.5])
    b_min = torch.tensor([-1, -1, -1])
    b_max = torch.tensor([1, 1, 1])

    # view_id_pairs = [ ## 12 views
    #     [0, 4],
    #     [96, 2],
    #     [180, 1],
    #     [276, 0],
    #     [36, 0],
    #     [60, 0],
    #     [120, 0],
    #     [156, 0],
    #     [216, 0],
    #     [240, 0],
    #     [300, 0],
    #     [336, 0],
    # ]
    
    # view_id_pairs = [ ## 12 views
    #     [0, 1],
    #     [96, 1],
    #     [180, 1],
    #     [276, 3],
    #     [36, 0],
    #     [60, 0],
    #     [120, 0],
    #     [156, 0],
    #     [216, 0],
    #     [240, 0],
    #     [300, 0],
    #     [336, 0],
    # ]
    if not test:
        view_id_pairs = [ ## 24 views
            [0, 4],
            # [96, 2],
            [96, 2],
            [180, 1],
            [276, 0], ## 4 basic views
            [48, 0],
            [132, 0],
            [228, 0],
            [312, 0], ## 8 views
            [24, 0],
            [120, 0],
            [204, 0],
            [300, 0], ## 12 views
            [72, 0],
            [156, 0], 
            [252, 0],
            [336, 0], ## 16 views
            [12, 0],
            [144, 0], 
            [216, 0],
            [324, 0], ## 20 views
            [36, 0], 
            [168, 0],
            [240, 0], 
            [288, 0] ## 24 views
        ]

        # view_id_pairs = [ ## 24 views
        #     [0, 4],
        #     # [96, 2],
        #     [78, 2],
        #     [198, 1],
        #     [312, 0], ## 4 basic views
        #     [48, 0],
        #     [132, 0],
        #     [228, 0],
        #     [312, 0], ## 8 views
        #     [24, 0],
        #     [120, 0],
        #     [204, 0],
        #     [300, 0], ## 12 views
        #     [72, 0],
        #     [156, 0], 
        #     [252, 0],
        #     [336, 0], ## 16 views
        #     [12, 0],
        #     [144, 0], 
        #     [216, 0],
        #     [324, 0], ## 20 views
        #     [36, 0], 
        #     [168, 0],
        #     [240, 0], 
        #     [288, 0] ## 24 views
        # ]
    else:
        view_id_pairs = [ ## 30 views
            [60, 0],
            [84, 1],
            [108, 2],
            [192, 4],
            [264, 3],
            [348, 2],
            [6, 3],
            [18, 4],
            [30, 3],
            [42, 2],
            [54, 1],
            [78, 3],
            [102, 4],
            [126, 3],
            [138, 2],
            [150, 4],
            [162, 1],
            [174, 2],
            [210, 4],
            [222, 3],
            [234, 2],
            [246, 1],
            [258, 3],
            [282, 4],
            [294, 3],
            [306, 2],
            [318, 1],
            [330, 2],
            [342, 3],
            [354, 2]
        ]

    intensity = torch.tensor(opt.init_intensity).float()

    if num_target is not None:
        view_id_pairs = view_id_pairs[:num_target]
    # view_ids = [240]
    # view_ids = [0]
    view_id_list = []
    # path_data = '/mnt/data1/lujiawei/thuman2_rescaled_prt_512_single_light_w_flash_no_env_persp_fov_42.00_unit_sphere_d2_near_1.00_far_5.00_dist_augment'
    path_data = opt.path_to_dataset
    path_obj = opt.path_to_obj

    if opt.use_spatial:
        path_smpl_joints = os.path.join(opt.path_to_SMPL_joints, subject, '%s_smplx_joints.npy' % subject)
        smpl_joints = torch.Tensor(np.load(path_smpl_joints)).float().unsqueeze(0)
        
    ## for error computing
    import trimesh
    mesh_name = subject if opt.dataset == 'Thuman2' else 'mesh-f%s'%subject.split('_')[-1]
    mesh = trimesh.load(os.path.join(path_obj, subject, '%s.obj'%(mesh_name)))
    mesh_verts = mesh.vertices
    mesh_faces = mesh.faces
    
    test_data['mesh_verts'] = mesh_verts
    test_data['mesh_faces'] = mesh_faces
    test_data['mesh'] = mesh
    for y_id, p_id in view_id_pairs:
        path_param = os.path.join(path_data, 'PARAM')
        path_img = os.path.join(path_data, 'RENDER')
        path_albedo = os.path.join(path_data, 'ALBEDO')
        path_shading = os.path.join(path_data, 'SHADING')
        path_mask = os.path.join(path_data, 'MASK')
        param_path = os.path.join(path_param, subject, '%d_%d_%02d.npy' % (y_id, 0, p_id))
        mask_path = os.path.join(path_mask, subject, '%d_%d_%02d.png' % (y_id, 0, p_id))
        img_path = os.path.join(path_img, subject, '%d_%d_%02d.png' % (y_id, 0, p_id))
        albedo_path = os.path.join(path_albedo, subject, '%d_%d_%02d.png' % (y_id, 0, p_id))
        shading_path = os.path.join(path_shading, subject, '%d_%d_%02d.png' % (y_id, 0, p_id))
        # img_path = os.path.join(path_img, subject, '%d_%d_%02d.jpg' % (y_id, 0, p_id))
        # albedo_path = os.path.join(path_albedo, subject, '%d_%d_%02d.jpg' % (y_id, 0, p_id))
        params = np.load(param_path, allow_pickle=True)
        mask = Image.open(mask_path).convert('L')
        img = Image.open(img_path).convert('RGB')
        albedo = Image.open(albedo_path).convert('RGB')
        shading = Image.open(shading_path).convert('RGB')

        dilate_kernel_size = opt.dilate_size
        dilate_kernel = np.ones((dilate_kernel_size,dilate_kernel_size), np.uint8)
        mask_dilate = cv2.dilate(np.array(mask), dilate_kernel, iterations=1)

        intrinsic = params.item().get('intrinsic')
        extrinsic = params.item().get('extrinsic')
        quat = get_quat_from_world_mat_np(extrinsic)
        c2w = np.linalg.inv(extrinsic)
        cam_center = c2w[:3,3]
        print(cam_center)
        print(np.linalg.norm(cam_center))

        gl_2_cv_matrix = np.eye(4)
        gl_2_cv_matrix[1,1] = -1
        intrinsic = gl_2_cv_matrix @ intrinsic
        calib = intrinsic @ extrinsic
        calib_list += [torch.Tensor(calib).float()]
        mask_list += [T.ToTensor()(mask).float()]
        mask_dilate_list += [T.ToTensor()(mask_dilate).float()]

        img_list += [T.ToTensor()(img).float()]
        albedo_list += [T.ToTensor()(albedo).float()]
        shading_list += [T.ToTensor()(shading).float()]
        extri_list.append( torch.Tensor(extrinsic).float())
        # extri_inv_list.append(torch.Tensor(np.linalg.inv(extrinsic)).float())
        intri_list.append( torch.Tensor(intrinsic).float())
        quat_list.append(torch.Tensor(quat).float())
        norm_list.append(torch.Tensor(np.eye(4)).float())
        view_id_list.append(y_id)
    
    
    
    length = b_max - b_min
    space_points = (torch.rand(3, n_sample_space) * length.unsqueeze(-1) + b_min.unsqueeze(-1)).float().unsqueeze(0)

    test_data['intensity'] = intensity
    test_data['smpl_joints'] = smpl_joints if opt.use_spatial else None
    test_data['b_min'] = b_min
    test_data['b_max'] = b_max
    test_data['name'] = subject
    test_data['z_center'] = torch.zeros(1, 3, 1) if opt.normalize_z else None
    test_data['view_ids'] = view_id_list
    test_data['pose'] = torch.stack(quat_list, dim=0)

    test_data['calib'] = torch.stack(calib_list, dim=0)
    test_data['extrinsic'] = torch.stack(extri_list, dim=0)
    # test_data['extrinsic_inv'] = torch.stack(extri_inv_list, dim=0)
    test_data['intrinsic'] = torch.stack(intri_list, dim=0)
    test_data['normal_matrices'] = torch.stack(norm_list, dim=0)
    
    test_data['in_ori_img'] = torch.ones_like(mask_list[0])
    test_data['img'] = torch.stack(img_list, dim=0)
    test_data['albedo'] = torch.stack(albedo_list, dim=0)
    test_data['shading'] = torch.stack(shading_list, dim=0)
    test_data['mask'] = torch.stack(mask_list, dim=0)
    test_data['mask_dilate'] = torch.stack(mask_dilate_list, dim=0)
    test_data['samples'] = space_points
    # test_data['surface_samples'] = surface_pts


    return test_data

if __name__ == "__main__":
    from datetime import datetime

    opt = get_options()
    # for k,v in sorted(vars(opt).items()):
    #     print(k,':',v)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    if opt.phase == 'train':
        today = datetime.today().strftime("%Y%m%d")
        # today = '20220420'
        # today = '20220530'
        # today = '20220714'
        # today = '20220715'
        # today = '20220721'
        # today = '20220729'
        # today = '20220902'
        # today = '20220923'
        # today = '20221031'
        # today = '20221120'
        today = '20221220'
        set_random_seed(opt.seed)
        print(opt.id)
        geo_ckpt_path = opt.load_pretrained_path
        train_unified(opt, opt.id, geo_ckpt_path, date=today)
    else:
        from utils.common_utils import make_testing_dirs
        # date = '20230822'
        # date = '20230902'
        # date = '20230925'
        date = '20221220'
        test_output_dir = make_testing_dirs(opt, opt.test_id, date)
        if opt.use_perspective:
            projection_mode = 'perspective'
        elif opt.use_CV_perspective:
            projection_mode = 'perspective_cv'
        else:
            projection_mode = 'orthogonal'
        test_ckpt_path = opt.load_pretrained_path
        ckpt = torch.load(test_ckpt_path, map_location=device)
        backbone = UNet_unified(opt, opt.num_views, projection_mode=projection_mode).to(device)
        backbone.load_state_dict(ckpt['G_state_dict'], strict=True)
        backbone.eval()
        n_sample_space = 200000

        dataset = make_dataset(opt, phase='test')

        # test_data = dataset[0] ## load test synthetic data by ids
        for test_data in dataset:
            with torch.no_grad():
                test_one_object(opt, backbone, date, test_data, test_result_dir=test_output_dir)

    pass


