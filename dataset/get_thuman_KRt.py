import os
from turtle import width
import numpy as np
import json, codecs
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
import shutil
from PIL import Image

dataset_folder = '/mnt/data1/lujiawei/'
# dataset_name = 'thuman2_rescaled_prt_512_single_light_w_flash_no_env_fov_68.40_unit_sphere_d2_near_0.10_far_4.00_dist_1.78_intensity_1.000_aug_p_-20.000_20.000_aug_y_-10.000_10.000_aug_dist_-0.300_0.200'
# dataset_name = 'customhuman_prt_512_single_light_flash_1_fov_68.40_unit_sphere_d2_near_0.10_far_4.00_dist_1.78_intensity_1.000_aug_p_-20.000_20.000_aug_y_-10.000_10.000_aug_dist_-0.300_0.200'
dataset_name = 'thuman2_rescaled_prt_512_single_light_flash_1_fov_68.40_unit_sphere_d2_near_0.10_far_4.00_dist_1.78_intensity_0.500_aug_p_-20.000_20.000_aug_y_-10.000_10.000_aug_dist_-0.300_0.200'
# dataset_name = 'thuman2_rescaled_prt_512_single_light_flash_0_fov_68.40_unit_sphere_d2_near_0.10_far_4.00_dist_1.78_intensity_0.500_aug_p_-20.000_20.000_aug_y_-10.000_10.000_aug_dist_-0.300_0.200'
workspace_folder = os.path.join(dataset_folder, dataset_name)

print(workspace_folder)
out_folder = 'IDR_format_transparent'
# subject = '0526'


subjects = [
 '0004',
 '0082',
 '0178',
 '0287',
 '0403',
 '0455',
 '0519'
]

# subjects = [
#     '0072_00022_04_00001',
#     '0137_00033_10_00081',
#     '0143_00034_06_00122',
#     '0186_00044_15_00141',
#     '0192_00045_08_00141',
#     '0208_00048_02_00021',
#     '0256_00058_7_00141',
#     '0259_00059_08_00041'
# ]

# view_id_pairs = [
#         [180, 0, 0],
#         [210, 20, 0],
#         [150, -30, 0]
#         ]


# view_id_pairs = [ ## 12 views
#         [0, 4],
#         [96, 2],
#         [180, 1],
#         [276, 0],
#         [36, 0],
#         [60, 0],
#         [120, 0],
#         [156, 0],
#         [216, 0],
#         [240, 0],
#         [300, 0],
#         [336, 0],
#     ]

view_id_pairs_12 = [ ## 12 views
        [0, 4],
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
    ]

view_id_pairs_24 = [ ## 24 views
        [0, 4],
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

view_id_pairs_test_30 = [ ## 30 views
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

# view_id_pairs = view_id_pairs_12
# view_id_pairs = view_id_pairs_24
view_id_pairs = view_id_pairs_test_30


for subject in subjects:
    out_folder_name = '%s_%dviews_test' % (subject, len(view_id_pairs))
    # out_folder_name = '%s_%dviews' % (subject, len(view_id_pairs))
    print(out_folder_name)
    jfpath = os.path.join(workspace_folder, out_folder, out_folder_name, 'cam_dict_norm.json')
    out_cam_path = os.path.join(workspace_folder, out_folder, out_folder_name, 'cameras.npz')
    mask_dir = os.path.join(workspace_folder, out_folder, out_folder_name, 'mask')
    normal_dir = os.path.join(workspace_folder, out_folder, out_folder_name, 'normal')
    albedo_dir = os.path.join(workspace_folder, out_folder, out_folder_name, 'albedo')
    img_dir = os.path.join(workspace_folder, out_folder, out_folder_name, 'image')
    os.makedirs(os.path.dirname(jfpath), exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(albedo_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # n_imgs = 60
    # n_augs = 10
    fov = np.radians(68.4)
    width = 512
    height= 512
    focal_x = width /  (np.tan(fov/2) * 2)
    focal_y = height /  (np.tan(fov/2) * 2)
    px = width/2
    py = height/2

    K = np.identity(4)
    K[0,0] = focal_x
    K[1,1] = focal_y
    K[0,2] = px
    K[1,2] = py
    print(K)

    id_txt_path = os.path.join(workspace_folder, 'all_ids.txt')


    final_dict  = {}
    cam_params_new = {}


    with open(id_txt_path, 'w') as f:
        count = 0
        # all_views = list(range(0,360,6))
        # for i in range(n_imgs):
        for i in range(len(view_id_pairs)):
            y, a = view_id_pairs[i]
            p = 0
            iname = '%d_%d_%02d.png' % (y, p, a)
            # for n in range(n_augs):
                # iname = '%d_0_%02d.jpg' % (all_views[i], n)
            print(count, iname)
            f.write('%s %s\n' % (str(count), iname[:-4]))
            param_path = os.path.join(workspace_folder, 'PARAM', subject, '%s.npy' % iname[:-4])
            param = np.load(param_path, allow_pickle=True)
            src_img_path = os.path.join(workspace_folder, 'RENDER', subject, iname)
            tgt_img_path = os.path.join(img_dir, '%06d.png'%count)
            src_mask_path = os.path.join(workspace_folder, 'MASK', subject, iname)
            tgt_mask_path = os.path.join(mask_dir, '%06d.png'%count)
            src_normal_path = os.path.join(workspace_folder, 'NORMAL', subject, iname)
            tgt_normal_path = os.path.join(normal_dir, '%06d.png'%count)
            src_albedo_path = os.path.join(workspace_folder, 'ALBEDO', subject, iname)
            tgt_albedo_path = os.path.join(albedo_dir, '%06d.png'%count)
            # shutil.copy(src_img_path, tgt_img_path)
            # shutil.copy(src_mask_path, tgt_mask_path)
            mask = Image.open(src_mask_path).convert('L')
            aa = Image.open(src_normal_path); aa.putalpha(mask);aa.save(tgt_normal_path)
            aa = Image.open(src_img_path); aa.putalpha(mask);aa.save(tgt_img_path)
            aa = Image.open(src_albedo_path); aa.putalpha(mask);aa.save(tgt_albedo_path)
            aa = Image.open(src_mask_path); aa.putalpha(mask);aa.save(tgt_mask_path)

            extrinsic = param.item().get('extrinsic')
            intrinsic = param.item().get('intrinsic')
            calib = intrinsic @ extrinsic
            w2c = extrinsic
            axis_adj = np.eye(4)
            axis_adj[1,1] = -1
            axis_adj[2,2] = -1
            w2c = np.matmul(axis_adj, w2c)
            world_mat = K @ w2c
            final_dict['%06d.png'%count] = {
                'K': K.reshape(16).tolist(),
                'W2C': w2c.reshape(16).tolist(),
                'img_size': [width, height],
                'calib':calib.reshape(16).tolist(),
                'intrinsic':intrinsic.reshape(16).tolist(),
                'P':world_mat.reshape(16).tolist()
            }

            cam_params_new['scale_mat_%d' % count] = np.eye(4)
            cam_params_new['world_mat_%d' % count] = world_mat
            cam_params_new['w2c_%d' % count] = w2c
            cam_params_new['c2w_%d' % count] = inv(w2c)
            cam_params_new['K_%d' % count] = K

            count += 1



    json.dump(final_dict, codecs.open(jfpath, 'w', encoding='utf-8'), sort_keys=True, indent=4)
    np.savez(out_cam_path, **cam_params_new)


