fdim=128
GPU=0
CUDA_VISIBLE_DEVICES=${GPU} python joint_optimize.py\
    --id "joint_optimize_thuman"\
    --finetune_real_data_train_dir './data/0004_24views'\
    --tune_datatype 'real'\
    --dataset 'Thuman2'\
    --num_views 4\
    --epochs 5000\
    --use_perspective\
    --cal_vhull_intersection_online\
    --load_size 512 512\
    --gen_init_mesh\
    --render_init_imgs\
    --use_linear_z\
    --optimize_pose\
    --lambda_pose 10\
    --opt_pose_epoch 500\
    --selected_train\
    --init_intensity 1.5\
    --lr_G 0.0001\
    --lr_pose 0.0001\
    --lr_intensity 0.001\
    --object_bounding_sphere 1\
    --eik_std 0.01\
    --dilate_size 10\
    --use_feature_confidence\
    --random_sample\
    --num_sample_dr 2048\
    --fov 68.4\
    --near 0.1\
    --far 4\
    --patch_size 49\
    --normalize_z\
    --angle_step 30\
    --random_scale\
    --random_trans\
    --offline_sample\
    --align_corner\
    --seed 19951023\
    --size 512\
    --batch_size 1\
    --epochs_warm_up 0\
    --z_size 2000\
    --sigma 0.1\
    --field_type 'sdf'\
    --filter_type 'UNet_s_deeper_128'\
    --num_sample_surface 5000\
    --num_sample_inout 5000\
    --num_sample_color 5000\
    --num_workers 0\
    --lambda_g1 1\
    --lambda_g2 1\
    --lambda_g1_end 15.0\
    --lambda_reg 0.1\
    --lambda_mask 50\
    --lambda_sign 0.2\
    --use_align_loss\
    --lambda_align 0.01\
    --use_backnormal_loss\
    --norm 'group'\
    --mlp_activation_type 'silu'\
    --mlp_activation_type_albedo 'silu'\
    --mlp_activation_type_light 'silu'\
    --mlp_norm_type 'none'\
    --mlp_norm_type_albedo 'none'\
    --mlp_norm_type_light 'none'\
    --mlp_dim ${fdim} 1024 512 256 128 1\
    --mlp_dim_albedo ${fdim} 1024 512 256 128 3\
    --mlp_dim_spec_albedo ${fdim} 1024 512 256 128 3\
    --mlp_dim_roughness ${fdim} 1024 512 256 128 1\
    --mlp_dim_light 256 256 128 64 32 28\
    --mlp_dim_indirect 132 1024 512 256 128 3\
    --mlp_res_layers_albedo 1 2 3\
    --mlp_mean_layer_albedo 3\
    --mlp_res_layers 1 2 3\
    --mlp_mean_layer 3\
    --use_transformer\
    --transformer_geo_d_model ${fdim}\
    --transformer_geo_d_inner 128\
    --transformer_tex_d_model ${fdim}\
    --transformer_tex_d_inner 128\
    --transformer_tex_dropout 0\
    --transformer_geo_dropout 0\
    --gamma 0.9\
    --k_type 'exp'\
    --init_k 5\
    --model_save_freq 1000\
    --gen_mesh_freq 100\
    --save_sample_points_freq 5000\
    --vis_sdf_normal_freq 0\
    --gen_vis_hull_freq 0\
    --visibility_save_freq 0\
    --image_proj_save_freq 100\
    --image_patch_save_freq 100\
    --render_img_freq 100\
    --test_freq 100\
    --no_num_eval\
    --resolution 128\
    --samp_indirect 1