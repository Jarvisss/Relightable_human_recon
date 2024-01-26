import torch
from torch.functional import norm
import torch.nn as nn
import numpy as np
import random
import os
from PIL import Image
from PIL.ImageFilter import GaussianBlur, MinFilter, MaxFilter
import torchvision.transforms.functional as F
from torch.nn import init
from torch.autograd import grad
from tqdm import tqdm

color_map = [
    [0,0,0],
    [255, 0, 0],
    [0,255,0],
    [0,0,255],
    [255,255,0],
    [0,255,255],
    [255,0,255],
]

def set_random_seed(seed):
    """
    set random seed
    """
    print('set seed to %d' % seed)
    random.seed(seed) #random and transforms
    np.random.seed(seed) #numpy
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) #gpu
    # torch.backends.cudnn.deterministic=True
    # torch.use_deterministic_algorithms(True)

def to8b(x):
    if x.shape[-1] ==1 :
        x = np.repeat(x, 3, -1)
    return np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)


def init_weights(m, init_type='normal', init_gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find(
            'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        init.normal_(m.weight.data, 1.0, init_gain)
        init.constant_(m.bias.data, 0.0)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    print('initialize network with %s' % init_type)
    net.apply(init_weights)  # apply the initialization function <init_func>
    return net

def make_training_dirs(opt, exp_name, date):
    """
    make visualization, log, checkpoint directories
    """
    # path_to_ckpt_dir = opt.root_dir+ '/checkpoints/{0}/{1}/'.format(date, exp_name)
    # path_to_visualize_dir = opt.root_dir+ '/visualize_result/{0}/{1}/'.format(date, exp_name)
    path_to_ckpt_dir = './checkpoints/{0}/{1}/'.format(date, exp_name)
    path_to_visualize_dir = './visualize_result/{0}/{1}/'.format(date, exp_name)
    

    if not os.path.isdir(path_to_ckpt_dir):
        os.makedirs(path_to_ckpt_dir)
    if not os.path.isdir(path_to_visualize_dir):
        os.makedirs(path_to_visualize_dir)
    return path_to_ckpt_dir, path_to_visualize_dir

def make_testing_dirs(opt, exp_name, date):
    """
    make visualization, log, checkpoint directories
    """
    path_to_test_dir = '/mnt/data2/lujiawei/tex2cloth_bak/test_result/{0}/{1}/'.format(date, exp_name)
    if not os.path.isdir(path_to_test_dir):
        os.makedirs(path_to_test_dir)
    return path_to_test_dir

def get_random_crop(mask, crop_width, crop_height):
    '''
    get random crop in a mask and return the indices  
    :param mask: [H,W] mask image of some view
    :param crop_width: int
    :param crop_height: int
    
    :return 
    :param crop_index:indices of the cropped region
    :param [cx, cy]: center of the cropped region
    :param [top,bottom,left,right]: tblr of the cropped region for further crop and cal vgg loss
    '''
    def get_random_crop_center(mask, border):
        
        xs, ys = torch.where((mask * border) >0.5) ## [1,1,H,W] -> [N]
        ## only sample patches with it's center is inside mask and not exceed border
        total_n = len(xs)
        center_idx = random.randint(0, total_n)
        cx, cy = xs[center_idx], ys[center_idx]
        return cx, cy
    border = torch.ones_like(mask)
    border[:crop_height//2,:] = 0
    border[-crop_height//2:,:] = 0
    border[:,:crop_width//2] = 0
    border[:,-crop_width//2:] = 0
    
    cx, cy = get_random_crop_center(mask, border)
    crop_reshape = torch.zeros_like(mask)
    top = cx - crop_height//2
    bottom = cx + crop_height//2
    left = cy - crop_width//2
    right = cy + crop_width//2
    crop_reshape[top:bottom, left:right] = 1
    ## get cropped mask ids
    cropped_idx = torch.where( crop_reshape.view(-1) > 0.5 )[0]
    return cropped_idx, [cx, cy],[top,bottom,left,right]



def transform_image(image, resize_param, method=Image.BILINEAR, affine=None, normalize=True, toTensor=True, fillWhiteColor=False):
    image = F.resize(image, resize_param, interpolation=method)
    if affine is not None:
        angle, translate, scale = affine['angle'], affine['shift'], affine['scale']
        fillcolor = None if not fillWhiteColor else (255,255,255)
        image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fillcolor=fillcolor)  
    if toTensor:
        image = F.to_tensor(image)
    if image.shape[0] ==1:
        image = torch.cat((image,image,image),dim=0)
    if image.shape[0] > 3:
        image = image[:3,...]
    if normalize:
        image = F.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    return image
def load_img_input(img_path, load_size, toTensor=True,normalize=True):
    return transform_image(Image.open(img_path), load_size, toTensor=toTensor, normalize=normalize)


def extract_bboxes(mask):

    """Compute bounding boxes from masks.

    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].

    """

    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)

    for i in range(mask.shape[-1]):

        m = mask[:, :, i]

        # Bounding box.

        horizontal_indicies = np.where(np.any(m, axis=0))[0]

        # print("np.any(m, axis=0)",np.any(m, axis=0))

        # print("p.where(np.any(m, axis=0))",np.where(np.any(m, axis=0)))

        vertical_indicies = np.where(np.any(m, axis=1))[0]

        if horizontal_indicies.shape[0]:

            x1, x2 = horizontal_indicies[[0, -1]]

            y1, y2 = vertical_indicies[[0, -1]]

            # x2 and y2 should not be part of the box. Increment by 1.

            x2 += 1

            y2 += 1

        else:

            # No mask for this instance. Might happen due to

            # resizing or cropping. Set bbox to zeros

            x1, x2, y1, y2 = 0, 0, 0, 0

        boxes[i] = np.array([y1, x1, y2, x2])

    return boxes.astype(np.int32)

def save_param_to_json(opt, path_to_ckpt):
    import json
    param_file = 'param.json'
    param_path = os.path.join(path_to_ckpt, param_file)
    with open(param_path, 'w') as f:
        json.dump(vars(opt), f, indent=4)

def shading_l2g(linear_shading:torch.Tensor):
    # return torch.clamp(torch.pow(shading,1/2.2)*0.5, 0., 1.)
    # hack for the exponent less than 1
    # linear shading comes from analytic function
    gamma_shading = torch.sign(linear_shading) * torch.pow(torch.abs(linear_shading), 1/2.2)
    return gamma_shading



def img_l2g(linear_input:torch.Tensor):
    # return torch.clamp(torch.pow(input,1/2.2), 0., 1.)
    # hack for the exponent less than 1
    gamma_img = torch.sign(linear_input) * torch.pow(torch.abs(linear_input), 1/2.2)
    return torch.clamp(gamma_img,0 ,1)


def img_g2l(input:torch.Tensor):
    return torch.clamp(torch.pow(input,2.2),0,1)


def crop(image, mask, padding_rate=0.1):
    h_min, w_min, h_max, w_max = extract_bboxes(mask)[0]
    delta = max(h_max - h_min, w_max - w_min)
    pad = int(delta * padding_rate)
    
    delta_h = int((delta - (h_max - h_min))/2)
    delta_w = int((delta - (w_max - w_min))/2)
    low_h = max(0, h_min-delta_h-pad)
    low_w = max(0, w_min-delta_w-pad)

    return image[low_h: h_max+delta_h+pad, low_w: w_max + delta_w + pad, :] 

def crop_padding(image, mask, padding_rate=0.1):
    h_min, w_min, h_max, w_max = extract_bboxes(mask)[0]
    cropped = image[h_min: h_max, w_min: w_max, :] 
    channels = image.shape[-1]
    delta = max(h_max - h_min, w_max - w_min)
    h_pad = (delta - (h_max - h_min))/2 + delta * padding_rate
    w_pad = (delta - (w_max - w_min))/2 + delta * padding_rate

    h_pad = int(h_pad)
    w_pad = int(w_pad)

    HW_max = max(h_max-h_min + h_pad * 2, w_max-w_min + w_pad * 2)

    new_shape = (HW_max, HW_max, channels)
    new_image = np.zeros(new_shape).astype(np.uint8)

    new_image[h_pad:  h_pad + h_max - h_min, w_pad : w_pad + w_max - w_min, :] = cropped.copy()

    return new_image

def adjust_lambda_g1(epoch, init_lg1, max_lg1):
    lambda_g1 = epoch + init_lg1
    lambda_g1 = min(lambda_g1, max_lg1)        

    return lambda_g1



def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        set_learning_rate(optimizer, lr)
    return lr

def set_learning_rate(optimizer, group_id, lr):
    optimizer.param_groups[group_id]['lr'] = lr
    # for param_group in optimizer.param_groups:
        # param_group['lr'] = lr
    # 
    return lr

def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        if thresh == 0.5:
            vol_pred = pred > thresh
            vol_gt = gt > thresh
        elif thresh == 0:
            vol_pred = pred <= thresh
            vol_gt = gt <= thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt


def safe_l2norm(input, dim=1, eps=1e-8):
    return (input+eps) / (input+eps).norm(dim=dim, keepdim=True)

def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    # inputs.requires_grad_(True)
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True)[0]
    return points_grad


def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

def steps_to_zero(start:torch.Tensor, stop:torch.Tensor,step:float):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out



def get_image_patch_by_center(image, center, patch_size=(64,64)):

    '''
    :@param image, [NK, C, H, W]
    :@param center, [NK, 2]
    :@param patch_size, [2]
    
    :@return 
    1. patch_list: list[C, H', W']
    2. bbox_list: list[4]
    '''
    ix = center[:, 0]
    iy = center[:, 1]

    px, py = patch_size

    N, C, IH, IW = image.shape

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)

    ix_ = torch.floor(ix+0.5)
    iy_ = torch.floor(iy+0.5)

    left = ix_ - px // 2
    right = ix_ + px // 2 + 1

    top = iy_ - py //2
    bottom = iy_ + py //2 + 1

    # torch.clamp(x_left, 0, IW-1, out=x_left)
    # torch.clamp(x_right, 0, IW-1, out=x_right)
    # torch.clamp(y_up, 0, IH-1, out=y_up)
    # torch.clamp(y_down, 0, IH-1, out=y_down)

    # patch_list = []
    # bbox_list = []
    # for b in range(N):
    #     patch = image[b, :, y_up[b].long(): y_down[b].long(), x_left[b].long():x_right[b].long()]
    #     patch_list += [patch]
    #     bbox_list += [y_up[b], y_down[b],x_left[b], x_right[b]]
    return top.long(), bottom.long(), left.long(), right.long()



def find_border(img):
    

    img = img.filter(MinFilter(11))
    img = np.array(img)
    
    img_1 = np.sum(img, axis=2) if len(img.shape)>2 else img
    img_x = np.sum(img_1, axis=0)
    img_y = np.sum(img_1, axis=1)
    x_min = img_x.shape[0]
    x_max = 0
    y_min = img_y.shape[0]
    y_max = 0
    for x in range(img_x.shape[0]):
        if img_x[x] > 0:
            x_min = x
            break
    for x in range(img_x.shape[0]-1, 0, -1):
        if img_x[x] > 0:
            x_max = x
            break
    for y in range(img_y.shape[0]):
        if img_y[y] > 0:
            y_min = y
            break
    for y in range(img_y.shape[0]-1, 0, -1):
        if img_y[y] > 0:
            y_max = y
            break
    return x_min, x_max, y_min, y_max

def reshape_multiview_images(image_tensor, calib_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    ).contiguous()

    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    ).contiguous()


    return image_tensor, calib_tensor


def calc_error(net, cuda, dataset, num_tests, thresh=0.5):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
    for idx in tqdm(range(num_tests)):
        data = dataset[idx * len(dataset) // num_tests]
        # retrieve the data
        image_tensor = data['img'].to(device=cuda)
        calib_tensor = data['calib'].to(device=cuda)
        sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
        label_tensor = data['labels'].to(device=cuda).unsqueeze(0)
        surf_tensor = data['surface_samples'].to(device=cuda).unsqueeze(0)
        surf_norm_tensor = data['surface_normals'].to(device=cuda).unsqueeze(0)
        
        error, error_dict = net.forward(image_tensor, sample_tensor, calib_tensor, labels_space=label_tensor, points_surf=surf_tensor, labels_surf=surf_norm_tensor)

        IOU, prec, recall = compute_acc(net.output_pred_space, label_tensor, thresh=thresh)

        # print(
        #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
        #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
        erorr_arr.append(error.item())
        IOU_arr.append(IOU.item())
        prec_arr.append(prec.item())
        recall_arr.append(recall.item())

    return np.average(erorr_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)

def calc_error_color(netG, netC, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    error_color_arr = []
    error_albedo_arr = []
    error_recon_arr = []
    error_light_arr = []

    for idx in tqdm(range(num_tests)):
        data = dataset[idx * len(dataset) // num_tests]
        # retrieve the data
        image_tensor = data['img'].to(device=cuda).unsqueeze(0)
        calib_tensor = data['calib'].to(device=cuda).unsqueeze(0)
        extrinsic_tensor = data['calib'].to(device=cuda).unsqueeze(0)
        color_sample_tensor = data['color_samples'].to(device=cuda).unsqueeze(0)
        labels = {
            'surface_color' : data['surface_color'].float().to(cuda).unsqueeze(0),
            'surface_albedo' : data['surface_albedo'].float().to(cuda).unsqueeze(0),
            'surface_normal' : data['surface_normal'].float().to(cuda).unsqueeze(0),
            'global_light' : data['global_light'].float().to(cuda).unsqueeze(0)
        }
        netG.filter(image_tensor)
        normals = netG.cal_sdf_grad(color_sample_tensor, calib_tensor)
        _, errorC, errorC_dict = netC.forward(image_tensor, netG.get_im_feat(), color_sample_tensor, calib_tensor, labels=labels, netG_normal=normals, extrinsic=extrinsic_tensor)

        # print('{0}/{1} | Error inout: {2:06f} | Error color: {3:06f}'
        #       .format(idx, num_tests, errorG.item(), errorC.item()))
        error_color_arr.append(errorC_dict['color_error'].item())
        error_albedo_arr.append(errorC_dict['albedo_error'].item())
        error_recon_arr.append(errorC_dict['recon_error'].item())
        error_light_arr.append(errorC_dict['light_error'].item())

    return np.average(error_color_arr), np.average(error_albedo_arr), np.average(error_recon_arr), np.average(error_light_arr)