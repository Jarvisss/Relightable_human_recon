import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import os
from utils.flow_utils import flow2img
from model.blocks import warp_flow
from utils.common_utils import color_map
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def vis_image_feature():
    pass

def vis_pt_normal_arrow(points_surf, points_surf_normal,save_surface_arrow_path, ratio=0.2, arrow_scale=0.2, fig_size=20):
    fig = plt.figure(figsize=(fig_size,fig_size))
    ax = fig.add_subplot(111, projection='3d')
    ## visualize the arrow of normal
    ax.plot(points_surf[:,0], points_surf[:,1], points_surf[:,2], 'o', markersize=1, color='g', alpha=1)
    ax.plot(points_surf[:,0] + points_surf_normal[:,0], points_surf[:,1] + points_surf_normal[:,1], points_surf[:,2] + points_surf_normal[:,2], 'o', markersize=1, color='r', alpha=0)
    
    all_ids = (list(range(points_surf.shape[0])))
    random.shuffle(all_ids)
    vis_ids = all_ids[:int(points_surf.shape[0] * ratio)]
    for i in (vis_ids):
        a = Arrow3D(
            [points_surf[i, 0], points_surf[i,0] + points_surf_normal[i,0] * arrow_scale], 
            [points_surf[i, 1], points_surf[i,1] + points_surf_normal[i,1] * arrow_scale], 
            [points_surf[i, 2], points_surf[i,2] + points_surf_normal[i,2] * arrow_scale],
            mutation_scale=5, 
            lw=1, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')
    ax.view_init(110,-85)
    plt.draw()
    plt.savefig(save_surface_arrow_path)
    plt.close(fig)
    
    pass

def vis_sdf_x_plane( sdf, save_dir, b_min, b_max, depth=128, res=256):
    save_img_dir = save_dir + '_sdf_x'
    os.makedirs(save_img_dir, exist_ok=True) 


    y, x = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, sdf[depth,:,:], cmap='RdBu', vmin=sdf.min(), vmax=sdf.max(), shading='auto')
    ax.set_title('yz plane at x=%d' % depth)
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    save_img_path = os.path.join(save_img_dir, '%d.png' % depth)

    plt.savefig(save_img_path)
    plt.close(fig)

def vis_sdf_y_plane( sdf, save_dir, bmin, bmax, depth=0, resx=256, resz=256):
    save_img_dir = save_dir + '_sdf_y'
    os.makedirs(save_img_dir, exist_ok=True)
    
    xmin, ymin, zmin = bmin
    xmax, ymax, zmax = bmax

    z, x = np.meshgrid(np.linspace(zmin, zmax, resz), np.linspace(xmin, xmax, resx))

    fig = plt.figure()
    inch_size = 5
    fig.set_size_inches(resx/resz * inch_size, inch_size, forward=False) ## set figure aspect
    ax = plt.Axes(fig, [0., 0., 1., 1.]) ## set fit whole figure
    ax.set_axis_off()
    fig.add_axes(ax)
    cs = ax.contourf(x, z, sdf[:,depth,:], 20)
    ct = ax.contour(cs, 10, linewidths=.5,colors='k')
    ax.clabel(ct, inline=True, fontsize=5)
    # c = ax.pcolormesh(x, y, sdf[:,:,depth], cmap='RdBu', vmin=sdf.min(), vmax=sdf.max(), shading='auto')
    save_img_path = os.path.join(save_img_dir, '%d.png' % depth)
    plt.show()
    # plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0,  dpi = resy)
    plt.savefig(save_img_path, dpi = resz)
    plt.close()

def vis_sdf_z_plane( sdf, save_dir, bmin, bmax, depth=0, resx=256, resy=256):
    import pdb
    save_img_dir = save_dir + '_sdf_z'
    os.makedirs(save_img_dir, exist_ok=True)
    
    xmin, ymin, zmin = bmin
    xmax, ymax, zmax = bmax

    y, x = np.meshgrid(np.linspace(ymin, ymax, resy), np.linspace(xmin, xmax, resx))

    fig = plt.figure()
    inch_size = 5
    fig.set_size_inches(resx/resy * inch_size, inch_size, forward=False) ## set figure aspect
    ax = plt.Axes(fig, [0., 0., 1., 1.]) ## set fit whole figure
    ax.set_axis_off()
    fig.add_axes(ax)
    cs = ax.contourf(x, y, sdf[:,:,depth], 20)
    ct = ax.contour(cs, 20, linewidths=.5,colors='k')
    ax.clabel(ct, inline=True, fontsize=5)
    # c = ax.pcolormesh(x, y, sdf[:,:,depth], cmap='RdBu', vmin=sdf.min(), vmax=sdf.max(), shading='auto')
    save_img_path = os.path.join(save_img_dir, '%d.png' % depth)
    plt.show()
    # plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0,  dpi = resy)
    plt.savefig(save_img_path, dpi = resy)
    plt.close()

def save_contour_image(data, cm, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.contour()
    plt.savefig(fn, dpi = height) 
    plt.close()

#-1 1 to 0 255
def to_image(tensor, denorm=False):
    assert(len(tensor.shape)==3)
    if tensor.shape[2]==1:
        tensor = torch.cat((tensor,)*3, dim=2)

    if denorm:
        return (tensor+1) * 127.5
    else:
        return tensor * 255.0



def tensor2im(x, display_batch=0,is_mask=False, out_size=(256,256)):
    '''
    take 4-d tensor as input, [B C H W]
    output 3-d image tensor, range(0,255), [H W C]
    output white image if tensor is None
    '''
    
    device = torch.device("cuda:0")
    if x.is_cuda:
        img = torch.ones((out_size[0],out_size[1],3)).to(device) * 255.0
    else:
        img = torch.ones((out_size[0],out_size[1],3)) * 255.0
    
    if x is None:
        return img
    if x.shape[-1] < 2:
        return img
    tensor = x.clone().detach()
    assert len(tensor.shape)==4
    tensor = F.interpolate(tensor, size=out_size, mode='bilinear',align_corners=True)
    img = tensor[display_batch].permute(1,2,0) 
    img = to_image(img)

    return img



def visualize(unlit_a, lit_a,normal,unlit_b,lit_b_gen, lit_b_gt):
    cpu = torch.device("cpu")
    device = torch.device("cuda:0")
    DISPLAY_BATCH = 0
    
    vis_unlit_a = to_image(unlit_a[DISPLAY_BATCH].permute(1,2,0))
    vis_lit_a = to_image(lit_a[DISPLAY_BATCH].permute(1,2,0))
    vis_normal = to_image(normal[DISPLAY_BATCH].permute(1,2,0))
    vis_unlit_b = to_image(unlit_b[DISPLAY_BATCH].permute(1,2,0))
    vis_lit_b_gen = to_image(lit_b_gen[DISPLAY_BATCH].permute(1,2,0))
    vis_lit_b_gt = to_image(lit_b_gt[DISPLAY_BATCH].permute(1,2,0))

    simp_img = torch.cat((vis_unlit_a, vis_lit_a,vis_normal, vis_unlit_b, vis_lit_b_gen,vis_lit_b_gt),dim=1)
    simp_img = simp_img.type(torch.uint8).to(cpu).numpy()
    return simp_img


def visualize_img_group(img_group, denorm=False):
    ''' input torch tensor images in range [-1, 1]
    '''
    cpu = torch.device("cpu")
    DISPLAY_BATCH = 0
    simp_img = torch.cat([to_image(img[DISPLAY_BATCH].permute(1,2,0), denorm) for img in img_group],dim=1)
    simp_img = simp_img.type(torch.uint8).to(cpu).numpy()
    return simp_img

    
def visualize_flow(img_b, im_a, render_b):
    cpu = torch.device("cpu")
    device = torch.device("cuda:0")
    DISPLAY_BATCH = 0
    out_size = im_a[0].shape[2:]
    
    im_b_img = to_image(img_b[DISPLAY_BATCH].permute(1,2,0))
    im_a_img = to_image(im_a[DISPLAY_BATCH].permute(1,2,0))
    render_b_img = to_image(render_b[DISPLAY_BATCH].permute(1,2,0))

    final_img = torch.cat((im_b_img, im_a_img, render_b_img), dim=1)
    
    final_img = final_img.type(torch.uint8).to(cpu).numpy()
    
    return final_img

def visualize_label(label, n_class):
    cpu = torch.device("cpu")
    DISPLAY_BATCH = 0
    label_numpy = label[DISPLAY_BATCH].type(torch.uint8).to(cpu).numpy()
    H, W = label_numpy.shape
    label_vis = np.zeros((H,W,3), dtype=np.uint8)
    for c in range(n_class):
        label_vis[label_numpy==c] = color_map[c % len(color_map)]
    return label_vis




