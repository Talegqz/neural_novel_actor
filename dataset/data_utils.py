# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import functools
import cv2
import math
import numpy as np
import imageio
from glob import glob
import os
import sys
import copy
import shutil
import skimage.metrics
import pandas as pd
import pylab as plt
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

def get_rank():    
    try:
        return du.get_rank()
    except AssertionError:
        return 0


def get_world_size():
    try:
        return du.get_world_size()
    except AssertionError:
        return 1
        

def parse_views(view_args):
    output = []
    try:
        xx = view_args.split(':')
        ids = xx[0].split(',')
        for id in ids:
            if '..' in id:
                a, b = id.split('..')
                output += list(range(int(a), int(b)))
            else:
                output += [int(id)]
        if len(xx) > 1:
            output = output[::int(xx[-1])]
    except Exception as e:
        raise Exception("parse view args error: {}".format(e))

    # print("parse: {} views".format(len(output)))
    # print("views are ",output)
    return output

def parse_character(view_args):
    output = []
    if view_args == '-1':
        return output
    try:
        xx = view_args.split(':')
        ids = xx[0].split(',')
        for id in ids:
            if '..' in id:
                a, b = id.split('..')
                output += list(range(int(a), int(b)))
            else:
                output += [int(id)]
        if len(xx) > 1:
            output = output[::int(xx[-1])]
    except Exception as e:
        raise Exception("parse characters args error: {}".format(e))

    print("parse: {} characters".format(len(output)))
    return output

def get_uv(H, W, h, w):
    """
    H, W: real image (intrinsics)
    h, w: resized image
    """
    uv = np.flip(np.mgrid[0: h, 0: w], axis=0).astype(np.float32)
    uv[0] = uv[0] * float(W / w)
    uv[1] = uv[1] * float(H / h)
    return uv, [float(H / h), float(W / w)]


def load_texcoords(filename):
    vt, ft = [], []
    for content in open(filename):
        contents = content.strip().split(' ')
        if contents[0] == 'vt':
            vt.append([float(a) for a in contents[1:]])
        if contents[0] == 'f':
            ft.append([int(a.split('/')[1]) for a in contents[1:] if a])
    return np.array(vt, dtype='float64'), np.array(ft, dtype='int32') - 1

def load_face(filename):
    ft = []
    for content in open(filename):
        contents = content.strip().split(' ')
        if contents[0] == 'f':
            ft.append([int(a) for a in contents[1:] if a])
    return np.array(ft, dtype='int32') - 1


def load_rgb(
    path, 
    resolution=None, 
    with_alpha=True, 
    bg_color=[1.0, 1.0, 1.0],
    min_rgb=-1,
    interpolation='AREA'):
    if with_alpha:
        img = imageio.imread(path)  # RGB-ALPHA
    else:
        img = imageio.imread(path)[:, :, :3]
    img = skimage.img_as_float32(img).astype('float32')
    if not with_alpha:
        return img

    H, W, D = img.shape
    if resolution is not None:
        h, w = resolution
    else:
        h, w = H, W

    if D == 3:
        img = np.concatenate([img, np.ones((img.shape[0], img.shape[1], 1))], -1).astype('float32')
    
    uv, ratio = get_uv(H, W, h, w)
    if (h < H) or (w < W):
        # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype('float32')
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype('float32')

    if min_rgb == -1:  # 0, 1  --> -1, 1
        img[:, :, :3] -= 0.5
        img[:, :, :3] *= 2.

    img[:, :, :3] = img[:, :, :3] * img[:, :, 3:] + np.asarray(bg_color)[None, None, :] * (1 - img[:, :, 3:])
    img[:, :, 3] = img[:, :, 3] * (img[:, :, :3] != np.asarray(bg_color)[None, None, :]).any(-1) 
    img = img.transpose(2, 0, 1)
    
    return img, uv, ratio


def load_depth(path, resolution=None, depth_plane=5):
    if path is None:
        return None
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # ret, img = cv2.threshold(img, depth_plane, depth_plane, cv2.THRESH_TRUNC)
    
    H, W = img.shape[:2]
    h, w = resolution
    if (h < H) or (w < W):
        img  = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype('float32')
        #img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    if len(img.shape) ==3:
        img = img[:,:,:1]
        img = img.transpose(2,0,1)
    else:
        img = img[None,:,:]
    return img


def load_mask(path, resolution=None):
    if path is None:
        return None
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    h, w = resolution
    H, W = img.shape[:2]
    if (h < H) or (w < W):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype('float32')
    img = img / (img.max() + 1e-7)
    return img


def load_matrix(path):
    lines = [[float(w) for w in line.strip().split()] for line in open(path)]
    if len(lines[0]) == 2:
        lines = lines[1:]
    if len(lines[-1]) == 2:
        lines = lines[:-1]
    return np.array(lines).astype(np.float32)

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose
    
def load_intrinsics(filepath, resized_width=None, invert_y=False):
    try:
        intrinsics = load_matrix(filepath)
        if intrinsics.shape[0] == 3 and intrinsics.shape[1] == 3:
            _intrinsics = np.zeros((4, 4), np.float32)
            _intrinsics[:3, :3] = intrinsics
            _intrinsics[3, 3] = 1
            intrinsics = _intrinsics
        if intrinsics.shape[0] == 1 and intrinsics.shape[1] == 16:
            intrinsics = intrinsics.reshape(4, 4)
        return intrinsics
    except ValueError:
        pass

    # Get camera intrinsics
    with open(filepath, 'r') as file:
        
        f, cx, cy, _ = map(float, file.readline().split())
    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])
    return full_intrinsic

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

    
def unflatten_img(img, width=512):
    sizes = img.size()
    height = sizes[-1] // width
    return img.reshape(*sizes[:-1], height, width)


def square_crop_img(img):
    if img.shape[0] == img.shape[1]:
        return img  # already square

    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def sample_pixel_from_image(
    num_pixel, num_sample, 
    mask=None, ratio=1.0,
    use_bbox=False, 
    center_ratio=1.0,
    width=512,
    patch_size=1):
    
    if patch_size > 1:
        assert (num_pixel % (patch_size * patch_size) == 0) \
            and (num_sample % (patch_size * patch_size) == 0), "size must match"
        _num_pixel = num_pixel // (patch_size * patch_size)
        _num_sample = num_sample // (patch_size * patch_size)
        height = num_pixel // width

        _mask = None if mask is None else \
            mask.reshape(height, width).reshape(
                height//patch_size, patch_size, width//patch_size, patch_size
            ).any(1).any(-1).reshape(-1)
        _width = width // patch_size
        _out = sample_pixel_from_image(_num_pixel, _num_sample, _mask, ratio, use_bbox, _width)
        _x, _y = _out % _width, _out // _width
        x, y = _x * patch_size, _y * patch_size
        x = x[:, None, None] + np.arange(patch_size)[None, :, None] 
        y = y[:, None, None] + np.arange(patch_size)[None, None, :]
        out = x + y * width
        return out.reshape(-1)

    if center_ratio < 1.0:
        r = (1 - center_ratio) / 2.0
        H, W = num_pixel // width, width
        mask0 = np.zeros((H, W))
        mask0[int(H * r): H - int(H * r), int(W * r): W - int(W * r)] = 1
        mask0 = mask0.reshape(-1)

        if mask is None:
            mask = mask0
        else:
            mask = mask * mask0
    
    if mask is not None:
        mask = (mask > 0.0).astype('float32')

    if (mask is None) or \
        (ratio <= 0.0) or \
        (mask.sum() == 0) or \
        ((1 - mask).sum() == 0):
        return np.random.choice(num_pixel, num_sample)

    if use_bbox:
        mask = mask.reshape(-1, width)
        x, y = np.where(mask == 1)
        mask = np.zeros_like(mask)
        mask[x.min(): x.max()+1, y.min(): y.max()+1] = 1.0
        mask = mask.reshape(-1)

    try:
        probs = mask * ratio / (mask.sum()) + (1 - mask) / (num_pixel - mask.sum()) * (1 - ratio)
        # x = np.random.choice(num_pixel, num_sample, True, p=probs)
        return np.random.choice(num_pixel, num_sample, True, p=probs)
    
    except Exception:
        return np.random.choice(num_pixel, num_sample)


def colormap(dz):
    return plt.cm.jet(dz)
    # return plt.cm.viridis(dz)
    # return plt.cm.gray(dz)


def recover_image(img, min_val=-1, max_val=1, width=512, bg=None, weight=None, raw=False):
    if raw: return img

    sizes = img.size()
    height = sizes[0] // width
    img = img.float().to('cpu')

    if len(sizes) == 1 and (bg is not None):
        bg_mask = img.eq(bg)[:, None].type_as(img)

    img = ((img - min_val) / (max_val - min_val)).clamp(min=0, max=1)
    if len(sizes) == 1:
        img = torch.tensor(colormap(img.numpy())[:, :3])
    if weight is not None:
        weight = weight.float().to('cpu')
        img = img * weight[:, None]

    if bg is not None:
        img = img * (1 - bg_mask) + bg_mask
    img = img.reshape(height, width, -1)
    return img

    
def write_images(writer, images, updates): 
    for tag in images:
        img = images[tag]
        tag, dataform = tag.split(':')
        writer.add_image(tag, img, updates, dataformats=dataform)


def compute_psnr(p, t):
    """Compute PSNR of model image predictions.
    :param prediction: Return value of forward pass.
    :param ground_truth: Ground truth.
    :return: (psnr, ssim): tuple of floats
    """
    ssim = skimage.metrics.structural_similarity(p, t, multichannel=True, data_range=1)
    psnr = skimage.metrics.peak_signal_noise_ratio(p, t, data_range=1)
    return ssim, psnr


def save_point_cloud(filename, xyz, rgb=None):
    if rgb is None:
        vertex = np.array([(xyz[k, 0], xyz[k, 1], xyz[k, 2]) for k in range(xyz.shape[0])], 
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    else:
        vertex = np.array([(xyz[k, 0], xyz[k, 1], xyz[k, 2], rgb[k, 0], rgb[k, 1], rgb[k, 2]) for k in range(xyz.shape[0])], 
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(filename)
    # from fairseq import pdb; pdb.set_trace()
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(open(filename, 'wb'))

def parse_extrinsics(extrinsics, world2camera=True):
    """ this function is only for numpy for now"""
    if extrinsics.shape[0] == 3 and extrinsics.shape[1] == 4:
        extrinsics = np.vstack([extrinsics, np.array([[0, 0, 0, 1.0]])])
    if extrinsics.shape[0] == 1 and extrinsics.shape[1] == 16:
        extrinsics = extrinsics.reshape(4, 4)
    if world2camera:
        extrinsics = np.linalg.inv(extrinsics).astype(np.float32)
    return extrinsics

class InfIndex(object):
    def __init__(self, index_list, shuffle=False):
        self.index_list = index_list
        self.size = len(index_list)
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        if self.shuffle:
            self._perm = np.random.permutation(self.index_list).tolist()
        else:
            self._perm = copy.deepcopy(self.index_list)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return self.size


# class Timer(StopwatchMeter):
#     def __enter__(self):
#         """Start a new timer as a context manager"""
#         self.start()
#         return self

#     def __exit__(self, *exc_info):
#         """Stop the context manager timer"""
#         self.stop()


# class GPUTimer(object):
#     def __enter__(self):
#         """Start a new timer as a context manager"""
#         self.start = torch.cuda.Event(enable_timing=True)
#         self.end = torch.cuda.Event(enable_timing=True)
#         self.start.record()
#         self.sum = 0
#         return self

#     def __exit__(self, *exc_info):
#         """Stop the context manager timer"""
#         self.end.record()
#         torch.cuda.synchronize()
#         self.sum = self.start.elapsed_time(self.end) / 1000.

def parse_views(view_args):
    output = []
    try:
        xx = view_args.split(':')
        ids = xx[0].split(',')
        for id in ids:
            if '..' in id:
                a, b = id.split('..')
                output += list(range(int(a), int(b)))
            else:
                output += [int(id)]
        if len(xx) > 1:
            output = output[::int(xx[-1])]
    except Exception as e:
        raise Exception("parse view args error: {}".format(e))

    print("parse: {} views".format(len(output)))
    return output

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = base_utils.project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box


class GaussianNoise(torch.nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma = 1.0, is_train=True, is_relative_detach=True, device = 'cpu'):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.training = is_train
        self.device = device
        self.register_buffer('noise', torch.tensor(0).to(self.device))


    def sample(self, size, x=None):
        if x is not None: 
            if self.training and self.sigma != 0:
                scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
                sampled_noise = self.noise.expand(*x.size).float().normal_() * scale
                output = x + sampled_noise
        else: 
            scale = self.sigma  
            sampled_noise = self.noise.expand(*size).float().normal_() * scale    
            output = sampled_noise      

        return output

from plyfile import PlyData, PlyElement
import numpy  as np
def write_ply(points, filename, text=False):
    """
    input: Nx3, write points to filename as PLY format.
    """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)
        
from plyfile import PlyData, PlyElement
import numpy  as np
def write_colored_ply(points, color, filename, text=False):
    """
    input: Nx3, write points to filename as PLY format.
    """
    # points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    # vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # color_data = [(color[i, 0], color[i, 1], color[i, 2]) for i in range(color.shape[0])]
    # color_data = np.array(color_data, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    # color = PlyElement.describe(color_data,'color')
    # with open(filename, mode='wb') as f:
    #     PlyData([el,color], text=text).write(f)
    vertices = np.empty(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'] = points[:,0].astype('f4')
    vertices['y'] = points[:,1].astype('f4')
    vertices['z'] = points[:,2].astype('f4')
    vertices['red'] = color[:,0].astype('u1')
    vertices['green'] = color[:,1].astype('u1')
    vertices['blue'] = color[:,0].astype('u1')
    # save as ply
    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write(filename)

from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

def gen_rays_between(data, idx_0, idx_1, ratio, Th, R, resolution_level=1):
    """
    Interpolate pose between two cameras.
    """
    H,W = data['image_size']
    l = resolution_level
    tx = torch.linspace(0, W - 1, W // l)
    ty = torch.linspace(0, H - 1, H // l)
    pixels_x, pixels_y = torch.meshgrid(tx, ty)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).cuda()  # W, H, 3

    p = torch.matmul(data['all_intrinsics_inv'][0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    rays_v = F.normalize(p, p=2, dim=-1)  # W, H, 3
    pose_all = data['all_pose']
    import pdb
    pdb.set_trace()
    pose_0 = pose_all[idx_0].detach().cpu().numpy()
    pose_1 = pose_all[idx_1].detach().cpu().numpy()
    #trans = ((pose_all[idx_0, :3, 3].cpu().numpy() - Th)@ R) * (1.0 - ratio) + ((pose_all[idx_1, :3, 3].cpu().numpy() - Th) @ R) * ratio
    trans = (pose_all[idx_0, :3, 3].cpu().numpy()) * (1.0 - ratio) + (pose_all[idx_1, :3, 3].cpu().numpy()) * ratio
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    
    rot_0 = pose_0[:3, :3] 
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)
    rot = torch.tensor(pose[:3, :3]).cuda()
    trans = torch.tensor(pose[:3, 3]).cuda()
    rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
    return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

# def gen_my_rays_between(data, idx_0, idx_1, ratio, Th, R, resolution_level=1):
#     """
#     Interpolate pose between two cameras.
#     """
#     H,W = data['image_size']
#     l = resolution_level
#     tx = torch.linspace(0, W - 1, W // l)
#     ty = torch.linspace(0, H - 1, H // l)
#     pixels_x, pixels_y = torch.meshgrid(tx, ty)
#     all_ext = data['all_ext']
#     all_ixt = data['all_ixt']
#     all_R = all_ext[:,:3,:3].cpu().numpy()
#     all_T = all_ext[:,:3,3].cpu().numpy()
#     all_K = all_ixt[:,:3,:3].cpu().numpy()
    
    
#     p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).cuda()  # W, H, 3
    
#     # p = torch.matmul(data['all_intrinsics_inv'][0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
#     # rays_v = F.normalize(p, p=2, dim=-1)  # W, H, 3
# #    pose_all = data['all_pose']

#     # pose_0 = all_ixt[idx_0].cpu().numpy()
#     # pose_1 = all_ixt[idx_1].cpu().numpy()
#     #trans = ((pose_all[idx_0, :3, 3].cpu().numpy() - Th)@ R) * (1.0 - ratio) + ((pose_all[idx_1, :3, 3].cpu().numpy() - Th) @ R) * ratio
#     # pose_0 = np.linalg.inv(pose_0)
#     # pose_1 = np.linalg.inv(pose_1)
    
#     rot_0 = all_R[idx_0] 
#     rot_1 = all_R[idx_1]
#     pose_0 = -np.dot(rot_0.T, all_T[idx_0]).ravel()
#     pose_1 = -np.dot(rot_1.T, all_T[idx_1]).ravel()
    
#     rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
#     key_times = [0, 1]
#     slerp = Slerp(key_times, rots)
#     rot = slerp(ratio)
#     pose = np.diag([1.0, 1.0, 1.0, 1.0])
#     pose = pose.astype(np.float32)
#     pose[:3, :3] = rot.as_matrix()
#     pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)
#     #pose = np.linalg.inv(pose)
#     rot = torch.tensor(pose[:3, :3]).cuda()
#     trans = torch.tensor(pose[:3, 3]).cuda()
#     # pixel_camera = np.dot(p,np.linalg.inv((all_K[idx_0]+all_K[idx_1])/2).T)
#     # pixel_world = np.dot(pixel_camera- trans.ravel())
#     # rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
#     # rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
#     ray_o = np.broadcast(trans,ray_d.shape)
#     # return rays_o.transpose(0, 1), rays_v.transpose(0, 1)
#     return ray_o, ray_d


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def lookat_ext(eye, target, up=(0, 1, 0)):
    fwd = np.asarray(target, np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.float64([right, down, fwd])
    tvec = -np.dot(R, eye)
    return R, tvec

def look_at(origin, target, world_up=np.array([0, 1, 0], dtype=np.float32)):
    """
    Get 4x4 camera to world space matrix, for camera looking at target
    """
    back = origin - target
    back /= np.linalg.norm(back)
    right = np.cross(world_up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)

    cam_to_world = np.empty((4, 4), dtype=np.float32)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = back
    cam_to_world[:3, 3] = origin
    cam_to_world[3, :] = [0, 0, 0, 1]
    return cam_to_world


def generate_cameras(dist=10, view_num=60, target= [0,0,0]):
    cams = []
    #target = [0, 0.95, 0]
    # up = [0, 0, 1]
    up = [0, 1, 0]
    for view_idx in range(view_num):
        angle = (math.pi * 2 / view_num) * view_idx
        eye = np.asarray([dist * math.sin(angle), 0, dist * math.cos(angle)])
        # eye = np.asarray([dist * math.cos(angle),dist * math.sin(angle),1.0])
        # eye = np.asarray([dist * math.cos(angle),dist * math.sin(angle),0.0])
        fwd = np.asarray(target, np.float64) - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)

        cams.append(
            {
                'center': eye, 
                'direction': fwd, 
                'right': right, 
                'up': -down, 
            }
        )

    return cams

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def fix_seed(seed):
    # # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
import random
import torchvision
import warnings
from skimage import img_as_ubyte, img_as_float
import PIL
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self):
        if self.brightness > 0:
            self.brightness_factor = random.uniform(
                max(0, 1 - self.brightness), 1 + self.brightness)
        else:
            self.brightness_factor = None

        if self.contrast > 0:
            self.contrast_factor = random.uniform(
                max(0, 1 - self.contrast), 1 + self.contrast)
        else:
            self.contrast_factor = None

        if self.saturation > 0:
            self.saturation_factor = random.uniform(
                max(0, 1 - self.saturation), 1 + self.saturation)
        else:
            self.saturation_factor = None

        if self.hue > 0:
            self.hue_factor = random.uniform(-self.hue, self.hue)
        else:
            self.hue_factor = None
        # return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            # brightness, contrast, saturation, hue = self.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if self.brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, self.brightness_factor))
            if self.saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, self.saturation_factor))
            if self.hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, self.hue_factor))
            if self.contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, self.contrast_factor))
            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage()] + img_transforms + [np.array,
                                                                                                     img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_clip = []
                for img in clip:
                    jittered_img = img
                    for func in img_transforms:
                        jittered_img = func(jittered_img)
                    jittered_clip.append(jittered_img.astype('float32'))
        elif isinstance(clip[0], PIL.Image.Image):
            print("PIL augmentation not supported")
            exit(1)
            # brightness, contrast, saturation, hue = self.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all videos
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return jittered_clip
    
from torchvision import transforms
# class nph_color_jitter():
#     def __init__(self):
#         self.jitter = 0    
#         self.ops = []
#         self.ops.extend(
#         [transforms.ColorJitter(brightness=(1.0 - self.jitter, 1.0 + self.jitter),
#                                 contrast=(1.0 - self.jitter, 1.0 + self.jitter), saturation=(1.0 - self.jitter, 1.0 + self.jitter),
#                                 hue=(-self.jitter, self.jitter)), ]
#     )
#     def get_ops(self):
#         self.ops = []
#         self.ops.extend(
#         [transforms.ColorJitter(brightness=(1.0 - self.jitter, 1.0 + self.jitter),
#                                 contrast=(1.0 - self.jitter, 1.0 + self.jitter), saturation=(1.0 - self.jitter, 1.0 + self.jitter),
#                                 hue=(-self.jitter, self.jitter)), ]
#     )
        
def nhp_ori_color_jitter():
    ops = []

    ops.extend(
        [transforms.ColorJitter(brightness=(0.2, 2),
                                contrast=(0.3, 2), saturation=(0.2, 2),
                                hue=(-0.5, 0.5)), ]
    )
    return transforms.Compose(ops)


def nhp_color_jitter(jitter = 0.0):
    ops = []

    # ops.extend(
    #     [transforms.ColorJitter(brightness=(0.2, 2),
    #                             contrast=(0.3, 2), saturation=(0.2, 2),
    #                             hue=(-0.5, 0.5)), ]
    # )
    # ops.extend(
    #     [transforms.ColorJitter(brightness=(0.8, 1.2),
    #                             contrast=(0.8, 1.2), saturation=(0.8, 1.2),
    #                             hue=(-0.2, 0.2)), ]
    # )
    ops.extend(
        [transforms.ColorJitter(brightness=(1.0 - jitter, 1.0 + jitter),
                                contrast=(1.0 - jitter, 1.0 + jitter), saturation=(1.0 - jitter, 1.0 + jitter),
                                hue=(-jitter, jitter)), ]
    )
    return transforms.Compose(ops)

def extreme_hue_color_jitter(jitter = 0.0):
    ops = []

    ops.extend(
        [transforms.ColorJitter(
                brightness=(1.0 - jitter/2, 1.0 + jitter),
                contrast=(1.0 - jitter/2, 1.0 + jitter),
                saturation=(1.0 - jitter/2, 1.0 + jitter),
                hue=(-jitter, jitter)), 
                ]
    )
    return transforms.Compose(ops)

def hue_color_jitter(jitter = 0.0):
    ops = []

    # ops.extend(
    #     [transforms.ColorJitter(brightness=(0.2, 2),
    #                             contrast=(0.3, 2), saturation=(0.2, 2),
    #                             hue=(-0.5, 0.5)), ]
    # )
    # ops.extend(
    #     [transforms.ColorJitter(brightness=(0.8, 1.2),
    #                             contrast=(0.8, 1.2), saturation=(0.8, 1.2),
    #                             hue=(-0.2, 0.2)), ]
    # )
    ops.extend(
        [transforms.ColorJitter(brightness=(1.0 - jitter/4, 1.0 + jitter/2),
                                contrast=(1.0 - jitter/4, 1.0 + jitter/2), saturation=(1.0 - jitter/4, 1.0 + jitter/2),
                                hue=(-jitter, jitter)), ]
    )
    return transforms.Compose(ops)

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def distributed_concat(tensor, num_total_examples):
    full_tensor = -torch.ones(num_total_examples)
    full_tensor[:tensor.size(0)] = tensor
    tensor = full_tensor
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)

    true_concat = torch.zeros(num_total_examples)
    true_concat = concat[concat!=-1]
    # truncate the dummy elements added by SequentialDistributedSampler
    return true_concat

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

