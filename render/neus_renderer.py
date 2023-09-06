import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
from clib._ext import mesh2sdf_gpu
from dataset.data_utils import write_ply
import time

def extract_fields(bound_min, bound_max, resolution, query_func, feature_net):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    if feature_net:
                        feature_net.fuse_codes(posed_space=True)    
                        posed_features, can_pts = feature_net(pts.reshape(-1,3).unsqueeze(0),posed_space=True)
                        feature_net.fuse_codes(posed_space=False)
                        canonical_features, _ = feature_net(can_pts.reshape(-1,3).unsqueeze(0),posed_space=False)
                        # features = feature_net(pts.view(-1,3).unsqueeze(0)).squeeze(0)
                        # features, can_pts = feature_net(pts.view(-1,3).unsqueeze(0))
                    else:
                        features = None
                    val = query_func(can_pts,canonical_features.squeeze(0)).reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_fields_with_smpl_sdf(bound_min, bound_max, resolution, query_func, feature_net, args, device='cpu'):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    triangles = feature_net.sp_input['smpl_posed_triangles']

    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    transformation, can_pts = feature_net.get_inverse_transform(pts.reshape(-1,3).unsqueeze(0))
                    val = query_func(can_pts,None).reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                   
                    if not args.is_train:
                        sdf = -mesh2sdf_gpu(pts.contiguous(),triangles)[0].clamp(-1.05,1.05).reshape(64,64,64).cpu().numpy()
                        val[sdf>args.min_dis_eps * args.scale_size] = 1
                        val[sdf<-args.min_dis_eps * args.scale_size] = -1 

                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, feature_net, args, device):
    print('threshold: {}'.format(threshold))
    u = extract_fields_with_smpl_sdf(bound_min, bound_max, resolution, query_func, feature_net, args, device)
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    del u
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]).to(cdf.device)
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(cdf.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

class NeuSRenderer(nn.Module):
    def __init__(self,
                 args,
                 conf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 feature_net,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        super(NeuSRenderer, self).__init__()
        
        self.args = args
        self.conf = conf
        # self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.feature_net = feature_net ## Given points, output latent features in this postion
        self.n_samples = self.args.n_samples
        # self.n_samples = n_samples
        self.n_importance = self.args.n_importance
        # self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = self.args.up_sample_steps
        self.perturb = perturb
        self.bball_radius = self.args.bball_radius
        self.eval_image = False
   
    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.norm(pts, p=2, dim=-1, keepdim=True).clamp(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.norm(pts, p=2, dim=-1)
        inside_sphere = (radius[:, :-1] < self.bball_radius) | (radius[:, 1:] < self.bball_radius) 
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]


        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]).to(rays_o.device), cos_val[:, :-1]], dim=-1).to(rays_o.device)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)

        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clamp(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
    
        if not last:

            transformation, can_pts = self.feature_net.get_inverse_transform(pts.reshape(-1,3).unsqueeze(0))
            new_sdf = self.sdf_network.sdf(can_pts.reshape(-1, 3)).reshape(batch_size, n_importance)
  
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf
    
    ## todo::
    def prepare_feature(self,data, latent_codes):
        self.feature_net.sp_input = data.sp_input
        self.feature_net.codes = latent_codes
        self.feature_net.fuse_codes_all()

    def render_normal(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = (far - near) / self.n_samples

        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(near.device)
        z_vals = near + (far - near) * z_vals[None, :]
        z_vals_outside = None
        assert self.n_outside == 0, print("outside not set!\n")
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5).to(near.device)
            z_vals = z_vals + t_rand * sample_dist
   
        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                
                transformation, can_pts = self.feature_net.get_inverse_transform(pts.reshape(-1,3).unsqueeze(0))
                sdf = self.sdf_network.sdf(can_pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
               
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        dists = torch.cat([dists, sample_dist], -1)
            
        mid_z_vals = z_vals + dists * 0.5
        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        pts.requires_grad_(True)
        transformation, can_pts = self.feature_net.get_inverse_transform(pts.reshape(-1,3).unsqueeze(0))

        sdf_nn_output = self.sdf_network(can_pts)
      
        sdf = sdf_nn_output[:, :1]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)

        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pts,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
            )[0]
        gradients = gradients.unsqueeze(1).squeeze().float()

        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clamp(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clamp(0.0, 1.0)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).to(alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        Rot = self.feature_net.sp_input['R'].squeeze(0).to(gradients.device)
        Rot_T = torch.inverse(Rot)
        gradients = torch.matmul(gradients, Rot_T)

        pts_norm = torch.norm(pts, p=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < self.bball_radius).float().detach()

        return {
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            'weights': weights,
            'inside_sphere': inside_sphere,
        }

    ## checked
    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        dists = torch.cat([dists, sample_dist], -1)
            
        mid_z_vals = z_vals + dists * 0.5
        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        pts.requires_grad_(True)

        transformation, can_pts = self.feature_net.get_inverse_transform(pts.reshape(-1,3).unsqueeze(0))

        ## get geometry
        sdf_nn_output = sdf_network(can_pts)

        sdf = sdf_nn_output[:, :1]

        feature_vector = sdf.float()

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
   
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pts,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
            )[0]
        gradients = gradients.unsqueeze(1).squeeze().float()

        ## get appearance features
        can_pts_wo_residual = transformation['can_pts_wo_residual']
        forward_posed_ppts = self.feature_net.ppts_canonical_to_posed(can_pts_wo_residual, transformation)

        surface_app_features = self.feature_net.get_app_surface_feature(can_pts,transformation=transformation)

        final_app_features = self.feature_net.get_pixel_aligned_feature(forward_posed_ppts, surface_feature = surface_app_features)

        
        if self.eval_image:
            with torch.no_grad():
                sampled_color = color_network(can_pts, gradients, dirs, feature_vector, final_app_features).reshape(batch_size, n_samples, 3)
        else:
            sampled_color = color_network(can_pts, gradients, dirs, feature_vector, final_app_features).reshape(batch_size, n_samples, 3)
            
        ## neus rendering equation
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clamp(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clamp(0.0, 1.0)


        pts_norm = torch.norm(pts, p=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        can_pts_norm = torch.norm(can_pts, p=2, dim=-1,keepdim=True).reshape(batch_size, n_samples)
        
        inside_sphere = (pts_norm < self.bball_radius).float().detach()
        relax_inside_sphere = (pts_norm < self.bball_radius + 0.2).float().detach()
        
        can_inside_sphere = (can_pts_norm < self.bball_radius).float().detach()
        can_relax_inside_sphere = (can_pts_norm < self.bball_radius + 0.2).float().detach()
        
        inside_sphere = inside_sphere 
        relax_inside_sphere = relax_inside_sphere
        
        # Eikonal loss
        gradient_error = (torch.norm(gradients.reshape(batch_size, n_samples, 3), p=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        
        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).to(alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)


        ## gradients change to canonical space
        Rot = self.feature_net.sp_input['R'].squeeze(0).to(gradients.device)
        Rot_T = torch.inverse(Rot)

        gradients = torch.matmul(gradients, Rot_T)

        sdf_min = sdf.reshape(batch_size, n_samples, 1).min(dim = 1)[0]


        return {
            'color': color,
            'sdf': sdf,
            'sdf_min': sdf_min,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
        }
        
    def forward(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):


        batch_size = len(rays_o)

        sample_dist = (far - near) / self.n_samples
   

        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(near.device)
        z_vals = near + (far - near) * z_vals[None, :]
        z_vals_outside = None
        assert self.n_outside == 0, print("outside not set!\n")
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5).to(near.device)
            z_vals = z_vals + t_rand * sample_dist

        background_alpha = None
        background_sampled_color = None


        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                
                transformation, can_pts = self.feature_net.get_inverse_transform(pts.reshape(-1,3).unsqueeze(0))
                sdf = self.sdf_network.sdf(can_pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                    
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance


        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)
        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)
        sdf_min = ret_fine['sdf_min']

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'sdf_min':sdf_min,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0, device='cpu'):
        return extract_geometry(bound_min,
                            bound_max,
                            resolution=resolution,
                            threshold=threshold,
                            query_func=lambda pts,features: -self.sdf_network.sdf(pts,features),
                            feature_net=self.feature_net,
                            args = self.args,
                            device = device)
 
