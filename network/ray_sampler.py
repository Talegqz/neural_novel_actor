import torch
from math import sqrt, exp


class ImgToPatch(object):
    def __init__(self, ray_sampler, hwf):
        self.ray_sampler = ray_sampler
        self.hwf = hwf      # camera intrinsics

    def __call__(self, img):
        rgbs = []
        for img_i in img:
            pose = torch.eye(4)         # use dummy pose to infer pixel values
            _, selected_idcs, pixels_i = self.ray_sampler(H=self.hwf[0], W=self.hwf[1], focal=self.hwf[2], pose=pose)
            if selected_idcs is not None:
                rgbs_i = img_i.flatten(1, 2).t()[selected_idcs]
            else:
                rgbs_i = torch.nn.functional.grid_sample(img_i.unsqueeze(0), 
                                     pixels_i.unsqueeze(0), mode='bilinear', align_corners=True)[0]
                rgbs_i = rgbs_i.flatten(1, 2).t()
            rgbs.append(rgbs_i)

        rgbs = torch.cat(rgbs, dim=0)       # (B*N)x3

        return rgbs


class RaySampler(object):
    def __init__(self, N_samples, orthographic=False):
        super(RaySampler, self).__init__()
        self.N_samples = N_samples
        self.scale = torch.ones(1,).float()
        self.return_indices = True
        self.orthographic = orthographic

    def __call__(self, H, W, rays_o, rays_d, img, msk, near, far):
        select_inds = self.sample_rays(H, W).to(rays_o.device)

        if self.return_indices:
            print("patch sample wrong!")
            exit(1)
            rays_o = rays_o.view(-1, 3)[select_inds]
            rays_d = rays_d.view(-1, 3)[select_inds]

            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds %  W) / float(W) - 0.5

            hw = torch.stack([h,w]).t()

        else:
            near = near.unsqueeze(-1)
            far = far.unsqueeze(-1)
            msk = msk.unsqueeze(-1)
            
            rays_o = torch.nn.functional.grid_sample(rays_o.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_d = torch.nn.functional.grid_sample(rays_d.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            near = torch.nn.functional.grid_sample(near.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            far = torch.nn.functional.grid_sample(far.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]

            img = torch.nn.functional.grid_sample(img.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            msk = torch.nn.functional.grid_sample(msk.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_o = rays_o.permute(1,2,0).view(-1, 3)
            rays_d = rays_d.permute(1,2,0).view(-1, 3)
            img = img.permute(1,2,0).view(-1, 3)
            # import pdb
            # pdb.set_trace()
            msk = msk.permute(1,2,0).view(-1, 1)
            near = near.permute(1,2,0).view(-1, 1)
            far = far.permute(1,2,0).view(-1, 1)

            return img, msk, rays_o, rays_d, near, far
            # hw = select_inds
            # select_inds = None

        # return torch.stack([rays_o, rays_d]), select_inds, hw

    def sample_rays(self, H, W):
        raise NotImplementedError


class FullRaySampler(RaySampler):
    def __init__(self, **kwargs):
        super(FullRaySampler, self).__init__(N_samples=None, **kwargs)

    def sample_rays(self, H, W):
        return torch.arange(0, H*W)


class FlexGridRaySampler(RaySampler):
    def __init__(self, N_samples, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1,
                 **kwargs):
        self.N_samples_sqrt = int(sqrt(N_samples))
        super(FlexGridRaySampler, self).__init__(self.N_samples_sqrt**2, **kwargs)

        self.random_shift = random_shift
        self.random_scale = random_scale

        self.min_scale = min_scale
        self.max_scale = max_scale

        # nn.functional.grid_sample grid value range in [-1,1]
        self.w, self.h = torch.meshgrid([torch.linspace(-1,1,self.N_samples_sqrt),
                                         torch.linspace(-1,1,self.N_samples_sqrt)])
        self.h = self.h.unsqueeze(2)
        self.w = self.w.unsqueeze(2)

        # directly return grid for grid_sample
        self.return_indices = False

        self.iterations = 0
        self.scale_anneal = scale_anneal

    def sample_rays(self, H, W):

        # if self.scale_anneal>0:
        #     k_iter = self.iterations // 1000 * 3
        #     min_scale = max(self.min_scale, self.max_scale * exp(-k_iter*self.scale_anneal))
        #     min_scale = min(0.9, min_scale)
        # else:
        min_scale = self.min_scale

        scale = 1
        if self.random_scale:
            scale = torch.Tensor(1).uniform_(min_scale, self.max_scale)
            h = self.h * scale 
            w = self.w * scale 

        if self.random_shift:
            max_offset = 1-scale.item()
            h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2
            w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2

            h += h_offset
            w += w_offset

        self.scale = scale

        return torch.cat([h, w], dim=2)
