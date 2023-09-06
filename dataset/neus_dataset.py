import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import imageio
from network.utils import get_rays,get_bound_2d_mask,get_near_far,get_total_near_far, sampling_without_replacement,get_nhp_near_far
from network.ray_sampler import FlexGridRaySampler

class NeusDataset:
    def __init__(self,data,args,device,no_gt_img = False):
        super(NeusDataset, self).__init__()

        self.device = device
        self.sp_input = data['sp_input']
        self.args = args
        self.smpl_bbox = data['sp_input']['smpl_bbox']
        self.world_bbox = data['sp_input']['world_bbox']
        
        if "input" in data.keys():
            self.prepared_input = data['input']

        if not no_gt_img:

            self.all_ext = data['loss_ext']
            self.all_ixt = data['loss_ixt'] 
            self.all_focal = data['loss_focal'] 
            self.all_principle = data['loss_principle'] 
            self.images = data['loss_image'].squeeze(0).permute(0,2,3,1)  # [Nv,H,W,3]
            self.masks = data['loss_mask'] 
            self.image_lst = data['loss_image_path']
            H,W = data['image_size']
            self.H = H
            self.W = W
        self.min_dis_eps = args.min_dis_eps  ## how far should we sample rays for a given smpl

        if self.args.patch_sample:
            self.patch_sampler = FlexGridRaySampler(
                                    N_samples=self.args.batch_size,
                                    min_scale=0.25,
                                    max_scale=0.5,
                                    scale_anneal=0.25)

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """

        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        # R = self.all_ext[idx_0][:3,:3].cpu().numpy()
        # T = self.all_ext[idx_0][:3,3].cpu().numpy()
 
        K = self.all_ixt[0][:3,:3].cpu().numpy()
        # K[:2] = K[:2] / resolution_level
        # K[:2] = K[:2]
        intrinsics_all_inv = torch.inverse(torch.tensor(K))
        pose_all = torch.inverse(self.all_ext).cpu().numpy()
        p = torch.matmul(intrinsics_all_inv, p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = F.normalize(p, p=2, dim=-1)  # W, H, 3
        pose_0 = pose_all[idx_0]
        pose_1 = pose_all[idx_1]
        trans = (pose_all[idx_0, :3, 3]) * (1.0 - ratio) + (pose_all[idx_1, :3, 3] * ratio)
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

    def nhp_crop_img(self, mask, image, mask_at_box):
        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        mask = mask[y:y + h, x:x + w]
        image = image[y:y + h, x:x + w]

        return mask, image

    def image_at(self, idx, resolution_level):
        img = self.images[idx].cpu().numpy()*255
        #img = cv2.imread(self.image_lst[idx])
        return (cv2.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def sample_patch_rays(self, img_idx, world_vertices, iterations = 0, resolution_level = 1,patch_num=1,patch_size=32):
        
        self.patch_sampler.iterations = iterations
        img = self.images[img_idx].float().cpu().numpy()
        H, W = img.shape[0]//resolution_level, img.shape[1]//resolution_level     
        img = cv2.resize(img, (W,H), interpolation=cv2.INTER_AREA)
        msk = self.masks[img_idx].cpu().numpy()
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        # img = all_image[0][view_idx].cpu().numpy().transpose(1,2,0)
        #img = all_image[0][view_idx].cpu().numpy()
        R = self.all_ext[img_idx][:3,:3].cpu().numpy()
        T = self.all_ext[img_idx][:3,3].cpu().numpy()
        K = self.all_ixt[img_idx][:3,:3].cpu().numpy()
        K[:2] = K[:2] / resolution_level
        
        bounds = self.world_bbox.cpu().numpy()
        
        ray_o, ray_d = get_rays(H, W, K, R, T)

        pose = np.concatenate([R, T[:,None]], axis=1)
        bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
        
        smpl_mask_index = (bound_mask != 0)
        x, y, w, h = cv2.boundingRect(smpl_mask_index.astype(np.uint8))
        if w<=patch_size:
            print(w)
            deta = patch_size-w
            if x-int(deta/2)<0:
                x = 0
                w = patch_size+2
            else:
                x = x-int(deta/2)
                w = patch_size+2
        
        if h<=patch_size:
            deta = patch_size-h
            print(h)
            if y-int(deta/2)<0:
                y = 0
                h = patch_size+2
            else:
                y = y-int(deta/2)
                h = patch_size+2
        
        msk = msk[y:y + h, x:x + w]
        img = img[y:y + h, x:x + w]
        ray_o = ray_o[y:y + h, x:x + w]
        ray_d = ray_d[y:y + h, x:x + w]

        # return mask, image
        # msk, img = self.nhp_crop_img(msk,img,smpl_mask_index)

        H, W = msk.shape

        norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
        ray_d = ray_d / norm_d


        # msk = msk * bound_mask
        # bound_mask[msk == 100] = 0

        # cv2.imwrite('img_crop.png', img*255)
        # cv2.imwrite('msk_crop.png', msk*255)
        # cv2.imwrite('bound_mask.png', bound_mask*255)
        
        bbox_near, bbox_far, mask_at_box = get_total_near_far(bounds, ray_o, ray_d)
        # print(bbox_far.max())
        # print(bbox_near.min())
        bbox_near[~mask_at_box]=bbox_near[mask_at_box].min()
        bbox_far[~mask_at_box]=bbox_far[mask_at_box].max()
        # print('mask box near far')
        # print(bbox_far[mask_at_box].max())
        # print(bbox_near[mask_at_box].min())


        rays_o = torch.tensor(ray_o)
        bbox_near = torch.tensor(bbox_near)
        bbox_far = torch.tensor(bbox_far)
        rays_d = torch.tensor(ray_d)
        # from clib import cloud_ray_intersect
        # try:
        #     hit_mask, hit_min_depth, hit_max_depth = cloud_ray_intersect(self.min_dis_eps, world_vertices.unsqueeze(0), rays_o.reshape(1,-1,3), rays_d.reshape(1,-1,3))
        # except Exception as e:
        #     import pdb
        #     pdb.set_trace()
        # '''
            # hit_mask: hit_min_depth < hit_max_depth
        # '''
        # hit_mask, hit_min_depth, hit_max_depth = hit_mask[0,:,0].bool(), hit_min_depth[0,:,0], hit_max_depth[0,:,0]
        # # near_, far_, mask_at_box = get_nhp_near_far(bounds, ray_o_, ray_d_, self.args.scale_size) 
        # # import pdb
        # # pdb.set_trace()
        # # ## faster
        # # bbox_near[mask_at_box] = np.maximum(hit_min_depth,bbox_near[mask_at_box])
        # # bbox_far[mask_at_box] = np.minimum(hit_max_depth,bbox_far[mask_at_box])
        # hit_min_depth = hit_min_depth.reshape(H,W)
        # # hit_min_depth = bbox_near
        # hit_max_depth = hit_max_depth.reshape(H,W)
        
        rgb_all = []
        msk_all = []
        ray_o_all = []
        ray_d_all = []
        near_all = []
        far_all = []
        for i in range(self.args.patch_num):
            # import pdb
            # pdb.set_trace()
        
            r_x = np.random.randint(0,H-self.args.patch_size)
            r_y = np.random.randint(0,W-self.args.patch_size)
            rgb_batch = img[r_x:r_x+self.args.patch_size,r_y:r_y+self.args.patch_size]
            msk_batch = msk[r_x:r_x+self.args.patch_size,r_y:r_y+self.args.patch_size]
            ray_o_batch = rays_o[r_x:r_x+self.args.patch_size,r_y:r_y+self.args.patch_size]
            ray_d_batch = rays_d[r_x:r_x+self.args.patch_size,r_y:r_y+self.args.patch_size]
            near_batch = bbox_near[r_x:r_x+self.args.patch_size,r_y:r_y+self.args.patch_size]
            far_batch = bbox_far[r_x:r_x+self.args.patch_size,r_y:r_y+self.args.patch_size]

            rgb_all.append(torch.from_numpy(rgb_batch))
            msk_all.append(torch.from_numpy(msk_batch))
            ray_o_all.append(ray_o_batch)
            ray_d_all.append(ray_d_batch)
            near_all.append(near_batch)
            far_all.append(far_batch)
        # import pdb
        # pdb.set_trace()
        rgb_all = torch.stack(rgb_all, dim = 0).view(-1,3)
        msk_all = torch.stack(msk_all, dim = 0).view(-1,1)
        ray_o_all = torch.stack(ray_o_all, dim = 0).view(-1,3)
        ray_d_all = torch.stack(ray_d_all, dim = 0).view(-1,3)
        near_all = torch.stack(near_all, dim = 0).view(-1,1)
        far_all = torch.stack(far_all, dim = 0).view(-1,1)

        # import pdb
        # pdb.set_trace()


        return rgb_all, ray_o_all, ray_d_all, near_all, far_all, None, None, msk_all

    ## todo:: remove
    def sample_ray_on_masks(self, img_idx, nrays, is_train = True ,resolution_level = 1):

        img = self.images[img_idx].float().cpu().numpy()
        H, W = img.shape[0]//resolution_level, img.shape[1]//resolution_level     
        img = cv2.resize(img, (W,H), interpolation=cv2.INTER_AREA)
        # cv2.imwrite('input_img.png',img*255)
        # exit(1)
        # msk = all_mask[view_idx].cpu().numpy()
        # img = all_image[0][view_idx].cpu().numpy().transpose(1,2,0)
        # #img = all_image[0][view_idx].cpu().numpy()
        
        # R = all_ext[view_idx][:3,:3].cpu().numpy()
        # T = all_ext[view_idx][:3,3].cpu().numpy()
        # K = all_ixt[view_idx][:3,:3].cpu().numpy()
        msk = self.masks[img_idx].cpu().numpy()
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        # img = all_image[0][view_idx].cpu().numpy().transpose(1,2,0)
        #img = all_image[0][view_idx].cpu().numpy()
        R = self.all_ext[img_idx][:3,:3].cpu().numpy()
        T = self.all_ext[img_idx][:3,3].cpu().numpy()
        K = self.all_ixt[img_idx][:3,:3].cpu().numpy()
        K[:2] = K[:2] / resolution_level
        
        bounds = self.world_bbox.cpu().numpy()
        
        ray_o, ray_d = get_rays(H, W, K, R, T)
        norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
        ray_d = ray_d / norm_d

        pose = np.concatenate([R, T[:,None]], axis=1)
        bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
        # import pdb
        # pdb.set_trace()
        ## mask: [H,W,3]?
        # cv2.imwrite('img.png',img*255)
        # cv2.imwrite('msk.png',msk*255)
        msk = msk * bound_mask
        bound_mask[msk == 100] = 0
        # cv2.imwrite('bound_mask.png',bound_mask*255)
        # import pdb
        # pdb.set_trace()
        # cv2.imwrite('msk++.png',msk*255)
        # exit(1)
        if is_train:
            nsampled_rays = 0
            # face_sample_ratio = self.cfg['dataset.face_sample_ratio']
            # body_sample_ratio = self.cfg['dataset.body_sample_ratio']
            face_sample_ratio = 0
            body_sample_ratio = 0.5
            ray_o_list = []
            ray_d_list = []
            rgb_list = []
            near_list = []
            msk_list = []
            far_list = []
            coord_list = []
            mask_at_box_list = []

            while nsampled_rays < nrays:
                n_body = int((nrays - nsampled_rays) * body_sample_ratio)
                n_face = int((nrays - nsampled_rays) * face_sample_ratio)
                n_rand = (nrays - nsampled_rays) - n_body - n_face

                # sample rays on body
                coord_body = np.argwhere(msk == 1)
                # if len(coord_body) ==0:
                # cv2.imwrite('debug/nb_sample/body_mask.png',(msk==1)*255)
                coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                        n_body)]
                # sample rays on face
                coord_face = np.argwhere(msk == 13)
                # cv2.imwrite('debug/nb_sample/face_mask.png',(msk==13)*255)
                if len(coord_face) > 0:
                    coord_face = coord_face[np.random.randint(
                        0, len(coord_face), n_face)]
                # sample rays in the bound mask
                coord = np.argwhere(bound_mask == 1)
                # cv2.imwrite('debug/nb_sample/rand_mask.png',(bound_mask == 1)*255)
                # exit(1)
                coord = coord[np.random.randint(0, len(coord), n_rand)]

                if len(coord_face) > 0:
                    coord = np.concatenate([coord_body, coord_face, coord], axis=0)
                else:
                    coord = np.concatenate([coord_body, coord], axis=0)

                ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
                ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
                rgb_ = img[coord[:, 0], coord[:, 1]]
                msk_ = msk[coord[:, 0], coord[:, 1]]
                if self.args.nhp_psnr:
                    near_, far_, mask_at_box = get_nhp_near_far(bounds, ray_o_, ray_d_, self.args.scale_size) 
                else:
                    near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_) 

                ray_o_list.append(ray_o_[mask_at_box])
                ray_d_list.append(ray_d_[mask_at_box])
                rgb_list.append(rgb_[mask_at_box])
                msk_list.append(msk_[mask_at_box])
                near_list.append(near_)
                far_list.append(far_)
                coord_list.append(coord[mask_at_box])
                mask_at_box_list.append(mask_at_box[mask_at_box])
                nsampled_rays += len(near_)

            ray_o = np.concatenate(ray_o_list).astype(np.float32)
            ray_d = np.concatenate(ray_d_list).astype(np.float32)
            rgb = np.concatenate(rgb_list).astype(np.float32)
            msk = np.concatenate(msk_list).astype(np.float32)
            near = np.concatenate(near_list).astype(np.float32)
            far = np.concatenate(far_list).astype(np.float32)
            coord = np.concatenate(coord_list)            
            mask_at_box = np.concatenate(mask_at_box_list)

        
            return rgb, ray_o, ray_d, near, far, coord, mask_at_box, msk
        else:
            rgb = img.reshape(-1, 3).astype(np.float32)
            ray_o = ray_o.reshape(-1, 3).astype(np.float32)
            ray_d = ray_d.reshape(-1, 3).astype(np.float32)
            if self.args.nhp_psnr:
                near, far, mask_at_box = get_nhp_near_far(bounds, ray_o, ray_d, self.args.scale_size) 
            else:
                near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d) 
            near = near.astype(np.float32)
            far = far.astype(np.float32)
            rgb = rgb[mask_at_box]
            ray_o = ray_o[mask_at_box]
            ray_d = ray_d[mask_at_box]
            coord = np.zeros([len(rgb), 2]).astype(np.int64)

            
            return img, ray_o, ray_d, near, far, coord, mask_at_box
 
    def sample_ray_on_bbox_novel_view(self, ray_o, ray_d):
        
        # pose = np.concatenate([R, T[:,None]], axis=1)

        # bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

        rays_o = ray_o.reshape(-1,3)
        rays_d = ray_d.reshape(-1,3)
        bounds = self.world_bbox.cpu().numpy()

        bbox_near, bbox_far, bound_mask = get_near_far(bounds, rays_o, rays_d)
        bound_mask = bound_mask.reshape(-1)
        
        rays_o = rays_o[np.where(bound_mask!=0),:][0]
        rays_d = rays_d[np.where(bound_mask!=0),:][0]
        
        norm_d = np.linalg.norm(rays_d, axis=-1, keepdims=True)
        viewdir = rays_d / norm_d
        # rays_o = torch.tensor(rays_o.copy()).to(self.device)
        # rays_d = torch.tensor(viewdir).to(self.device)
        
        return rays_o, viewdir, bbox_near, bbox_far, bound_mask

    def sample_ray_smpl_guided_novel_view(self, ray_o, ray_d, world_vertices):
        

        # pose = np.concatenate([R, T[:,None]], axis=1)

        # bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

        rays_o = ray_o.reshape(-1,3)
        rays_d = ray_d.reshape(-1,3)
        bounds = self.world_bbox.cpu().numpy()
        
        bbox_near, bbox_far, bound_mask = get_near_far(bounds, rays_o, rays_d)
        bound_mask = bound_mask.reshape(-1)
        
        rays_o = rays_o[np.where(bound_mask!=0),:][0]
        rays_d = rays_d[np.where(bound_mask!=0),:][0]
        # rgb = rgb[np.where(bound_mask!=0),:][0]
        
        # if is_train:
        #     rand_idx = np.random.randint(low = 0, high = rays_o.shape[0], size=[nrays])
        #     rays_o = rays_o[rand_idx]
        #     rays_d = rays_d[rand_idx]
        #     rgb = rgb[rand_idx]
            
        # bbox_near, bbox_far, mask_at_box = get_total_near_far(bounds, rays_o, rays_d)
        norm_d = np.linalg.norm(rays_d, axis=-1, keepdims=True)
        viewdir = rays_d / norm_d
        rays_o = torch.tensor(rays_o.copy()).to(self.device)
        rays_d = torch.tensor(viewdir).to(self.device)
        
        from clib import cloud_ray_intersect
        # import pdb
        # pdb.set_trace()
        hit_mask, hit_min_depth, hit_max_depth = cloud_ray_intersect(self.min_dis_eps, world_vertices.reshape(1,-1,3), 
            rays_o.reshape(1,-1,3), rays_d.reshape(1,-1,3))

        
        '''
            hit_mask: hit_min_depth < hit_max_depth
        '''
        hit_mask, hit_min_depth, hit_max_depth = hit_mask[0,:,0].bool(), hit_min_depth[0,:,0], hit_max_depth[0,:,0]
        hit_min_depth = torch.max(hit_min_depth,torch.tensor(bbox_near).to(self.device))
        hit_max_depth = torch.min(hit_max_depth,torch.tensor(bbox_far).to(self.device))
        

        hit_mask = hit_min_depth < hit_max_depth
        
        # near = hit_min_depth.cpu().numpy()
        # far = hit_max_depth.cpu().numpy()
        near = hit_min_depth[hit_mask].cpu().numpy()
        far = hit_max_depth[hit_mask].cpu().numpy()
        
        
        # if is_train:
        #     return rgb[hit_mask.cpu().numpy()], rays_o.squeeze(0)[hit_mask].cpu().numpy(), rays_d.squeeze(0)[hit_mask].cpu().numpy(), near, far, hit_mask.cpu().numpy()
        #     # return rgb, rays_o.squeeze(0).cpu().numpy(), rays_d.squeeze(0).cpu().numpy(), near, far, hit_mask.cpu().numpy()
        # else:
        mask_all = bound_mask
        mask_all[np.where(bound_mask!=0)] = hit_mask.cpu().numpy()
        '''
            mask_all: [H,W]
                1: intersect
                0: not intersect
        '''
        return rays_o.squeeze(0)[hit_mask].cpu().numpy(), rays_d.squeeze(0)[hit_mask].cpu().numpy(), near, far, mask_all
            # return img, rays_o.squeeze(0).cpu().numpy(), rays_d.squeeze(0).cpu().numpy(), near, far, mask_all

    def sample_ray_smpl_guided_NA_NB(self, img_idx, world_vertices, nrays, is_train = True ,resolution_level = 1):

        img = self.images[img_idx].float().cpu().numpy()
        H, W = img.shape[0]//resolution_level, img.shape[1]//resolution_level     
        img = cv2.resize(img, (W,H), interpolation=cv2.INTER_AREA)
        # cv2.imwrite('input_img.png',img*255)
        # exit(1)
        # msk = all_mask[view_idx].cpu().numpy()
        # img = all_image[0][view_idx].cpu().numpy().transpose(1,2,0)
        # #img = all_image[0][view_idx].cpu().numpy()
        
        # R = all_ext[view_idx][:3,:3].cpu().numpy()
        # T = all_ext[view_idx][:3,3].cpu().numpy()
        # K = all_ixt[view_idx][:3,:3].cpu().numpy()
        msk = self.masks[img_idx].cpu().numpy()
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        # img = all_image[0][view_idx].cpu().numpy().transpose(1,2,0)
        #img = all_image[0][view_idx].cpu().numpy()
        
        R = self.all_ext[img_idx][:3,:3].cpu().numpy()
        T = self.all_ext[img_idx][:3,3].cpu().numpy()
        K = self.all_ixt[img_idx][:3,:3].cpu().numpy()
        K[:2] = K[:2] / resolution_level
        
        bounds = self.world_bbox.cpu().numpy()
        
        ray_o, ray_d = get_rays(H, W, K, R, T)

        norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
        ray_d = ray_d / norm_d

        ray_o = ray_o.reshape(-1,3)
        ray_d = ray_d.reshape(-1,3)
        pose = np.concatenate([R, T[:,None]], axis=1)

        bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
        smpl_bound_mask = bound_mask.copy()
        msk = msk * bound_mask
        bound_mask[msk == 100] = 0
        # cv2.imwrite("debug/mask/msk.png",msk*255)
        # cv2.imwrite("debug/mask/bound_mask++.png",bound_mask*255)
        # bound_mask = bound_mask.reshape(-1)
            
        bbox_near, bbox_far, mask_at_box = get_total_near_far(bounds, ray_o, ray_d)
        
        # rays_o = torch.tensor(ray_o).to(self.device)
        # rays_d = torch.tensor(ray_d).to(self.device)
        
        ## faster
        rays_o = torch.tensor(ray_o[mask_at_box]).to(self.device)
        rays_d = torch.tensor(ray_d[mask_at_box]).to(self.device)
        
        from clib import cloud_ray_intersect
        hit_mask, hit_min_depth, hit_max_depth = cloud_ray_intersect(self.min_dis_eps, world_vertices.unsqueeze(0), rays_o.unsqueeze(0), rays_d.unsqueeze(0))


        '''
            hit_mask: hit_min_depth < hit_max_depth
        '''
        hit_mask, hit_min_depth, hit_max_depth = hit_mask[0,:,0].bool().cpu().numpy(), hit_min_depth[0,:,0].cpu().numpy(), hit_max_depth[0,:,0].cpu().numpy()
        
        # hit_min_depth = torch.max(hit_min_depth,torch.tensor(bbox_near).to(self.device))
        # hit_max_depth = torch.min(hit_max_depth,torch.tensor(bbox_far).to(self.device))
        
        ## faster
        bbox_near[mask_at_box] = np.maximum(hit_min_depth,bbox_near[mask_at_box])
        bbox_far[mask_at_box] = np.minimum(hit_max_depth,bbox_far[mask_at_box])
        hit_min_depth = bbox_near
        hit_max_depth = bbox_far
        
        hit_mask = hit_min_depth < hit_max_depth
        
        # mask_at_box = hit_mask.cpu().numpy()
        # bound_mask_index = np.where(bound_mask>0)
        # bound_mask[bound_mask_index] = hit_mask.cpu().squeeze(0).squeeze(-1).numpy()

        # bound_mask = hit_mask.cpu().numpy()
        # bound_mask = hit_mask.copy()
        bound_mask = hit_mask
        bound_mask = bound_mask.reshape(H,W)
        ray_o = ray_o.reshape(H,W,3)
        ray_d = ray_d.reshape(H,W,3)
        # bound_mask[msk == 100] = 0
        # cv2.imwrite('debug/mask/bound_mask++.png',bound_mask*255)
        if is_train:
            nsampled_rays = 0
            # face_sample_ratio = self.cfg['dataset.face_sample_ratio']
            # body_sample_ratio = self.cfg['dataset.body_sample_ratio']
            face_sample_ratio = 0
            body_sample_ratio = 0.5
            ray_o_list = []
            ray_d_list = []
            rgb_list = []
            msk_list = []
            near_list = []
            far_list = []
            coord_list = []
            mask_at_box_list = []

            while nsampled_rays < nrays:
                n_body = int((nrays - nsampled_rays) * body_sample_ratio)
                n_face = int((nrays - nsampled_rays) * face_sample_ratio)
                n_rand = (nrays - nsampled_rays) - n_body - n_face

                # sample rays on body
                coord_body = np.argwhere(msk == 1)
                # cv2.imwrite('debug/na_sample/body_mask.png',(msk==1)*255)
                coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                        n_body)]
                # sample rays on face
                coord_face = np.argwhere(msk == 13)
                # cv2.imwrite('debug/na_sample/face_mask.png',(msk==13)*255)
                if len(coord_face) > 0:
                    coord_face = coord_face[np.random.randint(
                        0, len(coord_face), n_face)]
                # sample rays in the bound mask
                coord = np.argwhere(bound_mask == 1)
                # cv2.imwrite('debug/na_sample/rand_mask.png',(bound_mask == 1)*255)
                # exit(1)
                coord = coord[np.random.randint(0, len(coord), n_rand)]

                if len(coord_face) > 0:
                    coord = np.concatenate([coord_body, coord_face, coord], axis=0)
                else:
                    coord = np.concatenate([coord_body, coord], axis=0)

                ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
                ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
                rgb_ = img[coord[:, 0], coord[:, 1]]
                msk_ = msk[coord[:, 0], coord[:, 1]]
                near_ = hit_min_depth.reshape(ray_o.shape[0],ray_o.shape[1])[coord[:, 0], coord[:, 1]]
                far_ = hit_max_depth.reshape(ray_o.shape[0],ray_o.shape[1])[coord[:, 0], coord[:, 1]]
                mask_at_box = hit_mask.reshape(ray_o.shape[0],ray_o.shape[1])[coord[:, 0], coord[:, 1]]
                # near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)
  
                ray_o_list.append(ray_o_[mask_at_box])
                ray_d_list.append(ray_d_[mask_at_box])
                rgb_list.append(rgb_[mask_at_box])
                msk_list.append(msk_[mask_at_box])
                near_list.append(near_[mask_at_box])
                far_list.append(far_[mask_at_box])
                coord_list.append(coord[mask_at_box])
                mask_at_box_list.append(mask_at_box[mask_at_box])
                nsampled_rays += len(near_)

            ray_o = np.concatenate(ray_o_list).astype(np.float32)
            ray_d = np.concatenate(ray_d_list).astype(np.float32)
            rgb = np.concatenate(rgb_list).astype(np.float32)
            msk = np.concatenate(msk_list).astype(np.float32)
            near = np.concatenate(near_list).astype(np.float32)
            far = np.concatenate(far_list).astype(np.float32)
            coord = np.concatenate(coord_list)            
            mask_at_box = np.concatenate(mask_at_box_list)

        
            return rgb, ray_o, ray_d, near, far, coord, mask_at_box, msk
        else:
            rgb = img.reshape(-1, 3).astype(np.float32)
            ray_o = ray_o.reshape(-1, 3).astype(np.float32)
            ray_d = ray_d.reshape(-1, 3).astype(np.float32)
            # near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
            near = hit_min_depth.reshape(-1).astype(np.float32)
            far = hit_max_depth.reshape(-1).astype(np.float32)
            mask_at_box = hit_mask.reshape(-1)
            # near = near.astype(np.float32)
            # far = far.astype(np.float32)
            rgb = rgb[mask_at_box]
            # import pdb
            # pdb.set_trace()
            near = near[mask_at_box]
            far = far[mask_at_box]
            ray_o = ray_o[mask_at_box]
            ray_d = ray_d[mask_at_box]
            coord = np.zeros([len(rgb), 2]).astype(np.int64)
            # norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
            # ray_d = ray_d / norm_d
            
            return img, ray_o, ray_d, near, far, smpl_bound_mask, mask_at_box