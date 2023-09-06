import os
from os.path import join
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import imageio
import trimesh
import glob
import torch
import json
import torch.nn.functional as F
from shutil import copyfile
import tqdm
from dataset import data_utils,base_dataset,neus_dataset
from network import latent_code
from network.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork
from render.neus_renderer import NeuSRenderer
from dataset.data_utils import fix_seed
import sys
sys.path.append('tools/get_blending_weight')
from config import cfg

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class Runner:
    def __init__(self, args, conf):
        self.device = torch.device('cuda')
        self.args = args
        self.conf = conf
        self.is_train = self.args.is_train
        assert(not self.is_train)
        if self.args.nhp_psnr:
            self.args.big_box = False

        fix_seed(42)

        # Configuration
        # if self.is_train:
        #     views = data_utils.parse_views(args.train_views)
        # else:
        #     views = data_utils.parse_views(args.test_views)
        views = None
        self.views = views
        self.base_dataset = base_dataset.BaseDataset(args, conf, args.root_path, views, args.subsample_valid, is_train=False) 

        if self.args.app_editing:
            self.app_target_dataset = base_dataset.BaseDataset(args, conf, args.root_path, views, args.subsample_valid, is_train=False, target_app=True)
        if self.args.pose_driven:
            self.pose_driven_dataset = base_dataset.BaseDataset(args, conf, args.root_path, self.views, args.subsample_valid, is_train=False)

        self.base_exp_dir = os.path.join(args.output_path,args.name)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.iter_step = 0
        
        self.use_white_bkgd = self.args.use_white_bkgd
        self.eval_batch_size = self.args.eval_batch_size
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
     
        self.writer = None

        # Networks
        self.sdf_network = SDFNetwork(args,self.conf['model'],**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(self.args.neus_variance).to(self.device)
        self.color_network = RenderingNetwork(self.args,**self.conf['model.rendering_network']).to(self.device)
        self.feature_net = latent_code.FeatureNet(self.args,self.conf['model'],device = self.device).to(self.device)

        self.renderer = NeuSRenderer(
                                     self.args,
                                     self.conf,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.feature_net,
                                     **self.conf['model.neus_renderer'])
        
        # Load checkpoint
        latest_model_name = None
        try:
            if self.args.load_model_path == None:
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]
                latest_model_name = os.path.join(self.base_exp_dir, 'checkpoints', latest_model_name)
            else:
                latest_model_name = self.args.load_model_path
                    
            if latest_model_name is not None:
                logging.info('Find checkpoint: {}\n'.format(latest_model_name))
                self.load_checkpoint(latest_model_name)
                print("Load model successfully!\n")
            else:
                if not self.is_train:
                    print("Finding model Failed!\n")
                    exit(1)           
        except Exception as e:
            print("Load model Failed!\n")
            print(e)
            exit(1)

    def get_cos_anneal_ratio(self):
        return 1.0
                    
    def prepare_novel_pose_ineterpolate_views(self,batch,character_id, shape, scale ,shape_id):
        if 'npy' in batch:
            batch = np.load(batch,allow_pickle=True).item()
        else:
            batch = json.load(open(batch))

        pose = np.array(batch['poses'])
        if self.args.novel_shape:
            shape[0][0] = -2
        else:
            pass

        data_shape = {}
        transform = {}
        transform['Rh'] = batch['Rh']
        transform['Th'] = batch['Th']
        transform['poses'] = pose
        transform['shapes'] = shape
        transform['scale'] = scale
        data_shape['transform'] = transform
        data_shape['id'] = character_id
        data = self.base_dataset.get_one_item_shape(data_shape,pose_driven=True)
        return data
          
    def render_interpolate_image(self, latent_codes, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        rays_o = rays_o.reshape(-1, 3).cpu().numpy()
        rays_d = rays_d.reshape(-1, 3).cpu().numpy()
        
        sp_input = self.dataset.sp_input
        
        Th = sp_input['Th'].squeeze(0).to(self.device)
        Rot = sp_input['R'].squeeze(0).to(self.device)
        H = self.dataset.H // resolution_level
        W = self.dataset.W // resolution_level
        
        if not self.args.neuralbody_sample:
            rays_o, rays_d, near, far, mask_at_box = self.dataset.sample_ray_smpl_guided_novel_view(
                rays_o, rays_d, sp_input['smpl_world_vertex'])
        else:
            rays_o, rays_d, near, far, mask_at_box = self.dataset.sample_ray_on_bbox_novel_view(rays_o, rays_d)
            
        assert mask_at_box.sum() > 0, 'can not find people in given camera!'
        
        rays_o = torch.tensor(rays_o.copy()).to(self.device).float()
        rays_d = torch.tensor(rays_d.copy()).to(self.device).float()
        near = torch.tensor(near).unsqueeze(-1).to(self.device)
        far = torch.tensor(far).unsqueeze(-1).to(self.device)
        
        rays_o = rays_o - Th
        rays_o = torch.matmul(rays_o, Rot)
        rays_d = torch.matmul(rays_d, Rot)
        
        mask_index = mask_at_box.reshape(H,W)
        mask_index = np.where(mask_at_box!=0)[0]

        # self.eval_batch_size = 10000
        rays_o = rays_o.reshape(-1, 3).split(self.eval_batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.eval_batch_size)
        near = near.reshape(-1,1).split(self.eval_batch_size)
        far = far.reshape(-1,1).split(self.eval_batch_size)
        
        
        out_rgb_fine = []
        out_normal_fine = []
        out_depth_fine = []
        self.renderer.prepare_feature(data=self.dataset,latent_codes = latent_codes)
        for rays_o_batch, rays_d_batch, near_batch, far_batch in zip(rays_o, rays_d, near, far):
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer(rays_o_batch,
                                              rays_d_batch,
                                              near_batch,
                                              far_batch,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            if feasible('depth'):
                out_depth_fine.append(render_out['depth'].detach().cpu().numpy())       
            del render_out
      
        if len(out_rgb_fine) > 0:
            img_fine = np.ones((H*W,3))*255. if self.use_white_bkgd else np.zeros((H*W,3))
            img_fine[mask_index] = (np.concatenate(out_rgb_fine, axis=0)*255).clip(0, 255)
            img_fine = img_fine.reshape([H, W, 3])  

            

        depth_img_fine = None
        if len(out_depth_fine) > 0:
            depth_img_fine = np.zeros((H*W))
            depth_img_fine[mask_index] = (np.concatenate(out_depth_fine, axis=0)*256).clip(0, 255)
            depth_img_fine = depth_img_fine.reshape([H, W, 1, -1])


        return img_fine, depth_img_fine
    
    def generate_pose_driven_results(self):
        if self.args.novel_pose_path != None:
            #     novel_pose_path = self.args.root_path + '/0/params'
        # else:
            novel_pose_path = self.args.novel_pose_path
            pose_transform = novel_pose_path+'/smpl_transform'
            target_id = novel_pose_path.split('/')[-1]
            if target_id == '':
                target_id = novel_pose_path.split('/')[-2]
            '''
                Use novel_pose camera
            '''
            all_int_path = []
            all_ext_path = []
            all_ixt = []
            all_ext = []
            if os.path.exists(novel_pose_path + '/intrinsic'):
                all_int_path.append(([sorted(glob.glob(novel_pose_path + '/intrinsic/*.txt'))]))
            if os.path.exists(novel_pose_path + '/extrinsic'):
                all_ext_path.append(([sorted(glob.glob(novel_pose_path + '/extrinsic/*.txt'))]))
            all_ext_path = all_ext_path[0][0]
            all_int_path = all_int_path[0][0]

            for view_idx in range(len(all_int_path)):
                extrinsics = data_utils.load_matrix(all_ext_path[view_idx]) 
                extrinsics = data_utils.parse_extrinsics(extrinsics, False).astype('float32')  # this is C2W
                intrinsics = data_utils.load_intrinsics(all_int_path[view_idx]).astype('float32')
    
                intrinsics[:2,:] = intrinsics[:2,:] * self.conf['dataset.ratio']
                extrinsics[:3,3] *= self.args.scale_size
                all_ext.append(extrinsics)
                all_ixt.append(intrinsics) 
            all_ext = torch.tensor(all_ext)
            all_ixt = torch.tensor(all_ixt)

        else:
            print("novel_pose_path not assign\n")
            exit(1)

        pose_loader = sorted(glob.glob(pose_transform+'/*.npy'))
        if len(pose_loader) == 0:
            pose_loader = sorted(glob.glob(pose_transform+'/*.json'))
        if self.args.test_start_end is not None:
            start, end = eval(self.args.test_start_end)
            pose_loader = pose_loader[start:end]
        if self.args.subsample_valid is not None:
            pose_loader = pose_loader[::self.args.subsample_valid]


        if self.args.aist_data:
            original_data =  self.base_dataset[0]
        else:

            original_data =  self.base_dataset[0]

        latent_codes_data = self.base_dataset.tocuda(original_data,self.device)
        character_id = latent_codes_data['character_id']
        shape = latent_codes_data['sp_input']['smpl_shape'].cpu().numpy()
        data_shape_id = latent_codes_data['shape_id']
        if self.args.aist_data:
            scale = latent_codes_data['sp_input']['smpl_params']['scale']
        else:
            scale = 1.0

        with torch.no_grad():
            latent_codes = self.renderer.feature_net.get_codes(latent_codes_data)

            if self.args.app_editing:
                uv_projection_geometry = self.renderer.feature_net.uv_projection
                app_data = self.app_target_dataset.tocuda(self.app_target_dataset[0],self.device)
                app_codes = self.renderer.feature_net.get_codes(app_data)
                if self.args.app_segmentation:
                    if self.args.app_with_codes:
                        latent_codes['app_codes_new'] = app_codes['app_codes']
                    latent_codes['app_map_new'] = app_codes['app_map']
                else:
                    if self.args.app_with_codes:
                        latent_codes['app_codes'] = app_codes['app_codes']
                    latent_codes['app_map'] = app_codes['app_map']

                self.renderer.feature_net.uv_projection_geometry = uv_projection_geometry
                
        self.renderer.feature_net.iteration = self.iter_step

        if self.args.fix_camera:
            camera_idx = self.args.fix_camera_id
            images_path = os.path.join(self.base_exp_dir, f'novel_pose_fixed_camera_id{camera_idx}/character_{character_id}/target_pose:{target_id}/images')
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(self.base_exp_dir, f'novel_pose_fixed_camera_id{camera_idx}/character_{character_id}/target_pose:{target_id}')
        else:
            images_path = os.path.join(self.base_exp_dir, f'novel_pose_interpolate/character_{character_id}/target_pose:{target_id}/images')
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(self.base_exp_dir, f'novel_pose_interpolate/character_{character_id}/target_pose:{target_id}')
        os.makedirs(images_path,exist_ok=True)
        
        resolution_level = 1

        H,W =  latent_codes_data['image_size']
        h = H//resolution_level
        w = W//resolution_level
        writer = cv.VideoWriter(os.path.join(video_path,'render.mp4'),fourcc, 15, (w, h))
        n_frames = 120
        images = []
        imageio_images = []
        view_id = 0
        img_idx0 = 0
        if self.args.aist_data:
            img_idx1 = 0
        else:
            img_idx1 = 12
        if self.args.fix_camera:
            img_idx0 = camera_idx
            img_idx1 = camera_idx


        for frame,batch in enumerate(tqdm.tqdm(pose_loader)):
            data = self.prepare_novel_pose_ineterpolate_views(batch, character_id, shape ,scale, data_shape_id)
            data = self.base_dataset.tocuda(data, self.device)
            self.dataset = neus_dataset.NeusDataset(data,args=self.args, device = self.device, no_gt_img = True)
            if self.args.fix_camera:
                self.dataset.all_ext = all_ext
                self.dataset.all_ixt = all_ixt
            else:
                self.dataset.all_ext = all_ext
                self.dataset.all_ixt = all_ixt
            
            self.dataset.W = W
            self.dataset.H = H
            # "--------------------------------------------------------------------"
            print(frame)
    
            if self.args.a_pose:
                n_frames = 60
                for frame in range(0,n_frames):
                    print(frame)
                    this_img = self.render_interpolate_image(latent_codes, img_idx0,
                                            img_idx1,
                                            np.sin(((view_id / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                        resolution_level=resolution_level)
                    view_id += 1
                    if view_id > n_frames:
                        img_idx0, img_idx1 = img_idx1, img_idx0
                        view_id = 0
                    cv.imwrite(os.path.join(images_path,f'{frame:03}.png'),this_img)
                    images.append(this_img.astype(np.uint8))
                    imageio_images.append(this_img[:,:,[2,1,0]].astype(np.uint8))
                if self.args.generate_mesh:
                    self.validate_mesh(frame,data,latent_codes_dataset=latent_codes_data,novel_pose = True, save_name = f'novel_pose_interpolate/character_{character_id}/target_pose:{target_id}', resolution = 256)
                for image in images:
                    writer.write(image)
                imageio.mimsave(os.path.join(video_path,'render.gif'),imageio_images,fps=18)
                writer.release()
                
                exit(1)
                
            this_img, depth_img = self.render_interpolate_image(latent_codes, img_idx0,
                                                img_idx1,
                                                np.sin(((view_id / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                        resolution_level=resolution_level)
            view_id += 1
            if view_id > n_frames:
                img_idx0, img_idx1 = img_idx1, img_idx0
                view_id = 0
                
            cv.imwrite(os.path.join(images_path,f'{frame:03}.png'),this_img)
            try:
                cv.imwrite(os.path.join(images_path,f'{frame:03}_depth.png'),depth_img[:,:,:,0])
            except Exception as e:
                pass
            images.append(this_img.astype(np.uint8))
            imageio_images.append(this_img[:,:,[2,1,0]].astype(np.uint8))
            if self.args.generate_mesh:
                self.validate_mesh(frame,data,latent_codes_dataset=latent_codes_data,novel_pose = True, save_name = video_path, resolution = 128)
        for image in images:
            writer.write(image)
        imageio.mimsave(os.path.join(video_path,'render.gif'),imageio_images,fps=18)
        writer.release()
                
    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(checkpoint_name, map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.feature_net.load_state_dict(checkpoint['feature_net'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        if self.args.is_train:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def validate_mesh(self, person_idx = -1, data = None, latent_codes = None, latent_codes_dataset = None, novel_pose = False, save_name = 'novel_pose', world_space=False, resolution=256, threshold=0.0):
        torch.cuda.empty_cache()
        self.base_dataset.iteration = self.iter_step 
        self.renderer.train()

        if data == None:
            if person_idx < 0:
                person_idx = np.random.randint(self.base_dataset.all_num)
            # data = self.base_dataset.tocuda(self.base_dataset[person_idx,0],self.device)
            index = person_idx * self.base_dataset.num_views
            data = self.base_dataset.tocuda(self.base_dataset[index],self.device)
            
        self.dataset = neus_dataset.NeusDataset(data,no_gt_img=novel_pose,args=self.args, device = self.device)
        self.dataset.app_image_id = person_idx
        if latent_codes == None:
            if latent_codes_dataset == None:
                if self.args.cross_train:
                    # character_id = base_data['character_id'] - self.base_dataset.character_min_id
                    self.base_dataset.latent_codes_data = True
                    character_id = self.base_dataset.characters_index[data['character_id']]
                    pose_per_character = self.base_dataset.frames
                    low_id = character_id * pose_per_character
                    high_id = (character_id + 1) * pose_per_character
                    latent_codes_id = np.random.randint(low_id,high_id,size=1)[0]
                    if self.args.monocular_view:
                        view_idx = self.views[0]
                    else:
                        view_idx = 0
                    latent_codes_dataset = self.base_dataset.tocuda(self.base_dataset[latent_codes_id * self.base_dataset.num_views + view_idx],self.device)
                    self.base_dataset.latent_codes_data = False
                else:
                    latent_codes_dataset = data

            with torch.no_grad():
                latent_codes = self.renderer.feature_net.get_codes(latent_codes_dataset)
       
        self.renderer.feature_net.sp_input = self.dataset.sp_input
        self.renderer.feature_net.codes = latent_codes
        self.renderer.feature_net.fuse_codes_all()
        bound_min = self.dataset.smpl_bbox[0]
        bound_max = self.dataset.smpl_bbox[1]


        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, device = self.device)

        if world_space:
            vertices = np.matmul(vertices, self.dataset.scale_mat[:3, :3]) + self.dataset.scale_mat[:3, 3][None]
        sp_input = data['sp_input']
        Th = sp_input['Th'].squeeze(0).cpu().numpy()
        Rot = sp_input['R'].squeeze(0).cpu().numpy()
        vertices = vertices @ Rot.T
        vertices = vertices + Th
        vertices /= self.args.scale_size

        mesh = trimesh.Trimesh(vertices, triangles)
        
        if not self.args.is_train:
            self.iter_step = -1
        if not novel_pose:
            os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.obj'.format(self.iter_step)))
        else:
            os.makedirs(f'{save_name}/meshes', exist_ok=True)
            mesh.export(os.path.join(f'{save_name}/meshes', f'{person_idx:03}.obj'))
        
        logging.info('End')
        del data
        
        return vertices, mesh

if __name__ == '__main__':

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    args, conf = cfg.parse_args()

    torch.cuda.set_device(args.gpu_id[0])
    runner = Runner(args,conf)


    if args.pose_driven:
        print('[MODE]: novel pose mode!\n')
        runner.generate_pose_driven_results()
    else:
        AssertionError('Not implemented yet!')