import enum
from pickle import FALSE
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import random
from torch.nn.modules.container import T
import yaml
import math
from dataset.data_utils import write_ply,load_face,batch_rodrigues,write_colored_ply
from network.utils import extract_feature, repeat_interleave, grad_grid_sample, extract_feature
from network.mlp import FCBlock
from network.embedder import PositionalEncoding, get_embedder
from network.gnn import GNN
from network.attention import MaskedSelfAttention
from network.image_encoder import ImageEncoder
from clib._ext import point_face_dist_forward
from clib import cloud_ray_intersect
from knn_cuda import KNN
## latentCode Net
class FeatureNet(nn.Module):
    '''
    Given sequence of multi-view image and SMPL, generate T-pose's latent codes
    '''
    def __init__(self, args, conf, device):
        super().__init__()
        self.args = args
        self.cfg = conf
        self.device = device
        self.iteration = 0
        if self.cfg['using_smplx']:
            self.latent_codes_num = 10475
            self.poses_num = 87 * 3
        else:
            self.latent_codes_num = 6890
            self.poses_num = 72 * 3
            
        self.latentCode_size = self.args.latent_codes_dim
        

        '''
            inverse part
        '''
        self.geo_image_encoder = ImageEncoder(in_dims=4) # todo:: change name

        self.app_image_encoder = ImageEncoder(in_dims=4)


        self.gnn_edge_PE = PositionalEncoding.from_conf(self.cfg['code'], d_in=4) 
        self.gnn_edge_dim = self.gnn_edge_PE.d_out

            

        ## geometry pixel features fusion
        self.self_attention_geo = MaskedSelfAttention(self.args,self.args.latent_codes_dim,self.args.latent_codes_dim)

        self.geo_pixel_embedding =  FCBlock(
                    self.args.latent_codes_dim, 2, self.args.latent_codes_dim, self.args.latent_codes_dim, 
                    outermost_linear=True, with_ln=True) 
        
        self.geo_cross_attention_embedding = FCBlock(
                self.args.latent_codes_dim*2, 0, self.args.latent_codes_dim*2, self.args.latent_codes_dim, 
                outermost_linear=True, with_ln=False) 
        
        self.geo_cross_attention = MaskedSelfAttention(self.args,self.args.latent_codes_dim,self.args.latent_codes_dim)

        

        self.app_cross_attention_embedding = FCBlock(
                self.args.latent_codes_dim*2, 0, self.args.latent_codes_dim*2, self.args.latent_codes_dim, 
                outermost_linear=True, with_ln=False) 
        self.app_cross_attention = MaskedSelfAttention(self.args,self.args.latent_codes_dim,self.args.latent_codes_dim)

        self.self_attention_app = MaskedSelfAttention(self.args,self.args.latent_codes_dim,self.args.latent_codes_dim)


        self.app_pixel_embedding =  FCBlock(
            self.args.latent_codes_dim, 2, self.args.latent_codes_dim, self.args.latent_codes_dim, 
            outermost_linear=True, with_ln=True) 


        gnn_layer = self.args.gnn_layer_num
        
        self.geo_gnn_inverse = GNN(self.args,self.cfg,self.device,self.latentCode_size , edge_input_size = self.gnn_edge_dim, num_message_passing_steps=gnn_layer) 
        
        self.app_gnn_inverse = GNN(self.args,self.cfg,self.device,self.latentCode_size , edge_input_size = self.gnn_edge_dim, num_message_passing_steps=gnn_layer) 


        '''
            forward part
        '''
        self.distance_PE = PositionalEncoding.from_conf(self.cfg['code'], d_in = 1)

        ## geometry
        self.geo_gnn_forward = GNN(self.args,self.cfg,self.device,node_input_size = self.latentCode_size, edge_input_size = self.gnn_edge_dim ,num_message_passing_steps = gnn_layer)

        self.app_gnn_forward = GNN(self.args,self.cfg,self.device,node_input_size = self.latentCode_size, edge_input_size = self.gnn_edge_dim ,num_message_passing_steps = gnn_layer)


        '''
            residual part
        '''
        self.residual_deform = FCBlock(
                        128, 2 , 24*4 + self.latentCode_size, 3, 
                        outermost_linear=True, with_ln=False)
      
                        
        self.residual_deform.net[-1].weight.data *= 0
        self.residual_deform.net[-1].bias.data *= 0
        
    
        ''''
            from smpl to spatial feature
        '''
        self.vector_fuse = FCBlock(
            self.latentCode_size * 2, 2, self.latentCode_size + 3, self.latentCode_size, 
                outermost_linear=True, with_ln=False)

        self.app_vector_fuse = FCBlock(
            self.latentCode_size * 2, 2, self.latentCode_size + 3, self.latentCode_size, 
                outermost_linear=True, with_ln=False)

        self.knn = KNN(k=self.args.knn_num, transpose_mode=True)
        

        self.occlusion_mesh = None

        if self.args.head_no_skeletal: ## todo:: check this
            self.head_id = torch.tensor(np.load('tools/head_id.npy',allow_pickle=True)).to(self.device).long()

    # checked
    def fuse_feature_with_distance(self, vertex, vertex_features, query_pts, appearance_feature=False):

        query_features = []
     
        k_dist, k_index = self.knn(vertex.unsqueeze(0), query_pts.unsqueeze(0))
        k_index = k_index.squeeze(0)
        k_dist = k_dist.squeeze(0) + 1e-9
        k_weight = F.normalize(1./(k_dist),p=1,dim = 1)
        k_vertices = vertex[k_index,:] #[42500,3,3]
        k_nearest_vector = (query_pts.unsqueeze(1) - k_vertices)


        k_nearest_features = vertex_features[k_index,:]

        vector_input = torch.cat([k_nearest_features,k_nearest_vector],dim = -1) #[points,K,features]
        if appearance_feature:
            features = self.app_vector_fuse(vector_input)
        else:
            features = self.vector_fuse(vector_input)
        k_weight = k_weight.unsqueeze(-1).expand(features.size(0),features.size(1),features.size(2))
        features = (features * k_weight).sum(dim = 1)
        query_features.append(features)

        query_features = torch.cat(query_features,dim = -1)

        return query_features
 
    # checked
    def get_codes(self, data):
        self.uv_projection = {}
        
        self.uv_projection['smpl_joints_RT'] = data['sp_input']['smpl_joints_RT_no_shape']
        self.uv_projection['smpl_posed_blending_shape'] = data['sp_input']['smpl_posed_blending_shape']

            
        Th = data['sp_input']['Th'].squeeze(0)
        Rot_T = data['sp_input']['R'].squeeze(0).permute(1,0)

        local_vertices = data['sp_input']['smpl_posed_vertex']
        world_vertices = data['sp_input']['smpl_world_vertex']

        images = data['latent_codes_image'] #[1,views,3,H,W]
        mask = data['latent_codes_mask']
        ext = data['latent_codes_ext']
        focal = data['latent_codes_focal']
        principle = data['latent_codes_principle']
        views = images.size(0)
        view_num = images.size(1)
        NS = focal.size(0)
        focal = focal
        c = principle
        loss = {}
        rotation = ext[:,:3,:3]
        translation = ext[:,:3,3]
        t_latent_codes_all_frame = []

        for view in range(views):
            image = images[view] # [Views, 3, H, W]
            image_batch = image
            mask_batch = mask
            rotation_batch = rotation
            translation_batch = translation
            focal_batch = focal
            c_batch = principle

            
            view_sum = image_batch.size(0)
                        
            focal_uv = focal_batch.unsqueeze(1).expand(view_sum,world_vertices.size(0),2)
            c_uv = c_batch.unsqueeze(1).expand(view_sum,world_vertices.size(0),2)
            self.uv_projection['view_sum'] = view_sum
            self.uv_projection['focal'] = focal_batch
            self.uv_projection['c'] = c_batch

            rotation_uv = rotation_batch.permute(0,2,1)
            translation_uv = translation_batch.unsqueeze(1)
            xyz = torch.matmul(world_vertices.expand(view_sum,self.latent_codes_num,3),rotation_uv) + translation_uv
            
            image_board = torch.tensor((image.size(2),image.size(3))).to(focal_uv.device)
            self.uv_projection['rotation_uv'] = rotation_uv
            self.uv_projection['translation_uv'] = translation_uv
            self.uv_projection['image_board'] = image_board
            self.uv_projection['Th'] = Th
            self.uv_projection['Rot_T'] = Rot_T
            self.uv_projection['image_batch'] = image_batch
            uv = xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
            uv *= focal_uv
            uv += c_uv
            
            camera_xyz = -torch.matmul(translation.unsqueeze(1),rotation_batch)
            self.uv_projection['camera_xyz'] = camera_xyz
            smpl_world_vex = world_vertices[:6890,:]

            camera_xyz = camera_xyz.expand(camera_xyz.size(0),world_vertices.size(0),world_vertices.size(1))
            self.uv_projection['world_vertices'] = smpl_world_vex
            camera_dirs = F.normalize(world_vertices.expand(camera_xyz.size(0),world_vertices.size(0),world_vertices.size(1)) - camera_xyz, p = 2 ,dim = -1)
  
            min_depth = self.args.occlusion_dis
            _, hit_min_depth, hit_max_depth = cloud_ray_intersect(min_depth, smpl_world_vex.unsqueeze(0), camera_xyz, camera_dirs)
            ppts_to_camera = torch.norm(world_vertices.expand(camera_xyz.size(0),world_vertices.size(0),world_vertices.size(1)) - camera_xyz, p = 2, dim = -1)
            hit_mask_for_smpl =  (ppts_to_camera < (hit_max_depth[:,:,0] + hit_min_depth[:,:,0])/2 ).float().unsqueeze(-1)

            
            mask_input_batch = []
            for i,mask_ori in enumerate(mask_batch):
                mask_input = torch.ones((mask_ori.size(0),mask_ori.size(1),1))
                mask_input[mask_ori == 0] = 0
                mask_input_batch.append(mask_input)
                # cv2.imwrite(f'debug/mask/mask_input_test_{i}.png',mask_input.cpu().numpy()*255)
                # cv2.imwrite(f'debug/mask/img_input_test_{i}.png',image_batch[i].permute(1,2,0).cpu().numpy()*255)

            mask_input = torch.stack(mask_input_batch,dim=0).permute(0,3,1,2).contiguous().to(mask_batch.device)
            mask_image_batch = torch.cat([image_batch,mask_input],dim=1)
  
            geo_map,_ = self.geo_image_encoder(mask_image_batch)

            app_map,_ = self.app_image_encoder(mask_image_batch)


            codes = {}
            feat_scale = []
            for i in range(len(geo_map)):
                feat_scale_i = torch.tensor([geo_map[i].shape[-1], geo_map[i].shape[-2]]).float().to(mask_batch.device)
                feat_scale_i = (feat_scale_i) / (feat_scale_i - 1)
                feat_scale.append(feat_scale_i)
            self.uv_projection['feat_scale'] = feat_scale

            
            geo_sampled = []
            for i in range(len(geo_map)):
                geo_sampled_i = extract_feature(geo_map[i], uv, feat_scale[i], image_board)
                geo_sampled.append(geo_sampled_i)

            geo_sampled = torch.cat(geo_sampled, dim = -1)
    
            # don't explicit use mask
            # mask_weights = F.normalize(hit_mask_for_smpl,p=1,dim=0)
            # weighted_mean = (geo_sampled * mask_weights).sum(0)
            # hit_mask_sign = (hit_mask_for_smpl.squeeze(-1) == 0)
            # hit_mask_index = hit_mask_sign.nonzero(as_tuple = False)[:,1]
            # geo_sampled[hit_mask_sign] = weighted_mean[hit_mask_index]
                    
                  
            geo_features = self.self_attention_geo(geo_sampled.permute(1,2,0).contiguous())
            geo_features = geo_features.mean(-1)
         
            sp_input = data['sp_input']
                
            geo_graph_input = sp_input['graph_input'].copy()
            geo_node_feat = geo_features.float()
                
            edge_feat = sp_input['graph_input']['posed_edge_feat'].float()
            edge_feat = self.gnn_edge_PE(edge_feat)
            geo_graph_input['edge_feat'] = edge_feat

            geo_graph_input['node_feat'] = geo_node_feat

            geo_codes = self.geo_gnn_inverse(geo_graph_input)
     
            codes['geo_map'] = geo_map
            codes['geo_codes'] = geo_codes
            

            
            app_sampled = []
            for i in range(len(app_map)):
                app_sampled_i = extract_feature(app_map[i], uv, feat_scale[i], image_board)
                app_sampled.append(app_sampled_i)

            app_sampled = torch.cat(app_sampled, dim = -1)

            # don't explicit use mask
            # mask_weights = F.normalize(hit_mask_for_smpl,p=1,dim=0)
            # weighted_mean = (app_sampled * mask_weights).sum(0)
            # hit_mask_sign = (hit_mask_for_smpl.squeeze(-1) == 0)
            # hit_mask_index = hit_mask_sign.nonzero(as_tuple = False)[:,1]
            # app_sampled[hit_mask_sign] = weighted_mean[hit_mask_index]
                   
            app_features = self.self_attention_app(app_sampled.permute(1,2,0).contiguous())
            app_features = app_features.mean(-1)
                       
            sp_input = data['sp_input']
                
            app_graph_input = sp_input['graph_input'].copy()
            app_node_feat = app_features.float()
                
            edge_feat = sp_input['graph_input']['posed_edge_feat'].float()
            edge_feat = self.gnn_edge_PE(edge_feat)
            app_graph_input['edge_feat'] = edge_feat

            app_graph_input['node_feat'] = app_node_feat
            app_codes = self.app_gnn_inverse(app_graph_input) # 6890,128
            codes['app_map'] = app_map
            codes['app_codes'] = app_codes
     

            return codes

    # checked  
    def ppts_canonical_to_posed(self,can_pts,transformation=None):
     
        weights = transformation['query_weights']
        vex_id = transformation['vex_id']
        bary_coords = transformation['bary_coords']
        ppts = can_pts.squeeze(0).float()
        scale_size = self.args.scale_size
        additional_Th = torch.tensor([0,0.3,0]).to(ppts.device) * scale_size
        ppts_normalized = ppts - additional_Th
        ppts_normalized /= scale_size 
        
        ppts_normalized = torch.cat([ppts_normalized, ppts_normalized.new_ones(ppts_normalized.size(0), 1)], -1)  # N x 4

        can_joints_RT = self.sp_input['can_pose_joints_RT_no_shape']

        new_weighted_RT = (weights @ can_joints_RT).reshape(-1, 4, 4)   # N x 4 x 4
        
        ppts_canonical = torch.einsum("ncd,nd->nc", torch.inverse(new_weighted_RT), ppts_normalized)
        
        joints_RT = self.uv_projection['smpl_joints_RT']
        weighted_RT = (weights @ joints_RT).reshape(-1, 4, 4)
        ppts_posed = torch.einsum("ncd,nd->nc", weighted_RT, ppts_canonical)
        ppts_posed *= scale_size 
        ppts_posed = ppts_posed[:,:3] + additional_Th

        bs_vex_posed = self.uv_projection['smpl_posed_blending_shape']
        bs_posed = (bs_vex_posed[vex_id,:] * bary_coords.unsqueeze(-1)).sum(1)
        ppts_posed = ppts_posed + bs_posed
        
        
        return ppts_posed

    # checked  
    def ppts_input_pose_to_driven_pose(self,ppts,transformation=None,geometry=False):
        if self.args.app_editing:
            geometry = True

        # only use for get pixel-aligned geometry features
        if transformation == None:
            ppts = ppts.squeeze(0).float() #[N,3]
            triangles = self.sp_input['smpl_posed_triangles']
            faces = self.sp_input['smpl_face'].squeeze(0)
            l_idx = torch.tensor([0,]).long().to(triangles.device)
            
            from clib._ext import point_face_dist_forward
            try:

                min_dis, min_face_idx, w0, w1, w2 = point_face_dist_forward(
                ppts, l_idx, triangles, l_idx, ppts.size(0)
                )
            except Exception as e:
                import pdb
                pdb.set_trace()
            vex_id = faces[min_face_idx].long() # B x 3 three vetices id. B: query point number
            bary_coords = torch.stack([w0, w1, w2], 1)   # B x 3  ## three vertiyces' weights
            vex_weight = self.sp_input['skinning_weight']
            weights = (vex_weight[vex_id] * bary_coords.unsqueeze(-1)).sum(1)
        else:
            weights = transformation['query_weights']
            bary_coords = transformation['bary_coords']    
            vex_id = transformation['vex_id']    

        # input pose's blend shape
        input_bs_posed = self.sp_input['smpl_posed_blending_shape']
        input_bs_posed = (input_bs_posed[vex_id,:] * bary_coords.unsqueeze(-1)).sum(1)

        ppts = ppts - input_bs_posed
        joints_RT = self.sp_input['smpl_joints_RT_no_shape']
        # Separate legs
        scale_size = self.args.scale_size
        additional_Th = torch.tensor([0,0.3,0]).to(ppts.device) * scale_size
        ppts_normalized = ppts - additional_Th
        ppts_normalized /= scale_size 
        ppts_normalized = torch.cat([ppts_normalized, ppts_normalized.new_ones(ppts_normalized.size(0), 1)], -1)  # N x 4
        weighted_RT = (weights @ joints_RT).reshape(-1, 4, 4)
        ppts_canonical = torch.einsum("ncd,nd->nc", torch.inverse(weighted_RT), ppts_normalized)
        if geometry:
            joints_RT = self.uv_projection_geometry['smpl_joints_RT']
        else:
            joints_RT = self.uv_projection['smpl_joints_RT']
        weighted_RT = (weights @ joints_RT).reshape(-1, 4, 4)

        ppts_posed = torch.einsum("ncd,nd->nc", weighted_RT, ppts_canonical)
        ppts_posed *= scale_size 
        ppts_posed = ppts_posed[:,:3] + additional_Th  

        if geometry:
            driven_bs_posed = self.uv_projection_geometry['smpl_posed_blending_shape']
        else:
            driven_bs_posed = self.uv_projection['smpl_posed_blending_shape']

        driven_bs_posed = (driven_bs_posed[vex_id,:] * bary_coords.unsqueeze(-1)).sum(1)
        ppts_posed = ppts_posed + driven_bs_posed

        return ppts_posed

    # checked
    def get_inverse_transform(self, ppts):
      
        vertices = self.sp_input['smpl_posed_vertex']
        ppts = ppts.squeeze(0).float() #[N,3]
        triangles = self.sp_input['smpl_posed_triangles']
        faces = self.sp_input['smpl_face'].squeeze(0)
        l_idx = torch.tensor([0,]).long().to(triangles.device)
        from clib._ext import point_face_dist_forward
        min_dis, min_face_idx, w0, w1, w2 = point_face_dist_forward(
            ppts, l_idx, triangles, l_idx, ppts.size(0)
        )
        vex_id = faces[min_face_idx].long() # B x 3 three vetices id. B: query point number
        bary_coords = torch.stack([w0, w1, w2], 1)   # B x 3  ## three vertiyces' weights

        vex_weight = self.sp_input['skinning_weight']
        weights = (vex_weight[vex_id] * bary_coords.unsqueeze(-1)).sum(1)
        if self.iteration >= self.args.begin_optimize_residual:
       
            batch_size = self.args.knn_batch_size
            from clib._ext import point_face_dist_forward
            residual_features_batch = []
            app_features_batch = []
            for i in range(0,ppts.size(0),batch_size):
                ppts_batch = ppts[i:i+batch_size,:]
                features = self.fuse_feature_with_distance(vertices, self.geo_posed_codes, ppts_batch)
                residual_features_batch.append(features)

            residual_features = torch.cat(residual_features_batch,dim = 0)
          
        
            transformation = {}
            transformation['query_weights'] = weights
            transformation['bary_coords'] = bary_coords
            transformation['vex_id'] = vex_id
            forward_ppts = self.ppts_input_pose_to_driven_pose(ppts.clone(),transformation)
            residual_features = self.get_pixel_aligned_feature(forward_ppts,surface_feature=residual_features,geometry=True)

            bs_vex_posed = self.sp_input['smpl_posed_blending_shape']
            bs_posed = (bs_vex_posed[vex_id,:] * bary_coords.unsqueeze(-1)).sum(1)
            ppts = ppts - bs_posed
            # write_ply(ppts.cpu().numpy(),'debug/canonical_shape/posed_no_shape_ppts.ply')
            joints = self.sp_input['smpl_local_joints_no_shape']
      
            
            ppts_joints_distance = ppts.unsqueeze(1) - joints.unsqueeze(0)
    
            ppts_joints_norm = ppts_joints_distance.norm(p=2,dim=-1,keepdim=True)
            # ppts_joints_distance /= ppts_joints_norm
            ppts_joints_distance = ppts_joints_distance / ppts_joints_norm
      
            coefficient_input = [ppts_joints_distance,ppts_joints_norm]

            coefficient_input = torch.cat(coefficient_input,dim=-1) # [ppts_number, joints_number, 9+4]
            
 
            joint_coefficient = coefficient_input
            
            residual_input = [joint_coefficient.view(ppts.size(0),-1), residual_features] 

            residual_input = torch.cat(residual_input, dim=-1) # B x (24*9 + 128)
            residual = self.residual_deform(residual_input) # B X 3
           
        else:
            bs_vex_posed = self.sp_input['smpl_posed_blending_shape']
            bs_posed = (bs_vex_posed[vex_id,:] * bary_coords.unsqueeze(-1)).sum(1)
            ppts = ppts - bs_posed
            
            
        scale_size = self.args.scale_size
        additional_Th = torch.tensor([0,0.3,0]).to(ppts.device) * scale_size
        ppts_normalized = ppts - additional_Th
        ppts_normalized = ppts_normalized / scale_size 

        joints_RT = self.sp_input['smpl_joints_RT_no_shape']

        weighted_RT = (weights @ joints_RT).reshape(-1, 4, 4)
        ppts_normalized = torch.cat([ppts_normalized, ppts_normalized.new_ones(ppts_normalized.size(0), 1)], -1)  # N x 4
        ppts_canonical = torch.einsum("ncd,nd->nc", torch.inverse(weighted_RT), ppts_normalized)


        can_joints_RT = self.sp_input['can_pose_joints_RT_no_shape']
        new_weighted_RT = (weights @ can_joints_RT).reshape(-1, 4, 4)   # N x 4 x 4
        ppts_canonical = torch.einsum("ncd,nd->nc", new_weighted_RT, ppts_canonical)

            
        ppts_canonical = ppts_canonical * scale_size
        ppts_canonical = ppts_canonical[:,:3] + additional_Th
        ppts_canonical_wo_residual = ppts_canonical.clone()
        ppts_canonical = ppts_canonical + residual
        
        return {'can_pts_wo_residual': ppts_canonical_wo_residual, 'residual': residual, 'query_weights' : weights, 'vex_id' : vex_id, 'bary_coords' : bary_coords} , ppts_canonical.float()

    # checked
    def get_app_surface_feature(self, ppts, forward_posed_ppts=None, transformation = None):
        ppts = ppts.reshape(-1,3)


        codes = self.app_can_codes

        vertices = self.sp_input['smpl_canonical_vertex_no_shape']
        triangles = self.sp_input['smpl_canonical_triangles_no_shape']

        batch_size = self.args.knn_batch_size
        features_batch = []

        for i in range(0,ppts.size(0),batch_size):
            ppts_batch = ppts[i:i+batch_size,:]

            features = self.fuse_feature_with_distance(vertices,codes,ppts_batch,appearance_feature=True)
            features_batch.append(features)

        query_features = torch.cat(features_batch,dim = 0)

        return query_features.float()

    # checked
    def get_pixel_aligned_feature(self, ppts, surface_feature = None, geometry=False):

        if self.args.app_editing and geometry:
            uv_projection = self.uv_projection_geometry
        else:
            uv_projection = self.uv_projection

        Th = uv_projection['Th']
        Rot = uv_projection['Rot_T']
        ## smpl_space to world space
        ppts = torch.matmul(ppts, Rot) + Th
        
        view_sum = uv_projection['view_sum']
        focal_uv = uv_projection['focal'].unsqueeze(1).expand(view_sum,ppts.size(0),2)
        c_uv = uv_projection['c'].unsqueeze(1).expand(view_sum,ppts.size(0),2)
        rotation_uv = uv_projection['rotation_uv']
        translation_uv = uv_projection['translation_uv']
        image_board = uv_projection['image_board']
        feat_scale = uv_projection['feat_scale']
        
        xyz = torch.matmul(ppts.expand(view_sum,ppts.size(0),3),rotation_uv) + translation_uv
        uv = xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
        uv *= focal_uv
        uv += c_uv        


        ## calculate occlusion mask
    
        camera_xyz = uv_projection['camera_xyz']
        camera_xyz = camera_xyz.expand(camera_xyz.size(0),ppts.size(0),ppts.size(1))
        world_vertices = uv_projection['world_vertices']
        camera_dirs = F.normalize(ppts.expand(camera_xyz.size(0),ppts.size(0),ppts.size(1)) - camera_xyz, p = 2 ,dim = -1)

        min_depth = self.args.occlusion_dis
        _, hit_min_depth, hit_max_depth = cloud_ray_intersect(min_depth, world_vertices.unsqueeze(0), camera_xyz.contiguous(), camera_dirs.contiguous())
        ppts_to_camera = torch.norm(ppts.expand(camera_xyz.size(0),ppts.size(0),ppts.size(1)) - camera_xyz, p = 2, dim = -1)
        hit_mask_for_smpl =  (ppts_to_camera < (hit_max_depth[:,:,0] + hit_min_depth[:,:,0])/2 ).float().unsqueeze(-1)

        
        if geometry:
            feature_map = self.codes['geo_map']
        else:
            feature_map = self.codes['app_map']

        feature_sampled = []
        for i in range(len(feature_map)):
            feature_sampled_i = extract_feature(feature_map[i], uv, feat_scale[i], image_board, allow_grad=True)
            feature_sampled.append(feature_sampled_i)
        feature_sampled = torch.cat(feature_sampled, dim = -1)
      
        # don't explicit use mask
        # mask_weights = F.normalize(hit_mask_for_smpl,p=1,dim=0)
        # weighted_mean = (feature_sampled * mask_weights).sum(0)
        # hit_mask_sign = (hit_mask_for_smpl.squeeze(-1) == 0)
        # hit_mask_index = hit_mask_sign.nonzero(as_tuple = False)[:,1]
        # feature_sampled[hit_mask_sign] = weighted_mean[hit_mask_index]

        ## occlusion-aware cross attention
        if geometry :
            feature_sampled = self.geo_pixel_embedding(feature_sampled)
            feature_sampled = self.geo_cross_attention(feature_sampled.permute(1,2,0),hit_mask_for_smpl.permute(1,2,0).contiguous()).mean(-1)          
            feature_sampled = torch.cat([feature_sampled,surface_feature],dim=-1)
            final_features = self.geo_cross_attention_embedding(feature_sampled)
             
        else:
            feature_sampled = self.app_pixel_embedding(feature_sampled)
            feature_sampled = self.app_cross_attention(feature_sampled.permute(1,2,0),hit_mask_for_smpl.permute(1,2,0).contiguous()).mean(-1)          
            feature_sampled = torch.cat([feature_sampled,surface_feature],dim=-1)
            final_features = self.app_cross_attention_embedding(feature_sampled)
         
        return final_features
 
    # checked
    def fuse_codes(self, posed_space=False, geometry=True):
        if geometry:
            codes = self.codes['geo_codes']
        else:
            codes = self.codes['app_codes']

        if posed_space:
            vertices_xyz = self.sp_input['smpl_posed_vertex']
        else:
            vertices_xyz = self.sp_input['smpl_canonical_vertex']
        

        node_feat = []
        node_feat.append(codes)
       
        node_feat = torch.cat(node_feat,dim=-1)

        graph_input = self.sp_input['graph_input'].copy()

        graph_input['node_feat'] = node_feat.float()
        if posed_space:
            edge_feat = self.sp_input['graph_input']['posed_edge_feat'].float()
        else:
            edge_feat = self.sp_input['graph_input']['can_edge_feat'].float()
        
        edge_feat = self.gnn_edge_PE(edge_feat)
        
        graph_input['edge_feat'] = edge_feat

        if geometry:
            output = self.geo_gnn_forward(graph_input)    
        else:
            output = self.app_gnn_forward(graph_input)

        gnn_features = output

        if posed_space:
            if geometry:
                self.geo_posed_codes = gnn_features
            else:
                AssertionError("posed appearance codes not used!")

        else:
            if geometry:
                AssertionError("canonical geometry codes not used!")
            else:
                if self.args.head_no_skeletal:
                    gnn_features[self.head_id] = 0

                self.app_can_codes = gnn_features


              
    def fuse_codes_all(self):
        if self.iteration >= self.args.begin_optimize_residual or not self.args.is_train:
            self.fuse_codes(geometry=True,posed_space=True)

        self.fuse_codes(geometry=False,posed_space=False)
        
    


