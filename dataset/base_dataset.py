import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import os
from os.path import join 
import imageio
import glob
import cv2
import torch
import torch.nn.functional as F
import copy
import time
import re
import tqdm
from . import data_utils
from .data_utils import load_face,  batch_rodrigues, parse_character, extreme_hue_color_jitter
import trimesh
import sys
sys.path.append('tools/get_blending_weight')
from smpl_webuser.serialization import load_model

class BaseDataset(data.Dataset):
    def __init__(self, args, conf, root_dir, views=None, subsample_interval = -1, is_train = True, iteration = 0, target_app = False):
        super().__init__()
        self.args = args
        self.cfg = conf
        self.device = torch.device(f'cuda:{self.args.gpu_id[0]}')
        self.root_dir = root_dir

        self.views = views            

        if target_app:
            self.characters = [self.args.app_target]
        else:
            self.characters = parse_character(self.args.characters)

        # if self.args.app_editing: ## todo::
            # self.characters.append(self.args.app_target)
            # self.characters = sorted(self.characters)

        if self.args.aist_data:
            self.args.nb_another_mask = False
            self.args.camera_undistort = False
        
        if self.args.na_data:
            self.args.camera_undistort = False
            self.args.nb_another_mask = False


        self.character_num = len(self.characters)
        self.train = is_train

        self.subsample_interval = subsample_interval

        

        self.smpl = self.find_smpl(self.root_dir) # 

        self.character_min_id = min(self.characters) 
        self.characters_index = {}
        ## characters name to index
        for i,j in enumerate(self.characters):
            self.characters_index[j] = i


        ## dataloader: frame start and end
        if self.train:
            self.start_end = args.train_start_end
        else:
            self.start_end = args.test_start_end
            
        

        _data_per_character = {}

        _data_per_character['transform'] =  self.smpl
  
        ## each characters has a list of smpl parameters
        self.frames = [0 for i in range(self.character_num)]

        if self.start_end is not None:
            start, end = eval(self.start_end)
            self.start_index = start
            for id in range(self.character_num):
                for key in _data_per_character:
                    _data_per_character[key][id] = _data_per_character[key][id][start:end]
                    self.frames[id] = len(_data_per_character[key][id]) 

        if self.subsample_interval > -1:   
            for id in range(self.character_num):

                for key in _data_per_character:
                    _data_per_character[key][id] = _data_per_character[key][id][::self.subsample_interval]
                    self.frames[id] = len(_data_per_character[key][id]) 
            

        self.all_shape_num = len([ii for i in self.smpl for ii in i])
        

        characters = self.characters
        character_num = self.character_num
        frames = self.frames
        _data_per_character['character_id'] = list([[characters[i] for times in range(frames[i]) ] for i in range(character_num)]) ## each data has a character id

        ## each data has a global frame id, sorted by character id
        acc_frames = 0
        self.acc_frames = []
        for i in range(self.character_num):
            self.acc_frames.append(acc_frames)
            acc_frames += self.frames[i]

        print('[INFO]: each character has {} frames'.format(self.frames))
        print('[INFO]: all loaded data num',self.all_shape_num)
        print('--------------------------------------------------------------------')

        ## data for smpl useage
        if self.args.aist_data:
            self.m, self.A, _ = load_model( 'tools/get_blending_weight/models/SMPL_MALE.pkl')
        else:
            self.m, self.A, _ = load_model( 'tools/get_blending_weight/SMPL_NEUTRAL.pkl')
        self.A_poses = np.load('tools/apose_pose.npy',allow_pickle=True)
        if self.cfg['model.using_smplx']:
            self.face = torch.from_numpy(load_face('tools/smplx_face.txt')).unsqueeze(0)
        else:
            self.face = torch.from_numpy(load_face('tools/smpl_face.txt')).unsqueeze(0)
        self.skinning_weight = torch.from_numpy(np.loadtxt( 'tools/skinning_weight.txt')).float()
        self.can_pose_joints_RT = torch.from_numpy(np.array(json.load(open('tools/transform_apose.json'))['joints_RT'])).permute(2,0,1).reshape(1,-1,16)
        smpl_canonical_pose = np.load('tools/apose_pose.npy',allow_pickle=True)
        self.smpl_canonical_pose = batch_rodrigues(torch.from_numpy(smpl_canonical_pose)).view(-1)

        # group the data..
        data_list = []
     
        for id in range(self.character_num):
            for frame in range(self.frames[id]): 
                element = {}
                for key in _data_per_character:
                    element[key] = _data_per_character[key][id][frame]
                data_list.append(element)

        # group the data together   
        self.shape_data = data_list
        _data_per_view = {}
        _data_per_view['rgb'] = self.find_rgb()
     
        if self.args.na_with_mask or self.args.realworld_data:
            self.rgb_with_bg = True
        if _data_per_view['rgb'] is not None:
            if self.rgb_with_bg:
                _data_per_view['mask'] = self.find_mask()
                if self.args.nb_another_mask:
                    _data_per_view['nb_another_mask'] = self.find_nb_another_mask()                    
            _data_per_view['ext'] = self.find_extrinsics()
            if self.args.camera_undistort:
                _data_per_view['undistort'] = self.find_undistort()
            if self.find_intrinsics_per_view() is not None:
                _data_per_view['ixt_v'] = self.find_intrinsics_per_view()
        
            self.summary_view_data(_data_per_view)

        else:
            print("[ERROR]: rgb images not found!\n")
            exit()




    def tocpu(self,data):
        for key,value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to('cpu')
            elif isinstance(value,dict):
                for sub_key,sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        data[key][sub_key] = sub_value.to('cpu')
                    elif isinstance(sub_value,dict):
                        for sub_sub_key,sub_sub_value in sub_value.items():
                            if torch.is_tensor(sub_sub_value):
                                data[key][sub_key][sub_sub_key] = sub_sub_value.to('cpu')
        return data

    def is_to_gpu(self,tensor):
        # if tensor.type()!="torch.IntTensor":
        #     return True
        # else:
        #     return False
        return True

    def tocuda(self, data, device = None, augmentation = None):
        if device == None:
            device = self.device
            
        new_data = dict()
        for key,value in data.items():
            if key == 'loss_image' or key == 'latent_codes_image' or key == 'loss_mask':
                new_data[key] = value.to(device)
            else:
                if torch.is_tensor(value) and self.is_to_gpu(value):
                    # print(value.type())
                    # print('ok')
                    new_data[key] = value.clone().to(device)
                    # data[key].to(device)
                elif isinstance(value,dict):
                    new_data[key] = {}
                    for sub_key,sub_value in value.items():
                        if torch.is_tensor(sub_value) and self.is_to_gpu(sub_value):
                            # print(sub_value.type())
                            # data[key][sub_key].to(device)
                            new_data[key][sub_key] = sub_value.clone().to(device)
                        elif isinstance(sub_value,dict):
                            new_data[key][sub_key] = {}
                            for sub_sub_key,sub_sub_value in sub_value.items():
                                # print(sub_sub_value.type())
                                if torch.is_tensor(sub_sub_value) and self.is_to_gpu(sub_sub_value):
                                    new_data[key][sub_key][sub_sub_key] =  sub_sub_value.clone().to(device)
                                    # data[key][sub_key][sub_sub_key].to(device)
                                else:
                                    new_data[key][sub_key][sub_sub_key] = sub_sub_value

                        else:
                            new_data[key][sub_key] = sub_value
                else:
                    new_data[key] = value

        return new_data
    
    def _load_view(self, packed_data, view_idx):
        ## load image, mask, camera data for one view
        extrinsics = data_utils.load_matrix(packed_data['ext'][view_idx]) 
        extrinsics = data_utils.parse_extrinsics(extrinsics, self.world2camera).astype('float32')  # this is C2W
        intrinsics = data_utils.load_intrinsics(packed_data['ixt_v'][view_idx]).astype('float32') \
            if packed_data.get('ixt_v', None) is not None else None
        if self.args.camera_undistort:
            undistort = np.load(packed_data['undistort'][view_idx],allow_pickle=True)
        mask = None
        nb_another_mask = None
        if packed_data.get('mask', None) is not None:
            mask = packed_data.get('mask', None)
        if packed_data.get('nb_another_mask', None) is not None:
            nb_another_mask = packed_data.get('nb_another_mask', None)
        return {
            'path': packed_data['rgb'][view_idx],
            'view': view_idx,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'undistort': undistort if self.args.camera_undistort else None,
            'mask': mask[view_idx] if mask is not None else None,
            'nb_another_mask': nb_another_mask[view_idx] if nb_another_mask is not None else None,
        }

        
    def _load_shape(self, packed_data):
        ## load shape data for each frame, e.g. smpl
        shape_id = packed_data['character_id']
        shape_data = {'id': shape_id}
        if packed_data.get('glb', None) is not None:   # additional global feature (if any)
            shape_data['global_index'] = np.loadtxt(packed_data['glb']).astype('int64')
        if packed_data.get('vertex', None) is not None:
            shape_data['vertex'] = np.load(packed_data.get('vertex', None)).astype('float32')
        if packed_data.get('smpl', None) is not None:
            shape_data['smpl'] = np.load(packed_data.get('smpl', None),allow_pickle = True).item()
        if packed_data.get('mesh', None) is not None and self.cfg['dataset.using_3d']:
            shape_data['mesh'] = trimesh.load(packed_data.get('mesh', None))  
        if packed_data.get('transform', None) is not None and self.cfg['dataset.using_blending_weight']:
            shape_data['transform'] = json.load(open(packed_data.get('transform', None)))
        if packed_data.get('jnt', None) is not None:
            data = json.load(open(packed_data['jnt']))
            for key in data:
                shape_data[key] = np.asarray(data[key]).astype('float32')
            
            rodrigues = lambda x : cv2.Rodrigues(x)[0]
   
            Rh = rodrigues(shape_data['rotation'].transpose(1,0))[:,0]
            if len(shape_data['pose'].shape) == 2:
                shape_data['pose'] = np.concatenate([Rh[None, :], shape_data['pose'][:, 3:]], -1)
            else:
                shape_data['pose'] = np.concatenate([Rh, shape_data['pose'][3:]], -1)
            
        if packed_data.get('tex', None) is not None:
            shape_data['tex'] = data_utils.load_rgb(packed_data['tex'], with_alpha=False)
        
        return shape_data

    def find_smpl(self, paths):
        if self.character_num == 0:
            ## load all character
            import re
            self.characters = [i for i in os.listdir(paths) if re.match(r'\d+',i)]  
            self.character_num = len(self.characters)
            print(f"[INFO]: Load all {self.characters} characters.")

        all_paths = []
        if self.cfg['model.using_smplx']:
            self.params_name = 'transform'
        else:
            self.params_name = 'smpl_transform'


        for id in self.characters:
            smpl_list = sorted(glob.glob(paths + f'/{id}/{self.params_name}/*.json'))
            all_paths.append(smpl_list)

        return all_paths

    def find_normal(self):
        all_normal= []
        if self.cfg['model.using_smplx']:
            normal_name = 'normal'
        else:
            normal_name = 'smpl_normal'
        for id in self.characters:
            home_path = self.root_dir
            if os.path.exists(home_path + f'/{id}/{normal_name}/'):
                normal_list = sorted(glob.glob(home_path + f'/{id}/{normal_name}/*.npy'))
            # assert len(tex_list) == len(self.paths), "the number of joints did not match"
            else:
                print("normal not found!\n")
                exit(1)
            all_normal.append(normal_list)
        return all_normal      

    def find_intrinsics(self):
        all_ixt = []
        for id in self.characters:
            home_path = self.root_dir
            if os.path.exists(home_path + '/intrinsic.txt'):
                ixt = home_path + '/intrinsic.txt'
            elif os.path.exists(home_path + '/intrinsics.txt'):
                ixt = home_path + '/intrinsics.txt'
            else:
                return None
            all_ixt.append(ixt)
        return all_ixt
    
    def find_vertex(self):
        all_vex= []
        if self.cfg['model.using_smplx']:
            vertex_name = 'vertex'
        else:
            vertex_name = 'smpl_vertex'
        for id in self.characters:
            home_path = self.root_dir
            if os.path.exists(home_path + f'/{id}/{vertex_name}/'):
                vex_list = sorted(glob.glob(home_path + f'/{id}/{vertex_name}/*.npy'))
            # assert len(tex_list) == len(self.paths), "the number of joints did not match"
            else:
                print("vertex not found!\n")
                exit(1)
            all_vex.append(vex_list)
        # return [i for ii in all_vex for i in ii]
        return all_vex
        
    def find_transform(self):
        all_transform= []
        if self.cfg['model.using_smplx']:
            transform_name = 'transform'
        else:
            transform_name = 'smpl_transform'
        for id in self.characters:
            home_path = self.root_dir
            if os.path.exists(home_path + f'/{id}/{transform_name}/'):
                transform_list = sorted(glob.glob(home_path + f'/{id}/{transform_name}/*.json'))
            # assert len(tex_list) == len(self.paths), "the number of joints did not match"
            else:
                print("transform not found!\n")
                exit(1)
            all_transform.append(transform_list)
        # return [i for ii in all_transform for i in ii]
        return all_transform
    
    
    def find_mesh(self):
        all_mesh = []
        for id in self.characters:
            home_path = self.root_dir
            fdir = 'mesh'
            if os.path.exists(home_path + f'/{id}/{fdir}'):
                mesh_list = sorted(glob.glob(home_path + f'/{id}/{fdir}/*.obj'))
            # assert len(tex_list) == len(self.paths), "the number of joints did not match"
                all_mesh.append(mesh_list)
        # return [i for ii in all_mesh for i in ii]
        return all_mesh

    def find_textures(self):
        all_tex = []
        for id in self.characters:
            home_path = self.root_dir
            fdir = 'tex'
            if os.path.exists(home_path + f'/{id}/{fdir}'):
                tex_list = sorted(glob.glob(home_path + f'/{id}/{fdir}/*.png'))
                all_tex.append(tex_list)
                # assert len(tex_list) == len(self.paths), "the number of joints did not match"
        # return [i for ii in all_tex for i in ii]
        return all_tex
    
    def find_rgb(self):
        if self.args.na_crop:
            self.rgb_with_bg = False
            return self.select([sorted(glob.glob(path.replace(self.params_name, 'rgb_crop').replace('.json', '/*.*g'))) for paths in self.smpl for path in paths])
        elif os.path.exists(os.path.join(self.root_dir,str(self.characters[0])+'/rgb/')):
            self.rgb_with_bg = False
            return self.select([sorted(glob.glob(path.replace(self.params_name, 'rgb').replace('.json', '/*.*g'))) for paths in self.smpl for path in paths])
        elif os.path.exists(os.path.join(self.root_dir,str(self.characters[0])+'/rgb_bg/')):
            self.rgb_with_bg = True
            before_select = [sorted(glob.glob(path.replace(self.params_name, 'rgb_bg').replace('.json', '/*.*g'))) for paths in self.smpl for path in paths]
            return self.select(before_select)
        else:
            return None

    def find_mask(self):
        # if os.path.exists(os.path.join(self.root_dir,'0/mask/')):
        if self.rgb_with_bg == True:
            if self.args.aist_data:
                return self.select([sorted(glob.glob(path.replace(self.params_name, 'mask_pgn').replace('.json', '/*.*g'))) for paths in self.smpl for path in paths])
            else:
                return self.select([sorted(glob.glob(path.replace(self.params_name, 'mask').replace('.json', '/*.*g'))) for paths in self.smpl for path in paths])
        else:
            return None

    def find_nb_another_mask(self):
        if self.rgb_with_bg == True:
            return self.select([sorted(glob.glob(path.replace(self.params_name, 'nb_another_mask').replace('.json', '/*.*g'))) for paths in self.smpl for path in paths])
        else:
            return None
        
    def find_intrinsics_per_view(self):
        all_int = []
        if self.args.na_crop:
            return self.select([sorted(glob.glob(path.replace(self.params_name, 'intrinsic_crop').replace('.json', '/*.txt'))) for paths in self.smpl for path in paths])
        for id in self.characters:
            home_path = self.root_dir
            if os.path.exists(home_path + f'/{id}' + '/intrinsic'):
                # all_int.append((self.select([sorted(glob.glob(home_path + f'/{id}' + '/intrinsic/*.txt'))]))*(self.all_num//self.character_num))
                all_int.append((self.select([sorted(glob.glob(home_path + f'/{id}' + '/intrinsic/*.txt'))]))*(self.frames[self.characters_index[id]]))
        return [i for ii in all_int for i in ii]

    def find_extrinsics(self):
        all_ext = []
        for id in self.characters:
            home_path = self.root_dir
            if os.path.exists(home_path + f'/{id}' + '/extrinsic'):
                self.world2camera = False
                path = home_path + f'/{id}' + '/extrinsic'
            elif os.path.exists(home_path + f'/{id}' + '/pose'):
                self.world2camera = True
                path = home_path + f'/{id}' +  '/pose'
            else:
                raise FileNotFoundError('world2camera or camera2world matrices not found.') 
            # all_ext.append((self.select([sorted(glob.glob(path + '*/*.txt'))]))*(self.all_num//self.character_num))
            all_ext.append((self.select([sorted(glob.glob(path + '*/*.txt'))]))*(self.frames[self.characters_index[id]]))
        return [i for ii in all_ext for i in ii]
    
    def find_undistort(self):
        all_ds = []
        for id in self.characters:
            home_path = self.root_dir
            if os.path.exists(home_path + f'/{id}' + '/undistort'):
                path = home_path + f'/{id}' + '/undistort'
            else:
                raise FileNotFoundError('Undistort parameters not found.') 
            # all_ds.append((self.select([sorted(glob.glob(path + '*/*.npy'))]))*(self.all_num//self.character_num))
            all_ds.append((self.select([sorted(glob.glob(path + '*/*.npy'))]))*(self.frames[self.characters_index[id]]))
        return [i for ii in all_ds for i in ii]
    
    def summary_view_data(self, _data_per_view):
        keys = [k for k in _data_per_view if _data_per_view[k] is not None]
        num_of_objects = len(_data_per_view[keys[0]])
        index_to_shape_id = []
        index_to_view_id = []
        shape_id_to_view_num = []
        shape_id_to_index = []
        view_data = []
        index = 0
      
        for shape_id in range(num_of_objects):
            shape_id_to_index.append(index)
            view_data_object = _data_per_view[keys[0]][shape_id]
            view_num = len(view_data_object)
            shape_id_to_view_num.append(view_num)
            element = {}
            for key in keys:
                element[key] = [_data_per_view[key][shape_id][i] for i in range(view_num)]
            for view_id in range(view_num):
                index_to_shape_id.append(shape_id)
                index_to_view_id.append(view_id)
                index += 1
            view_data.append(element)
            
        self.all_num = index
        self.index_to_shape_id = index_to_shape_id
        self.index_to_view_id = index_to_view_id
        self.shape_id_to_view_num = shape_id_to_view_num
        self.shape_id_to_index = shape_id_to_index
        self.view_data = view_data
    
    ## ground truth obj
    def find_mesh(self):
        all_mesh = []
        for id in self.characters:
            home_path = self.root_dir
            fdir = 'mesh'
            if os.path.exists(home_path + f'/{id}/{fdir}'):
                mesh_list = sorted(glob.glob(home_path + f'/{id}/{fdir}/*.obj'))
            # assert len(tex_list) == len(self.paths), "the number of joints did not match"
                all_mesh.append(mesh_list)
        return all_mesh        
        # return [i for ii in all_mesh for i in ii]        

    def find_samples(self):
        all_samples = []
        for id in self.characters:
            home_path = self.root_dir
            fdir = 'sample'
            if os.path.exists(home_path + f'/{id}/{fdir}'):
                sample_list = sorted(glob.glob(home_path + f'/{id}/{fdir}/samples*.mat'))
            # assert len(tex_list) == len(self.paths), "the number of joints did not match"
                all_samples.append(sample_list)
        # return [i for ii in all_samples for i in ii]    
        return all_samples

    def select(self, file_list):
        return file_list # do not select views

    def get_one_item_shape(self, data_shape, pose_driven = False):
        data = {}
        character_id = data_shape['id']
        data['character_id'] = character_id
        
        params = data_shape['transform']
        if self.args.na_data and not pose_driven:
            params = params[0]
        
        Rh = np.array(params['Rh']).astype(np.float32) # world_to_local
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            
        if self.args.na_data and not pose_driven:
            shape = np.array(params['shapes']).astype(np.float32)[:,:10]
        else:
            shape = np.array(params['shapes']).astype(np.float32)

        pose = np.array(params['poses']).astype(np.float32)

        if self.train and self.args.smpl_augmentation:
            shape_noise = np.random.normal(0.0, 0.05, shape.shape)
            pose_noise = np.random.normal(0.0, 0.05, pose.shape)
            shape += shape_noise
            pose += pose_noise

        Th = np.array(params['Th']).astype(np.float32) * self.args.scale_size  # local_to_world
        if self.args.aist_data:
            Th /= 100.
            smpl_scale = params['scale'] / 100
        else:
            smpl_scale = 1.0
        addtional_th = np.array([0.,-0.3,0.]) * self.args.scale_size
        Th += addtional_th @ R.T
        
        self.smpl_scale_size = smpl_scale * self.args.scale_size
        
        sp_input = {}
        R = torch.from_numpy(R)
        Th = torch.from_numpy(Th)
        sp_input['smpl_params'] = params
        sp_input['R'] = R
        sp_input['Th'] = Th
        sp_input['skinning_weight'] = self.skinning_weight



        '''
            canonical pose, original shape
        '''
        self.m.pose[:] = self.A_poses.reshape(-1)
        self.m.betas[:] = shape
        can_vertices = self.m.r.copy()
        can_vertices *= self.smpl_scale_size
        can_joints = self.m.J_transformed.r.copy()
        can_joints *= self.smpl_scale_size
        can_vertices -= addtional_th
        can_joints -= addtional_th
        can_vertices = torch.from_numpy(can_vertices).float()
        can_joints_RT = torch.from_numpy(self.A.r.copy()).float() # checked.
        can_joints_RT[:3,3,:] *= smpl_scale
        sp_input['can_pose_joints_RT'] = can_joints_RT.permute(2,0,1).reshape(1,-1,16)

        sp_input['smpl_canonical_joints'] = torch.from_numpy(can_joints)
        sp_input['smpl_canonical_vertex'] = can_vertices
        
        '''
            canonical pose, no shape
        '''
        # if self.args.na_data:
            # self.m.betas[:] = np.zeros(300)
        # else:
        self.m.betas[:] = np.zeros(10)
        can_vertices_no_shape = self.m.r.copy() 
        can_vertices_no_shape *= self.args.scale_size
        can_joints_no_shape = self.m.J_transformed.r.copy()
        can_joints_no_shape *= self.args.scale_size
        can_vertices_no_shape -= addtional_th
        can_joints_no_shape -= addtional_th
        can_vertices_no_shape = torch.from_numpy(can_vertices_no_shape).float()
        can_joints_RT_no_shape = torch.from_numpy(self.A.r.copy()).float() # checked.
        sp_input['can_pose_joints_RT_no_shape'] = can_joints_RT_no_shape.permute(2,0,1).reshape(1,-1,16)
      
        sp_input['smpl_canonical_blending_shape'] = can_vertices - can_vertices_no_shape
        
        
        '''
            original pose, no shape
        '''
        self.m.pose[:] = pose
        posed_vertex_no_shape = self.m.r.copy()  
        posed_vertex_no_shape *= self.args.scale_size
        posed_vertex_no_shape -= addtional_th
        posed_vertex_no_shape = torch.from_numpy(posed_vertex_no_shape).float()
        posed_joints_no_shape = self.m.J_transformed.r.copy()
        posed_joints_no_shape *= self.args.scale_size
        posed_joints_no_shape -= addtional_th
        posed_joints_no_shape = torch.from_numpy(posed_joints_no_shape).float()
        joints_RT_no_shape = torch.from_numpy(self.A.r.copy()).float() # checked.
        sp_input['smpl_joints_RT_no_shape'] = joints_RT_no_shape.permute(2,0,1).reshape(1,-1,16)
        '''
            original pose, original shape
        '''
        self.m.betas[:] = shape
        posed_vex = self.m.r.copy()
        posed_vex *= self.smpl_scale_size
        posed_joints = self.m.J_transformed.r.copy()
        posed_joints *= self.smpl_scale_size
        posed_vex -= addtional_th
        posed_joints -= addtional_th
        joints_RT = torch.from_numpy(self.A.r.copy()).float() # checked.
        joints_RT[:3,3,:] *= smpl_scale
        sp_input['smpl_joints_RT'] = joints_RT.permute(2,0,1).reshape(1,-1,16)
        posed_vex = torch.from_numpy(posed_vex).float()
        
        posed_joints = torch.from_numpy(posed_joints).float() # checked
        world_vex = torch.matmul(posed_vex, R.T).squeeze(0).float() + Th # checked

        min_xyz = world_vex.min(dim=0)[0]
        max_xyz = world_vex.max(dim=0)[0]

        if self.args.big_box:
            min_xyz -= 0.050 * self.args.scale_size
            max_xyz += 0.050 * self.args.scale_size
        else:
            min_xyz[2] -= 0.050 * self.args.scale_size
            max_xyz[2] += 0.050 * self.args.scale_size
        if self.args.na_data:
            min_xyz -= 0.050 * self.args.scale_size
            max_xyz += 0.050 * self.args.scale_size    
        can_bounds = torch.stack([min_xyz,max_xyz], dim = 0)

        min_xyz = posed_vex.min(dim=0)[0]
        max_xyz = posed_vex.max(dim=0)[0]
        if self.args.big_box:
            min_xyz -= 0.050 * self.args.scale_size
            max_xyz += 0.050 * self.args.scale_size
        else:

            min_xyz -= 0.050 * self.args.scale_size
            max_xyz += 0.050 * self.args.scale_size

        bounds = torch.stack([min_xyz,max_xyz], dim = 0)
        
        sp_input['smpl_posed_blending_shape'] = posed_vex - posed_vertex_no_shape
        sp_input['world_bbox'] = can_bounds
        sp_input['smpl_bbox'] = bounds

        posed_triangles = F.embedding(self.face.squeeze(0).long(), posed_vex).float()
        can_triangles = F.embedding(self.face.squeeze(0).long(), can_vertices).float()
        can_triangles_no_shape = F.embedding(self.face.squeeze(0).long(), can_vertices_no_shape).float()

        sp_input['smpl_posed_triangles'] = posed_triangles
        sp_input['smpl_canonical_triangles'] = can_triangles
        sp_input['smpl_canonical_triangles_no_shape'] = can_triangles_no_shape
        
        sp_input['smpl_posed_vertex'] = posed_vex
        sp_input['smpl_world_vertex'] = world_vex
        sp_input['smpl_canonical_vertex_no_shape'] = can_vertices_no_shape
        
        sp_input['smpl_local_joints'] = posed_joints
        sp_input['smpl_local_joints_no_shape'] = posed_joints_no_shape
        sp_input['smpl_face'] = self.face
        sp_input['smpl_pose'] = torch.from_numpy(pose).float()
        sp_input['smpl_shape'] = torch.from_numpy(shape).float()
        sp_input['smpl_pose_rotmatrix'] = batch_rodrigues(sp_input['smpl_pose'].view(-1,3)).view(-1)
        sp_input['smpl_canonical_pose_rotmatrix'] = self.smpl_canonical_pose 


        ## data preparation for gnn 
        edges_idx = torch.from_numpy(np.load('tools/smpl_edges.npy',allow_pickle=True))
        
        max_edge_length = 0.1 * self.args.scale_size

        posed_edges_length = F.embedding(edges_idx.long(),posed_vertex_no_shape)/max_edge_length

        posed_edges_length_norm2 = (posed_edges_length[:,0,:] - posed_edges_length[:,1,:]).norm(p=2,dim=-1,keepdim = True)
        
        posed_edges_length_abs3 = (posed_edges_length[:,0,:] - posed_edges_length[:,1,:])
        
        posed_edge_feat = torch.cat([posed_edges_length_norm2,posed_edges_length_abs3],dim=-1) # [N,4]

        can_edges_length = F.embedding(edges_idx.long(),can_vertices_no_shape)/max_edge_length

        can_edges_length_norm2 = (can_edges_length[:,0,:] - can_edges_length[:,1,:]).norm(p=2,dim=-1,keepdim = True)
        
        can_edges_length_abs3 = (can_edges_length[:,0,:] - can_edges_length[:,1,:])
            
        can_edge_feat = torch.cat([can_edges_length_norm2,can_edges_length_abs3],dim=-1) # [N,4]
        graph_input = {}

        graph_input['posed_edge_feat'] = posed_edge_feat
        graph_input['can_edge_feat'] = can_edge_feat
            
        graph_input['n_node'] = torch.tensor(posed_vex.size(0),device='cpu')
        graph_input['n_edge'] = torch.tensor(posed_edge_feat.size(0),device='cpu')
        graph_input['global'] = None
        graph_input['edge_index'] = edges_idx.permute(1,0)


        sp_input['graph_input'] = graph_input
        data['sp_input'] = sp_input
        

        return data

    def get_one_item_image(self, data_view, view_id):

        '''
            get images
        '''
        view_num = len(data_view)
        data = {}
        all_image = []
        latent_codes_ext = []
        latent_codes_ixt = []
        latent_codes_focal = []
        latent_codes_principle = []
        latent_codes_mask = []

        loss_image_path = []
        loss_mask = []
        loss_ext = []
        loss_ixt = []
        loss_focal = []
        loss_principle = []

    
        # if self.args.single_person and self.args.is_train:
        #     # view_all = np.random.permutation(self.views)
        #     view_all = np.random.permutation(self.views)
        #     view_id_list = [view_all[0]]

        if self.args.interpolate_views:
            if self.args.fix_camera:
                view_id_list = [0]
            elif self.args.aist_data or self.args.na_data:
                view_id_list = [0,2,4,6]
            else:
                view_id_list = [0,6,12,18]
        else:
            view_id_list = [view_id]

        loss_view_sum = len(view_id_list)
        for view_idx in view_id_list:
            view_info = data_view[view_idx]
            img_path = view_info['path']
            ixt = view_info['intrinsics'].copy()
            ext = view_info['extrinsics'].copy()
            loss_image_path.append(img_path)


            if self.args.realworld_data:
                img = cv2.imread(img_path).astype(np.float32) / 255.

                msk_path = view_info['mask']
                msk = imageio.imread(msk_path, as_gray=True) / 255.
                msk = (msk == 1.).astype(np.float32)
                ## erode msk
                border = 5
                kernel = np.ones((border, border), np.uint8)
                
                msk_erode = cv2.erode(msk.copy(), kernel)
                msk = msk_erode
                    

                mask = msk.copy() 
                x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

                
                H,W,_ = img.shape
                center_x = x+w/2
                center_y = y+h/2
                
                x = int(center_x-512)
                y = int(center_y-512)
                w = 1024
                h = 1024
                if x<0:
                    x=0
                if y<0:
                    y=0
                if x+w>=W:
                    x = W-1025
                if y+h>=H:
                    y = H-1025

                img = img[y:y + h, x:x + w]
                msk = msk[y:y + h, x:x + w]
                img[msk==0] = 0

                # cv2.imwrite("test_na_crop.png",img*255)
                ixt_offset_x = x
                ixt_offset_y = y


                H, W = 512, 512
                scale_ratio = np.array([512/img.shape[1],512/img.shape[0]])
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)


            elif self.args.na_data:
                if self.args.na_crop_inline:
                    img = cv2.imread(img_path).astype(np.float32) / 255.

                    if self.args.na_with_mask:
                        msk_path = view_info['mask']
                        msk = imageio.imread(msk_path, as_gray=True) / 255.
                        msk = (msk == 1.).astype(np.float32)
                        ## erode msk
                        border = 5
                        kernel = np.ones((border, border), np.uint8)
                        
                        msk_erode = cv2.erode(msk.copy(), kernel)
                        msk = msk_erode
                        
                    else:
                        msk = (img.sum(axis=-1,) != 3.).astype(np.float32)

                    mask = msk.copy() 
                    # cv2.imwrite("test_na_before_crop.png",img*255)
                    # cv2.imwrite("test_na_msk.png",mask*255)
                    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

                    
                    H,W,_ = img.shape
                    center_x = x+w/2
                    center_y = y+h/2
                    
                    x = int(center_x-256)
                    y = int(center_y-256)
                    w = 512
                    h = 512
                    if x<0:
                        x=0
                    if y<0:
                        y=0
                    if x+w>=W:
                        x = W-513
                    if y+h>=H:
                        y = H-513
                
                    # full_img = np.zeros_like(img)
                    # full_img[y:y + h, x:x + w] = 1
                    # cv2.imwrite("full_img.png",full_img*255)
                    # import pdb
                    # pdb.set_trace()
                    img = img[y:y + h, x:x + w]
                    msk = msk[y:y + h, x:x + w]
                    img[msk==0] = 0

                    # cv2.imwrite("test_na_crop.png",img*255)
                    ixt_offset_x = x
                    ixt_offset_y = y
                    scale_ratio = np.array([1,1])
                    H, W = 512, 512
                else:
                    img = cv2.imread(img_path).astype(np.float32) / 255.
                    msk = (img.sum(axis=-1) != 3.).astype(np.float32)
                    H, W = 512, 512

                    if self.cfg['dataset.mask_bkgd']:
                        img[msk == 0] = 0
                        if self.cfg['dataset.white_bkgd']:
                            img[msk == 0] = 1

                    scale_ratio = np.array([512/img.shape[1],512/img.shape[0]])
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
         
                msk_path = view_info['mask']
                undistort_img_path = img_path.replace('rgb_bg','undistort_img_beta')
                undistort_msk_path = msk_path.replace('mask','undistort_msk_beta')
                if self.args.camera_undistort and os.path.exists(undistort_img_path) and os.path.exists(undistort_msk_path):
                    try:
                        img = cv2.imread(undistort_img_path).astype(np.float32) / 255.
                        msk = imageio.imread(undistort_msk_path, as_gray=True)
                    except Exception as e:
                        img = cv2.imread(img_path).astype(np.float32) / 255.
                        msk_cihp = imageio.imread(msk_path, as_gray=True)
                        msk = (msk_cihp != 0).astype(np.uint8)
                        if self.args.nb_another_mask:
                            # msk_start = time.time()
                            nb_msk_path = view_info['nb_another_mask']
                            nb_msk_cihp = imageio.imread(nb_msk_path, as_gray=True)
                            nb_msk_cihp = (nb_msk_cihp != 0).astype(np.uint8)
                            # imageio.imwrite('nb_another_mask.png',nb_msk_cihp*255)
                            # imageio.imwrite('nb_mask.png',msk*255)
                            msk = (msk | nb_msk_cihp).astype(np.uint8)
                            # msk_end = time.time()
                            # print("msk_time%.3f"%(msk_end-msk_start))
                            # imageio.imwrite('nb_final_mask.png',msk*255)
                            # import pdb
                            # pdb.set_trace()
                            
                        border = 5
                        kernel = np.ones((border, border), np.uint8)
                        
                        msk_erode = cv2.erode(msk.copy(), kernel)
                        msk_dilate = cv2.dilate(msk.copy(), kernel)
                        msk[(msk_dilate - msk_erode) == 1] = 100
        

                        if self.args.camera_undistort:
                            # distort_start = time.time()
                            undistort = view_info['undistort']
                            msk = cv2.undistort(msk, ixt[:3,:3], undistort)
                            img = cv2.undistort(img, ixt[:3,:3], undistort)
                            if self.args.save_nb_data:
                                os.makedirs(undistort_img_path[:undistort_img_path.rfind('/')],exist_ok=True)
                                os.makedirs(undistort_msk_path[:undistort_msk_path.rfind('/')],exist_ok=True)
                                cv2.imwrite(undistort_img_path,img*255.)
                                imageio.imwrite(undistort_msk_path,msk)
                                if self.args.presave_dist_data:
                                    return
                else:
                    img = cv2.imread(img_path).astype(np.float32) / 255.
                    msk_cihp = imageio.imread(msk_path, as_gray=True)
                    msk = (msk_cihp != 0).astype(np.uint8)
                    if self.args.nb_another_mask:
                        nb_msk_path = view_info['nb_another_mask']
                        nb_msk_cihp = imageio.imread(nb_msk_path, as_gray=True)
                        nb_msk_cihp = (nb_msk_cihp != 0).astype(np.uint8)
                        # imageio.imwrite('nb_another_mask.png',nb_msk_cihp*255)
                        # imageio.imwrite('nb_mask.png',msk*255)
                        msk = (msk | nb_msk_cihp).astype(np.uint8)
       


                    border = 5
                    kernel = np.ones((border, border), np.uint8)
                    
                    msk_erode = cv2.erode(msk.copy(), kernel)
                    msk_dilate = cv2.dilate(msk.copy(), kernel)
                    msk[(msk_dilate - msk_erode) == 1] = 100
    

                    if self.args.camera_undistort:
                        # distort_start = time.time()
                        undistort = view_info['undistort']
                        msk = cv2.undistort(msk, ixt[:3,:3], undistort)
                        img = cv2.undistort(img, ixt[:3,:3], undistort)
                        if self.args.save_nb_data:
                            os.makedirs(undistort_img_path[:undistort_img_path.rfind('/')],exist_ok=True)
                            os.makedirs(undistort_msk_path[:undistort_msk_path.rfind('/')],exist_ok=True)
                            cv2.imwrite(undistort_img_path,img*255.)
                            imageio.imwrite(undistort_msk_path,msk)
                            if self.args.presave_dist_data:
                                return

            
                # distort_end = time.time()
                # print("distort_time%.3f"%(distort_end-distort_start))

                if self.args.aist_data:
                    H, W = 512, 512
                    scale_ratio = W / img.shape[0]
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    H, W = int(img.shape[0] * self.cfg['dataset.ratio']), int(img.shape[1] * self.cfg['dataset.ratio'])
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

                if self.cfg['dataset.mask_bkgd']:
                    img[msk == 0] = 0
                    if self.cfg['dataset.white_bkgd']:
                        img[msk == 0] = 1

                # img = img.transpose(2,0,1)
            
            img = img.transpose(2,0,1)  
            if self.args.na_data and self.args.na_crop_inline:
                ixt[0][2] -= ixt_offset_x
                ixt[1][2] -= ixt_offset_y
                if self.args.realworld_data:
                    ixt[:2,:] = ixt[:2,:] * self.cfg['dataset.ratio']
            elif self.args.aist_data or self.args.na_data:
                ixt[:2,:] = ixt[:2,:] * scale_ratio[:,None]
            else:
                ixt[:2,:] = ixt[:2,:] * self.cfg['dataset.ratio']

            ext[:3,3] *= self.args.scale_size
            if self.args.aist_data:
                ext[:3,3] /= 100.

            all_image.append(img)
            loss_mask.append(msk)
            loss_ext.append(ext)
            loss_ixt.append(ixt)
            loss_focal.append([ixt[0,0],ixt[1,1]])
            loss_principle.append([ixt[0,2],ixt[1,2]]) 

        if not self.train:
   
            if self.args.aist_data:
                latent_codes_view_list = [1,4,6]
            elif self.args.na_data:
                latent_codes_view_list = data_utils.parse_views(self.args.latent_codes_views)
            else:
                if self.args.app_editing:
                    latent_codes_view_list = [0,6,12,18]
                else:
                    latent_codes_view_list = [0,7,15]
            if self.args.latent_codes_views != None:
                latent_codes_view_list = data_utils.parse_views(self.args.latent_codes_views)
        else:

            all_latent_codes = list(range(view_num))
            all_latent_codes.remove(view_id)
            latent_codes_view_list = np.random.choice(all_latent_codes,3,replace=False)

        for id in latent_codes_view_list:
            view_info = data_view[id]
            img_path = view_info['path']
            # all_image_path.append(img_path)

            ixt = view_info['intrinsics'].copy()
            ext = view_info['extrinsics'].copy()

            if self.args.realworld_data:
                img = cv2.imread(img_path).astype(np.float32) / 255.

                msk_path = view_info['mask']
                msk = imageio.imread(msk_path, as_gray=True) / 255.
                msk = (msk == 1.).astype(np.float32)
                ## erode msk
                border = 5
                kernel = np.ones((border, border), np.uint8)
                
                msk_erode = cv2.erode(msk.copy(), kernel)
                msk = msk_erode
                    

                mask = msk.copy() 
                # cv2.imwrite("test_na_before_crop.png",img*255)
                # cv2.imwrite("test_na_msk.png",mask*255)
                x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

                
                # this_image[mask==1] = 0
                # mask = ((mask*(-1))+1).astype(np.bool)
                # x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                # print(x,y,w,h)
                H,W,_ = img.shape
                center_x = x+w/2
                center_y = y+h/2
                
                x = int(center_x-512)
                y = int(center_y-512)
                w = 1024
                h = 1024
                if x<0:
                    x=0
                if y<0:
                    y=0
                if x+w>=W:
                    x = W-1025
                if y+h>=H:
                    y = H-1025
            
                # full_img = np.zeros_like(img)
                # full_img[y:y + h, x:x + w] = 1
                # cv2.imwrite("full_img.png",full_img*255)
                # import pdb
                # pdb.set_trace()
                img = img[y:y + h, x:x + w]
                msk = msk[y:y + h, x:x + w]
                img[msk==0] = 0

                # cv2.imwrite("test_na_crop.png",img*255)
                ixt_offset_x = x
                ixt_offset_y = y


                H, W = 512, 512
                scale_ratio = np.array([512/img.shape[1],512/img.shape[0]])
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

            elif self.args.na_data:
                if self.args.na_crop_inline:
                    img = cv2.imread(img_path).astype(np.float32) / 255.

                    if self.args.na_with_mask:
                        msk_path = view_info['mask']
                        msk = imageio.imread(msk_path, as_gray=True) /255.
                        msk = (msk == 1.).astype(np.float32)

                        ## erode msk
                        border = 5
                        kernel = np.ones((border, border), np.uint8)
                        
                        msk_erode = cv2.erode(msk.copy(), kernel)
                        msk = msk_erode

                        # debug_img = img.copy()
                        # debug_img[msk==0] = 0
                        # # cv2.imwrite("debug_img.png", debug_img)
                        # ori_msk = imageio.imread(msk_path, as_gray=True)
                        # cv2.imwrite("debug_img.png", debug_img*255+ori_msk[:,:,None]/2)
                        # border = 5
                        # kernel = np.ones((border, border), np.uint8)
                        
                        # msk_erode = cv2.erode(msk.copy(), kernel)
                        # msk_dilate = cv2.dilate(msk.copy(), kernel)
                        # debug_img_erode = img.copy()
                        # debug_img_erode[msk_erode==0] = 0
                        # # cv2.imwrite("debug_img_erode.png", debug_img*255+msk_erode[:,:,None]*255/2)
                        # cv2.imwrite("debug_img_erode.png", debug_img_erode * 255)
                        # import pdb
                        # pdb.set_trace()
                    else:
                        msk = (img.sum(axis=-1,) != 3.).astype(np.float32)

                    mask = msk.copy()
                    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

                    
                    # this_image[mask==1] = 0
                    # mask = ((mask*(-1))+1).astype(np.bool)
                    # x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                    # print(x,y,w,h)
                    H,W,_ = img.shape
                    center_x = x+w/2
                    center_y = y+h/2
                    
                    x = int(center_x-256)
                    y = int(center_y-256)
                    w = 512
                    h = 512
                    if x<0:
                        x=0
                    if y<0:
                        y=0
                    if x+w>=W:
                        x = W-513
                    if y+h>=H:
                        y = H-513
                
                    # full_img = np.zeros_like(img)
                    # full_img[y:y + h, x:x + w] = 1
                    # cv2.imwrite("full_img.png",full_img*255)
                    # import pdb
                    # pdb.set_trace()
                    img = img[y:y + h, x:x + w]
                    msk = msk[y:y + h, x:x + w]
                    img[msk==0] = 0

                    ixt_offset_x = x
                    ixt_offset_y = y
                    scale_ratio = np.array([1,1])
                    H, W = 512, 512
                else:
                    img = cv2.imread(img_path).astype(np.float32) / 255.
                    msk = (img.sum(axis=-1) != 3.).astype(np.float32)

                    if self.cfg['dataset.mask_bkgd']:
                        img[msk == 0] = 0
                        if self.cfg['dataset.white_bkgd']:
                            img[msk == 0] = 1
                            
                    H, W = 512, 512
                    # W, H = img.shape[:-1]
                    # scale_ratio = np.array([W, H]) / img.shape[:-1] 
                    scale_ratio = np.array([512/img.shape[1],512/img.shape[0]])
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

            else:
                # print("not checked")
                # exit(1)
                msk_path = view_info['mask']
                undistort_img_path = img_path.replace('rgb_bg','undistort_img_beta')
                undistort_msk_path = msk_path.replace('mask','undistort_msk_beta')
                if self.args.camera_undistort and os.path.exists(undistort_img_path) and os.path.exists(undistort_msk_path):
                    try:
                        img = cv2.imread(undistort_img_path).astype(np.float32) / 255.
                        msk = imageio.imread(undistort_msk_path, as_gray=True)
                    except Exception as e:
                        img = cv2.imread(img_path).astype(np.float32) / 255.
                        msk_cihp = imageio.imread(msk_path, as_gray=True)
                        msk = (msk_cihp != 0).astype(np.uint8)
                        if self.args.nb_another_mask:
                            # msk_start = time.time()
                            nb_msk_path = view_info['nb_another_mask']
                            nb_msk_cihp = imageio.imread(nb_msk_path, as_gray=True)
                            nb_msk_cihp = (nb_msk_cihp != 0).astype(np.uint8)
                            # imageio.imwrite('nb_another_mask.png',nb_msk_cihp*255)
                            # imageio.imwrite('nb_mask.png',msk*255)
                            msk = (msk | nb_msk_cihp).astype(np.uint8)
                            # msk_end = time.time()
                            # print("msk_time%.3f"%(msk_end-msk_start))
                            # imageio.imwrite('nb_final_mask.png',msk*255)
                            # import pdb
                            # pdb.set_trace()
                            
                        border = 5
                        kernel = np.ones((border, border), np.uint8)
                        
                        msk_erode = cv2.erode(msk.copy(), kernel)
                        msk_dilate = cv2.dilate(msk.copy(), kernel)
                        msk[(msk_dilate - msk_erode) == 1] = 100
        

                        if self.args.camera_undistort:
                            # distort_start = time.time()
                            undistort = view_info['undistort']
                            msk = cv2.undistort(msk, ixt[:3,:3], undistort)
                            img = cv2.undistort(img, ixt[:3,:3], undistort)
                            if self.args.save_nb_data:
                                os.makedirs(undistort_img_path[:undistort_img_path.rfind('/')],exist_ok=True)
                                os.makedirs(undistort_msk_path[:undistort_msk_path.rfind('/')],exist_ok=True)
                                cv2.imwrite(undistort_img_path,img*255.)
                                imageio.imwrite(undistort_msk_path,msk)
                                if self.args.presave_dist_data:
                                    return
                else:
                    img = cv2.imread(img_path).astype(np.float32) / 255.
                    msk_cihp = imageio.imread(msk_path, as_gray=True)
                    msk = (msk_cihp != 0).astype(np.uint8)
                    if self.args.nb_another_mask:
                        # msk_start = time.time()
                        nb_msk_path = view_info['nb_another_mask']
                        nb_msk_cihp = imageio.imread(nb_msk_path, as_gray=True)
                        nb_msk_cihp = (nb_msk_cihp != 0).astype(np.uint8)
                        # imageio.imwrite('nb_another_mask.png',nb_msk_cihp*255)
                        # imageio.imwrite('nb_mask.png',msk*255)
                        msk = (msk | nb_msk_cihp).astype(np.uint8)
                        # msk_end = time.time()
                        # print("msk_time%.3f"%(msk_end-msk_start))
                        # imageio.imwrite('nb_final_mask.png',msk*255)
                        # import pdb
                        # pdb.set_trace()

                    border = 5
                    kernel = np.ones((border, border), np.uint8)
                    
                    msk_erode = cv2.erode(msk.copy(), kernel)
                    msk_dilate = cv2.dilate(msk.copy(), kernel)
                    msk[(msk_dilate - msk_erode) == 1] = 100
    

                    if self.args.camera_undistort:
                        # distort_start = time.time()
                        undistort = view_info['undistort']
                        msk = cv2.undistort(msk, ixt[:3,:3], undistort)
                        img = cv2.undistort(img, ixt[:3,:3], undistort)
                        if self.args.save_nb_data:
                            os.makedirs(undistort_img_path[:undistort_img_path.rfind('/')],exist_ok=True)
                            os.makedirs(undistort_msk_path[:undistort_msk_path.rfind('/')],exist_ok=True)
                            cv2.imwrite(undistort_img_path,img*255.)
                            imageio.imwrite(undistort_msk_path,msk)
                            if self.args.presave_dist_data:
                                return

                if self.args.aist_data:
                    H, W = 512 , 512
                    scale_ratio = W / img.shape[0]
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    H, W = int(img.shape[0] * self.cfg['dataset.ratio']), int(img.shape[1] * self.cfg['dataset.ratio'])
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

                if self.cfg['dataset.mask_bkgd']:
                    img[msk == 0] = 0
                    if self.cfg['dataset.white_bkgd']:
                        img[msk == 0] = 1

                # img = img.transpose(2,0,1)
            
            img = img.transpose(2,0,1) 
            if self.args.na_data and self.args.na_crop_inline:
                ixt[0][2] -= ixt_offset_x
                ixt[1][2] -= ixt_offset_y
                if self.args.realworld_data:
                    ixt[:2,:] = ixt[:2,:] * self.cfg['dataset.ratio']
            elif self.args.aist_data or self.args.na_data:
                ixt[:2,:] = ixt[:2,:] * scale_ratio[:,None]
            else:
                ixt[:2,:] = ixt[:2,:] * self.cfg['dataset.ratio']

        
    

            ext[:3,3] *= self.args.scale_size
            if self.args.aist_data:
                ext[:3,3] /= 100.
            all_image.append(img)
            latent_codes_mask.append(msk)
            latent_codes_ext.append(ext)
            latent_codes_ixt.append(ixt)
            latent_codes_focal.append([ixt[0,0],ixt[1,1]])
            latent_codes_principle.append([ixt[0,2],ixt[1,2]])
        
        all_image = torch.from_numpy(np.array(all_image)) #[views,3,H,W]
        
        if self.train:
            all_image = self.image_augmentation(all_image)
            
            all_mask = torch.from_numpy(np.concatenate([loss_mask,latent_codes_mask],axis=0))
            for i, msk in enumerate(all_mask):
                if self.cfg['dataset.mask_bkgd']:
                    all_image[i].permute(1,2,0)[msk == 0] = 0
                    if self.cfg['dataset.white_bkgd']:
                        all_image[i].permute(1,2,0)[msk == 0] = 1
                    
        # if self.args.interpolate_views:
        #     loss_image = all_image[0:4,...].unsqueeze(0) #[frames]
        #     latent_codes_image = all_image[4:,...].unsqueeze(0)
        # else:
        #     loss_image = all_image[0:1,...].unsqueeze(0) #[frames]
        #     latent_codes_image = all_image[1:,...].unsqueeze(0)

        loss_image = all_image[0:loss_view_sum,...].unsqueeze(0) #[frames]
        latent_codes_image = all_image[loss_view_sum:,...].unsqueeze(0)

        loss_mask = torch.from_numpy(np.array(loss_mask))
        loss_ext = torch.from_numpy(np.array(loss_ext))
        loss_ixt = torch.from_numpy(np.array(loss_ixt))
        loss_focal = torch.from_numpy(np.array(loss_focal))
        loss_principle = torch.from_numpy(np.array(loss_principle))
        
        latent_codes_ext = torch.from_numpy(np.array(latent_codes_ext))
        latent_codes_mask = torch.from_numpy(np.array(latent_codes_mask))
        latent_codes_ixt = torch.from_numpy(np.array(latent_codes_ixt))
        latent_codes_focal = torch.from_numpy(np.array(latent_codes_focal))
        latent_codes_principle = torch.from_numpy(np.array(latent_codes_principle))
        # all_image = torch.from_numpy(all_image) #[views,3,H,W]

        data['view_num'] = view_num
        data['image_size'] = [H,W]
        data['loss_image'] = loss_image
        data['loss_mask'] = loss_mask
        data['loss_image_path'] = loss_image_path
        data['loss_ext'] = loss_ext
        data['loss_ixt'] = loss_ixt
        data['loss_focal'] = loss_focal
        data['loss_principle'] = loss_principle

        data['latent_codes_image'] = latent_codes_image
        data['latent_codes_mask'] = latent_codes_mask
        data['latent_codes_ext'] = latent_codes_ext
        data['latent_codes_ixt'] = latent_codes_ixt
        data['latent_codes_focal'] = latent_codes_focal
        data['latent_codes_principle'] = latent_codes_principle

        return data
        

    def __getitem__(self, index):

        shape_id = self.index_to_shape_id[index]
        view_id = self.index_to_view_id[index]
        view_num = self.shape_id_to_view_num[shape_id]
        if self.train: ## color jitter for data augmentation
            prob = np.random.randint(9000000)
            torch.manual_seed(prob)
            self.image_augmentation = extreme_hue_color_jitter(min(self.iteration//self.args.pg_jitter_stage*0.05 + 0.05,0.5)) ## todo:: change name
        
                    
        data_shape = self._load_shape(self.shape_data[shape_id])
        data_view = [self._load_view(self.view_data[shape_id], i) for i in range(view_num)]

        cpu_data_image = self.get_one_item_image(data_view,view_id)

        cpu_data_others = self.get_one_item_shape(data_shape)

        cpu_data = {**cpu_data_others, **cpu_data_image}
        cpu_data['shape_id'] = shape_id
        cpu_data['view_id'] = view_id
        return cpu_data

    
    def __len__(self):
        return self.all_num






