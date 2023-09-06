'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]


About the Script:
=================
This script demonstrates a few basic functions to help users get started with using 
the SMPL model. The code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Save the resulting body as a mesh in .OBJ format


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python hello_smpl.py

'''

from smpl_webuser.serialization import load_model
import numpy as np
import cv2
import glob
import tqdm
import json
import torch
import os
from os.path import join
import sys
# sys.path.append('..')
# sys.path.append('./smpl_webuser')
np.random.seed(2)
## Load SMPL model (here we load the female model)
## Make sure path is correct

# import pdb;pdb.set_trace()
## Write to an .obj file
def save_obj(m, verts, fname):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        if m is not None:
            for f in m.f+1: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

    ## Print message
    print('..Output mesh saved to: ', fname)

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
        
def recover(verts, weights, A):
    # A_inv = np.linalg.inv(A.transpose((2,0,1))).transpose((1,2,0))  # N x 4 x 4
    # T_inv = A_inv.dot(weights.T)
    T = A.dot(weights.T)
    T_inv = np.linalg.inv(T.transpose((2,0,1))).transpose((1,2,0))  # N x 4 x 4
    shape_h = np.vstack((verts.T, np.ones((1, verts.shape[0]))))  # 4 x N

    shape_r = (
        T_inv[:,0,:] * shape_h[0, :].reshape((1, -1)) + 
        T_inv[:,1,:] * shape_h[1, :].reshape((1, -1)) + 
        T_inv[:,2,:] * shape_h[2, :].reshape((1, -1)) + 
        T_inv[:,3,:] * shape_h[3, :].reshape((1, -1))).T
    shape_r = shape_r[:,:3]
    return shape_r


# m, A, dd = load_model( '/home/wangyiming/na/tools/EasyMocap/data/smplx/smpl/SMPL_NEUTRAL.pkl')
m, A, dd = load_model( '/home/wangyiming/EasyMocap/data/smplx/smpl/SMPL_NEUTRAL.pkl')
# m, A, dd = load_model( '/home/wangyiming/EasyMocap/data/smplx/smpl/SMPL_MALE.pkl')
# m, A, dd = load_model( '/home/wangyiming/SIG2022/tools/get_blending_weight/SMPL_FEMALE.pkl')
# obj_fname='/HPS/HumanBodyRetargeting8/work/Code/smpl/new_canonical_pose/canonical.obj'
# save_obj(m, m.r, obj_fname)
print(A[:,:,10])

#======================================================================

# Assign random pose and shape parameters

# Transform files export
source_p = ''
target_p = ''

# dir_base = '/home/wangyiming/na/data/10c300f'
# dir_base = '/home/wangyiming/SIG2022/data/zju-mocap-full'
dir_base = '/home/wangyiming/SIG2022/data/zju-mocap-1200'
# dir_base = '/home/wangyiming/SIG2022/data/neus_zju'
# dir_base = '/home/wangyiming/SIG2022/data/geometry_test'
character = range(0,10)
total_frame = 1200
rodrigues = lambda x : cv2.Rodrigues(x)[0]

motions = []
for id in tqdm.tqdm(character,desc='character'):
    dir_id =  join(dir_base,str(id))        
    
    # smpl_dir = "/home/wangyiming/dataset/neuralbody/CoreView_"+str(id)+"/new_params"
    smpl_dir = "/home/wangyiming/dataset/neuralbody/CoreView_"+str(id)+"/new_params"
    # smpl_dir = "/home/wangyiming/SIG2022/data/geometry_test/0/smpl_params"
    if id == 9:
        smpl_dir = "/home/wangyiming/dataset/neuralbody/CoreView_"+str(id)+"/params"
    max_frame_len = len(sorted(os.listdir((smpl_dir))))
    for frame in tqdm.tqdm(range(min(total_frame,max_frame_len)),desc='frames'):
        #smpl_dir = f'/home/wangyiming/dataset/neuralbody/CoreView_{id+1}/new_params'
        if id == 9:
            frame += 810
        save_dir = join(dir_id,'smpl_transform/')
        os.makedirs(save_dir,exist_ok=True)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir += str(frame).rjust(6,'0')+'.json'   
        new_params = False
        if 'new' in os.path.basename(smpl_dir):
            new_params = True

        vertex_dir = join(dir_id,'smpl_vertex')
        os.makedirs(vertex_dir,exist_ok=True)
        smpl_path = os.path.join(smpl_dir, str(frame+1)+'.npy')
        # smpl_path = os.path.join(smpl_dir, str(frame).rjust(6,'0')+'.npy')
        # if id == 9:
        #     smpl_path = os.path.join(smpl_dir, str(frame+810)+'.npy')
        try:
            params = np.load(smpl_path, allow_pickle=True).item()
        except Exception as e:
            print(e)
            continue
        m.pose[:] = np.array(params['poses'])
        m.betas[:] = np.array(params['shapes'])
        Rh = np.array(params['Rh'])
        Th = np.array(params['Th'])
        rot = rodrigues(Rh[0])
        rot_t = rot.transpose(1,0)
        transl = Th
        vertices = np.matmul(m.r, rot_t) + transl
        # vex = np.load('/home/wangyiming/dataset/neuralbody/CoreView_0/new_vertices/1.npy',allow_pickle=True)
        # write_ply(vex,'../../debug/1.10/vex.ply')
        # write_ply(vertices,'../../debug/1.10/neutral.ply')
        # exit(1)
        
        np.save(join(vertex_dir,f'{frame:06}.npy'), vertices)
        # save_obj(m, vertices, os.path.join(output_mesh_folder + '/{}'.format(fname.split('/')[-1]).replace('json', 'obj')))
        joints = np.matmul(m.J_transformed.r, rot_t) + transl
        

        #if i == 0:
        #    motions.append(joints.tolist())
        #    motions.append(joints.tolist())
        motions.append(joints.tolist())

        transform = {
                    # 'translation': transl.tolist(), 
                    # 'rotation': rot_t.tolist(), 
                    'Rh': Rh.tolist(),
                    'Th': Th.tolist(),
                    'joints': joints.tolist(),
                    'joints_RT': A.r.tolist(), 
                    "poses": params['poses'].tolist(),
                    "shapes": params['shapes'].tolist()
                    }
        transform['motion'] = motions[-3:]
        
        with open(save_dir, 'w') as fw:
            json.dump(transform, fw)


        # ppts = torch.tensor(m.r)
        # ppts = np.load("/home/wangyiming/SIG2022/data/zju-mocap-full/0/smpl_vertex/000000.npy",allow_pickle=True)
        # ppts = (ppts - transl) @ rot_t.T
        # ppts = torch.tensor(ppts).double()
        # vex_weight = np.loadtxt( '/home/wangyiming/SIG2022/tools/skinning_weight.txt')
        # joints_RT = torch.tensor(A.r.transpose(2,0,1).reshape(1,-1,16))
        # # face_weights = F.embedding(faces.long(), face_weight)  # num_face x 3 x 24
        # # weights = (face_weights[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1) # num_samples x 24
        # weights = torch.tensor(vex_weight)
        # weighted_RT = torch.tensor((weights @ joints_RT).reshape(-1, 4, 4))
        # ppts_normalized = torch.cat([ppts, ppts.new_ones(ppts.size(0), 1)], -1)  # N x 4
        # ppts_canonical = torch.einsum("ncd,nd->nc", torch.inverse(weighted_RT), ppts_normalized)
        
        # write_ply(ppts_canonical.cpu().numpy(),'../../debug/blending_weight/vex_canonical.ply')
        # # # Separate legs
        # t_pose_joint_RT = torch.tensor(json.load(open('/home/wangyiming/SIG2022/tools/transform_tpose.json'))['joints_RT']).permute(2,0,1).reshape(1,-1,16).double()
        # new_weighted_RT = (weights @ t_pose_joint_RT).reshape(-1, 4, 4)   # N x 4 x 4
        # ppts_canonical = torch.einsum("ncd,nd->nc", new_weighted_RT, ppts_canonical)
        # ppts_canonical = ppts_canonical[:,:3]
        
        # write_ply(ppts_canonical.cpu().numpy(),'../../debug/blending_weight/vex_canonical_seperate.ply')
        # write_ply(ppts.cpu().numpy(),'../../debug/blending_weight/vex_original.ply')
        

        # exit(1)