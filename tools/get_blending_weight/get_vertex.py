
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
# m, A, dd = load_model( '/home/wangyiming/EasyMocap/data/smplx/smpl/SMPL_NEUTRAL.pkl')
# m, A, dd = load_model( '/home/wangyiming/EasyMocap/data/smplx/smpl/SMPL_MALE.pkl')
m, A, dd = load_model( '/home/sig/SIG2022/tools/get_blending_weight/SMPL_FEMALE.pkl')
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
dir_base = '/home/sig/SIG2022/data/zju-mocap-1200'
# dir_base = '/home/wangyiming/SIG2022/data/neus_zju'
# dir_base = '/home/wangyiming/SIG2022/data/geometry_test'
character = range(0,10)
total_frame = 1200
rodrigues = lambda x : cv2.Rodrigues(x)[0]

motions = []
for id in tqdm.tqdm(character,desc='character'):
    dir_id =  join(dir_base,str(id))        
    
    # smpl_dir = "/home/wangyiming/dataset/neuralbody/CoreView_"+str(id)+"/new_params"
    # smpl_dir = "/home/wangyiming/dataset/neuralbody/CoreView_"+str(id)+"/new_params"/
    smpl_dir = '/home/sig/SIG2022/data/zju-mocap-1200/'+str(id)+'/smpl_transform'
    # smpl_dir = "/home/wangyiming/SIG2022/data/geometry_test/0/smpl_params"
    # if id == 9:
        # smpl_dir = "/home/wangyiming/dataset/neuralbody/CoreView_"+str(id)+"/params"
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

        vertex_dir = join(dir_id,'smpl_vertex')
        os.makedirs(vertex_dir,exist_ok=True)
        smpl_path = os.path.join(smpl_dir, str(frame).rjust(6,'0')+'.json')
        # smpl_path = os.path.join(smpl_dir, str(frame).rjust(6,'0')+'.npy')
        # if id == 9:
        #     smpl_path = os.path.join(smpl_dir, str(frame+810)+'.npy')
        try:
            params =json.load(open(smpl_path))
        except Exception as e:
            print(e)
            continue
        m.pose[:] = np.array(params['poses'])
        Rh = np.array(params['Rh'])
        Th = np.array(params['Th'])
        rot = rodrigues(Rh[0])
        rot_t = rot.transpose(1,0)
        transl = Th
        vertices = np.matmul(m.r, rot_t) + transl
        np.save(join(vertex_dir,f'{frame:06}.npy'), vertices)
        # joints = np.matmul(m.J_transformed.r, rot_t) + transl