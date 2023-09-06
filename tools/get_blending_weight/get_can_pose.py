import json
import numpy as np

from plyfile import PlyData, PlyElement
def write_ply(points, filename, text=False):
    """
    input: Nx3, write points to filename as PLY format.
    """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)
        
def rotationPose3d(pose_3d, R):
    r_pose_3d = np.zeros(pose_3d.shape)
    
    # If R is rotation A->B, then A*R = B, * is dot.
    for i in range(14):
        r_pose_3d[i] = np.dot(R, pose_3d[i])
        
    return r_pose_3d

def calAxisXYZ(a, b, need_R=False):
    rotationAxis = np.cross(a, b)
    rotationAngle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    norm = np.linalg.norm(rotationAxis)
    rotationMatrix = np.zeros([3, 3])
    if norm == 0:
        wx, wy, wz =0., 0., 0.
    else:
        norm_rotationAxis = [i / norm for i in rotationAxis]
        wx, wy, wz = norm_rotationAxis
    sin, cos = np.sin(rotationAngle), np.cos(rotationAngle)
    
    rotationMatrix[0][0] = cos + (wx ** 2) * (1 - cos)
    rotationMatrix[0][1] = wx*wy*(1 - cos) - wz*sin
    rotationMatrix[0][2] = wy*sin + wx*wz*(1 - cos)
    
    rotationMatrix[1][0] = wz*sin + wx*wy*(1 - cos)
    rotationMatrix[1][1] = cos + (wy ** 2) * (1 - cos)
    rotationMatrix[1][2] = wy*wz*(1 - cos) - wx*sin
    
    rotationMatrix[2][0] = wx*wz*(1 - cos) - wy*sin
    rotationMatrix[2][1] = wx*sin + wy*wz*(1 - cos)
    rotationMatrix[2][2] = cos + (wz ** 2) * (1 - cos)
    
    ax = np.arctan2(rotationMatrix[2][1], rotationMatrix[2][2])
    ay = np.arctan2(-rotationMatrix[2][0], np.sqrt(rotationMatrix[2][1] ** 2 + rotationMatrix[2][2] ** 2))
    az = np.arctan2(rotationMatrix[1][0], rotationMatrix[0][0])
    
    if not need_R:
        return ax, ay, az
    else:
        return ax, ay, az, rotationMatrix
    
def pose2smpl(rot_all, kin_tree):
    
    smpl_pose = np.zeros([24, 3])
    
    def ab2smpl(a, b):
        ax, ay, az = calAxisXYZ(a, b)
        smpl = -ax, -ay, az
        return smpl
    smpl_pose[0] = [0,0,0]
    for i in range(1,24):
        # smpl_pose[i] = ab2smpl(pose_3d[kin_tree[i,0]],pose_3d[kin_tree[i,1]])
        rot_1 = rot_all[kin_tree[i,1]]
        rot_0 = rot_all[kin_tree[i,0]]
        rot = rot_1 @ rot_0.T
        vec = cv2.Rodrigues(rot)[0]
        smpl_pose[i] = vec[:,0]
    # Right hip
    
    return smpl_pose

from smpl_webuser.serialization import load_model

m, A, dd = load_model( '/home/wangyiming/EasyMocap/data/smplx/smpl/SMPL_NEUTRAL.pkl')
kintree = dd['kintree_table']
kintree = kintree.transpose(1,0)
# m.pose[:] = np.array(pose.reshape(1,72))

RT = np.array(json.load(open('/home/wangyiming/SIG2022/tools/transform_tpose_new.json'))['joints_RT'])
abs_rot = RT.transpose(2,0,1)[:,:3,:3]
# asb_Rh = np.zeros((24,3,1),dtype=np.float32)
import cv2
# for i in range(abs_rot.shape[0]):
    # asb_Rh[i] = cv2.Rodrigues(abs_rot[i])[0]
# pose = pose2smpl(asb_Rh[:,:,0])
pose_3d = np.array(json.load(open('/home/wangyiming/SIG2022/tools/transform_tpose_new.json'))['joints'])


# a = trans_pose_3d[8]
# b = [1, 0, 0]
# _, _, _, R = calAxisXYZ(a, b, need_R=True)
# r_pose_3d = rotationPose3d(trans_pose_3d, R)

# pose = pose2smpl(pose_3d,kintree)
pose = pose2smpl(abs_rot,kintree)
m.pose[:] = pose.reshape(1,72)
np.save('/home/wangyiming/SIG2022/tools/apose_pose.npy',pose, allow_pickle=True)
new_RT = A.r
vertices = m.r
write_ply(vertices,'../../debug/1.10/jinlile.ply')
import pdb
pdb.set_trace()

