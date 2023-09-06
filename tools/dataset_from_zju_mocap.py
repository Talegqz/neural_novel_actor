from curses.panel import new_panel
import pickle
import json
import os
from os.path import join
import numpy as np
import cv2
import re
import tqdm
from multiprocessing import Pool

total_num = 10
# data_from = '/home/gqz/dataset/zju_data_source/'
# data_to = '/home/gqz/SIG2022/data/neuralbody'

data_from = 'zju-moca[/' # zju-mocap
data_to = 'data/neuralbody'
total_frame = 2000
per_thread = 4
new_params = True


def get_ext_ixt(num):
    input_base = f'{data_from}/CoreView_{num}/annots.json'
    if os.path.exists(input_base):
        cams = json.load(open(input_base, 'r'))
    else:
        input_base = f'{data_from}/CoreView_{num}//annots.npy'
        cams = np.load(input_base,allow_pickle=True).item()

    if num == 0 or num==1:
        cams = cams['cams']['20190823']
        cam_num = 21
    else:
        cam_num = 23
        cams = cams['cams']

    output_base = f'{data_to}/{num}'
    ext_dir = join(output_base,'extrinsic')
    int_dir = join(output_base,'intrinsic')
    try:
        shutil.rmtree(ext_dir)
    except:
        pass
    try:
        shutil.rmtree(int_dir)
    except:
        pass
    os.makedirs(ext_dir,exist_ok=True)
    os.makedirs(int_dir,exist_ok=True)
    for i in range(cam_num):
        with open(join(ext_dir,f'{i:03}.txt'),'w+') as f:
            if num == 2: 
                RT = np.vstack((np.hstack((cams['R'][i],(np.array(cams['T'][i])/1).tolist())),[0,0,0,1]))
            else:
                RT = np.vstack((np.hstack((cams['R'][i],(np.array(cams['T'][i])/1000).tolist())),[0,0,0,1]))
            for row in range(4):
                for col in range(4):
                    f.write(str(RT[row][col])+' ')
                f.write('\n')
        with open(join(int_dir,f'{i:03}.txt'),'w+') as f:
            K = cams['K'][i]
            for row in range(3):
                for col in range(3):
                    f.write(str(K[row][col])+' ')
                f.write('\n')


def get_rgb_bg(num,j=0,l=1):
    base_dir = f'{data_from}/CoreView_{num}'
    cams = os.listdir(base_dir)
    cams = list(filter(lambda x: re.match('Camera*', x) != None,cams))        
    max_frame_len = len(sorted(os.listdir((join(base_dir,re.sub(r'\d+',str(1),cams[0]))))))
    all_frames = range(min(total_frame,max_frame_len))
    if num == 9:
        all_frames = all_frames[810:]
    all_frames = all_frames[j::l]
    for frame in tqdm.tqdm(all_frames):
        output_dir = f'{data_to}/{num}/rgb_bg/{frame:06}'            
        os.makedirs(output_dir, exist_ok = True)
        for cam_id, cam in enumerate(cams):
                #idx = re.findall(r'\d+', cam)[0]
            file_dir = join(base_dir,re.sub(r'\d+',str(cam_id+1),cam))
            files = sorted(os.listdir(file_dir))

            image = cv2.imread(join(file_dir,files[frame]))
            cv2.imwrite(join(output_dir,f'{cam_id:03}_{frame:06}.png'),image)

def get_transform(num):
    if new_params:
        smpl_dir = f"{data_from}/CoreView_{num}/new_params"
        if num == 9:
            smpl_dir = f"{data_from}/CoreView_{num}/params"
    else:
        smpl_dir = f"{data_from}/CoreView_{num}/params"
        
    save_dir = join(f'{data_to}/{num}','smpl_transform/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    import re
    all_frame = sorted(os.listdir(smpl_dir), key = lambda x: int(re.findall(r"\d+",x)[0]))
    for frame, params in tqdm.tqdm(enumerate(all_frame)):

        params = np.load(join(smpl_dir,params), allow_pickle=True).item()
        transform = {
                    'Rh': params['Rh'].tolist(),
                    'Th': params['Th'].tolist(),
                    "poses": params['poses'].tolist(),
                    "shapes": params['shapes'].tolist()}
        if num == 9:
            frame += 810
        with open(join(save_dir,f'{frame:06}.json'), 'w') as fw:
            json.dump(transform, fw, indent = 2)

def get_mask(num,j=0,l=1):
    base_dir = f'{data_from}/CoreView_{num}/mask'
    cams = os.listdir(base_dir)
    cams = list(filter(lambda x: re.match('Camera*', x) != None,cams))        
    max_frame_len = len(sorted(os.listdir((join(base_dir,re.sub(r'\d+',str(1),cams[0]))))))
    all_frames = range(min(total_frame,max_frame_len))
    all_frames = all_frames[j::l]
    if num == 9:
        return
    for frame in tqdm.tqdm(all_frames):
        # print(all_frames[1])
        # if num == 9:
        #     frame += 810
        output_dir = f'{data_to}/{num}/mask/{frame:06}'            
        os.makedirs(output_dir, exist_ok = True)
        # print(output_dir)
        for cam_id, cam in enumerate(cams):
            file_dir = join(base_dir,re.sub(r'\d+',str(cam_id+1),cam))
            files = sorted(os.listdir(file_dir))
            # print(files[frame])
            # print(file_dir)
            # print(join(file_dir,files[frame]))
            try:
                image = cv2.imread(join(file_dir,files[frame]))
            except Exception as e:
                print(e)
            cv2.imwrite(join(output_dir,f'{cam_id:03}_{frame:06}.png'),image)

def get_nb_another_mask(num,j=0,l=1):
    base_dir = f'{data_from}/CoreView_{num}/mask_cihp'
    cams = os.listdir(base_dir)
    cams = list(filter(lambda x: re.match('Camera*', x) != None,cams))        
    max_frame_len = len(sorted(os.listdir((join(base_dir,re.sub(r'\d+',str(1),cams[0]))))))
    all_frames = range(min(total_frame,max_frame_len))
    all_frames = all_frames[j::l]
    for frame in tqdm.tqdm(all_frames):
        if num == 9:
            frame += 810
        output_dir = f'{data_to}/{num}/nb_another_mask/{frame:06}'            
        os.makedirs(output_dir, exist_ok = True)
        for cam_id, cam in enumerate(cams):
                #idx = re.findall(r'\d+', cam)[0]
            file_dir = join(base_dir,re.sub(r'\d+',str(cam_id+1),cam))
            files = sorted(os.listdir(file_dir))
            if num == 9:
                image = cv2.imread(join(file_dir,files[frame-810]))
            else:
                image = cv2.imread(join(file_dir,files[frame]))
            cv2.imwrite(join(output_dir,f'{cam_id:03}_{frame:06}.png'),image)

def get_undistort(num):
    input_base = f'{data_from}/CoreView_{num}/annots.json'
    if os.path.exists(input_base):
        cams = json.load(open(input_base, 'r'))
    else:
        input_base = f'{data_from}/CoreView_{num}//annots.npy'
        cams = np.load(input_base,allow_pickle=True).item()

    if num ==0 or num==1:
        cams = cams['cams']['20190823']
        cam_num = 21
    else:
        cam_num = 23
        cams = cams['cams']

    output_base = f'{data_to}/{num}'
    ds_dir = join(output_base,'undistort')
    try:
        shutil.rmtree(ds_dir)
    except:
        pass

    os.makedirs(ds_dir,exist_ok=True)
    for i in range(cam_num):
        ds = np.array(cams['D'][i])
        np.save(join(ds_dir,f'{i:03}.npy'),ds)

if __name__=='__main__':

    for num in total_num:
        get_ext_ixt(num)
        get_transform(num)
        get_undistort(num)

        p = Pool(per_thread*3)
        for i in range(per_thread):
            p.apply_async(get_mask, args=(num,i,per_thread,))
            p.apply_async(get_nb_another_mask, args=(num,i,per_thread,))
            p.apply_async(get_rgb_bg, args=(num,i,per_thread,))
           

            
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')