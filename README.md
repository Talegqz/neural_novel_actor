# Neural Novel Actor: Learning a Generalized Animatable Neural Representation for Human Actors.
### [[Project]](https://talegqz.github.io/neural_novel_actor/)[ [Paper]](https://arxiv.org/abs/2208.11905)


> Neural Novel Actor: Learning a Generalized Animatable Neural Representation for Human Actors.
>[Qingzhe Gao*](https://talegqz.github.io/), [Yiming Wang*](https://19reborn.github.io/), [Libin Liu†](http://libliu.info/), [Lingjie Liu†](https://lingjie0206.github.io/), [Christian Theobalt](http://people.mpi-inf.mpg.de/~theobalt/), [Baoquan Chen†](https://cfcs.pku.edu.cn/baoquan/)

> TVCG 2023

<!-- todo:: add demo gif
<img src="docs/demo.gif" height="342"/> -->
### Updates

- [x] [09/06/2023] Released official test codes and pretrained checkpoints!

## Installation

First clone this repository and all its submodules using the following command:
```
git clone --recursive https://github.com/Talegqz/neural_novel_actor
cd neural_novel_actor
```

Then install dependencies with conda and pip:

```
conda create -n nna python=3.8
conda activate nna

pip install -r requirements.txt

python setup.py build_ext --inplace

pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

```

## Dataset
We provide a script to convert the [ZJU-dataset](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset) to our data convention, which can be found in `tools/dataset_from_zju_mocap.py`.


## Test
First download the pretrained checkpoints from [Google Drive](https://drive.google.com/file/d/17f1TBhlORBJ5a5b_UK2Bp9He47RZrcls/view?usp=sharing), and then put it in the `save` folder. 
You can then generate pose driven results using the following command.

```
bash generate.sh
```

## Prepare your own data
To test the model on your own dataset, please organize your dataset in following structure:


The data is organized like:
```
<dataset_path>/0          # character id
|-- intrinsic             # camera intrinsic for each camera, fixed across all frames 
    |-- 0000.txt
    |-- 0001.txt
    ...
|-- extrinsic                  # camera extrinsic for each camera, fixed across all frames
    |-- 0000.txt
    |-- 0001.txt
    ...
|-- smpl_transform             # json files defined the target pose transformation (produced by EasyMocap) 
    |-- 000000.json       
    |-- 000001.json  
    ...
|-- rgb_bg                   # ground-truth RGB image for each frame and each camera
    |-- 000000            # frame id
        |-- 0000.png
        |-- 0001.png
        ...
    |-- 000001            # camera id
        |-- 0000.png
        |-- 0001.png
        ...
    ...     
|-- mask                   # ground-truth mask image for each frame and each camera
    |-- 000000            # frame id
        |-- 0000.png
        |-- 0001.png
        ...
    |-- 000001            # camera id
        |-- 0000.png
        |-- 0001.png
        ...
    ...     
```

## Citation

@article{gao2023neural,
  title={Neural novel actor: Learning a generalized animatable neural representation for human actors},
  author={Gao, Qingzhe and Wang, Yiming and Liu, Libin and Liu, Lingjie and Theobalt, Christian and Chen, Baoquan},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2023},
  publisher={IEEE}
}