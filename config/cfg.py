from email.policy import default
import sys
import os


import argparse
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter
import json

from yaml import parse

def parse_args(
    callback=None,
    training=False,
    default_conf="config/default.conf",
    default_expname="example",
    default_num_epochs=10000000,
    default_lr=1e-4,
    default_gamma=1.00,
    default_weight_decay=0,
    default_datadir="data/neuralbody",
    default_outputdir='save/test',
    default_ray_batch_size=50000,
):
    parser = argparse.ArgumentParser()

    ## logging dir
    parser.add_argument("--conf", "-c", type=str, default=default_conf)
    parser.add_argument(
        "--name", "-n", type=str, default=default_expname, help="experiment name"
    )
    parser.add_argument(
        "--gpu-id", type=str, default="0", help="GPU(s) to use, space delimited")
    parser.add_argument("--resume",action='store_true',help="continue training or evaluate")
    parser.add_argument("--train-lpips",action='store_true',help="training of lpips")
    parser.add_argument(
        "--checkpoints-path",
        type=str,
        default="checkpoints",
        help="checkpoints output directory",
    )
    parser.add_argument(
        "--tensorboard-path",
        type=str,
        default="tensorboard",
        help="tensorboard output directory",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="log",
        help="log output directory",
    )    
    parser.add_argument(
        "--mesh-path",
        type=str,
        default="mesh",
        help="mesh output directory",
    )    



    ## loss
    parser.add_argument('--patch-sample',action='store_true')
    parser.add_argument('--patch-num',type=int,default=1)
    parser.add_argument('--patch-size',type=int,default=30)
    parser.add_argument('--lpips-loss-weight', type=float, default=1)
    parser.add_argument('--igr-weight',type=float,default=0.1)
    parser.add_argument('--mask-weight',type=float,default=0.1)

    ## dataset params
    parser.add_argument('--characters',type=str,default='-1')

    parser.add_argument('--aist-data',action='store_true')
    parser.add_argument('--na-data',action='store_true')
    parser.add_argument('--na-with-mask',action='store_true')
    parser.add_argument('--realworld-data',action='store_true')
    parser.add_argument('--na-crop',action='store_true')
    parser.add_argument('--na-crop-inline',action='store_true')
    parser.add_argument("--latent-codes-views", type=str, default=None, 
                        help="get latent codes from given views")
    parser.add_argument('--nhp-eval',action='store_true') ## todo
    parser.add_argument('--nhp-psnr',action='store_true') ## todo
    parser.add_argument('--big-box',action='store_true') ## todo
    parser.add_argument('--save-nb-data',action='store_true')
    parser.add_argument('--nb-another-mask',action='store_true')
    parser.add_argument('--camera-undistort',action='store_true')
    parser.add_argument('--scale-size',type=float, default=1.0)
    parser.add_argument('--bball-radius',type=float, default=1.0)

    ## training params
    parser.add_argument('--local_rank',type=int,default=-1)
    parser.add_argument('--save-ep',type=int,default=200,help='save epoch interval')
    parser.add_argument('--save-latest-ep',type=int,default=10,help='save latest epoch')    
    parser.add_argument('--eval-ep',type=int,default=10,help='save latest epoch')  
    parser.add_argument('--render-ep',type=int,default=100,help='render latest epoch')
    parser.add_argument('--record-interval',type=int,default=5)
    parser.add_argument('--log-interval',type=int,default=1)
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--gamma", type=float, default=default_gamma, help="learning rate decay factor")   
    parser.add_argument("--weight-decay", type=float, default=default_weight_decay, help="weight decay factor")     
    parser.add_argument('--optim',type=str,default='adam',help="optimizer type")
    parser.add_argument("--ray-batch-size", "-R", type=int, default=default_ray_batch_size, help="Ray batch size")
    parser.add_argument("--root-path", default=default_datadir, help="data-directory for validation")
    parser.add_argument('--output-path', default=default_outputdir, type=str, help='output directiory')
    parser.add_argument("--valid-data", default=None, help="data-directory for validation")
    parser.add_argument('--fix-latent-codes',action='store_true')
    parser.add_argument("--transparent-background", type=str, default="1.0",
                        help="background color if the image is transparent")
    parser.add_argument("--train-views", type=str, default="0..20", 
                        help="views sampled for training, you can set specific view id, or a range")
    parser.add_argument("--valid-views", type=str, default="0..20",
                        help="views sampled for validation,  you can set specific view id, or a range")
    parser.add_argument("--test-views", type=str, default=None,
                        help="views sampled for rendering, only used for showing rendering results.")
    parser.add_argument("--subsample-valid", type=int, default=1,
                        help="if set > -1, subsample the validation (when training set is too large)")
    parser.add_argument("--subsample-train", type=int, default=1,
                        help="if set > -1, subsample the validation (when training set is too large)")
    parser.add_argument("--train-start-end", type=str, default=None)
    parser.add_argument("--test-start-end", type=str, default=None)
    parser.add_argument('--is-train', action='store_true',
                        help="Training mode")
    parser.add_argument('--generate-mesh', action='store_true',
                        help="generate mesh")
    parser.add_argument('--save-freq',type=int,default=1000)
    parser.add_argument('--val-freq',type=int,default=1000)
    parser.add_argument('--val-mesh-freq',type=int,default=1000)
    parser.add_argument('--preload-sdf',action='store_true')
    parser.add_argument('--preload-sdf-path',type=str)
    parser.add_argument('--smpl-augmentation',action='store_true')
    parser.add_argument('--finetune-sdf',action='store_true')
    parser.add_argument('--finetune-sdf-end',type=int,default=200000)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--patch-ray-batch-size',type=int,default=1024)
    parser.add_argument('--eval-batch-size',type=int,default=512)
    parser.add_argument('--end-iter',type=int,default=100000)
    parser.add_argument('--begin-optimize-residual',type=int,default=-1)
    parser.add_argument('--pg-jitter-stage',type=float,default=1000)

    ## eval params
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--app-editing',action='store_true')
    parser.add_argument('--app-target',type=int,default=0)
    parser.add_argument('--original-shape',action='store_true')
    parser.add_argument('--novel-shape',action='store_true')
    parser.add_argument('--a-pose',action='store_true')
    parser.add_argument('--fix-camera',action='store_true')
    parser.add_argument('--fix-camera-id',type=int,default=0)
    parser.add_argument('--pose-driven', action='store_true')
    parser.add_argument('--novel-pose-path', type=str,default=None)
    parser.add_argument('--load-model-path', type=str,default=None)
    parser.add_argument('--novel-views', type=str)
    parser.add_argument('--interpolate-views', action='store_true')

    ## render params
    parser.add_argument('--use-white-bkgd',action='store_true')
    parser.add_argument('--occlusion-dis',type=float,default=0.03)
    parser.add_argument('--n-samples',type=int,default=32)
    parser.add_argument('--n-importance',type=int,default=32)
    parser.add_argument('--up-sample-steps',type=int,default=4)
    parser.add_argument("--min-dis-eps", type=float, default=0.08, help="neural actor sample distance")
    parser.add_argument('--neuralbody-sample',action='store_true')

    
    ## model params
    parser.add_argument('--head-no-skeletal',action='store_true') ## todo::remove this
    parser.add_argument('--gnn-layer-num',type=int,default=3)
    parser.add_argument('--latent-codes-dim',type=int,default=56)
    parser.add_argument('--knn-num',type=int,default=8)
    parser.add_argument('--knn-batch-size',type=int,default=500000)
    parser.add_argument('--neus-variance',type=float,default=0.3)

    if callback is not None:
        parser = callback(parser)
    args = parser.parse_args()


    args.checkpoints_path = os.path.join(args.output_path, args.checkpoints_path, args.name)
    args.tensorboard_path = os.path.join(args.output_path, args.tensorboard_path, args.name)
    
    args.mesh_path = os.path.join(args.output_path, args.mesh_path, args.name)
    args.render_path = os.path.join(args.output_path, 'render', args.name)
    
    
    args.log_path = os.path.join(args.output_path, args.name,args.log_path)
    if args.is_train:
        os.makedirs(args.log_path, exist_ok=True)

    conf = ConfigFactory.parse_file(args.conf)


    args.gpu_id = list(map(int, args.gpu_id.split()))


     
    if args.is_train:
        with open(args.log_path+'/config.conf', 'w') as configfile:
            configfile.write(HOCONConverter.to_hocon(conf))
        with open(args.log_path+'/args.json','w') as argsfile:
            json.dump(args.__dict__, argsfile, indent=2)

    return args, conf
