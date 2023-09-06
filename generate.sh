CUDA_VISIBLE_DEVICES=3 \
NCCL_P2P_DISABLE=1 \
python \
    generate.py \
    --conf config/default.conf \
    --name test\
    --output-path save/generate \
    --root-path data/neuralbody \
    --gpu-id 0\
    --nhp-psnr\
    --big-box\
    --head-no-skeletal\
    --resume\
    --camera-undistort\
    --nb-another-mask\
    --scale-size 0.8 \
    --load-model-path save/released_checkpoints.pth\
    --subsample-valid 1 \
    --test-start-end 300,400\
    --latent-codes-views 0,6,12,18\
    --characters 7\
    --pose-driven\
    --novel-pose-path data/neuralbody/4\
    --interpolate-views\
    --generate-mesh\
    --fix-camera\
    --fix-camera-id 0\