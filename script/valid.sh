#!/bin/bash

DatasetPath=/data/datasets/dataset/mfmos_data_aug
bev_res_path=/data/datasets/dataset/kitti_dataset/motionbev_residual/
ModelPath=/data/czy/CV-MOS/pretrain_weight/
SavePath=/data/czy/CV-MOS/log/valid_old
SPLIT=valid # valid or test

# If you want to use SIEM, set pointrefine on
export CUDA_VISIBLE_DEVICES=2 && python3 infer.py -d $DatasetPath \
                                                  - brp $bev_res_path \
                                                  -m $ModelPath \
                                                  -l $SavePath \
                                                  -s $SPLIT \
                                                  --pointrefine \
#                                                  --movable # Whether to save the label of movable objects
