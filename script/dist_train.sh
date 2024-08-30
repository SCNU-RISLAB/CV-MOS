#!/bin/bash

dataset_path=/data/datasets/dataset/mfmos_data_aug
arch_config=./train_yaml/ddp_mos_coarse_stage.yml
data_config=./config/labels/semantic-kitti-mos.raw.yaml
bev_res_path=/data/datasets/dataset/kitti_dataset/motionbev_residual
log_path=/data/2022/zeyu/CV-MOS-apollo/log/apollo_finetune
pretrain=/data/2022/zeyu/CV-MOS-apollo/pretrain_weight

export CUDA_VISIBLE_DEVICES=0 && python3 -m torch.distributed.launch --nproc_per_node=1 \
                                           ./train.py -d $dataset_path \
                                                      -ac $arch_config \
                                                      -dc $data_config \
                                                      -l $log_path \
                                                      -p $pretrain \
                                                      -brp $bev_res_path
