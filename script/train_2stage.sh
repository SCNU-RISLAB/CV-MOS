#!/bin/bash

dataset_path=/data/zeyu/datasets/kitti
arch_config=./train_yaml/mos_pointrefine_stage.yml
data_config=./config/labels/semantic-kitti-mos.raw.yaml
log_path=./log/valid_2stage
bev_res_path=/data/zeyu/datasets/kitti/sequences/
first_stage_model_path=/data/zeyu/mos/CV-MOS3090/pretrain_weight/Train_AUG3_77.4/2024-3-12-14_07

export CUDA_VISIBLE_DEVICES=3 && python train_2stage.py -d $dataset_path \
                                                        -ac $arch_config \
                                                        -dc $data_config \
                                                        -brp $bev_res_path \
                                                        -l $log_path\
                                                        -p $first_stage_model_path