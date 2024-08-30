#!/bin/bash

dataset_path=/data/zeyu/datasets/kitti
arch_config=./train_yaml/ddp_mos_coarse_stage.yml
data_config=./config/labels/semantic-kitti-mos.raw.yaml
log_path=/data/zeyu/mos/CV-MOS3090/log/train
pretrain=None

# export MASTER_ADDR=127.0.1.2
# export MASTER_PORT=29502
export CUDA_VISIBLE_DEVICES=2,3  && python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 \
                                           ./train.py -d $dataset_path \
                                                      -ac $arch_config \
                                                      -dc $data_config \
                                                      -l $log_path \
                                                    #   -p $pretrain \
