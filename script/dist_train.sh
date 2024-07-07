#!/bin/bash

DatasetPath=/data/2022/mos-dataset/Apollo
ArchConfig=./train_yaml/ddp_mos_coarse_stage.yml
DataConfig=./config/labels/apollo.yaml
LogPath=/data/2022/zeyu/CV-MOS-apollo/log/apollo_finetune
pretrain=/data/2022/zeyu/CV-MOS-apollo/pretrain_weight
export CUDA_VISIBLE_DEVICES=4,5,6,7 && python3 -m torch.distributed.launch --nproc_per_node=4 \
                                           ./train.py -d $DatasetPath \
                                                      -ac $ArchConfig \
                                                      -dc $DataConfig \
                                                      -l $LogPath \
                                                      -p $pretrain
