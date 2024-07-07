#!/bin/bash

DatasetPath=/data/2022/mos-dataset/Apollo
ArchConfig=./train_yaml/mos_pointrefine_stage.yml
DataConfig=./config/labels/apollo.yaml
LogPath=./log/apollo_finetune2stage
FirstStageModelPath=/data/2022/zeyu/CV-MOS-apollo/log/apollo_finetune/2024-4-15-01:34

export CUDA_VISIBLE_DEVICES=6 && python train_2stage.py -d $DatasetPath \
                                                        -ac $ArchConfig \
                                                        -dc $DataConfig \
                                                        -l $LogPath \
                                                        -p $FirstStageModelPath