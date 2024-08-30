#!/bin/bash

dataset_path=/data/zeyu/datasets/kitti/
model_path=/data/zeyu/mos/CV-MOS3090/log/valid_2stage/2024-8-29-16:09/
save_path=/data/czy/CV-MOS/log/valid
split=valid # valid or test

# If you want to use SIEM, set pointrefine on
export CUDA_VISIBLE_DEVICES=2 && python3 infer.py -d $dataset_path \
                                                  -m $model_path \
                                                  -l $save_path \
                                                  -s $split \
                                                #   --pointrefine \
#                                                  --movable # Whether to save the label of movable objects
