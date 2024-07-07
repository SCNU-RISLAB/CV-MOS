#!/bin/bash

DatasetPath=/data/2022/mos-dataset/Apollo
ModelPath=/data/2022/zeyu/CV-MOS-apollo/log/apollo_finetune2stage/2024-4-16-00:46
SavePath=/data/2022/zeyu/CV-MOS-apollo/log/apollo_finetune2stage
SPLIT=valid # valid or test

# If you want to use SIEM, set pointrefine on
export CUDA_VISIBLE_DEVICES=7 && python3 infer.py -d $DatasetPath \
                                                  -m $ModelPath \
                                                  -l $SavePath \
                                                  -s $SPLIT \
                                                  --pointrefine \
#                                                  --movable # Whether to save the label of movable objects
