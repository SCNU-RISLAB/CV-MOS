#!/bin/bash

DatasetPath=/data/2022/mos-dataset/Apollo
PredictionsPath=/data/2022/zeyu/CV-MOS-apollo/log/apollo_finetune2stage
DataConfig=./config/labels/apollo.yaml
SPLIT=valid # valid or test

python3 utils/evaluate_mos.py -d $DatasetPath \
                              -p $PredictionsPath \
                              -dc $DataConfig \
                              -s $SPLIT \
