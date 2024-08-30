#!/bin/bash

DatasetPath=/data/datasets/dataset/mfmos_data_aug
PredictionsPath=/data/czy/CV-MOS/log/valid_old
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml
SPLIT=valid # valid or test

python3 utils/evaluate_mos.py -d $DatasetPath \
                              -p $PredictionsPath \
                              -dc $DataConfig \
                              -s $SPLIT \
