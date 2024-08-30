#!/bin/bash

dataset_path=/data/datasets/dataset/mfmos_data_aug
predictions_path=/data/czy/CV-MOS/log/valid_old
data_config=./config/labels/semantic-kitti-mos.raw.yaml
split=valid # valid or test

python3 utils/evaluate_mos.py -d $dataset_path \
                              -p $predictions_path \
                              -dc $data_config \
                              -s $split \
