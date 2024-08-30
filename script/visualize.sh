#!/bin/bash

dataset_path=dataroot
seq=08
data_config=./config/labels/semantic-kitti-mos.raw.yaml
version=fuse # version in ["moving", "movable", "fuse"] for predictions
#predictionpath=./log/valid/predictions

python3 utils/visualize_mos.py -d $dataset_path \
                               -s $seq \
                               -c $data_config \
                               -v $version \
                               # -p $predictionpath
