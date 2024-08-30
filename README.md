# [TIM2024]CV-MOS: A Cross-View Model for Motion Segmentation

<div align="center">
<div>
    <a href="http://arxiv.org/abs/2408.13790"><img src="http://img.shields.io/badge/paper-arXiv.cs.CV%3A2401.17023-B31B1B.svg"></a>
    <a href="https://www.bilibili.com/video/BV1ZM411Z7rP/?share_source=copy_web&vd_source=902841b9751bc137897f677e1ea56624"><img src="http://img.shields.io/badge/video-bilibili%3ACV--MOS-FF6699.svg"></a>
  </div>
</div>

## 📖How to use
### 📦pretrained model
Our pretrained model (validation with the IoU of **_77.5%_**, test with the IoU of **_79.2%_**) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1FURxrcwXTilTxCkn6uky9sutAKG-Nq1_?usp=sharing)
### 📚Dataset 
Download SemanticKITTI dataset from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) (including **Velodyne point clouds**, **calibration data** and **label data**).
#### Preprocessing
After downloading the dataset, the residual maps as the input of the model during training need to be generated.
Run [auto_gen_residual_images.py](./utils/auto_gen_residual_images.py) or [auto_gen_residual_images_mp.py](./utils/auto_gen_residual_images_mp.py)(with multiprocess) and [auto_gen_polar_sequential_residual_images.py](./bev_utils/generate_residual/utils/auto_gen_polar_sequential_residual_images.py),
and check that the path is correct before running.

The structure of one of the folders in the entire dataset is as follows:
```
DATAROOT
└── sequences
    ├── 00
    │   ├── poses.txt
    │   ├── calib.txt
    │   ├── times.txt
    │   ├── labels
    │   ├── residual_images(the bev residual images)
    │   ├── residual_images_1
    │   ├── residual_images_10
    │   ├── residual_images_11
    │   ├── residual_images_13
    │   ├── residual_images_15
    │   ├── residual_images_16
    │   ├── residual_images_19
    │   ├── residual_images_2
    │   ├── residual_images_22
    │   ├── residual_images_3
    │   ├── residual_images_4
    │   ├── residual_images_5
    │   ├── residual_images_6
    │   ├── residual_images_7
    │   ├── residual_images_8
    │   ├── residual_images_9
    │   └── velodyne
   ...
```
If you don't need to do augmentation for residual maps, you just need the folder with num [1, 2, 3, 4, 5, 6, 7, 8].

### 💾Environment
Our environment: Ubuntu 18.04, CUDA 11.2 

Use conda to create the conda environment and activate it:
```shell
conda env create -f environment.yml
conda activate cvmos
```
#### TorchSparse
Install torchsparse which is used in [2stage](./modules/PointRefine/spvcnn.py) using the commands:
```shell
sudo apt install libsparsehash-dev 
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

### 📈Train 1stage
Check the path in [dist_train.sh](./script/dist_train.sh), and run it to train:
```shell
bash script/dist_train.sh
```
You can change the number of GPUs as well as ID to suit your needs.
#### Train 2stage
Once you have completed the first phase of training above, you can continue with SIEM training to get an improved performance.

Check the path in [train_2stage.sh](./script/train_2stage.sh) and run it to train the SIEM **(only available on single GPU)**:
```shell
bash script/train_siem.sh
```

### 📝Validation and Evaluation
Check the path in [valid.sh](./script/valid.sh) and [evaluate.sh](./script/evaluate.sh).

Then, run them to get the predicted results and IoU in the paper separately:
```shell
bash script/valid.sh
# evaluation after validation
bash script/evaluate.sh
```
You can also use our pre-trained model which has been provided above to validate its performance.


### 👀Visualization
#### Single-frame visualization
Check the path in [visualize.sh](./script/visualize.sh), and run it to visualize the results in 2D and 3D:
```shell
bash script/visualize.sh
```
If -p is empty: only ground truth will be visualized.

If -p set the path of predictions: both ground truth and predictions will be visualized.
![Single frame visualization](./assets/VisualizeSingleFrame.jpg)
#### Get the sequences video
Check the path in [viz_seqVideo.py](./utils/viz_seqVideo.py), and run it to visualize the entire sequence in the form of a video.


## 👏Acknowledgment
This repo is based on MF-MOS, MotionBEV, GFNet... We are very grateful for their excellent work.
