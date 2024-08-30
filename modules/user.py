#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
import imp
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import __init__ as booger

from tqdm import tqdm
from modules.KNN import KNN
# from modules.SalsaNextWithMotionAttention import *
from modules.CVMOS import *

from modules.PointRefine.attn_spvcnn import SPVCNN
# from modules.PointRefine.spvcnn_lite import SPVCNN

class User:
    def __init__(
            self,
             arch,
             data,
             datadir,
             outputdir,
             modeldir,
             split,
             point_refine=False,
             save_movable=False
    ):
        # parameters
        self.arch = arch
        self.data = data
        self.datadir = datadir
        self.outputdir = outputdir
        self.modeldir = modeldir
        self.split = split
        self.post = None
        self.infer_batch_size = 1
        self.point_refine = point_refine
        self.save_movable = save_movable
        # get the data
        parserModule = imp.load_source("parserModule",
                                       f"{booger.TRAIN_PATH}/common/dataset/{self.data['name']}/parser.py")
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=self.data["split"]["train"],
                                          valid_sequences=self.data["split"]["valid"],
                                          test_sequences=self.data["split"]["test"],
                                          split=self.split,
                                          labels=self.data["labels"],
                                          residual_aug=self.arch["train"]["residual_aug"],
                                          valid_residual_delta_t=self.arch["train"]["valid_residual_delta_t"],
                                          color_map=self.data["color_map"],
                                          learning_map=self.data["moving_learning_map"],
                                          movable_learning_map=self.data["movable_learning_map"],
                                          learning_map_inv=self.data["moving_learning_map_inv"],
                                          movable_learning_map_inv=self.data["movable_learning_map_inv"],
                                          sensor=self.arch["dataset"]["sensor"],
                                          max_points=self.arch["dataset"]["max_points"],
                                          batch_size=self.infer_batch_size,
                                          workers=2, # self.arch["train"]["workers"],
                                          gt=True,
                                          shuffle_train=False)

        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if not point_refine:
                self.model = CVMOS(nclasses=self.parser.get_n_classes(),
                                   movable_nclasses=self.parser.get_n_classes(movable=True),
                                   params=arch, num_batch=self.infer_batch_size)
                self.polar_preprocess = self.model.polar_preprocess
                self.model = nn.DataParallel(self.model)
                checkpoint = "CVMOS_valid_best"
                w_dict = torch.load(f"{self.modeldir}/{checkpoint}", map_location=lambda storage, loc: storage)
                self.model.load_state_dict(w_dict['state_dict'], strict=True)

                self.set_knn_post()
            else:
                self.model = CVMOS(nclasses=self.parser.get_n_classes(),
                                   movable_nclasses=self.parser.get_n_classes(movable=True),
                                   params=arch, num_batch=self.infer_batch_size)
                self.polar_preprocess = self.model.polar_preprocess
                self.model = nn.DataParallel(self.model)
                checkpoint = "CVMOS_2stage_valid_best"
                w_dict = torch.load(f"{self.modeldir}/{checkpoint}", map_location=lambda storage, loc: storage)
                self.model.load_state_dict({f"module.{k}":v for k,v in w_dict['main_state_dict'].items()}, strict=True)

                net_config = {'num_classes': self.parser.get_n_classes(), 'cr': 1.0, 'pres': 0.05, 'vres': 0.05}
                self.refine_module = SPVCNN(num_classes=net_config['num_classes'],
                                            cr=net_config['cr'],
                                            pres=net_config['pres'],
                                            vres=net_config['vres'])
                # self.refine_module = nn.DataParallel(self.refine_module)
                w_dict = torch.load(f"{modeldir}/{checkpoint}", map_location=lambda storage, loc: storage)
                # self.refine_module.load_state_dict({f"module.{k}":v for k,v in w_dict['refine_state_dict'].items()}, strict=True)
                self.refine_module.load_state_dict({f"{k}": v for k, v in w_dict['refine_state_dict'].items()},
                                                   strict=True)

        self.set_gpu_cuda()

    def set_knn_post(self):
        # use knn post processing?
        if self.arch["post"]["KNN"]["use"]:
            self.post = KNN(self.arch["post"]["KNN"]["params"], self.parser.get_n_classes())

    def set_gpu_cuda(self):
        # GPU?
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()
            if self.point_refine:
                self.refine_module.cuda()


    def infer(self):
        cnn, knn = [], []

        if self.split == 'valid':
            self.infer_subset(loader=self.parser.get_valid_set(),
                              to_orig_fn=self.parser.to_original,
                              cnn=cnn, knn=knn)
        elif self.split == 'train':
            self.infer_subset(loader=self.parser.get_train_set(),
                              to_orig_fn=self.parser.to_original,
                              cnn=cnn, knn=knn)
        elif self.split == 'test':
            self.infer_subset(loader=self.parser.get_test_set(),
                              to_orig_fn=self.parser.to_original,
                              cnn=cnn, knn=knn)
        elif self.split is None:
            self.infer_subset(loader=self.parser.get_train_set(),
                              to_orig_fn=self.parser.to_original,
                              cnn=cnn, knn=knn)
            # do valid set
            self.infer_subset(loader=self.parser.get_valid_set(),
                              to_orig_fn=self.parser.to_original,
                              cnn=cnn, knn=knn)
            # do test set
            self.infer_subset(loader=self.parser.get_test_set(),
                              to_orig_fn=self.parser.to_original,
                              cnn=cnn, knn=knn)
        else:
            raise NotImplementedError

        print(f"Mean CNN inference time:{'%.8f'%np.mean(cnn)}\t std:{'%.8f'%np.std(cnn)}")
        print(f"Mean KNN inference time:{'%.8f'%np.mean(knn)}\t std:{'%.8f'%np.std(knn)}")
        print(f"Total Frames: {len(cnn)}")
        print("Finished Infering")

        return

    def infer_subset(self, loader, to_orig_fn, cnn, knn):

        # switch to evaluate mode
        self.model.eval()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():

            end = time.time()

            # for i, (proj_in, proj_mask, _, _, path_seq, path_name,
            #         p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints)\
            #         in enumerate(tqdm(loader, ncols=80)):
                # first cut to rela size (batch size one allows it)

            for i, (proj_in, proj_mask, _, _, path_seq, path_name,
                    p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints, polar_data, p2r_matrix) in enumerate(tqdm(loader, ncols=80)):

                p_x = p_x[0, :npoints]
                p_y = p_y[0, :npoints]
                proj_range = proj_range[0, :npoints]
                unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]

                ########################## polar ##############################################################
                train_grid, num_pt, bev_residual_data = polar_data
                p2r_matrix = p2r_matrix[:, :, :, :2]
                xy_ind = train_grid[:, :, :-1]  # bev投影要把z切掉

                if self.gpu:
                    proj_in = proj_in.cuda()
                    p_x = p_x.cuda()
                    p_y = p_y.cuda()

                    ### bev
                    bev_residual_data = bev_residual_data.cuda()
                    xy_ind = xy_ind.cuda()
                    p2r_matrix = p2r_matrix[:, :, :, :2].cuda()

                    if self.post:
                        proj_range = proj_range.cuda()
                        unproj_range = unproj_range.cuda()

                cat_bev_res_fea, unq, unq_inv, batch_size, cur_dev = self.polar_preprocess(bev_residual_data, xy_ind, num_pt)

                # compute output
                end = time.time()
                # compute output
                proj_output, _, movable_proj_output, _ = self.model(proj_in,  cat_bev_res_fea, unq, unq_inv, batch_size, cur_dev, p2r_matrix, train_mode=False)
                res = time.time() - end
                cnn.append(res)

                proj_argmax = proj_output[0].argmax(dim=0)
                movable_proj_argmax = movable_proj_output[0].argmax(dim=0)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()
                # print(f"Network seq {path_seq} scan {path_name} in {res} sec")

                # if knn --> use knn to postprocess
                # 	else put in original pointcloud using indexes
                if self.post:
                    unproj_argmax = self.post(proj_range, unproj_range,
                                              proj_argmax, p_x, p_y)
                    if self.save_movable:
                        movable_unproj_argmax = self.post(proj_range, unproj_range,
                                                          movable_proj_argmax, p_x, p_y)
                else:
                    unproj_argmax = proj_argmax[p_y, p_x]
                    if self.save_movable:
                        movable_unproj_argmax = movable_proj_argmax[p_y, p_x]

                # measure elapsed time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                knn.append(res)
                # print(f"KNN Infered seq {path_seq} scan {path_name} in {res} sec")

                # save scan # get the first scan in batch and project scan
                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)

                # map to original label
                pred_np = to_orig_fn(pred_np)

                path = os.path.join(self.outputdir, "sequences", path_seq, "predictions", path_name)
                pred_np.tofile(path)

                if self.save_movable:
                    movable_pred_np = movable_unproj_argmax.cpu().numpy()
                    movable_pred_np = movable_pred_np.reshape((-1)).astype(np.int32)

                    # map to original label
                    movable_pred_np = to_orig_fn(movable_pred_np, movable=True)
                    path = os.path.join(self.outputdir, "sequences", path_seq, "predictions_movable", path_name)
                    movable_pred_np.tofile(path)

                    movable_pred_np[np.where(pred_np == 251)] = 251
                    path = os.path.join(self.outputdir, "sequences", path_seq, "predictions_fuse", path_name)
                    movable_pred_np.tofile(path)