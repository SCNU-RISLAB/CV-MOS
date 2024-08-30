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
from modules.user import User
# from modules.SalsaNextWithMotionAttention import *

# from modules.PointRefine.spvcnn import SPVCNN
# from modules.PointRefine.spvcnn_lite import SPVCNN
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
from torchsparse import SparseTensor


class UserRefine(User):
    def __init__(self, arch, data, datadir, outputdir, modeldir, split, bev_res_path: str, save_movable=False):

        super(UserRefine, self).__init__(arch, data, datadir, outputdir, modeldir, split,
                                         point_refine=True, save_movable=save_movable, bev_res_path=bev_res_path)

    def infer(self):
        coarse, reproj, refine = [], [], []

        if self.split == 'valid':
            self.infer_subset(loader=self.parser.get_valid_set(),
                              to_orig_fn=self.parser.to_original,
                              coarse=coarse, reproj=reproj, refine=refine)
        elif self.split == 'train':
            self.infer_subset(loader=self.parser.get_train_set(),
                              to_orig_fn=self.parser.to_original,
                              coarse=coarse, reproj=reproj, refine=refine)
        elif self.split == 'test':
            self.infer_subset(loader=self.parser.get_test_set(),
                              to_orig_fn=self.parser.to_original,
                              coarse=coarse, reproj=reproj, refine=refine)
        elif self.split is None:
            self.infer_subset(loader=self.parser.get_train_set(),
                              to_orig_fn=self.parser.to_original,
                              coarse=coarse, reproj=reproj, refine=refine)
            self.infer_subset(loader=self.parser.get_valid_set(),
                              to_orig_fn=self.parser.to_original,
                              coarse=coarse, reproj=reproj, refine=refine)
            self.infer_subset(loader=self.parser.get_test_set(),
                              to_orig_fn=self.parser.to_original,
                              coarse=coarse, reproj=reproj, refine=refine)
        else:
            raise NotImplementedError

        print(f"Mean Coarse inference time:{'%.8f'%np.mean(coarse)}\t std:{'%.8f'%np.std(coarse)}")
        print(f"Mean Reproject inference time:{'%.8f'%np.mean(reproj)}\t std:{'%.8f'%np.std(reproj)}")
        print(f"Mean Refine inference time:{'%.8f'%np.mean(refine)}\t std:{'%.8f'%np.std(refine)}")
        print(f"Total Frames: {len(coarse)}")
        print("Finished Infering")

        return

    def infer_subset(self, loader, to_orig_fn, coarse, reproj, refine):

        # switch to evaluate mode
        self.model.eval()
        self.refine_module.eval()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():

            end = time.time()

            for i, (proj_in, proj_mask, _, _, path_seq, path_name,
                    p_x, p_y, proj_range, unproj_range, _, unproj_xyz, _, _, npoints, polar_data, p2r_matrix)\
                    in enumerate(tqdm(loader, ncols=80)):

                # first cut to rela size (batch size one allows it)
                p_x = p_x[0, :npoints]
                p_y = p_y[0, :npoints]
                proj_range = proj_range[0, :npoints]
                unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]
                points_xyz = unproj_xyz[0, :npoints]

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
                proj_output, last_feature, movable_proj_output, _ = self.model(proj_in,  cat_bev_res_fea, unq, unq_inv, batch_size, cur_dev, p2r_matrix, train_mode=False)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                coarse.append(res)

                if self.save_movable:
                    movable_proj_argmax = movable_proj_output[0].argmax(dim=0)
                    if self.post:
                        movable_unproj_argmax = self.post(proj_range, unproj_range,
                                                          movable_proj_argmax, p_x, p_y)
                    else:
                        movable_unproj_argmax = movable_proj_argmax[p_y, p_x]

                end = time.time()
                # print(f"CoarseModule seq {path_seq} scan {path_name} in {res} sec")

                """ Reproject 2D features to 3D based on indices and form sparse Tensor"""
                points_feature = last_feature[0, :, p_y, p_x]
                coords = np.round(points_xyz[:, :3].cpu().numpy() / 0.05)
                coords -= coords.min(0, keepdims=1)
                coords, indices, inverse = sparse_quantize(coords, return_index=True, return_inverse=True)
                coords = torch.tensor(coords, dtype=torch.int, device='cuda')
                feats = points_feature.permute(1,0)[indices] #torch.tensor(, dtype=torch.float)
                inputs = SparseTensor(coords=coords, feats=feats)
                inputs = sparse_collate([inputs]).cuda()
                """"""""""""""""""""""""

                # measure elapsed time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                reproj.append(res)
                end = time.time()
                # print(f"DataConvert seq {path_seq} scan {path_name} in {res} sec")

                """ Input to PointHead, refine prediction """
                predict = self.refine_module(inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                refine.append(res)
                # print(f"RefineModule seq {path_seq} scan {path_name} in {res} sec")

                predict = predict[inverse] #.permute(1,0)
                unproj_argmax = predict.argmax(dim=1)

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
