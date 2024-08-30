#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import datetime
import os
import time
import imp
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from modules.trainer import Trainer
import __init__ as booger

import torch.optim as optim
from tensorboardX import SummaryWriter as Logger
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import warmupLR

# from modules.SalsaNextWithMotionAttention import SalsaNextWithMotionAttention
# from modules.loss.Lovasz_Softmax import Lovasz_softmax, Lovasz_softmax_PointCloud
from modules.tools import AverageMeter, iouEval, save_checkpoint, show_scans_in_training, save_to_txtlog, make_log_img

from modules.PointRefine.attn_spvcnn import SPVCNN
# from modules.PointRefine.spvcnn_lite import SPVCNN
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
from torchsparse import SparseTensor


class TrainerRefine(Trainer):
    def __init__(self, arch, data, datadir, logdir, bev_res_path, path=None):

        super(TrainerRefine, self).__init__(arch, data, datadir, logdir, path, bev_res_path, point_refine=True)

        """ New variables for the PointRefine module """
        net_config = {'num_classes': self.parser.get_n_classes(),
                      'cr': 1.0, 'pres': 0.05, 'vres': 0.05}
        self.refine_module = SPVCNN(num_classes=net_config['num_classes'],
                                    cr=net_config['cr'],
                                    pres=net_config['pres'],
                                    vres=net_config['vres'])

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 0:
                self.refine_module.cuda()
            if torch.cuda.device_count() > 1:
                self.refine_module = nn.DataParallel(self.refine_module)

        self.set_refine_optim_scheduler()
        """"""""""""""""""""""""""""""""""""""""""""""""

    def set_refine_optim_scheduler(self):
        """
            Used to set the optimizer and scheduler for PointRefine module
        """
        self.refine_optimizer = optim.Adam(
            [{'params': self.refine_module.parameters()}], self.arch["train"]["lr"])

        self.refine_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.refine_optimizer, 0.95)

    def train(self):

        self.init_evaluator()

        # train for n epochs
        for epoch in range(self.epoch, self.arch["train"]["max_epochs"]):

            # train for 1 epoch
            acc, iou, loss, update_mean, hetero_l = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                                     model=self.model,
                                                                     criterion=self.criterion,
                                                                     optimizer=self.optimizer,
                                                                     epoch=epoch,
                                                                     evaluator=self.evaluator,
                                                                     scheduler=self.scheduler,
                                                                     color_fn=self.parser.to_color,
                                                                     report=self.arch["train"]["report_batch"],
                                                                     show_scans=self.arch["train"]["show_scans"])

            # update the info dict and save the training checkpoint
            self.update_training_info(epoch, acc, iou, loss, update_mean, hetero_l)

            rand_img = []
            # evaluate on validation set
            if epoch % self.arch["train"]["report_epoch"] == 0 and epoch >= 1:
                acc, iou, loss, rand_img, hetero_l = self.validate(val_loader=self.parser.get_valid_set(),
                                                                   model=self.model,
                                                                   criterion=self.criterion,
                                                                   evaluator=self.evaluator,
                                                                   class_func=self.parser.get_xentropy_class_string,
                                                                   color_fn=self.parser.to_color,
                                                                   save_scans=self.arch["train"]["save_scans"])

                self.update_validation_info(epoch, acc, iou, loss, hetero_l)

            # save to tensorboard log
            Trainer.save_to_tensorboard(logdir=self.logdir,
                                        logger=self.tb_logger,
                                        info=self.info,
                                        epoch=epoch,
                                        w_summary=self.arch["train"]["save_summary"],
                                        model=self.model_single,
                                        img_summary=self.arch["train"]["save_scans"],
                                        imgs=rand_img)

        print('Finished Training')

        return

    def train_epoch(self, train_loader, model, criterion, optimizer,
                    epoch, evaluator, scheduler, color_fn, report=10,
                    show_scans=False):

        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        update_ratio_meter = AverageMeter()

        # empty the cache to train now
        # if self.gpu:
        #     torch.cuda.empty_cache()

        # switch to train mode
        start_epoch = 5
        # switch to train mode
        if epoch <= start_epoch:
            model.eval()
        else:
            model.train()

        self.refine_module.train()

        end = time.time()

        for i, (in_vol, proj_mask, all_proj_labels, unproj_labels, path_seq, path_name,
                p_x, p_y, proj_range, unproj_range, _, unproj_xyz, _, unproj_remissions, n_points, polar_data,
                p2r_matrix) \
                in enumerate(train_loader):

            proj_labels, movable_proj_labels = all_proj_labels

            # measure data loading time
            self.data_time_t.update(time.time() - end)

            ########################## polar #######################################
            train_grid, num_pt, bev_residual_data = polar_data
            p2r_matrix = p2r_matrix[:, :, :, :2]
            unproj_labels = unproj_labels.cuda().long()
            xy_ind = train_grid[:, :, :-1]  # bev投影要把z切掉

            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()
                movable_proj_labels = movable_proj_labels.cuda().long()
                p2r_matrix = p2r_matrix.cuda()

                ### bev
                bev_residual_data = bev_residual_data.cuda()
                xy_ind = xy_ind.cuda()

            # compute output
            # output = model(in_vol)
            cat_bev_res_fea, unq, unq_inv, batch_size, cur_dev = self.polar_preprocess(bev_residual_data, xy_ind,
                                                                                       num_pt)
            if epoch <= start_epoch:
                with torch.no_grad():
                    output, last_feature, movable_output, movalbe_last_feature = model(in_vol, cat_bev_res_fea, unq, unq_inv, batch_size, cur_dev, p2r_matrix, train_mode=False)
            else:
                output, last_feature, movable_output, movalbe_last_feature = model(in_vol, cat_bev_res_fea, unq, unq_inv, batch_size, cur_dev, p2r_matrix, train_mode=False)

            tmp_pred = []
            tmp_labels = []
            for j in range(len(n_points)):
                _npoints = n_points[j]
                _px = p_x[j, :_npoints]
                _py = p_y[j, :_npoints]
                _unproj_labels = unproj_labels[j, :_npoints]
                _points_xyz = unproj_xyz[j, :_npoints]

                # put in original pointcloud using indexes
                _points_feature = last_feature[j, :, _py, _px]

                # Filter out duplicate points
                coords = np.round(_points_xyz[:, :3].cpu().numpy() / 0.05)
                coords -= coords.min(0, keepdims=1)
                coords, indices, inverse = sparse_quantize(coords, return_index=True, return_inverse=True)
                coords = torch.tensor(coords, dtype=torch.int, device='cuda')

                feats = _points_feature.permute(1, 0)[indices]

                inputs = SparseTensor(coords=coords, feats=feats)
                inputs = sparse_collate([inputs]).cuda()

                _predict = self.refine_module(inputs)
                _predict = _predict[inverse].permute(1, 0)

                tmp_pred.append(_predict)
                tmp_labels.append(_unproj_labels)

            predict = torch.cat(tmp_pred, -1).unsqueeze(0)
            unproj_labels = torch.cat(tmp_labels).unsqueeze(0)

            loss_m = criterion(torch.log(predict.clamp(min=1e-8)).double(), unproj_labels).float() + 1.5 * self.ls(predict,
                                                                                                             unproj_labels)

            if epoch <= start_epoch:
                self.refine_optimizer.zero_grad()
            else:
                self.refine_optimizer.zero_grad()
                optimizer.zero_grad()
            if self.n_gpus > 1:
                idx = torch.ones(self.n_gpus).cuda()
                loss_m.backward(idx)
            else:
                loss_m.backward()

            if epoch <= start_epoch:
                self.refine_optimizer.step()
            else:
                self.refine_optimizer.step()
                optimizer.step()

            # measure accuracy and record loss
            loss = loss_m.mean()
            with torch.no_grad():
                evaluator.reset()
                # argmax = output.argmax(dim=1)
                # evaluator.addBatch(argmax, proj_labels)
                argmax = predict.argmax(dim=1)
                evaluator.addBatch(argmax, unproj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()

            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            # get gradient updates and weights, so I can print the relationship of
            # their norms
            # update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]

            if show_scans:
                show_scans_in_training(
                    proj_mask, in_vol, argmax, proj_labels, color_fn)

            if i % self.arch["train"]["report_batch"] == 0:
                str_line = ('Lr: {lr:.3e} | '
                            # 'Update: {umean:.3e} mean,{ustd:.3e} std | '
                            'Epoch: [{0}][{1}/{2}] | '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                            'MovingLoss {moving_loss.val:.4f} ({moving_loss.avg:.4f}) | '
                            'MovingAcc {moving_acc.val:.3f} ({moving_acc.avg:.3f}) | '
                            'MovingIoU {moving_iou.val:.3f} ({moving_iou.avg:.3f}) | [{estim}]').format(
                    epoch, i, len(train_loader), batch_time=self.batch_time_t,
                    data_time=self.data_time_t, moving_loss=losses, moving_acc=acc, moving_iou=iou, lr=lr,
                    # umean=update_mean, ustd=update_std,
                    estim=self.calculate_estimate(epoch, i))
                print(str_line)
                save_to_txtlog(self.logdir, 'log.txt', str_line)

            # step scheduler
            # scheduler.step()
            if i != 0 and i % 1000 == 0:
                self.refine_scheduler.step()
                # evaluate on validation set
                tmp_evaluator = iouEval(self.parser.get_n_classes(), self.device, self.ignore_class)

                _acc, _iou, _loss, rand_img, _hetero_l = self.validate(val_loader=self.parser.get_valid_set(),
                                                                       model=self.model,
                                                                       criterion=self.criterion,
                                                                       evaluator=tmp_evaluator,
                                                                       class_func=self.parser.get_xentropy_class_string,
                                                                       color_fn=self.parser.to_color,
                                                                       save_scans=self.arch["train"]["save_scans"])

                self.update_validation_info(epoch, _acc, _iou, _loss, _hetero_l, i)

            # step scheduler
            scheduler.step()

        return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg, hetero_l.avg

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans=False):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        rand_imgs = []

        # switch to evaluate mode
        model.eval()
        evaluator.reset()
        self.refine_module.eval()

        # empty the cache to infer in high res
        # if self.gpu:
        #     torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, all_proj_labels, unproj_labels, path_seq, path_name,
                    p_x, p_y, proj_range, unproj_range, _, unproj_xyz, _, unproj_remissions, n_points, polar_data,
                    p2r_matrix) \
                    in enumerate(tqdm(val_loader, desc="Validation:", ncols=80)):

                proj_labels, movable_proj_labels = all_proj_labels

                ########################## polar #######################################
                train_grid, num_pt, bev_residual_data = polar_data
                p2r_matrix = p2r_matrix[:, :, :, :2]
                unproj_labels = unproj_labels.cuda().long()
                xy_ind = train_grid[:, :, :-1]  # bev投影要把z切掉

                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda().long()
                    movable_proj_labels = movable_proj_labels.cuda().long()
                    p2r_matrix = p2r_matrix.cuda()

                    ### bev
                    bev_residual_data = bev_residual_data.cuda()
                    xy_ind = xy_ind.cuda()

                # compute output
                # output = model(in_vol)
                cat_bev_res_fea, unq, unq_inv, batch_size, cur_dev = self.polar_preprocess(bev_residual_data, xy_ind,
                                                                                           num_pt)
                with torch.no_grad():
                    output, last_feature, movable_output, movable_last_feature = model(in_vol, cat_bev_res_fea, unq, unq_inv, batch_size, cur_dev, p2r_matrix)

                """ Reproject 2D features to 3D based on indices and form sparse Tensor"""
                tmp_pred = []
                tmp_labels = []
                for j in range(len(n_points)):
                    _npoints = n_points[j]
                    _px = p_x[j, :_npoints]
                    _py = p_y[j, :_npoints]
                    _unproj_labels = unproj_labels[j, :_npoints]
                    _points_xyz = unproj_xyz[j, :_npoints]

                    # put in original pointcloud using indexes
                    _points_feature = last_feature[j, :, _py, _px]

                # Filter out duplicate points
                coords = np.round(_points_xyz[:, :3].cpu().numpy() / 0.05)
                coords -= coords.min(0, keepdims=1)
                coords, indices, inverse = sparse_quantize(coords, return_index=True, return_inverse=True)
                coords = torch.tensor(coords, dtype=torch.int, device='cuda')
                feats = _points_feature.permute(1, 0)[indices]
                # feats = torch.cat((_points_xyz[indices] , _points_feature.permute(1, 0)[indices]), dim=1)
                # feats = torch.cat((_points_xyz[indices], _range_pred.permute(1, 0)[indices], _points_feature.permute(1, 0)[indices]), dim=1)

                inputs = SparseTensor(coords=coords, feats=feats)
                inputs = sparse_collate([inputs]).cuda()

                # _concate_feat = torch.cat((_points_xyz, _range_pred.permute(1, 0)), dim=1).unsqueeze(0) 
                _predict = self.refine_module(inputs)
                _predict = _predict[inverse].permute(1, 0)

                tmp_pred.append(_predict)
                tmp_labels.append(_unproj_labels)

                predict = torch.cat(tmp_pred, -1).unsqueeze(0)
                unproj_labels = torch.cat(tmp_labels).unsqueeze(0)
                """"""""""""""""""""""""

                log_out = torch.log(predict.clamp(min=1e-8))
                jacc = self.ls(predict, unproj_labels)
                # wce = criterion(log_out, proj_labels)
                wce = criterion(log_out.double(), unproj_labels).float()
                loss = wce + jacc

                # measure accuracy and record loss
                # argmax = output.argmax(dim=1)
                # evaluator.addBatch(argmax, proj_labels)
                argmax = predict.argmax(dim=1)
                evaluator.addBatch(argmax, unproj_labels)
                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(), in_vol.size(0))
                wces.update(wce.mean().item(), in_vol.size(0))

                if save_scans:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)
                    rand_imgs.append(out)

                # measure elapsed time
                self.batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            str_line = ("*" * 80 + '\n'
                                   'Validation set:\n'
                                   'Time avg per batch {batch_time.avg:.3f}\n'
                                   'MovingLoss avg {moving_loss.avg:.4f}\n'
                                   'Jaccard avg {jac.avg:.4f}\n'
                                   'WCE avg {wces.avg:.4f}\n'
                                   'MovingAcc avg {moving_acc.avg:.6f}\n'
                                   'MovingIoU avg {moving_iou.avg:.6f}').format(
                batch_time=self.batch_time_e, moving_loss=losses,
                jac=jaccs, wces=wces, moving_acc=acc, moving_iou=iou)
            print(str_line)
            save_to_txtlog(self.logdir, 'log.txt', str_line)

            # print also classwise
            for i, jacc in enumerate(class_jaccard):
                self.info["valid_classes/" + class_func(i)] = jacc
                str_line = 'IoU class {i:} [{class_str:}] = {jacc:.6f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc)
                print(str_line)
                save_to_txtlog(self.logdir, 'log.txt', str_line)
            str_line = '*' * 80
            print(str_line)
            save_to_txtlog(self.logdir, 'log.txt', str_line)

        return acc.avg, iou.avg, losses.avg, rand_imgs, hetero_l.avg

    def update_training_info(self, epoch, acc, iou, loss, update_mean, hetero_l):
        # update info
        self.info["train_update"] = update_mean
        self.info["train_loss"] = loss
        self.info["train_acc"] = acc
        self.info["train_iou"] = iou
        self.info["train_hetero"] = hetero_l

        # remember best iou and save checkpoint
        state = {'epoch': epoch,
                 'main_state_dict': self.model.state_dict(),
                 'refine_state_dict': self.refine_module.state_dict(),
                 'main_optimizer': self.optimizer.state_dict(),
                 'refine_optimizer': self.refine_optimizer.state_dict(),
                 'info': self.info,
                 'scheduler': self.scheduler.state_dict(),
                 'refine_scheduler': self.refine_scheduler.state_dict()}
        try:
            save_checkpoint(state, self.logdir, suffix="")
        except TypeError:
            state.pop('scheduler')
            state.pop('refine_scheduler')
            save_checkpoint(state, self.logdir, suffix="")

        if self.info['train_iou'] > self.info['best_train_iou']:
            print("Best mean iou in training set so far, save model!")
            self.info['best_train_iou'] = self.info['train_iou']
            state = {'epoch': epoch,
                     'main_state_dict': self.model.state_dict(),
                     'refine_state_dict': self.refine_module.state_dict(),
                     'main_optimizer': self.optimizer.state_dict(),
                     'refine_optimizer': self.refine_optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict(),
                     'refine_scheduler': self.refine_scheduler.state_dict()}
            try:
                save_checkpoint(state, self.logdir, suffix="_train_best")
            except TypeError:
                state.pop('scheduler')
                state.pop('refine_scheduler')
                save_checkpoint(state, self.logdir, suffix="_train_best")

    def update_validation_info(self, epoch, acc, iou, loss, hetero_l, iter=None):
        # update info
        self.info["valid_loss"] = loss
        self.info["valid_acc"] = acc
        self.info["valid_iou"] = iou
        self.info['valid_heteros'] = hetero_l

        # remember best iou and save checkpoint
        if self.info['valid_iou'] > self.info['best_val_iou']:
            str_line = ("Best mean iou in validation so far, save model!\n" + "*" * 80)
            print(str_line)
            save_to_txtlog(self.logdir, 'log.txt', str_line)

            self.info['best_val_iou'] = self.info['valid_iou']

            # save the weights!
            state = {'epoch': epoch,
                     'main_state_dict': self.model.state_dict(),
                     'refine_state_dict': self.refine_module.state_dict(),
                     'main_optimizer': self.optimizer.state_dict(),
                     'refine_optimizer': self.refine_optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict(),
                     'refine_scheduler': self.refine_scheduler.state_dict()
                     }
            try:
                save_checkpoint(state, self.logdir, suffix="_2stage_valid_best")
            except TypeError:
                state.pop('scheduler')
                state.pop('refine_scheduler')
                save_checkpoint(state, self.logdir, suffix="_2stage_valid_best")
            if iter is None:
                save_checkpoint(state, self.logdir, suffix=f"_2stage_valid_best_{epoch}")
            else:
                save_checkpoint(state, self.logdir, suffix=f"_2stage_valid_best_{epoch}_it{iter}")

