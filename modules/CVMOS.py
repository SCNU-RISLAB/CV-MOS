# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from einops import rearrange, repeat
from utils.utils import *

from modules.BaseBlocks import MetaKernel, ResContextBlock, ResBlock, UpBlock
from modules.motionbev_basic_blocks import inconv, down


class B2R_flow(nn.Module):
    def __init__(self, fea_dim, data_type):
        super(B2R_flow, self).__init__()
        self.fea_dim = fea_dim
        self.data_type = data_type

        self.fusion = nn.Sequential(
            nn.Conv2d(fea_dim * 2, fea_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(fea_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fea_dim),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fea_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, flow_matrix, range_fea, polar_fea):
        """
        range_fea: [N, C1, 64, 2048]
        polar_fea: [N, C2, 480, 360]
        flow_matrix: [N, 64, 2048, 2], need to be [-1, 1] for grid sample
        """
        # rescale the flow matrix
        _, _, H, W = range_fea.shape
        N, C, _, _ = polar_fea.shape
        # because for F.grid_sample, i,j,k index w,h,d (i.e., reverse order)
        flow_matrix = torch.flip(flow_matrix, dims=[-1])
        flow_matrix_scaled = F.interpolate(rearrange(flow_matrix, "b h w c -> b c h w").float(),
                                           (H, W), mode='nearest')  # N*2*H*W  通过F.interpolate将flow_matrix缩放到采样后的尺寸
        flow_matrix_scaled = rearrange(flow_matrix_scaled, "b c h w -> b h w c")  # N*H*W*2
        # https://blog.csdn.net/weixin_45657478/article/details/128080374  关于F.grid_sample的用法
        flow_fea = F.grid_sample(input=polar_fea, grid=flow_matrix_scaled, padding_mode='zeros',
                                 align_corners=False)  # N*C*H*W

        fea = torch.cat((range_fea, flow_fea), dim=1)
        res = self.fusion(fea)
        res = res * self.attention(res)
        fea = range_fea + res

        return fea


class CVMOS(nn.Module):
    def __init__(self,
                 nclasses,
                 movable_nclasses,
                 params,
                 num_batch=None,
                 point_refine=None,
                 dilation: int = 1,
                 input_batch_norm: bool = True,
                 group_conv: bool = False,
                 circular_padding: bool = False
                 ):
        super().__init__()
        self.nclasses = nclasses
        self.use_attention = "MGA"
        self.point_refine = point_refine

        self.range_channel = 5
        print("Channel of range image input = ", self.range_channel)
        print("Number of residual images input = ", params['train']['n_input_scans'])

        self.downCntx = ResContextBlock(self.range_channel, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)
        print("params['train']['batch_size']", params['train']['batch_size'])
        self.metaConv = MetaKernel(num_batch=int(params['train']['batch_size']) if num_batch is None else num_batch,
                                   feat_height=params['dataset']['sensor']['img_prop']['height'],
                                   feat_width=params['dataset']['sensor']['img_prop']['width'],
                                   coord_channels=self.range_channel)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4))
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False, kernel_size=(2, 4))

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        self.range_upBlock1 = UpBlock(256, 128, 0.2)
        self.range_upBlock2 = UpBlock(128, 64, 0.2)
        self.range_upBlock3 = UpBlock(64, 32, 0.2)
        # self.range_upBlock4 = UpBlock(64, 32, 0.2, drop_out=False)

        # Context Block for residual image
        self.RI_downCntx = ResContextBlock(params['train']['n_input_scans'], 32)
        # self.RI_downCntx2 = ResContextBlock(32, 32)
        # self.RI_downCntx3 = ResContextBlock(32, 32)

        self.RI_resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4))
        self.RI_resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock5 = ResBlock(2 * 4 * 32, 4 * 4 * 32, 0.2, pooling=False, kernel_size=(2, 4))

        self.logits3 = nn.Conv2d(32, nclasses, kernel_size=(1, 1))
        self.movable_logits = nn.Conv2d(32, movable_nclasses, kernel_size=(1, 1))

        if self.use_attention == "MGA":
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.conv1x1_conv1_channel_wise = nn.Conv2d(32, 32, 1, bias=True)
            self.conv1x1_conv1_spatial = nn.Conv2d(32, 1, 1, bias=True)

            self.conv1x1_layer0_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
            self.conv1x1_layer0_spatial = nn.Conv2d(64, 1, 1, bias=True)

            self.conv1x1_layer1_channel_wise = nn.Conv2d(128, 128, 1, bias=True)
            self.conv1x1_layer1_spatial = nn.Conv2d(128, 1, 1, bias=True)

            self.conv1x1_layer2_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
            self.conv1x1_layer2_spatial = nn.Conv2d(256, 1, 1, bias=True)

            self.conv1x1_layer3_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
            self.conv1x1_layer3_spatial = nn.Conv2d(256, 1, 1, bias=True)

            self.conv1x1_layer4_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
            self.conv1x1_layer4_spatial = nn.Conv2d(256, 1, 1, bias=True)
        else:
            pass  # raise NotImplementedError

        bev_res_fea_dim = 8
        self.polar_PPmodel = nn.Sequential(
            nn.BatchNorm1d(bev_res_fea_dim),

            nn.Linear(bev_res_fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 64),

        )

        self.polar_compress = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # for polar net
        bev_channels = [64, 128, 256, 512]
        self.polar_inc = inconv(bev_channels[0] // 2, bev_channels[0] // 2, dilation, input_batch_norm,
                                circular_padding)
        self.polar_down1 = down(bev_channels[0] // 2, bev_channels[0], dilation, group_conv, circular_padding)
        self.polar_down2 = down(bev_channels[0], bev_channels[1], dilation, group_conv, circular_padding)
        self.polar_down3 = down(bev_channels[1], bev_channels[2], dilation, group_conv, circular_padding)
        self.polar_down4 = down(bev_channels[2], bev_channels[2], dilation, group_conv, circular_padding)

        # flow
        self.flow_down0_p2r = B2R_flow(bev_channels[0], torch.float32)
        self.flow_down1_p2r = B2R_flow(bev_channels[1], torch.float32)
        self.flow_down2_p2r = B2R_flow(bev_channels[2], torch.float32)
        self.flow_down3_p2r = B2R_flow(bev_channels[2], torch.float32)

    @staticmethod
    def polar_preprocess(pt_fea, xy_ind, num_pt_each):
        cur_dev = pt_fea[0].get_device()
        batch_size = len(pt_fea)

        # get the valid part
        pt_fea_valid = []
        xy_ind_valid = []
        for i_batch in range(len(pt_fea)):
            pt_fea_valid.append(pt_fea[i_batch, :num_pt_each[i_batch], :])
            xy_ind_valid.append(xy_ind[i_batch, :num_pt_each[i_batch], :])

        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind_valid[i_batch], (1, 0), 'constant', value=i_batch))

        cat_pt_fea = torch.cat(pt_fea_valid, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)
        return cat_pt_fea, unq, unq_inv, batch_size, cur_dev

    @staticmethod
    def polar_reformat_data(processed_pooled_data, unq, batch_size, cur_dev, data_type):
        # stuff pooled data into 4D tensor
        out_data_dim = [batch_size, *[480, 360, processed_pooled_data.shape[-1]]]
        out_data = torch.zeros(out_data_dim, dtype=data_type).to(cur_dev)
        out_data[unq[:, 0], unq[:, 1], unq[:, 2], :] = processed_pooled_data
        out_data = rearrange(out_data, "b h w c -> b c h w")
        # if self.local_pool_op is not None:
        #     out_data = self.local_pool_op(out_data)

        return out_data

    def encoder_attention_module_MGA_tmc(self, img_feat, flow_feat, conv1x1_channel_wise, conv1x1_spatial):
        """
            flow_feat_map:  [bsize, 1, h, w]
            feat_vec:       [bsize, channel, 1, 1]
            channel_attentioned_img_feat:  [bsize, channel, h, w]
        """
        # spatial attention
        flow_feat_map = conv1x1_spatial(flow_feat)
        flow_feat_map = nn.Sigmoid()(flow_feat_map)
        spatial_attentioned_img_feat = flow_feat_map * img_feat

        # channel-wise attention
        feat_vec = self.avg_pool(spatial_attentioned_img_feat)
        feat_vec = conv1x1_channel_wise(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec

        final_feat = channel_attentioned_img_feat + img_feat
        return final_feat

    def forward(self, x, cat_bev_res_fea, unq, unq_inv, batch_size, cur_dev, p2r_matrix, train_mode: bool = False):
        """
            x: shape [bs, c, h, w],  c = range image channel + num of residual images
            *_downCntx:[bs, .., h, w]
            RI_down0c: [bs, c', h/2, w/2]       RI_down0b:  [bs, c', h, w]
            RI_down1c: [bs, c'', h/4, w/4]      RI_down1b:  [bs, c'', h/2, w/2]
            RI_down2c: [bs, c'', h/8, w/8]      RI_down2b:  [bs, c'', h/4, w/4]
            RI_down3c: [bs, c'', h/16, w/16]    RI_down3b:  [bs, c'', h/8, w/8]
            up4e: [bs, .., h/8, w/8]
            up3e: [bs, .., h/4, w/4]
            up2e: [bs, .., h/2, w/2]
            up1e: [bs, .., h, w]
            logits: [bs, num_class, h, w]
        """
        ############## bev residual data process ############################################################
        # cat_bev_res_fea, unq, unq_inv, batch_size, cur_dev = self.polar_preprocess(bev_res_fea, xy_ind, num_pt_each)
        processed_cat_bev_res_fea = self.polar_PPmodel(cat_bev_res_fea)
        # torch scatter does not support float16
        pooled_data, pooled_idx = torch_scatter.scatter_max(processed_cat_bev_res_fea, unq_inv, dim=0)
        processed_pooled_data = self.polar_compress(pooled_data)

        # 将特征放到网格中
        out_data = self.polar_reformat_data(processed_pooled_data, unq, batch_size, cur_dev, data_type=torch.float32)
        _, _, polar_h, polar_w = out_data.shape

        ###################### bev down sample ###########################
        polar_x = self.polar_inc(out_data)
        polar_down0 = self.polar_down1(polar_x)  # 1/2
        polar_down1 = self.polar_down2(polar_down0)  # 1/4
        polar_down2 = self.polar_down3(polar_down1)  # 1/8
        polar_down3 = self.polar_down4(polar_down2)  # 1/16

        #######################################################################################################
        # print("x shape is {}".format(x.shape))
        # split the input data to range image (5 channel) and residual images
        current_range_image = x[:, :self.range_channel, :, :]
        residual_images = x[:, self.range_channel:, :, :]

        # print("residual_images {}".format(residual_images.shape))
        ###### the Encoder for residual image #############
        RI_downCntx = self.RI_downCntx(residual_images)
        # print("RI_downCntx {}".format(RI_downCntx.shape))

        ###### the Encoder for range image ######       # range (3, 5, 64, 2048)
        downCntx = self.downCntx(current_range_image)  # (3, 32, 64, 2048)
        # Use MetaKernel to capture more spatial information
        downCntx = self.metaConv(data=downCntx,
                                 coord_data=current_range_image,
                                 data_channels=downCntx.size()[1],
                                 coord_channels=current_range_image.size()[1],
                                 kernel_size=3)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        Range_down0c, Range_down0b = self.RI_resBlock1(downCntx)  # (64, 32, 1024), (64, 64, 2048)
        Range_down1c, Range_down1b = self.RI_resBlock2(Range_down0c)  # (128, 16, 512), (128, 32, 1024)
        Range_down2c, Range_down2b = self.RI_resBlock3(Range_down1c)  # (256, 8, 256) , (256, 16, 512)
        Range_down3c, Range_down3b = self.RI_resBlock4(Range_down2c)  # (256, 4, 128) , (256, 8, 256)
        # Range_down4c = self.RI_resBlock5(Range_down3c)                  # (512, 4, 128)

        ###### Bridging two specific branches using MotionGuidedAttention ######
        if self.use_attention == "MGA":
            downCntx = self.encoder_attention_module_MGA_tmc(RI_downCntx, downCntx, self.conv1x1_conv1_channel_wise,
                                                             self.conv1x1_conv1_spatial)
        elif self.use_attention == "Add":
            downCntx += RI_downCntx
        down0c, down0b = self.resBlock1(downCntx)

        down0c = self.flow_down0_p2r(p2r_matrix.clone(), down0c, polar_down0)  # (64, 32, 512)  (64, 240, 180)
        if self.use_attention == "MGA":
            down0c = self.encoder_attention_module_MGA_tmc(down0c, Range_down0c, self.conv1x1_layer0_channel_wise,
                                                           self.conv1x1_layer0_spatial)
        elif self.use_attention == "Add":
            down0c += Range_down0c
        down1c, down1b = self.resBlock2(down0c)

        down1c = self.flow_down1_p2r(p2r_matrix.clone(), down1c, polar_down1)
        if self.use_attention == "MGA":
            down1c = self.encoder_attention_module_MGA_tmc(down1c, Range_down1c, self.conv1x1_layer1_channel_wise,
                                                           self.conv1x1_layer1_spatial)
        elif self.use_attention == "Add":
            down1c += Range_down1c
        down2c, down2b = self.resBlock3(down1c)

        down2c = self.flow_down2_p2r(p2r_matrix.clone(), down2c, polar_down2)
        if self.use_attention == "MGA":
            down2c = self.encoder_attention_module_MGA_tmc(down2c, Range_down2c, self.conv1x1_layer2_channel_wise,
                                                           self.conv1x1_layer2_spatial)
        elif self.use_attention == "Add":
            down2c += Range_down2c
        down3c, down3b = self.resBlock4(down2c)  # (1, 256, 4, 128) (1, 256, 8, 256)

        down3c = self.flow_down3_p2r(p2r_matrix.clone(), down3c, polar_down3)
        if self.use_attention == "MGA":
            down3c = self.encoder_attention_module_MGA_tmc(down3c, Range_down3c, self.conv1x1_layer3_channel_wise,
                                                           self.conv1x1_layer3_spatial)
        elif self.use_attention == "Add":
            down3c += Range_down3c
        down5c = self.resBlock5(down3c)  # (1, 256, 4, 128)

        ###### the Decoder, same as SalsaNext ######
        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)

        logits = self.logits3(up1e)
        logits = F.softmax(logits, dim=1)

        if train_mode:  # 只有训练模式才走movable的上采样
            range_up4e = self.range_upBlock1(Range_down3b, Range_down2b)
            range_up3e = self.range_upBlock2(range_up4e, Range_down1b)
            range_up2e = self.range_upBlock3(range_up3e, Range_down0b)
            # range_up1e = self.range_upBlock1(range_up2e, Range_down0b)

            movable_logits = self.movable_logits(range_up2e)
            movable_logits = F.softmax(movable_logits, dim=1)
        else:
            movable_logits, range_up2e = deepcopy(logits.detach()), deepcopy(up1e.detach())

        return logits, up1e, movable_logits, range_up2e


if __name__ == '__main__':
    from torchinfo import summary

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input = torch.randn(1, 13, 64, 2048)
    arch_cfg = load_yaml('../train_yaml/mos_coarse_stage.yml')
    # print(arch_cfg)
    model = CVMOS(nclasses=3, movable_nclasses=3, params=arch_cfg, num_batch=1)
    # print(model)
    output, movable_output = model(input)  # (1, 3, 64, 2048)

    summary(model, (1, 13, 64, 2048))

    print(output.shape)
    print(movable_output.shape)