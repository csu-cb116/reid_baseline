# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch import nn

from reid.models.hpm.hpm_module import global_pcb, spp_vertical, weight_init, pcb_block
from reid.models.rga.models_utils.rga_branches import RGA_Branch


# ===============
#    RGA Model
# ===============

class RGA_Pyramid_Model(nn.Module):
    '''
    Backbone: ResNet-50 + RGA modules.
    '''

    def __init__(self, num_classes, num_stripes=8, local_conv_out_channels=256, num_feat=2048, height=256, width=128,
                 dropout=0, last_stride=1, branch_name='rgasc', scale=8, d_scale=8, avg=False):
        super(RGA_Pyramid_Model, self).__init__()
        self.num_feat = num_feat
        self.dropout = dropout
        self.num_classes = num_classes
        self.branch_name = branch_name
        self.num_stripes = num_stripes

        print('Num of features: {}.'.format(self.num_feat))

        if 'rgasc' in branch_name:
            spa_on = True
            cha_on = True
        elif 'rgas' in branch_name:
            spa_on = True
            cha_on = False
        elif 'rgac' in branch_name:
            spa_on = False
            cha_on = True
        else:
            raise NameError

        self.backbone = RGA_Branch(last_stride=last_stride,
                                   spa_on=spa_on, cha_on=cha_on, height=height, width=width,
                                   s_ratio=scale, c_ratio=scale, d_ratio=d_scale)
        self.num_ftrs = list(self.backbone.layer4)[-1].conv1.in_channels
        # global
        self.global_pooling = nn.AdaptiveMaxPool2d(1)
        self.global_conv = nn.Conv2d(self.num_ftrs, local_conv_out_channels, 1, bias=False)
        self.global_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.global_relu = nn.ReLU(inplace=True)
        self.global_fc = nn.Linear(local_conv_out_channels, num_classes, bias=False)

        weight_init(self.global_conv)
        weight_init(self.global_bn)
        weight_init(self.global_fc)

        # 2x
        self.pcb2_pool_list, self.pcb2_conv_list, self.pcb2_batchnorm_list, self.pcb2_relu_list, self.pcb2_fc_list = pcb_block(
            self.num_ftrs, 2, local_conv_out_channels, num_classes, avg)
        # 4x
        self.pcb4_pool_list, self.pcb4_conv_list, self.pcb4_batchnorm_list, self.pcb4_relu_list, self.pcb4_fc_list = pcb_block(
            self.num_ftrs, 4, local_conv_out_channels, num_classes, avg)
        # 8x
        self.pcb8_pool_list, self.pcb8_conv_list, self.pcb8_batchnorm_list, self.pcb8_relu_list, self.pcb8_fc_list = pcb_block(
            self.num_ftrs, 8, local_conv_out_channels, num_classes, avg)

    def forward(self, inputs):
        # im_input = inputs[0]
        feats = self.backbone(inputs)
        assert feats.size(2) % self.num_stripes == 0
        feat_list, logits_list = global_pcb(feats, self.global_pooling, self.global_conv, self.global_bn,
                                            self.global_relu, self.global_fc, [], [])
        feat_list, logits_list = spp_vertical(feats, self.pcb2_pool_list, self.pcb2_conv_list,
                                              self.pcb2_batchnorm_list, self.pcb2_relu_list, self.pcb2_fc_list, 2,
                                              feat_list, logits_list)
        feat_list, logits_list = spp_vertical(feats, self.pcb4_pool_list, self.pcb4_conv_list,
                                              self.pcb4_batchnorm_list, self.pcb4_relu_list, self.pcb4_fc_list, 4,
                                              feat_list, logits_list)

        # feat_list, logits_list = spp_vertical(feats, self.pcb8_pool_list, self.pcb8_conv_list,
        #                                       self.pcb8_batchnorm_list, self.pcb8_relu_list, self.pcb8_fc_list, 8,
        #                                       feat_list, logits_list)

        return logits_list, feat_list

    def load_param(self, model_path):
        self.backbone.load_param(model_path)
