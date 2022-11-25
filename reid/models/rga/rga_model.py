# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch import nn

from reid.models.rga.models_utils.rga_branches import RGA_Branch


# ===============
#    RGA Model 
# ===============

class ResNet50_RGA_Model(nn.Module):
    '''
    Backbone: ResNet-50 + RGA modules.
    '''

    def __init__(self, num_feat=2048, height=256, width=128,
                 dropout=0, num_classes=0, last_stride=1, branch_name='rgasc', scale=8, d_scale=8):
        super(ResNet50_RGA_Model, self).__init__()
        self.num_feat = num_feat
        self.dropout = dropout
        self.num_classes = num_classes
        self.branch_name = branch_name
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

    def forward(self, inputs):
        # im_input = inputs[0]
        feat_ = self.backbone(inputs)
        return feat_

    def load_param(self, model_path):
        self.backbone.load_param(model_path)
