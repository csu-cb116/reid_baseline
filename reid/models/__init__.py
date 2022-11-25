# from __future__ import absolute_import
#
# from .resnet import *
# from .vip.Vip import vip_tiny, vip_small, vip_medium, vip_base
#
# __model_factory = {
#     # image classification models
#     'resnet50': resnet50,
#     'resnet50_fc512': resnet50_fc512,
#     'vip_tiny': vip_tiny,
#     'vip_small': vip_small,
#     'vip_medium': vip_medium,
#     'vip_base': vip_base
#
# }
#
#
# def get_names():
#     return list(__model_factory.keys())
#
#
# def init_model(name, *args, **kwargs):
#     if name not in list(__model_factory.keys()):
#         raise KeyError('Unknown model: {}'.format(name))
#     return __model_factory[name](*args, **kwargs)
from .baseline import Baseline