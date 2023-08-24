# -*- coding: utf-8 -*-

"""Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import pdb
import numpy as np
import torch
import torch.nn as nn

from tool.use_torch import modelzoo
import torch.nn.functional as F
from mobilenetv2_backbone import MobileNetV2
PYTORCH_VERSION = torch.__version__


class Normalize(nn.Module):
    """Normalize"""
    def __init__(self, norm=255, mean=(0.408, 0.459, 0.482), std=(0.225, 0.224, 0.229)):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.FloatTensor(mean) * norm)
        self.register_buffer('var', torch.FloatTensor(std) ** 2)
        self.register_buffer('weight', torch.ones(len(mean)) / norm)
        self.register_buffer('bias', torch.zeros(len(mean)))

    def forward(self, inp_tsr):
        out_tsr = nn.functional.batch_norm(
            input=inp_tsr,
            running_mean=self.mean,
            running_var=self.var,
            weight=self.weight,
            bias=self.bias,
            training=False,
            momentum=0.0,
            eps=0.0
        )
        return out_tsr

class DeployModel(nn.Module):
    """Deploy Model"""
    def __init__(self, model, norm=255, mean=(0.408, 0.459, 0.482), std=(0.225, 0.224, 0.229)):
        super(DeployModel, self).__init__()
        self.model = self.load_model(model)
        self.norm = Normalize(norm, mean, std)
        self.softmax = nn.Softmax(dim=-1)

    def load_model(self, model):
        if hasattr(model, 'module'): # nn.DataParallel
            model = model.module
        return model

    def forward(self, inp):
        #out = self.norm(inp)
        out = self.model(inp)
        #out = self.softmax(out)
        return out


def save_model(model, model_path):
    """Save Model"""
    if hasattr(model, 'module'):
        torch.save(model.module, model_path)
    else:
        torch.save(model, model_path)
    return


def load_model(model, model_path):
    """Load Model"""
    model = torch.load(model_path, map_location='cpu')
    return model


def save_model_state_dict(model, model_state_dict_path):
    """Save Model State Dict"""
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), model_state_dict_path)
    else:
        torch.save(model.state_dict(), model_state_dict_path)
    return

def load_model_state_dict(model, model_state_dict_path):
    """Load Model State Dict"""
    # Compatible Setting
    if PYTORCH_VERSION[:3] == '0.2':
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            """New Function in Pytorch 0.4"""
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    # Load State Dict
    state_dict = torch.load(model_state_dict_path, map_location='cpu')
    print("state_dict", state_dict.keys())
    # Compatible Setting
    if PYTORCH_VERSION[:5] < '0.4.1':
        pop_list = []
        for key in state_dict:
            if 'num_batches_tracked' in key:
                pop_list.append(key)
        for key in pop_list:
            state_dict.pop(key)
    # Load Params
    if hasattr(model, 'module'):  # torch.nn.DataParallel
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return

if __name__ == '__main__':
    model_path = '/home/sail/zhang.hongshuang/DTS_pk/models/SynergyNet_checkpoint_epoch_80.pth.tar'
    onnx_model_path = '/home/sail/zhang.hongshuang/DTS_pk/onnx/SynergyNet_MobileNetV2.onnx'
    input_size = [120, 120]

    model = MobileNetV2()
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)['state_dict']
    model_dict = model.state_dict()

    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]

    model.load_state_dict(model_dict, strict=False)

    deploy_model = DeployModel(model)
    dummy_input = torch.randn((1, 3, input_size[0], input_size[1]))
    deploy_model.eval()
    torch.onnx.export(
        model=deploy_model,
        args=dummy_input,
        f=onnx_model_path,
        verbose=False,
    )
