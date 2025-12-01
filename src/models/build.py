import sys
sys.path.append("../src/models")
from swin import BaseSwinUnet
from restormer import BaseRestormer
from bunet import BaseUnet
import torch
import os
# pretrained_dict = torch.load(save_path)
# model_dict = model.state_dict()
#
# # 1. filter out unnecessary keys
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['unet.conv.weight','unet.conv.bias','output.weight','output.bias']}
# # 2. overwrite entries in the existing state dict
# model_dict.update(pretrained_dict)
# # 3. load the new state dict
# model.load_state_dict(pretrained_dict, strict=False)

def model_load_weights(model, load_path):
    if not os.path.exists(load_path):
        print('weights not found, run init')
        return model, False
    try:
        print('try to load weights')
        model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')), strict=False)
        print('weights loaded successfully')
    except:
        print('fail to load weights')
        print('try to load pretrained from denoise weights')
        pretrained_dict = torch.load(load_path, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k not in ['unet.conv.weight', 'unet.conv.bias', 'output.weight', 'output.bias']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        print('weights loaded successfully')
    return model, True

def build_model(model, problem, activation=None):
    if model == 'restormer':
        if problem == 'denoise':
            model = BaseRestormer(inp_channels=1, out_channels=1, dim=24, activation=activation)
        elif problem == 'firstbreak':
            model = BaseRestormer(inp_channels=1, out_channels=2, dim=24)
        else:
            raise ValueError('Undefined problem!')
    elif model == 'swin':
        if problem == 'denoise':
            model = BaseSwinUnet(in_chans=1, num_classes=1, embed_dim=48, activation=activation)
        elif problem == 'firstbreak':
            model = BaseSwinUnet(in_chans=1, num_classes=2, embed_dim=48)
        else:
            raise ValueError('Undefined problem!')
    elif model == 'unet':
        if problem == 'denoise':
            model = BaseUnet(in_channels=1, out_channels=1, activation=activation)
        elif problem == 'firstbreak':
            model = BaseUnet(in_channels=1, out_channels=2)
        else:
            raise ValueError('Undefined problem!')
    else:
        raise ValueError('Undefined model!')
    return model