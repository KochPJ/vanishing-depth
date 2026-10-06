import os
import sys


if __name__ == '__main__':
    sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from utils.stuff import load_fitting_state_dict
from model.DFormer.models.encoders.DFormer import DFormer_Tiny, DFormer_Small, DFormer_Base, DFormer_Large
from model.DFormer.models.encoders.DFormerv2 import DFormerv2_S, DFormerv2_B, DFormerv2_L
import torch.nn.functional as F


class DformerWrapper(nn.Module):
    def __init__(self, model_size='base', pretrained_path=None, upsample_outs=False):
        super().__init__()

        encoder = {
            'tiny': DFormer_Tiny, 
            'small': DFormer_Small,
            'base': DFormer_Base,
            'DFormer_Large': DFormer_Large
        }[model_size]

        self.arch = encoder()
        self.model_size = model_size
        self.interm_channels = [self.arch.dims[0], self.arch.dims[1], self.arch.dims[2], self.arch.dims[3]]
        self.upsample_outs = upsample_outs
        self.patch_size = (14, 14)

        if pretrained_path:
            sd = torch.load(pretrained_path)['state_dict']
           
            # This is normal according to the authors of deformer https://github.com/VCIP-RGBD/DFormer/issues/36
            # This is about some cls heads used in the pretraining and not needed for segmentation anymore
            '''                   
            wrong key stages.3.1.layer_scale_1_e torch.Size([256])
            wrong key stages.3.1.layer_scale_2_e torch.Size([256])
            wrong key stages.3.1.attn.proj_e.weight torch.Size([256, 1024])
            wrong key stages.3.1.attn.proj_e.bias torch.Size([256])
            wrong key stages.3.1.mlp_e2.norm.weight torch.Size([256])
            wrong key stages.3.1.mlp_e2.norm.bias torch.Size([256])
            wrong key stages.3.1.mlp_e2.fc1.weight torch.Size([1024, 256])
            wrong key stages.3.1.mlp_e2.fc1.bias torch.Size([1024])
            wrong key stages.3.1.mlp_e2.pos.weight torch.Size([1024, 1, 3, 3])
            wrong key stages.3.1.mlp_e2.pos.bias torch.Size([1024])
            wrong key stages.3.1.mlp_e2.fc2.weight torch.Size([256, 1024])
            wrong key stages.3.1.mlp_e2.fc2.bias torch.Size([256])
            wrong key pred.weight torch.Size([1000, 768])
            wrong key pred.bias torch.Size([1000])

            '''
            self.arch = load_fitting_state_dict(self.arch, sd)
            
    def forward(self, x, xd=None, depth_scales=None, masks=None):

        #if xd is not None:
        #    if xd.shape[1] == 1:
        #        xd = xd.repeat((1,3,1,1))
        #else:
        #    xd = torch.zeros(x.shape, dtype=x.dtype, device=x.device)

        outs = self.arch(x, xd)

        if self.upsample_outs:
            for i in range(len(outs)):
                h, w = outs[i].shape[-2:]
                outs[i] = F.interpolate(outs[i], (h*2, w*2), None, 'bilinear', None, recompute_scale_factor=None)

        #print('dformer outs', [o.shape for o in outs])
        outs = [None] + outs + [None] # pad for consisten output (first = input size -> discarded anyways) last = cls token -> not needed and not used anyways

        return outs



class DformerV2Wrapper(nn.Module):
    def __init__(self, model_size='base', pretrained_path=None, upsample_outs=False, syncbn=False, drop_path_rate=0.1):
        super().__init__()

        encoder = {
            'small': DFormerv2_S,
            'base': DFormerv2_B,
            'DFormer_Large': DFormerv2_L
        }[model_size]

        self.model_size = model_size
        if syncbn:
            norm_cfg = dict(type="SyncBN", requires_grad=True)
        else:
            norm_cfg = dict(type="BN", requires_grad=True)
        self.arch = encoder(drop_path_rate=drop_path_rate, norm_cfg=norm_cfg)
        
        self.interm_channels = [self.arch.interm_channels[0], self.arch.interm_channels[1], self.arch.interm_channels[2], self.arch.interm_channels[3]]
        self.upsample_outs = upsample_outs
        self.patch_size = (14, 14)

        if pretrained_path:
            sd = torch.load(pretrained_path)['state_dict']
            # This is normal according to the authors of deformer https://github.com/VCIP-RGBD/DFormer/issues/36
            # This is about some cls heads used in the pretraining and not needed for segmentation anymore
            
            self.arch = load_fitting_state_dict(self.arch, sd)
            print('loaded pretrained weights from {}'.format(pretrained_path))
            #input()
            
            
    def forward(self, x, xd=None, depth_scales=None, masks=None):

        #if xd is not None:
        #    if xd.shape[1] == 1:
        #        xd = xd.repeat((1,3,1,1))
        #else:
        #    xd = torch.zeros(x.shape, dtype=x.dtype, device=x.device)

        outs = self.arch(x, xd)

        if self.upsample_outs:
            for i in range(len(outs)):
                h, w = outs[i].shape[-2:]
                outs[i] = F.interpolate(outs[i], (h*2, w*2), None, 'bilinear', None, recompute_scale_factor=None)

        outs = [None] + outs + [None] # pad for consisten output (first = input size -> discarded anyways) last = cls token -> not needed and not used anyways

        return outs




if __name__ == '__main__':

    device = torch.device('cuda:0')
    

    return_layers=[2, 5, 8, 11]
    model = DformerV2Wrapper(pretrained_path='./data/backbones/DFormerv2_Base_pretrained.pth', upsample_outs=True).to(device)
    

    print(model)
    

    x = torch.rand((1, 3, 224, 224), device=device).to(device)
    xd = torch.rand((1, 1, 224, 224), device=device).to(device)

    out = model(x, xd)
    for i, o in enumerate(out):
        if o is None:
            print(i, type(o))
        else:
            print(i, type(o), o.shape)

