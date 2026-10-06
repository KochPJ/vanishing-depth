import os
import sys

if __name__ == '__main__':
    sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.stuff import load_fitting_state_dict
from model.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2

from model.dinov2.models.vision_transformer import vit_small as dinov2_small
from model.dinov2.models.vision_transformer import vit_base as dinov2_base
from model.dinov2.models.vision_transformer import vit_large as dinov2_large



dav2_model_configs = {
    'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'giant': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


class DepthAnythingV2Wrapper(nn.Module):
    def __init__(self, encoder = 'base', upsample_outs=True):
        super().__init__()
        
        cfg = dav2_model_configs[encoder]
        self.arch = DepthAnythingV2(**cfg)        
        sd = torch.load('./data/backbones/depth_anything_v2_{}.pth'.format(cfg['encoder']), map_location='cpu')
        self.arch = load_fitting_state_dict(self.arch, sd)
        self.patch_size = 14

        self.encoder = encoder
        
        self.return_layers = self.arch.intermediate_layer_idx[self.arch.encoder]
        self.cfg = cfg
        self.upsample_outs = upsample_outs
        del self.arch.depth_head
        self.interm_channels = [768, 768, 768, 768]
        
    def forward(self, x, xd=None, depth_scales=None):

        H, W = x.shape[-2:]
        h_, w_ = H//14, W//14
        h_decode, w_decode = (H // 32)*32, (W // 32)*32

        outs_da2 = self.arch.pretrained.get_intermediate_layers(x, self.return_layers, return_class_token=False, reshape=True)
        #outs = [o[0] for o in outs]
        outs = []
        for i, x_ in enumerate(outs_da2):
            #bs, l, d = x_.shape
            #x_ = x_.reshape(bs, int(l / h_), int(l / w_), -1).permute(0, 3, 1, 2)
            if self.upsample_outs:
                t_size = (int(h_decode / (2**(i+1))), int(w_decode / (2**(i+1))))
                x_ = F.interpolate(x_, t_size, None, 'bilinear', None, recompute_scale_factor=None)
            #print(i, x_.shape)
            outs.append(x_)
            
        #print('out', [o.shape for o in outs])
                            
        return [None] + outs + [None]



class DinV2andDepthAnythingV2Wrapper(nn.Module):
    def __init__(self, encoder = 'base', upsample_outs=True):
        super().__init__()
        
        cfg = dav2_model_configs[encoder]
        self.arch = DepthAnythingV2(**cfg)        
        print('loading depth anything v2')
        sd = torch.load('./data/backbones/depth_anything_v2_{}.pth'.format(cfg['encoder']), map_location='cpu')
        self.arch = load_fitting_state_dict(self.arch, sd)
        self.encoder = encoder
        self.cfg = cfg
        self.upsample_outs = upsample_outs
        del self.arch.depth_head
        self.interm_channels = [768*2, 768*2, 768*2, 768*2]
        self.patch_size = 14

        dino_encoder = {'small': dinov2_small, 'base': dinov2_base, 'large': dinov2_large}[encoder]
        self.return_layers = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [5, 12, 18, 24]
        }[encoder]

        

        self.dino = dino_encoder(patch_size=14,
                                img_size=518, init_values=1.0,
                                block_chunks=0, 
                                norm_return_layers=False)
        print('loading dino v2')
        sd = torch.load('./data/backbones/dinov2_vit{}14_pretrain.pth'.format(encoder[0]), map_location='cpu')
        self.dino = load_fitting_state_dict(self.dino, sd)
        #input()

        
    def forward(self, x, xd=None, depth_scales=None):

        H, W = x.shape[-2:]
        h_, w_ = H//14, W//14
        h_decode, w_decode = (H // 32)*32, (W // 32)*32

        x_dino = x.clone()

        outs_da2 = self.arch.pretrained.get_intermediate_layers(x, self.return_layers, return_class_token=False, reshape=True)
        dino_outs = self.dino.get_intermediate_layers(x_dino, self.return_layers, return_class_token=False, reshape=True)
        #print(len(dino_outs))

        #outs = [o[0] for o in outs]
        #dino_outs = [o[0] for o in dino_outs]
        outs = []
        for i, (x1, x2) in enumerate(zip(dino_outs, outs_da2)):
            x_ = torch.cat([x1, x2], dim=1)
            #bs, l, d = x_.shape
            #x_ = x_.reshape(bs, int(l / h_), int(l / w_), -1).permute(0, 3, 1, 2)
            #print(i, x_.shape, x1.shape, x2.shape)
            if self.upsample_outs:
                t_size = (int(h_decode / (2**(i+1))), int(w_decode / (2**(i+1))))
                x_ = F.interpolate(x_, t_size, None, 'bilinear', None, recompute_scale_factor=None)
                #print('i', i, x_.shape, t_size)
            outs.append(x_)
        #print('out', [o.shape for o in outs])
                            
        return [None] + outs + [None]

if __name__ == '__main__':

    device = torch.device('cuda:0')

    model = DinV2andDepthAnythingV2Wrapper().to(device)

    print(model)


    x = torch.rand((1, 3, 168, 168), device=device)
    xd = torch.rand((1, 1, 168, 168), device=device)

    out = model(x, xd)
    for i, o in enumerate(out):
        if o is None:
            print(i, type(o))
            
        else:

            print(i, type(o), o.shape)

