# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# CroCo model during pretraining
# --------------------------------------------------------


import os
import sys


if __name__ == '__main__':
    sys.path.append(os.getcwd())


import torch
import torch.nn as nn
import copy


from model.mast3r.model import load_model

from utils.stuff import load_fitting_state_dict
import torch.nn.functional as F


class MASt3REncoderWrapper(nn.Module):

    def __init__(self,
                pretrained_path=None,
                upsample_outs=False,
                return_layers=[5, 12, 18, 24]
                ):

        super(MASt3REncoderWrapper, self).__init__()

        # load model and remove decoder 
        self.arch = load_model(pretrained_path, torch.device('cpu'))
        self.arch.downstream_head1 = None
        self.arch.downstream_head2 = None
        self.arch.decoder_embed = None
        self.arch.dec_blocks = None
        self.arch.dec_blocks2 = None
        self.arch.dec_norm = None
        self.patch_size = 16

        self.upsample_outs = upsample_outs
        self.return_layers = return_layers
        self.interm_channels = [1024, 1024, 1024, 1024]        
       

    def forward(self, x, xd=None, depth_scales=None, mask=None):

        B, _, H, W = x.shape

        H_ = H // 16
        W_ = W // 16
        h_decode, w_decode = (H // 32)*32, (W // 32)*32

        true_shape = torch.tensor(x.shape[-2:])[None].repeat(B, 1)

        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.arch.patch_embed(x, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.arch.enc_pos_embed is None

        # now apply the transformer encoder and normalization        
        outs = []
        i = 0
        for ii, blk in enumerate(self.arch.enc_blocks):
            x = blk(x, pos)
            if ii+1 in self.return_layers:
                s = self.arch.enc_norm(x).view((B, H_, W_, -1)).permute((0, 3, 1, 2))
                if self.upsample_outs:
                    t_size = (int(h_decode / (2**(i+1))), int(w_decode / (2**(i+1))))
                    s = F.interpolate(s, t_size, None, 'bilinear', None, recompute_scale_factor=None)

                outs.append(s)
                i+=1

        outs = [None] + outs + [None] # pad for consisten output (first = input size -> discarded anyways) last = cls token -> not needed and not used anyways

        return outs 


if __name__ == '__main__':

    device = torch.device('cpu')
    

    return_layers=[2, 5, 8, 11]
    model = MASt3REncoderWrapper(pretrained_path='./data/backbones/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', upsample_outs=True).to(device)
    

    print(model)
    

    x = torch.rand((1, 3, 224, 224), device=device).to(device)

    out = model(x)
    for i, o in enumerate(out):
        if o is None:
            print(i, type(o))
        else:
            print(i, type(o), o.shape)