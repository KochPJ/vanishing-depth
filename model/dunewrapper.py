
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import timm
import torch
import types

from PIL import Image
import torch.nn as nn
import inspect
import torch
import warnings
import math
from math import lcm
import numpy as np
import torch.nn.functional as F
from functools import partial
try:
    from .dino import DinoRGBDWrapper, DinoRGBWrapper
    from timm.layers import PatchEmbed
    from .fusion import SqueezeAndExciteFusionAdd
    from .dune.model.encoder.vision_transformer import vit_small as dune_small
    from .dune.model.encoder.vision_transformer import vit_base as dune_base
    from .dune.model.encoder.vision_transformer import vit_large as dune_large
        
    from model.dinov2.models.vision_transformer import vit_small as dinov2_small
    from model.dinov2.models.vision_transformer import vit_base as dinov2_base
    from model.dinov2.models.vision_transformer import vit_large as dinov2_large
    from utils.stuff import load_fitting_state_dict
except:
    pass




class DuneDinoV2(nn.Module):
    def __init__(self, encoder = 'base', upsample_outs=True):
        super().__init__()

        dune_encoder = {
            'small': dune_small, 
            'base': dune_base,
            'large': dune_large
        }[encoder]

        self.return_layers = {
                'small': [2, 5, 8, 11],
                'base': [2, 5, 8, 11],
                'large': [5, 12, 18, 24]
            }[encoder]
        
        self.arch = dune_encoder(
            patch_size=14,
            return_layers=self.return_layers,
            image_size=448, num_register_tokens=4, layerscale_init=0.0001, qkv_bias=True, ln_affine=True, block_chunks=0
        )        
        print('loading depth anything v2')
        
        sd = {}
        n_ = 8
        for sdkey, sdv in torch.load(torch.load('./data/backbones/dune_vit{}14_448.pth'.format(cfg['encoder']), map_location='cpu')['model'].items()):
            if 'teacher' in sdkey or 'projectors' in sdkey:#not needed..
                continue
            if 'encoder.' == sdkey[:n_]: #not needed..
                sdkey = sdkey[n_:]
            sdkey = sdkey.replace('blocks.0.', 'blocks.') # removes the block_chungs=1
            sd[sdkey] = sdv

        self.arch = load_fitting_state_dict(self.arch, sd)
        self.encoder = encoder
        self.upsample_outs = upsample_outs

        self.interm_channels = [768*2, 768*2, 768+2, 768*2]

        dino_encoder = {'small': dinov2_small, 'base': dinov2_base, 'large': dinov2_large}[encoder]
        self.dino = dino_encoder(patch_size=14,
                                img_size=518, init_values=1.0,
                                block_chunks=0, 
                                norm_return_layers=False)
        print('loading dino v2')
        sd = torch.load('./data/backbones/dinov2_vit{}14_pretrain.pth'.format(encoder[0]), map_location='cpu')
        self.dino = load_fitting_state_dict(self.dino, sd)
        self.patch_size = 14
        #input()

        
    def forward(self, x, xd=None, depth_scales=None):

        H, W = x.shape[-2:]
        h_, w_ = H//14, W//14
        h_decode, w_decode = (H // 32)*32, (W // 32)*32

        x_dino = x.clone()

        outs_dune = self.arch.get_intermediate_layers(x, self.return_layers, return_class_token=False, reshape=True)
        dino_outs = self.dino.get_intermediate_layers(x_dino, self.return_layers, return_class_token=False, reshape=True)
        #print(len(dino_outs))

        #outs = [o[0] for o in outs]
        #dino_outs = [o[0] for o in dino_outs]
        outs = []
        for i, (x1, x2) in enumerate(zip(dino_outs, outs_dune)):
            x_ = torch.cat([x1, x2], dim=1)
            #bs, l, d = x_.shape
            #x_ = x_.reshape(bs, int(l / h_), int(l / w_), -1).permute(0, 3, 1, 2)
            #print(i, x_.shape, x1.shape, x2.shape)
            if self.upsample_outs:
                t_size = (int(h_decode / (2**(i+1))), int(w_decode / (2**(i+1))))
                x_ = F.interpolate(x_, t_size, None, 'bilinear', None, recompute_scale_factor=None)
                #print('i', i, x_.shape, t_size)
            outs.append(x_)
                            
        return [None] + outs + [None]



def get_dune_encoder(args, return_layers=True, vit_return_layers=None, rgb_only=False, cat_outs=False,
                       fusion_layer_ids=None, resize_layers=True, down_sample_last_layer=True,
                       norm_return_layers=False, return_before_fusion=None,
                       norm_return_layers_rgb=False, rand_disable_modality=False,
                       add_depth_scales_to_cls_token=False, dino_size='small', fuse_cls_tolken=False):

    dino_encoder = {
        'small': dune_small, 
        'base': dune_base,
        'large': dune_large
    }[dino_size]

    

    if fusion_layer_ids is None:
        fusion_layer_ids = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [5, 12, 18, 24]
        }[dino_size]

    if vit_return_layers is None:
        vit_return_layers = fusion_layer_ids

    if return_before_fusion is None:
        return_before_fusion = cat_outs
    #if not return_layers:
    #    vit_return_layers = False

    print('fusion_layer_ids', fusion_layer_ids)
    print('vit_return_layers', vit_return_layers)
    print('return_before_fusion', return_before_fusion)
    print('cat_outs', cat_outs)
    print('norm_return_layers', norm_return_layers)
    print('norm_return_layers_rgb', norm_return_layers_rgb)
    print('rand_disable_modality', rand_disable_modality)
    print('add_depth_scales_to_cls_token', add_depth_scales_to_cls_token)
    print('dino_encoder', dino_encoder)
    print('fuse_cls_tolken', fuse_cls_tolken)


    vit_rgb = dino_encoder(
        patch_size=14,
        return_layers=vit_return_layers,
        fusion_layer_ids=fusion_layer_ids,
        image_size=448, num_register_tokens=4, layerscale_init=0.0001, qkv_bias=True, ln_affine=True, block_chunks=0, 
        norm_return_layers=norm_return_layers_rgb
    )


    print('return_rdps', args.return_rdps)

    if not rgb_only:
        depth_channels = args.depth_channels
        try:
            if args.with_intr:
                depth_channels = depth_channels * 3
        except:
            pass

        vit_depth = dino_encoder(
            patch_size=14,
            in_chans=depth_channels,
            return_layers=vit_return_layers,
            return_rdps=args.return_rdps,
            fusion_layer_ids=fusion_layer_ids,
            add_fusion=True, image_size=448, num_register_tokens=4, layerscale_init=0.0001, qkv_bias=True, ln_affine=True, block_chunks=0, 
            norm_return_layers=norm_return_layers,
            return_before_fusion=return_before_fusion,
            add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
            fuse_cls_tolken=fuse_cls_tolken
        )


    else:
        print('RGB Only encoder')

    if isinstance(args.pretrained_path, str):
        if os.path.exists(args.pretrained_path):
            sd = {}
            n_ = 8
            for sdkey, sdv in torch.load(args.pretrained_path, map_location='cpu')['model'].items():
                if 'teacher' in sdkey or 'projectors' in sdkey:#not needed..
                    continue
                if 'encoder.' == sdkey[:n_]: #not needed..
                    sdkey = sdkey[n_:]
                sdkey = sdkey.replace('blocks.0.', 'blocks.') # removes the block_chungs=1
                sd[sdkey] = sdv

            vit_rgb = load_fitting_state_dict(vit_rgb, sd)
            print('loaded state dict on vit rgb from: {}'.format(args.pretrained_path))
            
            
            if not rgb_only:
                sd = {}
                n_ = 8
                for sdkey, sdv in torch.load(args.pretrained_path, map_location='cpu')['model'].items():
                    if 'teacher' in sdkey or 'projectors' in sdkey: #not needed..
                        continue
                    if 'encoder.' == sdkey[:n_]: #not needed..
                        sdkey = sdkey[n_:]
                    sdkey = sdkey.replace('blocks.0.', 'blocks.')  # removes the block_chungs=1
                    sd[sdkey] = sdv
                
                vit_depth = load_fitting_state_dict(vit_depth, sd)
                #vit_depth.load_state_dict(sd)
                print('loaded state dict on vit depth from: {}'.format(args.pretrained_path))
    

    if not rgb_only:
        print('Dune RGBD Wrapper')
        model = DinoRGBDWrapper(vit_rgb, vit_depth, return_layers=return_layers, return_rdps=args.return_rdps,
                                cat_outs=cat_outs, resize_layers=resize_layers,
                                down_sample_last_layer=down_sample_last_layer,
                                rand_disable_modality=rand_disable_modality)
    else:
        print('Dune DinoRGB Wrapper')
        model = DinoRGBWrapper(vit_rgb, return_layers=return_layers,
                               resize_layers=resize_layers, down_sample_last_layer=down_sample_last_layer)
    return model


