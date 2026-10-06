
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
except:
    pass

from timm.layers import resample_abs_pos_embed

class FiT3DWrapper(nn.Module):
    def __init__(self, arch, return_layers=None, add_fusion=False, return_rdps=False, norm_return_layers=False, in_chans=None, img_size=518, patch_size=14,
                    add_depth_scales_to_cls_token=False, fuse_cls_tolken=False,  fusion_layer_ids=None, return_before_fusion=False):
        super().__init__()
        self.arch = arch

        self.num_channels = self.arch.embed_dim  
        self.embed_dim = self.arch.embed_dim  
        if fusion_layer_ids is None:
            fusion_layer_ids = []
     

        if in_chans is not None:
            self.arch.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=self.num_channels, strict_img_size=False, output_fmt='NHWC')

        self.patch_embed = self.arch.patch_embed
        self.patch_size = patch_size
        self.fusion_layers = None
        self.return_rdps = False
        self.add_depth_scales_to_cls_token = add_depth_scales_to_cls_token
        self.fuse_cls_tolken = fuse_cls_tolken
        self.fusion_layer_ids = fusion_layer_ids
        self.add_fusion = add_fusion
        self.norm_return_layers = norm_return_layers
        self.return_before_fusion = return_before_fusion
        if return_layers is None:
            return_layers = []
        else:
            if self.add_fusion:
                self.fusion_layers = nn.ModuleList([
                    SqueezeAndExciteFusionAdd(channels=self.num_channels, return_rgb_depth_p=return_rdps)
                    for _ in range(len(return_layers))
                ])
                self.return_rdps = return_rdps

        self.return_layers = return_layers

    def my_pos_embed(self, x: torch.Tensor, depth_scales=None) -> torch.Tensor:
        """Apply positional embedding to input."""

        #print(x.shape, type(depth_scales))
        if self.arch.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.arch.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.arch.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.arch.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.arch.no_embed_class else self.arch.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.arch.pos_embed

        to_cat = []
        if self.arch.cls_token is not None:
            if depth_scales is not None:
                #print('depth?')
                if depth_scales.dim() == 2:
                    depth_scales = depth_scales.unsqueeze(1)

                if self.add_depth_scales_to_cls_token:
                    cls_token = self.arch.cls_token.expand(x.shape[0], -1, -1)
                    cls_token = cls_token.add(depth_scales)
                else:
                    cls_token = depth_scales

            else: 
                cls_token = self.arch.cls_token.expand(x.shape[0], -1, -1)
            to_cat.append(cls_token)
            
        if self.arch.reg_token is not None:
            to_cat.append(self.arch.reg_token.expand(x.shape[0], -1, -1))

        #print('cat', [c.shape for c in to_cat])
        if self.arch.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            #print(x.shape, pos_embed.shape)
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed
            #print(x.shape, pos_embed.shape)
            
            
        return self.arch.pos_drop(x)

        
    def forward(self, x, merge_layers=None, depth_scales=None):
        
        x = self.arch.patch_embed(x)
        x = self.my_pos_embed(x, depth_scales=depth_scales)  # add depthscales to the cls token else original self.arch._pos:embed(x)
        x = self.arch.patch_drop(x)
        x = self.arch.norm_pre(x)


        xs = []
        xs_fusion = []
        rdps = []
        j = 0
        #print('return_layers', self.return_layers)

        for i, blk in enumerate(self.arch.blocks):
            x = blk(x)
            #print('fx', x.shape)

            if i in self.return_layers and self.return_before_fusion:
                #print('returning', i, x.shape)
                if self.norm_return_layers:
                    xs.append(self.arch.norm(x))
                    #print('norming', i, xs[-1].shape)
                else:
                    #print('no norm here', i, xs[-1].shape)
                    xs.append(x)

            if merge_layers is not None:
                if i in self.fusion_layer_ids and self.add_fusion:

                    if not self.fuse_cls_tolken:
                        cls_tolken = x[:, 0].unsqueeze(1)
                        bs, c, dims = x[:, 1:].shape

                        x = self.fusion_layers[j](
                            x[:, 1:].reshape(-1, dims, 1, 1),
                            merge_layers[j][:, 1:].reshape(-1, dims, 1, 1)
                        )
                    else:
                        bs, c, dims = x.shape
                        x = self.fusion_layers[j](
                            x.reshape(-1, dims, 1, 1),
                            merge_layers[j].reshape(-1, dims, 1, 1)
                        )
                        #print('fusion of cls token')

                    if self.return_rdps:
                        #print('x', type(x))
                        x, rdp = x
                        #print('unpacking rpds', rdp, i)
                        rdps.append(rdp)

                    j += 1
                    x = x.reshape(bs, c, dims)
                    if not self.fuse_cls_tolken:
                        x = torch.cat((cls_tolken, x), dim=1)

                    xs_fusion.append(x)
                    #print('xs_fusion', i, xs_fusion[-1].shape)
                    

            elif i in self.fusion_layer_ids:
                #print('fusing', i, x.shape)
                xs_fusion.append(x)

            if i in self.return_layers and not self.return_before_fusion:
                # print('returning', i, x.shape)
                if self.norm_return_layers:
                    # print('norming', i)
                    xs.append(self.arch.norm(x))
                else:
                    #print('no norm there')
                    xs.append(x)

        x = self.arch.norm(x)
        if len(xs) > 0:
            xs.append(x[:, 0])
            if len(xs_fusion) > 0:
                xs_fusion.append(x[:, 0])
                xs = {'return': xs, 'fusion': xs_fusion}

            if self.return_rdps:
                #print('retruning rdps')
                return (xs, rdps)
            else:
                return xs
        else:
            return x[:, 0]

        '''

        xs = []
        rdps = []
        j = 0
        for i, blk in enumerate(self.arch.blocks):
            x = blk(x)
            if merge_layers is not None:
                if i in self.return_layers and self.add_fusion:
                    bs, c, dims = x.shape
                    x = self.fusion_layers[j](x.reshape(-1, dims, 1, 1),
                                              merge_layers[j].reshape(-1, dims, 1, 1)).reshape(bs, c, dims)
                    if isinstance(x, tuple):
                        x, rdp = x
                        rdps.append(rdp)
                    j += 1
            if i in self.return_layers:
                if self.norm_return_layers:
                    xs.append(self.arch.norm(x))
                else:
                    xs.append(x)

        x = self.arch.norm(x)
        if len(xs) > 0:
            xs.append(x[:, 0])
            if self.return_rdps:
                return (xs, rdps)
            else:
                return xs
        else:
            return x[:, 0]
        '''


def get_fit3d_encoder(args, return_layers=True, vit_return_layers=None, rgb_only=False, cat_outs=False,
                       fusion_layer_ids=None, resize_layers=True, down_sample_last_layer=True,
                       norm_return_layers=False, return_before_fusion=None,
                       norm_return_layers_rgb=False, rand_disable_modality=False,
                       add_depth_scales_to_cls_token=False, dino_size='small', fuse_cls_tolken=False):

    dino_encoder = {
        'small': 'dinov2_small_fine', 
        'base': 'dinov2_base_fine',
        'large': 'Not Implemented'
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

    vit_rgb = FiT3DWrapper(
        arch=torch.hub.load("ywyue/FiT3D", dino_encoder),
        return_layers=vit_return_layers,
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

        vit_depth = FiT3DWrapper(
            arch=torch.hub.load("ywyue/FiT3D", dino_encoder),
            return_layers=vit_return_layers,
            norm_return_layers=norm_return_layers,
            in_chans=depth_channels,
            return_rdps=args.return_rdps,
            fusion_layer_ids=fusion_layer_ids,
            add_fusion=True,
            return_before_fusion=return_before_fusion,
            add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
            fuse_cls_tolken=fuse_cls_tolken
        )


    else:
        print('RGB Only encoder')
    

    if not rgb_only:
        print('Fit3D RGBD Wrapper')
        model = DinoRGBDWrapper(vit_rgb, vit_depth, return_layers=return_layers, return_rdps=args.return_rdps,
                                cat_outs=cat_outs, resize_layers=resize_layers,
                                down_sample_last_layer=down_sample_last_layer,
                                rand_disable_modality=rand_disable_modality)
    else:
        print('Fit3D DinoRGB Wrapper')
        model = DinoRGBWrapper(vit_rgb, return_layers=return_layers,
                               resize_layers=resize_layers, down_sample_last_layer=down_sample_last_layer)
    return model



if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.getcwd())
    device = torch.device('cuda:0')
    encoder = torch.hub.load("ywyue/FiT3D", 'dinov2_base_fine')

    return_layers=[2, 5, 8, 11]
    model = FiT3DWrapper(encoder, return_layers=return_layers, fusion_layer_ids=[]).to(device)
    

    print(model)
    
    print(inspect.getfile(model.arch.__class__))

    x = torch.rand((1,3, 224, 224), device=device)

    out = model(x)
    for i, o in enumerate(out):
        if o is None:
            print(i, type(o))
            
        else:

            print(i, type(o), o.shape)

