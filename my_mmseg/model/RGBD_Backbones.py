'''
Copied and edidted from:
https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L257

'''
import os.path

import torch.nn as nn
import torch
import warnings
import math
import numpy as np
import time
import torch.nn.functional as F
from functools import partial
from model.fusion import SqueezeAndExciteFusionAdd
from model.dinov2.models.vision_transformer import vit_small as dinov2_small
from model.dinov2.models.vision_transformer import vit_base as dinov2_base
from model.dinov2.models.vision_transformer import vit_large as dinov2_large

try:
    from model.dinov3.models.vision_transformer import vit_small as dinov3_small
    from model.dinov3.models.vision_transformer import vit_base as dinov3_base
    from model.dinov3.models.vision_transformer import vit_large as dinov3_large
except:
    pass

from model.dune.model.encoder.vision_transformer import vit_small as dune_small
from model.dune.model.encoder.vision_transformer import vit_base as dune_base
from model.dune.model.encoder.vision_transformer import vit_large as dune_large

from model.dformerwrapper import DformerWrapper, DformerV2Wrapper

try:
    from model.crocoencoderwrapper import CroCoEncoderWrapper
    from model.mast3rencoderwrapper import MASt3REncoderWrapper
except:
    pass

from model.fit3d import FiT3DWrapper
#from model.EVA.eva2_wrapper import evav2_base
from utils.stuff import load_fitting_state_dict
from datasets.transforms import DepthPositionalEncoding
from model.dino import DinoRGBDWrapper, DinoRGBWrapper
from model.multimaewrapper import MultiMAEWrapper #
from model.depthanythingv2wrapper import DinV2andDepthAnythingV2Wrapper, DepthAnythingV2Wrapper
from model.omnidcwrapper import OmniDCWrapper
from model.dmd3cwrapper import DMD3CWrapper


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from mmengine.model import BaseModule, ModuleList
from mmseg.registry import MODELS

import sys
import os
sys.path.append(os.getcwd())
from my_mmseg.utils.loading import log_message
from my_mmseg.model.Omnivore_Backbone import Omnivore




def get_dinov2_encoder(return_layers=True, vit_return_layers=None, rgb_only=False, cat_outs=True,
                       fusion_layer_ids=None, resize_layers=True, down_sample_last_layer=False,
                       norm_return_layers=True, return_before_fusion=None,
                       norm_return_layers_rgb=True, rand_disable_modality=False,
                       add_depth_scales_to_cls_token=False, dino_size='base', depth_channels=32,
                       with_intr=False, pretrained_path=None, encoder_path=None):

    dino_encoder = {
        'small': dinov2_small, 
        'base': dinov2_base,
        'large': dinov2_large
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


    vit_rgb = dino_encoder(patch_size=14,
                           return_layers=vit_return_layers,
                           fusion_layer_ids=fusion_layer_ids,
                           img_size=518, init_values=1.0,
                           block_chunks=0, norm_return_layers=norm_return_layers_rgb)
    
    # state_dict_rgb = vit_rgb.state_dict()
    # print('keys in vit-rgb state_dict:')
    # for key in state_dict_rgb.keys():
    #     print(key)

    if not rgb_only:
        if with_intr:
            depth_channels = depth_channels * 3        
        vit_depth = dino_encoder(patch_size=14,
                                 in_chans=depth_channels,
                                 return_layers=vit_return_layers,
                                 return_rdps=False,
                                 fusion_layer_ids=fusion_layer_ids,
                                 add_fusion=True, img_size=518, init_values=1.0, block_chunks=0,
                                 norm_return_layers=norm_return_layers,
                                 return_before_fusion=return_before_fusion,
                                 add_depth_scales_to_cls_token=add_depth_scales_to_cls_token)
    else:
        print('RGB Only encoder')
       
    # print('type of VIT depth', type(vit_depth))
    # state_dict = vit_depth.state_dict()
    # print('keys in vit-depth state_dict:')
    # for key in state_dict.keys():
    #     print(key)
    

    #print('pretrained_path', pretrained_path)
    #print('pretrained_path type is', type(pretrained_path))
    #print('encoder path is ', encoder_path)

    if isinstance(pretrained_path, str) and encoder_path is None:
        print('pretrained_path', pretrained_path)
        log_message('pretrained_path: ' + pretrained_path)
        if os.path.exists(pretrained_path):
            
            sd = torch.load(pretrained_path, map_location='cpu')
            vit_rgb = load_fitting_state_dict(vit_rgb, sd)
            print('loaded state dict on vit rgb from: {}'.format(pretrained_path))
            if not rgb_only:
                vit_depth = load_fitting_state_dict(vit_depth, sd)
                # vit_depth.load_state_dict(sd)
                print('loaded state dict on vit depth from: {}'.format(pretrained_path))

        else:
            print('path does exist {}'.format(os.path.exists(pretrained_path)))
        #input()
    #input()

    if not rgb_only:
        print('DinoRGBDWrapper')
        # print('Im here ')
        # input()
        model = DinoRGBDWrapper(vit_rgb, vit_depth, return_layers=return_layers, return_rdps=False,
                                cat_outs=cat_outs, resize_layers=resize_layers,
                                down_sample_last_layer=down_sample_last_layer,
                                rand_disable_modality=rand_disable_modality)
    else:
        print('DinoRGBWrapper')
        model = DinoRGBWrapper(vit_rgb, return_layers=return_layers,
                               resize_layers=resize_layers, down_sample_last_layer=down_sample_last_layer)

          
    if encoder_path is not None:
        print('loading from ', encoder_path)
        sd = torch.load(encoder_path, map_location='cpu')['state_dict']
        model = load_fitting_state_dict(model, sd)

    #input('Are the weights loaded corret in the backbone?')
        

    return model




@MODELS.register_module()
class RGBDVit(BaseModule):
     
     

    def __init__(self,
                 img_size=224,
                 patch_size=14,
                 rgb_only=False,
                 encoder_path=None,
                 pretrained_path=None,
                 version='base',
                 freeze_encoder=True,
                 depth_channels=32,
                 init_cfg =None,
                 **kwargs

                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        

        
        # self.pretrained_path = pretrained_path

        # print('pretrained_path', pretrained_path)
        # print('pretrained_path type is', type(pretrained_path))
        # print('encoder path is ', encoder_path)
        # input()
        # pretrained_path, dino_version = None, 'base' 
        
        
        if not isinstance(rgb_only, bool):
            raise TypeError(f"'rgb_only' must be a boolean value , but got {type(rgb_only).__name__}")

        if encoder_path is not None and not isinstance(encoder_path, str):
            raise TypeError(f"'encoder_path' must be a string object, but got {type(encoder_path).__name__}")

        if pretrained_path is not None and not isinstance(pretrained_path, str):
            raise TypeError(f"'pretrained_path' must be a string object, but got {type(pretrained_path).__name__}")

        if not isinstance(version, str):
            raise TypeError(f"'version' must be a boolean value , but got {type(version).__name__}")


        self.img_size = img_size
        self.patch_size = patch_size
        self.rgb_only = rgb_only
        self.encoder_path = encoder_path
        self.pretrained_path = pretrained_path
        self.version = version
        self.depth_channels = depth_channels

        if init_cfg is not None and 'checkpoint' in init_cfg:
            self.pretrained_path = init_cfg['checkpoint']


        if init_cfg is not None and 'version' in init_cfg:
            dino_version = init_cfg['version']

        
        if self.version not in ['small', 'base','large']:
            raise ValueError(f"Invalid version '{self.version}'. Allowed versions for the DinoV2 models are 'small', 'base', 'large'.")
        #print('dino_version', dino_version)
        # print('init_cfg', init_cfg)

        # encoder_path = '/mnt/logicNAS/Exchange/paul/vanishing-depth-models/20240908_1725793388_Dino_UNet-Long32c-15m-Fix15m32c_448_encoder.plt'

        self.encoder = get_dinov2_encoder(rgb_only=self.rgb_only, pretrained_path=self.pretrained_path, dino_size=self.version, encoder_path=self.encoder_path, cat_outs=True, depth_channels=depth_channels)   
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
                
    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #
        if not self.training:
            print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        
        #print('input size inside RGBDVit', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        
        return embs


def get_dinov3_encoder(return_layers=True, vit_return_layers=None, rgb_only=False, cat_outs=False,
                       fusion_layer_ids=None, resize_layers=True, down_sample_last_layer=True,
                       norm_return_layers=False, return_before_fusion=None,
                       norm_return_layers_rgb=False, rand_disable_modality=False,
                       add_depth_scales_to_cls_token=False, dino_size='base', fuse_cls_tolken=False,
                       depth_channels=32, return_rdps=False,
                       with_intr=False, pretrained_path=None, encoder_path=None):

    dino_encoder = {
        'small': dinov3_small, 
        'base': dinov3_base,
        'large': dinov3_large
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

    vit_rgb = dino_encoder(patch_size=16,
                           return_layers=vit_return_layers,
                           fusion_layer_ids=fusion_layer_ids,
                           layerscale_init=1e-5, mask_k_bias=True,  n_storage_tokens=4,
                           img_size=518, norm_return_layers=norm_return_layers_rgb)
    print('return_rdps', return_rdps)

    if not rgb_only:
        depth_channels = depth_channels
        try:
            if with_intr:
                depth_channels = depth_channels * 3
        except:
            pass
        vit_depth = dino_encoder(
            patch_size=16,
            in_chans=depth_channels,
            return_layers=vit_return_layers,
            return_rdps=return_rdps,
            fusion_layer_ids=fusion_layer_ids,
            add_fusion=True, img_size=518,
            layerscale_init=1e-5, mask_k_bias=True, n_storage_tokens=4,
            norm_return_layers=norm_return_layers,
            return_before_fusion=return_before_fusion,
            add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
            fuse_cls_tolken=fuse_cls_tolken
        )
    else:
        print('RGB Only encoder')
    #args.pretrained_path = './data/backbones/dinov2_vits14_pretrain.pth'
    # print('args.pretrained_path', args.pretrained_path)
    if isinstance(pretrained_path, str):
        if os.path.exists(pretrained_path):
            sd = torch.load(pretrained_path, map_location='cpu')   
            vit_rgb = load_fitting_state_dict(vit_rgb, sd)
            print('loaded state dict on vit rgb from: {}'.format(pretrained_path))
            if not rgb_only:
                vit_depth = load_fitting_state_dict(vit_depth, sd)
                #vit_depth.load_state_dict(sd)
                print('loaded state dict on vit depth from: {}'.format(pretrained_path))
        #input()

    if not rgb_only:
        print('DinoRGBDWrapper')
        model = DinoRGBDWrapper(vit_rgb, vit_depth, return_layers=return_layers, return_rdps=return_rdps,
                                cat_outs=cat_outs, resize_layers=resize_layers,
                                down_sample_last_layer=down_sample_last_layer,
                                rand_disable_modality=rand_disable_modality)
    else:
        print('DinoRGBWrapper')
        model = DinoRGBWrapper(vit_rgb, return_layers=return_layers,
                               resize_layers=resize_layers, down_sample_last_layer=down_sample_last_layer)
    return model


@MODELS.register_module()
class DinoV3(BaseModule):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 rgb_only=False,
                 encoder_path=None,
                 pretrained_path=None,
                 version='base',
                 freeze_encoder=True,
                 depth_channels=32,
                 init_cfg =None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        if not isinstance(rgb_only, bool):
            raise TypeError(f"'rgb_only' must be a boolean value , but got {type(rgb_only).__name__}")

        if encoder_path is not None and not isinstance(encoder_path, str):
            raise TypeError(f"'encoder_path' must be a string object, but got {type(encoder_path).__name__}")

        if pretrained_path is not None and not isinstance(pretrained_path, str):
            raise TypeError(f"'pretrained_path' must be a string object, but got {type(pretrained_path).__name__}")

        if not isinstance(version, str):
            raise TypeError(f"'version' must be a boolean value , but got {type(version).__name__}")

        self.img_size = img_size
        self.patch_size = patch_size
        self.rgb_only = rgb_only
        self.encoder_path = encoder_path
        self.pretrained_path = pretrained_path
        self.version = version
        self.depth_channels = depth_channels

        if init_cfg is not None and 'checkpoint' in init_cfg:
            self.pretrained_path = init_cfg['checkpoint']

        if init_cfg is not None and 'version' in init_cfg:
            dino_version = init_cfg['version']

        if self.version not in ['small', 'base','large']:
            raise ValueError(f"Invalid version '{self.version}'. Allowed versions for the DinoV2 models are 'small', 'base', 'large'.")
       
        self.encoder = get_dinov3_encoder(rgb_only=self.rgb_only, pretrained_path=self.pretrained_path, dino_size=self.version, encoder_path=self.encoder_path, cat_outs=True, depth_channels=depth_channels)   
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            print('freezing encoder')
            for p in self.encoder.parameters():
                p.requires_grad = False
        print('all good? continue in 5s')
        time.sleep(5)



    def forward(self, inputs):
        inputs = inputs.to(next(self.parameters()).device) #
        if not self.training:
            print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        
        #print('input size inside RGBDVit', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        return embs

'''
def eva_load_fitting_state_dict(arch, state_dict):
    keys = list(arch.state_dict().keys())
    keys_left = list(arch.state_dict().keys())
    wrong_shape_keys = []
    #print('model keys', keys)
    wrong_key = 0
    wrong_shape = 0
    okay = 0
    #print('sd keys', state_dict.keys())
    for key, v in state_dict.items():
        if key not in keys:
            wrong_key += 1
            print('wrong key', key)
            continue
        sd = {key: v}
        del keys_left[keys_left.index(key)]
        try:
            arch.load_state_dict(sd, strict=False)
        except Exception as e:
            if '.pos_embed' in key:
                pos_embed_checkpoint = state_dict[key]
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_patches = arch.color_encoder.patch_embed.num_patches
                num_extra_tokens = arch.color_encoder.pos_embed.shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Pos Embed {} interpolate from {}x{} to {}x{}".format(key, orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2).float()
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens.float(), size=(new_size, new_size), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    sd = {key: new_pos_embed}
                    arch.load_state_dict(sd, strict=False)
                okay += 1


            #print(state_dict.keys())
            elif 'rope' in key:              
            
                rope_embed_checkpoint = state_dict[key]
                embedding_size = rope_embed_checkpoint.shape[-1]
                num_patches = arch.color_encoder.patch_embed.num_patches
                num_extra_tokens = arch.color_encoder.pos_embed.shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged

                
                #print('key', key, orig_size, new_size, rope_embed_checkpoint.shape)
                if orig_size != new_size:
                    print("Rope {} interpolate from {}x{} to {}x{}".format(key, orig_size, orig_size, new_size, new_size))
                    
                    # only the position tokens are interpolated
                    rope_tokens = rope_embed_checkpoint.reshape(orig_size, orig_size, embedding_size).unsqueeze(0).permute(0, 3, 1, 2).float()   #.view((-1, rope_embed_checkpoint.shape[-1]))
                    
                    #pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2).float()
                    rope_tokens = torch.nn.functional.interpolate(
                        rope_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    
                    rope_tokens = rope_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
                    
                    #new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    #state_dict[key] = rope_tokens
                    sd = {key: rope_tokens}
                    arch.load_state_dict(sd, strict=False)
                okay += 1

            else: 
                print('wrong shape', e)
                wrong_shape += 1
                wrong_shape_keys.append(keys)
            
            #input()
            continue
        okay += 1

    msg = 'Loaded {}/{} weights, wrong key: {}, wrong shape: {}, missing: {}'.format(
        okay, len(keys), wrong_key, wrong_shape, len(keys) - (okay+wrong_key+wrong_shape))
    print(msg)
    #print('keys_left', keys_left)
    #print('wrong_shape_keys', wrong_shape_keys)
    return arch



def get_evav2_encoder(return_layers=True, vit_return_layers=None, rgb_only=False, cat_outs=False,
                       fusion_layer_ids=None, resize_layers=True, down_sample_last_layer=True,
                       return_before_fusion=None, rand_disable_modality=False,
                       add_depth_scales_to_cls_token=False, eva_size='base', fuse_cls_tolken=False,
                       pretrained_path=None, depth_channels=32, with_intr=False, encoder_path=None,
                       patch_size=14, img_size=224):


    eva_encoder = {
        'small': evav2_base, 
        'base': evav2_base,
        'large': evav2_base
    }[eva_size]


    if fusion_layer_ids is None:
        fusion_layer_ids = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [5, 12, 18, 24]
        }[eva_size]

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
    print('rand_disable_modality', rand_disable_modality)
    print('add_depth_scales_to_cls_token', add_depth_scales_to_cls_token)
    print('eva_encoder', eva_encoder)
    print('fuse_cls_tolken', fuse_cls_tolken)


    if isinstance(pretrained_path, str):
        if len(pretrained_path) == 0:
            pretrained_path = None


    print('patch_size', patch_size)
    print('img_size', img_size)
    print('pretrained', pretrained_path)
    #print('pretrained', os.path.exists(pretrained_path))


    vit_rgb = eva_encoder(patch_size=patch_size,
                           return_layers=vit_return_layers,
                           fusion_layer_ids=fusion_layer_ids,
                           img_size=img_size,
                           pretrained=pretrained_path)
                           
    #input()


    if not rgb_only:
        
        try:
            if with_intr:
                depth_channels = depth_channels * 3
        except:
            pass
        vit_depth = eva_encoder(patch_size=patch_size,
                                 in_chans=depth_channels,
                                 return_layers=vit_return_layers,
                                 return_rdps=False,
                                 fusion_layer_ids=fusion_layer_ids,
                                 add_fusion=True, img_size=img_size,
                                 return_before_fusion=return_before_fusion,
                                 add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                 fuse_cls_tolken=fuse_cls_tolken,
                                pretrained=pretrained_path)
    else:
        print('RGB Only encoder')

    if not rgb_only:
        print('DinoRGBDWrapper')
        model = DinoRGBDWrapper(vit_rgb, vit_depth, return_layers=return_layers, return_rdps=False,
                                cat_outs=cat_outs, resize_layers=resize_layers,
                                down_sample_last_layer=down_sample_last_layer,
                                rand_disable_modality=rand_disable_modality)
    else:
        print('DinoRGBWrapper')
        model = DinoRGBWrapper(vit_rgb, return_layers=return_layers,
                               resize_layers=resize_layers, down_sample_last_layer=down_sample_last_layer)

    
          
    if encoder_path is not None:
        print('loading from ', encoder_path)
        sd = torch.load(encoder_path, map_location='cpu')['state_dict']
        model = eva_load_fitting_state_dict(model, sd)

    input('Are the weights loaded corret in the backbone?')
        
    return model



@MODELS.register_module()
class EVA02(BaseModule):

    def __init__(self,
                 img_size=518,
                 patch_size=14,
                 rgb_only=False,
                 encoder_path=None,
                 pretrained_path=None,
                 version='base',
                 freeze_encoder=True,
                 depth_channels=32,
                 init_cfg=None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        

        
        # self.pretrained_path = pretrained_path

        # print('pretrained_path', pretrained_path)
        # print('pretrained_path type is', type(pretrained_path))
        # print('encoder path is ', encoder_path)
        # input()
        # pretrained_path, dino_version = None, 'base' 
        
        
        if not isinstance(rgb_only, bool):
            raise TypeError(f"'rgb_only' must be a boolean value , but got {type(rgb_only).__name__}")

        if encoder_path is not None and not isinstance(encoder_path, str):
            raise TypeError(f"'encoder_path' must be a string object, but got {type(encoder_path).__name__}")

        if pretrained_path is not None and not isinstance(pretrained_path, str):
            raise TypeError(f"'pretrained_path' must be a string object, but got {type(pretrained_path).__name__}")

        if not isinstance(version, str):
            raise TypeError(f"'version' must be a boolean value , but got {type(version).__name__}")


        self.img_size = img_size
        self.patch_size = patch_size
        self.rgb_only = rgb_only
        self.encoder_path = encoder_path
        self.pretrained_path = pretrained_path
        self.version = version
        self.depth_channels = depth_channels

        if init_cfg is not None and 'checkpoint' in init_cfg:
            self.pretrained_path = init_cfg['checkpoint']


        if init_cfg is not None and 'version' in init_cfg:
            dino_version = init_cfg['version']

        
        if self.version not in ['small', 'base','large']:
            raise ValueError(f"Invalid version '{self.version}'. Allowed versions for the DinoV2 models are 'small', 'base', 'large'.")
        #print('dino_version', dino_version)
        # print('init_cfg', init_cfg)

        # encoder_path = '/mnt/logicNAS/Exchange/paul/vanishing-depth-models/20240908_1725793388_Dino_UNet-Long32c-15m-Fix15m32c_448_encoder.plt'

        self.encoder = get_evav2_encoder(rgb_only=self.rgb_only, pretrained_path=self.pretrained_path, eva_size=self.version, encoder_path=self.encoder_path, cat_outs=True, patch_size=self.patch_size, img_size=self.img_size,
                                         depth_channels=depth_channels)  
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        #input()
            
    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #
        if not self.training:
            print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        
        #print('input size inside RGBDVit', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
            
        
        return embs
'''

@MODELS.register_module()
class MultiMAE(BaseModule):

    def __init__(self,
                 freeze_encoder=True,
                 init_cfg=None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        self.encoder = MultiMAEWrapper(upsample_outs=True)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder

    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #
        #if not self.training:
        #    print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        
        #print('input size inside MultiMAE', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
            
        return embs



def get_fit3d_encoder(return_layers=True, vit_return_layers=None, rgb_only=False, cat_outs=False,
                      fusion_layer_ids=None, resize_layers=True, down_sample_last_layer=True,
                      norm_return_layers=False, return_before_fusion=None,
                      norm_return_layers_rgb=False, rand_disable_modality=False,
                      add_depth_scales_to_cls_token=False, dino_size='small', fuse_cls_tolken=False, depth_channels=32, return_rdps=False, with_intr=False):

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


    print('return_rdps', return_rdps)

    if not rgb_only:
        depth_channels = depth_channels
        try:
            if with_intr:
                depth_channels = depth_channels * 3
        except:
            pass

        vit_depth = FiT3DWrapper(
            arch=torch.hub.load("ywyue/FiT3D", dino_encoder),
            return_layers=vit_return_layers,
            norm_return_layers=norm_return_layers,
            in_chans=depth_channels,
            return_rdps=return_rdps,
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
        model = DinoRGBDWrapper(vit_rgb, vit_depth, return_layers=return_layers, return_rdps=return_rdps,
                                cat_outs=cat_outs, resize_layers=resize_layers,
                                down_sample_last_layer=down_sample_last_layer,
                                rand_disable_modality=rand_disable_modality)
    else:
        print('Fit3D RGB Wrapper')
        model = DinoRGBWrapper(vit_rgb, return_layers=return_layers,
                               resize_layers=resize_layers, down_sample_last_layer=down_sample_last_layer)
    return model



@MODELS.register_module()
class Fit3D(BaseModule):

    def __init__(self,
                 freeze_encoder=True,
                 rgb_only=False,
                 encoder_path=None,
                 version='base',
                 depth_channels=32,
                 init_cfg=None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        self.encoder = get_fit3d_encoder(rgb_only=rgb_only, dino_size=version, depth_channels=depth_channels)
        
        if encoder_path is not None:
            print('loading from ', encoder_path)
            sd = torch.load(encoder_path, map_location='cpu')['state_dict']
            self.encoder = load_fitting_state_dict(self.encoder, sd)

            #input('Are the weights loaded corret in the backbone?')
        
        
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder



    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #
        #if not self.training:
        #    print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        
        #rint('input size inside MultiMAE', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]

      
        return embs




def get_dune_encoder(return_layers=True, vit_return_layers=None, rgb_only=False, cat_outs=False,
                       fusion_layer_ids=None, resize_layers=True, down_sample_last_layer=True,
                       norm_return_layers=False, return_before_fusion=None,
                       norm_return_layers_rgb=False, rand_disable_modality=False,
                       add_depth_scales_to_cls_token=False, dino_size='small', fuse_cls_tolken=False, depth_channels=32, return_rdps=False,
                       with_intr=False, pretrained_path=None, encoder_path=None):

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
        image_size=448, num_register_tokens=4, layerscale_init=0.0001, qkv_bias=True, ln_affine=True,
        block_chunks=0, norm_return_layers=norm_return_layers_rgb
    )


    print('return_rdps', return_rdps)

    if not rgb_only:
        depth_channels = depth_channels
        try:
            if with_intr:
                depth_channels = depth_channels * 3
        except:
            pass

        vit_depth = dino_encoder(
            patch_size=14,
            in_chans=depth_channels,
            return_layers=vit_return_layers,
            return_rdps=return_rdps,
            fusion_layer_ids=fusion_layer_ids,
            add_fusion=True, image_size=448, num_register_tokens=4, layerscale_init=0.0001, qkv_bias=True, ln_affine=True,
            norm_return_layers=norm_return_layers,
            return_before_fusion=return_before_fusion,
            add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
            fuse_cls_tolken=fuse_cls_tolken
        )


    else:
        print('RGB Only encoder')

    if isinstance(pretrained_path, str):
        if os.path.exists(pretrained_path):
            sd = {}
            n_ = 8
            for sdkey, sdv in torch.load(pretrained_path, map_location='cpu')['model'].items():
                if 'teacher' in sdkey or 'projectors' in sdkey:
                    continue
                if 'encoder.' == sdkey[:n_]:
                    sdkey = sdkey[n_:]
                sdkey = sdkey.replace('blocks.0.', 'blocks.')
                sd[sdkey] = sdv

            vit_rgb = load_fitting_state_dict(vit_rgb, sd)
            print('loaded state dict on vit rgb from: {}'.format(pretrained_path))
            
            
            if not rgb_only:
                sd = {}
                n_ = 8
                for sdkey, sdv in torch.load(pretrained_path, map_location='cpu')['model'].items():
                    if 'teacher' in sdkey or 'projectors' in sdkey:
                        continue
                    if 'encoder.' == sdkey[:n_]:
                        sdkey = sdkey[n_:]
                    #sdkey = sdkey.replace('blocks.0.', 'blocks.')
                    sd[sdkey] = sdv
                
                vit_depth = load_fitting_state_dict(vit_depth, sd)
                #vit_depth.load_state_dict(sd)
                print('loaded state dict on vit depth from: {}'.format(pretrained_path))
        #input()
    

    if not rgb_only:
        print('Fit3D RGBD Wrapper')
        model = DinoRGBDWrapper(vit_rgb, vit_depth, return_layers=return_layers, return_rdps=return_rdps,
                                cat_outs=cat_outs, resize_layers=resize_layers,
                                down_sample_last_layer=down_sample_last_layer,
                                rand_disable_modality=rand_disable_modality)
    else:
        print('Fit3D DinoRGB Wrapper')
        model = DinoRGBWrapper(vit_rgb, return_layers=return_layers,
                               resize_layers=resize_layers, down_sample_last_layer=down_sample_last_layer)
    return model


@MODELS.register_module()
class DuneVit(BaseModule):
     
     

    def __init__(self,
                 img_size=448,
                 patch_size=14,
                 rgb_only=False,
                 encoder_path=None,
                 pretrained_path=None,
                 version='base',
                 freeze_encoder=True,
                 depth_channels=32,
                 init_cfg =None,
                 **kwargs

                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

            
        if not isinstance(rgb_only, bool):
            raise TypeError(f"'rgb_only' must be a boolean value , but got {type(rgb_only).__name__}")

        if encoder_path is not None and not isinstance(encoder_path, str):
            raise TypeError(f"'encoder_path' must be a string object, but got {type(encoder_path).__name__}")

        if pretrained_path is not None and not isinstance(pretrained_path, str):
            raise TypeError(f"'pretrained_path' must be a string object, but got {type(pretrained_path).__name__}")

        if not isinstance(version, str):
            raise TypeError(f"'version' must be a boolean value , but got {type(version).__name__}")


        self.img_size = img_size
        self.patch_size = patch_size
        self.rgb_only = rgb_only
        self.encoder_path = encoder_path
        self.pretrained_path = pretrained_path
        self.version = version
        self.depth_channels = depth_channels

        if init_cfg is not None and 'checkpoint' in init_cfg:
            self.pretrained_path = init_cfg['checkpoint']


        if init_cfg is not None and 'version' in init_cfg:
            dino_version = init_cfg['version']

        
        if self.version not in ['small', 'base','large']:
            raise ValueError(f"Invalid version '{self.version}'. Allowed versions for the DinoV2 models are 'small', 'base', 'large'.")
        #print('dino_version', dino_version)
        # print('init_cfg', init_cfg)

        # encoder_path = '/mnt/logicNAS/Exchange/paul/vanishing-depth-models/20240908_1725793388_Dino_UNet-Long32c-15m-Fix15m32c_448_encoder.plt'

        self.encoder = get_dune_encoder(rgb_only=self.rgb_only, pretrained_path=self.pretrained_path, dino_size=self.version, encoder_path=self.encoder_path, cat_outs=True, depth_channels=depth_channels)   
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
                
    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #
        if not self.training:
            print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        
        #print('input size inside RGBDVit', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        
        return embs


    
@MODELS.register_module()
class DFormer(BaseModule):

    def __init__(self,
                 freeze_encoder=True,
                 init_cfg=None,
                 pretrained_path=None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        self.encoder = DformerWrapper(upsample_outs=True, pretrained_path=pretrained_path)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder

    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #
        #if not self.training:
        #    print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        
        #print('input size inside MultiMAE', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]

        #for i, e in enumerate(embs):
        #    print(i, e.shape)
            
        return embs


    
@MODELS.register_module()
class DFormerV2(BaseModule):

    def __init__(self,
                 freeze_encoder=True,
                 init_cfg=None,
                 pretrained_path=None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        self.encoder = DformerV2Wrapper(upsample_outs=True, pretrained_path=pretrained_path)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder

    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #
        #if not self.training:
        #    print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        
        #print('input size inside MultiMAE', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
            
        return embs


@MODELS.register_module()
class CroCoV2(BaseModule):

    def __init__(self,
                 freeze_encoder=True,
                 init_cfg=None,
                 pretrained_path=None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        self.encoder = CroCoEncoderWrapper(upsample_outs=True, pretrained_path=pretrained_path)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder

    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #
        #if not self.training:
        #    print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        
        #print('input size inside MultiMAE', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
            
        return embs


@MODELS.register_module()
class MASt3R(BaseModule):

    def __init__(self,
                 freeze_encoder=True,
                 init_cfg=None,
                 pretrained_path=None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        self.encoder = MASt3REncoderWrapper(upsample_outs=True, pretrained_path=pretrained_path)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder

    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #
        #if not self.training:
        #    print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        
        #print('input size inside MultiMAE', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
            
        return embs


@MODELS.register_module()
class DMD3C(BaseModule):
    def __init__(self,
                 freeze_encoder=True,
                 init_cfg=None,
                 pretrained_path=None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        self.encoder = DMD3CWrapper()
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder
        
        #if self.rgb_only:
        #    self.interm_channels = [64, 128, 256, 512]
        #else:
        #    self.interm_channels = [128, 256, 512, 1024]

        #patch size 14

    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #

        
        #if not self.training:
        #    print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        
        #print('input size inside MultiMAE', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
            
        return embs

@MODELS.register_module()
class OmniDC(BaseModule):

    def __init__(self,
                 freeze_encoder=True,
                 init_cfg=None,
                 pretrained_path=None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        self.encoder = OmniDCWrapper()
        #[64, 64, 128, 256]
        # patch 16
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder

    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #
        #if not self.training:
        #    print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        
        #print('input size inside MultiMAE', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
            
        return embs

@MODELS.register_module()
class DepthAnythingV2(BaseModule):

    def __init__(self,
                 freeze_encoder=True,
                 init_cfg=None,
                 version='base',
                 pretrained_path=None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        self.encoder = DepthAnythingV2Wrapper(encoder=version)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder

    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) #
        #if not self.training:
        #    print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()
        #self.interm_channels = [768, 768, 768, 768]
        # patch 14
        
        #print('input size inside MultiMAE', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
            
        return embs


@MODELS.register_module()
class DinoV2DepthAnythingV2(BaseModule):

    def __init__(self,
                 freeze_encoder=True,
                 init_cfg=None,
                 version='base',
                 pretrained_path=None,
                 **kwargs
                 ):
        # print(f'RGBDVit __init__ called with: img_size ={img_size} , patch_size:{patch_size}, init_cfg={init_cfg} , kwargs:{kwargs}')
        super().__init__(init_cfg=init_cfg)

        self.encoder = DinV2andDepthAnythingV2Wrapper(encoder=version)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder

    def forward(self, inputs):
        
        # embs = self.encoder(inputs[:, :3], depth[:, 4:], inputs[:, 3].flatten(1)[:self.encoder.out_channels]) # for RGBD
       
        inputs = inputs.to(next(self.parameters()).device) 

        #self.interm_channels = [768*2, 768*2, 768*2, 768*2]
        # patch 14        
        #print('input size inside MultiMAE', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)[1:-1]
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)[1:-1]
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])[1:-1]
            
        return embs
