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
import torch.nn.functional as F
from functools import partial
from model.omnivore.omniwrapper import OmniVoreWrapper
from utils.stuff import load_fitting_state_dict
from datasets.transforms import DepthPositionalEncoding
from model.dino import DinoRGBDWrapper, DinoRGBWrapper
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from mmengine.model import BaseModule, ModuleList
from mmseg.registry import MODELS

import sys
sys.path.append('/home/chowanki/git/vanishing-depth-self-supervised')
from my_mmseg.utils.loading import log_message



@MODELS.register_module()
class Omnivore(BaseModule):
     
     

    def __init__(self,
                 img_size=224,
                 patch_size=14,
                 rgb_only= False,
                 encoder_path = None,
                 pretrained_path= None,
                 version='swinB',
                 freeze_encoder=True,
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

        if init_cfg is not None and 'checkpoint' in init_cfg:
            self.pretrained_path = init_cfg['checkpoint']


        if init_cfg is not None and 'version' in init_cfg:
            dino_version = init_cfg['version']

        
        #if self.version not in ['small', 'base','large']:
        #    raise ValueError(f"Invalid version '{self.version}'. Allowed versions for the DinoV2 models are 'small', 'base', 'large'.")
        #print('dino_version', dino_version)
        # print('init_cfg', init_cfg)

        # encoder_path = '/mnt/logicNAS/Exchange/paul/vanishing-depth-models/20240908_1725793388_Dino_UNet-Long32c-15m-Fix15m32c_448_encoder.plt'

        self.encoder = OmniVoreWrapper(
            version=version, 
            resize_layers=True,
            down_sample_last_layer=False, 
            n_return_layers=4, 
            num_classes=0, 
            freeze_encoder=freeze_encoder
            )

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        #print('freeting ', self.freeze_encoder)
        #input()
        
    def forward(self, inputs):
        
        inputs = inputs.to(next(self.parameters()).device) #
        #if not self.training:
        #    print('inputs in VIT14RGBD', inputs.shape) # ([1, 35, 1036, 2058])
        # input()

        #print('input size inside RGBDVit', inputs.size()) # debug
        if self.freeze_encoder:
            with torch.no_grad():
                if len(inputs[0]) == 3:
                    embs = self.encoder(inputs)
                else:
                    embs = self.encoder(inputs[:, :3], inputs[:, 3:])
        else:
            if len(inputs[0]) == 3:
                embs = self.encoder(inputs)
            else:
                embs = self.encoder(inputs[:, :3], inputs[:, 3:])

        #for i, e in enumerate(embs):
        #    if e is not None:
        #        print(i, type(e), e.shape) 
        #    else:
        #        print(i, type(e))
        #input()
            
        return embs[1:-1]  # first one is None and last one is the cls token, in between we have the feature maps
