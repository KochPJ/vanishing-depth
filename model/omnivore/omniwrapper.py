
import os.path
import torch.nn as nn
import torch
import warnings
import math
import numpy as np
import torch.nn.functional as F


class OmniVoreWrapper(nn.Module):
    def __init__(self, version='swinB', resize_layers=True, down_sample_last_layer=True, n_return_layers=4, num_classes=0, freeze_encoder=True):
        super().__init__()

        if version in ['swinB', 'swinL']:
            model_name = "omnivore_{}".format(version + '_imagenet21k')
        else:
            model_name = "omnivore_{}".format(version)

        self.version = version
        self.model_name = model_name       
        self.color_encoder = torch.hub.load("facebookresearch/omnivore:main", model=model_name).trunk  # rgbd encoder, naming for lr assignment
        self.patch_size = (16, 16)
        self.down_sample_last_layer = down_sample_last_layer
        self.n_return_layers = n_return_layers
        self.resize_layers = resize_layers
        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            print('Freezing Omnivore Encoder')
            for param in self.color_encoder.parameters():
                param.requires_grad = False
        
        self.interm_channels =  {
            'swinT': [192, 384, 768, 768],
            'swinS': [192, 384, 768, 768],
            'swinB': [256, 512, 1024, 1024],
            'swinL': [384, 768, 1536, 1536],
        }[self.version]

        self.out_channels = self.interm_channels[-1]
        if num_classes > 0:
            layer_name = 'stage' # applies the norm (is trained for classification)
        else:
            layer_name = 'interim' # does not apply the norm

        self.num_classes = num_classes
        self.feature_keys = sorted(['{}{}'.format(layer_name if index < 2 else 'interim' , 3 - index) for index in range(self.n_return_layers)]) if self.n_return_layers else None

        self.fc = None
        if self.num_classes > 0:
            if self.n_return_layers > 1:
                self.out_channels = sum(self.interm_channels[-self.n_return_layers])
                
            self.fc = nn.Linear(self.out_channels, num_classes)
            self.out_channels = num_classes
        

    def forward(self, x, xd=None, depth_scales=None):
        #print('max', torch.max(xd))
        _, _, h, w = x.shape

        #print('x', x.shape)
        if xd is not None:
            if xd.dim() == 3:
                xd = xd.unsqueeze(1)
            #print('xd', xd.shape)
        h_, w_ = h / self.patch_size[0], w / self.patch_size[1]
        h_decode, w_decode = (h // 32)*32, (w // 32)*32

        if xd is not None:
            x = torch.cat([x, xd], dim=1)[:, :, None, ...]
        else:
            x = x[:, :, None, ...]

        if self.freeze_encoder:
            with torch.no_grad():
                self.color_encoder.eval()
                features = self.color_encoder(x, out_feat_keys=self.feature_keys)  
        else:
            features = self.color_encoder(x, out_feat_keys=self.feature_keys)  

        if self.fc is not None:
            if isinstance(features, list):
                features = torch.cat([torch.mean(feat, [-3, -2, -1]) for feat in features], dim=-1)              
            return self.fc(features)

        else:
            out = []
            if self.resize_layers:
                for i, x in enumerate(features):
                    t_size = (int(h_decode / (2**(i+1))), int(w_decode / (2**(i+1))))
                    out.append(F.interpolate(x[:, :, 0], t_size, None, 'bilinear', None, recompute_scale_factor=None))
                    #print(out[-1].shape)
            else:
                out = [x[:, :, 0] for x in features]

            if self.down_sample_last_layer:
                t_size = (int(h_decode / (2**(len(features)+1))), int(w_decode / (2**(len(features)+1))))
                out.append(F.interpolate(features[-1][:, :, 0], t_size, None, 'bilinear', None, recompute_scale_factor=None))
            
            out = [None] + out # for the unet decoder which ignores the first (input) layer with same resolution
            out.append(torch.mean(out[-1], [-2, -1]))
            return out

