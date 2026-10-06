import os
import warnings

import numpy as np
import json
from tqdm import tqdm
import pickle
import torch.utils.data as data

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from pathlib import Path

class Uniformat(data.Dataset):
    def __init__(self, name, root='./data/uniformat_release', mode="train", depth_mul=1.0, transforms=None):

        name = name.replace('Omnirelease-', '')
        self.name = name
        self.root = os.path.join(root, name)
        self.depth_scale_multiplier = depth_mul
        self.mode = mode

        self.transforms = transforms

        print('Loading Uniformat...')
        print(self.root, len(list(os.listdir(self.root))))

        self.sample_list = sorted([os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith('.npy')])
        print('self.sample_list', len(self.sample_list))
        print( self.sample_list[0])
        # self.sample_list = [self.sample_list[k] for k in np.random.choice(range(len(self.sample_list)), 10, replace=False)]


    def __len__(self):
        return len(self.sample_list)
        # return 10

    def __getitem__(self, index):
        sample = self.load_sample(index)
        sample['mode'] = self.mode
        sample['sample_index'] = index
        #print(sample.keys())
        if self.transforms is not None:
            sample = self.transforms(sample)
            #sample = {}
            for key, v in sample.items():
                if key in ['img', 'depth', 'gt_depth', 'ori_depth']:
                    sample[key] = v.float()
            
            if 'depth_scales' in sample and sample.get('depth_scales') is None:
                del sample['depth_scales']

        #print(sample.keys())
        return sample

    def load_sample(self, idx):
        # image_file, depth_file, K = self.sample_list[idx]
        
        filedir = self.sample_list[idx]
        data_dict = dict(np.load(filedir, allow_pickle=True).item())

        rgb = Image.fromarray(data_dict['rgb'], mode='RGB')

        dep_sp = torch.as_tensor(data_dict['dep'], dtype=torch.float32)
        dep = torch.as_tensor(data_dict['gt'], dtype=torch.float32)
        
        if 'ETH3D' in filedir:
            K = torch.Tensor(
                [[425, 0, 320],
                 [0, 425, 240],
                 [0, 0, 1.0]]
            )
        elif 'iBims' in filedir:
            K = torch.Tensor(
                [[490, 0, 320],
                 [0, 490, 240],
                 [0, 0, 1.0]]
            )
        elif 'KITTI' in filedir:
            K = torch.Tensor(
                [[data_dict['K'][0], 0, data_dict['K'][2]],
                 [0, data_dict['K'][1], data_dict['K'][3]],
                 [0, 0, 1.0]]
            )
        else:
            K = torch.Tensor(data_dict['K'])

        dep_sp = dep_sp * self.depth_scale_multiplier
        dep = dep * self.depth_scale_multiplier



        output = {
            'img': rgb, 
            'depth': dep_sp, 
            'gt_depth': dep, 
            'depth_scale': 1000, #meter to mm
            #'K': K, 
            #'pattern': PATTERN_IDS[self.args.inference_pattern_type]
        }

        return output