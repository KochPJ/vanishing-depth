import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import transforms3d as t3d
from pathlib import Path
import struct
import math


class IPKData(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True, mode_tag=None,
                 with_mask=False, load_from_checkpoint=True,  depth_mul=1, with_bbox=False):
        
        self.root = './data/gurkenumgebung'
        self.ime = './data/gurke-ime'
        self.ipk = './data/gurkenipkrgbd'
        self.transforms = transforms
        self.with_intr = with_intr
        self.with_depth = with_depth
        self.with_mask = with_mask
        self.names = {'0': 'ipk'}
        self.depth_mul = depth_mul
        self.with_bbox = with_bbox      
        

        self.dirs = []
        
        for root, dirs, files in os.walk(self.ime):
            if files:
                print(root, len(files))
            for f in files:
                if '_depth.png' in f:
                    depth_path = os.path.join(root, f)
                    img_path = os.path.join(root, f.replace('_depth.png', '_rgb.png'))
                    scale = 1
                    max_depth = 3000
                    if os.path.exists(img_path) and os.path.exists(depth_path):
                        self.dirs.append((img_path, depth_path, scale, max_depth))
        
        for root, dirs, files in os.walk(self.root):
            
            if files:
                print(root, len(files))        
            
            for f in files:
                if 'depth_' in f:
                    depth_path = os.path.join(root, f)
                    img_path = depth_path.replace('depth_', 'rgb_')
                    scale = 1
                    max_depth = 3500
                    if os.path.exists(img_path) and os.path.exists(depth_path):
                        self.dirs.append((img_path, depth_path, scale, max_depth))      
       
        for root, dirs, files in os.walk(self.ipk):
            if files:
                print(root, len(files))        
            
            if 'depth_mm' in root:
                for f in files:
                    depth_path = os.path.join(root, f)
                    img_path = depth_path.replace('depth_mm', 'rgb')
                    scale = 1
                    max_depth = 5000
                    if os.path.exists(img_path) and os.path.exists(depth_path):
                        self.dirs.append((img_path, depth_path, scale, max_depth))      

        self.class_names = ['ipk']
        self.num_classes = len(self.class_names)

    def __getitem__(self, index):
        path = self.dirs[index]
        sample = self.load_sample(path)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array
        img_path, depth_path, depth_scale, max_depth = path
  

        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        if self.with_depth:
            
            sample['depth_scale'] = depth_scale
            if self.depth_mul is not None:
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul
                sample['depth_mul'] = self.depth_mul

       
            sample['depth'] = torch.from_numpy(np.array(Image.open(depth_path), dtype=np.float32))
            sample['depth'][sample['depth'] > max_depth] = 0

       
        return sample


    def plot_sample(self, sample):
        plt.subplot(2, 2, 1)
        plt.imshow(sample['img'])
        if 'depth' in sample:
            d = sample['depth']
            if 'depth_scale' in sample:
                d = d * sample['depth_scale']
                d = d / 1000
                
            plt.subplot(2, 2, 2)
            plt.imshow(d)

            
            d2 = math.pi * 2 * d
            plt.subplot(2, 2, 3)
            plt.imshow(d2.sin())
           

            d3 =  math.pi * 2 * d*20
            plt.subplot(2, 2, 4)
            plt.imshow(d3.sin())
            
            


    
        print('show')
        plt.show()

if __name__ == '__main__':

    import sys
    sys.path.append(os.getcwd())

    ds = IPKData(None, with_depth=True)

    print(len(ds))
    print('names', ds.names)
    indexes = np.random.permutation(len(ds))
    #indexes = range(len(ds))
    #for i in indexes:
    #    print(i)
    #    sample = ds.__getitem__(i)
    #    ds.plot_sample(sample)
    
    for i in indexes:
        sample = ds.__getitem__(i)
        
        print('{} / {} | {}'.format(i+1, len(ds), sample['depth'].shape))
        print('intr', sample.get('intr'))
        ds.plot_sample(sample)
        