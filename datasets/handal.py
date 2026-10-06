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


class HandalDataset(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True,
                        with_mask=False, load_from_checkpoint=True,
                     depth_mul=1, with_bbox=False, intr_mat=False, mode=None):
        if root is None:
            root = './data/handal'

        self.me = 'handal'
        self.root = root
        self.transforms = transforms
        self.with_intr = with_intr
        self.with_depth = with_depth
        self.with_mask = with_mask
        self.names = {}
        self.depth_mul = depth_mul
        self.with_bbox = with_bbox
        self.intr_mat = intr_mat

        if mode is not None:
            file_name = '{}_{}_dataset_config.json'.format(self.me, mode)
        else:
            file_name = '{}_dataset_config.json'.format(self.me)

        load_path = os.path.join(self.root, file_name)
        cp = None
        if load_from_checkpoint:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    cp = json.load(f)
        ids = []
        self.dirs = []
        if cp is None:
            for path, folders, files in os.walk(self.root):
                if 'rgb' not in folders and 'depth_nerf':
                    continue
                
                if mode == 'train':
                    if 'train' not in path:
                        continue
                elif mode == 'test':
                    if 'test' not in path:
                        continue
                    
                for img_id in os.listdir(os.path.join(path, 'rgb')):
                    self.dirs.append([path, img_id.split('.')[0]])                        
        
            with open(load_path, 'w') as f:
                json.dump(
                    {
                        'names': self.names,
                        'dirs': self.dirs
                        }, 
                        f)
        else:
            self.names = cp['names']
            self.dirs = cp['dirs']

        self.class_names = [self.names[key]['name'] for key in sorted(list(self.names.keys()))]
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
        path, idx = path


        img_path = os.path.join(path, 'rgb', idx + '.jpg')
        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')


        if self.with_intr or self.with_depth:
            sample['depth_scale'] = 1
            if self.depth_mul is not None:
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul
                sample['depth_mul'] = self.depth_mul

            intr = np.zeros((3,3))

            if self.intr_mat:
                sample['intr'] = torch.as_tensor(intr, dtype=torch.float32)
            else:
                cx, cy, fx, fy = intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
                sample['intr'] = [cx, cy, fx, fy]

        if  self.with_depth:
            # read depth information to numpy array
            depth_path = os.path.join(path, 'depth_nerf', idx + '.png')
            with open(depth_path, 'rb') as f:
                sample['depth'] = torch.as_tensor(np.array(Image.open(f), dtype=np.float32), dtype=torch.float32)
                #sample['depth'] = np.array(Image.open(f)).astype(np.float32)

        if self.with_mask:
            
            mask_path = os.path.join(path, 'mask', idx + '_000000.png')
            #print(os.path.exists(mask_path))
            with open(mask_path, 'rb') as f:
                sample['mask'] = np.array(Image.open(f))


        return sample

    def load_ann(self, path):
        with open(os.path.join(path, 'scene_gt.json')) as f:
            data = json.load(f)
        return data

    def plot_sample(self, sample):
        plt.subplot(2, 3, 1)
        plt.imshow(sample['img'])
        if 'depth' in sample:
            d = sample['depth']
            if 'depth_scale' in sample:
                d = d * sample['depth_scale']
                d = d / 1000
            plt.subplot(2, 3, 2)
            plt.imshow(d)
        if 'mask' in sample:
            plt.subplot(2, 3, 3)
            plt.imshow(sample['mask'])
        
        plt.show()

if __name__ == '__main__':
    ds = HandalDataset(None, with_depth=True, root='/mnt/wimi/publicdata/handal/', mode='test', with_mask=False)

    print(len(ds))
    print(ds.names)
    indexes = np.random.permutation(len(ds))
    indexes = range(len(ds))
    for i in indexes:
        sample = ds.__getitem__(i)
        ds.plot_sample(sample)
