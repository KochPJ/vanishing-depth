import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import json
import numpy as np
import torchvision.transforms as transforms
import argparse

import cv2
from os import listdir
import json
import numpy as np
import time
from math import cos, sin
import matplotlib.pyplot as plt
import random

'''
get object150_info.csv from
 https://github.com/CSAILVision/sceneparsing/blob/master/convertFromADE/mapFromADE.txt
 https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
 
'''


class ADE20K(data.Dataset):
    def __init__(self, transforms=None, root=None, mode=None, balanced_sampling=False, sample_with_p=False):
        if root is None:
            root = './data/ADE20K_2021_17_01'

        lut = {}
        self.classes = []
        self.mode = mode
        with open(os.path.join(root, 'object150_info.csv'), 'r') as f:
            lines = f.readlines()
            for i, l in enumerate(lines):
                lines[i] = l.replace('\n', '').split(',')
                if i == 0:
                    for key in lines[i]:
                        lut[key] = []
        keys = list(lut.keys())
        for l in lines[1:]:
            for v, key in zip(l, keys):
                if key != 'Name':
                    if key == 'Ratio':
                        v = float(v)
                    else:
                        v = int(v)
                else:
                    self.classes.append(v)
                lut[key].append(v)


        self.lut = {}
        self.root = root
        self.transforms = transforms
        self.dirs = []
        self.class_names = self.classes
        self.balanced_sampling = balanced_sampling
        self.sample_with_p = sample_with_p

        with open(os.path.join(root, 'mapFromADE.txt'), 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.replace('\n', '-').replace(' ', '-').replace('\t', '-').split('-')
                nl = []
                for v in l:
                    if v == '-' or v == '':
                        continue
                    else:
                        nl.append(v)

                if len(nl) != 2:
                    raise ValueError(l, nl)
                self.lut[int(nl[1])] = int(nl[0])

        self.names = {i+1: {'name': name,
                            'samples': []} for i, name in enumerate(self.classes)}

        self.num_classes = len(self.classes)

        cfg_path = os.path.join(root, '{}_segmentation_config.json'.format(mode))
        if not os.path.exists(cfg_path) or mode != 'train' or (mode == 'train' and not self.balanced_sampling):
            m = {'train': 'training', 'valid': 'validation'}[mode]
            for root, folders, files in os.walk(os.path.join(root, 'images', 'ADE', m)):
                for f in files:
                    #print(f)
                    if '.jpg' not in f:
                        continue
                    rgb = os.path.join(root, f)
                    mask = rgb.replace('.jpg', '_seg.png')
                    info = rgb.replace('.jpg', '.json')

                    if not os.path.exists(rgb):
                        raise ValueError('rgb not exist', rgb)

                    if not os.path.exists(mask):
                        raise ValueError('mask not exist', mask)
                    if not os.path.exists(mask):
                        raise ValueError('info not exist', info)

                    if self.balanced_sampling and self.mode == 'train':
                        s = self.load_sample([rgb, mask, info])
                        uni = torch.unique(s['mask'])
                        for u in uni:
                            u = int(u)
                            if u == 0:
                                continue
                            else:
                                self.names[u]['samples'].append([rgb, mask, info])
                    self.dirs.append([rgb, mask, info])

            if self.balanced_sampling and self.mode == 'train':
                cfg = {'names': self.names, 'dirs': self.dirs}
                with open(cfg_path, 'w') as f:
                    json.dump(cfg, f)
        else:
            print('loading config from {}'.format(cfg_path))
            with open(cfg_path, 'rb') as f:
                cfg = json.load(f)
            self.names = cfg['names']
            self.dirs = cfg['dirs']

        if self.balanced_sampling and self.mode == 'train' and self.sample_with_p:
            self.ps = np.array([1 - (len(v['samples']) / len(self.dirs)) for v in self.names.values()])
            self.ps = self.ps / np.sum(self.ps)
            #print(np.sum(self.ps))
            #print(self.ps)
            if np.sum(self.ps) != 1.0:
                idx = np.argmax(self.ps)
                self.ps[idx] += 1 - np.sum(self.ps)
        else:
            self.ps = None

    def __getitem__(self, index):
        if self.balanced_sampling and self.mode == 'train':
            if self.ps is not None:
                #print('with ps')
                index = str(int(np.random.choice(list(self.names.keys()), size=1, p=self.ps)))
            else:
                index = str(int(np.random.choice(list(self.names.keys()), size=1)))

            path = np.random.choice(list(range(len(self.names[index]['samples']))), size=1)
            path = self.names[index]['samples'][int(path)]
        else:
            path = self.dirs[index]

        sample = self.load_sample(path)
        if self.transforms is not None:
            s = self.transforms(sample)
            sample = {}
            for key, v in s.items():
                if key in ['img', 'mask']:
                    sample[key] = v.float()
        return sample

    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array

        img_path, mask_path, info = path

        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        with open(mask_path, 'rb') as f:
            mask = np.array(Image.open(f).convert('RGB'))

        mask = (np.int32(mask[:, :, 0]) / 10) * 256 + np.int32(mask[:, :, 1])
        sample['mask'] = np.zeros(mask.shape, dtype=np.float32)

        #with open(info) as f:
        #    content = json.load(f)['annotation']['object']

        #name = {c['name_ndx']: c['raw_name'] for c in content}
        #print(name)

        for u in np.unique(mask):
            if u == 0:
                continue
            index = self.lut.get(int(u))
            if index is None:
                #print('u', int(u), name[u])
                continue
            sample['mask'][mask == u] = index
        sample['mask'] = torch.from_numpy(sample['mask']).float()
        return sample



    def plot_sample(self, sample):
        plt.subplot(1, 2, 1)
        plt.imshow(sample['img'])

        plt.subplot(1, 2, 2)
        plt.imshow(sample['mask'])
        #print(sample['mask'].shape)
        plt.show()


if __name__ == '__main__':
    ds = ADE20K(None, mode='train', balanced_sampling=True, sample_with_p=True)
    print('len', len(ds))
    #print(ds.lut)
    print(ds.num_classes)
    #for key, v in ds.lut.items():
    #    print(key, v)

    for key, v in ds.names.items():
        print(key, len(v['samples']) / len(ds) * 100, v['name'])

    cls = {}
    for i in range(len(ds)):
        print('{} / {}'.format(i+1, len(ds)))
        sample = ds.__getitem__(i)

        h, w = sample['mask'].shape
        a = h*w
        uni, counts = torch.unique(sample['mask'], return_counts=True)
        #coutns = counts/a

        for u_, c_ in zip(uni, counts):
            c = float(c_) / a
            u = int(u_)
            if u == 0:
                continue
            elif u not in cls:
                cls[u] = []
                print(len(cls), sorted(list(cls.keys())))
            cls[u].append(c)

        if i > 1000:
            break
        #ds.plot_sample(sample)

    print('######################################')
    for c, v in cls.items():
        cls[c] = sum(v) / len(ds)
        #print(cls, b)

    vs = list(cls.values())
    s = sorted(vs)
    keys = list(cls.keys())
    indx = [keys[vs.index(s_)] for s_ in s]

    for i, k in enumerate(indx):
        print(i, k, cls[k])



