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

class ADE20K(data.Dataset):
    def __init__(self, transforms=None, root=None, mode=None, balanced_sampling=False, sample_with_p=False,
                 with_cls_classes=False, with_mask=True, mode_tag='segmentation'):
        if root is None:
            root = './data/ADEChallengeData2016'

        lut = {}
        self.classes = []
        self.mode = mode
        self.with_cls_classes = with_cls_classes
        self.with_mask = with_mask
        self.mode_tag = mode_tag

        with open(os.path.join(root, 'objectInfo150.txt'), 'r') as f:
            lines = f.readlines()
            for i, l in enumerate(lines):
                lines[i] = l.replace('\n', '').replace(' ', '').split('\t')
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
                        try:
                            v = int(v)
                        except Exception as e:
                            print(key, v, l)
                            raise e

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

        self.cls_samples = {}
        self.cls_lut = {}
        with open(os.path.join(root, 'sceneCategories.txt'), 'r') as f:
            lines = f.readlines()
            for i, l in enumerate(lines):
                path, cls = l.replace('\n', '').split(' ')
                if cls not in self.cls_samples:
                    self.cls_samples[cls] = []
                self.cls_samples[cls].append(path)
                self.cls_lut[path] = cls

        if self.mode_tag == 'segmentation':
            self.names = {str(i + 1): {'name': name,
                                  'samples': []} for i, name in enumerate(self.classes)}
            self.num_classes = len(self.classes)
        elif self.with_cls_classes:

            self.cls_classes = list(self.cls_samples.keys())
            self.names = {str(i): {'name': name,
                              'samples': []} for i, name in enumerate(self.cls_classes)}
            self.num_classes = len(self.cls_classes)

        cfg_path = os.path.join(root, '{}_{}_config.json'.format(mode, mode_tag))
        if not os.path.exists(cfg_path) or mode != 'train' or (mode == 'train' and not self.balanced_sampling):
            m = {'train': 'training', 'valid': 'validation', 'test': 'validation'}[mode]
            for root, folders, files in os.walk(os.path.join(root, 'images', m)):
                for f in files:
                    #print(f)
                    if '.jpg' not in f:
                        continue
                    rgb = os.path.join(root, f)
                    mask = rgb.replace('/images/', '/annotations/').replace('.jpg', '.png')

                    if not os.path.exists(rgb):
                        raise ValueError('rgb not exist', rgb)

                    if not os.path.exists(mask):
                        raise ValueError('mask not exist', mask)

                    img_name = f.replace('.jpg', '')
                    if self.balanced_sampling and self.mode == 'train':
                        s = self.load_sample([rgb, mask, img_name])
                        if self.mode_tag == 'segmentation' and self.with_mask:
                            uni = torch.unique(s['mask'])
                            for u in uni:
                                u = str(int(u))
                                if u == 0:
                                    continue
                                else:
                                    self.names[u]['samples'].append([rgb, mask, img_name])

                        elif self.mode_tag == 'classification' and self.with_cls_classes:
                            self.names[str(s['cls'])]['samples'].append([rgb, mask, img_name])

                    self.dirs.append([rgb, mask, img_name])

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
        if self.balanced_sampling and self.mode == 'train' and False:
            if self.ps is not None:
                #print('with ps')
                index = str(int(np.random.choice(list(self.names.keys()), size=1, p=self.ps)))
                #print('here')
            else:
                index = str(int(np.random.choice(list(self.names.keys()), size=1)))

            path = np.random.choice(list(range(len(self.names[index]['samples']))), size=1)
            path = self.names[index]['samples'][int(path)]
        else:
            path = self.dirs[index]

        sample = self.load_sample(path)
        if self.transforms is not None:
            sample = self.transforms(sample)
            for key, v in sample.items():
                if key in ['img', 'mask']:
                    sample[key] = v.float()
        return sample

    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array
        try:
            img_path, mask_path, img_name = path
        except:
            img_path, mask_path = path
            img_name = None

        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        if self.with_mask:
            with open(mask_path, 'rb') as f:
                mask = np.array(Image.open(f))

            sample['mask'] = torch.from_numpy(mask).float()

        if self.with_cls_classes:
            sample['cls'] = int(self.cls_classes.index(self.cls_lut[img_name]))
            pass

        return sample

    def plot_sample(self, sample):
        plt.subplot(1, 2, 1)
        plt.imshow(sample['img'])
        if 'cls' in sample:
            plt.title('{} | {}'.format(sample['cls'], self.names[str(sample['cls'])]['name']))

        if 'mask' in sample:
            plt.subplot(1, 2, 2)
            plt.imshow(sample['mask'])
            #print(sample['mask'].shape)
        plt.show()


if __name__ == '__main__':
    ds = ADE20K(None, mode='train', balanced_sampling=True, sample_with_p=False,
                mode_tag='classification', with_mask=False, with_cls_classes=True)
    print('len', len(ds))
    #print(ds.lut)
    print(ds.num_classes)
    #for key, v in ds.lut.items():
    #    print(key, v)

    for key, v in ds.names.items():
        if len(v['samples']) > 0:
            print(key, len(v['samples']) / len(ds) * 100, v['name'])

    cls = {}
    for i in range(len(ds)):
        print('{} / {}'.format(i+1, len(ds)))
        sample = ds.__getitem__(i)

        if 'mask' in sample:
            #ds.plot_sample(sample)
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
        elif 'cls' in sample:
            c = str(sample['cls'])
            if c not in cls:
                cls[c] = 1
                for key, v in cls.items():
                    print(key, v)
            else:
                cls[c] += 1
        #ds.plot_sample(sample)


    print('######################################')
    n = 3
    counter = 0
    for key, v in cls.items():
        if v >= n:
            print(key, v, ds.cls_classes[int(key)])
            counter += 1

    print(counter, np.max(list(cls.values())))

    input()
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



