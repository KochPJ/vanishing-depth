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


class Scannet(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True,
                        with_mask=False, load_from_checkpoint=True,
                     depth_mul=1, with_bbox=False, intr_mat=False, mode=None):
        if root is None:
            root = './data/scannet'

        self.me = 'scannet'
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
                if 'color' not in folders or 'depth' not in folders:
                    continue
                
                #print(path, len(os.listdir(os.path.join(path, 'color'))))
                for img_id in os.listdir(os.path.join(path, 'color')):
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
        path, img_id = path

        img_path = os.path.join(path, 'color', img_id + '.jpg')
        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')


        if self.with_intr or self.with_depth:
            sample['depth_scale'] = 1
            if self.depth_mul is not None:
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul
                sample['depth_mul'] = self.depth_mul

            if self.with_intr:
                intr_path = os.path.join(path, 'intrinsic', 'intrinsic_color.txt')
                with open(intr_path) as f:
                    cam = [line.rstrip() for line in f]
                    intr = np.array([[float(v) for v in c.split(' ')] for c in cam]).flatten().reshape((4, 4))[:3, :3]
                    #print(intr)
                    
                if self.intr_mat:
                    sample['intr'] = torch.as_tensor(intr, dtype=torch.float32)
                else:
                    cx, cy, fx, fy = intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
                    sample['intr'] = [cx, cy, fx, fy]

        if  self.with_depth:
            # read depth information to numpy array           
            depth_path = os.path.join(path, 'depth', img_id + '.png')
            sample['depth'] = Image.open(depth_path)
                    
            sample['img'] = sample['img'].resize(sample['depth'].size)
            sample['depth'] = torch.as_tensor(np.array(sample['depth'], dtype=np.float32), dtype=torch.float32)

        if self.with_mask:
            mask_path = os.path.join(path, 'label', img_id + '.png')
            #print(os.path.exists(mask_path))
            sample['mask'] = Image.open(mask_path)
            if self.with_depth:
                sample['mask'] = sample['mask'].resize(sample['img'].size) #, resample=Image.NEAREST)
            sample['mask'] = np.array(sample['mask'])
            
        return sample

    def load_ann(self, path):
        with open(os.path.join(path, 'scene_gt.json')) as f:
            data = json.load(f)
        return data

    def plot_sample(self, sample):
        plt.subplot(2, 3, 1)
        plt.imshow(sample['img'])
        plt.title('img')
        if 'depth' in sample:
            d = sample['depth']
            if 'depth_scale' in sample:
                d = d * sample['depth_scale']
                d = d / 1000
            plt.subplot(2, 3, 2)
            plt.imshow(d)
            plt.title('depth')

            depth = sample['depth'].numpy()
            depth = np.array(depth / np.max(depth) * 255, dtype=np.uint8)
            color = np.array(sample['img'])
            color[:, :, 0] = depth
            plt.subplot(2, 3, 3)
            plt.imshow(color)
            plt.title('overlay')
            
    
        if 'mask' in sample:
            plt.subplot(2, 3, 4)
            plt.imshow(sample['mask'])
            
        
        plt.show()

if __name__ == '__main__':
    ds = Scannet(None, with_depth=True, root='/mnt/wimi/publicdata/scannet/', mode=None, with_mask=False, with_intr=True, 
                    load_from_checkpoint=False)

    print(len(ds))
    print(ds.names)
    #indexes = np.random.permutation(len(ds))
    ##indexes = range(len(ds))
    #for i in indexes:
    #    sample = ds.__getitem__(i)
    #    ds.plot_sample(sample)

    

    mean = []
    mean_gt = []
    std = []
    std_gt = []
    zeros = []
    zeros_gt = []
    
    for i in range(len(ds)):
        try:
            sample = ds.__getitem__(i)
        except:
            continue
        #print('{} / {} | {}'.format(i+1, len(ds), sample['depth'].shape))
        #print('intr', sample.get('intr'))
        ds.plot_sample(sample)
        d = sample['depth'] * sample['depth_scale'] / 1000
        mean.append(torch.mean(d[d > 0]))
        std.append(torch.std(d[d > 0]))
        zeros.append( 1 - (d.numpy() > 0) / np.prod(d.numpy().shape))

        if 'gt_depth' in sample:
            d = sample['gt_depth'] * sample['depth_scale'] / 1000
            mean_gt.append(torch.mean(d[d > 0]))
            std_gt.append(torch.std(d[d > 0]))
            zeros_gt.append( 1 - (d.numpy() > 0) / np.prod(d.numpy().shape))
        
        if i%1000 == 0:
            print('{} / {} | {}, mean: {}, std: {}, zeros: {}'.format(i+1, len(ds), sample['depth'].shape, np.mean(mean), np.mean(std), np.mean(zeros)))
            
    print('mean', np.mean(mean))
    print('std', np.mean(std))
    print('zeros', np.mean(zeros))
        
    if len(mean_gt) > 0:
        print('mean_gt', np.mean(mean_gt))
        print('std_gt', np.mean(std_gt))
        print('zeros_gt', np.mean(zeros_gt))

