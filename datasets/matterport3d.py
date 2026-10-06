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


class Matterport(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True,
                        with_mask=False, load_from_checkpoint=True,
                     depth_mul=1, with_bbox=False, intr_mat=False, mode=None):
        if root is None:
            root = './data/matterport'

        self.me = 'matterport'
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
                if 'matterport_color_images' not in folders:
                    continue
                    
                for img_id in os.listdir(os.path.join(path, 'matterport_color_images')):
                    tag = img_id.split('.')[0]
                    try:
                        name, image, version = tag.split('_')
                    except:
                        continue

                    self.dirs.append([path, False, name, image, version])                
        
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
        path, undistored, name, image, version = path
        depth = image.replace('i', 'd')
        if undistored:
            color_path, depth_path, intr_path = 'undistorted_color_images', 'undistorted_depth_images', ''
        else:
            color_path, depth_path, intr_path = 'matterport_color_images', 'matterport_depth_images', 'matterport_camera_intrinsics'



        img_path = os.path.join(path, color_path, name + '_' + image + '_' + version + '.jpg')
        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')


        if self.with_intr or self.with_depth:
            sample['depth_scale'] = 0.25
            if self.depth_mul is not None:
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul
                sample['depth_mul'] = self.depth_mul

            if intr_path:
                intr_path = os.path.join(path, intr_path, name + '_intrinsics_' + image.replace('i', '')  + '.txt')
                with open(intr_path) as f:
                    cam = [line.rstrip() for line in f][0]
                    w, h, cx, cy, fx, fy, d0, d1, d2, d3, d4 = [float(v) for v in cam.split(' ')]
                    
                if self.intr_mat:    
                    intr = np.zeros((3, 3))
                    intr[2, 2] = 1
                    intr[0, 0] = fx
                    intr[1, 1] = fy
                    intr[0, 2] = cx
                    intr[1, 2] = cy                
                    sample['intr'] = torch.as_tensor(intr, dtype=torch.float32)
                else:
                    #cx, cy, fx, fy = intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
                    sample['intr'] = [cx, cy, fx, fy]

        if  self.with_depth:
            # read depth information to numpy array           
            depth_path = os.path.join(path, depth_path, name + '_' + depth + '_' + version + '.png')
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
    ds = Matterport(None, with_depth=True, root='/mnt/wimi/publicdata/matterport/', mode=None, with_mask=False, with_intr=True)

    print(len(ds))
    print(ds.names)
    #indexes = np.random.permutation(len(ds))
    indexes = range(len(ds))


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
        m = float(torch.mean(d[d > 0]))
        s = float(torch.std(d[d > 0]))

        if np.isnan(m) or np.isnan(s) or np.isinf(m) or np.isinf(s):
            continue


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
            
    print('mean', np.nanmean(mean))
    print('std', np.nanmean(std))
    print('zeros', np.nanmean(zeros))
        
    if len(mean_gt) > 0:
        print('mean_gt', np.nanmean(mean_gt))
        print('std_gt', np.nanmean(std_gt))
        print('zeros_gt', np.nanmean(zeros_gt))
        


    #for i in indexes:
    #    sample = ds.__getitem__(i)
    #    ds.plot_sample(sample)
