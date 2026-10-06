import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import copy


class KittiDataset(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True, load_from_checkpoint=True, with_pc=False, mode='valid', use_depth_gt=False, depth_mul=None):
        if root is None:
            root = './data/kitti_depth_completion'
        self.root = root
        self.transforms = transforms
        self.with_intr = with_intr
        self.with_depth = with_depth
        self.use_depth_gt = use_depth_gt
        self.names = {}
        self.depth_mul = depth_mul
        self.cam_mat = np.array([[0.58, 0, 0.5, 0],
                                 [0, 1.92, 0.5, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], dtype=np.float32)

        self.dirs = []
        self.mode_path = {'valid': 'val_selection_cropped', 'test': 'test_depth_completion_anonymous'}[mode]
        for img_name in os.listdir(os.path.join(self.root, 'depth_selection', self.mode_path, 'image')):


            if mode == 'test':
                img_path = os.path.join(self.root, 'depth_selection', self.mode_path, 'image', img_name)
                depth_path = os.path.join(self.root, 'depth_selection', self.mode_path, 'velodyne_raw', img_name)
                self.dirs.append([img_path, depth_path, None])            
            else:
                #'2011_09_26_drive_0002_sync_image_0000000005_image_02'
                #'2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02'
                #'2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02'
                img_path = os.path.join(self.root, 'depth_selection', self.mode_path, 'image', img_name)
                depth_path = os.path.join(self.root, 'depth_selection', self.mode_path, 'velodyne_raw', img_name.replace('_sync_image_', '_sync_velodyne_raw_'))
                depth_gt_path = os.path.join(self.root, 'depth_selection', self.mode_path, 'groundtruth_depth', img_name.replace('_sync_image_', '_sync_groundtruth_depth_'))
                self.dirs.append([img_path, depth_path, depth_gt_path])

        
    def __getitem__(self, index):
        path = self.dirs[index]
        sample = self.load_sample(path)
        if self.transforms is not None:
            s = self.transforms(sample)
            sample = {}
            for key, v in s.items():
                if key in ['img', 'mask', 'depth', 'gt_depth', 'ori_depth', 'depth_scales']:
                    if v is not None:
                        sample[key] = v.float()
                if key in ['depth_mul', 'sample_id', 'height', 'width']:
                    sample[key] = v

        return sample

    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array
        img_path, depth_path, depth_gt_path = path


        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        if self.with_intr or self.with_depth:
            sample['depth_scale'] = 1000
            if self.depth_mul is not None:
                sample['depth_mul'] = self.depth_mul
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul

            intr = copy.deepcopy(self.cam_mat)
            cx, cy, fx, fy = intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
            w, h = sample['img'].size
            sample['intr'] = [cx*w, cy*h, fx*w, fy*h]

        if self.with_depth:
            # read depth information to numpy array
            with open(depth_path, 'rb') as f:
                #depth = np.array(cv2.imread(depth_path, -1)).astype(np.float32) / 256
                #print(depth.shape)
                depth = np.array(Image.open(f), dtype=np.float32) / 256
                sample['depth'] = torch.as_tensor(depth, dtype=torch.float32)
            
            if depth_gt_path is not None:
                with open(depth_gt_path, 'rb') as f:
                    #depth = np.array(cv2.imread(depth_path, -1)).astype(np.float32) / 256
                    #print(depth.shape)
                    gt_depth = np.array(Image.open(f), dtype=np.float32) / 256
                    if self.use_depth_gt:
                        sample['depth'] = torch.as_tensor(gt_depth, dtype=torch.float32)
                    else:
                        sample['gt_depth'] = torch.as_tensor(gt_depth, dtype=torch.float32)          
            else:
                sample_id = img_path.split('/')[-1].split('.')[0]
                #print(img_path, sample_id, len(sample_id))
                sample['sample_id'] = int(sample_id)

                w, h = sample['img'].size
                sample['height'] = h
                sample['width'] = w

        return sample

    def load_ann(self, path):
        with open(os.path.join(path, 'scene_gt.json')) as f:
            data = json.load(f)
        return data

    def plot_sample(self, sample):
        plt.subplot(1,3,1)
        plt.imshow(sample['img'])
        if 'depth' in sample:
            d = sample['depth'].numpy()
            if 'depth_scale' in sample:
                d = d*sample['depth_scale']
                d = d / 1000
            plt.subplot(1,3,2)
            plt.title('depth {}% [{}]'.format(np.round(np.sum(d > 0) / np.prod(d.shape) * 100, 3), d.shape))
            plt.imshow(d)
        if 'gt_depth' in sample:
            d = sample['gt_depth'].numpy()
            if 'depth_scale' in sample:
                d = d*sample['depth_scale']
                d = d / 1000
            plt.subplot(1,3,3)
            plt.title('depth gt {}% [{}]'.format(np.round(np.sum(d > 0) / np.prod(d.shape) * 100, 3), d.shape))
            plt.imshow(d)
        plt.show()

if __name__ == '__main__':
    ds = KittiDataset(None, mode='test')
    mean = []
    mean_gt = []
    std = []
    std_gt = []
    zeros = []
    zeros_gt = []
    
    for i in range(len(ds)):
        sample = ds.__getitem__(i)
        print('{} / {} | {}'.format(i+1, len(ds), sample['depth'].shape))
        #if 'sample_id' in sample:
        #    print(sample['sample_id'], str(sample['sample_id']).zfill(10), len(str(sample['sample_id']).zfill(10)))
        #    input()
        
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
            
    print('mean', np.mean(mean))
    print('std', np.mean(std))
    print('zeros', np.mean(zeros))
        
    print('mean_gt', np.mean(mean_gt))
    print('std_gt', np.mean(std_gt))
    print('zeros_gt', np.mean(zeros_gt))
        