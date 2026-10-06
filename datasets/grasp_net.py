import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io as scio


class GraspNetDataset(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True, mode_tag=None,
                 with_mask=False, load_from_checkpoint=True, with_pc=False, mode=None, p_eval=500,
                 depth_mul=1):
        if root is None:
            root = './data/GraspNet-1Billion'
        self.root = root
        self.transforms = transforms
        self.with_intr = with_intr
        self.with_depth = with_depth
        self.with_mask = with_mask
        self.with_pc = with_pc
        self.names = {str(i+1).zfill(4): {'name': str(i+1).zfill(4)} for i in range(88)}
        self.depth_mul = depth_mul
        self.mode = mode

        if mode_tag is not None:
            mode = mode + '_' + mode_tag
        if mode is not None:
            file_name = 'GraspNet_{}_dataset_config.json'.format(mode)
        else:
            file_name = 'GraspNet_dataset_config.json'

        load_path = os.path.join(self.root, file_name)
        cp = None
        if load_from_checkpoint:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    cp = json.load(f)

        self.dirs = []
        test_dirs = []
        if cp is None:
            for path, folders, files in os.walk(self.root):
                if 'camK.npy' in files:
                    samples = [sample.split('.')[0] for sample in list(os.listdir(os.path.join(path, 'rgb')))]
                    for sample in samples:
                        try:
                            _ = self.load_sample((path, sample))
                            if 'test' in path:
                                test_dirs.append((path, sample))
                            else:
                                self.dirs.append((path, sample))
                        except Exception as e:
                            print(path, sample, e)
                            continue

            if mode is not None:
                if mode_tag is not None:
                    train_name = 'GraspNet_{}_dataset_config.json'.format('train_'.format(mode_tag))
                    valid_name = 'GraspNet_{}_dataset_config.json'.format('valid_'.format(mode_tag))
                    test_name = 'GraspNet_{}_dataset_config.json'.format('test_'.format(mode_tag))
                else:
                    train_name = 'GraspNet_{}_dataset_config.json'.format('train')
                    valid_name = 'GraspNet_{}_dataset_config.json'.format('valid')
                    test_name = 'GraspNet_{}_dataset_config.json'.format('test')

                random.shuffle(self.dirs)
                random.shuffle(test_dirs)

                if p_eval < 1:
                    n = int(len(self.dirs) * p_eval)
                else:
                    n = int(p_eval)

                with open(os.path.join(root, test_name), 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': test_dirs}, f)
                with open(os.path.join(root, valid_name), 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': self.dirs[:n]}, f)
                with open(os.path.join(root, train_name), 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': self.dirs[n:]}, f)
                if 'train' in mode:
                    self.dirs = self.dirs[n:]
                elif 'valid' in mode:
                    self.dirs = self.dirs[:n]
                elif 'test' in mode:
                    self.dirs = test_dirs
            else:
                with open(load_path, 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': self.dirs}, f)
        else:
            self.names = cp['names']
            self.dirs = cp['dirs']

        self.class_names = [self.names[key]['name'] for key in sorted(list(self.names.keys()))]
        self.num_classes = len(self.class_names)

    def __getitem__(self, index):
        path = self.dirs[index]
        sample = self.load_sample(path)
        sample['sample_index'] = index
        sample['mode'] = self.mode
        if self.transforms is not None:
            s = self.transforms(sample)
            sample = {}
            for key, v in s.items():
                if key in ['img', 'mask', 'depth', 'gt_depth', 'depth_scales']:
                    sample[key] = v.float()
        return sample

    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array
        path, name = path

        img_path = os.path.join(path, 'rgb', name + '.png')
        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        if self.with_pc or self.with_depth:
            # read depth information to numpy array
            depth_path = os.path.join(path, 'depth', name + '.png')
            with open(depth_path, 'rb') as f:
                sample['depth'] = torch.as_tensor(np.array(Image.open(f), dtype=np.float32), dtype=torch.float32)
                #sample['depth'] = np.array(Image.open(f)).astype(np.float32)
            sample['depth_scale'] = 1
            if self.depth_mul is not None:
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul
                sample['depth_mul'] = self.depth_mul

        if self.with_mask:
            mask_path = os.path.join(path, 'label', name + '.png')
            with open(mask_path, 'rb') as f:
                sample['mask'] = torch.from_numpy(np.array(Image.open(f)).astype(np.float32))

        if self.with_intr or self.with_pc:
            #intr = np.load(os.path.join(path, 'camK.npy'))
            meta = scio.loadmat(os.path.join(path, 'meta', name + '.mat'))
            intr = meta['intrinsic_matrix']
            cx, cy, fx, fy = intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
            sample['intr'] = [cx, cy, fx, fy]

        if self.with_pc:
            w, h = sample['img'].size
            xmap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(w), axis=0), h, axis=0), dtype=torch.float32)
            ymap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(h), axis=1), w, axis=1), dtype=torch.float32)
            cx, cy, fx, fy = sample['intr']
            pt2 = (torch.as_tensor(sample['depth'], dtype=torch.float32) * sample['depth_scale']) / 1000  # mm to meter
            pt0 = (xmap - cx) * pt2 / fx
            pt1 = (ymap - cy) * pt2 / fy
            sample['pc'] = torch.stack((pt0, pt1, pt2), axis=0)

        return sample


    def plot_sample(self, sample):
        #overlay = np.array(sample['img']).astype(np.float32)
        #overlay[:, :, 0] = overlay[:, :, 0]*0.7 + ((sample['depth']/np.max(sample['depth']))*255) *0.3
        plt.subplot(1,3,1)
        plt.imshow(sample['img'])
        plt.subplot(1,3,2)
        plt.imshow(sample['depth'])
        plt.subplot(1,3,3)
        plt.imshow(sample['mask'])
        plt.show()

if __name__ == '__main__':
    ds = GraspNetDataset(None, with_pc=False, mode='train', with_mask=True, p_eval=0, load_from_checkpoint=True)
    print(len(ds))
    print(ds.names)
    print(len(ds.names))
    for i in np.random.permutation(len(ds)):
        sample = ds.__getitem__(i)
        ds.plot_sample(sample)
