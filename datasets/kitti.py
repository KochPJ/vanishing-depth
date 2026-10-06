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
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True, mode_tag=None,
                 with_mask=False, load_from_checkpoint=True, with_pc=False, mode=None, p_eval=500,
                 depth_mul=10):
        if root is None:
            root = './data/kitti'
        self.root = root
        self.transforms = transforms
        self.with_intr = with_intr
        self.with_depth = with_depth
        self.with_mask = with_mask
        self.with_pc = with_pc
        self.names = {}
        self.depth_mul = depth_mul
        self.cam_mat = np.array([[0.58, 0, 0.5, 0],
                                 [0, 1.92, 0.5, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], dtype=np.float32)

        if mode_tag is not None:
            mode = mode + '_' + mode_tag
        if mode is not None:
            file_name = 'kitti_{}_dataset_config.json'.format(mode)
        else:
            file_name = 'kitti_dataset_config.json'

        load_path = os.path.join(self.root, file_name)
        cp = None
        if load_from_checkpoint:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    cp = json.load(f)
        ids = []
        self.dirs = []
        test_dirs = []
        if cp is None:
            cls_counter = 1
            dir_lut = {}
            for folder in os.listdir(os.path.join(self.root, 'images')):
                #print(folder)
                if not os.path.isdir(os.path.join(self.root, 'images', folder)):
                    continue
                for date in os.listdir(os.path.join(self.root, 'images', folder)):
                    dir_lut[date] = folder
                    print(folder, date)


            for path, folders, files in os.walk(os.path.join(self.root, 'depth')):
                if len(folders) == 0:
                    d = path.split('/')
                    im_set, date = d[-1], d[-4]
                    if date not in dir_lut:
                        continue
                    sub_folder = dir_lut[date]
                    for f in files:
                        depth_path = os.path.join(path, f)
                        rgb_path = os.path.join(self.root, 'images', sub_folder, date, im_set, 'data', f)
                        if not os.path.exists(rgb_path):
                            continue
                        if '/train/' in depth_path:
                            self.dirs.append([rgb_path, depth_path])
                        else:
                            test_dirs.append([rgb_path, depth_path])

            if mode is not None:
                if mode_tag is not None:
                    train_name = 'kitti_{}_{}_dataset_config.json'.format('train', mode_tag)
                    valid_name = 'kitti_{}_{}_dataset_config.json'.format('valid', mode_tag)
                    test_name = 'kitti_{}_{}_dataset_config.json'.format('test', mode_tag)
                else:
                    train_name = 'kitti_{}_dataset_config.json'.format('train')
                    valid_name = 'kitti_{}_dataset_config.json'.format('valid')
                    test_name = 'kitti_{}_dataset_config.json'.format('test')

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
        if self.transforms is not None:
            s = self.transforms(sample)
            sample = {}
            for key, v in s.items():
                if key in ['img', 'mask', 'depth', 'ori_depth']:
                    sample[key] = v.float()
        return sample

    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array
        img_path, depth_path = path

        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        if self.with_intr or self.with_pc or self.with_depth:
            sample['depth_scale'] = 10000
            if self.depth_mul is not None:
                sample['depth_mul'] = self.depth_mul
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul

            intr = copy.deepcopy(self.cam_mat)
            cx, cy, fx, fy = intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
            w, h = sample['img'].size
            sample['intr'] = [cx*w, cy*h, fx*w, fy*h]

        if self.with_pc or self.with_depth:
            # read depth information to numpy array
            with open(depth_path, 'rb') as f:
                #depth = np.array(cv2.imread(depth_path, -1)).astype(np.float32) / 256
                #print(depth.shape)
                depth = np.array(Image.open(f), dtype=np.float32) / 256
                height, width = depth.shape
                sample['depth'] = torch.as_tensor(depth, dtype=torch.float32)

        return sample

    def load_ann(self, path):
        with open(os.path.join(path, 'scene_gt.json')) as f:
            data = json.load(f)
        return data

    def plot_sample(self, sample):
        plt.subplot(1,2,1)
        plt.imshow(sample['img'])
        if 'depth' in sample:
            d = sample['depth']
            if 'depth_scale' in sample:
                d = d*sample['depth_scale']
                d = d / 1000
            plt.subplot(1,2,2)
            plt.imshow(d)
            plt.show()

if __name__ == '__main__':
    ds = KittiDataset(None, root='/mnt/wimi/publicdata/kitti', with_pc=False, mode='train', with_intr=True,
                        load_from_checkpoint=True, p_eval=0)
    for i in np.random.permutation(len(ds)):
        sample = ds.__getitem__(i)
        print('intr', sample.get('intr'))
        ds.plot_sample(sample)
