import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import random


class TLESSDataset(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True, mode_tag=None,
                 with_mask=False, load_from_checkpoint=True, with_pc=False, mode=None, p_eval=500,
                 depth_mul=1):
        if root is None:
            root = './data/T-Less'
        self.root = root
        self.transforms = transforms
        self.with_intr = with_intr
        self.with_depth = with_depth
        self.with_mask = with_mask
        self.with_pc = with_pc
        self.names = {}
        self.depth_mul = depth_mul

        if mode_tag is not None:
            mode = mode + '_' + mode_tag
        if mode is not None:
            file_name = 't_less_{}_dataset_config.json'.format(mode)
        else:
            file_name = 't_less_dataset_config.json'

        load_path = os.path.join(self.root, file_name)
        cp = None
        if load_from_checkpoint:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    cp = json.load(f)

        self.dirs = []
        test_dirs = []
        ids = []
        if cp is None:
            cls_counter = 0
            for path, folders, files in os.walk(self.root):
                if 'scene_camera.json' in files and 'scene_gt.json' in files:
                    try:
                        ann = self.load_ann(path)
                        for key, value in ann.items():
                            try:
                                _ = self.load_sample((path, key))
                                if 'test' in path:
                                    test_dirs.append((path, key))
                                else:
                                    self.dirs.append((path, key))
                                classes = [int(v['obj_id']) for v in value]
                                for cls_id in classes:
                                    if cls_id not in ids:
                                        ids.append(cls_id)
                                        self.names[str(cls_counter).zfill(4)] = {'name': cls_id}
                                        cls_counter += 1
                            except Exception as e:
                                print(path, key, e)
                                continue

                    except Exception as e:
                        continue

            if mode is not None:
                if mode_tag is not None:
                    train_name = 't_less_{}_dataset_config.json'.format('train_ {}'.format(mode_tag))
                    valid_name = 't_less_{}_dataset_config.json'.format('valid_{}'.format(mode_tag))
                    test_name = 't_less_{}_dataset_config.json'.format('test_{}'.format(mode_tag))
                else:
                    train_name = 't_less_{}_dataset_config.json'.format('train')
                    valid_name = 't_less_{}_dataset_config.json'.format('valid')
                    test_name = 't_less_{}_dataset_config.json'.format('test')

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
                if key in ['img', 'mask', 'depth']:
                    sample[key] = v.float()
        return sample

    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array
        path, idx = path
        name = str(idx).zfill(6)
        img_path = os.path.join(path, 'rgb', name + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(path, 'rgb', name + '.png')

        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        if self.with_intr or self.with_pc or self.with_depth:
            with open(os.path.join(path, 'scene_camera.json')) as f:
                scene_camera = json.load(f)[str(idx)]

            sample['depth_scale'] = scene_camera['depth_scale']
            if self.depth_mul is not None:
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul
                sample['depth_mul'] = self.depth_mul

            intr = np.array(scene_camera['cam_K']).reshape((3,3))

            cx, cy, fx, fy = intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
            sample['intr'] = [cx, cy, fx, fy]

        if self.with_pc or self.with_depth:
            # read depth information to numpy array
            depth_path = os.path.join(path, 'depth', name + '.png')
            with open(depth_path, 'rb') as f:
                sample['depth'] = torch.as_tensor(np.array(Image.open(f), dtype=np.float32), dtype=torch.float32)
                #sample['depth'] = np.array(Image.open(f)).astype(np.float32)

        if self.with_mask:
            depth_path = os.path.join(path, 'mask_visib')
            ann = self.load_ann(path)[str(idx)]
            classes = [int(v['obj_id']) for v in ann]
            masks = sorted([m for m in list(os.listdir(depth_path)) if m[:6] == name])
            w, h = sample['img'].size
            mask = np.zeros((h, w), dtype=np.uint8)
            for cls, m in zip(classes, masks):
                with open(os.path.join(depth_path, m), 'rb') as f:
                    ml = np.array(Image.open(f).convert('RGB'))[:, :, 0]
                    mask[ml != 0] = cls
            sample['mask'] = mask


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

    def load_ann(self, path):
        with open(os.path.join(path, 'scene_gt.json')) as f:
            data = json.load(f)
        return data

    def plot_sample(self, sample):
        plt.subplot(2,3,1)
        plt.imshow(sample['img'])
        if 'depth' in sample:
            d = sample['depth']
            if 'depth_scale' in sample:
                d = d*sample['depth_scale']
                d = d / 1000
            plt.subplot(2,3,2)
            plt.imshow(d)
        if 'mask' in sample:
            plt.subplot(2,3,3)
            plt.imshow(sample['mask'])
        if 'pc' in sample:
            plt.subplot(2, 3, 4)
            plt.imshow(np.abs(sample['pc'][0, :, :]))
            plt.subplot(2, 3, 5)
            plt.imshow(np.abs(sample['pc'][1, :, :]))
            plt.subplot(2, 3, 6)
            plt.imshow(sample['pc'][2, :, :])
        plt.show()

if __name__ == '__main__':
    ds = TLESSDataset(None, with_pc=False, mode='test', load_from_checkpoint=True, p_eval=0)
    print(len(ds))
    print(ds.names)
    for i in np.random.permutation(len(ds)):
        sample = ds.__getitem__(i)
        ds.plot_sample(sample)
