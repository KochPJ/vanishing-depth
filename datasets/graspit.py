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
import open3d as o3d


class GraspIt(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True, mode_tag=None,
                 with_mask=False, load_from_checkpoint=True, with_pc=False, mode=None, p_eval=500,
                 depth_mul=1):
        if root is None:
            root = './data/graspit'
        self.root = root
        self.transforms = transforms
        self.scenes = []
        self.frames = []
        self.classes = ['background', 'plane', 'table', 'objects']
        for scenes in os.listdir(os.path.join(self.root, 'scenes')):
            for scene in os.listdir(os.path.join(self.root, 'scenes', scenes)):
                self.scenes.append((scenes, scene))
                for frame in os.listdir(os.path.join(self.root, 'scenes', scenes, scene)):
                    if 'frame_' in frame:
                        self.frames.append((scenes, scene, frame))
        

    def __getitem__(self, index):
        path = self.frames[index]
        sample = self.load_sample(path)
        sample['sample_index'] = index
        if self.transforms is not None:
            s = self.transforms(sample)
            sample = {}
            for key, v in s.items():
                if key in ['img', 'mask', 'depth', 'gt_depth', 'depth_scales']:
                    sample[key] = v.float()
        return sample

    def __len__(self):
        return len(self.frames)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array
        scenes, scene, frame = path

        frame_idx = frame.split('_')[1].zfill(4)

        frame_root = os.path.join(self.root, 'scenes', scenes, scene, frame)
        scene_root = os.path.join(self.root, 'scenes', scenes, scene)


        img_path = os.path.join(frame_root, 'rgb_{}.png'.format(frame_idx))
        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        W, H = sample['img'].size

        # read depth information to numpy array
        
        depth_path = os.path.join(frame_root, 'depth.png'.format(frame_idx))
        with open(depth_path, 'rb') as f:
            sample['depth'] = torch.as_tensor(np.array(Image.open(f), dtype=np.float32), dtype=torch.float32)
            #sample['depth'] = np.array(Image.open(f)).astype(np.float32)
        sample['depth_scale'] = 0.2

        print(os.listdir(frame_root))
        
        mask_info_path = os.path.join(frame_root, 'semantic_segmentation_labels_{}.json'.format(frame_idx))
        with open(mask_info_path, 'rb') as f:
            mask_info = json.load(f)
            print(mask_info)
        
        mask_path = os.path.join(frame_root, 'semantic_segmentation_{}.png'.format(frame_idx))
        mask_classes = []

        with open(mask_path, 'rb') as f:
            sample['mask_rgb'] = Image.open(f).convert('RGB')
            mask = np.array(sample['mask_rgb'])

            sample['mask'] = np.zeros((H, W), dtype=np.float32)
            for mask_idx, (rgb_m, cls_name) in enumerate(mask_info.items()):
                name = cls_name['class'].lower() 
                if name in self.classes:
                    mask_classes.append(self.classes.index(name))
                else:
                    mask_classes.append(self.classes.index('objects'))
                colour = np.array([int(v) for v in rgb_m.replace('(', '').replace(')', '').replace(' ', '').split(',')[:3]])
                m = np.all(mask == colour, axis=-1)
                sample['mask'][m] = mask_idx+1

        sample['mask_classes'] = torch.from_numpy(np.array(mask_classes, dtype=np.uint8))
        

        return sample


    def plot_sample(self, sample):
        #overlay = np.array(sample['img']).astype(np.float32)
        #overlay[:, :, 0] = overlay[:, :, 0]*0.7 + ((sample['depth']/np.max(sample['depth']))*255) *0.3
        print(sample['mask_classes'])
        plt.subplot(1,4,1)
        plt.imshow(sample['img'])
        plt.subplot(1,4,2)
        plt.imshow(sample['depth']*sample['depth_scale'] / 1000)
        plt.subplot(1,4,3)
        plt.imshow(sample['mask_rgb'])
        plt.subplot(1,4,4)
        plt.imshow(sample['mask'])
        plt.show()

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.getcwd())

    ds = GraspIt(None, with_pc=False, mode='train', with_mask=True, p_eval=0, load_from_checkpoint=True)
    print(len(ds))
    for i in np.random.permutation(len(ds)):
        sample = ds.__getitem__(i)
        ds.plot_sample(sample)
