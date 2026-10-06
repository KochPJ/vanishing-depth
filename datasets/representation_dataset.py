import torch.utils.data as data
from PIL import Image
import torch
import datasets.transforms as tfs
import numpy as np
import copy
import os


class VanishingDepthDataset(data.Dataset):
    def __init__(self, datasets: dict, mode, transforms=None, samples_per_epoch=1e4,
                 with_depth=False, with_threshold=True, with_intr=False, load_valid_gt_masks=True,
                 valid_gt_mask_save_root:str = './data/valid_gt_masks',
                 overwrite_existing_valid_gt_masks=False, check_gt_masks: bool = False, with_dino_head=False,
                 max_gt_masks=500, dataset_sampling=None, with_depth_scales=True):

        self.datasets = datasets
        self.dataset_sampling = dataset_sampling
        if self.dataset_sampling is None and mode == 'train':
            self.dataset_sampling = [1 for _ in range(len(datasets))]
        elif mode == 'train':
            assert len(self.datasets) == len(self.dataset_sampling)
        if self.dataset_sampling is not None:
            self.dataset_sampling = np.array(dataset_sampling)
            self.dataset_sampling = self.dataset_sampling / np.sum(self.dataset_sampling)
            if np.sum(self.dataset_sampling) != 1:
                self.dataset_sampling[0] += 1-np.sum(self.dataset_sampling)
            print('##### train sampling distribution #####')
            for key, p in zip(self.datasets.keys(), self.dataset_sampling):
                print('     {} | p={}'.format(key, p))

        self.mode = mode
        self.transforms = transforms
        self.samples_per_epoch = samples_per_epoch
        self.with_depth = with_depth
        self.with_threshold = with_threshold
        self.with_intr = with_intr
        self.with_depth_scales = with_depth_scales
        self.ds_index = []
        self.ds_keys = []
        for key, v in self.datasets.items():
            self.ds_index.append(len(v))
            self.ds_keys.append(key)
        self.ds_index = list(np.cumsum(self.ds_index))

        self.sample_keys = ['img', 'gt_mask', 'gt_depth', 't']
        if with_depth:
            self.sample_keys.append('depth')
            #self.sample_keys.append('depth_mul')
            if self.with_depth_scales:
                self.sample_keys.append('depth_scales')            
        if with_intr:
            self.sample_keys.append('intr')
            self.sample_keys.append('pc')
        if with_dino_head:
            self.sample_keys.append('dino_depth')
            if self.with_depth_scales:
                self.sample_keys.append('dino_depth_scales')
        #print('mode: {} | sample_keys = {}'.format(self.mode, self.sample_keys))

        self.valid_gt_masks = {}
        self.valid_gt_mask_save_root = valid_gt_mask_save_root
        self.max_gt_masks = max_gt_masks
        self.load_valid_gt_masks = load_valid_gt_masks
        if with_intr:
            self.valid_gt_mask_save_root += '_3d'

        if load_valid_gt_masks:
            if not os.path.exists(self.valid_gt_mask_save_root):
                raise ValueError('path {} does not exist'.format(self.valid_gt_mask_save_root))
            saved_indexs = list(os.listdir(self.valid_gt_mask_save_root))

            if check_gt_masks:
                for i in range(len(self.train_samples)):
                    if '{}.mask'.format(i%self.max_gt_masks) not in saved_indexs:
                        raise ValueError('Index {} has no saved gt_mask. Please make sure all gt masks exist')

            if not overwrite_existing_valid_gt_masks:
                self.load_gt_masks_from_disk()
        self.epoch = 0
        self.total_steps = None
        self.current_step = None
        self.gaussian_mean = None
        self.max_gaussian_mean = None
        self.p_max = None
        self.t_min = None
        self.t_max = None
        self.std = None
        self.max_std = None
        self.step_size = None
        self.step_size_std = None

    def __getitem__(self, index):
        failed = False
        if self.mode in ['valid', 'test']:
            d = 0
            for ds_name, l in zip(self.ds_keys, self.ds_index):
                if index < l:
                    index = index-d
                    break
                else:
                    d = l
            
            try:
                sample = self.datasets[ds_name].__getitem__(index)
                if 'depth' in sample:
                    if torch.sum(sample['depth'] > 0) < 100:
                        raise ValueError('not enough depth pixels, only {}'.format(torch.sum(sample['depth'] > 0)))

            except Exception as e:
                #raise e
                try:
                    path = self.datasets[ds_name].dirs[index]
                except:
                    path = '[BROKEN INDEX]'
                
                #raise e
                #print('Sample {} broke at dataset {} index {}'.format(path, ds_name, index))
                #print('OK Error MSG: {}'.format(e))
                failed = True


        sample_done = False
        while not sample_done:
            error = False
            if self.mode == 'train' or failed:
                errors = 0
                while True:
                    try:
                        if self.dataset_sampling is not None:
                            ds_name = np.random.choice(np.array(list(self.datasets.keys())), p=self.dataset_sampling)
                        else:
                            ds_name = np.random.choice(np.array(list(self.datasets.keys())))
                        index = int(np.random.randint(len(self.datasets[ds_name])))
                        sample = self.datasets[ds_name].__getitem__(index)
                        if 'depth' in sample:
                            if torch.sum(sample['depth'] > 0) < 100:
                                raise ValueError('not enough depth pixels, only {}'.format(torch.sum(sample['depth'] > 0)))
                        break
                    except Exception as e:
                        #raise e
                        try:
                            path = self.datasets[ds_name].dirs[index]
                        except:
                            path = '[BROKEN INDEX]'
                        #print('Sample {} broke at dataset {} index {}'.format(path, ds_name, index))
                        #print('Here Error MSG: {}'.format(e))
                        #raise e
                        errors += 1
                        if errors > 10:
                            print('Too many errors, what is happening?!?!')
                            print('Sample {} broke at dataset {} index {}'.format(path, ds_name, index))
                            print('This Error MSG: {}'.format(e))
                            raise e

            if self.mode == 'train' and self.with_depth and self.with_threshold:
                sample['t'] = self.get_threshold()

            if self.mode in ['valid', 'test'] and self.load_valid_gt_masks:
                if str(index%self.max_gt_masks) in self.valid_gt_masks:
                    sample = self.load_valid_gt_mask(sample, index)

            try:
                if self.transforms is not None:
                    sample['mode'] = self.mode
                    sample['epoch'] = self.epoch
                    sample = self.transforms(sample)
            except Exception as e:
                #print('Sample {} broke at dataset {} index {} for transforms'.format(path, ds_name, index))
                #print('Transforms Error MSG: {}'.format(e))
                #raise e
                continue

            if self.mode in ['valid', 'test'] and self.load_valid_gt_masks:
                if str(index%self.max_gt_masks) not in self.valid_gt_masks:
                    self.save_valid_gt_mask(sample, index)

            #print()
            #print(sample['img'].shape, sample['depth'].shape)
            #print('before', sample.keys(), self.sample_keys)
            sample = {key: sample[key] for key in self.sample_keys if key in sample}
            #if 'depth_mul' in self.sample_keys and 'depth_mul' not in sample:
            #    sample['depth_mul'] = 1


            

            if 'depth' in sample:
                if torch.isnan(sample['depth']).any():
                    error = True

            if 'depth_scales' in sample:
                if torch.isnan(sample['depth_scales']).any():
                    error = True

            if not error:
                sample_done = True


            #print('after', {key: type(value) for key, value in sample.items()})
            #print(sample.keys())
        return sample

    def load_gt_masks_from_disk(self):
        saved_indexs = list(os.listdir(self.valid_gt_mask_save_root))
        for path in saved_indexs:
            tag = path.split('.')[-1]
            if tag == 'mask':
                self.valid_gt_masks[path.split('.')[0]] = os.path.join(self.valid_gt_mask_save_root, path)

    def load_valid_gt_mask(self, sample, index):
        path = self.valid_gt_masks[str(index%self.max_gt_masks)]
        sample['gt_mask'] = torch.load(path)
        return sample

    def save_valid_gt_mask(self, sample, index):
        path = os.path.join(self.valid_gt_mask_save_root, '{}.mask'.format(index))
        self.valid_gt_masks[str(index%self.max_gt_masks)] = path
        torch.save(sample['gt_mask'], path)


    def __len__(self):
        if self.mode in ['valid', 'test']:
            l = 0
            for ds in self.datasets.values():
                l += len(ds)
        else:
            l = self.samples_per_epoch
        return l

    def init_threshold(self, total_steps, gaussian_mean=0.5, max_gaussian_mean=0.75, t_min=0.50,
                       t_max=1.0, std=0.25, max_std=0.5, current_step=0, num_workers=1, warmup_steps=0,
                       epoch=0):
        self.epoch = epoch
        self.total_steps = int(total_steps / num_workers)
        self.current_step = int(current_step / num_workers)
        self.gaussian_mean = gaussian_mean
        self.max_gaussian_mean = max_gaussian_mean
        self.max_std = max_std
        self.t_min = t_min
        self.t_max = t_max
        self.std = std
        self.step_size = (max_gaussian_mean - gaussian_mean) / self.total_steps
        self.step_size_std = (max_std - std) / self.total_steps
        self.warmup_steps = int(warmup_steps / num_workers)

        # set the start mean and std according to the current step
        if self.current_step >= self.warmup_steps:
            self.gaussian_mean += self.step_size * (self.current_step - self.warmup_steps)
            self.std += self.step_size_std * (self.current_step - self.warmup_steps)

        if self.gaussian_mean > self.max_gaussian_mean:
            self.gaussian_mean = self.max_gaussian_mean

        if self.std > self.max_std:
            self.std = self.max_std


    def get_threshold(self):
        t = torch.randn(1) * self.std + self.gaussian_mean
        if self.t_min is not None:
            if t < self.t_min:
                t = self.t_min
        if self.t_max is not None:
            if t > self.t_max:
                t = self.t_max

        if self.gaussian_mean < self.max_gaussian_mean and self.current_step >= self.warmup_steps:
            self.gaussian_mean += self.step_size

        if self.std < self.max_std and self.current_step >= self.warmup_steps:
            self.std += self.step_size_std

        self.current_step += 1
        return torch.Tensor([t])

'''
    def init_new_epoch(self):
        if self.scene_wise_sampling and self.mode == 'train':
            self.train_samples = []
            for ds_name in self.ds_lut.keys():
                if len(self.ds_lut[ds_name]) < self.min_l:
                    self.ds_lut[ds_name] += list(np.random.permutation(self.datasets[ds_name].dirs))
                self.train_samples += [[ds_name, self.ds_lut[ds_name].pop()] for _ in range(self.min_l)]
        elif not self.train_samples:
            self.train_samples = []
            np.random.seed(42)
            for ds_name in self.datasets.keys():
                if self.scene_wise_sampling:
                    np.random.permutation(self.datasets[ds_name].dirs)
                    if len(self.datasets[ds_name].dirs) < self.samples_per_epoch:
                        raise ValueError('The dataset {} has only {} samples, wich is not enough for the {} samples '
                                         'per dataset.'.format(ds_name,
                                                        len(self.datasets[ds_name].dirs),
                                                        self.samples_per_epoch))
                    for path in self.datasets[ds_name].dirs[:self.samples_per_epoch]:
                        self.train_samples.append((ds_name, path))
                else:
                    for path in self.datasets[ds_name].dirs:
                        self.train_samples.append((ds_name, path))
'''

if __name__ == '__main__':
    from sunrgbd import SUNRGBD_Dataset
    from linemod import LineModeDataset
    from t_less import TLESSDataset
    from grasp_net import GraspNetDataset
    from ycb_video import YCBVDataset

    import matplotlib.pyplot as plt

    datasets = {'SUNRGBD': SUNRGBD_Dataset(with_intr=True, mode='train', load_from_checkpoint=True),
                'LineMod': LineModeDataset(with_intr=True, mode='train', load_from_checkpoint=True),
                'T-Less': TLESSDataset(with_intr=True, mode='train', load_from_checkpoint=True),
                'Graspnet': GraspNetDataset(with_intr=True, mode='train', load_from_checkpoint=True),
                'YCB_video': YCBVDataset(with_intr=True, mode='train', load_from_checkpoint=True)}
    #global_crops_scale = (0.4, 1.)
    #local_crops_scale = (0.05, 0.4)
    #num_local_crops = 8

    ds = VanishingDepthDataset(datasets=datasets, mode='train', scene_wise_sampling=True, samples_per_epoch=20000,
                               with_depth=True, with_intr=True)

    transforms = [tfs.ColorJitter(),
                  tfs.RandomGray(),
                  tfs.Depth2distance(to_meter=True),
                  tfs.GTDepth(),
                  tfs.DepthNoise(),
                  tfs.DepthToPointCloud(),
                  tfs.RandomHorizontalFlip(),
                  tfs.RandomVerticalFlip(),
                  tfs.RandomResizedCrop(size=512),
                  tfs.VanishingDepth(fpn=True, fpn_layers=5)]

    import math

    preprocesses = [#tfs.ToTensor(),
                    #tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                    tfs.DepthPositionalEncoding(scale=math.pi*2),
                    ]

    transforms += preprocesses
    import torchvision.transforms as pytfs
    ds.transforms = pytfs.Compose(transforms)

    ds.init_threshold(total_steps=100 * len(ds),
                      max_gaussian_mean=0,
                      gaussian_mean=0.0, std=0.0, t_min=0.0,
                      current_step=0, num_workers=1,
                      warmup_steps=0)

    print('dataset size', len(ds), ds.min_l)

    for i in np.random.permutation(list(range(len(ds)))):

        print(i)
        sample = ds.__getitem__(i)

        print('sample', sample.keys())
        img = np.array(sample['img'])
        depth = np.array(sample['depth'][-1])
        gt_depth = np.array(sample['gt_depth'][0])
        print(gt_depth.shape, sample['depth'].shape)
        print(img.shape, depth.shape, sample['intr'], sample['t'])
        plt.subplot(3,3,1)
        plt.imshow(img)


        plt.subplot(3,3,2)
        plt.imshow(depth[0])
        plt.subplot(3,3,3)
        plt.imshow(depth[1])
        plt.subplot(3,3,4)
        plt.imshow(depth[2])

        plt.subplot(3, 3, 5)
        plt.imshow(gt_depth[0])
        plt.subplot(3, 3, 6)
        plt.imshow(gt_depth[1])
        plt.subplot(3, 3, 7)
        plt.imshow(gt_depth[2])

        plt.show()

