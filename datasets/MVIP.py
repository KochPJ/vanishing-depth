import copy
import json
import random

import torch.utils.data as data
import os
from PIL import Image
import torch
import datasets.transforms as tfs
from utils.MVIP_utils import *
import numpy as np
import torchvision.transforms as transforms


def get_dataset(args):
    ds = {'train': {}, 'valid': {}, 'test': {}, 'meta': {}}
    data_views = ['cam_{}'.format(v.zfill(2)) for v in args.data_views.split('-')]
    input_views = ['cam_{}'.format(v.zfill(2)) for v in args.views.split('-')]
    rotations = args.rotations.split('-')

    for cls in os.listdir(args.path):
        with open(os.path.join(args.path, cls, 'meta.json')) as f:
            ds['meta'][cls] = json.load(f)

        for mode in ['valid', 'test']:
            ds[mode][cls] = []
            tdir = os.path.join(args.path, cls, '{}_data'.format(mode))

            for ts in os.listdir(tdir):
                sample = {}
                sdir = os.path.join(tdir, ts)
                for v in os.listdir(sdir):
                    if v in input_views:
                        _, v_id = v.split('_')
                        sample[v_id] = {
                            'rgb': os.path.join(sdir, v, '{}_rgb.png'.format(v_id)),
                            'mask': os.path.join(sdir, v, '{}_rgb_mask_gen.png'.format(v_id)),
                            'depth': os.path.join(sdir, v, '{}_depth.png'.format(v_id)),
                            'rgb_mean': os.path.join(sdir, v, '{}_rgb_mean.png'.format(v_id)),
                            'mask_mean': os.path.join(sdir, v, '{}_rgb_mean_mask_gen.png'.format(v_id)),
                            'depth_mean': os.path.join(sdir, v, '{}_depth_mean.png'.format(v_id)),
                            'meta': os.path.join(sdir, v, '{}_meta.json'.format(v_id)),
                            'view': v
                        }
                if args.multiview:
                    ds[mode][cls].append(sample)
                else:
                    ds[mode][cls] += list(sample.values())

        tdir = os.path.join(args.path, cls, 'train_data')

        ds['train'][cls] = []
        for pos in os.listdir(tdir):
            if pos == 'background':
                back = True
            else:
                back = False
            #    continue

            pdir = os.path.join(tdir, pos)
            for rot in os.listdir(pdir):
                if rot not in rotations:
                    continue

                sdir = os.path.join(pdir, rot)
                sample = {}
                for v in os.listdir(sdir):
                    if v in data_views:
                        _, v_id = v.split('_')
                        sample[v_id] = {
                            'rgb': os.path.join(sdir, v, '{}_rgb.png'.format(v_id)),
                            'mask': os.path.join(sdir, v, '{}_rgb_mask_gen.png'.format(v_id)),
                            'depth': os.path.join(sdir, v, '{}_depth.png'.format(v_id)),
                            'rgb_mean': os.path.join(sdir, v, '{}_rgb_mean.png'.format(v_id)),
                            'mask_mean': os.path.join(sdir, v, '{}_rgb_mean_mask_gen.png'.format(v_id)),
                            'depth_mean': os.path.join(sdir, v, '{}_depth_mean.png'.format(v_id)),
                            'meta': os.path.join(sdir, v, '{}_meta.json'.format(v_id)),
                            'rot': rot,
                            'view': v
                        }

                if args.multiview:
                    ds['train'][cls].append(sample)
                else:
                    ds['train'][cls] += list(sample.values())

    #print('ds size', sum(len(v) for v in ds['train'].values()))
    #print(len(ds['train'].keys()))
    if args.balanced_sampling:
        for cls in ds['train'].keys():
            l = len(ds['train'][cls])
            if l != len(rotations):
                random.shuffle(ds['train'][cls])
                if l > len(rotations):
                    ds['train'][cls] = ds['train'][cls][:12]
                else:
                    n = len(rotations)-l
                    ds['train'][cls] = ds['train'][cls] + ds['train'][cls][:n]
    return ds


class MVIP(data.Dataset):
    def __init__(self, args, data, meta, mode=False, classes=None):
        self.args = args
        self.data = data
        self.meta = meta
        self.mode = mode

        if classes is not None:
            self.classes = classes
        else:
            self.classes = sorted(list(data.keys()))
        self.num_classes = len(self.classes)

        if mode != 'train':
            self.eval = True
        else:
            self.eval = False

        self.input_keys = args.input_keys.split('-')
        self.load_keys = args.load_keys.split('-')
        #self.views = args.views.split('-')
        self.input_views = [v.zfill(2) for v in args.views.split('-')]
        self.rotations = args.rotations.split('-')

        self.samples = []
        self.samples_ = {}

        for cls, samples in self.data.items():
            if args.shuf_views and not self.eval:
                self.samples_[cls] = {}
            cls_id = self.classes.index(cls)
            for s in samples:
                if args.multiview:
                    for view in s.keys():
                        s[view]['cls_id'] = cls_id
                        s[view]['cls'] = cls
                else:
                    s['cls_id'] = cls_id
                    s['cls'] = cls
                self.samples.append(s)

                if args.shuf_views and not self.eval:
                    for view in s.keys():
                        if view not in self.samples_[cls]:
                            self.samples_[cls][view] = []
                        self.samples_[cls][view].append(s[view])

        if args.shuf_views and not self.eval:
            self.ori_samples = copy.deepcopy(self.samples)
        else:
            self.ori_samples = None

        self.shuffle_data()

        if 'depth' in self.args.input_keys and (args.depth_mean is None or args.depth_std is None) \
                and (not args.with_positional_encoding and 'depth' in args.input_keys):
            print('Getting Depth mean and std')
            depth_mean, depth_std, ps = self.get_depth_mean_std()
            print('Depth mean: {}'.format(depth_mean))
            print('Depth std: {}'.format(depth_std))
            print('p >Zero: {}'.format(ps))
            self.args.depth_mean = depth_mean
            self.args.depth_std = depth_std

        if self.eval:
            self.augs = None
        else:
            augs = [ColorJitter(), RandomColor()]
            if args.flip_aug:
                augs.append(RandomFlip())
            if args.rotation_aug:
                augs.append(RandomRotation())
            #if 'depth' in args.input_keys:
            #    augs.append(DepthNoise(zero_p=args.zero_p, noise_color_p=args.noise_color_p))
            self.augs = transforms.Compose(augs)

        print('mode: {}, Augmentations: {}'.format(self.mode, self.augs))
        self.multi_scale = args.multi_scale_training if not self.eval else False
        self.color_pre = transforms.Compose([
            RoiCrop(),
            Resize(width=args.width, height=args.height,
                   multi_scale_training=self.multi_scale,
                   training_scale_low=args.training_scale_low, training_scale_high=args.training_scale_high)
        ])
        print('mode: {}, Color pre: {}'.format(self.mode, self.color_pre))
        self.tensor_pre = [
            ToTensor()
        ]

        if args.with_depth:
            try:
                depth_mul = args.depth_mul
            except:
                depth_mul = 1.0

            
            self.tensor_pre.append(Depth2distance(to_meter=args.to_meter,
                                                  constant_depth_offset=args.constant_depth_offset,
                                                  depth_mul=depth_mul))
            self.tensor_pre.append(CatDepth())

        self.tensor_pre.append(
            Normalize(norm_depth=args.norm_depth)
        )

        if args.with_positional_encoding and 'depth' in args.input_keys:
            self.args.depth_mean = None
            self.args.depth_std = None
            self.tensor_pre.append(
                tfs.DepthPositionalEncoding(depth_channels=args.depth_channels,
                                                temperature=args.temperature,
                                                position_offset=args.position_offset,
                                                scale=args.scale,
                                                zero_eps=args.zero_eps,
                                                flatten_4th_dim=True if args.with_intr else False,
                                                div_factors=args.pe_div_factors if
                                                args.set_pe_div_factors else None,
                                                with_dino_head=args.with_dino_head,
                                                #with_dino_head=False,
                                                encoder_hidden_dims=args.encoder_hidden_dims,
                                                enable_dino_head_epoch=args.enable_dino_head_epoch,
                                                get_depth_scales=args.with_depth_scales,
                                                max_depth=args.max_depth)
                             
                )
        elif 'depth' in args.input_keys:
            args.norm_depth = False
            if args.depth2disparty:
                self.tensor_pre.append(
                    tfs.DisparityAndRescale(
                        rescale_factor=args.depth_rescale_div, 
                        depth2disparty=args.depth2disparty,
                        min_depth=0.1,
                        max_depth=50
                        )
                )
            self.tensor_pre.append(
                tfs.NormalizeDepth(
                    mean=args.depth_mean, std=args.depth_std
                )
            )
            print('Depth Mean: {}, Depth STD = {}'.format(self.args.depth_mean, self.args.depth_std))

        self.tensor_pre = transforms.Compose(self.tensor_pre)
        print('mode: {}, Tensor pre: {}'.format(self.mode, self.tensor_pre))

        self.checking_batch_size = False
        self.batch_size = args.batch_size

    def __getitem__(self, index):
        x, y = self.load_sample(index)

        x['mode'] = self.mode
        if self.multi_scale:
            x['batch_size'] = self.batch_size
            if self.checking_batch_size:
                x['checking_batch_size'] = True
                self.checking_batch_size = False

        x = self.color_pre(x)
        if self.augs is not None:
            x = self.augs(x)

        if not self.args.visualize_samples:
            x = self.tensor_pre(x)
            if self.args.multiview:
                x = {key: x[key] for key in self.input_keys}
            else:
                x = {key: x[key][0] for key in self.input_keys}
        else:
            if isinstance(y, torch.Tensor):
                y = torch.where(y != 0), y[torch.where(y != 0)]
            x = {key: x[key] for key in self.input_keys}

        return x, y

    def __len__(self):
        return len(self.samples)

    def load_sample(self, index):

        s = self.samples[index]
        if self.args.multiview:
            sample = {}
            ys = []

            views = sorted(list(s.keys()))
            if len(views) > len(self.input_views):
                if self.eval:
                    views = [v for v in views if v in self.input_views]
                else:
                    views = [views[int(i)] for i in np.random.choice(list(range(len(views))),
                                                                     size=len(self.input_views), replace=False)]

            if self.args.random_view_order and not self.eval:
                random.shuffle(views)

            sample['view_ids'] = views
            for view in views:
                x, y = self.load_view(s[view])
                for key in x:
                    if key not in sample:
                        sample[key] = []
                    sample[key].append(x[key])
                ys.append(y)

            if not self.args.shuf_views_cw and self.args.shuf_views:
                y = torch.zeros(self.num_classes)
                uni, c = torch.unique(torch.tensor(ys), return_counts=True)
                y[uni.long()] = c/torch.sum(c)
            else:
                y = ys[0]
        else:
            sample, y = self.load_view(s)
            for key in sample.keys():
                sample[key] = [sample[key]]

        return sample, y

        pass

    def load_view(self, view):
        sample = {}
        if 'x' in self.load_keys:
            sample['x'] = Image.open(view['rgb'])
        if 'mask' in self.load_keys:
            sample['mask'] = Image.open(view['mask'])
        if 'depth' in self.load_keys:
            sample['depth'] = Image.open(view['depth'])
        y = view['cls_id']
        return sample, y

    def shuffle_data(self):
        if not self.eval and self.args.shuf_views:
            #print('Shuffeling Data: shuf_views_vw: {}, shuf_views_cw: {}, p_shuf_cw: {}, p_shuf_vw: {}'.format(
            #    self.args.shuf_views_vw, self.args.shuf_views_cw, self.args.p_shuf_cw, self.args.p_shuf_vw
            #))
            self.samples = []
            for s in self.ori_samples:
                views = list(s.keys())
                for view in views:
                    if not self.args.shuf_views_cw and np.random.rand() < self.args.p_shuf_cw:
                        cls = self.classes[int(np.random.randint(0, self.num_classes))]
                    else:
                        cls = s[view]['cls']

                    if not self.args.shuf_views_vw and np.random.rand() < self.args.p_shuf_vw:
                        v = views[int(np.random.randint(0, len(s)))]
                    else:
                        v = view
                    s_ = self.samples_[cls][v]
                    s[view] = s_[int(np.random.randint(0, len(s_)))]
                self.samples.append(s)

    def get_depth_mean_std(self):
        means = []
        stds = []
        ps = []
        for index in range(len(self)):
            sample, _ = self.load_sample(index)
            for depth in sample['depth']:
                depth = np.array(depth)
                means.append(np.mean(depth))
                stds.append(np.std(depth))
                h, w = depth.shape
                ps.append(float(np.sum(depth > 0) / (h * w)))
        mean = float(np.mean(means))
        std = float(np.mean(stds))
        ps = float(np.mean(ps))
        return mean, std, ps
