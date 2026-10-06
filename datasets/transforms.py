# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import ImageFilter, ImageOps
import numpy as np
import copy
import math
from torchvision.transforms import InterpolationMode
from utils.perlin_noise import random_binary_perlin_noise
from datasets.hha import getHHA
import matplotlib.pyplot as plt


class ColorJitter(object):
    """
        Applies ColorJitter on PIL Image
    """
    def __init__(self, p=1.0, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        self.p = p
        self.tf = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p={0}, '.format(self.p)
        format_string += 'tf={0})'.format(self.tf)
        return format_string

    def __call__(self, sample):
        if np.random.rand() < self.p:
            sample['img'] = self.tf(sample['img'])
        return sample


class RandomDisabler(object):
    """
        Applies ColorJitter on PIL Image
    """
    def __init__(self, p=0.5):
        self.p = p

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p={0})'.format(self.p)
        return format_string

    def __call__(self, sample):
        if np.random.rand() < self.p:
            keys = []
            if 'x' in sample:
                keys.append('x')
            if 'depth' in sample:
                keys.append('depth')
            if 'img' in sample:
                keys.append('img')
            key = np.random.choice(keys)
            sample[key] = torch.rand(sample[key].shape, dtype=sample[key].dtype)
        return sample


class RandomGray(object):
    """
        Converts PIL image into Grayscale
    """
    def __init__(self, p=0.2):
        self.p = p
        self.tf = transforms.RandomGrayscale(p=1)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p={0}, '.format(self.p)
        format_string += 'tf={0})'.format(self.tf)
        return format_string

    def __call__(self, sample):
        if np.random.rand() < self.p:
            sample['img'] = self.tf(sample['img'])
        return sample




class RandomRotation(object):
    """
        Flips PIL image and np.array PointCloud
    """
    def __init__(self, min_degree=-180,  max_degree=180):

        self.min_degree = min_degree
        self.max_degree = max_degree

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'min_degree={0}, '.format(self.min_degree)
        format_string += ', max_degree={0})'.format(self.max_degree)
        return format_string

    def __call__(self, sample):
        angle = random.uniform(self.min_degree, self.max_degree)
        sample['img'] = F.rotate(sample.get('img'), angle)
        if 'depth' in sample:
            sample['depth'] = F.rotate(sample.get('depth').unsqueeze(0), angle).squeeze(0)
        if 'mask' in sample:
            sample['mask'] = F.rotate(sample.get('mask').unsqueeze(0), angle).squeeze(0)

        return sample



class  RandomHorizontalFlip(object):
    """
        Flips PIL image and np.array PointCloud
    """
    def __init__(self, p=0.5):
        self.p = p
        self.tf = transforms.RandomHorizontalFlip(p=1)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p={0}, '.format(self.p)
        format_string += 'tf={0})'.format(self.tf)
        return format_string

    def __call__(self, sample):
        if np.random.rand() < self.p:
            sample['img'] = self.tf(sample['img'])
            if 'pc' in sample:
                sample['pc'] = self.tf(sample['pc'])
                sample['pc'][0] = -sample['pc'][0]  # also flip the x sign
            if 'depth' in sample:
                sample['depth'] = self.tf(sample['depth'])
            if 'hha' in sample:
                sample['hha'] = self.tf(sample['hha'])
            if 'gt_depth' in sample:
                sample['gt_depth'] = self.tf(sample['gt_depth'])
            if 'mask' in sample:
                sample['mask'] = self.tf(sample['mask'])

        return sample


class RandomVerticalFlip(object):
    """
        Flips PIL image and np.array PointCloud
    """
    def __init__(self, p=0.5):
        self.p = p
        self.tf = transforms.RandomVerticalFlip(p=1)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p={0}, '.format(self.p)
        format_string += 'tf={0})'.format(self.tf)
        return format_string

    def __call__(self, sample):
        if np.random.rand() < self.p:
            sample['img'] = self.tf(sample['img'])
            if 'pc' in sample:
                sample['pc'] = self.tf(sample['pc'])
                sample['pc'][1] = -sample['pc'][1]  # also flip the y sign
            if 'depth' in sample:
                sample['depth'] = self.tf(sample['depth'])
            if 'hha' in sample:
                sample['hha'] = self.tf(sample['hha'])
            if 'gt_depth' in sample:
                sample['gt_depth'] = self.tf(sample['gt_depth'])
            if 'mask' in sample:
                sample['mask'] = self.tf(sample['mask'])
        return sample


class ToTensor(object):
    """
        Converts the PIL image into a tensor and norms to range [0->1]
        Converts the PointCloud to tensor leaves it in [mm]
    """
    def __init__(self, unsqueeze_depth=False, norm_depth=False, depth_mean=1.0, depth_mean_jitter=0.05):
        self.tf = transforms.ToTensor()
        self.unsqueeze_depth = unsqueeze_depth
        self.norm_depth = norm_depth
        self.depth_mean = depth_mean
        self.depth_mean_jitter = depth_mean_jitter

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'tf={0}, '.format(self.tf)
        format_string += 'norm_depth={0}, '.format(self.norm_depth)
        format_string += 'depth_mean={0}, '.format(self.depth_mean)
        format_string += 'depth_mean_jitter={0}, '.format(self.depth_mean_jitter)
        format_string += 'unsqueeze_depth={0})'.format(self.unsqueeze_depth)
        return format_string


    def __call__(self, sample):
        if 'img' in sample:
            sample['img'] = self.tf(sample['img'])
        if 'depth' in sample:
            if isinstance(sample['depth'], Image.Image):
                sample['depth'] = torch.tensor(np.array(sample['depth']))
            if self.norm_depth:
                if sample.get('mode', '') == 'train':
                    depth_mean = self.depth_mean + \
                                 float((np.random.rand() - 0.5) * 2 * self.depth_mean_jitter) * self.depth_mean
                else:
                    depth_mean = self.depth_mean
                mask = sample['depth'] > 0
                sample['depth'][mask] += depth_mean - torch.mean(sample['depth'][mask])

            if self.unsqueeze_depth:
                if sample['depth'].dim() == 2:
                    sample['depth'] = sample['depth'].unsqueeze(0)
        img = sample['img']
        return sample




class Normalize(object):
    """
        Normalises the tensor image
    """
    def __init__(self, mean, std):
        self.tf = transforms.Normalize(mean, std)
        self.mean = mean
        self.std = std

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'std={0}, '.format(self.mean)
        format_string += 'mean={0})'.format(self.std)
        return format_string

    def __call__(self, sample):

        sample['img'] = self.tf(sample['img'])
        return sample



class DepthGTCopy(object):

    def __repr__(self):
        format_string = self.__class__.__name__ + '()'
        return format_string

    def __call__(self, sample):
        if 'depth' in sample:
            sample['depth_gt'] = copy.deepcopy(sample['depth'])
            sample['depth'] = torch.zeros(sample['depth'].shape, dtype=sample['depth'].dtype,
                                          device=sample['depth'].device)
        return sample


class NormalizeDepth(object):
    """
        Normalises the tensor image
    """
    def __init__(self, mean, std):
        self.tf = transforms.Normalize(mean, std)
        self.mean = mean
        self.std = std

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'mean={0}, '.format(self.mean)
        format_string += 'std={0})'.format(self.std)
        return format_string

    def __call__(self, sample):
        #print('what im doing here')
        if sample['depth'].dim() == 2:
            sample['depth'] = sample['depth'].unsqueeze(0)
        elif sample['depth'].dim() == 3:
            sample['depth'] = sample['depth'].unsqueeze(1)
        #print(sample['depth'].shape)

        #print('--------')
        #print(torch.max(sample['depth']), torch.min(sample['depth'][sample['depth']>0]), torch.std(sample['depth']), torch.mean(sample['depth']))
        
        #plot_depth(sample['depth'], 'norm beginning {} in meter {} | '.format('depth', sample['in_meter']))
        sample['depth'] = self.tf(sample['depth'])
        #print(torch.max(sample['depth']), torch.min(sample['depth'][sample['depth']>0]), torch.std(sample['depth']), torch.mean(sample['depth']))
        
        #print(sample['depth'].shape)
        #print(torch.mean(sample['depth']), torch.std(sample['depth']))
        
        #plot_depth(sample['depth'], 'norm end {} in meter {} | '.format('depth', sample['in_meter']))
        return sample


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'radius_min={0}, '.format(self.radius_min)
        format_string += 'radius_max={0})'.format(self.radius_max)
        return format_string

    def __call__(self, sample):
        if np.random.rand() <= self.p:
            sample['img'].filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))
        return sample


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            sample['img'] = ImageOps.solarize(sample['img'])
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p={0})'.format(self.p)
        return format_string



class DisparityAndRescale(object):
    def __init__(self, rescale_factor=75.0, depth2disparty=True, baseline=0.05, focal_length=None, min_depth=0.01, max_depth=75.0,
                    clamp_max_depth=False, clamp_min_depth=False):
        self.rescale_factor = rescale_factor
        self.depth2disparty = depth2disparty
        self.baseline = baseline
        self.focal_length = focal_length
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.clamp_max_depth = clamp_max_depth
        self.clamp_min_depth = clamp_min_depth
        self.eps = 1e-4

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'rescale_factor={0}, '.format(self.rescale_factor)
        format_string += 'depth2disparty={0}, '.format(self.depth2disparty)
        format_string += 'baseline={0}, '.format(self.baseline)
        format_string += 'min_depth={0}, '.format(self.min_depth)
        format_string += 'max_depth={0}, '.format(self.max_depth)
        format_string += 'focal_length={0})'.format(self.focal_length)
        return format_string

    def __call__(self, sample):
        #print('#####')
        #plot_depth(sample['depth'], 'dis beginning {} in meter {} | '.format('depth', sample['in_meter']))
        mask = sample['depth'] <= self.min_depth
        if self.depth2disparty:
            baseline = sample.get('baseline', self.baseline)
            focal_length = sample.get('focal_length', self.focal_length)

            if focal_length is None:
                focal_length = min(sample['depth'].shape) * 0.9
                #print('here')
            #print('focal', focal_length)     
            #print('baseline', baseline)

            #input()       


            sample['depth'] = torch.clamp(sample['depth'], min=self.min_depth, max=self.max_depth)
            #print('base', baseline, focal_length, sample['depth'].shape, sample['depth'].dtype)
            #print(torch.max(sample['depth']), torch.min(sample['depth'][sample['depth']>0]), torch.std(sample['depth']), torch.mean(sample['depth']))
            
            sample['depth'] = (baseline * focal_length) / sample['depth']
            #sample['depth'] = baseline * focal_length * sample['depth']

            #print(torch.max(sample['depth']), torch.min(sample['depth'][sample['depth']>0]), torch.std(sample['depth']), torch.mean(sample['depth']))
        
        if self.clamp_max_depth:
            sample['depth'] = torch.clamp(sample['depth'], max=self.max_depth)

        if self.clamp_min_depth:
            sample['depth'] = torch.clamp(sample['depth'], min=self.min_depth)

        sample['depth'][mask] = 0

        
        #plot_depth(sample['depth'], 'mid beginning {} in meter {} | '.format('depth', sample['in_meter']))
            
        #print(torch.max(sample['depth']), torch.min(sample['depth'][sample['depth']>0]), torch.std(sample['depth']), torch.mean(sample['depth']))

        sample['depth'] = sample['depth'] / self.rescale_factor

        
        #plot_depth(sample['depth'], 'end beginning {} in meter {} | '.format('depth', sample['in_meter']))

        #print(torch.max(sample['depth']), torch.min(sample['depth'][sample['depth']>0]), torch.std(sample['depth']), torch.mean(sample['depth']))
        #input()
        return sample




class RandomResizedCrop(object):
    """
    Randomly Crops and resizes the PIL image and the PointCloud
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BICUBIC,
                 mask_size=None):
        self.scale = scale
        self.ratio = ratio
        self.tf = transforms.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.mask_size = mask_size

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'scale={0}, '.format(self.scale)
        format_string += 'ratio={0}, '.format(self.ratio)
        format_string += 'tf={0}, '.format(self.tf)
        format_string += 'mask_size={0})'.format(self.mask_size)
        return format_string

    def __call__(self, sample):
        i, j, h, w = self.tf.get_params(sample['img'], self.scale, self.ratio)
        sample['img'] = F.crop(sample['img'], i, j, h, w)

        if 'pc' in sample:
            sample['pc'] = F.crop(sample['pc'], i, j, h, w)

        if 'depth' in sample:
            sample['depth'] = F.crop(sample['depth'], i, j, h, w)

        if 'hha' in sample:
            sample['hha'] = F.crop(sample['hha'], i, j, h, w)

        if 'gt_depth' in sample:
            sample['gt_depth'] = F.crop(sample['gt_depth'], i, j, h, w)

        if 'mask' in sample:
            sample['mask'] = F.crop(sample['mask'], i, j, h, w)

        if h != w:
            m = max(h, w)
            img = np.zeros((m, m, 3), dtype=np.uint8)
            x0 = (m - h) // 2
            y0 = (m - w) // 2
            img[x0:x0+h, y0:y0+w] = sample['img']
            sample['img'] = Image.fromarray(img)

            if 'pc' in sample:
                pc = torch.zeros((3, m , m), dtype=torch.float32)
                pc[:, x0:x0 + h, y0:y0 + w] = sample['pc']
                sample['pc'] = pc

            if 'depth' in sample:
                if len(sample['depth'].shape) > 2:
                    pc = torch.zeros((3, m, m), dtype=torch.float32)
                    pc[:, x0:x0 + h, y0:y0 + w] = sample['depth']
                    sample['depth'] = pc
                else:
                    depth = torch.zeros((m , m), dtype=torch.float32)
                    depth[x0:x0 + h, y0:y0 + w] = sample['depth']
                    sample['depth'] = depth

            if 'hha' in sample:
                pc = torch.zeros((3, m, m), dtype=torch.float32)
                pc[:, x0:x0 + h, y0:y0 + w] = sample['hha']
                sample['hha'] = pc

                #depth = np.zeros((m, m, 3), dtype=np.uint8)
                #depth[x0:x0 + h, y0:y0 + w] = sample['hha']
                #sample['hha'] = Image.fromarray(depth)

            if 'gt_depth' in sample:
                if len(sample['gt_depth'].shape) > 2:
                    pc = torch.zeros((3, m, m), dtype=torch.float32)
                    pc[:, x0:x0 + h, y0:y0 + w] = sample['gt_depth']
                    sample['gt_depth'] = pc
                else:
                    depth = torch.zeros((m , m), dtype=torch.float32)
                    depth[x0:x0 + h, y0:y0 + w] = sample['gt_depth']
                    sample['gt_depth'] = depth

            if 'mask' in sample:
                mask = torch.zeros((m, m), dtype=torch.float32)
                mask[x0:x0 + h, y0:y0 + w] = sample['mask']
                sample['mask'] = mask

        sample['img'] = F.resize(sample['img'], self.tf.size, self.tf.interpolation, antialias=True)

        if 'pc' in sample:
            sample['pc'] = F.resize(sample['pc'], self.tf.size, self.tf.interpolation, antialias=True)

        if 'depth' in sample:
            sample['depth'] = F.resize(sample['depth'].unsqueeze(0), self.tf.size,
                                       InterpolationMode.NEAREST).squeeze(0)

        if 'hha' in sample:
            sample['hha'] = F.resize(sample['hha'], self.tf.size, self.tf.interpolation, antialias=True)

        if 'gt_depth' in sample:
            sample['gt_depth'] = F.resize(sample['gt_depth'].unsqueeze(0), self.tf.size,
                                          InterpolationMode.NEAREST).squeeze(0)
        if 'mask' in sample:
            if self.mask_size is None:
                sample['mask'] = F.resize(sample['mask'].unsqueeze(0), self.tf.size,
                                          InterpolationMode.NEAREST).squeeze(0)
            else:

                sample['mask'] = F.resize(sample['mask'].unsqueeze(0), self.mask_size,
                                          InterpolationMode.NEAREST).squeeze(0)

            #print('crop random', sample['mask'].shape, min(torch.unique(sample['mask'])), max(torch.unique(sample['mask'])))

        return sample


class Pad(object):
    def __init__(self, size_divisor=32, return_roi=False):
        self.size_divisor = size_divisor
        self.return_roi = return_roi

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'size_divisor={0}, '.format(self.size_divisor)
        format_string += 'return_roi={0})'.format(self.return_roi)
        return format_string

    def __call__(self, sample):
        w, h = sample['img'].size
        w_ = w // self.size_divisor
        h_ = h // self.size_divisor
        if w % self.size_divisor != 0:
            w_ += 1
        if h % self.size_divisor != 0:
            h_ += 1
        w_ = self.size_divisor * w_
        h_ = self.size_divisor * h_
        w1 = int((w_ - w) // 2)
        h1 = int((h_ - h) // 2)
        x = np.zeros((h_, w_, 3), dtype=np.uint8)
        x[h1:h1 + h, w1:w1 + w] = np.array(sample['img'], dtype=np.uint8)
        if self.return_roi:
            sample['roi'] = [h1, h1+h, w1, w1+w]
            print('roi in tf', sample['roi'])

        sample['img'] = Image.fromarray(x)

        if 'depth' in sample:
            if len(sample['depth'].shape) == 2:
                depth = torch.zeros((h_, w_), dtype=torch.float32)
                depth[h1:h1 + h, w1:w1 + w] = sample['depth']
                sample['depth'] = depth
            else:
                depth = torch.zeros((h_, w_, 3), dtype=torch.float32)
                depth[h1:h1 + h, w1:w1 + w, :] = sample['depth']
                sample['depth'] = depth
        return sample


class Resize(object):
    def __init__(self, max_size=32, keep_size=False, interpolation=InterpolationMode.BICUBIC, min_size=None,
                 factor=None):
        self.max_size = max_size
        self.min_size = min_size
        self.keep_size = keep_size
        self.interpolation = interpolation
        self.factor = factor


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'max_size={0}, '.format(self.max_size)
        format_string += 'min_size={0}, '.format(self.min_size)
        format_string += 'keep_size={0}, '.format(self.keep_size)
        format_string += 'interpolation={0})'.format(self.interpolation)
        return format_string

    def __call__(self, sample):
        w, h = sample['img'].size
        if self.min_size is not None:
            if min(w, h) <= self.min_size:
                return sample

        if self.factor is not None:
            w_ = int(self.factor * w)
            h_ = int(self.factor * h)

        elif self.keep_size:
            if self.min_size is not None:
                f = self.min_size / min(w, h)
            else:
                f = self.max_size / max(w, h)
            w_ = int(f * w)
            h_ = int(f * h)
        else:
            if isinstance(self.max_size, tuple):
                w_, h_ = self.max_size
            else:
                w_, h_ = self.max_size, self.max_size

        sample['img'] = F.resize(sample['img'], (h_, w_), self.interpolation)

        for dkey in ['depth', 'gt_depth', 'depth_auc']:
            if dkey in sample:
                sample[dkey] = F.resize(sample[dkey].unsqueeze(0), (h_, w_), InterpolationMode.NEAREST).squeeze(0)

        #print('mask', min(torch.unique(sample['mask'])), max(torch.unique(sample['mask'])))
        if 'mask' in sample:
            sample['mask'] = F.resize(sample['mask'].unsqueeze(0), (h_, w_), InterpolationMode.NEAREST).squeeze(0)

        #print('mask2', min(torch.unique(sample['mask'])), max(torch.unique(sample['mask'])))
        return sample


class ResizePad(object):
    def __init__(self, div_factor=14, interpolation=InterpolationMode.BICUBIC):
        self.div_factor = div_factor
        self.interpolation = interpolation

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'div_factor={0}, '.format(self.div_factor)
        format_string += 'interpolation={0})'.format(self.interpolation)
        return format_string

    def __call__(self, sample):
        if isinstance(sample['img'], torch.Tensor):
            _, h, w = sample['img'].shape
        else:
            w, h = sample['img'].size

        w_issue = w%self.div_factor > 0
        h_issue = h%self.div_factor > 0

        #print('image', sample['img'].size, sample.keys())
        if w_issue or h_issue:
            if w_issue:
                w_ = (w // self.div_factor + 1) * self.div_factor
            else:
                w_ = w
            
            if h_issue:
                h_ = (h // self.div_factor + 1) * self.div_factor
            else:
                h_ = h

            sample['img'] = F.resize(sample['img'], (h_, w_), self.interpolation)
            #print('image', sample['img'].size)

            for dkey in ['depth', 'dino_depth']:
                if dkey in sample:
                    if sample[dkey].dim() == 2:
                        sample[dkey] = F.resize(sample[dkey].unsqueeze(0), (h_, w_),
                                                   InterpolationMode.NEAREST).squeeze(0)
                    else:
                        sample[dkey] = F.resize(sample[dkey], (h_, w_), InterpolationMode.NEAREST)
                    #print(dkey, sample[dkey].shape)

        return sample


class CenteredResizedCrop(object):
    """
    Randomly Crops and resizes the PIL image and the PointCloud
    """
    def __init__(self, size, interpolation=InterpolationMode.BICUBIC, mode='max'):
        self.size = size
        self.interpolation = interpolation
        self.mode = mode

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'size={0}, '.format(self.size)
        format_string += 'mode={0}, '.format(self.mode)
        format_string += 'interpolation={0})'.format(self.interpolation)
        return format_string


    def __call__(self, sample):
        w, h = sample['img'].size
        if self.mode == 'max':
            m = max(w, h)
        else:
            m = min(w, h)
            
        i = int((h-m) / 2)
        j = int((w-m) / 2)

        #print(w, h, i, j, m)

        sample['img'] = F.crop(sample['img'], i, j, m, m)
        #print(sample['img'].size)

        if 'pc' in sample:
            sample['pc'] = F.crop(sample['pc'], i, j, m, m)

        if 'depth' in sample:
            sample['depth'] = F.crop(sample['depth'], i, j, m, m)

        if 'hha' in sample:
            sample['hha'] = F.crop(sample['hha'], i, j, m, m)

        if 'gt_depth' in sample:
            sample['gt_depth'] = F.crop(sample['gt_depth'], i, j, m, m)

        if 'mask' in sample:
            sample['mask'] = F.crop(sample['mask'], i, j, m, m)

        sample['img'] = F.resize(sample['img'], self.size, self.interpolation, antialias=True)
        #import  matplotlib.pyplot as plt
        #plt.imshow(sample['img'])
        #plt.show()

        if 'pc' in sample:
            sample['pc'] = F.resize(sample['pc'], self.size, self.interpolation, antialias=True)

        if 'hha' in sample:
            sample['hha'] = F.resize(sample['hha'], self.size, self.interpolation, antialias=True)

        if 'depth' in sample:
            sample['depth'] = F.resize(sample['depth'].unsqueeze(0), self.size, InterpolationMode.NEAREST, antialias=True).squeeze(0)

        if 'gt_depth' in sample:
            sample['gt_depth'] = F.resize(sample['gt_depth'].unsqueeze(0), self.size, InterpolationMode.NEAREST, antialias=True).squeeze(0)

        if 'mask' in sample:
            sample['mask'] = F.resize(sample['mask'].unsqueeze(0), self.size, InterpolationMode.NEAREST, antialias=True).squeeze(0)
        #print(sample['img'].size)
        return sample


class StackImagePC(object):
    def __call__(self, sample):
        sample = torch.cat((sample['img'], sample['pc']), dim=0)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '()'
        return format_string

class GetBBoxFromMask(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        uni = torch.unique(sample['mask'])
        if uni[0] == 0:
            uni = uni[1:]

        bboxes = []
        h, w = sample['mask'].shape
        for u in uni:
            ys, xs = torch.where(sample['mask'] == u)
            x0, y0, x1, y1 = min(xs)/w, min(ys)/h, max(xs)/w, max(ys)/h
            wb = x1 - x0
            hb = y1 - y0
            cx = x0 + wb/2
            cy = y0 + hb/2
            bboxes.append(torch.tensor([cx, cy, wb, hb]).unsqueeze(0))
        sample['cls'] = uni.long()
        sample['bbox'] = torch.cat(bboxes)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '()'
        return format_string



class DepthPositionalEncoding(object):
    def __init__(self, depth_channels=32, temperature=10000, scale=0, normalize=False,
                 position_offset=0.0, zero_eps=1e-6, flatten_4th_dim=True, div_factors=None,
                 with_dino_head=False, enable_dino_head_epoch=0, encoder_hidden_dims=384,
                 get_depth_scales=True, 
                 max_depth=15.0):
        self.depth_channels = depth_channels
        self.temperature = temperature
        self.scale = scale
        self.normalize = normalize        
        self.position_offset = position_offset
        self.zero_eps = zero_eps
        self.flatten_4th_dim = flatten_4th_dim
        self.with_dino_head = with_dino_head
        self.enable_dino_head_epoch = enable_dino_head_epoch
        self.encoder_hidden_dims = encoder_hidden_dims
        self.depth_scales_bin_size = encoder_hidden_dims // 3
        self.depth_scales_bin_center = self.depth_scales_bin_size // 2
        self.get_depth_scales = get_depth_scales
        self.max_depth = max_depth
        self.norm_with_max_scale = True if temperature < 1 else False


        if div_factors is not None:
            dim_t = [float(df) for df in div_factors.split('/')]            
            a = []
            for df in dim_t:
                a.append(df)
                a.append(df)
            dim_t = a
            self.dim_t = torch.tensor(dim_t, dtype=torch.float32)

        else:
            dim_t = torch.arange(self.depth_channels, dtype=torch.float32)
            print('dim-t shape in DPE', dim_t.shape)
            dim_t = (2 * torch.div(dim_t, 2, rounding_mode='trunc')) / self.depth_channels
            self.dim_t = self.temperature ** dim_t  # dim_t // 2

        #print('temperature scales: {}'.format(self.dim_t.numpy()))

        p_meter = self.scale / self.dim_t.numpy() / (math.pi * 2)
        #print('cm dist: {}'.format(self.dim_t.numpy()))
        #print('max dist per layer dist: {}'.format(1 / p_meter))

        

        print('max dist per layer dist [%]: {}'.format([np.round(v, 4) for i, v in enumerate(1/p_meter) if i%2 == 0]))
        print('max dist per layer dist [m]: {}'.format([np.round(v, 4) for i, v in enumerate(self.max_depth/p_meter) if i%2 == 0]))
        print('max dist per layer dist [mm]: {}'.format([np.round(v, 4) for i, v in enumerate(self.max_depth*1000/p_meter) if i%2 == 0]))

        print('max dist of first layer: {}, 1 = {}%, 0.1 = {}%'.format(np.round(1/p_meter[0], 6),
                                                                     np.round(p_meter[0]*100, 6),
                                                                  np.round(p_meter[0], 6)))
        print('max dist of last layer: {}, 1 = {}%, 0.001 = {}%'.format(np.round(1/p_meter[-1], 6),
                                                                    np.round(p_meter[-1] * 10000, 6),
                                                                  np.round(p_meter[-1]*1, 6)))


        self.dim_t_dict = {}

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'dim_t={0}'.format(self.dim_t)
        format_string += ', depth_channels={0}'.format(self.depth_channels)
        format_string += ', temperature={0}'.format(self.temperature)
        format_string += ', scale={0}'.format(self.scale)
        format_string += ', normalize={0}'.format(self.normalize)
        format_string += ', position_offset={0}'.format(self.position_offset)
        format_string += ', zero_eps={0}'.format(self.zero_eps)
        format_string += ', flatten_4th_dim={0}'.format(self.flatten_4th_dim)
        format_string += ', temperature={0}'.format(self.temperature)
        format_string += ', max_depth={0}'.format(self.max_depth)
        format_string += ', norm_with_max_scale={0}'.format(self.norm_with_max_scale)
        format_string += ', get_depth_scales={0}'.format(self.get_depth_scales)
        format_string += ', with_dino_head={0})'.format(self.with_dino_head)
        return format_string

    def __call__(self, sample):
        #
        #a, b = sample['gt_depth'].shape
        #print('a: {}'.format(torch.sum(sample['gt_depth'] > 0) / (a * b) * 100))

        #h, w = sample['depth'].shape
        #print('b: {}'.format(torch.sum(sample['depth'] > 0) / (h * w) * 100))
        #print(a,b, h, w)
        #print(sample['gt_depth'].shape, sample['depth'].shape)
        # print('sample is inside transforms.py transforms', sample)
        if 'depth' in sample:
            #print('before', sample['depth'].shape)
            if sample['depth'] is not None:
                #print(torch.mean(sample['depth']), torch.mean(sample['depth'][sample['depth'] > 0]))
                if self.with_dino_head and sample.get('epoch', 0) >= self.enable_dino_head_epoch - 1:
                    if isinstance(sample['gt_depth'], list):
                        sample['dino_depth'], sample['dino_depth_scales'] = self.encode(sample['gt_depth'][0].clone())
                    else:
                        sample['dino_depth'], sample['dino_depth_scales'] = self.encode(sample['gt_depth'].clone())
                sample['depth'], sample['depth_scales'] = self.encode(sample['depth'])
                
                #print('depth_scales type in DepthPosEncTranform in transforms',type(sample['depth_scales']))
                # print('depth_scales in DepthPosEncTranform in transform', sample['depth_scales'])
                # print('depth_scales in DepthPosEncTranform in transforms',sample['depth_scales'].shape)
                #print('depth data in DepthPosEncTranform in transforms',sample['depth'].shape)
                
                #print('after', sample['depth'].shape)
                # return sample
            
          
        
            #print('after2', sample['depth'].shape)
        # input()
        return sample

    def encode(self, depth):
        #if depth.device != self.dim_t.device:
        #    self.dim_t = self.dim_t.to(depth.device)
        #print('here', torch.mean(depth), torch.std(depth))
        #if depth.device !
        #print('depth dim is', depth.dim())
        #print('depth shape in DPE before encode', depth.shape)
        d_id = str(depth.get_device())
        if d_id == '-1':
            dim_t = self.dim_t
        else:
            if d_id not in self.dim_t_dict:
                self.dim_t_dict[d_id] = self.dim_t.to(depth.device)
            dim_t = self.dim_t_dict[d_id]

        if self.zero_eps > 0:
            depth[depth == 0] = self.zero_eps
        
        if self.get_depth_scales:
            depth_scales = torch.zeros(self.encoder_hidden_dims)
        else:
            depth_scales = None

        if depth.dim() == 2:
            if self.get_depth_scales:
                #print('max_depth in DPE',torch.max(depth))
                upper = format(float(torch.max(depth)), '20f').replace(' ', '')
                #print('upper in DPE', upper)
                if '.' in upper:
                    upper, lower = upper.split('.')
                else:
                    lower = '0'

                embed = torch.tensor([0.1 + float(v)/10 for v in upper+lower]) # map 0->9 to 0.1:1.0. Zeros are 0.1 and missing values are 0.0
                
                #print('embed shape in DPE',embed.shape)
                #print('len embed in DPE', len(embed))
                
                in_index = 2 * self.depth_scales_bin_size + (self.depth_scales_bin_center-len(upper))
                #print('in index in DPE', in_index)
                depth_scales[in_index:in_index+len(embed)] = embed
                depth = depth.div(depth.max()) # normalize depth
                #print('depth shape inside DPE before norm_max_scale',depth.shape)
            elif self.norm_with_max_scale:
                depth = depth.div(self.max_depth)
                #print('depth shape inside DPE after norm_max_scale',depth.shape)
            if self.scale > 0:
                depth = depth * self.scale

            depth = depth[:, :, None] / dim_t
            #print('depth shape inside DPE before permute',depth.shape)
            depth = torch.stack((depth[:, :, 0::2].sin(), depth[:, :, 1::2].cos()), dim=3).flatten(2)
            depth = depth.permute(2, 0, 1)
            #print('depth shape inside DPE',depth.shape)
        elif depth.dim() == 3 and len(depth) == 3: # depth 3D (PointCloud)
            if self.get_depth_scales:
                for index, d in enumerate(depth):
                    upper, lower = format(float(torch.max(depth)), '20f').replace(' ', '').split('.')
                    embed = torch.tensor([0.1 + float(v)/10 for v in upper+lower]) # map 0->9 to 0.1:1.0. Zeros are 0.1 and missing values are 0.0
                    in_index = index * self.depth_scales_bin_size + (self.depth_scales_bin_center-len(upper))
                    depth_scales[in_index:in_index+len(embed)] = embed
                    depth[index] = d.div(d.max())
            elif self.norm_with_max_scale:
                depth = depth.div(self.max_depth)

            if self.scale > 0:
                depth = depth * self.scale

            depth = depth[:, :, :, None] / dim_t            
            depth = torch.stack((depth[:, :, :, 0::2].sin(), depth[:, :, :, 1::2].cos()), dim=4).flatten(3)
                    
            depth = depth.permute(0, 3, 1, 2)
            if self.flatten_4th_dim:
                c, dims, h, w = depth.shape
                depth = depth.reshape(dims * c, h, w)
            

        elif depth.dim() == 3: # multiple depth versions
            if self.get_depth_scales:
                depth_scales = depth_scales.unsqueeze(0).repeat(len(depth), 1)
            depth_ = []
            for ii, d in enumerate(depth):

                #(d, 'pde {} |'.format(ii), self.zero_eps)
                
                if self.get_depth_scales:
                    upper, lower = format(float(torch.max(d)), '20f').replace(' ', '').split('.')
                    embed = torch.tensor([0.1 + float(v)/10 for v in upper+lower]) # map 0->9 to 0.1:1.0. Zeros are 0.1 and missing values are 0.0
                    in_index = 2 * self.depth_scales_bin_size + (self.depth_scales_bin_center-len(upper))
                    depth_scales[ii, in_index:in_index+len(embed)] = embed
                    d = d.div(d.max()) # normalize depth
                elif self.norm_with_max_scale:
                    d = d.div(self.max_depth)

                if self.scale > 0:
                    d = d * self.scale

                d = d[:, :, None] / dim_t
                d = torch.stack((d[:, :, 0::2].sin(), d[:, :, 1::2].cos()), dim=3).flatten(2)
                d = d.permute(2, 0, 1).unsqueeze(0)
                depth_.append(d)
                #print(d.shape)
            depth = torch.cat(depth_, dim=0)
            #print('depth shape inside DPE',depth.shape)
        elif depth.dim() == 4: # depth 3D (PointCloud)
            #print('depth here', depth.shape, dim_t.shape)
            
            depth = depth.squeeze(1)
            #print('depth here', depth.shape, dim_t.shape)
            
            if self.get_depth_scales:
                upper, lower = format(float(torch.max(depth)), '20f').replace(' ', '').split('.')
                embed = torch.tensor([0.1 + float(v)/10 for v in upper+lower]) # map 0->9 to 0.1:1.0. Zeros are 0.1 and missing values are 0.0
                in_index = index * self.depth_scales_bin_size + (self.depth_scales_bin_center-len(upper))
                depth_scales[in_index:in_index+len(embed)] = embed
                depth[index] = d.div(d.max())

            elif self.norm_with_max_scale:
                depth = depth.div(self.max_depth)

            if self.scale > 0:
                depth = depth * self.scale

            depth = depth[:, :, :, None] / dim_t            
            depth = torch.stack((depth[:, :, :, 0::2].sin(), depth[:, :, :, 1::2].cos()), dim=4).flatten(3)
                    
            depth = depth.permute(0, 3, 1, 2)
            #if len(depth) > 1:
            #    depth = depth.unsqueeze(0)


        #elif depth.dim() == 4:
        #    depths = []
        #    for i, d in enumerate(depth):
        #        #if d.device != self.dim_t.device:
        #        #    self.dim_t = self.dim_t.to(d.device)
        #        d = d[:, :, :, None] / dim_t
        #        d = torch.stack((d[:, :, :, 0::2].sin(),
        #                            d[:, :, :, 1::2].cos()), dim=4).flatten(3)        #    
        #        d = d.permute(0, 3, 1, 2)
        #        if self.flatten_4th_dim:
        #            c, dims, h, w = d.shape
        #            d = d.reshape(dims * c, h, w).unsqueeze(0)
        #        depths.append(d)
        #    depth = torch.cat(depths)
        else:
            raise NotImplementedError('Depth encoding for dim {} is not implemented, got shape'.format(
                depth.dim(), depth.shape))

        return depth, depth_scales
        
def plot_depth(d, msg='', t=0.0):
    m = d > t
    c = float(sum(m.flatten()))
    p = float(np.prod([float(v) for v in d.shape]))
    if isinstance(d, torch.Tensor):
        print('{} mean: {}, std: {}, min: {}, max: {}, p: {}%'.format(msg, float(torch.mean(d[m])), float(torch.std(d[m])), float(torch.min(d[m])), float(torch.max(d[m])), np.round(100*c/p, 3)))
    else:
        print('{} mean: {}, std: {}, min: {}, max: {}, p: {}%'.format(msg, float(np.mean(d[m])), float(np.std(d[m])), float(np.min(d[m])), float(np.max(d[m])), np.round(100*c/p,3)))


class Depth2distance(object):
    def __init__(self, to_meter: bool = True, max_dist=150, depth_mul=1.0, random_to_m=0.0, rescale=False, cut_dist=None,
        rescale_p=None, allways_rescale=False, allways_rescale_p=None, strech_depth=False, strech_depth_p=None):
        self.to_meter = to_meter
        self.max_dist = max_dist
        self.depth_mul = depth_mul
        self.random_to_m = random_to_m
        self.rescale = rescale
        if cut_dist is None:
            cut_dist = max_dist
        self.cut_dist = cut_dist

        if rescale_p is None:
            rescale_p = 0.95


        self.rescale_p = rescale_p
        self.allways_rescale = allways_rescale
        if allways_rescale_p is None:
            allways_rescale_p = 1.0

        self.allways_rescale_p = allways_rescale_p
        self.strech_depth = strech_depth
        if strech_depth_p is None:
            strech_depth_p = 1.0
        self.strech_depth_p = strech_depth_p


    def __call__(self, sample):
        #print('depth_scale in Depth2distance', sample['depth_scale'])
        #print('there', sample.get('depth_scale', 1.0))
        sample['depth_scale'] = sample.get('depth_scale', 1.0) * self.depth_mul
        
        for dkey in ['gt_depth', 'depth_auc', 'depth']:
            if dkey in sample:
                #print('---------------------')
                #print(dkey, 'start', torch.mean(sample[dkey][sample[dkey]>0]),torch.std(sample[dkey][sample[dkey]>0]))
                sample[dkey] = sample[dkey] * sample['depth_scale']
                in_meter = False
                if self.to_meter:
                    in_meter = True
                    sample[dkey] = sample[dkey] / 1000
                elif self.random_to_m < np.random.rand():
                    in_meter = True
                    sample[dkey] = sample[dkey] / 1000
                sample['in_meter'] = in_meter

                #plot_depth(sample[dkey], 'here {} in meter {} | '.format(dkey, sample['in_meter']))

                if self.max_dist is not None:       
                    max_dist = self.max_dist
                    cut_dist = self.cut_dist
                    if not in_meter:
                        max_dist = max_dist * 1000
                        cut_dist = cut_dist * 1000

                    if self.rescale:
                        dkey_max = torch.max(sample[dkey])
                        ds = sample.get('depth_mul', 1.0)
                        #print('this', dkey_max, max_dist, ds, max_dist*self.rescale_p)
                        
                        if self.allways_rescale:
                            cs = sample.get('const_scale', None)
                            if cs is None:
                                sample['const_scale'] = cut_dist * self.allways_rescale_p / dkey_max
                            sample[dkey] = sample[dkey] * sample['const_scale'] 
                        dkey_max = torch.max(sample[dkey])

                        #print('strech_depth', self.strech_depth, ds, self.strech_depth_p)
                        if self.strech_depth: 
                            #if ds != 1.0: #dkey_max < max_dist*self.strech_depth_p or
                            if ds == 1.0:
                                if sample.get('mode', '') == 'train':    
                                    scale = (0.9 + float(np.random.rand() * 0.1))
                                else:
                                    scale = self.strech_depth_p
                                ds = (1 / dkey_max) * (max_dist * scale)
                                sample['depth_mul'] = ds                     
                            sample[dkey] = sample[dkey] * ds 
                            #print('stratching {} max {} with {} to {}'.format(dkey, dkey_max, ds, torch.max(sample[dkey]))) 

                        elif dkey_max > max_dist*self.rescale_p or ds != 1.0:
                            if ds == 1.0:
                                if sample.get('mode', '') == 'train':    
                                    scale = (0.9 + float(np.random.rand() * 0.1))
                                else:
                                    scale = self.rescale_p
                                ds = (1 / dkey_max) * (max_dist * scale)
                                sample['depth_mul'] = ds                     
                            sample[dkey] = sample[dkey] * ds 
                            #print('rescaling {} max {} with {} to {}'.format(dkey, dkey_max, ds, torch.max(sample[dkey]))) 
                                                        
                    sample[dkey][sample[dkey] > cut_dist] = 0
                #print('data mul', sample['depth_mul'])
                #plot_depth(sample[dkey], 'again {} in meter {} | '.format(dkey, sample['in_meter']))
                    
                #print(dkey, 'end', torch.mean(sample[dkey][sample[dkey]>0]),torch.std(sample[dkey][sample[dkey]>0]))
                #print('#################')

        #if 'depth_scale_rescaled' in sample:
        #    sample['depth_scale'] = sample['depth_scale'] * sample['depth_scale_rescaled']
        #print('sample depth shape in Depth2Distance is ', sample['depth'].shape)
        # input()
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'to_meter={0}'.format(self.to_meter)
        format_string += ', max_dist={0}'.format(self.max_dist)
        format_string += ', cut_dist={0}'.format(self.cut_dist)
        format_string += ', rescale={0}'.format(self.rescale)
        format_string += ', rescale_p={0}'.format(self.rescale_p)
        format_string += ', allways_rescale={0}'.format(self.allways_rescale)
        format_string += ', strech_depth={0}'.format(self.strech_depth)
        format_string += ', strech_depth_p={0}'.format(self.strech_depth_p)
        format_string += ', depth_mul={0})'.format(self.depth_mul)
        return format_string

class GTDepth(object):
    def __init__(self, fpn=False, fpn_layers: int = 5, thresholds=None, over_write_depth_gt=False,
                 add_depth_2_auc=False, p_perlin=None):
        self.fpn = fpn
        self.fpn_layers = fpn_layers
        self.thresholds = thresholds
        self.over_write_depth_gt = over_write_depth_gt
        self.add_depth_2_auc = add_depth_2_auc
        self.p_perlin = 0.0 if p_perlin is None else p_perlin

    def __call__(self, sample):
        if 'gt_depth' not in sample:
            sample['gt_depth'] = copy.deepcopy(sample['depth'])


        if 'ori_depth' not in sample:
            sample['ori_depth'] = copy.deepcopy(sample['depth'])


        elif self.over_write_depth_gt and self.thresholds is not None:
            sample['depth'] = copy.deepcopy(sample['gt_depth'])

        if 'depth_auc' in sample:
            if self.add_depth_2_auc:
                sample['depth_auc'] = torch.cat([sample['depth'].unsqueeze(0), sample['depth_auc']])
            sample['depth'] = sample['depth_auc']

        elif self.thresholds is not None:

            h, w = sample['gt_depth'].shape
            depth = torch.zeros((len(self.thresholds), h, w),
                                dtype=sample['gt_depth'].dtype,
                                device=sample['gt_depth'].device)

            if sample.get('sample_index') is not None:
                #print('seed', sample.get('sample_index'))
                torch.manual_seed(int(sample.get('sample_index')))
                np.random.seed(int(sample.get('sample_index')))

            mask = torch.rand(depth.shape, device=depth.device)
            for i, (t, m) in enumerate(zip(self.thresholds, mask)):
                if np.random.rand() < self.p_perlin:
                    m_ = random_binary_perlin_noise(shape=(h, w), t=t)
                    #print('perlin', t, torch.sum(m_) / (h*w))
                else:
                    m_ = m <= t
                    #print('random', t, torch.sum(m_) / (h*w))

                depth[i][m_] = sample['gt_depth'][m_]
                '''
                if False:
                    plt.subplot(2, 2, 1)
                    plt.imshow(sample['img'])
                    plt.subplot(2, 2, 2)
                    plt.imshow(sample['gt_depth'])
                    plt.subplot(2, 2, 3)
                    plt.imshow(m_.numpy())
                    plt.subplot(2, 2, 4)
                    plt.imshow(depth[i].numpy())
                    plt.show()
                '''

            sample['depth'] = depth



        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'fpn={0}'.format(self.fpn)
        format_string += ', fpn_layers={0})'.format(self.fpn_layers)
        return format_string

class DepthNoise(object):
    def __init__(self, depth_noise=0.003, min_depth=0.0, max_depth=500, p_chaos = 0.5, t_chaos=0.0001):
        self.depth_noise = depth_noise
        self.min_depth = min_depth
        self.max_depth = max_depth * 0.95
        self.p_chaos = p_chaos
        self.t_chaos = t_chaos



    def __call__(self, sample):
        depth_nosie = ((torch.rand(sample['depth'].shape) - 0.5) * 2) * self.depth_noise
        noise_mask = sample['depth'] > 0
        sample['depth'][noise_mask] += depth_nosie[noise_mask]

        if np.random.rand() < self.p_chaos:
            noise_mask = torch.rand(sample['depth'].shape) < self.t_chaos # where
            depth_nosie = torch.rand(sample['depth'].shape)    # how much
            if np.random.rand() < 0.5: # scalar
                depth_nosie = depth_nosie * self.max_depth # full chaos
            else:
                depth_nosie = depth_nosie * (torch.max(sample['depth']) * 1.5) # relative chaos 
            sample['depth'][noise_mask] = depth_nosie[noise_mask] # apply total chaos values    
        
        sample['depth'][sample['depth'] < self.min_depth] = 0
        sample['depth'][sample['depth'] > self.max_depth] = self.max_depth

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += ', min_depth={0}'.format(self.min_depth)
        format_string += ', max_depth={0}'.format(self.max_depth)
        format_string += ', p_chaos={0}'.format(self.p_chaos)
        format_string += ', t_chaos={0}'.format(self.t_chaos)
        format_string += ', depth_noise={0})'.format(self.depth_noise)
        return format_string

        
class CopyImages(object):
    def __init__(self, keys=None):
        self.keys = keys
        if self.keys is None:
            self.keys = ['img', 'depth']
        
    def __call__(self, sample):
        for key in self.keys:
            if key not in sample:
                continue
            sample[key+'_copy'] = copy.deepcopy(sample[key])
        return sample


class CourruptDepth(object):
    def __init__(self, missing=None, bad=None, noisy=None, constant=None, offset=0.05, max_depth=15.0):
        self.missing = missing
        self.bad = bad
        self.noisy = noisy
        self.constant = constant
        self.offset = offset
        self.max_depth = max_depth
        

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'missing={0}, '.format(self.missing)
        format_string += 'bad={0}, '.format(self.bad)
        format_string += 'noisy={0}, '.format(self.noisy)
        format_string += 'constant={0}, '.format(self.constant)
        format_string += 'max_depth={0}, '.format(self.max_depth)
        format_string += 'offset={0})'.format(self.offset)
        return format_string

    def __call__(self, sample):

        if self.missing is not None:
            mask = torch.rand(sample['depth'].shape) < self.missing
            sample['depth'][mask] = 0

        if self.bad is not None:
            mask = torch.rand(sample['depth'].shape) < self.bad
            noise = torch.rand(sample['depth'].shape, dtype=sample['depth'].dtype, device=sample['depth'].device) * self.max_depth
            sample['depth'][mask] = noise[mask]

        if self.constant is not None:
            mask = torch.rand(sample['depth'].shape) < self.constant            
            sample['depth'][mask] = sample['depth'][mask] + self.offset

        
        if self.noisy is not None:
            mask = torch.rand(sample['depth'].shape) < self.noisy
            noise = torch.normal(0, self.offset, size=sample['depth'].shape)
            sample['depth'][mask] = sample['depth'][mask] + noise[mask]
            sample['depth'][sample['depth'] < 0] = 0
        
        return sample


class RandomDepth(object):
    def __init__(self, max_depth=100, min_depth=0.1, bins=None, p_add=0.25, p_rescale=0.25, p_depth_jitter=0.25,
                 bin_jitter=0.1, depth_jitter=0.2, start_epoch=0, rescale=False):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.bins = bins
        self.bin_jitter = bin_jitter
        if self.bins is None:
            self.bins = [
                [self.min_depth * (1 + bin_jitter) + float(v)/10 for v in range(0, 10)], # very close (depth < 1m)
                [float(v)/10 for v in range(5, 22, 2)], # near (0.5m< depth <2.1m)
                [float(v)/10 for v in range(5, 51, 5)], # room (0.5m< depth <5m)
                [float(v) for v in range(1, 16)], # indoor (1m< depth <15m)
                [float(v) for v in range(50, int(self.max_depth * 0.75) + 1, 10)] # far (5m< depth <max depth m)
                #self.min_depth * (1 + bin_jitter), 0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 30, 50, 70, 150, 250,
                #self.max_depth * (1 - self.bin_jitter)
            ]
            self.bins = [[b for b in bins if b < self.max_depth] for bins in self.bins]
            self.bins = [b for b in self.bins if len(b) > 0]            
            
        self.p_add = p_add
        self.p_rescale = p_rescale + self.p_add
        self.p_depth_jitter = p_depth_jitter + self.p_rescale
        self.depth_jitter = depth_jitter
        self.start_epoch = start_epoch
        self.rescale = rescale
        print('Random Depth: {}'.format(self))

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'bins={0}, '.format(['({}->{}))'.format(b[0], b[-1]) for b in self.bins])
        format_string += 'max_depth={0}, '.format(self.max_depth)
        format_string += 'min_depth={0}, '.format(self.min_depth)
        format_string += 'bin_jitter={0}, '.format(self.bin_jitter)
        format_string += 'p_add={0}, '.format(self.p_add)
        format_string += 'p_rescale={0}, '.format(self.p_rescale)
        format_string += 'p_depth_jitter={0}, '.format(self.p_depth_jitter)
        format_string += 'depth_jitter={0}, '.format(self.depth_jitter)
        format_string += 'rescale={0}, '.format(self.rescale)
        format_string += 'start_epoch={0})'.format(self.start_epoch)
        return format_string

    def __call__(self, sample):

        if sample.get('epoch', 0) < self.start_epoch:
            return sample

        depth_ori = copy.deepcopy(sample['depth'])
        try:
            p = np.random.rand()
            in_meter = sample.get('in_meter', False)
            if not in_meter:
                sample['depth'] = sample['depth'] / 1000

            #print('p', p)
            if p < self.p_add:
                mask = sample['depth'] > 0
                j = float((np.random.rand() - 0.5) * 2)
                sample['depth'][mask] += torch.std(sample['depth'][mask]) * j
                if torch.min(sample['depth'][mask]) < self.min_depth:
                    sample['depth'][mask] += -torch.min(sample['depth'][mask]) + self.min_depth               

            elif p < self.p_rescale:
                zero_mask = sample['depth'] <= 0
                bin_id = int(np.random.choice(range(len(self.bins)), 1)[0])
                m = np.random.choice(self.bins[bin_id], 2, replace=False)
                j1 = 1 + (float(np.random.rand()) * ((self.bin_jitter * 2) - self.bin_jitter))
                j2 = 1 + (float(np.random.rand()) * ((self.bin_jitter * 2) - self.bin_jitter))
                a = min(m) * j1
                b = max(m) * j2
                if a < self.min_depth:
                    a = self.min_depth
                if b > self.max_depth:
                    b = self.max_depth
                sample['depth'] = sample['depth'] - torch.min(sample['depth'][~zero_mask])
                sample['depth'] = sample['depth'] / torch.max(sample['depth'])
                sample['depth'] = sample['depth'] * (b - a) + a
                sample['depth'][zero_mask] = 0
                #print('scale', a, b)

            elif p < self.p_depth_jitter:
                j = 1 + (float(np.random.rand()) * ((self.depth_jitter * 2) - self.depth_jitter))
                sample['depth'] = sample['depth'] * j
                #print('mul', j)

            
            if self.rescale:
                if torch.max(sample['depth']) > self.max_depth:                 
                    sample['depth'] = (sample['depth'] / torch.max(sample[dkey])) * (max_dist * (0.9 + float(np.random.rand() * 0.1)))

            sample['depth'][sample['depth'] < self.min_depth] = 0
            sample['depth'][sample['depth'] > self.max_depth] = 0

            if not in_meter:
                sample['depth'] = sample['depth'] * 1000

        except Exception as e:
            #print('error at RandomDist Transform: {}'.format(e))
            #raise e
            sample['depth'] = depth_ori
            return sample
        return sample

class VanishingDepthValid(object):
    def __init__(self, thresholds, fpn: bool = False, fpn_layers: int = 5, stack_thresholds: bool = False,
                 perlin_nosie: float = 0.75, use_perlin=True):

        self.thresholds = thresholds
        self.fpn = fpn
        self.fpn_layers = fpn_layers
        self.stack_thresholds = stack_thresholds
        self.perlin_nosie = perlin_nosie
        self.use_perlin = use_perlin

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'thresholds={0}, '.format(self.thresholds)
        format_string += 'fpn={0}, '.format(self.fpn)
        format_string += 'fpn_layers={0}, '.format(self.fpn_layers)
        format_string += 'stack_thresholds={0}, '.format(self.stack_thresholds)
        format_string += 'perlin_nosie={0}, '.format(self.perlin_nosie)
        format_string += 'use_perlin={0})'.format(self.use_perlin)
        return format_string

    def __call__(self, sample):
        #print('vd', sample.keys())
        if self.fpn:
            s = sample['gt_depth'].shape
            if len(s) == 2:
                h, w = s
            else:
                h,w = s[1], s[2]
            gt_depth = [sample['gt_depth']]
            for _ in range(self.fpn_layers - 1):
                w = int(w / 2)
                h = int(h / 2)
                gt_depth.append(
                    F.resize(sample['gt_depth'].unsqueeze(0), (h, w), InterpolationMode.NEAREST, antialias=True).squeeze(0))
            sample['gt_depth'] = gt_depth

        if 'gt_mask' not in sample:
            if self.stack_thresholds:
                if self.fpn:
                    gt_mask = [[] for _ in range(self.fpn_layers)]
                else:
                    gt_mask = []
                for t in self.thresholds:
                    s = sample['depth'].shape
                    if len(s) == 2:
                        h, w = s
                        depth = sample['depth']
                    else:
                        h, w = s[1], s[2]
                        depth = sample['depth'][2]
                    
                    pnoise=np.random.rand()
                    
                    print(self.perlin_nosie, pnoise)
                    if pnoise < self.perlin_nosie:
                        pperlin=np.random.rand()
                        print(self.use_perlin, pperlin)
                        if pperlin < 0.5 and self.use_perlin:
                            mask = random_binary_perlin_noise(shape=(h, w), t=t)
                            print('this?')
                        else:
                            print('here')
                            h_small = np.random.rand() * h * 0.75
                            w_small = np.random.rand() * w * 0.75
                            if h_small < 5:
                                h_small = 5
                            if w_small < 5:
                                w_small = 5
                            mask = torch.rand((int(h_small), int(w_small)))
                            mask = F.resize(mask.unsqueeze(0), (h, w), InterpolationMode.NEAREST, antialias=True).squeeze(0)
                            mask = mask < t
                        #print('t perlin', t)
                    else:
                        mask = torch.rand((h, w))
                        mask = mask < t
                        #print('t rand', t)
                        #print('there')

                    mask[depth == 0] = False
                    mask = mask.unsqueeze(0)
                    if self.fpn:
                        _, h, w = mask.shape
                        mask = [mask]
                        for _ in range(self.fpn_layers - 1):
                            h = int(h / 2)
                            w = int(w / 2)
                            mask.append(F.resize(mask[0], (h, w), InterpolationMode.NEAREST, antialias=True))
                    if self.fpn:
                        for i, m in enumerate(mask):
                            gt_mask[i].append(mask[i])
                    else:
                        gt_mask.append(mask)

                if self.fpn:
                    for i in range(len(gt_mask)):
                        gt_mask[i] = torch.cat(gt_mask[i], dim=0)
                    c = len(gt_mask[0])
                else:
                    gt_mask = torch.cat(gt_mask, dim=0)
                    c = len(gt_mask)

                sample['gt_mask'] = gt_mask
                if len(sample['depth'].shape) == 2:
                    sample['depth'] = sample['depth'].unsqueeze(0).repeat((c, 1, 1))
                else:
                    sample['depth'] = sample['depth'].unsqueeze(0).repeat((c, 1, 1, 1))
            else:
                t = np.random.choice(self.thresholds)
                s = sample['depth'].shape
                if len(s) == 2:
                    h, w = s
                    depth = sample['depth']
                else:
                    h, w = s[1], s[2]
                    depth = sample['depth'][2]

                if np.random.rand() < self.perlin_nosie:
                    if self.use_perlin:
                        mask = random_binary_perlin_noise(shape=(h, w), t=t)
                    else:
                        h_small = np.random.rand() * h * 0.75
                        w_small = np.random.rand() * w * 0.75
                        if h_small < 5:
                            h_small = 5
                        if w_small < 5:
                            w_small = 5
                        mask = torch.rand((int(h_small), int(w_small)))
                        mask = F.resize(mask.unsqueeze(0), (h, w), InterpolationMode.NEAREST, antialias=True).squeeze(0)
                        mask = mask < t

                    #mask = random_binary_perlin_noise(shape=(h, w), t=t)
                else:
                    mask = torch.rand((h, w))
                    mask = mask < t
                mask[depth == 0] = False
                mask = mask.unsqueeze(0)
                if self.fpn:
                    _, h, w = mask.shape
                    mask = [mask]
                    for _ in range(self.fpn_layers - 1):
                        h = int(h / 2)
                        w = int(w / 2)
                        mask.append(F.resize(mask[0], (h, w), InterpolationMode.NEAREST, antialias=True))
                sample['gt_mask'] = mask
        else:
            if self.stack_thresholds:
                if self.fpn:
                    c = len(sample['gt_mask'][0])
                else:
                    c = len(sample['gt_mask'])

                #print('hha valid', sample.keys())
                if 'hha' in sample:
                    sample['depth'] = sample['hha']

                if len(sample['depth'].shape) == 2:
                    sample['depth'] = sample['depth'].unsqueeze(0).repeat((c, 1, 1))
                else:
                    sample['depth'] = sample['depth'].unsqueeze(0).repeat((c, 1, 1, 1))


        if isinstance(sample['gt_mask'], list):
            sample['depth'][sample['gt_mask'][0]] = 0
        else:
            if len(sample['depth'].shape)== 4:
                sample['depth'][sample['gt_mask'].unsqueeze(1).repeat(1,3,1,1)] = 0
            else:
                sample['depth'][sample['gt_mask']] = 0

        #print('asdfsa', sample['depth'].shape)

        return sample


class VanishingDepth(object):
    def __init__(self, fpn=False, fpn_layers=5, perlin_nosie: float = 0.75, use_perlin=True, p_pixel=0.0):
        self.fpn = fpn
        self.fpn_layers = fpn_layers
        self.perlin_nosie = perlin_nosie
        self.use_perlin = use_perlin
        self.p_pixel = p_pixel

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'fpn={0}, '.format(self.fpn)
        format_string += 'fpn_layers={0}, '.format(self.fpn_layers)
        format_string += 'perlin_nosie={0}, '.format(self.perlin_nosie)
        format_string += 'p_pixel={0}, '.format(self.p_pixel)
        format_string += 'use_perlin={0})'.format(self.use_perlin)
        return format_string

    def __call__(self, sample):
        if self.fpn:
            s = sample['gt_depth'].shape
            if len(s) == 2:
                h, w = s
            else:
                h, w = s[1], s[2]

            gt_depth = [sample['gt_depth']]
            for _ in range(self.fpn_layers - 1):
                w = int(w / 2)
                h = int(h / 2)
                gt_depth.append(
                    F.resize(sample['gt_depth'].unsqueeze(0), (h, w), InterpolationMode.NEAREST, antialias=True).squeeze(0))
            sample['gt_depth'] = gt_depth

        if 'gt_mask' not in sample:
            s = sample['depth'].shape
            if len(s) > 2:
                s = s[1:]

            if np.random.rand() > self.p_pixel:
                if self.use_perlin and np.random.rand() < self.perlin_nosie:
                    mask = random_binary_perlin_noise(shape=s, t=sample['t'])
                    #print('perin')
                else:
                    #print('resized')
                    h, w = s[-2:] 
                    h_small = np.random.rand() * h * 0.75
                    w_small = np.random.rand() * w * 0.75
                    if h_small < 10:
                        h_small = 10
                    if w_small < 10:
                        w_small = 10
                    mask = torch.rand((int(h_small), int(w_small)))
                    mask = F.resize(mask.unsqueeze(0), s, InterpolationMode.NEAREST, antialias=True).squeeze(0)
                    mask = mask < sample['t']
                    #print('there', sample['t'], h_small, w_small, np.sum(mask.numpy()) / (h*w), mask.shape, h, w)

                #mask = random_binary_perlin_noise(shape=s, t=sample['t'])
                #print('t perlin', sample['t'])
            else:
                #print('here', sample['t'])
                mask = torch.rand(s)
                mask = mask < sample['t']
                #print('t rand', sample['t'])

            if len(sample['depth'].shape) > 2:
                mask[sample['depth'][0] == 0] = False
            else:
                mask[sample['depth'] == 0] = False

            if self.fpn:
                h, w = mask.shape
                mask = [mask]
                for _ in range(self.fpn_layers - 1):
                    h = int(h/2)
                    w = int(w/2)
                    mask.append(F.resize(mask[0].unsqueeze(0), (h, w), InterpolationMode.NEAREST, antialias=True).squeeze(0))
            sample['gt_mask'] = mask

        if 'hha' in sample:
            sample['depth'] = sample['hha']

        if isinstance(sample['gt_mask'], list):
            if len(sample['depth'].shape) == 2:
                sample['depth'][sample['gt_mask'][0]] = 0
            else:
                for i in range(sample['depth'].shape[0]):
                    sample['depth'][i][sample['gt_mask'][0]] = 0
        else:
            if len(sample['depth'].shape) == 2:
                sample['depth'][sample['gt_mask']] = 0
            else:
                for i in range(sample['depth'].shape[0]):
                    sample['depth'][i][sample['gt_mask']] = 0

        return sample

    def box_zero(self, area):
        boxes = np.random.randint(1, 5)

        for _ in np.arange(boxes):
            if area > 0:
                cur_area = int(area * np.random.rand())
                area -= cur_area
                a, b = self.random_box()


    def random_box(self, area, w, h):
        s = np.random.rand()
        if s < 0.2:
            s = 0.2
        elif s > 0.8:
            s = 0.8

        a = int(np.sqrt(area) * s)
        m = max(w, h)
        if a > m:
            a = m
        b = int(area / a)
        return a, b


class DepthToPointCloud_old(object):
    def __init__(self, depth_zeros=0.2, depth_noise=0.005):
        self.depth_zeros = depth_zeros
        self.depth_noise = depth_noise

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'depth_zeros={0}, '.format(self.depth_zeros)
        format_string += 'depth_noise={0})'.format(self.depth_noise)
        return format_string

    def __call__(self, sample):
        w, h = sample['img'].size
        xmap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(w), axis=0), h, axis=0), dtype=torch.float32)
        ymap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(h), axis=1), w, axis=1), dtype=torch.float32)
        cx, cy, fx, fy = sample['intr']
        pt2 = (torch.as_tensor(sample['depth'], dtype=torch.float32) * sample['depth_scale']) / 1000  # mm to meter

        t = torch.rand(1) * self.depth_zeros
        pt2[torch.rand(pt2.shape) < t] = 0  # set some depth points randomly to 0
        depth_nosie = ((torch.rand(pt2.shape) - 0.5) * 2) * self.depth_noise  # adding +- random noise
        pt2[pt2 != 0] += depth_nosie[pt2 != 0]
        pt2[pt2 < 0] = 0
        pt0 = (xmap - cx) * pt2 / fx
        pt1 = (ymap - cy) * pt2 / fy
        sample['pc'] = torch.stack((pt0, pt1, pt2), axis=0)
        return sample

class DepthToPointCloud(object):
    def __init__(self, abs_values=True, center_jitter=0.02, focal_jitter=0.20, jitter=True,
                 p_disable_xy=0.0):
        self.abs_values = abs_values
        self.intr = [0.5, 0.5, 1.5, 1.5]
        self.center_jitter = center_jitter
        self.focal_jitter = focal_jitter
        self.jitter = jitter
        self.p_disable_xy = p_disable_xy

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'abs_values={0}, '.format(self.abs_values)
        format_string += 'intr={0}, '.format(self.intr)
        format_string += 'center_jitter={0}, '.format(self.center_jitter)
        format_string += 'focal_jitter={0})'.format(self.focal_jitter)
        return format_string

    def __call__(self, sample):

        if 'intr' not in sample:
            cx, cy, fx, fy = copy.deepcopy(self.intr)
            w, h = sample['img'].size
            sample['intr'] = [cx*w, cy*h, fx*w, fy*w]# focal with the same h or w
            #print('created intr', sample['intr'])

        intr = sample['intr']
        if not isinstance(intr, list):
            intr = [intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]]

        if sample.get('mode', '') == 'train' and self.jitter:
            #print('before jitter intr', sample['intr'])
            if np.random.rand() < 0.5:
                focal_jitterx = (float(np.random.rand() - 0.5) * 2 * self.focal_jitter)
                focal_jittery = (float(np.random.rand() - 0.5) * 2 * self.focal_jitter)
            else:
                focal_jitterx = (float(np.random.rand() - 0.5) * 2 * self.focal_jitter)
                focal_jittery = focal_jitterx

            intr[0] += intr[0] * (float(np.random.rand() - 0.5) * 2 * self.center_jitter)
            intr[1] += intr[1] * (float(np.random.rand() - 0.5) * 2 * self.center_jitter)
            intr[2] += intr[2] * focal_jitterx
            intr[3] += intr[3] * focal_jittery
            #print('after jitter intr', sample['intr'])

        if np.random.rand() >= self.p_disable_xy:
            if 'gt_depth' in sample:
                sample['gt_depth'] = self.to_pc(sample['gt_depth'], intr)

            if 'depth' in sample:
                sample['depth'] = self.to_pc(sample['depth'], intr)
        else:
            if 'gt_depth' in sample:
                sample['gt_depth'] = self.expand_depth(sample['gt_depth'])

            if 'depth' in sample:
                sample['depth'] = self.expand_depth(sample['depth'])
        return sample
    
    
    def expand_depth(self, depth):
        pt2 = torch.as_tensor(depth, dtype=torch.float32)
        pt0 = pt1 = torch.zeros(depth.shape, dtype=depth.dtype, device=depth.device)
        return torch.stack((pt0, pt1, pt2), axis=0)

    def to_pc(self, depth, intr):
        pt2 = torch.as_tensor(depth, dtype=torch.float32)
        h, w = depth.shape
        cx, cy, fx, fy = intr
        xmap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(w), axis=0), h, axis=0), dtype=torch.float32)
        ymap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(h), axis=1), w, axis=1), dtype=torch.float32)
        pt0 = (xmap - cx) * pt2 / fx
        pt1 = (ymap - cy) * pt2 / fy
        if self.abs_values:
            pt0 = torch.abs(pt0)
            pt1 = torch.abs(pt1)
        return torch.stack((pt0, pt1, pt2), axis=0)

class DepthToHHA(object):
    def __init__(self):
        self.tf = transforms.ToTensor()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'tf={0})'.format(self.tf)
        return format_string


    def __call__(self, sample):
        print('to hha')
        if 'depth' in sample and 'intr' in sample:
            cx, cy, fx, fy = sample['intr']
            h, w = sample['depth'].shape
            print(h, w, cx, cy, fx, fy)

            cam_mat = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape((3, 3))
            if isinstance(sample['depth'], list):
                sample['hha'] = [self.to_hha(d, cam_mat) for d in sample['depth']]
            else:
                sample['hha'] = self.to_hha(sample['depth'], cam_mat)
        print('hha', sample.keys())
        return sample

    def to_hha(self, d, intr):
        d = np.array(d)
        d = getHHA(d, d, intr)
        print('gethha', d.shape)
        #d = self.tf()
        d = np.array(d, dtype=np.uint8)

        #return Image.fromarray(np.array(d, dtype=np.uint8))
        return self.tf(d)





