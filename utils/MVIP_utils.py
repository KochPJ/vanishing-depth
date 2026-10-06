import torchvision.transforms as transforms
import sys
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torch
import random
from utils.perlin_noise import random_binary_perlin_noise
from utils.stuff import gray_scale, Solarization, GaussianBlur


class ColorJitter:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.tf = transforms.ColorJitter(brightness=self.brightness,
                                         contrast=self.contrast,
                                         saturation=self.saturation,
                                         hue=self.hue)

    def __call__(self, sample):
        sample['x'] = [self.tf(x) for x in sample['x']]
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class Identity:
    def __call__(self, sample):
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '()'
        return format_string

class RandomColor:
    def __init__(self, img_key='x', add_identity=False):
        tfs = [gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)]
        self.add_identity = add_identity
        if self.add_identity:
            tfs.append(Identity())
        self.tf = transforms.RandomChoice(tfs)
        self.img_key = img_key
        self.add_identity = add_identity

    def __call__(self, sample):
        if isinstance(sample.get(self.img_key), list):
            sample[self.img_key] = [self.tf(x) for x in sample[self.img_key]]
        else:
            sample[self.img_key] = self.tf(sample[self.img_key])
        
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '(img_key:{}, add_identity: {})'.format(self.img_key,
                                                                                          self.add_identity)
        return format_string



class DepthNoise:
    def __init__(self, max_noise: float = 5.0, to_meter=False, random_offset_range: float = 300,
                 perlin_noise_p: float = 0.5, zero_p: float = 0.0, zero_max: float = 0.2,
                 offset_p: float = 1.0, noise_color_p: float = 0.0, noise_color_max: float = 0.2,
                 p_random_color_noise: float = 0.33, p_random_color_gray: float = 0.33, color_key: str = 'x'):
        if to_meter:
            max_noise = max_noise / 1000
            random_offset_range = random_offset_range / 1000

        self.max_noise = max_noise
        self.random_offset_range = random_offset_range
        self.offset_p = offset_p
        if self.offset_p > 0:
            self.apply_offset = True
        else:
            self.apply_offset = True

        self.perlin_noise_p = perlin_noise_p

        self.zero_max = zero_max
        self.zero_p = zero_p
        if self.zero_p > 0:
            self.apply_zeros = True
        else:
            self.apply_zeros = False

        self.noise_color_p = noise_color_p
        self.noise_color_max = noise_color_max
        if self.noise_color_p > 0:
            self.apply_noise_color = True
        else:
            self.apply_noise_color = False
        self.p_random_color_noise = p_random_color_noise
        self.p_random_color_gray = p_random_color_gray
        self.color_key = color_key

    def __call__(self, sample):
        if 'depth' in sample:
            do_color = True if self.color_key in sample and self.apply_noise_color else False
            if isinstance(sample['depth'], list):
                for i in range(len(sample['depth'])):
                    sample['depth'][i], depth_zeros = self.apply_depth_noise(sample['depth'][i])
                    if do_color and not depth_zeros:
                        sample[self.color_key][i] = self.apply_color_noise(sample[self.color_key][i])
            else:
                sample['depth'], depth_zeros = self.apply_depth_noise(sample['depth'])
                if do_color and not depth_zeros:
                    sample[self.color_key] = self.apply_color_noise(sample[self.color_key])
        return sample

    def apply_depth_noise(self, depth):
        depth = np.array(depth)
        depth_zeros = False
        if self.apply_zeros:
            if np.random.rand() < self.zero_p:
                h, w = depth.shape
                t = float(np.random.rand()) * self.zero_max

                if np.random.rand() < self.perlin_noise_p:
                    mask = random_binary_perlin_noise(shape=(h, w), t=t)
                else:
                    mask = torch.rand((h, w))
                    mask = mask < t

                depth[mask] = 0
                depth_zeros = True

        if self.apply_offset:
            if np.random.rand() < self.offset_p:
                offset = float((np.random.rand() - 0.5) * 2) * self.random_offset_range
                mask = depth > 0
                depth[mask] = depth[mask] + offset
                depth[depth < 0] = 0

        noise = (torch.rand(depth.shape).numpy() * 2) - 1
        noise = np.array(noise * self.max_noise, dtype=np.int32)
        depth[depth > self.max_noise] += noise[depth > self.max_noise]
        depth = Image.fromarray(depth)
        return depth, depth_zeros

    def apply_color_noise(self, color):
        if np.random.rand() < self.noise_color_p:
            w, h = color.size
            color = np.array(color, dtype=np.uint8)
            t = float(np.random.rand()) * self.noise_color_max
            if np.random.rand() < self.perlin_noise_p:
                mask = random_binary_perlin_noise(shape=(h, w), t=t)
            else:
                mask = torch.rand((h, w))
                mask = mask < t

            r = float(np.random.rand())
            if r < self.p_random_color_noise:
                noise = np.array(torch.rand(color.shape).numpy() * 255, dtype=np.uint8)
                color[mask] = noise[mask]
            elif r < self.p_random_color_gray + self.p_random_color_noise:
                color[mask] = int(np.random.randint(100, 150))
            else:
                color[mask] = 0

            color = Image.fromarray(color)
        return color

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'max_noise={0}'.format(self.max_noise)
        format_string += ', random_offset_range={0}'.format(self.random_offset_range)
        format_string += ', offset_p={0}'.format(self.offset_p)
        format_string += ', apply_offset={0}'.format(self.apply_offset)
        format_string += ', perlin_noise_p={0}'.format(self.perlin_noise_p)
        format_string += ', zero_max={0}'.format(self.zero_max)
        format_string += ', zero_p={0}'.format(self.zero_p)
        format_string += ', apply_zeros={0}'.format(self.apply_zeros)
        format_string += ', noise_color_p={0}'.format(self.noise_color_p)
        format_string += ', noise_color_max={0}'.format(self.noise_color_max)
        format_string += ', apply_noise_color={0}'.format(self.apply_noise_color)
        format_string += ', p_random_color_noise={0}'.format(self.p_random_color_noise)
        format_string += ', p_random_color_gray={0}'.format(self.p_random_color_gray)
        format_string += ', color_key={0})'.format(self.color_key)
        return format_string


class RandomFlip:
    def __init__(self, p: float = 0.5, horizontal: bool = True, vertical: bool = True, diagonal: bool = True,
                 uniform: bool = False):
        self.p = {'p': p}
        self.horizontal = horizontal
        self.vertical = vertical
        self.diagonal = diagonal
        self.n = 0
        self.uniform = uniform

        if self.horizontal:
            self.n += p
        self.p['h'] = self.n
        if self.vertical:
            self.n += 1
        self.p['v'] = self.n
        if self.diagonal:
            self.n += 1
        self.p['d'] = self.n

    def __call__(self, sample):

        x = sample.get('x')
        m = sample.get('mask')
        d = sample.get('depth')
        l = None
        for s in [x, m, d]:
            if s is not None:
                l = [None for _ in range(len(s))]
                break
        x = sample.get('x', l)
        m = sample.get('mask', l)
        d = sample.get('depth', l)

        if self.uniform:
            x, m, d = random_flip([x, m, d], self.p, self.n, self.horizontal, self.vertical, self.diagonal)
        else:
            out = [random_flip([x_, m_, d_], self.p, self.n, self.horizontal, self.vertical, self.diagonal)
                   for x_, m_, d_ in zip(x, m, d)]
            x = [o[0] for o in out]
            m = [o[1] for o in out]
            d = [o[2] for o in out]

        if x[0] is not None:
            sample['x'] = x
        if m[0] is not None:
            sample['mask'] = m
        if d[0] is not None:
            sample['depth'] = d

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p={0}'.format(self.p['p'])
        format_string += ', horizontal={0}'.format(self.horizontal)
        format_string += ', vertical={0}'.format(self.vertical)
        format_string += ', diagonal={0}'.format(self.diagonal)
        format_string += ', uniform={0})'.format(self.uniform)
        return format_string


class Normalize:
    def __init__(self, mean=None, std=None, norm_depth=False, depth_mean=1.5, depth_mean_jitter=0.05):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std
        self.norm_depth = norm_depth
        self.depth_mean = depth_mean
        self.depth_mean_jitter = depth_mean_jitter

    def __call__(self, sample):
        for i in range(sample['x'].shape[0]):
            #print(i, sample['x'][i].shape)
            sample['x'][i] = F.normalize(sample['x'][i], self.mean, self.std)

        #if self.depth_mean is not None and self.depth_std is not None and 'depth' in sample:
        #    for i in range(len(sample['depth'])):
        #        sample['depth'][i] = F.normalize(sample['depth'][i].unsqueeze(0), self.depth_mean, self.depth_std)
        #    #print(sample['depth'].shape, 'tensor')
        if 'depth' in sample and self.norm_depth:
            #before = torch.mean(sample['depth'][sample['depth']>0])
            for i in range(len(sample['depth'])):
                if sample.get('mode', '') == 'train':
                    depth_mean = self.depth_mean + \
                                 float((np.random.rand() - 0.5) * 2 * self.depth_mean_jitter) * self.depth_mean
                else:
                    depth_mean = self.depth_mean
                mask = sample['depth'][i] > 0
                sample['depth'][i][mask] += depth_mean - torch.mean(sample['depth'][i][mask])

            #print('after', before, torch.mean(sample['depth'][sample['depth']>0]))
            #sample['depth'][i] = F.normalize(sample['depth'][i].unsqueeze(0), self.depth_mean, self.depth_std)
            # print(sample['depth'].shape, 'tensor')

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'mean={0}'.format(self.mean)
        format_string += ', std={0}'.format(self.std)
        format_string += ', norm_depth={0}, '.format(self.norm_depth)
        format_string += ', depth_mean={0}'.format(self.depth_mean)
        format_string += ', depth_mean_jitter={0})'.format(self.depth_mean_jitter)
        return format_string


class ToTensor:
    def __call__(self, sample):
        sample['x'] = torch.cat([F.to_tensor(x).unsqueeze(0) for x in sample['x']])
        if 'depth' in sample:
            sample['depth'] = torch.cat([torch.as_tensor(np.array(x, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
                                         for x in sample['depth']]).unsqueeze(1)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'

class CatDepth:
    def __call__(self, sample):
        if 'depth' in sample:
            if isinstance(sample['depth'], list):
                sample['depth'] = torch.cat(sample['depth'])
            if sample['depth'].dim() == 3:
                sample['depth'] = sample['depth'].unsqueeze(1)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RoiCrop:
    def __init__(self, roi_inflation: float = 0.1,
                 train_random_zoom_max: float = 0.1, train_random_zoom_min: float = -0.1, zoom_mean: float = 0.0,
                 train_center_jitter: bool = True, train_no_crop: float = 0.0, zoom_std: float = 0.33,
                 p_min: float = 0.1, distribution: str = 'uni'):

        self.roi_inflation = roi_inflation + 1
        self.zoom_min = train_random_zoom_min
        self.zoom_max = train_random_zoom_max
        self.zoom_dist = self.zoom_max - self.zoom_min
        self.p_min = p_min
        self.zoom_mean = zoom_mean
        self.zoom_std = zoom_std
        self.distribution = distribution
        self.min_to_mean_zoom = self.zoom_mean - self.zoom_min
        self.max_to_mean_zoom = self.zoom_max - self.zoom_mean

        self.center_jitter = train_center_jitter
        self.train_no_crop = train_no_crop

    def get_normal_zoom(self):
        a = np.random.normal(0, self.zoom_std)
        if a > 1:
            a = 1
        elif a < -1:
            a = -1

        if a < 0:
            a = self.zoom_mean + (a * self.min_to_mean_zoom)
        else:
            a = self.zoom_mean + (a * self.max_to_mean_zoom)

        return a

    def get_p_zoom(self):
        r = np.random.rand()
        a = np.random.rand()
        if r < self.p_min:
            a = self.zoom_mean - (a * self.min_to_mean_zoom)
        else:
            a = self.zoom_mean + (a * self.max_to_mean_zoom)

        return a

    def get_balanced_zoom(self):
        a = (np.random.rand() - 0.5) * 2
        if a < 0:
            a = self.zoom_mean + (a * self.min_to_mean_zoom)
        else:
            a = self.zoom_mean + (a * self.max_to_mean_zoom)

        return a

    def get_uni_zoom(self):
        a = (np.random.rand() * self.zoom_dist) + self.zoom_min
        return a

    def __call__(self, sample):
        if 'mask' in sample and not sample.get('disable_roi-crop', False):
            for i, mask in enumerate(sample['mask']):
                if sample.get('mode') != 'train' or np.random.rand() > self.train_no_crop:
                    mask = np.array(mask, dtype=np.uint8)
                    crop = get_mask_extrems(mask)
                    wc, hc = get_wh(crop)
                    cx, cy = get_center(crop)
                    m = int(max(wc, hc) * self.roi_inflation)
                    if sample.get('mode') == 'train':
                        if self.distribution == 'uni':
                            a = self.get_uni_zoom()
                        elif self.distribution == 'normal':
                            a = self.get_normal_zoom()
                        elif self.distribution == 'p_min':
                            a = self.get_p_zoom()
                        elif self.distribution == 'balanced':
                            a = self.get_balanced_zoom()
                        else:
                            a = self.get_uni_zoom()
                        a = int(m * a)
                        m += a
                        if self.center_jitter and a > 0:
                            center_offset_x = int((np.random.rand() * a) - (a // 2))
                            center_offset_y = int((np.random.rand() * a) - (a // 2))
                            cx += center_offset_x
                            cy += center_offset_y
                    m = int(m // 2)
                    crop = (cx - m, cy - m, cx + m, cy + m)
                    sample['x'][i] = sample['x'][i].crop(crop)
                    sample['mask'][i] = sample['mask'][i].crop(crop)
                    if 'depth' in sample:
                        sample['depth'][i] = sample['depth'][i].crop(crop)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'roi_inflation={0}, '.format(self.roi_inflation)
        format_string += 'zoom_range=({0}, {1}), '.format(self.zoom_min, self.zoom_max)
        format_string += 'center_jitter={0}, '.format(self.center_jitter)
        format_string += 'train_no_crop={0})'.format(self.train_no_crop)
        return format_string

class Pad:
    def __init__(self, size_divisor: int = 32):
        self.size_divisor = size_divisor

    def __call__(self, sample):
        for i in range(len(sample['x'])):
            w, h = sample['x'][i].size
            w_ = w // self.size_divisor
            h_ = h // self.size_divisor
            skip = True
            if w % self.size_divisor != 0:
                w_ += 1
                skip = False
            if h % self.size_divisor != 0:
                h_ += 1
                skip = False
            if skip:
                continue
            w_ = self.size_divisor * w_
            h_ = self.size_divisor * h_
            w1 = int((w_ - w) // 2)
            h1 = int((h_ - h) // 2)
            x = np.zeros((h_, w_, 3), dtype=np.uint8)
            x[h1:h1 + h, w1:w1 + w] = np.array(sample['x'][i], dtype=np.uint8)
            sample['x'][i] = Image.fromarray(x)
            if 'depth' in sample:
                x = np.zeros((h_, w_), dtype=np.float32)
                x[h1:h1 + h, w1:w1 + w] = sample['depth'][i]
                sample['depth'][i] = Image.fromarray(x)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'size_divisor={0})'.format(self.size_divisor)
        return format_string


class Depth2distance(object):
    def __init__(self, to_meter: bool = True, constant_depth_offset: float = 0.0, depth_mul: float = 1.0):
        self.to_meter = to_meter
        self.constant_depth_offset = constant_depth_offset
        self.depth_mul = depth_mul
        if self.constant_depth_offset != 0:
            self.apply_constant_depth_offset = True
        else:
            self.apply_constant_depth_offset = False
        self.apply_depth_mul = True if depth_mul != 1.0 else False

    def __call__(self, sample):
        if 'depth' not in sample:
            return sample

        if 'depth_scale' in sample:
            #sample['depth'] = [d * sample['depth_scale'] for d in sample['depth']]
            sample['depth'] = sample['depth'] * sample['depth_scale']

        if self.to_meter:
            sample['depth'] = sample['depth'] / 1000 #[d / 1000 for d in sample['depth']]

        if self.apply_depth_mul:
            sample['depth'] = sample['depth'] * self.depth_mul

        if self.apply_constant_depth_offset:
            #print('constant offset', self.constant_depth_offset, self.apply_constant_depth_offset)
            #print(type(sample['depth']), type(sample['depth'][0]))
            #print(torch.mean(sample['depth']))
            #sample['depth'][sample['depth'] > 0] += self.constant_depth_offset
            if self.constant_depth_offset > 0:
                #print('const')
                sample['depth'][sample['depth'] > 0] += self.constant_depth_offset
            else:
                #print('rand')
                if self.constant_depth_offset == -1:
                    sample['depth'] = torch.rand(sample['depth'].shape)
                else:
                    sample['x'] = torch.rand(sample['x'].shape)
            #sample['depth'] =
            # self.constant_depth_offset for d in sample['depth']]
            #print(torch.mean(sample['depth']))

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'constant_depth_offset={0}, '.format(self.constant_depth_offset)
        format_string += 'depth_mul={0}, '.format(self.depth_mul)
        format_string += 'apply_constant_depth_offset={0}, '.format(self.apply_constant_depth_offset)
        format_string += 'apply_depth_mul={0}, '.format(self.apply_depth_mul)
        format_string += 'to_meter={0})'.format(self.to_meter)
        return format_string

class Resize:
    def __init__(self, height: int = 224, width: int = 224, keep_ratio: bool = True, fill: int = 0,
                 padding_mode: str = 'constant', multi_scale_training: bool = True, training_scale_low: float = 0.25,
                 training_scale_high: float = 0.1):
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.height = height
        self.width = width
        self.ratio = height / width
        self.keep_ratio = keep_ratio
        self.fill = fill
        self.padding_mode = padding_mode

        self.multi_scale_training = multi_scale_training
        self.training_scale_high = training_scale_high
        self.training_scale_low = training_scale_low

        if 0 < self.training_scale_low < 0.9 and 0 < self.training_scale_high < 0.9 and self.multi_scale_training:
            self.scale_range = (min(int(self.width * (1 - self.training_scale_low)),
                                    int(self.height * (1 - self.training_scale_low))),
                                max(int(self.width * (1 + self.training_scale_high)),
                                    int(self.height * (1 + self.training_scale_high))))
        else:
            self.scale_range = None

        self.call_counter = 0
        self.set_max = False
        self.img_scale = (self.width, self.height)
        self.init_resize(default=True)

    def init_resize(self, default=False, set_max=False):
        if self.scale_range is not None and not default and self.multi_scale_training:
            if set_max:
                self.img_scale = (self.scale_range[1], self.scale_range[1])
            else:
                r = random.randint(self.scale_range[0], self.scale_range[1])
                self.img_scale = (r, r)
        else:
            self.img_scale = (self.width, self.height)

    def __call__(self, sample):
        if self.scale_range is not None:
            if sample.get('mode') == 'train':
                if sample.get('checking_batch_size'):
                    if not self.set_max:
                        self.init_resize(set_max=True)
                        self.set_max = True
                    self.call_counter += 1
                elif (self.call_counter == sample.get('batch_size') or self.call_counter == 0):
                    self.init_resize()
                    self.call_counter = 1
                    self.set_max = False
                else:
                    self.call_counter += 1
                    self.set_max = False

        size = sample.get('resize_size', self.img_scale) # (self.width, self.height)

        if self.keep_ratio:
            for i in range(len(sample['x'])):
                sample['x'][i] = resize_keep_ratio(sample['x'][i], size[0], size[1], self.fill, self.padding_mode)

                if 'depth' in sample:
                    sample['depth'][i] = resize_keep_ratio(sample['depth'][i],
                                                           size[0], size[1], self.fill, self.padding_mode,
                                                           interpolation=Image.NEAREST)
                if 'mask' in sample:
                    sample['mask'][i] = resize_keep_ratio(sample['mask'][i],
                                                           size[0], size[1], self.fill, self.padding_mode,
                                                           interpolation=Image.NEAREST)
        else:
            for i in range(len(sample['x'])):
                sample['x'][i] = sample['x'][i].resize(size=size)
                if 'depth' in sample:
                    sample['depth'][i] = sample['depth'][i].resize(size=size, resample=Image.NEAREST)
                if 'mask' in sample:
                    sample['mask'][i] = sample['mask'][i].resize(size=size, resample=Image.NEAREST)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'height={0}, '.format(self.height)
        format_string += 'width={0}, '.format(self.width)
        format_string += 'ratio={0}, '.format(self.ratio)
        format_string += 'keep_ratio={0}, '.format(self.keep_ratio)
        format_string += 'fill={0}, '.format(self.fill)
        format_string += 'padding_mode={0})'.format(self.padding_mode)
        return format_string

def bar_progress(mode, epoch, step, steps, lr, loss, acc, old_loss, old_acc, times, topk, rps=None, dps=None):

    progress = (epoch-1) * steps.get('train', 0) + (epoch-1) * steps.get('valid', 0)
    if mode == 'train':
        progress += step
        d_valid_steps = steps.get('valid', 0)
        d_train_steps = steps.get('train', 0) - step
        d_test_steps = steps.get('test', 0)
    elif mode == 'valid':
        progress += steps.get('train', 0) + step
        d_valid_steps = steps.get('valid', 0) - step
        d_train_steps = 0
        d_test_steps = steps.get('test', 0)
    else:
        progress += steps.get('train', 0) + steps.get('valid', 0) + step
        d_valid_steps = 0
        d_train_steps = 0
        d_test_steps = steps.get('test', 0) - step

    progress = progress / (steps.get('epochs', 0) * steps.get('train', 0) +
                           steps.get('epochs', 0) * steps.get('valid', 0) + steps.get('test', 0))

    loss_delta = float(np.round(loss - old_loss, 4)) if old_loss is not None else None
    acc_delta = float(np.round(acc-old_acc, 4)) if old_acc is not None else None

    a = step
    b = steps[mode]
    c = '{} step'.format(mode)

    d_epoch = steps.get('epochs', 0) - epoch

    tepoch = np.mean(times.get('epoch', 0)) if len(times.get('epoch', [])) > 0 else None
    ttrain = np.mean(times.get('train', 0))
    tvalid = np.mean(times.get('valid', 0)) if len(times.get('valid', [])) > 0 else None
    ttest = np.mean(times.get('test', 0)) if len(times.get('test', [])) > 0 else None

    if tvalid is None:
        tvalid = ttrain
    if ttest is None:
        ttest = tvalid
    if tepoch is None:
        tepoch = ttrain * steps.get('train', 0) + tvalid * steps.get('valid', 0)

    eta = d_epoch * tepoch + d_train_steps * ttrain + d_valid_steps * tvalid + ttest * d_test_steps
    if not eta:
        eta = 1
    elif np.isnan(eta):
        eta = 1

    hours = int(eta / 3600)
    if hours > 24:
        days = int(hours / 24)
        hours -= days * 24
        eta -= (days * 24) * 3600
    else:
        days = 0
    eta -= hours * 3600
    minutes = int(eta / 60)
    eta -= minutes * 60
    sec = int(eta)

    eta = ''
    if days > 0:
        eta += str(days) + ':'
    eta += '{}:{}:{}'.format(str(hours) if len(str(hours)) > 2 else str(hours).zfill(2),
                             str(minutes).zfill(2), str(sec).zfill(2))

    if lr is not None:
        if isinstance(lr, list):
            if len(lr) > 3:
                lr = float(np.round(sum([v for v in lr]) / len(lr), 10))
            else:
                lr = [float(np.round(v, 10)) for v in lr]
        else:
            lr = float(np.round(lr, 10))

    if rps is not None:
        if len(rps) > 0:
            rps = float(np.mean(rps))
        else:
            rps = None

    if dps is not None:
        if len(dps) > 0:
            dps = float(np.mean(dps))
        else:
            dps = None

    if dps is not None and rps is not None:
        dps = float(np.round(dps / (rps + dps), 4))
        rps = float(np.round(rps / (rps + dps), 4))

    progress_message = "eta: {} | {}% [{}/{}] ep | {}% [{}/{}] {} | lr: {} | avg.l: {} |" \
                       "loss delta: {} | top {}: {}% | top {} delta: {}% | rps: {} | dps: {}".format(
        eta,
        float(np.round(progress * 100, 4)),
        epoch,
        steps.get('epochs', 0),
        float(np.round(a / b * 100, 2)),
        a,
        b,
        c,
        lr,
        float(np.round(loss, 4)),
        loss_delta,
        topk,
        float(np.round(acc, 4)),
        topk,
        acc_delta,
        rps,
        dps)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()




def resize_keep_ratio(image, wt, ht, fill=0, padding_mode='constant', interpolation=Image.Resampling.BICUBIC):
    if isinstance(image, Image.Image):
        w, h = image.size
    else:
        if len(image.shape) == 2:
            h, w = image.shape
        elif len(image.shape) == 3:
            h, w, c = image.shape

    scale = wt / w
    if h * scale > ht:
        scale = ht / h

    new_w = int(np.round(w * scale, 0))
    new_h = int(np.round(h * scale, 0))

    if isinstance(image, Image.Image):
        image = image.resize((new_w, new_h), interpolation)
    else:
        image = F.resize(image,
                         size=(new_h, new_w),
                         interpolation=interpolation)
    w_padding = (wt - new_w) / 2
    h_padding = (ht - new_h) / 2

    l_pad = w_padding if w_padding % 1 == 0 else w_padding + 0.5
    t_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    r_pad = w_padding if w_padding % 1 == 0 else w_padding - 0.5
    b_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5

    padding = [int(l_pad), int(t_pad), int(r_pad), int(b_pad)]

    image = F.pad(image, padding, fill, padding_mode)

    return image


def get_mask_extrems(mask):
    mask = np.array(mask)
    ys, xs = np.where(mask != 0)
    if len(ys) > 3 and len(xs) > 3:
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    else:
        h, w = mask.shape[:2]
        if w == h:
            return (0, 0, w, h)
        else:
            l = min(h, w)
            h1 = int((h - l) // 2)
            w1 = int((w - l) // 2)
            return (w1, h1, l + w1, l + h1)


def get_center(bbox):
    return (int(bbox[0] + ((bbox[2]-bbox[0])/2)), int(bbox[1] + ((bbox[3]-bbox[1])/2)))


def get_wh(bbox):
    return (bbox[2]-bbox[0], bbox[3]-bbox[1])


def random_flip(x, p: dict, n: int, horizontal: bool, vertical: bool, diagonal: bool):
    if random.random() < p['p']:
        f = random.random()
        if horizontal and f <= p['h'] / n:
            if isinstance(x, list):
                for i, s in enumerate(x):
                    if s is not None:
                        x[i] = F.hflip(s)

                        #x = [F.hflip(s) for s in x if s i not None]
            else:
                if x is not None:
                    x = F.hflip(x)
        elif vertical and f <= p['v'] / n:
            if isinstance(x, list):
                #if x[0] is not None:
                #    x = [F.vflip(s) for s in x]
                for i, s in enumerate(x):
                    if s is not None:
                        x[i] = F.vflip(s)
            else:
                if x is not None:
                    x = F.vflip(x)
        elif diagonal and f <= p['d'] / n:
            if random.random() < 0.5:
                if isinstance(x, list):
                    for i, s in enumerate(x):
                        if s is not None:
                            x[i] = F.hflip(s)
                            x[i] = F.vflip(s)
                        #else:
                        #x = [F.hflip(s) for s in x]
                        #x = [F.vflip(s) for s in x]
                else:
                    if x is not None:
                        x = F.hflip(x)
                        x = F.vflip(x)
            else:
                if isinstance(x, list):
                    #if x[0] is not None:
                    for i, s in enumerate(x):
                        if s is not None:
                            #x = [F.vflip(s) for s in x]
                            #x = [F.hflip(s) for s in x]
                            x[i] = F.vflip(s)
                            x[i] = F.hflip(s)
                else:
                    if x is not None:
                        x = F.vflip(x)
                        x = F.hflip(x)
    return x


def lookup(n):
    """
            Args:
                n (int): number of plots
            returns:
                x, y (int): matplotlib grid x, y
            """
    if n == 1:
        x, y = 1, 1
    elif n == 2:
        x, y = 1, 2
    elif n == 3:
        x, y = 1, 3
    elif n == 4:
        x, y = 2, 2
    elif n in [5, 6]:
        x, y = 2, 3
    elif n in [7, 8]:
        x, y = 2, 4
    elif n == 9:
        x, y = 3, 3
    elif n in [10, 11, 12]:
        x, y = 3, 4
    elif n in [13, 14, 15]:
        x, y = 3, 5
    elif n == 16:
        x, y = 4, 4
    elif n in [17, 18, 19, 20]:
        x, y = 4, 5
    elif n in [21, 22, 23, 24, 25]:
        x, y = 5, 5
    elif n in [26, 27, 28, 29, 30]:
        x, y = 5, 6
    elif n in [31, 32, 33, 34, 35, 36]:
        x, y = 6, 6
    elif n in [37, 38, 39, 40, 41, 42]:
        x, y = 6, 7
    elif n in [43, 44, 45, 46, 47, 48, 49]:
        x, y = 7, 7
    elif n in [50, 51, 52, 53, 54, 55, 56]:
        x, y = 7, 8
    elif n in [57, 58, 59, 60, 61, 62, 63, 64]:
        x, y = 8, 8
    elif n in [65, 66, 67, 68, 69, 70, 71, 72]:
        x, y = 8, 9
    else:
        raise NotImplementedError
    return int(x), int(y)


class TopKAccuracy(object):
    def __init__(self, k: int):
        self.res = []
        self.i = []
        self.k = k

    def add(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k.
        Arguments:
            output (tensor): prediction of a classifier
            target (tensor): labels corresponding to input of classifier
        """
        with torch.no_grad():
            batch_size = target.size(0)

            # for one hot encoding take the maximum argument of the target
            if target.dim() == 2:
                batch_size = target.size(0)
                conf, pred = output.topk(self.k, 1, True, True)
                pred = pred.t()

                # convert one hot to a prediction
                target = torch.argmax(target, dim=1)
            else:
                _, pred = output.topk(self.k, 1, True, True)
                pred = pred.t()

            correct = pred.eq(target[None])

            correct_k = correct[:self.k].flatten().sum(dtype=torch.float32).cpu().numpy()
            self.res.append(correct_k * (100.0 / batch_size))
            self.i.append(batch_size)

    def reset(self):
        """Resets the memory of the metric.
        """
        self.res = []
        self.i = []

    def result(self):
        """Calculates the resulting accuracy.
        Returns:
            float: resulting accuracy
        """
        n_images = np.sum(self.i)
        result = 0
        for i, res in enumerate(self.res):
            result += (self.i[i]/n_images)*res
        return float(result)


class RandomRotation:
    def __init__(self, min_degree=-180,  max_degree=180, uniform: bool = False):
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.uniform = uniform

    def __call__(self, sample):
        if self.uniform:
            angle = random.uniform(self.min_degree, self.max_degree)
            sample['x'] = [F.rotate(x, angle) for x in sample.get('x')]
            if 'depth' in sample:
                sample['depth'] = [F.rotate(x, angle) for x in sample.get('depth')]
            if 'mask' in sample:
                sample['mask'] = [F.rotate(x, angle) for x in sample.get('mask')]
        else:
            for i, x in enumerate(sample['x']):
                angle = random.uniform(self.min_degree, self.max_degree)
                sample['x'][i] = F.rotate(x, angle)
                if 'depth' in sample:
                    sample['depth'][i] = F.rotate(sample['depth'][i], angle)
                if 'mask' in sample:
                    sample['mask'][i] = F.rotate(sample['mask'][i], angle)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'min_degree={0}'.format(self.min_degree)
        format_string += ', max_degree={0}'.format(self.max_degree)
        format_string += ', uniform={0})'.format(self.uniform)
        return format_string

