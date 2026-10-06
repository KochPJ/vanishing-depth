# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
import warnings
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import math
import mmcv
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS
import torch
from mmengine.registry import MODELS
from mmengine.model.base_model.data_preprocessor import ImgDataPreprocessor
Number = Union[int, float]




@TRANSFORMS.register_module()
class NormalizeRGBD(BaseTransform):
    """Normalize the image.

    Required Keys:

    - img

    Modified Keys:

    - img

    Added Keys:

    - img_norm_cfg

      - mean
      - std
      - to_rgb


    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB before
            normlizing the image. If ``to_rgb=True``, the order of mean and std
            should be RGB. If ``to_rgb=False``, the order of mean and std
            should be the same order of the image. Defaults to True.
    """

    def __init__(self,
                 mean: Sequence[Number],
                 std: Sequence[Number],
                 to_rgb: bool = True) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results: dict) -> dict:
        """Function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, key 'img_norm_cfg' key is added in to
            result dict.
        """

        results['img'][:, :, :3] = mmcv.imnormalize(results['img'][:, :, :3],
                                                    self.mean[:3],
                                                    self.std[:3],
                                                    self.to_rgb)
        if len(self.mean) > 3 and 'depth' in results:
            results['depth'] = (results['depth'] - self.mean[3]) / self.std[3]
            
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

    
@TRANSFORMS.register_module()
class DepthPositionalEncoding(BaseTransform):
    def __init__(self, depth_channels=32, temperature=0.0003, scale=math.pi*2, normalize=True, 
                 position_offset=0.0, zero_eps=1e-6, flatten_4th_dim=True, div_factors=None,
                 with_dino_head=False, enable_dino_head_epoch=0, encoder_hidden_dims=768,get_depth_scales=False, 
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
            dim_t = torch.arange(self.depth_channels, dtype=torch.float32)  # dim_t is each frequency bands
            dim_t = (2 * torch.div(dim_t, 2, rounding_mode='trunc')) / self.depth_channels
            self.dim_t = self.temperature ** dim_t  # dim_t // 2


        print('Temperature in DPE:', self.temperature)
        print('temperature scales: {}'.format(self.dim_t.numpy()))

        p_meter = self.scale / self.dim_t.numpy() / (math.pi * 2)  # p_meter is periodicity of each frequency bands
        
        # print('max dist per layer dist [%]: {}'.format([np.round(v, 4) for i, v in enumerate(1/p_meter) if i%2 == 0]))
        # print('max dist per layer dist [m]: {}'.format([np.round(v, 4) for i, v in enumerate(self.max_depth/p_meter) if i%2 == 0]))
        # print('max dist per layer dist [mm]: {}'.format([np.round(v, 4) for i, v in enumerate(self.max_depth*1000/p_meter) if i%2 == 0]))

        # print('max dist of first layer: {}, 1 = {}%, 0.1 = {}%'.format(np.round(1/p_meter[0], 6),
        #                                                              np.round(p_meter[0]*100, 6),
        #                                                           np.round(p_meter[0], 6)))
        # print('max dist of last layer: {}, 1 = {}%, 0.001 = {}%'.format(np.round(1/p_meter[-1], 6),
        #                                                             np.round(p_meter[-1] * 10000, 6),
        #                                                           np.round(p_meter[-1]*1, 6)))

        print('cm dist: {}'.format(self.dim_t.numpy()))
        print('max dist per layer dist: {}'.format(1 / p_meter))

        print('max dist of first layer: {}m, 1m = {}%, 1cm = {}%'.format(np.round(1 / p_meter[0], 4),
                                                                         np.round(p_meter[0] * 100, 4),
                                                                         np.round(p_meter[0], 3)))
        print('max dist of last layer: {}m, 1m = {}%, 1cm = {}%'.format(np.round(1 / p_meter[-1], 4),
                                                                        np.round(p_meter[-1] * 100, 4),
                                                                        np.round(p_meter[-1] * 1, 4)))
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
        format_string += ', with_dino_head={0})'.format(self.with_dino_head)
        return format_string

    def transform(self, results):
        #
        #a, b = sample['gt_depth'].shape
        #print('a: {}'.format(torch.sum(sample['gt_depth'] > 0) / (a * b) * 100))

        #h, w = sample['depth'].shape
        #print('b: {}'.format(torch.sum(sample['depth'] > 0) / (h * w) * 100))
        #print(a,b, h, w)
        #print(sample['gt_depth'].shape, sample['depth'].shape)
        # print('Dictionary keys from Results in DPTEnc transform:')
        # for i, v in results.items():
        #     print(i)
        # print('img shape in transform is DPE', results['img'].shape)
        
        depth = torch.as_tensor(np.array(results['img'][:, :, 3], dtype=np.float32), dtype=torch.float32) 
        results['img'] = results['img'][:, :, :3]
        if self.get_depth_scales :
            results['max_depth_emb'] = torch.zeros(depth.shape).unsqueeze(0)
            s = results['max_depth_emb'].shape
            results['max_depth_emb'] = results['max_depth_emb'].flatten()
        
        #print(self.__repr__())
        #print('in', depth.shape)
        depth, max_depth_emb = self.encode(depth)
        #print('out', depth.shape)
        # results['max_depth_emb'] = torch.zeros(depth.shape).unsqueeze(0)
        # s = results['max_depth_emb'].shape
        # results['max_depth_emb'] = results['max_depth_emb'].flatten()
        # results['img'] = results['img'][:, :, :3]
        # depth, max_depth_emb = self.encode(depth)
        # print('max_depth_emb shape in transform is ', max_depth_emb.shape)
        # print('results[max_depth_emb][:len(max_depth_emb)] shape in transform is ', results['max_depth_emb'][:len(max_depth_emb)].shape)
        # print('img shape in transform is DPE', results['img'].shape)
        # results['max_depth_emb'][:len(max_depth_emb)] = max_depth_emb
        # results['max_depth_emb'] = results['max_depth_emb'].reshape(s)

        if max_depth_emb is not None:
            results['max_depth_emb'][:len(max_depth_emb)] = max_depth_emb
            results['max_depth_emb'] = results['max_depth_emb'].reshape(s)
        
        results['depth'] = depth
        
        # debug
        # print('max_depth_emb inside DepthPosEncTranform',results['max_depth_emb'].shape)
        # has_non_zero = torch.any(results['max_depth_emb']!=0)
        # print('depth array inside loaddepth has non zero:',has_non_zero)

        # has_nan = torch.any(torch.isnan(results['max_depth_emb']))
        # print('depth has any None element ?', has_nan)
        # print('max_depth_enb inside DepthPosEncTranform',results['max_depth_emb'])
        # print('image type in DPE transform', type(results['img']))
        # print('depth type in DPE transform', type(results['depth']))
        # print('depth data inside DepthPosEncTranform',results['depth'].shape)
        # input()
        return results

    def encode(self, depth):
        #if depth.device != self.dim_t.device:
        #    self.dim_t = self.dim_t.to(depth.device)
        # print('here encode', torch.mean(depth[depth > 0]), torch.std(depth[depth > 0]), torch.min(depth[depth > 0]), torch.max(depth[depth > 0]))
        #if depth.device !
        # print('depth dim is', depth.dim())
        # print('norm_with_max_scale in DPE encode', self.norm_with_max_scale)
        d_id = str(depth.get_device())
        if d_id == '-1':
            dim_t = self.dim_t
        else:
            if d_id not in self.dim_t_dict:
                self.dim_t_dict[d_id] = self.dim_t.to(depth.device)
            dim_t = self.dim_t_dict[d_id]

        zero_mask = depth == 0
        if self.zero_eps > 0:
            depth[zero_mask] = self.zero_eps
                
        # depth_scales = torch.zeros(self.encoder_hidden_dims)
        if self.get_depth_scales:
            depth_scales = torch.zeros(self.encoder_hidden_dims)
        else:
            depth_scales = None
        
        if depth.dim() == 2:
            if self.get_depth_scales:
                upper, lower = format(float(torch.max(depth)), '20f').replace(' ', '').split('.')
                embed = torch.tensor([0.1 + float(v)/10 for v in upper+lower]) # map 0->9 to 0.1:1.0. Zeros are 0.1 and missing values are 0.0
                in_index = 2 * self.depth_scales_bin_size + (self.depth_scales_bin_center-len(upper))
                depth_scales[in_index:in_index+len(embed)] = embed
                depth = depth.div(depth.max()) # normalize depth
                
            elif self.norm_with_max_scale:
                depth = depth.div(self.max_depth)
                # print('normalized depth with max depth')
                
            if self.scale > 0:
                depth = depth * self.scale

            depth = depth[:, :, None] / dim_t
            depth = torch.stack((depth[:, :, 0::2].sin(),
                                    depth[:, :, 1::2].cos()), dim=3).flatten(2)

            depth = depth.permute(2, 0, 1)
        elif depth.dim() == 3:
            if self.get_depth_scales:
                for index, d in enumerate(depth):
                    upper, lower = format(float(torch.max(depth)), '20f').replace(' ', '').split('.')
                    embed = torch.tensor([0.1 + float(v)/10 for v in upper+lower]) # map 0->9 to 0.1:1.0. Zeros are 0.1 and missing values are 0.0
                    
                    in_index = index * self.depth_scales_bin_size + (self.depth_scales_bin_center-len(upper))
                    
                    depth_scales[in_index:in_index+len(embed)] = embed
                    depth[index] = d.div(d.max())
            
            elif self.norm_with_max_scale:
                depth = depth.div(self.max_depth)
                

            # print('emb shape', embed.shape)
            # print('depth_scales[in_index:in_index+len(embed)] shape', depth_scales[in_index:in_index+len(embed)].shape)
            if self.scale > 0:
                depth = depth * self.scale

            depth = depth[:, :, :, None] / dim_t
            
            depth = torch.stack((depth[:, :, :, 0::2].sin(),
                                    depth[:, :, :, 1::2].cos()), dim=4).flatten(3)
                    
            depth = depth.permute(0, 3, 1, 2)
            if self.flatten_4th_dim:
                c, dims, h, w = depth.shape
                depth = depth.reshape(dims * c, h, w)
        else:
            raise NotImplementedError('Depth encoding for dim {} is not implemented, got shape'.format(
                depth.dim(), depth.shape))

        # max_value = torch.max(depth)
        # min_value = torch.min(depth)

        # print('max_value is in DPE after calc',max_value)
        # print('min_value is in DPE after calc',min_value)

        # print('depth shape inside encoder DPE', depth.shape)

        
        return depth, depth_scales

    
@TRANSFORMS.register_module()
class DisparityAndRescale(BaseTransform):
    def __init__(self, rescale_factor=75.0, depth2disparty=True, baseline=0.05, focal_length=512, min_depth=0.01, max_depth=75.0,
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

        
    def transform(self, results):
        depth = torch.as_tensor(np.array(results['img'][:, :, 3], dtype=np.float32), dtype=torch.float32) 
        results['depth'] = self.disparty({'depth': depth})['depth'].unsqueeze(0)
        return results


    def disparty(self, sample):

        mask = sample['depth'] <= self.min_depth
        if self.depth2disparty:
            baseline = sample.get('baseline', self.baseline)
            focal_length = sample.get('focal_length', self.focal_length)
            if focal_length is None:
                focal_length = min(sample['depth'].shape) * 0.9
                print('focal_length', focal_length, min(sample['depth'].shape), sample['depth'].shape)

            sample['depth'] = torch.clamp(sample['depth'], min=self.min_depth, max=self.max_depth)            
            sample['depth'] = (baseline * focal_length) / sample['depth']

        if self.clamp_max_depth:
            sample['depth'] = torch.clamp(sample['depth'], max=self.max_depth)

        if self.clamp_min_depth:
            sample['depth'] = torch.clamp(sample['depth'], min=self.min_depth)

        sample['depth'][mask] = 0
        sample['depth'] = sample['depth'] / self.rescale_factor
        return sample




