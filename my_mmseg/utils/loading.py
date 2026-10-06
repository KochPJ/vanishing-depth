# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import cv2
import mmengine.fileio as fileio
import numpy as np
import torch
from PIL import Image

import os
import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Printer(BaseTransform):

    def __init__(
        self,
        name = None,
        
        ) -> None:
        super().__init__()

        self.name = name
        
    def transform(self, results: dict) -> Optional[dict]:
        print('################################')
        if self.name is not None:
            print('##### {} #####'.format(self.name))

        for key, item in results.items():
            if isinstance(item, np.ndarray):
                if key in ['img']:
                    try:
                        print('     {}: {} {} | mean {}, std {}, min {}, max {}'.format(key, type(item), item.shape,
                                                                [np.mean(i) for i in item.transpose((2,0,1))], 
                                                                [np.std(i) for i in item.transpose((2,0,1))], 
                                                                [np.min(i[i>0]) for i in item.transpose((2,0,1))], 
                                                                [np.max(i) for i in item.transpose((2,0,1))]))
                    except:
                        print('     {}: {} {} | mean {}, std {}, max {}'.format(key, type(item), item.shape,
                                                                [np.mean(i) for i in item.transpose((2,0,1))], 
                                                                [np.std(i) for i in item.transpose((2,0,1))], 
                                                                [np.max(i) for i in item.transpose((2,0,1))]))
                                                            
                else:
                    print('     {}: {} {}'.format(key, type(item), item.shape))
                

            elif isinstance(item, torch.Tensor):
                try:
                    print('     {}: {} {} | mean {}, std {}, min {}, max {}'.format(key, type(item), item.shape,
                          [torch.mean(i) for i in item], [torch.std(i) for i in item], [torch.min(i[i>0]) for i in item], [torch.max(i) for i in item]))
                except:
                    print('     {}: {} {} | mean {}, std {}, max {}'.format(key, type(item), item.shape,
                          [torch.mean(i) for i in item], [torch.std(i) for i in item], [torch.max(i) for i in item]))
                
            else:
                print('     {}: {}'.format(key, type(item)))
        print('################################')

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}()')
        return repr_str

def log_message(message, log_file= 'log.txt'):
    with open(log_file , 'a') as f:
        f.write(message + '\n')



@TRANSFORMS.register_module()
class LoadAnnotationsRGBD(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        with_depth = True,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
        *args, **kwargs
        ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        self.with_depth = with_depth
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def _load_depth_map(self, results: dict) -> None:
        """
        Loads the depth map for depth annotations 
        using depth_map_path from results dict

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.

        """
        if self.file_client_args is not None:
            file_client = fileio.FileClient.infer_client(self.file_client_args, results['depth_map_path'])
            img_bytes = file_client.get(results['depth_map_path'])
        
        else:
            img_bytes = fileio.get(results['depth_map_path'], backend_args=self.backend_args)

        depth_map = mmcv.imfrombytes(img_bytes , flag = 'unchanged',
                    backend = self.imdecode_backend).squeeze().astype(np.float32)

        results['gt_depth_map'] = depth_map
        results['depth_fields'].append('gt_depth_map') 

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """
        results = super().transform(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_depth:
            self._load_depth_map(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'with_depth={self.with_depth})'
        repr_str += f'backend_args={self.backend_args})'
        return repr_str



@TRANSFORMS.register_module()
class LoadDepth(BaseTransform):
    """
    Loads the depth map and stacks it onto the img for further transforms
    """
    def __init__(self, depthscale2meter=1, norm_depth=0, div_nr=0, max_depth=1000, rescale_depth=0, depth_mul=None, depth_deleted=None, random_depth=None, depth_offset=None):
        self.depthscale2meter = depthscale2meter
        self.norm_depth = norm_depth
        self.div_nr = div_nr
        self.max_depth = max_depth
        self.rescale_depth = rescale_depth
        self.rescale_p = 0.95
        self.training = False
        self.depth_mul = depth_mul
        self.depth_deleted = depth_deleted
        self.random_depth = random_depth
        self.depth_offset = depth_offset

    def transform(self, results: dict) -> Optional[dict]:
        img = results['img']
        # print('img shape before depth concat', img.shape) # (1024,2048,3) 3 rgb channels
        depth = np.array(Image.open(results['depth_map_path']), dtype=np.float32)
        # print('here loading', np.mean(depth[depth > 0]), np.std(depth[depth > 0]), np.min(depth[depth > 0]), np.max(depth[depth > 0]))
        
        # depth = cv2.imread(results['depth_map_path'], cv2.IMREAD_UNCHANGED).astype(np.float32)
        # print('depth array after loading and before concat loaddepth', depth)
    
        # has_non_zero = np.any(depth!=0)
        # print('depth array inside loaddepth has non zero:',has_non_zero)

        # has_nan = np.any(np.isnan(depth))
        # print('depth has any None element ?', has_nan)

        # print('depth array from Image shape is loaddepth :', depth.shape)
        
        #print('here loading', np.mean(depth[depth > 0]), np.std(depth[depth > 0]), np.min(depth[depth > 0]), np.max(depth[depth > 0]))

        #print('shape size', np.sum(depth > 0) / np.prod(depth.shape))

        if self.depth_deleted is not None:
            m = torch.rand(depth.shape) < self.depth_deleted
            depth[m] = 0       
        
        #print('shape again', np.sum(depth > 0) / np.prod(depth.shape))

        mask = depth > 0
        if self.norm_depth > 0:
            depth[mask] = (depth[mask] - 1) / self.norm_depth
            mask = depth > 0


        if self.div_nr != 0:
            depth[mask] = self.div_nr / depth[mask]
        
        if self.depthscale2meter > 0:
            depth = depth * self.depthscale2meter
            
        if self.depth_mul is not None:
            depth[mask] = depth[mask] * self.depth_mul
            mask = depth > 0
        
        if self.depth_offset is not None:
            #print('berofre', np.mean(depth))
            depth[mask] += self.depth_offset
            mask = depth > 0
            #print('after', np.mean(depth), self.depth_offset)
            #input()
        
        if self.random_depth is not None:
            m = torch.rand(depth.shape) < self.random_depth
            noise = torch.rand(depth.shape) * self.max_depth
            depth[m] = noise[m]
            #print('self.random', self.random_depth, self.max_depth)
        
        #print('shape there', np.sum(depth > 0) / np.prod(depth.shape))


        #print('here begin', np.mean(depth[depth > 0]), np.std(depth[depth > 0]), np.min(depth[depth > 0]), np.max(depth[depth > 0]))
        
        #print('')
        depth[~mask] = 0
        depth[depth < 0] = 0
        if self.max_depth != None:
            depth[depth > self.max_depth] = 0
        
        #print('here mid', np.mean(depth[depth > 0]), np.std(depth[depth > 0]), np.min(depth[depth > 0]), np.max(depth[depth > 0]))
        if len(depth.shape) == 2:
            depth = np.expand_dims(depth, axis=-1)

        # print('depth shape before depth concat loaddepth', depth.shape)
        # max_value = np.max(depth)
        # min_value = np.min(depth)

        # print('max_value is in loaddepth depth after max-depth logic',max_value)
        # print('min_value is in loaddepth depth after max-depth logic',min_value)
        # has_non_zero = np.any(depth!=0)
        # print('depth array inside loaddepth has non zero before cat:',has_non_zero)

        # has_nan = np.any(np.isnan(depth))
        # print('depth has any None element before cat?', has_nan)


        #print('rescale_depth',self.rescale_depth)
        if self.rescale_depth != 0:
            dkey_max = np.max(depth)
            #print('dkey_max', dkey_max)
            if dkey_max > self.rescale_depth*self.rescale_p:
                if self.training:
                    scale = (0.9 + float(np.random.rand() * 0.1))
                else:
                    scale = self.rescale_p
                ds = (1 / dkey_max) * (self.rescale_depth * scale)
                depth = depth * ds 
                   
        if self.max_depth != None:              
            depth[depth > self.max_depth] = 0

        #print('here end', np.mean(depth[depth > 0]), np.std(depth[depth > 0]), np.min(depth[depth > 0]), np.max(depth[depth > 0]))
        
        #input()

        # img = np.concatenate((img.astype(np.float32), depth), axis=2)
        img = np.concatenate((img, depth), axis=2)
        
        
        #print('img shape after depth concat loaddepth', img.shape) # (1024,2048,4) 4 rgbd channels
        results['img'] = img
        # debug
        # print('Type of results in LoadDepth ', type(results))
        # print('Dictionary results in LoadDepth', results)
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}()')
        return repr_str


# mmseg_custom_transforms.py

# Register this module so that dict(type='CourruptDepth', ...) works in cfg pipelines

import numpy as np
from typing import Optional, Dict
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CourruptDepth(BaseTransform):
    """
    Corrupts the depth channel in results['img'] (assumes RGBD with depth in the last channel).
    Corruption modes:

      - missing: set depth to 0 with probability p
      - bad: replace with uniform noise in [0, max_depth] with probability p

      - constant: add a constant offset with probability p
      - noisy: add Gaussian noise N(0, offset) with probability p

    Args:
        missing (float | None): probability of zeroing depth values
        bad (float | None): probability of replacing values with uniform noise in [0, max_depth]
        noisy (float | None): probability of adding Gaussian noise N(0, offset)
        constant (float | None): probability of adding a fixed offset
        offset (float): noise/constant magnitude
        max_depth (float): upper bound for 'bad' uniform noise
    """

    def __init__(
        self,
        missing: Optional[float] = None,
        bad: Optional[float] = None,
        noisy: Optional[float] = None,
        constant: Optional[float] = None,
        offset: float = 0.05,
        max_depth: float = 15.0
    ) -> None:
        super().__init__()
        self.missing = missing
        self.bad = bad
        self.noisy = noisy
        self.constant = constant
        self.offset = float(offset)
        self.max_depth = float(max_depth)

    def transform(self, results: Dict) -> Optional[Dict]:
        # Expect RGBD image in results['img'], last channel is depth
        if 'img' not in results:
            return results

        img = results['img']
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] < 4:
            # Not RGBD or unexpected format; do nothing
            return results

        depth = img[..., -1].astype(np.float32)
        rng = np.random.default_rng()

        # Missing depth: set to zero
        if self.missing is not None and self.missing > 0:
            #print('missing', self.missing)
            mask = rng.random(depth.shape) < self.missing
            depth[mask] = 0.0

        # Bad depth: random uniform in [0, max_depth]
        if self.bad is not None and self.bad > 0:
            #print('bad', self.bad)
            mask = rng.random(depth.shape) < self.bad
            noise = rng.random(depth.shape, dtype=np.float32) * self.max_depth
            depth[mask] = noise[mask]

        # Constant offset
        if self.constant is not None and self.constant > 0:

            #print('constant', self.constant)
            mask = rng.random(depth.shape) < self.constant
            depth[mask] = depth[mask] + self.offset

        # Noisy: Gaussian noise
        if self.noisy is not None and self.noisy > 0:
            
            #print('noisy', self.noisy)
            mask = rng.random(depth.shape) < self.noisy
            noise = rng.normal(0.0, self.offset, size=depth.shape).astype(np.float32)
            depth[mask] = depth[mask] + noise[mask]

        # Clamp negative values to zero
        np.maximum(depth, 0.0, out=depth)

        # Write back
        img[..., -1] = depth
        results['img'] = img
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(missing={self.missing}, bad={self.bad}, '
                f'noisy={self.noisy}, constant={self.constant}, '
                f'offset={self.offset}, max_depth={self.max_depth})')