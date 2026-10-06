
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms import RandomCrop
from torchvision.transforms import InterpolationMode
import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.image.geometric import _scale_size
import torch
import torchvision.transforms.functional as F
import cv2
#from mmcv.transforms.processing.utils import cache_randomness


@TRANSFORMS.register_module()
class ClampLabels(BaseTransform):
    def __init__(self, max_label=18):
        self.max_label = max_label

    def transform(self, results):
        if 'gt_semantic_seg' in results:
            print('unique labels:',np.unique(results['gt_semantic_seg'], return_counts=True))
            results['gt_semantic_seg'] = np.clip(results['gt_semantic_seg'], 0, self.max_label)
            
        #else:
            #print("Key 'gt_semantic_seg' not found in results")
        return results
            

@TRANSFORMS.register_module()
class ResizeRGBD(MMCV_Resize):
    """Resize images & seg & depth map.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Seg map, depth map and other relative annotations are
    then resized with the same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_seg_map (optional)
    - gt_depth_map (optional)

    Modified Keys:

    - img
    - gt_seg_map
    - gt_depth_map

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self, scale= None, scale_factor= None, keep_ratio=False, clip_object_border= True, backend='cv2'):
        super().__init__(scale, scale_factor, keep_ratio, clip_object_border, backend)
        self.rgb_interpolation = 'bilinear'
        self.depth_interpolation= 'nearest'

    def _resize_img(self, results: dict) -> None:
        # print('backend in resizergbd is :',self.backend)
        if results.get('img', None) is not None:
            img = results['img']
            
            if img.shape[2] > 3:
                rgb_channels = img[:,:,:3]
                depth_channels = img[:,:,3:]
                has_depth = True
            
            # rgb_only
            else:
                rgb_channels = img
                depth_channels = None
                has_depth = False
            

            if self.keep_ratio :
                rgb_resized , scale_factor = mmcv.imrescale(
                    rgb_channels,
                    results['scale'],
                    return_scale = True,
                    interpolation = self.rgb_interpolation,
                    backend = self.backend
                )
                if has_depth:
                    depth_resized = mmcv.imrescale(
                        depth_channels,
                        results['scale'],
                        interpolation = self.depth_interpolation,
                        backend = self.backend

                    )

                new_h, new_w = rgb_resized.shape[:2]
                h, w = rgb_channels.shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h

            else:
                rgb_resized , w_scale, h_scale = mmcv.imresize(
                    rgb_channels,
                    results['scale'],
                    return_scale=True,
                    interpolation = self.rgb_interpolation,
                    backend = self.backend
                )
                if has_depth:
                    depth_resized = mmcv.imresize(
                        depth_channels,
                        results['scale'],
                        interpolation = self.depth_interpolation,
                        backend= self.backend
                    )
            # print('resized image shape in _resize_img', rgb_resized.shape)
            # print('resized depth shape in _resize_img', depth_resized.shape)
            if has_depth:
                results['img'] = np.dstack((rgb_resized, depth_resized))
            else:
                results['img'] = rgb_resized
            
            results['img_shape'] = results['img'].shape[:2]
            # results['scale_factor']= np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            results['scale_factor']= np.array([w_scale, h_scale] , dtype=np.float32)
            results['keep_ratio'] = self.keep_ratio


    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        for seg_key in results.get('seg_fields', []):
            if results.get(seg_key, None) is not None:
                # print('gt_seg_map type in ResizeRGBDPad ###########',type(results[seg_key]))
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[seg_key],
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                else:
                    gt_seg = mmcv.imresize(
                        results[seg_key],
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                results[seg_key] = gt_seg
    
    def _resize_depth(self, results: dict) -> None:

        if 'gt_depth_map' in results:
            if self.keep_ratio:
                gt_depth = mmcv.imrescale(
                    results['gt_depth_map'],
                    results['scale'],
                    interpolation = self.depth_interpolation,
                    backend = self.backend  
                )
            else:
                gt_depth = mmcv.imresize(
                    results['gt_depth_map'],
                    results['scale'],
                    interpolation = self.depth_interpolation,
                    backend = self.backend  
                )
            results['gt_depth_map'] = gt_depth 
            # print('resized gt_depth_map shape', results['gt_depth_map'].shape)
        # Handling depth_fields
        for depth_key in results.get('depth_fields',[]):
            if results.get(depth_key) is not None:
                if self.keep_ratio :
                    resized_depth = mmcv.imrescale(
                        results[depth_key],
                        results['scale'],
                        interpolation = self.depth_interpolation,
                        backend = self.backend
                    )
                else:
                    resized_depth = mmcv.imresize(
                        results[depth_key],
                        results['scale'],
                        interpolation = self.depth_interpolation,
                        backend = self.backend   
                    )

                # print('resized depth_fields shape', resized_depth.shape)
                results[depth_key] = resized_depth 

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1],
                                           self.scale_factor)  # type: ignore
        self._resize_img(results)
        self._resize_seg(results)
        # self._resize_depth(results)
        
        return results

'''
# todo implement this for rgbd, now it is still in rgb mode 
class RandomResizeRGBD(BaseTransform):
    """Random resize images & bbox & keypoints.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if ``scale`` is a sequence of tuple

    .. math::
        target\\_scale[0] \\sim Uniform([scale[0][0], scale[1][0]])
    .. math::
        target\\_scale[1] \\sim Uniform([scale[0][1], scale[1][1]])

    Following the resize order of weight and height in cv2, ``scale[i][0]``
    is for width, and ``scale[i][1]`` is for height.

    - if ``scale`` is a tuple

    .. math::
        target\\_scale[0] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[0]
    .. math::
        target\\_scale[0] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[1]

    Following the resize order of weight and height in cv2, ``ratio_range[0]``
    is for width, and ``ratio_range[1]`` is for height.

    - if ``keep_ratio`` is True, the minimum value of ``target_scale`` will be
      used to set the shorter side and the maximum value will be used to
      set the longer side.

    - if ``keep_ratio`` is False, the value of ``target_scale`` will be used to
      reisze the width and height accordingly.

    Required Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (tuple or Sequence[tuple]): Images scales for resizing.
            Defaults to None.
        ratio_range (tuple[float], optional): (min_ratio, max_ratio).
            Defaults to None.
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.

    Note:
        By defaults, the ``resize_type`` is "Resize", if it's not overwritten
        by your registry, it indicates the :class:`mmcv.Resize`. And therefore,
        ``resize_kwargs`` accepts any keyword arguments of it, like
        ``keep_ratio``, ``interpolation`` and so on.

        If you want to use your custom resize class, the class should accept
        ``scale`` argument and have ``scale`` attribution which determines the
        resize shape.
    """

    def __init__(
        self,
        scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
        ratio_range: Tuple[float, float] = None,
        resize_type: str = 'Resize',
        **resize_kwargs,
    ) -> None:

        self.scale = scale
        self.ratio_range = ratio_range

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Reisize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})

    @staticmethod
    def _random_sample(scales: Sequence[Tuple[int, int]]) -> tuple:
        """Private function to randomly sample a scale from a list of tuples.

        Args:
            scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in scales, which specify the lower
                and upper bound of image scales.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        assert mmengine.is_list_of(scales, tuple) and len(scales) == 2
        scale_0 = [scales[0][0], scales[1][0]]
        scale_1 = [scales[0][1], scales[1][1]]
        edge_0 = np.random.randint(min(scale_0), max(scale_0) + 1)
        edge_1 = np.random.randint(min(scale_1), max(scale_1) + 1)
        scale = (edge_0, edge_1)
        return scale

    @staticmethod
    def _random_sample_ratio(scale: tuple, ratio_range: Tuple[float,
                                                              float]) -> tuple:
        """Private function to randomly sample a scale from a tuple.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``scale`` to
        generate sampled scale.

        Args:
            scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``scale``.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        assert isinstance(scale, tuple) and len(scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(scale[0] * ratio), int(scale[1] * ratio)
        return scale

    @cache_randomness
    def _random_scale(self) -> tuple:
        """Private function to randomly sample an scale according to the type
        of ``scale``.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        if mmengine.is_tuple_of(self.scale, int):
            assert self.ratio_range is not None and len(self.ratio_range) == 2
            scale = self._random_sample_ratio(
                self.scale,  # type: ignore
                self.ratio_range)
        elif mmengine.is_seq_of(self.scale, tuple):
            scale = self._random_sample(self.scale)  # type: ignore
        else:
            raise NotImplementedError('Do not support sampling function '
                                      f'for "{self.scale}"')

        return scale

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, ``img``, ``gt_bboxes``, ``gt_semantic_seg``,
            ``gt_keypoints``, ``scale``, ``scale_factor``, ``img_shape``, and
            ``keep_ratio`` keys are updated in result dict.
        """
        results['scale'] = self._random_scale()
        self.resize.scale = results['scale']
        results = self.resize(results)
        return results


    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'resize_cfg={self.resize_cfg})'
        return repr_str
'''

@TRANSFORMS.register_module()
class RandomCropRGBD(RandomCrop):
    def __init__(self, crop_size: Union[int, Tuple[int, int]],
                cat_max_ratio: float = 1,
                ignore_index: int = 255):
        super().__init__(crop_size, cat_max_ratio, ignore_index)

    def crop_bbox(self, results: dict) -> tuple:
        return super().crop_bbox(results)

    def crop(self, img: np.ndarray, crop_bbox:tuple) -> np.ndarray:

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        
        # print('random crop img shape', results['img'].shape)
        
        if 'gt_depth_map' in results:
            results['gt_depth_map'] = self.crop(results['gt_depth_map'], crop_bbox ) 

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@TRANSFORMS.register_module()
class RandomFlipRGBD(MMCV_RandomFlip):
    def __init__(self, prob = None, direction = 'horizontal', swap_seg_labels= None):
        super().__init__(prob , direction)
        self.swap_seg_labels = swap_seg_labels

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes and semantic segmentation map."""
        # flip image
        # print('random flip img shape', results['img'].shape)
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])
        
        img_shape = results['img'].shape[:2]

        # flip seg map
        for key in results.get('seg_fields', []):
            if results.get(key, None) is not None:
                results[key] = self._flip_seg_map(
                    results[key], direction=results['flip_direction']).copy()
                results['swap_seg_labels'] = self.swap_seg_labels

        for key in results.get('depth_fields',[]):
            if results.get(key, None) is not None:
                results[key] = mmcv.imflip(
                    results[key], direction = results['flip_direction']).copy()

    def transform(self, results):
        return super().transform(results)


@TRANSFORMS.register_module()
class RGBDResizePad(BaseTransform):
    def __init__(self, div_factor=14, interpolation=InterpolationMode.BICUBIC):
        self.div_factor = div_factor
        self.interpolation = interpolation

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'div_factor={0}, '.format(self.div_factor)
        format_string += 'interpolation={0})'.format(self.interpolation)
        return format_string

    
    def transform(self, results: dict) -> dict:
        """Function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, key 'img_norm_cfg' key is added in to
            result dict.
        """
        h, w, c = results['img'].shape
        # print('image height inside the transform before modulo div factor', h)
        # print('image width inside the transform before modulo div factor', w)
        # print('results img shape in RGBDResize', results['img'].shape)
        # print('results[:, :, :3] shape in RGBDResize', results['img'][:, :, :3].shape)
        # input()
        # print('results depth shape in RGBDResize', results['depth'].shape)
        # print('results max_depth_emb shape in RGBDResize', results['max_depth_emb'].shape)
        # print('results depth type', type(results['depth']))
        # print('results max_depth_emb type', type(results['max_depth_emb']))
        w_issue = w%self.div_factor > 0
        h_issue = h%self.div_factor > 0

        # print('before', results['img'].shape)
        
        #print('results pad1', results.keys())

        if w_issue or h_issue:
            if w_issue:
                w_ = (w // self.div_factor + 1) * self.div_factor
            else:
                w_ = w
            
            if h_issue:
                h_ = (h // self.div_factor + 1) * self.div_factor
            else:
                h_ = h
        
            #print('issue w or h', w_issue, h_issue, c)
            #print('results pad', results['img'].shape)
        
            img = torch.from_numpy(results['img'][:, :, :3].copy()).permute((2,0,1))
            #print('?', img.shape)
            img = F.resize(img, (h_, w_), self.interpolation)
            #print('???', img.shape)

            if 'max_depth_emb' in results:
                max_depth_emb = results['max_depth_emb']
                results['max_depth_emb'] = F.resize(max_depth_emb, (h_, w_), InterpolationMode.NEAREST)
                # print('results img shape in RGBDResize after max-depth concat', img.shape)
                # img = torch.cat([img, max_depth_emb])

            #print('results', results.keys())
            if 'depth' in results:
                depth = results['depth']
                #print(depth.shape, h_, w_)
                results['depth'] = F.resize(depth, (h_, w_), InterpolationMode.NEAREST)
                #print('res depth', results['depth'].shape)
                # img = torch.cat([img, depth])
                # print('depth shape inside RGBDResizepad', results['depth'].shape)
            
            # print('results img shape in RGBDResize after F.pad', img.shape)
            
            if c > 3:
                depth = torch.from_numpy(results['img'][:, :, 3:].copy()).permute((2,0,1))
                depth = F.resize(depth, (h_, w_), InterpolationMode.NEAREST)
                #print('depth', depth.shape)
                #img = torch.cat([img, depth])
                #print('img', img.shape)
                results['img'] = results['img'][:, :, :3].copy()
                results['depth'] = depth

            # for key in results.get('seg_fields', []):
            #     if results.get(key, None) is not None:
            #         print('gt_seg_map type in RGBDResizePad',type(results[key]))
            #         print('gt_seg_map shape in RGBDResizePad before interpolation',results[key].shape)
            #         gt_seg_map = results[key].copy()
            #         gt_seg_map_resized =  cv2.resize(gt_seg_map, (w_,h_), interpolation= cv2.INTER_NEAREST)
            #         results[key] = gt_seg_map_resized
            #         print('gt_seg_map shape in RGBDResizePad after interpolation',gt_seg_map_resized.shape)
            
            
            results['img'] = img.permute((1, 2, 0)).numpy()
        else:
            #print('no w or h issue')
            #print('results pad', results['img'].shape)
            # unpack depth from img
            if 'depth' not in results:
                results['depth'] = torch.from_numpy(results['img'][:, :, 3:].copy()).permute((2,0,1))
            results['img'] = results['img'][:, :, :3].copy()
            #print('results', results['img'].shape, results['depth'].shape)

        
        #print('results pad', results.keys())
        
        # print('after', results['img'].shape)

        
        return results
