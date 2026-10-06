# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class PackRGBDSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        # print('results img shape', results['img'].shape)
        # print('image type inside the Packsegrgbd', type(results['img']))
        # print('results[:, :, :3] shape', results['img'][:, :, :3].shape)
        # print('results depth shape', results['depth'].shape)
        # print('results max_depth_emb shape', results['max_depth_emb'].shape)
        #print('results pack seg', results.keys())
        if 'img' in results:
            img = results['img'][:, :, :3]
            # depth = results['img'][:,:,3:]  # debug for viz using only loaddepth
            # print('depth shape in packrgbd for Viz', depth.shape)
            
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                #print('here')
                #print('1', np.mean(img), np.std(img))
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
                #print('2', torch.mean(img), torch.mean(img))
            
            # # debug for viz using only loaddepth
            # img = to_tensor(img)
            # print(' img shape after transpose', img.shape)
            # depth= to_tensor(depth).permute(2,0,1)
            # img = torch.cat([img, depth]) # debug for viz using only loaddepth
            # print('image shape in packrgbd before max_depth_emb concat', img.shape)
            # print('max_depth_emb shape in packrgbd before max_depth_emb concat', results['max_depth_emb'].shape)
                        
            if 'max_depth_emb' in results:
                img = torch.cat([img, results['max_depth_emb']])
        
            #print('Results[max_depth_emb] shape is:', results['max_depth_emb'].shape)
            # print('image shape in packrgbd before depth concat', img.shape)
            
            if 'depth' in results:
                # depth = results['depth']
                # if depth.is_contiguous():
                #     print('depth tensor is contiguous')
                # else:
                #     print('depth is not contiguous')
                #print(type(img), img.shape, type(results['depth']), results['depth'].shape)
                img = torch.cat([img, results['depth']])
            
            # print('image shape in packrgbd after depth concat', img.shape)
           

            packed_results['inputs'] = img
        
        # debug
        # print('Results[depth] shape in Packrgbd', results['depth'].shape)
        # print('Results[img] shape in Packrgbd', results['img'].shape)
        # print('packed results input shape in packrgbd', packed_results['inputs'].shape)
        # input()
        
        data_sample = SegDataSample()
        for seg_key in results.get('seg_fields',['gt_seg_map']):
            if seg_key in results:
                if len(results[seg_key].shape) == 2:
                    data = to_tensor(results[seg_key][None,
                                                        ...].astype(np.int64))
                else:
                    warnings.warn('Please pay attention your ground truth '
                                'segmentation map, usually the segmentation '
                                'map is 2D, but got '
                                f'{results[seg_key].shape}')
                    data = to_tensor(results[seg_key].astype(np.int64))
                gt_sem_seg_data = dict(data=data)
                data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results and results['gt_edge_map'] is not None:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_depth_map' in results and results['gt_depth_map'] is not None:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str