import copy
import os.path
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union
import json
import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose

from mmseg.registry import DATASETS

@DATASETS.register_module()
class BaseSUNRGBDSegDataset(BaseDataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── depth_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{depth_map_suffix}
        │   │   │   │   ├── yyy{depth_map_suffix}
        │   │   │   │   ├── zzz{depth_map_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of BaseSegDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/en/tutorials/new_dataset.md`` for more details.


    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as
            specify classes to load. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img_path=None, seg_map_path=None).
        img_suffix (str): Suffix of images. Default: '.jpg'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=True``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default to False.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """
    METAINFO: dict = dict()

    def __init__(self,
                 ann_file: str = '',
                 split: str = 'train',
                 img_suffix='.jpg',
                 depth_map_suffix='.png',
                 seg_map_suffix='.png',
                 requires_ann=True,
                 requires_depth=True,
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path='', seg_map_path='', depth_map_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 backend_args: Optional[dict] = None) -> None:

        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.depth_map_suffix = depth_map_suffix
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None
        self.requires_ann = requires_ann
        self.requires_depth = requires_depth

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray
        self.split= split
        self.validation_dirs = self._load_validation_dirs(ann_file) if ann_file else []
         
        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        # Get label map for custom classes
        new_classes = self._metainfo.get('classes', None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label))

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # # Join paths.
        # if self.data_root is not None:
        #     self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'


    @classmethod
    def get_label_map(cls,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:
        """Require label mapping.

        The ``label_map`` is a dictionary, its keys are the old label ids and
        its values are the new label ids, and is used for changing pixel
        labels in load_annotations. If and only if old classes in cls.METAINFO
        is not equal to new classes in self._metainfo and nether of them is not
        None, `label_map` is not None.

        Args:
            new_classes (list, tuple, optional): The new classes name from
                metainfo. Default to None.


        Returns:
            dict, optional: The mapping from old classes in cls.METAINFO to
                new classes in self._metainfo
        """
        old_classes = cls.METAINFO.get('classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):

            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO['classes']):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in METAINFO.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None

    def _update_palette(self) -> list:
        """Update palette after loading metainfo.

        If length of palette is equal to classes, just return the palette.
        If palette is not defined, it will randomly generate a palette.
        If classes is updated by customer, it will return the subset of
        palette.

        Returns:
            Sequence: Palette for current dataset.
        """
        palette = self._metainfo.get('palette', [])
        classes = self._metainfo.get('classes', [])
        # palette does match classes
        if len(palette) == len(classes):
            return palette

        if len(palette) == 0:
            # Get random state before set seed, and restore
            # random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            new_palette = np.random.randint(
                0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.label_map is not None:
            new_palette = []
            # return subset of palette
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError('palette does not match classes '
                             f'as metainfo is {self._metainfo}.')
        return new_palette


    def _load_validation_dirs(self, ann_file: str) -> List[str]:
        '''
        with open(ann_file, 'r') as f:
            data=json.load(f)

        list_dir = data.get('dirs',[])
        print('length of ann file list dir is :',len(list_dir))
        print(self.data_root, list_dir[0])
        normalized_dirs = []
        for val_dir in list_dir:
            dir_split = val_dir.split('/')
            idx = dir_split.index('train')
            normalized_dirs.append('/'.join(dir_split[idx:-1]))
        '''
        import scipy

        split = scipy.io.loadmat(ann_file, squeeze_me=True, struct_as_record=False)
        print(split.keys())
        split_train = split['alltrain']
        split_train = [f.replace('/n/fs/sun3d/data/SUNRGBD', '/train') for f in split_train]
        split_train = [f[:-1] if f[-1] == '/' else f for f in split_train]


        #normalized_dirs = ['/'.join(val_dir.replace('./data/SunRGBD/SUNRGBD/', '').split('/')[:-1]) for val_dir in list_dir] 
        print('split_train', split_train[0])
        #input()

        return split_train

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file."""
        data_list = []
        skipped = 0
        added = 0
        base_dirs = [
            'train/kv1',
            'train/kv2',
            'train/realsense',
            'train/xtion' 
        ]
        # print('data prefix', self.data_prefix)
        # print('reduce zero label', self.reduce_zero_label)
        # input()

        for base_dir in base_dirs:
            full_base_dir = osp.join(self.data_root, base_dir)
            # print(f'checking base dir: {full_base_dir}')

            # List all subdirectories recursively
            all_entries = fileio.list_dir_or_file(
                dir_path=full_base_dir,
                list_dir=True,
                recursive=True,
                backend_args=self.backend_args
            )

            # Initialize paths to None
            img_dir, depth_dir, seg_mask_dir = None, None, None

            # Check each entry
            for entry in all_entries:
                entry_path = osp.join(full_base_dir, entry)
                if not osp.isdir(entry_path):
                    continue
                
                # print('entry_path is: ',entry_path)

                relative_entry_path = osp.relpath(entry_path, self.data_root)

                # print('rel entry_path is: ',relative_entry_path)

                if self.validation_dirs:
                    # print('normalized val dir:', normalized_val_dirs)
                    found = False
                    for vdir in self.validation_dirs:
                        if relative_entry_path in vdir:
                            found = True
                            break
                    
                    if not found and self.split == 'train':
                        skipped += 1
                        continue

                    elif found and self.split != 'train':
                        skipped += 1
                        continue
                        

                    #if self.split != 'train' and relative_entry_path in self.validation_dirs:
                    #    continue

                    

                    #if self.split == 'train' and relative_entry_path in self.validation_dirs:
                    #    print('skipping', relative_entry_path, relative_entry_path in self.validation_dirs)
                    #    skipped += 1
                    #    continue
                    
                    #print('use relative_entry_path', relative_entry_path, relative_entry_path in self.validation_dirs, self.validation_dirs[0])
                    #input()

                # basename = osp.basename(entry_path)
                # print(f'checking entry path: {entry_path}')
                # print(f'basename is {basename}')
                # print(f"expected image path is {self.data_prefix['img_path']}")
                # print(f"expected depth_map path is {self.data_prefix['depth_map_path']}")
                # print(f"expected seg_map_path is {self.data_prefix['seg_map_path']}")

                # if basename == self.data_prefix['img_path']:
                #     img_dir = entry_path
                #     # print(f'found image dir: {img_dir}')
                # elif basename == self.data_prefix['depth_map_path']:
                #     depth_dir = entry_path
                #     # print(f'found depth dir: {depth_dir}')
                # elif basename == self.data_prefix['seg_map_path']:
                #     seg_mask_dir = entry_path
                #     # print(f'found seg_mask_dir: {seg_mask_dir}')

                img_dir = osp.join(entry_path, self.data_prefix['img_path'])
                depth_dir = osp.join(entry_path, self.data_prefix['depth_map_path'])
                seg_mask_dir = osp.join(entry_path, self.data_prefix['seg_map_path'])

                if not all(map(osp.isdir, (img_dir, depth_dir, seg_mask_dir))):
                    continue

                # Ensure the required subdirectories exist
                if img_dir and depth_dir and seg_mask_dir:
                    # print(f'Found valid timestamp dir: {osp.dirname(entry_path)}')

                    # Load the image file
                    img_files = list(fileio.list_dir_or_file(
                        dir_path=img_dir,
                        list_dir=False,
                        suffix=self.img_suffix,
                        recursive=False,
                        backend_args=self.backend_args))

                    # Load the depth file
                    #depth_files = list(fileio.list_dir_or_file(
                    #    dir_path=depth_dir,
                    #    list_dir=False,
                    #    suffix=self.depth_map_suffix,
                    #    recursive=False,
                    #    backend_args=self.backend_args))
                    depth_files = list(os.listdir(depth_dir))
                    if len(depth_files) != 1:
                        raise ValueError('depth files from {} should be 1: {}'.format(depth_dir, depth_files))


                    #print('depth_dir', depth_dir)
                    #print('depth_files', depth_files)
                    #input()

                    # Load the segmentation mask file
                    seg_mask_files = list(fileio.list_dir_or_file(
                        dir_path=seg_mask_dir,
                        list_dir=False,
                        suffix=self.seg_map_suffix,
                        recursive=False,
                        backend_args=self.backend_args))

                    # Ensure that there is at least one file in each subdirectory
                    if img_files and depth_files and seg_mask_files:
                        # print(f'Found data in: {osp.dirname(entry_path)}')

                        data_info = {
                            'img_path': osp.join(img_dir, img_files[0]),
                            'depth_map_path': osp.join(depth_dir, depth_files[0]),
                            'seg_map_path': osp.join(seg_mask_dir, seg_mask_files[0]),
                            'label_map': self.label_map,
                            'reduce_zero_label': self.reduce_zero_label,
                            'seg_fields': [],
                            'depth_fields': []
                        }
                        data_list.append(data_info)
                        added += 1


                    else:
                        print(f'Missing data in: {osp.dirname(entry_path)}')

                    #print('added/skipped = {}/{}'.format(added, skipped), relative_entry_path)
                    # Reset paths for the next set
                    img_dir, depth_dir, seg_mask_dir = None, None, None

        if not data_list:
            print('No data Found')

        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def load_data_list(self) -> List[dict]:
    #     """Load annotation from directory or annotation file."""
    #     data_list = []
    #     base_dirs = [
    #         'train/realsense',
    #         'train/xtion',
    #         'train/kv1',
    #         'train/kv2'
    #     ]
    #     print('data prefix', self.data_prefix)
    #     input()

    #     for base_dir in base_dirs:
    #         full_base_dir = osp.join(self.data_root, base_dir)
    #         print(f'checking base dir: {full_base_dir}')

    #         # List all subdirectories and files recursively
    #         all_entries = fileio.list_dir_or_file(
    #             dir_path=full_base_dir,
    #             list_dir=True,
    #             recursive=True,
    #             backend_args=self.backend_args
    #         )

    #         # Filter out directories that match the expected subfolder names
    #         dir_groups = {} 
    #         for entry in all_entries:
    #             entry_path = osp.join(full_base_dir, entry)
    #             if not osp.isdir(entry_path):
    #                 continue

    #             parent_dir = osp.dirname(entry_path)
    #             if parent_dir not in dir_groups:
    #                 dir_groups[parent_dir] = []
    #             dir_groups[parent_dir].append(entry_path)

    #             for parent_dir, subdirs in dir_groups.items():
    #                 img_dir, depth_dir, seg_mask_dir = None, None, None
    #                 print(f'checking entry path: {entry_path}')


    #                 for subdir in subdirs:
    #                     # Check if the directory name matches any of the expected names
    #                     basename = osp.basename(entry_path)
    #                     print(f'basename is {basename}')
    #                     print(f"expected image path is {self.data_prefix['img_path']}")
    #                     print(f"expected depth_map path is {self.data_prefix['depth_map_path']}")
    #                     print(f"expected seg_map_path is {self.data_prefix['seg_map_path']}")

    #                     if basename == self.data_prefix['img_path']:
    #                         img_dir = entry_path
    #                         print(f'found image dir: {img_dir}')
    #                     elif basename == self.data_prefix['depth_map_path']:
    #                         depth_dir = entry_path
    #                         print(f'found depth dir: {depth_dir}')
    #                     elif basename == self.data_prefix['seg_map_path']:
    #                         seg_mask_dir = entry_path
    #                         print(f'found seg_mask_dir: {seg_mask_dir}')

    #                 # Ensure the required subdirectories exist
    #                 if img_dir and depth_dir and seg_mask_dir:
    #                     print(f'Found valid timestamp dir: {osp.dirname(entry_path)}')

    #                     # Load the image file
    #                     img_files = list(fileio.list_dir_or_file(
    #                         dir_path=img_dir,
    #                         list_dir=False,
    #                         suffix=self.img_suffix,
    #                         recursive=False,
    #                         backend_args=self.backend_args))

    #                     # Load the depth file
    #                     depth_files = list(fileio.list_dir_or_file(
    #                         dir_path=depth_dir,
    #                         list_dir=False,
    #                         suffix=self.depth_map_suffix,
    #                         recursive=False,
    #                         backend_args=self.backend_args))

    #                     # Load the segmentation mask file
    #                     seg_mask_files = list(fileio.list_dir_or_file(
    #                         dir_path=seg_mask_dir,
    #                         list_dir=False,
    #                         suffix=self.seg_map_suffix,
    #                         recursive=False,
    #                         backend_args=self.backend_args))

    #                     # Ensure that there is at least one file in each subdirectory
    #                     if img_files and depth_files and seg_mask_files:
    #                         print(f'Found data in: {osp.dirname(entry_path)}')

    #                         data_info = {
    #                             'img_path': osp.join(img_dir, img_files[0]),
    #                             'depth_map_path': osp.join(depth_dir, depth_files[0]),
    #                             'seg_map_path': osp.join(seg_mask_dir, seg_mask_files[0]),
    #                             'label_map': self.label_map,
    #                             'reduce_zero_label': self.reduce_zero_label,
    #                             'seg_fields': [],
    #                             'depth_fields': []
    #                         }
    #                         data_list.append(data_info)
    #                     else:
    #                         print(f'Missing data in: {osp.dirname(entry_path)}')

    #     if not data_list:
    #         print('No data Found')

    #     data_list = sorted(data_list, key=lambda x: x['img_path'])
    #     return data_list





"""
    def load_data_list(self) -> List[dict]:
        
        data_list = []
        base_dirs = [
            'train/realsense',
            'train/xtion',
            'train/kv1',
            'train/kv2'
        ] 
        print('data prefix', self.data_prefix)
        input()
        for base_dir in base_dirs:
            
            full_base_dir = osp.join(self.data_root, base_dir)
            print(f'checking base dir:{full_base_dir}')

            for folder in fileio.list_dir_or_file(
                    dir_path=full_base_dir,
                    list_dir=True,
                    recursive=True,
                    backend_args=self.backend_args):
                
                timestamp_path = osp.join(full_base_dir, folder)
                
                if not (osp.isdir(timestamp_path)): 
                    continue

                print(f'checking timestamp dir:{timestamp_path}')

                img_dir, depth_dir, seg_mask_dir = None, None, None

                for subfolder in fileio.list_dir_or_file(
                    dir_path = timestamp_path,
                    list_dir = True,
                    recursive=True,
                    backend_args = self.backend_args):

                    subfolder_path = osp.join(timestamp_path, subfolder)

                    if osp.isdir(subfolder_path):
                        basename = osp.basename(subfolder_path)
                        print(f'checking subfolder path:{subfolder_path}')
                        print(f'basename is {basename}')
                        print(f"expected image path is {self.data_prefix['img_path']}")
                        print(f"expected depth_map path is {self.data_prefix['depth_map_path']}")
                        print(f"expected seg_map_path is {self.data_prefix['seg_map_path']}")
                        if basename == self.data_prefix['img_path']:
                            img_dir = subfolder_path
                            print(f'found image dir : {img_dir}')
                        elif basename == self.data_prefix['depth_map_path']:
                            depth_dir = subfolder_path
                            print(f'found depth dir : {depth_dir}')
                        elif basename == self.data_prefix['seg_map_path']:
                            seg_mask_dir = subfolder_path
                            print(f'found seg_mask_dir : {seg_mask_dir}')

                print(f'Image dir :{img_dir}')
                print(f'Depth dir :{depth_dir}')
                print(f'seg_mask dir :{seg_mask_dir}')

                # Ensure the required subdirectories exist
                if not (img_dir and depth_dir and seg_mask_dir):
                    print(f'Skipping invalid dir:{timestamp_path}')
                    continue

                # Load the image file
                img_files = list(fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=False,
                    backend_args=self.backend_args))
                
                # Load the depth file
                depth_files = list(fileio.list_dir_or_file(
                    dir_path=depth_dir,
                    list_dir=False,
                    suffix=self.depth_map_suffix,
                    recursive=False,
                    backend_args=self.backend_args))

                # Load the segmentation mask file
                seg_mask_files = list(fileio.list_dir_or_file(
                    dir_path=seg_mask_dir,
                    list_dir=False,
                    suffix=self.seg_map_suffix,
                    recursive=False,
                    backend_args=self.backend_args))

                # one of each file in the subdirectories
                if img_files and depth_files and seg_mask_files:
                    print(f'Found data in :{timestamp_path}')

                    data_info = {
                        'img_path': osp.join(img_dir, img_files[0]),
                        'depth_map_path': osp.join(depth_dir, depth_files[0]),
                        'seg_map_path': osp.join(seg_mask_dir, seg_mask_files[0]),
                        'label_map': self.label_map,
                        'reduce_zero_label': self.reduce_zero_label,
                        'seg_fields': [],
                        'depth_fields': []
                    }
                    data_list.append(data_info)
                else:
                    print(f'Missing data in:{timestamp_path}')
        if not data_list :
            print('No data Found')

        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list

"""