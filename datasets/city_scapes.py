import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import namedtuple

# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('pole group', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]
seg_labels = ['road', 'sidewalk', 'parking', 'rail track',	'person', 'rider', 'car', 'truck',
              'bus', 'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence',
              'guard rail', 'bridge', 'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'vegetation',
              'terrain', 'sky', 'ground', 'dynamic', 'static']


class CityScapesDataset(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True, mode_tag=None,
                 with_mask=False, load_from_checkpoint=True, with_pc=False, mode=None, p_eval=500,
                 depth_mul=10, copy_seg_ds=False, out_root=None, num_samples=None):
        if root is None:
            root = './data/cityscapes'
        self.root = root
        self.transforms = transforms
        self.with_intr = with_intr
        self.with_depth = with_depth
        self.with_mask = with_mask
        self.with_pc = with_pc
        self.depth_mul = depth_mul
        self.label = labels
        self.names = {}
        self.cls_color_ids = {}
        self.cls_names = ['background']
        self.class_names = []
        self.mode = mode
        id_cls = 1
        for l in labels:
            if not l.ignoreInEval:
                self.names[l.id] = {'name': l.name, 'id': id_cls}
                self.cls_color_ids[sum(l.color)] = {'name': l.name, 'id': id_cls}
                self.cls_names.append(l.name)
                self.class_names.append(l.name)
                id_cls += 1
            else:
                pass #print(l.name)
        #print(self.seg_labels)
        self.num_classes = len(self.names)
        uni, counts = np.unique(list(self.cls_color_ids.keys()), return_counts=True)
        assert max(counts) == 1
        #print(len(uni), uni, counts)
        #input()

        if mode_tag is not None:
            mode = mode + '_' + mode_tag 
        if mode is not None:
            file_name = 'cityscapes_{}_dataset_config.json'.format(mode)
        else:
            file_name = 'cityscapes_dataset_config.json'

        load_path = os.path.join(self.root, file_name)
        print('path inside the city_scapes.py is',load_path)
        cp = None
        if load_from_checkpoint:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    cp = json.load(f)
        ids = []
        self.dirs = []
        test_dirs = []
        val_dirs = []
        if cp is None:
            if mode_tag == 'segmentation':

                for path, folders, files in os.walk(os.path.join(self.root, 'gtFine')):
                    # print('path inside the city_scapes.py is',path)
                    # input()
                    if len(folders) == 0:
                        d = path.split('/')
                        city, t_mode = d[-1], d[-2]
                        for f in files:
                            if '.png' not in f:
                                continue
                            mask_path = os.path.join(path, f)
                            i_id = f.split('_')
                            depth_path = os.path.join(self.root,
                                                      'disparity_sequence',
                                                      t_mode, city,
                                                      '{}_{}_{}_disparity.png'.format(i_id[0], i_id[1], i_id[2]))
                            rgb_path = os.path.join(self.root,
                                                    'leftImg8bit',
                                                    t_mode, city,
                                                    '{}_{}_{}_leftImg8bit.png'.format(i_id[0], i_id[1], i_id[2]))
                            if t_mode == 'train':
                                self.dirs.append([rgb_path, depth_path, mask_path])
                            elif t_mode == 'val':
                                val_dirs.append([rgb_path, depth_path, mask_path])
                            elif t_mode == 'test':
                                test_dirs.append([rgb_path, depth_path, mask_path])

                            if copy_seg_ds:
                                import shutil
                                for sample in [rgb_path, depth_path, mask_path]:
                                    out_src = sample.replace(self.root, out_root)
                                    new_folder = '/'.join(out_src.split('/')[:-1])
                                    if not os.path.exists(new_folder):
                                        os.makedirs(new_folder)
                                    shutil.copy(sample, out_src)

                            # print(os.path.exists(rgb_path), os.path.exists(depth_path), os.path.exists(mask_path))
            else:
                for path, folders, files in os.walk(os.path.join(self.root, 'leftImg8bit')):
                    if len(folders) == 0:
                        d = path.split('/')
                        city, mode = d[-1], d[-2]
                        for f in files:
                            rgb_path = os.path.join(path, f)
                            i_id = f.split('_')
                            depth_path = os.path.join(self.root,
                                                      'disparity_sequence',
                                                      mode, city,
                                                      '{}_{}_{}_disparity.png'.format(i_id[0], i_id[1], i_id[2]))

                            if not os.path.exists(depth_path):
                                continue

                            if '/test/' in rgb_path:
                                test_dirs.append([rgb_path, depth_path])
                            else:
                                self.dirs.append([rgb_path, depth_path])

            if mode is not None:
                if mode_tag is not None:
                    train_name = 'cityscapes_{}_{}_dataset_config.json'.format('train', mode_tag)
                    valid_name = 'cityscapes_{}_{}_dataset_config.json'.format('valid', mode_tag)
                    test_name = 'cityscapes_{}_{}_dataset_config.json'.format('test', mode_tag)
                else:
                    train_name = 'cityscapes_{}_dataset_config.json'.format('train')
                    valid_name = 'cityscapes_{}_dataset_config.json'.format('valid')
                    test_name = 'cityscapes_{}_dataset_config.json'.format('test')

                if len(val_dirs) == 0:
                    random.shuffle(self.dirs)
                    # random.shuffle(test_dirs)
                    if p_eval < 1:
                        n = int(len(self.dirs) * p_eval)
                    else:
                        n = int(p_eval)

                    val_dirs = self.dirs[:n]
                    self.dirs = self.dirs[n:]

                with open(os.path.join(root, test_name), 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': test_dirs}, f)
                with open(os.path.join(root, valid_name), 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': val_dirs}, f)
                with open(os.path.join(root, train_name), 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': self.dirs}, f)
                if 'train' in mode:
                    pass
                elif 'valid' in mode:
                    self.dirs = val_dirs
                elif 'test' in mode:
                    self.dirs = test_dirs
            else:
                with open(load_path, 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': self.dirs}, f)
        else:
            # self.names = cp['names']
            self.dirs = cp['dirs']
        if num_samples is not None and num_samples > 0:
            self.dirs = self.dirs[ :num_samples] 
        # self.class_names = [self.names[key]['name'] for key in sorted(list(self.names.keys()))]
        # self.num_classes = len(self.class_names)

    def __getitem__(self, index):
        path = self.dirs[index]
        sample = self.load_sample(path)
        sample['sample_index'] = index
        sample['mode'] = self.mode
        if self.transforms is not None:
            s = self.transforms(sample)
            sample = {}
            for key, v in s.items():
                if key in ['img', 'mask', 'depth', 'gt_depth', 'depth_scales']:
                    if v is not None:
                        sample[key] = v.float()
                if key in ['depth_mul']:
                    sample[key] = v
        return sample

    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array
        if len(path) == 2:
            img_path, depth_path = path
            mask_path = None
        else:
            img_path, depth_path, mask_path = path

        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')
            img = np.array(sample['img'])
            max_value = np.max(img)
            min_value = np.min(img)
            #print('img shape is in city_scapes',img.shape)
            #print('max_value is in city_scapes',max_value)
            #print('min_value is in city_scapes',min_value)

        if self.with_intr or self.with_pc or self.with_depth:
            sample['depth_scale'] = 1000
            if self.depth_mul is not None:
                sample['depth_mul'] = self.depth_mul
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul

            #cx, cy, fx, fy = CityscapesRGBDDatasetintr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
            #sample['intr'] = [cx, cy, fx, fy]

        if self.with_pc or self.with_depth:
            # read depth information to numpy array
            with open(depth_path, 'rb') as f:
                # depth = np.array(cv2.imread(depth_path, -1)).astype(np.float32) / 256
                # print(depth.shape)
                depth = np.array(Image.open(f), dtype=np.float32)

                max_value = np.max(depth)
                min_value = np.min(depth)

                #print('max_value is in city_scapes',max_value)
                #print('min_value is in city_scapes',min_value)

                mask = depth > 0
                depth[mask] = (depth[mask] - 1) / 256
                mask = depth > 0
                depth[mask] = (0.209313 * 2262.52) / depth[mask]
                depth[~mask] = 0
                depth[depth < 0] = 0
                depth[depth > 150] = 0
                sample['depth'] = torch.as_tensor(depth, dtype=torch.float32)
                
                max_value = torch.max(sample['depth'])
                min_value = torch.min(sample['depth'])

                #print('max_value of depth after clamp is in city_scapes',max_value)
                #print('min_value of depth after clamp is in city_scapes',min_value)
                #print('sample depth shape in city_scapes', sample['depth'].shape)
                
        if self.with_mask and mask_path is not None:
            with open(mask_path, 'rb') as f:
                mask = torch.from_numpy(np.array(Image.open(f)))
                if len(mask.shape) > 2:
                    mask = torch.sum(mask[:, :, :3], dim=-1)
                    id_lut = self.cls_color_ids
                    #sample['plot'] = True
                else:
                    id_lut = self.names

                sample['mask'] = torch.zeros(mask.shape, dtype=torch.float32)
                for u in torch.unique(mask):
                    c = int(u)
                    if c > 1000:
                        c = c // 1000
                    c = int(c)
                    if c in id_lut:
                        sample['mask'][mask == u] = id_lut[c]['id']

                    # print(sample['mask'].shape)
                    # print(torch.unique(sample['mask']))
                sample['mask'] = torch.as_tensor(sample['mask'], dtype=torch.float32)

        return sample

    def load_ann(self, path):
        with open(os.path.join(path, 'scene_gt.json')) as f:
            data = json.load(f)
        return data

    def plot_sample(self, sample):
        plt.subplot(1, 3, 1)
        plt.imshow(sample['img'])
        print('img size', sample['img'].size)
        if 'depth' in sample:
            d = sample['depth']
            if 'depth_scale' in sample:
                d = d * sample['depth_scale']
                d = d / 1000
            plt.subplot(1, 3, 2)
            plt.imshow(d)
        if 'mask' in sample:
            plt.subplot(1, 3, 3)
            plt.imshow(sample['mask'])
            print({int(u): self.cls_names[int(u)] for u in torch.unique(sample['mask'])})

            #print(sample['mask'].shape)
        plt.show()


if __name__ == '__main__':
    ds = CityScapesDataset(None, with_pc=False, mode='train', with_mask=False, mode_tag=None, p_eval=0,
                           with_depth=True, depth_mul=1, load_from_checkpoint=True, root='/home/chowanki/git/vanishing-depth-self-supervised/data/cityscapes_segmentation')
    print('len', len(ds))
    print(ds.names)
    print(ds.num_classes)
    cls = []
    for i in np.random.permutation(len(ds)):
        sample = ds.__getitem__(i)
        #classes = torch.unique(sample['mask'])
        ## print(classes)
        #for c in classes:
        #    c = int(c)
        #    # print(c)
        #    if c not in cls:
        #        cls.append(c)
        #print(sample.keys())
        #if sample.get('plot', True):
        ds.plot_sample(sample)
