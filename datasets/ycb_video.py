import sys
sys.path.append('/home/kochpaul/git/vanishing-depth-self-supervised')

import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
from utils.stuff import get_empty_ann_file
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import transforms3d as t3d


class YCBVDataset(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True, mode_tag=None,
                 with_mask=False, load_from_checkpoint=True, with_pc=False, mode=None, with_bbox=False, with_pose=False,
                 with_synt_data=False, p_eval=500, with_coco=False, pc_points=512, depth_mul=1, intr_mat=False):
        if root is None:
            root = './data/YCV_Video'

        self.symmetries_file = 'ycbv_symmetries.json'
        self.models_dir = 'ycbv_models'
        self.root = root
        self.transforms = transforms
        self.with_intr = with_intr
        self.with_depth = with_depth
        self.with_synt_data = with_synt_data
        self.with_mask = with_mask
        self.with_pc = with_pc
        self.with_bbox = with_bbox
        self.with_pose = with_pose
        self.pc_points = pc_points
        self.depth_mul = depth_mul
        self.intr_mat = intr_mat
        self.sym = [12, 15, 18, 19, 20]
        self.mode = mode
        self.names = {
            '0001': {'name': '002_master_chef_can', 'c': [1.3360, -0.5000, 3.5105]},
            '0002': {'name': '003_cracker_box', 'c': [0.5575, 1.7005, 4.8050]},
            '0003': {'name': '004_sugar_box', 'c': [1.3360, -0.5000, 3.5105]},
            '0004': {'name': '005_tomato_soup_can', 'c': [-0.0240, -1.5270, 8.4035]},
            '0005': {'name': '006_mustard_bottle', 'c': [1.2995, 2.4870, -11.8290]},
            '0006': {'name': '007_tuna_fish_can', 'c': [-0.1565, 0.1150, 4.2625]},
            '0007': {'name': '008_pudding_box', 'c': [1.1645, -4.2015, 3.1190]},
            '0008': {'name': '009_gelatin_box', 'c': [1.4460, -0.5915, 3.6085]},
            '0009': {'name': '010_potted_meat_can', 'c': [2.4195, 0.3075, 8.0715]},
            '0010': {'name': '011_banana', 'c': [-18.6730, 12.1915, -1.4635]},
            '0011': {'name': '019_pitcher_base', 'c': [5.3370, 5.8855, 25.6115]},
            '0012': {'name': '021_bleach_cleanser', 'c': [4.9290, -2.4800, -13.2920]},
            '0013': {'name': '024_bowl', 'c': [-0.2270, 0.7950, -2.9675]},
            '0014': {'name': '025_mug', 'c': [-8.4675, -0.6995, -1.6145]},
            '0015': {'name': '035_power_drill', 'c': [9.0710, 20.9360, -2.1190]},
            '0016': {'name': '036_wood_block', 'c': [1.4265, -2.5305, 17.1890]},
            '0017': {'name': '037_scissors', 'c': [1.3360, -0.5000, 3.5105]},
            '0018': {'name': '040_large_marker', 'c': [0.0460, -2.1040, 0.3500]},
            '0019': {'name': '051_large_clamp', 'c': [10.5180, -1.9640, -0.4745]},
            '0020': {'name': '052_extra_large_clamp', 'c': [-0.3950, -10.4130, 0.1620]},
            '0021': {'name': '061_foam_brick', 'c': [-0.0805, 0.0805, -8.2435]}
        }

        #if self.with_pose:
        #    import open3d as o3d
        #    self.pcs = {index: o3d.io.read_point_cloud(os.path.join(root, 'ycbv_models', 'models', 'obj_{}.ply'.format(
        #        index.zfill(6)))) for index in self.names.keys()}

        if mode is not None:
            if mode_tag is not None:
                file_name = 'ycb_v_{}_{}_dataset_config.json'.format(mode, mode_tag)
            else:
                file_name = 'ycb_v_{}_dataset_config.json'.format(mode)
        else:
            file_name = 'ycb_v_dataset_config.json'

        load_path = os.path.join(self.root, file_name)
        cp = None
        if load_from_checkpoint:
            print('loading checkpoint from {}'.format(load_path))
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    cp = json.load(f)
                #print('cp loaded from {}'.format(load_path))

        self.dirs = []
        test_dirs = []
        if cp is None:
            for path, folders, files in os.walk(self.root):
                if not with_synt_data:
                    if 'synt' in path:
                        continue
                if 'scene_camera.json' in files and 'scene_gt.json' in files:
                    try:
                        ann = self.load_ann(path)
                        for key, value in ann.items():
                            try:
                                _ = self.load_sample((path, key))
                                if 'ycbv_test_all' in path:
                                    test_dirs.append((path, key))
                                elif 'ycbv_train_' in path:
                                    self.dirs.append((path, key))
                            except Exception as e:
                                print(path, key, e)
                    except:
                        continue
            if mode is not None:
                if mode_tag is not None:
                    train_name = 'ycb_v_{}_{}_dataset_config.json'.format('train', mode_tag)
                    valid_name = 'ycb_v_{}_{}_dataset_config.json'.format('valid', mode_tag)
                    test_name = 'ycb_v_{}_{}_dataset_config.json'.format('test', mode_tag)
                else:
                    train_name = 'ycb_v_{}_dataset_config.json'.format('train')
                    valid_name = 'ycb_v_{}_dataset_config.json'.format('valid')
                    test_name = 'ycb_v_{}_dataset_config.json'.format('test')

                random.shuffle(self.dirs)
                random.shuffle(test_dirs)

                if p_eval < 1:
                    n = int(len(self.dirs) * p_eval)
                else:
                    n = int(p_eval)

                with open(os.path.join(root, test_name), 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': test_dirs}, f)
                with open(os.path.join(root, valid_name), 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': self.dirs[:n]}, f)
                with open(os.path.join(root, train_name), 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': self.dirs[n:]}, f)
                if 'train' in mode:
                    self.dirs = self.dirs[n:]
                elif 'valid' in mode:
                    self.dirs = self.dirs[:n]
                elif 'test' in mode:
                    self.dirs = test_dirs
            else:
                with open(load_path, 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': self.dirs}, f)
        else:
            self.names = cp['names']
            self.dirs = cp['dirs']

        self.class_names = [self.names[key]['name'] for key in sorted(list(self.names.keys()))]
        self.num_classes = len(self.class_names)

        self.with_coco = with_coco
        self.coco_ann = None
        self.coco = None

        self.data_infos = None
        self.img_ids = None
        self.cat_ids = None
       

    def __getitem__(self, index):

        path = self.dirs[index]
        sample = self.load_sample(path)

        sample['sample_index'] = index
        sample['mode'] = self.mode

        if self.transforms is not None:
            sample = self.transforms(sample)
            #sample = {}
            for key, v in sample.items():
                if key in ['img', 'mask', 'depth', 'gt_depth']:
                    sample[key] = v.float()

        
        sample['image_id'] = int(path[1])
        #print('path', path[0])
        sample['scene_id'] = int(path[0].split('/')[-1])
        
        return sample

    def __len__(self):
        if self.coco is not None:
            return len(self.data_infos)
        else:
            return len(self.dirs)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array
        path, idx = path
        name = str(idx).zfill(6)
        img_path = os.path.join(path, 'rgb', name + '.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(path, 'rgb', name + '.jgp')
        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        sample['img_path'] = img_path

        if self.with_intr or self.with_pc or self.with_depth:
            with open(os.path.join(path, 'scene_camera.json')) as f:
                scene_camera = json.load(f)[str(idx)]

            sample['depth_scale'] = scene_camera['depth_scale']
            if self.depth_mul is not None:
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul
                sample['depth_mul'] = self.depth_mul
            intr = np.array(scene_camera['cam_K']).reshape((3,3))
            if self.intr_mat:
                sample['intr'] = torch.as_tensor(intr, dtype=torch.float32)
            else:
                cx, cy, fx, fy = intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
                sample['intr'] = [cx, cy, fx, fy]
            #print(intr)
            #print(scene_camera['depth_scale'])

        if self.with_pc or self.with_depth:
            # read depth information to numpy array
            depth_path = os.path.join(path, 'depth', name + '.png')
            with open(depth_path, 'rb') as f:
                sample['depth'] = torch.as_tensor(np.array(Image.open(f), dtype=np.float32), dtype=torch.float32)

        if self.with_mask:
            depth_path = os.path.join(path, 'mask_visib')
            ann = self.load_ann(path)[str(idx)]
            classes = [int(v['obj_id']) for v in ann]
            masks = sorted([m for m in list(os.listdir(depth_path)) if m[:6] == name])
            w, h = sample['img'].size
            mask = torch.zeros((h, w), dtype=torch.int8)
            for cls, m in zip(classes, masks):
                with open(os.path.join(depth_path, m), 'rb') as f:
                    ml = np.array(Image.open(f).convert('RGB'))[:, :, 0]
                    mask[ml != 0] = cls
            sample['mask'] = mask


        if self.with_bbox:
            with open(os.path.join(path, 'scene_gt_info.json')) as f:
                gt_info = json.load(f)[str(idx)]
            with open(os.path.join(path, 'scene_gt.json')) as f:
                gt = json.load(f)[str(idx)]
            sample['ori_size'] = sample['img'].size
            sample['bboxes'] = torch.from_numpy(
                np.array([boxes['bbox_visib'] for boxes in gt_info], dtype=np.float32))

            #print(sample['bboxes'])
            sample['bboxes'][:, :2] += sample['bboxes'][:, 2:] / 2
            #print(sample['ori_size'])
            #print(sample['bboxes'])
            sample['bboxes'] = sample['bboxes'] / torch.tensor([[sample['ori_size'][0], sample['ori_size'][1],
                                                                 sample['ori_size'][0], sample['ori_size'][1]]])
            #print(sample['bboxes'])
            sample['cls'] = torch.from_numpy(np.array([cls['obj_id'] for cls in gt], dtype=np.float32))
            #print(sample['cls'])
            #print(sample['cls'])

        if self.with_pose:
            with open(os.path.join(path, 'scene_gt.json')) as f:
                gt = json.load(f)[str(idx)]
            sample['quat'] = torch.from_numpy(np.array([t3d.quaternions.mat2quat(
                np.array(cls['cam_R_m2c']).reshape((3, 3))) for cls in gt], dtype=np.float32))
            sample['rot'] = torch.from_numpy(np.array([
                np.array(cls['cam_R_m2c']).reshape((3, 3)) for cls in gt], dtype=np.float32))
            sample['pos'] = torch.from_numpy(
                np.array([np.array(cls['cam_t_m2c']) / 1000 for cls in gt], dtype=np.float32))

        if self.with_pc:
            w, h = sample['img'].size
            xmap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(w), axis=0), h, axis=0), dtype=torch.float32)
            ymap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(h), axis=1), w, axis=1), dtype=torch.float32)
            cx, cy, fx, fy = sample['intr']
            pt2 = (torch.as_tensor(sample['depth'], dtype=torch.float32) * sample['depth_scale']) / 1000  # mm to meter
            pt0 = (xmap - cx) * pt2 / fx
            pt1 = (ymap - cy) * pt2 / fy
            sample['pc'] = torch.stack((pt0, pt1, pt2), axis=0)

        return sample

    def load_ann(self, path):
        with open(os.path.join(path, 'scene_gt.json')) as f:
            data = json.load(f)
        return data

    def plot_sample(self, sample, scale_depth=True):
        plt.subplot(2, 2, 1)
        plt.imshow(sample['img'])
        if 'depth' in sample:
            print('mean c', torch.mean(sample['depth'][sample['depth'] > 0]),
                  torch.std(sample['depth'][sample['depth'] > 0]),
                  torch.min(sample['depth'][sample['depth'] > 0]),
                  torch.max(sample['depth'][sample['depth'] > 0])
                  )
            d = sample['depth']

            if 'depth_scale' in sample and scale_depth:
                d = d * sample['depth_scale']
                d = d / 1000
            plt.subplot(2,2 , 2)
            plt.imshow(d)
        if 'mask' in sample:
            plt.subplot(2, 2 , 3)
            plt.imshow(sample['mask'])
            mask = sample['mask'].numpy()
            h_, w_ = mask.shape
            print(h_, w_)
            uni = np.unique(mask)[1:]
            print(uni)
            for u in uni:
                ys, xs = np.where(mask == u)
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                w, h = x1-x0, y1-y0

                bbox = [x0+w/2, y0+h/2, w, h]
                bbox = [bbox[0] / w_, bbox[1] / h_, bbox[2] / w_, bbox[3] / h_]
                print(u, bbox)

        if 'bboxes' in sample:
            import cv2
            
            bbimg = np.array(sample['img'], dtype=np.uint8)
            plt.subplot(2, 2, 4)
            for bbox in sample['bboxes']:
                cx, cy, w, h = bbox
                w_, h_ = sample['img'].size
                cx = int(cx*w_)
                cy = int(cy*h_)
                w = int(w*w_)
                h = int(h*h_)
                print(x1, y1, w, h)
                cv2.rectangle(bbimg, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), color=(255,0,0), thickness=2)
            plt.imshow(bbimg)

        if 'pc' in sample:
            plt.subplot(2, 2, 4)
            plt.imshow(np.abs(sample['pc'][0, :, :]))
            plt.subplot(2, 2, 5)
            plt.imshow(np.abs(sample['pc'][1, :, :]))
            plt.subplot(2, 2, 6)
            plt.imshow(sample['pc'][2, :, :])

        plt.show()

    def create_coco_ann_file(self):
        ann_file = get_empty_ann_file()

        for cat_id, d in self.names.items():
            ann_file['categories'].append({'id': int(cat_id), 'name': d['name']})
        ann_id = 0
        image_id = 0
        for i in range(len(self.dirs)):
            path = self.dirs[i]
            try:
                sample = self.load_sample(path)
            except:
                sample = {}

            curr_anns = []
            for bbox, cls in sample.get('bboxes', []):
                ann = {
                    'id': ann_id,
                    'category_id': cls,
                    'image_id': image_id,
                    'bbox': bbox,
                    'iscrowd': 0,
                    'area': float(bbox[2] * bbox[3])
                }
                if ann['area'] < 10:
                    continue
                curr_anns.append(ann)
                ann_id += 1

            if len(curr_anns) == 0:
                continue

            # add image and anns
            w, h = sample['img'].size
            ann_file['images'].append({
                'id': image_id,
                'width': w,
                'height': h,
                'file_name': str(sample['img_path']),
                'path': path
            })
            for ann in curr_anns:
                ann_file['annotations'].append(ann)
            image_id += 1
        return ann_file


if __name__ == '__main__':

    #import open3d as o3d
    #p = '/home/kochpaul/git/vanishing-depth-self-supervised/data/YCV_Video/ycbv_models/models/obj_000001.ply'
    #pc = o3d.io.read_point_cloud(p)
    ##o3d.visualization.draw_geometries([pc])
    #c = [1.3360, -0.5000, 3.5105]
    #print(pc.get_center())
    #points = np.array(pc.points)
    #for i in range(3):
    #    print(np.mean(points[:, i]), np.std(points[:, i]), np.min(points[:, i]), np.max(points[:, i]))
    import os
    import sys
    #sys.path.append(os.getcwd())
    
    from datasets.transforms import RandomDepth, Depth2distance
    
    

    import torchvision.transforms as pytfs

    max_dist = 15
    t = None #pytfs.Compose([Depth2distance(max_dist=max_dist), RandomDepth(max_depth=max_dist)])


    ds = YCBVDataset(mode='test', transforms=None, load_from_checkpoint=True, with_intr=True,
                              with_mask=True, with_bbox=True, with_pose=True, intr_mat=True,
                              mode_tag='pose', with_synt_data=True)
    print(len(ds))
    print(ds.names)
    print(ds.num_classes)
    classes = []
    for i in np.random.permutation(len(ds)):
        sample = ds.__getitem__(i)
        #print(sample.keys())
        #print(sample['cls'])
        #for cls in sample['cls']:
        #    c = float(cls)
        #    if c not in classes:
        #        classes.append(c)
        #        print(len(classes), sorted(classes))

        #print('intr', sample.get('intr'))
        #print('bbox', sample.get('bboxes'))

        #print('cls names', {int(c): ds.names[str(int(c)).zfill(4)]['name'] for c in sample.get('cls')})
        print(sample['scene_id'], sample['image_id'])
        ds.plot_sample(sample, scale_depth=False)


