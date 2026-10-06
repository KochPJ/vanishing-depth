import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import copy



class NyuDepthv2Dataset(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True,
                 with_mask=False, load_from_checkpoint=True, with_pc=False, mode=None, p_eval=500,
                 depth_mul=1, mode_tag=None, with_scene_cls=False, balanced_sampling=False, copy_past=None):
        if root is None:
            root = './data/NYU_Depth_v2'
        self.root = root
        self.transforms = transforms
        self.with_intr = with_intr
        self.with_depth = with_depth
        self.with_mask = with_mask
        self.with_pc = with_pc
        self.p_eval = p_eval
        self.depth_mul = depth_mul
        self.mode = mode
        self.with_scene_cls = with_scene_cls
        self.cam_params = [3.2558244941119034e+02, 2.5373616633400465e+02,
                           5.1885790117450188e+02, 5.1946961112127485e+02]

        self.copy_past = copy_past
        #self.seg_labels = {
        #    '1': 38, '2': 38, '3': 38, '4': 38, '5': 38, '6': 38, '7': 38, '8': 38, '9': 38, '10': 38, '11': 38,
        #    '12': 38, '13': 38, '14': 38, '15': 38, '16': 38, '17': 38, '18': 38, '19': 38, '20': 38, '21': 38,
        #    '22': 38, '23': 27, '24': 27, '25': 27, '26': 27, '27': 27, '28': 28, '29': 29, '30': 30, '31': 31,
        #    '32': 32, '33': 33, '34': 34, '35': 35, '36': 36, '37': 37}

        self.seg_classes = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf',
                            'picture','counter','blinds', 'desk','shelves','curtain','dresser','pillow','mirror',
                            'floor mat','clothes','ceiling','books','refridgerator', 'television','paper','towel',
                            'shower curtain','box','whiteboard','person','night stand','toilet', 'sink','lamp',
                            'bathtub','bag','otherstructure', 'furniture','otherprop']

        self.cls_classes = ['bedroom', 'kitchen', 'living_room', 'bathroom', 'dining room',
                            'office', 'home_office', 'classroom', 'bookstore', 'others']

        if mode_tag == 'classification':
            self.names = {str(i+1): {'name': n} for i, n in enumerate(self.cls_classes)}
        else:
            self.names = {str(i+1): {'name': n} for i, n in enumerate(self.seg_classes)}

        #print(len(self.seg_classes), 'seg_classes', self.seg_classes)
        self.seg_labels = {}

        if mode_tag is not None:
            mode = mode + '_' + mode_tag
        if mode is not None:
            file_name = 'NYU_Depth_v2_{}_dataset_config.json'.format(mode)
        else:
            file_name = 'NYU_Depth_v2_dataset_config.json'

        load_path = os.path.join(self.root, file_name)
        print('nyu depth v2 load path', load_path)
        cp = None
        if load_from_checkpoint:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    cp = json.load(f)

        if mode_tag in ['segmentation', 'classification'] and cp is None:
            with open(os.path.join(root, 'seg_train.txt'), 'rb') as f:
                datalist = f.readlines()

            self.dirs = [[os.path.join(root, 'test', 'images'),
                l.decode('utf-8').strip('\n').split('\t')[0].split('/')[1].split('.')[0]] for l in datalist]
            with open(os.path.join(root, 'seg_val.txt'), 'rb') as f:
                datalist = f.readlines()
            self.test_dirs = [[os.path.join(root, 'test', 'images'),
                l.decode('utf-8').strip('\n').split('\t')[0].split('/')[1].split('.')[0]] for l in datalist]

            print('train samples', len(self.dirs))
            print('test val samples', len(self.test_dirs))

            train_name = 'NYU_Depth_v2_{}_dataset_config.json'.format('train_{}'.format(mode_tag))
            valid_name = 'NYU_Depth_v2_{}_dataset_config.json'.format('valid_{}'.format(mode_tag))
            test_name = 'NYU_Depth_v2_{}_dataset_config.json'.format('test_{}'.format(mode_tag))
            with open(os.path.join(root, test_name), 'w') as f:
                json.dump({'names': self.names,
                           'dirs': self.test_dirs}, f)
            with open(os.path.join(root, valid_name), 'w') as f:
                json.dump({'names': self.names,
                           'dirs': self.test_dirs}, f)
            with open(os.path.join(root, train_name), 'w') as f:
                json.dump({'names': self.names,
                           'dirs': self.dirs}, f)

        elif cp is None and mode_tag not in ['segmentation', 'classification']:
            self.dirs = []
            self.test_dirs = []
            for path, folders, files in os.walk(self.root):
                if len(folders) == 0:
                    if 'train' in path:
                        ppms = sorted([f for f in files if '.ppm' == f[-4:]])
                        pgms = sorted([f for f in files if '.pgm' == f[-4:]])
                        for ppm, pgm in zip(ppms, pgms):
                            try:
                                sample = self.load_sample([path, ppm, pgm])
                                self.dirs.append([path, ppm, pgm])
                            except:
                                pass
                    else:
                        rgbs = [[path, f.split('_')[0]] for f in os.listdir(path) if 'rgb.png' in f]
                        for rgb in rgbs:
                            try:
                                _ = self.load_sample(rgb)
                                self.test_dirs.append(rgb)
                                #print('adding test', rgb)
                            except Exception as e:
                                #print(rgbs)
                                #print(self.dirs[-1])
                                #raise e
                                pass

            if mode is not None:
                print('total train samples: {}'.format(len(self.dirs)))
                print('total test samples: {}'.format(len(self.test_dirs)))

                if mode_tag is not None:
                    train_name = 'NYU_Depth_v2_{}_dataset_config.json'.format('train_{}'.format(mode_tag))
                    valid_name = 'NYU_Depth_v2_{}_dataset_config.json'.format('valid_{}'.format(mode_tag))
                    test_name = 'NYU_Depth_v2_{}_dataset_config.json'.format('test_{}'.format(mode_tag))
                else:
                    train_name = 'NYU_Depth_v2_{}_dataset_config.json'.format('train')
                    valid_name = 'NYU_Depth_v2_{}_dataset_config.json'.format('valid')
                    test_name = 'NYU_Depth_v2_{}_dataset_config.json'.format('test')

                random.shuffle(self.dirs)
                if p_eval < 1:
                    n = int(len(self.dirs) * p_eval)
                else:
                    n = int(p_eval)

                with open(os.path.join(root, test_name), 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': self.test_dirs}, f)
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
                    self.dirs = self.test_dirs
                else:
                    raise NotImplementedError('unknown mode {}'.format(mode))
            else:
                with open(load_path, 'w') as f:
                    json.dump({'names': self.names,
                               'dirs': self.dirs}, f)
        else:
            self.names = cp['names']
            self.dirs = cp['dirs']

        self.class_names = [self.names[key]['name'] for key in sorted(list(self.names.keys()))]
        self.num_classes = len(self.class_names)

        #with open(os.path.join(root, train_name), 'w') as f:
        #            json.dump({'dirs': self.dirs[:n]}, f)
        #        with open(os.path.join(root, valid_name), 'w') as f:
        #            json.dump({'dirs': self.dirs[n:]}, f)
        #    else:
        #        with open(load_path, 'w') as f:
        #            json.dump({'dirs': self.dirs}, f)
        #else:
        #    self.dirs = cp['dirs']
        self.balanced_sampling = balanced_sampling
        self.balanced_dirs = {}
        if self.balanced_sampling and self.with_scene_cls:
            for index, path in enumerate(self.dirs):
                path, n = path

                with open(os.path.join(path, '{}_scene.json'.format(str(int(n) - 1).zfill(6)))) as jf:
                    cls = json.load(jf)['cls']
                if cls in self.cls_classes:
                    cls = int(self.cls_classes.index(cls))
                else:
                    cls = int(self.cls_classes.index('others'))
                if cls not in self.balanced_dirs:
                    self.balanced_dirs[cls] = []
                self.balanced_dirs[cls].append(index)

            for key, v in self.balanced_dirs.items():
                print(key, self.cls_classes[key], len(v), 100 * len(v) / sum([len(v) for v in self.balanced_dirs.values()]))
            #print(self.balanced_dirs.keys())

    def __getitem__(self, index):

        if self.balanced_sampling and self.mode == 'train':
            cls_index = int(np.random.choice(list(self.balanced_dirs.keys()), size=1))
            nr = int(np.random.choice(list(range(len(self.balanced_dirs[cls_index]))), size=1))
            #print(cls_index, nr)
            index = self.balanced_dirs[cls_index][nr]
            #print(cls_index, nr, index)

        path = self.dirs[index]


        #path = self.dirs[index]
        sample = self.load_sample(path)
        sample['mode'] = self.mode
        sample['sample_index'] = index
        #print(sample.keys())
        if self.transforms is not None:
            s = self.transforms(sample)
            sample = {}
            for key, v in s.items():
                if v is None:
                    continue

                if key in ['x', 'mask', 'depth', 'gt_depth', 'img', 'depth_scales', 'ori_depth']:
                    sample[key] = v.float()
                elif key in ['cls', 'depth_mul']:
                    sample[key] = v

        if self.with_scene_cls:
            if self.with_depth:
                sample = ({'x': sample['x'], 'depth': sample['depth']}, sample['cls'])
            else:
                sample = ({'x': sample['x']}, sample['cls'])

        #del sample['mode']
        #print(sample.keys())
        return sample

    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array
        if len(path) == 3:
            path, ppm, pgm = path
            pmm = '.'.join(ppm.split('.')[:-1]) + '.dump'
            pmm = 'a'+pmm[1:]
        else:
            path, n = path
            pgm = '{}_depth.png'.format(str(int(n)-1).zfill(6))
            ppm = '{}_rgb.png'.format(str(int(n)-1).zfill(6))
            pmm = '{}.png'.format(str(int(n)).zfill(6))
            #print(pgm, ppm, pmm)

        img_path = os.path.join(path, ppm)
        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        if self.with_pc or self.with_depth:
            # read depth information to numpy array
            depth_path = os.path.join(path, pgm)
            with open(depth_path, 'rb') as f:
                sample['depth'] = torch.as_tensor(np.array(Image.open(f), dtype=np.float32), dtype=torch.float32) #/ 10.0 * 65535.0
                if '.pgm' in depth_path:
                    sample['depth'][sample['depth'] <= 2047] = \
                        0.1236 * torch.tan(sample['depth'][sample['depth'] <= 2047] / 2842.5 + 1.1863)

                    sample['depth'][sample['depth'] > 65000] = 0
                    #sample['depth'] = np.array(Image.open(f)).astype(np.float32)
                    sample['depth_scale'] = 0.1
                else:
                    sample['depth_scale'] = 1

                if self.depth_mul is not None:
                    sample['depth_scale'] = sample['depth_scale'] / self.depth_mul
                    sample['depth_mul'] = self.depth_mul

        if self.with_mask:
            #print(os.listdir(os.path.join(path, 'label')))
            mask_path = os.path.join(path, pmm)
            if '.png' in mask_path:
                with open(mask_path, 'rb') as f:
                    #sample['mask'] = torch.from_numpy(np.array(Image.open(f)).astype(np.float32))
                    mask = np.array(Image.open(f), dtype=np.int8)
                sample['mask'] = torch.from_numpy(np.array(mask + 1, dtype=np.uint8)) #torch.zeros(mask.shape, dtype=torch.float32)


        if self.with_intr or self.with_pc:
            sample['intr'] = copy.deepcopy(self.cam_params)

        if self.with_pc:
            w, h = sample['img'].size
            xmap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(w), axis=0), h, axis=0), dtype=torch.float32)
            ymap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(h), axis=1), w, axis=1), dtype=torch.float32)
            cx, cy, fx, fy = sample['intr']
            pt2 = (torch.as_tensor(sample['depth'], dtype=torch.float32) * sample['depth_scale']) / 1000  # mm to meter
            pt0 = (xmap - cx) * pt2 / fx
            pt1 = (ymap - cy) * pt2 / fy
            sample['pc'] = torch.stack((pt0, pt1, pt2), axis=0)

        if self.with_scene_cls:
            cls_path = ppm.replace('_rgb.png', '_scene.json')

            with open(os.path.join(path, cls_path)) as jf:
                cls = json.load(jf)['cls']

            if cls in self.cls_classes:
                sample['cls'] = int(self.cls_classes.index(cls))
            else:
                sample['cls'] = int(self.cls_classes.index('others'))


        return sample


    def plot_sample(self, sample):
        #overlay = np.array(sample['img']).astype(np.float32)
        #overlay[:, :, 0] = overlay[:, :, 0]*0.7 + ((sample['depth']/np.max(sample['depth']))*255) *0.3
        y = 1
        if self.with_mask:
            y += 1

        if self.with_depth:
            y += 1

        plt.subplot(1,y,1)
        plt.imshow(sample['img'])
        print('img', sample['img'].size)
        c = 2
        #print(y, c)
        if self.with_depth:
            plt.subplot(1,y,c)
            #plt.imshow(overlay.astype(np.uint8))
            #plt.subplot(1,3,3)
            depth = sample['depth'] * sample['depth_scale'] / 1000
            #depth[depth > 5] = 0
            depth[depth < 1] = 0
            plt.imshow(depth)
            
            print('depth', depth.shape)
            c += 1

        if 'mask' in sample:
            plt.subplot(1, y, c)
            plt.imshow(sample['mask'])
            print('mask', sample['mask'].shape)
            c += 1

        plt.show()

    def save_sample(self, sample, index):

        depth = Image.fromarray(np.array(sample['depth'].numpy() * sample['depth_scale'], dtype=np.uint16))
        img = sample['img']
        mask = Image.fromarray(np.array(sample['mask'].numpy(), dtype=np.uint8)).resize(depth.size, Image.NEAREST)

        outpath = os.path.join(self.copy_past, self.mode, 'depth')
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        depth.save(os.path.join(outpath, '{}_depth.png'.format(str(i).zfill(6))))


        outpath = os.path.join(self.copy_past, self.mode, 'images')
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        img.save(os.path.join(outpath, '{}_rgb.png'.format(str(i).zfill(6))))
        

        outpath = os.path.join(self.copy_past, self.mode, 'masks')
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        mask.save(os.path.join(outpath, '{}_mask.png'.format(str(i).zfill(6))))
        
        print('img', img.size)
        print('depth', depth.size)
        print('mask', mask.size)





if __name__ == '__main__':
    ds = NyuDepthv2Dataset(mode='train', load_from_checkpoint=True, with_mask=True, with_depth=True,
                           mode_tag='segmentation', with_intr=True, with_scene_cls=False, p_eval=0, copy_past='./data/NYU_V2_Seg')

    print(len(ds), ds.dirs[-1], ds.dirs[0])
    print(ds.names)
    indexes = np.random.permutation(len(ds))
    #indexes = range(len(ds))
    #for i in np.random.permutation(len(ds)):
    for i in range(len(ds)):
        #i = len(ds)-i-1
        print('{} / {}'.format(i+1, len(ds)))


        sample = ds.__getitem__(i)
        #print('intr', sample.get('intr'))
        ds.save_sample(sample, i)
        #ds.plot_sample(sample)
    input()

    for key in ds.seg_classes:
        if key not in ds.seg_labels:
            print('miss', key)


'''
    if args.nyu_depth:
        from datasets.nyu_depth_v2 import NyuDepthv2Dataset
        print('Creating NYU Depth V2 dataset')
        datasets['nyu_depth'] = NyuDepthv2Dataset(mode='train', load_from_checkpoint=True, with_intr=args.with_intr)
        valid_datasets['nyu_depth'] = NyuDepthv2Dataset(mode='valid', load_from_checkpoint=True, with_intr=args.with_intr)
        test_datasets['nyu_depth'] = NyuDepthv2Dataset(mode='test', load_from_checkpoint=True, with_intr=args.with_intr)
        print('Train NYU Depth V2 created with "{}" samples.'.format(len(datasets['nyu_depth'])))
        print('Valid NYU Depth V2 created with "{}" samples.'.format(len(valid_datasets['nyu_depth'])))
        print('Test NYU Depth V2 created with "{}" samples.'.format(len(test_datasets['nyu_depth'])))
'''