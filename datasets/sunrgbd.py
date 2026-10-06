import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import json
from os import listdir
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.io as scio
import random


class SUNRGBD_Dataset(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_pc=False, mode_tag=None,
                 with_depth=True, with_mask=False, load_from_checkpoint=True, mode=None, p_eval=500,
                 depth_mul=1, with_scene_cls=False, balanced_sampling=False, sample_with_p=False,
                 depth_mode='depth'):
        if root is None:
            root = './data/SunRGBD/SUNRGBD'
        self.root = root
        #get_mask(root)

        self.transforms = transforms
        self.with_pc = with_pc
        self.with_depth = with_depth
        self.with_mask = with_mask
        self.with_intr = with_intr
        self.balanced_sampling = balanced_sampling
        self.sample_with_p = sample_with_p
        self.depth_mul = depth_mul
        self.with_scene_cls = with_scene_cls
        self.depth_mode = depth_mode
        self.mode = mode
        #self.scene_classes = [
        #    'basement', 'bathroom', 'bedroom', 'bookstore', 'cafeteria', 'classroom', 'coffee_room', 'computer_room',
        #    'conference_room', 'corridor', 'dinette', 'dining_area', 'dining_room', 'discussion_area', 'exhibition',
        #    'furniture_store', 'gym', 'home', 'home_office', 'hotel_room', 'idk', 'indoor_balcony', 'kitchen', 'lab',
        #    'laundromat', 'lecture_theatre', 'library', 'living_room', 'lobby', 'mail_room', 'office', 'office_dining',
        #    'office_kitchen', 'playroom', 'printer_room', 'reception', 'reception_room', 'recreation_room',
        #    'rest_space', 'stairs', 'storage_room', 'study', 'study_space'
        #]
        self.scene_classes = [
            'bathroom', 'rest_space', 'bedroom', 'classroom', 'office', 'furniture_store', 'living_room', 'kitchen',
            'corridor', 'lab', 'conference_room', 'dining_area', 'dining_room', 'discussion_area', 'home_office',
            'study_space', 'library', 'lecture_theatre', 'computer_room'#, 'others'
            #'bookstore', 'cafeteria', 'classroom', 'coffee_room', 'computer_room',
            #'conference_room', 'corridor', 'dinette', 'dining_area', 'dining_room', 'discussion_area', 'exhibition',
            #'furniture_store', 'gym', 'home', 'home_office', 'hotel_room', 'idk', 'indoor_balcony', 'kitchen', 'lab',
            #'laundromat', 'lecture_theatre', 'library', 'living_room', 'lobby', 'mail_room', 'office', 'office_dining',
            #'office_kitchen', 'playroom', 'printer_room', 'reception', 'reception_room', 'recreation_room',
            #'rest_space', 'stairs', 'storage_room', 'study', 'study_space'
        ]

        # following https://github.com/facebookresearch/omnivore/issues/12
        self.sensor_to_params = {
            "kv1": {
                "baseline": 0.075,
            },
            "kv1_b": {
                "baseline": 0.075,
            },
            "kv2": {
                "baseline": 0.075,
            },
            "realsense": {
                "baseline": 0.095,
            },
            "xtion": {
                "baseline": 0.095, # guessed based on length of 18cm for ASUS xtion v1
            },
        }

        self.focal_dict = {
            '529.5': 'kv2',
            '519.469611': 'kv1',
            '520.7444': 'kv1',
            '570.342224': 'xtion',
            '570.342205': 'xtion',
            '570.342218': 'xtion',
            '533.069214': 'xtion',
            '691.584229': 'realsense',
            '693.74469': 'realsense'
        } 
        
        print('cls classes: ', len(self.scene_classes))

        if not self.with_scene_cls:

            self.names = {i + 1: {'name': cls, 'samples': []} for i, cls in enumerate([
                    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                    'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
                    'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv', 'paper',
                    'towel', 'shower_curtain', 'box', 'whiteboard', 'person', 'night_stand', 'toilet', 'sink',
                    'lamp', 'bathtub', 'bag'
                ])
            }
        else:
            self.names = {str(i): {'name': cls, 'samples': []} for i, cls in enumerate(self.scene_classes)}


        if mode_tag is not None:
            mode = mode + '_' + mode_tag
        if mode is not None:
            file_name = 'sunrgbd_{}_dataset_config.json'.format(mode)
        else:
            file_name = 'sunrgbd_dataset_config.json'

        load_path = os.path.join(self.root, file_name)
        cp = None
        print('load path', load_path)
        if load_from_checkpoint:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    cp = json.load(f)
            else:
                print('no path ', load_path)
        ids = []
        self.dirs = []
        self.test_dirs = []
        if cp is None:
            train_skip = 0
            test_skip = 0
            added = 0
            test_added = 0
            for path, folders, files in os.walk(self.root):
                if 'intrinsics.txt' in files and 'depth' in folders and 'image' in folders:
                    if 'train' in path:
                        try:
                            s = self.load_sample(path)
                            self.dirs.append([path, s['cls']])
                            added += 1
                        except Exception as e:
                            train_skip += 1
                            print('added [train: {}, test: {}] | skips [train: {}, test: {}] | '
                                  'Skip train {} due to {}'.format(added, test_added, train_skip, test_skip, path, e))
                            #raise e
                    else:
                        try:
                            s = self.load_sample(path)
                            self.test_dirs.append(path)
                            test_added += 1
                        except Exception as e:
                            test_skip += 1
                            print('added [train: {}, test: {}] | skips [train: {}, test: {}] | '
                                  'Skip test {} due to {}'.format(added, test_added, train_skip, test_skip, path, e))
                            continue

            if mode is not None:
                if mode_tag is not None:
                    train_name = 'sunrgbd_{}_dataset_config.json'.format('train_{}'.format(mode_tag))
                    valid_name = 'sunrgbd_{}_dataset_config.json'.format('valid_{}'.format(mode_tag))
                    test_name = 'sunrgbd_{}_dataset_config.json'.format('test_{}'.format(mode_tag))
                else:
                    train_name = 'sunrgbd_{}_dataset_config.json'.format('train')
                    valid_name = 'sunrgbd_{}_dataset_config.json'.format('valid')
                    test_name = 'sunrgbd_{}_dataset_config.json'.format('test')

                if mode_tag in ['segmentation', 'classification']:
                    allsplit_dir = os.path.join(self.root, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
                    split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
                    split_train = split['alltrain']
                    split_train = [f.replace('/n/fs/sun3d/data/SUNRGBD', self.root + '/train') for f in split_train]
                    split_train = [f[:-1] if f[-1] == '/' else f for f in split_train]
                    print(len(split_train), split_train)
                    test_split, train_split = [], []
                    
                    for f in self.dirs:
                        if f[0] in split_train:
                            #print(f)
                            #print(self.names)
                            train_split.append(f[0])
                            self.names[str(f[1])]['samples'].append(f[0])
                        else:
                            test_split.append(f[0])

                    self.dirs = train_split
                    self.test_dirs = test_split
                    #print(len(self.dirs), len(self.test_dirs))
                    #input()

                    with open(os.path.join(root, test_name), 'w') as f:
                        json.dump({'names': self.names,
                                   'dirs': self.test_dirs}, f)
                    with open(os.path.join(root, valid_name), 'w') as f:
                        json.dump({'names': self.names,
                                   'dirs': self.test_dirs}, f)
                    with open(os.path.join(root, train_name), 'w') as f:
                        json.dump({'names': self.names,
                                   'dirs': self.dirs}, f)

                else:
                    random.shuffle(self.dirs)
                    if p_eval < 1:
                        n = int(len(self.dirs) * p_eval)
                    else:
                        n = int(p_eval)
                        assert n > 0

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

        if self.with_scene_cls:
            self.class_names = self.scene_classes
        else:
            self.class_names = [v['name'] for v in self.names.values()]
        self.num_classes = len(self.class_names)

        #todo here
        #for key, v in self.names.items():
        #    del_index = []
        #    print(v)
        #    for i, p in enumerate(v['samples']):
        #        if p not in self.dirs:
        #            del_index.append(i)
        #    for i in del_index[::-1]:
        #        del v['samples'][i]
        #    self.names[key] = v


        if self.balanced_sampling and self.mode == 'train' and self.sample_with_p:
            self.ps = np.array([1 - (len(v['samples']) / len(self.dirs)) for v in self.names.values()])
            self.ps = self.ps / np.sum(self.ps)
            for cls, p in zip(self.class_names, self.ps):
                print('cls {}, p: {}'.format(cls, p*100))
            #print(np.sum(self.ps))
            #print(self.ps)
            if np.sum(self.ps) != 1.0:
                idx = np.argmax(self.ps)
                self.ps[idx] += 1 - np.sum(self.ps)
        else:
            self.ps = None

    def __getitem__(self, index):

        if self.balanced_sampling and self.mode == 'train':
            if self.ps is not None:
                cls_key = str(int(np.random.choice(list(self.names.keys()), size=1, p=self.ps)))
            else:
                cls_key = str(int(np.random.choice(list(self.names.keys()), size=1)))
            
            path_index = np.random.choice(list(range(len(self.names[cls_key]['samples']))), size=1)
            path = self.names[cls_key]['samples'][int(path_index)]
        else:
            path = self.dirs[index]

        sample = self.load_sample(path)
        if self.transforms is not None:
            s = self.transforms(sample)
            sample = {}
            for key, v in s.items():
                if key in ['img', 'mask', 'depth', 'x']:
                    sample[key] = v.float()
                elif key in ['ori_img', 'ori_depth', 'cls']:
                    sample[key] = v

            if self.with_scene_cls:
                if self.with_depth:
                    sample = ({'x': sample['x'], 'depth': sample['depth']}, sample['cls'])
                else:
                    sample = ({'x': sample['x']}, sample['cls'])
        
        #print(sample[0].keys())
        return sample


    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path, extr=None):
        sample = {}
        # read RGB information to numpy array
        img_path = os.path.join(path, 'image')
        img_path = os.path.join(img_path, listdir(img_path)[0])
        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')
        if self.with_pc or self.with_depth:
            # read depth information to numpy array
            depth_path = os.path.join(path, self.depth_mode)
            depth_path = os.path.join(depth_path, listdir(depth_path)[0])
            with open(depth_path, 'rb') as f:
                sample['depth'] = torch.as_tensor(np.array(Image.open(f), dtype=np.float32), dtype=torch.float32)
                #sample['depth'] = np.array(Image.open(f)).astype(np.float32)
            sample['depth_scale'] = {
                'depth': 0.25,
                'depth_gen': 1
            }[self.depth_mode]
            #print(sample['depth_scale'], self.depth_mode)
            if self.depth_mul is not None:
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul
                sample['depth_mul'] = self.depth_mul
            
            #print(sample['depth_scale'], self.depth_mode)


        if self.with_mask:
            mask_path = os.path.join(path, 'seg_mask', 'seg_mask.png')
            sample['mask'] = torch.from_numpy(np.array(Image.open(mask_path), dtype=np.uint8))
            #print(sample['mask'].shape)

        if self.with_intr or self.with_scene_cls:
            with open(os.path.join(path, 'intrinsics.txt'), 'r') as file:
                rows = file.read().strip().split('\n')
            intr = np.array([[float(v) for v in row.split(' ')] for row in rows]).reshape((3, 3))
            cx, cy, fx, fy = intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
            sample['focal_length'] = max(fx, fy)
            sample['intr'] = [cx, cy, fx, fy]
            #print('focal_length', sample['focal_length'])
        #else:
        #    print('intr', self.with_intr, self.with_scene_cls)

        if self.with_scene_cls:
            with open(os.path.join(path, 'scene.txt')) as f:
                cls = f.readlines()[0]
                if cls not in self.scene_classes:
                    cls = 'others'

                sample['cls'] = int(self.scene_classes.index(cls))

                if self.mode == 'train':
                    cam_type = path.split('/')[-3]
                    if cam_type not in self.sensor_to_params and '/xtion/' in path:
                        cam_type = 'xtion'           
                else:
                    cam_type = self.focal_dict.get(str(sample.get('focal_length')))

                sample['baseline'] = self.sensor_to_params.get(cam_type, {'baseline': None})['baseline'] 
                sample['cam_type'] = cam_type

                if cam_type is None or sample['baseline'] is None:
                    print('cam_type', cam_type)
                    print('path', path)
                    print('baseline', sample['baseline'])
                    print('focal_length', sample['focal_length'])

        if self.with_pc:
            cx, cy, fx, fy = sample['intr']
            w, h = sample['img'].size
            xmap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(w), axis=0), h, axis=0), dtype=torch.float32)
            ymap = torch.as_tensor(np.repeat(np.expand_dims(np.arange(h), axis=1), w, axis=1), dtype=torch.float32)

            pt2 = (torch.as_tensor(sample['depth'], dtype=torch.float32) * sample['depth_scale']) / 1000  # mm to meter
            pt0 = (xmap - cx) * pt2 / fx
            pt1 = (ymap - cy) * pt2 / fy
            sample['pc'] = torch.stack((pt0, pt1, pt2), axis=0)

        return sample

    def plot_sample(self, sample):
        plt.subplot(2,3,1)
        plt.imshow(sample['img'])
        w, h = sample['img'].size
        print(sample['img'].size, w/h)
        if 'depth' in sample:
            depth = sample['depth']
            if 'depth_scale' in sample:
                depth = depth * sample['depth_scale']
            
            if not sample.get('in_meter'):
                depth = depth / 1000
                
            plt.subplot(2,3,2)
            plt.imshow(depth)
        if 'mask' in sample:
            plt.subplot(2,3,3)
            plt.imshow(sample['mask'])

        if 'pc' in sample:
            plt.subplot(2, 3, 4)
            plt.imshow(np.abs(sample['pc'][:, :, 0]))
            plt.subplot(2, 3, 5)
            plt.imshow(np.abs(sample['pc'][:, :, 1]))
            plt.subplot(2, 3, 6)
            plt.imshow(sample['pc'][:, :, 2])
        plt.show()

def get_mask(data_dir=None):
    import h5py
    import mat73
    if data_dir is None:
        data_dir = '/path/to/SUNRGB-D'
    SUNRGBDMeta_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
    SUNRGBD2Dseg_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
    SUNRGBD2Dseg = mat73.loadmat(SUNRGBD2Dseg_dir)
    SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                   struct_as_record=False)['SUNRGBDMeta']
    seglabel = SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

    for i, meta in enumerate(SUNRGBDMeta):
        meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
        real_dir = meta_dir.replace('/n/fs/sun3d/data/SUNRGBD', data_dir+'/train')

        if os.path.exists(real_dir):
            label_path = os.path.join(real_dir, 'seg_mask')
            mask_path = os.path.join(label_path, 'seg_mask.png')
            if not os.path.exists(label_path):
                os.makedirs(label_path)

            if not os.path.exists(mask_path):
                mask = Image.fromarray(np.array(seglabel[i], dtype=np.uint8))
                mask.save(mask_path)

            #mask = Image.fromarray(np.array(seglabel[i], dtype=np.uint8))
        #plt.imshow(mask)
        #plt.show()
        #input()




if __name__ == '__main__':
    #root = './data/SunRGBD/SUNRGBD'
    #allsplit_dir = os.path.join(root, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
    #split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
    #print(split.keys())
    #split_train = split['alltrain']
    #split_train = [f.replace('/n/fs/sun3d/data/SUNRGBD', root + '/train') for f in split_train]
    #split_train = [f[:-1] if f[-1] == '/' else f for f in split_train]
    #print(split_train)

    #print([f for f in split_train  if f[-1] != '/'])

    #print(split_train[-2:])
    #'./data/SunRGBD/SUNRGBD/train/xtion/sun3ddata/mit_76_studyroom/76-1studyroom1/0000003-000000067032'

    ds = SUNRGBD_Dataset(None, mode='train', with_mask=False, mode_tag='classification', with_scene_cls=True,
                         balanced_sampling=True, sample_with_p=False, load_from_checkpoint=False)

    #ds_valid = SUNRGBD_Dataset(None, mode='valid', with_mask=False, mode_tag='classification', with_scene_cls=True,
    #                     balanced_sampling=False, sample_with_p=False)

    ds_test = SUNRGBD_Dataset(None, 
        mode='test', with_mask=False, mode_tag='classification', with_scene_cls=True,
                              balanced_sampling=False, sample_with_p=False, load_from_checkpoint=True)

    modes = ['train', 'valid', 'test']
    with_depth = True
    rgb_only = False
    transforms, valid_transforms = None, None
    datasets = {
            mode: SUNRGBD_Dataset(transforms=transforms if mode == 'train' else valid_transforms,
                                  with_depth=with_depth if not rgb_only else False,
                                  with_mask=False,
                                  with_scene_cls=True,
                                  mode=mode,
                                  mode_tag='classification',
                                  balanced_sampling=False if mode == 'train' else False, sample_with_p=False, load_from_checkpoint=True)
            for mode in modes
        }

    #ds = datasets['train']
    #ds_test = datasets['test']

    print(len(ds.dirs), ds.dirs[-1])
    print(len(ds_test.dirs), ds_test.dirs[-1])
    print(len(ds.class_names), ds.class_names)
    print(ds.names.keys())
    # input()
    classes = {}
    classes_test = {}
    #indexes = np.random.permutation(len(ds))
    indexes = range(len(ds))
    print(ds.names)
    for i in indexes:
        print('i', i)
        
        sample = ds.__getitem__(i)
        #if i == 100:
        #    break
        if ds.with_scene_cls:
            cls = sample['cls']
            if cls not in classes:
                classes[cls] = 1
                print('################# {} #################'.format(len(classes)))
                for key, v in classes.items():
                    print(key, v)
            else:
                classes[cls] += 1

        else:
            ds.plot_sample(sample)

    print('############################')
    print('test')

    print('############################')

    indexes = range(len(ds_test))
    for i in indexes:
        sample = ds_test.__getitem__(i)
        # print('i', i)
        # if i == 100:
        #    break
        if ds_test.with_scene_cls:
            cls = sample['cls']
            # print(cls, type(cls))
            # input()
            if cls not in classes_test:
                classes_test[cls] = 1
                print('################# {} #################'.format(len(classes_test)))
                for key, v in classes_test.items():
                    print(key, v)
            else:
                classes_test[cls] += 1

        else:
            ds.plot_sample(sample)

    print('############## train ##############')
    print('total', sum(list(classes.values())))
    for key, v in classes.items():
        print(key, ds.class_names[key], v, (100 * v / len(ds)) / 92 * 100)


    print('############## test ##############')
    print('total', sum(list(classes_test.values())))
    for key, v in classes_test.items():
        print(key, ds.class_names[key], v, (100 * v / len(ds_test)) / 92 * 100)


    print('########### both #################')
    print('total', sum(list(classes_test.values())) + sum(list(classes.values())))
    for key, v in classes.items():
        v2 = classes_test[key]
        print(key, ds.class_names[key], v+v2, (100 * v / (len(ds) + len(ds_test))) / 92 * 100)

