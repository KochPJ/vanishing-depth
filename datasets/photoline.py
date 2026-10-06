import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import transforms3d as t3d


class Photoline(data.Dataset):
<<<<<<< HEAD

    def __init__(self, transforms=None, root=None, mode='train',
<<<<<<< HEAD
<<<<<<< HEAD
                 n_eval=500, sample_per_epoch=1):
=======
=======
                 n_eval=500, sample_per_epoch=1):
=======
    def __init__(self, transforms=None, root=None, mode='train',
>>>>>>> pose
                 n_eval=10, sample_per_epoch=1):
>>>>>>> main
        if root is None:
            root = './data/photoline'

        self.symmetries_file = 'photoline_symmetries.json'
        self.models_dir = 'photoline_models'
        self.root = root
        self.transforms = transforms
        self.sample_per_epoch = sample_per_epoch

        self.train_dirs = []
        self.valid_dirs = []
        self.test_dirs = []
        self.photoline_names= ['left', 'right']
        self.names = {str(i+1).zfill(4): {'name': v} for i, v in enumerate(self.photoline_names)}
<<<<<<< HEAD
<<<<<<< HEAD
=======
        valid_rechts = True
        valid_links_weiß = False
        valid_links_schwarz = False
>>>>>>> pose
        
<<<<<<< HEAD
        path = os.path.join(self.root, 'img_links')
        for i, img in enumerate(sorted(list(os.listdir(path)))):
            if i < n_eval:
                self.valid_dirs.append([path, img, 1])
            else:
<<<<<<< HEAD
=======
        valid_rechts = True
        valid_links_weiß = False
        valid_links_schwarz = False
=======
                self.train_dirs.append([path, img, 1])

>>>>>>> pose
        
        path = os.path.join(self.root, 'img_rechts')
        for i, img in enumerate(sorted(list(os.listdir(path)))):
            if i < n_eval:
                self.valid_dirs.append([path, img, 2])
            else:
                self.train_dirs.append([path, img, 2])


        path = os.path.join(self.root, 'img_test')
        for i, fdir in enumerate(sorted(list(os.listdir(path)))):            
            path_test = os.path.join(path, fdir)
            imgs = os.listdir(path_test)
=======

        train_rechts = True
        train_links_weiß = False
        train_links_schwarz = False
        shuffle = False


        path = os.path.join(self.root, 'img_links_white')
        data = sorted(list(os.listdir(path)))
        if shuffle:
            random.shuffle(data)

        for i, img in enumerate(data):
            if i < n_eval and valid_links_weiß:
                self.valid_dirs.append([path, img, 1])
            elif train_links_weiß:
                self.train_dirs.append([path, img, 1])


        path = os.path.join(self.root, 'img_links_schwarz')
        data = sorted(list(os.listdir(path)))
        if shuffle:
            random.shuffle(data)

        for i, img in enumerate(data):
            if i < n_eval and valid_links_schwarz:
                self.valid_dirs.append([path, img, 1])
            elif train_links_schwarz:
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> pose
                self.train_dirs.append([path, img, 1])

        
        path = os.path.join(self.root, 'img_rechts')
<<<<<<< HEAD
<<<<<<< HEAD
        for i, img in enumerate(sorted(list(os.listdir(path)))):
            if i < n_eval:
                self.valid_dirs.append([path, img, 2])
            else:
=======
=======
>>>>>>> pose
        data = sorted(list(os.listdir(path)))

        if shuffle:
            random.shuffle(data)

        for i, img in enumerate(data):
            if i < n_eval and valid_rechts:
                self.valid_dirs.append([path, img, 2])
            elif train_rechts:
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> pose
                self.train_dirs.append([path, img, 2])


        path = os.path.join(self.root, 'img_test')
        for i, fdir in enumerate(sorted(list(os.listdir(path)))):            
            path_test = os.path.join(path, fdir)
<<<<<<< HEAD
<<<<<<< HEAD
            imgs = os.listdir(path_test)
=======
=======
>>>>>>> pose
            imgs = sorted(list(os.listdir(path_test)))
>>>>>>> main
            for img in imgs:
                if 'rechts' in fdir:
                    self.test_dirs.append([path_test, img, 2])
                elif 'links' in fdir:
                    self.test_dirs.append([path_test, img, 1])
                else:
                    raise Valueerror()

<<<<<<< HEAD
        if mode == 'train'
            self.dirs = self.train_dirs 
        
        elif mode == 'valid'
            self.dirs = self.valid_dirs + self.test_dirs

        elif mode == 'test'
            self.dirs = self.test_dirs 
        else:
            raise Valueerror()

=======
        

        if mode == 'train':
            self.dirs = self.train_dirs
        
        elif mode == 'valid':
            self.dirs = self.valid_dirs

        elif mode == 'test':
            self.dirs = self.valid_dirs
        else:
            raise Valueerror()

        print({'{}'.format(i): os.path.join(path_test, img) for i, (path_test, img, _) in enumerate(self.valid_dirs)})
        #input()
            

>>>>>>> main
        self.class_names = self.photoline_names
        self.num_classes = len(self.class_names)

    def __getitem__(self, index):
        
        if index > len(self.dirs):
            print('here {} {} {}'.format(index, index%len(self.dirs), len(self.dirs)))
            index = index%len(self.dirs)

        info = self.dirs[index]
        
        sample = self.load_sample(info)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.dirs * self.sample_per_epoch)

    def load_sample(self, info):
        sample = {}
        # read RGB information to numpy array
        
        path, img_name, cls_id = info

        
        img_path = os.path.join(path, img_name)
        
        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')
<<<<<<< HEAD

<<<<<<< HEAD
<<<<<<< HEAD
        sample['bboxes'] = [[0.5, 0.5, 1.0, 1.0]]
        sample['cls'] = [int(cls_id)]
=======
=======
        sample['bboxes'] = [[0.5, 0.5, 1.0, 1.0]]
        sample['cls'] = [int(cls_id)]
=======

>>>>>>> pose
        w, h = sample['img'].size

        if 'img_test' in img_path:
            
            sample['img'] = sample['img'].crop((0, 0, int(h*1.77777777778), h))
            #print(w, h, sample['img'].size)
        
        sample['img'] = sample['img'].resize((1024//2, 576//2))
        
        #print(w, h, sample['img'].size)

        sample['bboxes'] = torch.from_numpy(np.array([[0.5, 0.5, 1.0, 1.0]], dtype=np.float32))
        sample['cls'] =  torch.from_numpy(np.array([int(cls_id)], dtype=np.float32))
>>>>>>> main
        
        pose_path = img_path.replace('/img_', '/pose_').replace('.jpg', '.npy')
        if os.path.exists(pose_path):
            pose = np.load(pose_path)
        else:
            print('pose does not exist', pose_path)
<<<<<<< HEAD
            pose = np.identiy(4)
=======
            pose = np.identity(4, dtype=np.float32)
>>>>>>> main

        #print(pose, pose.shape)
        sample['quat'] = torch.from_numpy(np.array([t3d.quaternions.mat2quat(pose[:3, :3])], dtype=np.float32))
        sample['rot'] = torch.from_numpy(np.array([pose[:3, :3]], dtype=np.float32))
        sample['pos'] = torch.from_numpy(np.array([pose[:3, 3]], dtype=np.float32))
<<<<<<< HEAD
<<<<<<< HEAD
=======
        sample['intr'] = torch.from_numpy(np.array([490/2, 290/2, 695/2, 691/2], dtype=np.float32))
        
>>>>>>> main
=======
        return sample
=======
        sample['intr'] = torch.from_numpy(np.array([490/2, 290/2, 695/2, 691/2], dtype=np.float32))
        
>>>>>>> pose
        
        return sample

>>>>>>> main

    def plot_sample(self, sample):
        
        plt.imshow(sample['img'])
<<<<<<< HEAD
<<<<<<< HEAD
        print(sample['quat'], sample['rot'], sample['pos'], sample['bboxes'], sample['cls'])
        plt.show()

if __name__ == '__main__':
    ds = Photoline(mode='train')
=======
=======
>>>>>>> pose
        print(sample['quat'], sample['rot'], sample['pos'], sample['bboxes'], sample['cls'], sample['intr'])
        plt.show()

if __name__ == '__main__':
<<<<<<< HEAD
    ds = Photoline(mode='train')
=======
    ds = Photoline(mode='test')
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> pose

>>>>>>> main
    print(len(ds))
    print(ds.names)
    indexes = np.random.permutation(len(ds))
    indexes = range(len(ds))
    for i in indexes:
        sample = ds.__getitem__(i)
        #print(sample['cls'])
        #print(sample['bboxes'])
        ds.plot_sample(sample)
