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
from pathlib import Path
import struct


class BlendedMVS(data.Dataset):
    def __init__(self, transforms=None, root=None, with_intr=False, with_depth=True, mode_tag=None,
                 with_mask=False, load_from_checkpoint=True,  depth_mul=1, with_bbox=False):
        if root is None:
            root = './data/BlendedMVS++'

        self.me = 'BlendedMVS'

        self.root = root
        self.transforms = transforms
        self.with_intr = with_intr
        self.with_depth = with_depth
        self.with_mask = with_mask
        self.names = {}
        self.depth_mul = depth_mul
        self.with_bbox = with_bbox

        
        file_name = '{}_dataset_config.json'.format(self.me)

        load_path = os.path.join(self.root, file_name)
        cp = None
        if load_from_checkpoint:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    cp = json.load(f)

        self.dirs = []

        if cp is None:
            cls_counter = 1
            for scan in os.listdir(self.root):
                #print(scan, os.path.exists(os.path.join(self.root, scan, 'cams')), os.path.join(self.root, scan, 'cams'))
                if not os.path.exists(os.path.join(self.root, scan, 'cams')):
                    continue
                print(self.root, scan)
                    
                for cam in os.listdir(os.path.join(self.root, scan, 'cams')):
                    if 'pair' in cam:
                        continue
                    #print(cam)
                    #if '.txt' in cam:
                    #    continue                    
                    index = cam.split('_')[0]

                    if '.txt' in index:
                        continue

                    
                    self.dirs.append([
                        os.path.join(self.root, scan, 'cams', cam),
                        os.path.join(self.root, scan, 'blended_images', index + '.jpg'),
                        os.path.join(self.root, scan, 'rendered_depth_maps', index + '.pfm')
                    ])
                
            
            with open(load_path, 'w') as f:
                json.dump({'names': self.names,
                            'dirs': self.dirs}, f)
        else:
            self.names = cp['names']
            self.dirs = cp['dirs']

        self.class_names = [self.names[key]['name'] for key in sorted(list(self.names.keys()))]
        self.num_classes = len(self.class_names)

    def __getitem__(self, index):
        path = self.dirs[index]
        sample = self.load_sample(path)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path):
        sample = {}
        # read RGB information to numpy array
        cam_path, img_path, depth_path = path
  

        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        if self.with_intr or self.with_depth:
            
            sample['depth_scale'] = 1000
            if self.depth_mul is not None:
                sample['depth_scale'] = sample['depth_scale'] / self.depth_mul
                sample['depth_mul'] = self.depth_mul

                  
            with open(cam_path) as f:
                cam = [line.rstrip() for line in f][7:10]
                intr = np.array([[float(v) for v in c.split(' ')] for c in cam]).flatten().reshape((3, 3))
                        
            cx, cy, fx, fy = intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
            sample['intr'] = [cx, cy, fx, fy]

        if self.with_depth:
            # read depth information to numpy array
            #with open(depth_path, 'rb') as f:
                #sample['depth'] = torch.as_tensor(np.array(Image.open(f), dtype=np.float32), dtype=torch.float32)
                #sample['depth'] = np.array(Image.open(f)).astype(np.float32)
            sample['depth'] = self.read_pfm(depth_path)

       
        return sample

    def load_ann(self, path):
        with open(os.path.join(path, 'scene_gt.json')) as f:
            data = json.load(f)
        return data
        
    def read_pfm(self, filename):
        with Path(filename).open('rb') as pfm_file:

            line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
            assert line1 in ('PF', 'Pf')
            
            channels = 3 if "PF" in line1 else 1
            width, height = (int(s) for s in line2.split())
            scale_endianess = float(line3)
            bigendian = scale_endianess > 0
            scale = abs(scale_endianess)

            buffer = pfm_file.read()
            samples = width * height * channels
            assert len(buffer) == samples * 4
            
            fmt = f'{"<>"[bigendian]}{samples}f'
            decoded = struct.unpack(fmt, buffer)
            shape = (height, width, 3) if channels == 3 else (height, width)
            depth = torch.from_numpy(np.array(np.flipud(np.reshape(decoded, shape)), dtype=np.float32) * scale)
            

            return depth

    def load_pfm(self, file):
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = file.readline().decode('UTF-8').rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        # scale = float(file.readline().rstrip())
        scale = float((file.readline()).decode('UTF-8').rstrip())
        if scale < 0: # little-endian
            data_type = '<f'
        else:
            data_type = '>f' # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
        return data


    def plot_sample(self, sample):
        plt.subplot(2, 3, 1)
        plt.imshow(sample['img'])
        if 'depth' in sample:
            d = sample['depth']
            if 'depth_scale' in sample:
                d = d * sample['depth_scale']
                d = d / 1000
                
            plt.subplot(2, 3, 2)
            plt.imshow(d)
        if 'mask' in sample:
            plt.subplot(2, 3, 3)
            plt.imshow(sample['mask'])
        if 'pc' in sample:
            plt.subplot(2, 3, 4)
            plt.imshow(np.abs(sample['pc'][0, :, :]))
            plt.subplot(2, 3, 5)
            plt.imshow(np.abs(sample['pc'][1, :, :]))
            plt.subplot(2, 3, 6)
            plt.imshow(sample['pc'][2, :, :])

        print('show')
        plt.show()

if __name__ == '__main__':
    ds = BlendedMVS(None, with_intr=True, with_depth=True, root='/mnt/wimi/publicdata/BlendedMVS++', load_from_checkpoint=True)

    print(len(ds))
    print('names', ds.names)
    #indexes = np.random.permutation(len(ds))
    indexes = range(len(ds))
    #for i in indexes:
    #    print(i)
    #    sample = ds.__getitem__(i)
    #    ds.plot_sample(sample)

    

    mean = []
    mean_gt = []
    std = []
    std_gt = []
    zeros = []
    zeros_gt = []
    
    for i in range(len(ds)):
        try:
            sample = ds.__getitem__(i)
        except:
            continue

        #print('{} / {} | {}'.format(i+1, len(ds), sample['depth'].shape))
        #print('intr', sample.get('intr'))
        #ds.plot_sample(sample)
        d = sample['depth'] * sample['depth_scale'] / 1000
        mean.append(torch.mean(d[d > 0]))
        std.append(torch.std(d[d > 0]))
        zeros.append( 1 - (d.numpy() > 0) / np.prod(d.numpy().shape))

        if 'gt_depth' in sample:
            d = sample['gt_depth'] * sample['depth_scale'] / 1000
            mean_gt.append(torch.mean(d[d > 0]))
            std_gt.append(torch.std(d[d > 0]))
            zeros_gt.append( 1 - (d.numpy() > 0) / np.prod(d.numpy().shape))

        
        if i%1000 == 0:
            print('{} / {} | {}, mean: {}, std: {}, zeros: {}'.format(i+1, len(ds), sample['depth'].shape, np.mean(mean), np.mean(std), np.mean(zeros)))
            
    print('mean', np.mean(mean))
    print('std', np.mean(std))
    print('zeros', np.mean(zeros))
        
    if len(mean_gt) > 0:
        print('mean_gt', np.mean(mean_gt))
        print('std_gt', np.mean(std_gt))
        print('zeros_gt', np.mean(zeros_gt))
