import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import matplotlib.pyplot as plt


       
def plot_depth(d, msg=''):
    m = d > 0
    c = sum(m.flatten())
    p = np.prod([float(v) for v in d.shape])
    if isinstance(d, torch.Tensor):
        print('{} mean: {}, std: {}, min: {}, max: {}, p: {}%'.format(msg, float(torch.mean(d[m])), float(torch.std(d[m])), float(torch.min(d[m])), float(torch.max(d[m])), np.round(100*c/p, 3)))
    else:
        print('{} mean: {}, std: {}, min: {}, max: {}, p: {}%'.format(msg, float(np.mean(d[m])), float(np.std(d[m])), float(np.min(d[m])), float(np.max(d[m])), np.round(100*c/p,3)))



class VoidDataset(data.Dataset):
    def __init__(self, transforms=None, root=None, mode=None, auc_thresholds=None, use_depth_gt=False,
                 with_intr=False, void='void_150'):

        if root is None:
            root = './data/void/void_release/{}'.format(void)
        self.root = root
        self.transforms = transforms
        self.mode = mode
        self.auc_thresholds = auc_thresholds
        self.depth_threshold = 0.05
        self.use_depth_gt = use_depth_gt
        self.with_intr = with_intr
        self.void = void

        self.dirs = []

        with open(os.path.join(root, '{}_image.txt'.format(mode))) as f:
            lines = f.readlines()
            for l in lines:
                l = l.replace('\n', '')
                l = l.replace('{}/data/'.format(void), '')
                self.dirs.append(l)
            #print(lines)
        #input()
        self.root = os.path.join(self.root, 'data')

        #for folder in os.listdir(root):
        #    if os.path.isdir(os.path.join(self.root, folder)):
        #        if 'image' not in list(os.listdir(os.path.join(self.root, folder))):
        #            continue
        #        for img in os.listdir(os.path.join(self.root, folder, 'image')):
        #            self.dirs.append([folder, img])

    def __getitem__(self, index):
        path = self.dirs[index]
        sample = self.load_sample(path, index)
        sample['mode'] = self.mode
        sample['sample_index'] = index
        #print(sample.keys())
        if self.transforms is not None:
            sample = self.transforms(sample)
            #sample = {}
            for key, v in sample.items():
                if key in ['img', 'depth', 'gt_depth', 'ori_depth']:
                    sample[key] = v.float()
            
            if 'depth_scales' in sample and sample.get('depth_scales') is None:
                del sample['depth_scales']

        #print(sample.keys())
        return sample

    def __len__(self):
        return len(self.dirs)

    def load_sample(self, path, index):
        sample = {}
        folder, img = path.split('/image/')

        img_path = os.path.join(self.root, folder, 'image', img)
        with open(img_path, 'rb') as f:
            sample['img'] = Image.open(f).convert('RGB')

        depth_path = os.path.join(self.root, folder, 'ground_truth', img)
        with open(depth_path, 'rb') as f:
            depth = np.array(Image.open(f), dtype=np.float32)
            depth = depth / 256.0
            depth[depth <= 0] = 0.0

        
        #plot_depth(depth, 'void depth')

        #if self.mode == 'train':
        sparse_path = os.path.join(self.root, folder, 'sparse_depth', img)
        
        #else:
        #    sparse_path = os.path.join(self.root, folder, 'validity_map', img)

        with open(sparse_path, 'rb') as f:
            sparse_depth = np.array(Image.open(f), dtype=np.float32) 
            sparse_depth = sparse_depth / 256.0
            sparse_depth[sparse_depth <= 0] = 0.0

            #print(np.mean(sparse_depth[sparse_depth>0]), np.mean(depth[depth>0]), np.mean(np.abs(sparse_depth[sparse_depth>0] - depth[sparse_depth>0] )))
            #sparse_depth = np.zeros(depth.shape, dtype=np.float32)
            #sparse_depth[sparse_map > 0] = depth[sparse_map > 0]

        sample['gt_depth'] = torch.as_tensor(depth, dtype=torch.float32)

        if self.use_depth_gt:
            sample['depth'] = torch.as_tensor(depth, dtype=torch.float32)
        else:
            sample['depth'] = torch.as_tensor(sparse_depth, dtype=torch.float32)


        #plot_depth(sample['depth'], 'void depth out')
        #plot_depth(sample['gt_depth'], 'void gt_depth out')

        sample['depth_scale'] = 1000

        '''
        if self.auc_thresholds is not None and False:
            froot = os.path.join(self.root, folder, 'vd-auc')
            if not os.path.exists(froot):
                os.makedirs(froot)

            ffile = os.path.join(froot, img)
            if not os.path.exists(ffile):
                print('Creating file for auc {}'.format(index))
                w, h = sample['img'].size
                torch.manual_seed(index)
                mask = torch.rand((h, w, len(self.auc_thresholds)))
                for i, t in enumerate(self.auc_thresholds):
                    mask[:, :, i][mask[:, :, i] > t] = 0
                mask[mask != 0] = 1
                depth_auc = sample['gt_depth'].unsqueeze(-1).repeat(1, 1, len(self.auc_thresholds))
                depth_auc[mask == 0] = 0
                sample['depth_auc'] = depth_auc
                mask = mask.view(h, w * len(self.auc_thresholds))
                #print(mask.shape)
                mask = np.array(mask.numpy(), dtype=np.uint8)
                mask = Image.fromarray(mask)
                mask.save(os.path.join(ffile))
            else:
                mask = torch.from_numpy(np.array(Image.open(ffile)))
                w, h = sample['img'].size
                mask = mask.view(h, w, len(self.auc_thresholds))
                depth_auc = sample['gt_depth'].unsqueeze(-1).repeat(1, 1, len(self.auc_thresholds))
                depth_auc[mask == 0] = 0
                sample['depth_auc'] = depth_auc
            sample['depth_auc'] = sample['depth_auc'].permute(2, 0, 1)
        '''

        if self.with_intr:
            intr = []
            with open(os.path.join(self.root, folder, 'K.txt')) as f:
                lines = f.readlines()
                for l in lines:
                    l = l.replace('\n', '').split(' ')
                    for v in l:
                        intr.append(float(v))


            intr = np.array(intr).reshape((3, 3))
            cx, cy, fx, fy = intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]
            sample['intr'] = [cx, cy, fx, fy]

        return sample


    def plot_sample(self, sample):
        plt.subplot(2,3,1)
        plt.imshow(sample['img'])
        w, h = sample['img'].size
        for dkey in ['depth', 'gt_depth', 'depth_auc']:
            if dkey in sample:
                print(dkey, torch.mean(sample[dkey][sample[dkey] > 0]))

        plt.subplot(2,3,2)
        plt.imshow(sample['depth'] * sample['depth_scale'])
        a = np.sum(sample['depth'].numpy() > 0)
        plt.title('{} | {}'.format(a, np.round(a/ (w*h ) * 100 ), 6))
        plt.subplot(2,3,3)
        plt.imshow(sample['gt_depth'] * sample['depth_scale'])
        plt.subplot(2,3,4)
        if 'depth_auc' in sample:
            plt.imshow(sample['depth_auc'][0] * sample['depth_scale'])
            a = np.sum(sample['depth_auc'][0].numpy() > 0)
            plt.title('{} | {}'.format(a, np.round(a/ (w*h ) * 100 ), 4))
            plt.subplot(2,3,5)
            plt.imshow(sample['depth_auc'][10] * sample['depth_scale'])
            a = np.sum(sample['depth_auc'][10].numpy() > 0)
            plt.title('{} | {}'.format(a, np.round(a/ (w*h ) * 100 ), 4))
            plt.subplot(2,3,6)
            plt.imshow(sample['depth_auc'][19] * sample['depth_scale'])
            a = np.sum(sample['depth_auc'][19].numpy() > 0)
            plt.title('{} | {}'.format(a, np.round(a/ (w*h ) * 100 ), 4))
        plt.show()



if __name__ == '__main__':
    torch.manual_seed(42)
    thresholds = np.array(list(range(1, 100, 5))) / 100
    print(thresholds)
    print(len(thresholds))
    #thresholds = None
    ds = VoidDataset(mode='test', auc_thresholds=None, use_depth_gt=False, void='void_1500')
    index = list(range(len(ds)))
    print(len(ds))
    #random.shuffle(index)

    mean = []
    mean_gt = []
    std = []
    std_gt = []
    zeros = []
    zeros_gt = []
    
    for i in range(len(ds)):
        sample = ds.__getitem__(i)
        print('{} / {} | {}'.format(i+1, len(ds), sample['depth'].shape))
        #print('intr', sample.get('intr'))
        #ds.plot_sample(sample)
        mean.append(torch.mean(sample['depth'][sample['depth'] > 0]))
        std.append(torch.std(sample['depth'][sample['depth'] > 0]))
        zeros.append( 1 - (np.sum(sample['depth'].numpy() > 0) / np.prod(sample['depth'].numpy().shape)))

        if 'gt_depth' in sample:
            mean_gt.append(torch.mean(sample['gt_depth'][sample['gt_depth'] > 0]))
            std_gt.append(torch.std(sample['gt_depth'][sample['gt_depth'] > 0]))
            zeros_gt.append( 1 - (np.sum(sample['gt_depth'].numpy() > 0) / np.prod(sample['gt_depth'].numpy().shape)))
    
    print('mean', np.mean(mean))
    print('std', np.mean(std))
    print('zeros', np.mean(zeros))
        
    print('mean_gt', np.mean(mean_gt))
    print('std_gt', np.mean(std_gt))
    print('zeros_gt', np.mean(zeros_gt))

    '''
    means = []
    stds = []
    zeros = []
    counts = []
    for i in index:
        print('{}/{} | {}%'.format(i+1, len(index), np.round(((i+1) / len(index)) * 100, 4) ))
        sample = ds.__getitem__(i)
        #ds.plot_sample(sample)
        m = sample['gt_depth'] > 0
        means.append(float(torch.mean(sample['gt_depth'])))
        stds.append(float(torch.std(sample['gt_depth'])))
        h, w = m.shape
        #print(h, w)
        zeros.append( float(np.sum(sample['gt_depth'].numpy() > 0)) / (h*w))
        counts.append(float(torch.sum(sample['depth'] > 0) / torch.sum(sample['gt_depth'] > 0) * 100))

    mean = np.mean(means)
    std = np.mean(stds)
    ps = np.mean(zeros)
    counts = np.mean(counts)
    print('mean', mean)
    print('std', std)
    print('ps', ps)
    print('counts', counts)

        #ds.plot_sample(sample)

    '''







