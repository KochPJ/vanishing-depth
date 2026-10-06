from typing import Any
import os
from torchvision.datasets import ImageNet
from PIL import Image
import torch
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageNetRGBD():
    
    def __init__(self, root, split, transform=None, rgb_only=False):
        
        self.rgb_only = rgb_only
        self.root = root
        self.split = split
        self.transform = transform

        self.samples = []

        self.cls_names = list(sorted(os.listdir(os.path.join(self.root, self.split))))
        assert len(self.cls_names) == 1000
        
        for cls_id, name in enumerate(self.cls_names):
            for file in os.listdir(os.path.join(self.root, self.split, name)):
                if '_depth.png' in file:
                    continue
                elif '.JPEG' in file:
                    self.samples.append((os.path.join(self.root, self.split, name, file), cls_id))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        for _ in range(10):
            try:
                path, target = self.samples[index]
                #print('here path', path)
                sample = {
                    'img': self.loader(path)
                }

                #print(np.mean(np.array(sample['img'])))
                #import matplotlib.pyplot as plt
                #plt.imshwo(sample['img'])
                #plt.show()

                if not self.rgb_only:
                    sample['depth'] = self.load_depth(path)
                    sample['depth_scale'] = 1

                if self.transform is not None:
                    sample = self.transform(sample)

                if not self.rgb_only:
                    sample = ({'x': sample.get('x', sample.get('img')), 'depth': sample['depth']}, target)
                else:
                    sample = ({'x': sample.get('x', sample.get('img'))}, target)
                    
                return sample
            except Exception as e:
                print(index, 'failed: ', e)
                if index < 20:
                    index += 1
                if index > 50:
                    index = index - 1
                else:
                    index += 1


    def load_depth(self, path):
        depth_path = path.replace('.JPEG', '_depth.png')

        #depth = None
        #print('pp', depth_path, os.path.exists(depth_path))
        #input()
        with open(depth_path, 'rb') as f:
            depth = torch.as_tensor(np.array(Image.open(f), dtype=np.float32), dtype=torch.float32)
        return depth

    def loader(self, path):
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        return image

    def __len__(self) -> int:
        return len(self.samples)



if __name__ == '__main__':
    ds = ImageNetRGBD('./data/imagenet', 'train')
    index = list(range(len(ds)))
    print('data set len', len(ds))
    #random.shuffle(index)
    import matplotlib.pyplot as plt
    for i in range(len(ds)):
        sample, label = ds.__getitem__(i)
        plt.subplot(1,2,1)
        plt.imshow(sample['x'])
        plt.title(label)
        plt.subplot(1,2,2)
        plt.imshow(sample['depth'])
        plt.show()


