
'https://github.com/visipedia/inat_comp/blob/master/2018/README.md'
'https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py'

import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
import random
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')

def rgbd_loader(path):
    
    depth_path = path.replace('.JPEG', '_depth.png')
    sample = {
        'img': Image.open(path).convert('RGB'),
        'depth': torch.as_tensor(np.array(Image.open(depth_path), dtype=np.float32), dtype=torch.float32)
    }
    return sample

def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic


class INaturalist(data.Dataset):
    def __init__(self, root, ann_file, with_depth=False, transforms=None):
        
        if '.json' not in ann_file:
            ann_file = ann_file + '.json'

        if root not in ann_file:
            ann_file = os.path.join(root, ann_file)

        # load annotations
        print('Loading annotations from: {}'.format(ann_file))
        print('Loading depth {}'.format(with_depth))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]
        self.with_depth = with_depth

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # print out some stats
        print '\t' + str(len(self.imgs)) + ' images'
        print '\t' + str(len(set(self.classes))) + ' classes'

        self.root = root
        
        self.loader = rgbd_loader if with_depth else default_loader 

        # augmentation params
        #self.im_size = [299, 299]  # can change this to train on higher res
        #self.mu_data = [0.485, 0.456, 0.406]
        #self.std_data = [0.229, 0.224, 0.225]
        #self.brightness = 0.4
        #self.contrast = 0.4
        #self.saturation = 0.4
        #self.hue = 0.25

        # augmentations

        self.transforms = transforms
        #self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        #self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        #self.flip_aug = transforms.RandomHorizontalFlip()
        #self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        #self.tensor_aug = transforms.ToTensor()
        #self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        im_id = self.ids[index]
        img = self.loader(path)
        species_id = self.classes[index]
        tax_ids = self.classes_taxonomic[species_id]

        #if self.is_train:
        #    img = self.scale_aug(img)
        #    img = self.flip_aug(img)
        #    img = self.color_aug(img)
        #else:
        #    img = self.center_crop(img)

        #img = self.tensor_aug(img)
        #img = self.norm_aug(img)
        img = self.transforms(img)

        return img, species_id # im_id, species_id, tax_ids

    def __len__(self):
        return len(self.imgs)