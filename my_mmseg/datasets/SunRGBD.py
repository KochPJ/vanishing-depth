from mmseg.registry import DATASETS
from mmengine.registry import init_default_scope
init_default_scope('mmseg')

import os
print(os.getcwd())
import sys
sys.path.append('/home/chowanki/git/vanishing-depth-self-supervised')
from tqdm import tqdm
from collections import defaultdict
from mmseg.registry import DATASETS
from my_mmseg.datasets.base_Sunrgbd_dataset import BaseSUNRGBDSegDataset
from my_mmseg.utils.loading import *
from my_mmseg.utils.processing import *
from my_mmseg.utils.transforms import ClampLabels
from my_mmseg.utils.formatting import *
import matplotlib.pyplot as plt
import numpy as np

@DATASETS.register_module()
class SunRGBDDataset(BaseSUNRGBDSegDataset):
    #https://rgbd.cs.princeton.edu/supp.pdf
    
    METAINFO = dict(
        classes=(
                    'wall', 'floor', 'cabinet', 'bed', 'chair',
                    'sofa', 'table', 'door', 'window', 'bookshelf',
                    'picture', 'counter', 'blinds', 'desk', 'shelves',
                    'curtain', 'dresser', 'pillow', 'mirror',
                    'floor mat', 'clothes', 'ceiling', 'books',
                    'fridge', 'tv', 'paper', 'towel', 'shower curtain',
                    'box', 'whiteboard', 'person', 'night stand',
                    'toilet', 'sink', 'lamp', 'bathtub', 'bag'),
        
        palette=[   [119, 119, 119], [244, 243, 131],
                    [137, 28, 157], [150, 255, 255], [54, 114, 113],
                    [0, 0, 176], [255, 69, 0], [87, 112, 255], [0, 163, 33],
                    [255, 150, 255], [255, 180, 10], [101, 70, 86],
                    [38, 230, 0], [255, 120, 70], [117, 41, 121],
                    [150, 255, 0], [132, 0, 255], [24, 209, 255],
                    [191, 130, 35], [219, 200, 109], [154, 62, 86],
                    [255, 190, 190], [255, 0, 255], [192, 79, 212],
                    [152, 163, 55], [230, 230, 230], [53, 130, 64],
                    [155, 249, 152], [87, 64, 34], [214, 209, 175],
                    [170, 0, 59], [255, 0, 0], [193, 195, 234], [70, 72, 115],
                    [255, 255, 0], [52, 57, 131], [12, 83, 45]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='seg_mask.png',
                 depth_map_suffix='.png',
                 requires_depth=True,
                 requires_ann=True,
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, depth_map_suffix=depth_map_suffix,
            requires_depth=requires_depth, requires_ann=requires_ann, reduce_zero_label = reduce_zero_label, **kwargs)

            #wall, floor, cabinet, bed, chair,
            #sofa, table, door, window, bookshelf, 
            #picture, counter, blinds, desk, shelves, 
            #curtain, dresser, #pillow, mirror, 
            #floor mat, clothes, ceiling, books, 
            #fridge, tv, paper, towel, shower curtain, 
            #box, whiteboard, person, nightstand, 
            #toilet, sink, lamp, bathtub, bag)
        METAINFO = dict(
            classes=(
                        'wall', 'floor', 'cabinet', 'bed', 'chair',
                        'sofa', 'table', 'door', 'window', 'bookshelf',
                        'picture', 'counter', 'blinds', 'desk', 'shelves',
                        'curtain', 'dresser', 'pillow', 'mirror', 
                        'floor mat', 'clothes', 'ceiling', 'books', 
                        'fridge', 'tv', 'paper', 'towel', 'shower curtain',
                        'box', 'whiteboard', 'person', 'night stand', 
                        'toilet', 'sink', 'lamp', 'bathtub', 'bag'),
            
            palette=[   [119, 119, 119], [244, 243, 131],
                        [137, 28, 157], [150, 255, 255], [54, 114, 113],
                        [0, 0, 176], [255, 69, 0], [87, 112, 255], [0, 163, 33],
                        [255, 150, 255], [255, 180, 10], [101, 70, 86],
                        [38, 230, 0], [255, 120, 70], [117, 41, 121],
                        [150, 255, 0], [132, 0, 255], [24, 209, 255],
                        [191, 130, 35], [219, 200, 109], [154, 62, 86],
                        [255, 190, 190], [255, 0, 255], [192, 79, 212],
                        [152, 163, 55], [230, 230, 230], [53, 130, 64],
                        [155, 249, 152], [87, 64, 34], [214, 209, 175],
                        [170, 0, 59], [255, 0, 0], [193, 195, 234], [70, 72, 115],
                        [255, 255, 0], [52, 57, 131], [12, 83, 45]])
        print('METAINFO, palette', len(METAINFO['classes']), len(METAINFO['palette']))
        print('SUN-RGBD dataset lenth = {}, number of classes = {}'.format(len(self), len(self._metainfo['classes'])))
        #input()


def label_to_color(label_map, palette):
    color_map = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_map[label_map == label] = color
    return color_map

def detected_classes(label_map, class_names):
    unique_labels = np.unique(label_map)
    detected_classes = [class_names[label] for label in unique_labels if 0 <= label < len(class_names)]
    print(f'Detected classes: {detected_classes}')
    return detected_classes

def verify_grayscale_class_indices(grayscale_label_map, class_names):
    unique_values = np.unique(grayscale_label_map)
    print(f'Unique grayscale values: {unique_values}')
    valid_values = set(range(len(class_names)))
    invalid_values = [val for val in unique_values if val not in valid_values]
    if invalid_values:
        print(f'Invalid values: {invalid_values}')
    else:
        print('All grayscale values are valid')

def visualize_maps(img, grayscale_seg_map, color_label_map, palette):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title('Input Image')

    axes[1].imshow(grayscale_seg_map, cmap='gray')
    axes[1].set_title('Grayscale Seg Map')

    axes[2].imshow(color_label_map)
    axes[2].set_title('Colorized Seg Map')

    plt.show(block=False)

def display_legends(class_names, palette):
    fig, ax = plt.subplots(figsize=(10,20))
    normalized_colors = normalize_colors(palette)
    legend_entries =[plt.Line2D([0] ,[0], marker = 'o', color = 'w' , markerfacecolor = color, markersize = 10) for color in normalized_colors] 

    ax.legend(legend_entries , class_names , loc= 'best', bbox_to_anchor = (1,.5), fontsize='small', ncol=4)

    ax.axis('off')
    plt.show()

def viz_img(img_path, seg_map_path, classes, palette):
    # Load the images
    img = Image.open(img_path).convert('RGB')
    grayscale_seg_map = Image.open(seg_map_path).convert('L')

    # Convert images to numpy arrays
    img_np = np.array(img)
    grayscale_seg_map_np = np.array(grayscale_seg_map)

    # Check the shapes and types
    print(f'Image shape: {img_np.shape}, dtype: {img_np.dtype}')
    print(f'Grayscale seg map shape: {grayscale_seg_map_np.shape}, dtype: {grayscale_seg_map_np.dtype}')

    # Convert grayscale segmentation map to color
    color_label_map = label_to_color(grayscale_seg_map_np, palette)

    # Verify class indices
    verify_grayscale_class_indices(grayscale_seg_map_np, classes)

    # Detect classes
    detected_classes(grayscale_seg_map_np, classes)

    # Visualize the maps
    visualize_maps(img_np, grayscale_seg_map_np, color_label_map, palette)

    display_legends(classes, palette)




def normalize_colors(class_colors):
    return [[c/255.0 for c in color] for color in class_colors]


def visualize_sample(sample, class_colors, class_names):

    img = sample['inputs'][:3,:,:].permute(1,2,0).numpy()
    print('img shape in viz', img.shape)
    print('img dtype in Viz', img.dtype)

    img = img.astype(np.uint8)

    gt_seg_map = sample['data_samples'].gt_sem_seg.data.squeeze(0).numpy()
    print('gt_seg_map shape in viz', gt_seg_map.shape)
    depth_channels = sample['inputs'][3:, :,:].numpy()
    print('depth_channels shape in viz', depth_channels.shape)
    depth_map = depth_channels[0,:,:] 
    # depth_map = np.mean(depth_channels, axis=0)
    # depth_map = depth_channels.squeeze(0) # when using only loaddepth class and not DPE
    print('depth_map shape in viz ', depth_map.shape)

    # print('depth_map min in Viz', np.min(depth_map))
    # print('depth_map max in Viz', np.max(depth_map))
    
    
    verify_grayscale_class_indices(gt_seg_map, class_names)
    
    detected_classes(gt_seg_map, class_names)

    #depth_map_normalized = (depth_map - np.min(depth_map))/ (np.max(depth_map) - np.min(depth_map))
    depth_map_normalized = depth_map / np.max(depth_map)

    gt_color = np.zeros((gt_seg_map.shape[0], gt_seg_map.shape[1], 3), dtype = np.uint8)

    for class_id, color in enumerate(class_colors):
        mask = (gt_seg_map == class_id)
        gt_color[mask] = color 

    fig, ax = plt.subplots(1,3, figsize=(20,10))

    ax[0].imshow(img)
    ax[0].set_title('Input Image')
    ax[0].axis('off')
     
    ax[1].imshow(gt_color)
    ax[1].set_title('Ground Truth Segmentation')
    ax[1].axis('off')

    ax[2].imshow(depth_map, cmap= 'viridis')
    ax[2].set_title('Depth Map')
    ax[2].axis('off')

    normalized_colors = normalize_colors(class_colors)

    handles =[plt.Line2D([0] ,[0], marker = 'o', color = 'w' , markerfacecolor = color, markersize = 10) for color in normalized_colors]
    ax[1].legend(handles, class_names, loc= 'center left', bbox_to_anchor= (1.0,0.5), fontsize= 'small', ncol=1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    #data_root = '/home/chowanki/git/vanishing-depth-self-supervised/data/SunRGBD/SUNRGBD'
    
    data_root = '/mnt/data/publicdata/SunRGBD/SUNRGBD'
    # For RGBD
    data_prefix = dict(img_path='image', seg_map_path='seg_mask', depth_map_path='depth')

    #ann_file_path = data_root + '/sunrgbd_test_segmentation_dataset_config.json'
    #ann_file_path = None #data_root+'/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'
    
    ann_file_path_val = data_root + '/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'

    classes=[ 
                    'wall', 'floor', 'cabinet', 'bed', 'chair',
                    'sofa', 'table', 'door', 'window', 'bookshelf',
                    'picture', 'counter', 'blinds', 'desk', 'shelves',
                    'curtain', 'dresser', 'pillow', 'mirror',
                    'floor mat', 'clothes', 'ceiling', 'books',
                    'fridge', 'tv', 'paper', 'towel', 'shower curtain',
                    'box', 'whiteboard', 'person', 'night stand',
                    'toilet', 'sink', 'lamp', 'bathtub', 'bag'] 
    palette=[       [119, 119, 119], [244, 243, 131],
                    [137, 28, 157], [150, 255, 255], [54, 114, 113],
                    [0, 0, 176], [255, 69, 0], [87, 112, 255], [0, 163, 33],
                    [255, 150, 255], [255, 180, 10], [101, 70, 86],
                    [38, 230, 0], [255, 120, 70], [117, 41, 121],
                    [150, 255, 0], [132, 0, 255], [24, 209, 255],
                    [191, 130, 35], [219, 200, 109], [154, 62, 86],
                    [255, 190, 190], [255, 0, 255], [192, 79, 212],
                    [152, 163, 55], [230, 230, 230], [53, 130, 64],
                    [155, 249, 152], [87, 64, 34], [214, 209, 175],
                    [170, 0, 59], [255, 0, 0], [193, 195, 234], [70, 72, 115],
                    [255, 255, 0], [52, 57, 131], [12, 83, 45]]
    
    rescale_depth = 15
    depth_mul = 1.0
    depth_deleted = None
    random_depth = 0.05
    depth_offset = 0.0
    max_depth = 15 * depth_mul + depth_offset
    
    # For RGBD pipeline
    train_pipeline = [
        dict(type='LoadImageFromFile', to_float32= True),
        dict(type='LoadDepth', depthscale2meter=0.00025, norm_depth=0, div_nr=0, max_depth=max_depth, rescale_depth=rescale_depth, depth_mul=depth_mul, depth_deleted=depth_deleted, random_depth=random_depth, depth_offset=depth_offset),
        #dict(type='LoadAnnotations', reduce_zero_label = True),
        #dict(type='ResizeRGBD',scale=(1280, 960),keep_ratio=True),
        #dict(type='RandomCropRGBD', crop_size=(512, 1024), cat_max_ratio=0.75),
        #dict(type='RandomFlipRGBD', prob=0.5),
        #dict(type='NormalizeRGBD', mean=[123.7709, 116.7460, 104.0937], std=[68.5005, 66.6322, 70.3232], to_rgb= True),
        #dict(type='Printer'),
        #dict(type='DepthPositionalEncoding', depth_channels=32, encoder_hidden_dims=768),
        #dict(type='Printer'),
        # dict(type='ClampLabels', max_label=18),
        # dict(type='Printer'),
        #dict(type='RGBDResizePad', div_factor=14),
        #dict(type='PackRGBDSegInputs')
    ]


    # dataset = SunRGBDDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False,
    #                                 pipeline=train_pipeline, requires_depth=True, requires_ann=True, reduce_zero_label= False)


    # print("length of the dataset is:",len(dataset))
    # print(dataset.get_data_info(0))
    # print("dataset ,metainfo \n:",dataset.metainfo)
    # sample = dataset[5]

    print('Total number of classes in SUNrgbd is : ',len(classes))   

    train_dataset = SunRGBDDataset(data_root=data_root, data_prefix=data_prefix, ann_file=ann_file_path_val, split='val', test_mode=False,
                                    pipeline=train_pipeline, requires_depth=True, requires_ann=True, reduce_zero_label= True)

    print("length of train the dataset is:",len(train_dataset))

    #print(train_dataset.get_data_info(0))
    #print("train dataset ,metainfo \n:",train_dataset.metainfo)

    

    #sample = train_dataset[500]

    # val_dataset = SunRGBDDataset(data_root=data_root, data_prefix=data_prefix, ann_file= ann_file_path, split= 'val', test_mode=False,
    #                                 pipeline=train_pipeline, requires_depth=True, requires_ann=True, reduce_zero_label= True)

    # print("length of val the dataset is:",len(val_dataset))
    # print(val_dataset.get_data_info(0))
    # print("val dataset ,metainfo \n:",val_dataset.metainfo)
    # sample = val_dataset[5]


    # for k, v in sample.items():
    #     print('keys in sample are:', k)
    
    # # print('dataset sample 0 ', sample['gt_seg_map'])
    # # print('dataset sample seg map shape 0 ', sample['gt_seg_map'].shape)
    # # print("sample img shape \n:",sample['img'].shape)
    # # print("sample img type \n:",type(sample['img']))
    # # print("sample depth shape \n:",sample['gt_depth_map'].shape)
    # # print("sample depth fields \n:",sample['depth_fields'])
    
    # print("sample input shape \n:",sample['inputs'].shape)
    
    # # input()
    
    # print("data samples \n:",sample['data_samples'])
    # print("data samples seg fields \n:",sample['seg_fields'])
    # print("data samples label maps \n:",sample['label_map'])
    # print("data samples gt_sem_seg shape \n:",sample['data_samples'].gt_sem_seg.shape)
    #visualize_sample(sample , palette, classes)
    #input('continue?')
    #for i in range(len(train_dataset)):
    #    print('{}/{}'.format(i+1, len(train_dataset)))
    #    sample = train_dataset.__getitem__(i)

    mean_depth = []
    std_depth = []
    zeros = []
    for i in range(len(train_dataset)):
        
        #meta = train_dataset.get_data_info(i)
        print('-----------------------------------------------------')
        #print('{}/{} {} {} {}'.format(i, len(dataset), meta['seg_map_path'], meta['img_path'], meta['depth_map_path']))
        sample = train_dataset.__getitem__(i)

        depth = sample['img'][:, :, -1]
        mean_depth.append(np.mean(depth[depth>0]))
        std_depth.append(np.std(depth[depth>0]))

        zeros.append(1 - (np.sum(depth > 0) / np.prod(depth.shape)))

        
        print('{}/{} {}% mean depth: {}, std depth: {}, zeros: {}'.format(i, len(train_dataset), np.round(i / len(train_dataset) * 100, 2), np.mean(mean_depth), np.mean(std_depth), np.mean(zeros)))
        #input()
        #sample = dataset.__getitem__(i+1)
        #try:
        #    sample = dataset.__getitem__(i)
        #except Exception as e:
        #    print('failed with {}'.format(e))
        #    input('continue')



    # viz_img(img_path, seg_map_path, classes, palette)