# Copyright (c) OpenMMLab. All rights reserved.
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
from mmengine.dataset import Compose
from my_mmseg.datasets.base_rgbd_mmseg_dataset import BaseRGBDSegDataset
from my_mmseg.utils.loading import *
from my_mmseg.utils.processing import *
from my_mmseg.utils.transforms import *
from my_mmseg.utils.formatting import *
from my_mmseg.utils.data_preprocessor import *
import matplotlib.pyplot as plt
import numpy as np

@DATASETS.register_module()
class COCOStuffRGBDDataset(BaseRGBDSegDataset):
    """
    COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    
    """
    METAINFO = dict(
        classes=(
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
            'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
            'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
            'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
            'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
            'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
            'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
            'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
            'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
            'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
            'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
            'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
            'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
            'stone', 'straw', 'structural-other', 'table', 'tent',
            'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
            'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
            'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
            'window-blind', 'window-other', 'wood'),
        palette=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                 [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                 [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                 [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                 [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                 [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                 [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                 [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                 [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                 [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                 [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                 [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                 [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                 [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
                 [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
                 [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
                 [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
                 [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                 [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
                 [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
                 [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
                 [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
                 [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                 [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
                 [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
                 [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
                 [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
                 [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                 [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
                 [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
                 [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                 [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
                 [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                 [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
                 [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
                 [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
                 [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
                 [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                 [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
                 [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
                 [64, 192, 96], [64, 160, 64], [64, 64, 0]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_labelTrainIds.png',
                 depth_map_suffix='_depth.png',
                 requires_depth=True,
                 requires_ann=True,
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, depth_map_suffix=depth_map_suffix,
            requires_depth=requires_depth, requires_ann=requires_ann, **kwargs)
        
        #print('CoCo-Stuff dataset lenth = {}, number of classes = {}'.format(len(self), len(self._metainfo['classes'])))
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

    print('depth_map min in Viz', np.min(depth_map))
    print('depth_map max in Viz', np.max(depth_map))


    #depth_map_normalized = (depth_map - np.min(depth_map))/ (np.max(depth_map) - np.min(depth_map))
    depth_map_normalized = depth_map / 15

    gt_color = np.zeros((gt_seg_map.shape[0], gt_seg_map.shape[1], 3), dtype = np.uint8)

    for class_id, color in enumerate(class_colors):
        mask = (gt_seg_map == class_id)
        gt_color[mask] = color 

    fig, ax = plt.subplots(1,3, figsize=(20,7))

    ax[0].imshow(img)
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    
    ax[1].imshow(gt_color)
    ax[1].set_title('Ground Truth Segmentation')
    ax[1].axis('off')

    ax[2].imshow(depth_map_normalized, cmap= 'viridis')
    ax[2].set_title('Depth Map')
    ax[2].axis('off')

    normalized_colors = normalize_colors(class_colors)

    handles =[plt.Line2D([0] ,[0], marker = 'o', color = 'w' , markerfacecolor = color, markersize = 10) for color in normalized_colors]
    ax[1].legend(handles, class_names, loc= 'right', bbox_to_anchor= (2.6,0.5), fontsize= 'small', ncol=1)

    plt.tight_layout()
    plt.show()


def min_max_dimensions(dataset):
    min_height = float('inf')
    max_height = 0
    min_width = float('inf')
    max_width = 0
    length = len(dataset)
    for i in range(length ):

        sample = dataset[i]
        ori_shape = sample['data_samples'].metainfo['ori_shape']

        height , width = ori_shape 

        if height < min_height:
            min_height = height
        if height > max_height:
            max_height = height
        if width < min_width:
            min_width = width
        if width > max_width:
            max_width = width

        if i % 1000 == 0:
            print(f'processed{i}/{length } samples')
    
    print(f'Minimum height:{min_height}') # 154
    print(f'Max height :{max_height}') # 640
    print(f'Min width :{min_width}') # 256
    print(f'Max width :{max_width}') # 640



def load_single_sample(img_path, seg_map_path, depth_map_path):
    img = Image.open(img_path).convert('RGB')
    seg_map = Image.open(seg_map_path).convert('L')
    depth_map = Image.open(depth_map_path).convert('L')

    img_np = np.array(img)
    seg_map_np = np.array(seg_map)
    depth_map_np = np.array(depth_map)
    reduce_zero_label = False
    seg_fields=[] 
    sample = dict(
        img=img_np,
        gt_semantic_seg=seg_map_np,
        gt_depth_map=depth_map_np,
        img_path = img_path,
        seg_map_path = seg_map_path,
        depth_map_path = depth_map_path,
        reduce_zero_label = reduce_zero_label,
        seg_fields= seg_fields
    )
    return sample

def apply_transforms(sample, pipeline):
    pipeline = Compose(pipeline)
    transformed_sample = pipeline(sample)
    return transformed_sample

def visualize_transformed_sample(transformed_sample, classes, palette):
    print('shape of img sample', transformed_sample['inputs'][:3,:,:].permute(1,2,0).shape)
    print('shape of seg_map sample', transformed_sample['data_samples'].gt_sem_seg.data.shape)
    print('shape of depth_map sample', transformed_sample['inputs'][3:,:,:].shape)
    
    
    img = transformed_sample['inputs'][:3,:,:].permute(1,2,0).numpy().astype(np.uint8)
    seg_map = transformed_sample['data_samples'].gt_sem_seg.data.squeeze(0).numpy()
    depth_map = transformed_sample['inputs'][3:,:,:].numpy()

    color_seg_map = label_to_color(seg_map, palette)
    verify_grayscale_class_indices(seg_map, classes)
    detected_classes(seg_map, classes)
    
    fig, axes = plt.subplots(1,4, figsize= (24,6))

    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(seg_map, cmap='gray')
    axes[1].set_title('Grayscale Seg Map')
    axes[1].axis('off')

    axes[2].imshow(color_seg_map)
    axes[2].set_title('Colorized Seg Map')
    axes[2].axis('off')



    # visualize_maps(img, seg_map, color_seg_map, palette)
    
    
    depth_map_1 = depth_map[0, : , :] 
    depth_map_normalized = depth_map_1 / 15
    axes[3].imshow(depth_map_normalized, cmap='viridis')
    axes[3].set_title('Depth Map channel 1')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show(block= False)
    display_legends(classes, palette)


if __name__ == '__main__':

    data_root = '/mnt/data/publicdata/coco'
    # For RGBD
    data_prefix = dict(img_path='images/train2017', seg_map_path='annotations/train2017',
                      depth_map_path='images/train2017')

    img_path = data_root + '/images/train2017/000000092107.jpg'
    seg_map_path = data_root + '/annotations/train2017/000000092107_labelTrainIds.png'
    depth_map_path = data_root + '/images/train2017/000000092107_depth.png'
    classes=[ 
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
                'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
                'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
                'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
                'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
                'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
                'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
                'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
                'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
                'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
                'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
                'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
                'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
                'stone', 'straw', 'structural-other', 'table', 'tent',
                'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
                'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
                'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
                'window-blind', 'window-other', 'wood'] 
                
    palette=[   [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
                [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
                [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
                [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
                [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
                [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
                [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
                [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
                [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
                [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
                [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
                [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
                [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
                [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
                [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
                [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
                [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
                [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
                [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
                [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
                [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
                [64, 192, 96], [64, 160, 64], [64, 64, 0]]

    rescale_depth = 15.0
    max_depth = 15.0
    depth_mul = 100
    # For RGBD pipeline
    train_pipeline = [
        dict(type='LoadImageFromFile', to_float32= True, imdecode_backend= 'pillow'),
        dict(type='LoadDepth', depthscale2meter=0.001, norm_depth=0, div_nr=0, max_depth=max_depth, rescale_depth=rescale_depth, depth_mul=depth_mul),
        dict(type='LoadAnnotations', imdecode_backend= 'pillow'),
        dict(type='ResizeRGBD',scale=(1280, 960),keep_ratio=True),
        dict(type='RandomCropRGBD', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlipRGBD', prob=0.5),
        dict(type='NormalizeRGBD', mean=[123.7709, 116.7460, 104.0937], std=[68.5005, 66.6322, 70.3232], to_rgb= True),
        #dict(type='Printer'),
        #dict(type='DepthPositionalEncoding', depth_channels=32, encoder_hidden_dims=768),
        #dict(type='Printer'),
        # dict(type='ClampLabels', max_label=18),
        #dict(type='Printer'),
        #dict(type='RGBDResizePad', div_factor=14),
        #dict(type='PackRGBDSegInputs')
    ]
    
    # sample = load_single_sample(img_path , seg_map_path, depth_map_path)
    # transformed_sample = apply_transforms(sample, train_pipeline)
    # visualize_transformed_sample(transformed_sample, classes, palette)
    
    
    print(classes[44] , classes[45], classes[50], classes[60], classes[109], classes[125], classes[155])
    # print(classes[0] , classes[2], classes[128], classes[157])
    # viz_img(img_path, seg_map_path, classes, palette)

    dataset = COCOStuffRGBDDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False,
                                    pipeline=train_pipeline, requires_depth=True, requires_ann=True)

    # min_max_dimensions(dataset)

    print("length of the dataset is:",len(dataset))
    print(dataset.get_data_info(0))
    print("dataset ,metainfo \n:",dataset.metainfo)
    sample = dataset[3744]

    # print('Total number of classes in coco is : ',len(classes))  # 171 checked 


    # for k, v in sample.items():
    #     print('keys in sample are:', k)
    
    # # print('dataset sample 0 ', sample['gt_seg_map'])
    # # print("sample img shape \n:",sample['img'].shape)
    # # print("sample img type \n:",type(sample['img']))
    # # print("sample depth shape \n:",sample['gt_depth_map'].shape)
    # # print("sample depth fields \n:",sample['depth_fields'])
    
    # print("sample input shape \n:",sample['inputs'].shape)
    
    # # input()
    
    # print("data samples \n:",sample['data_samples'])
    # # print("data samples seg fields \n:",sample['seg_fields'])
    # # print("data samples label maps \n:",sample['label_map'])
    # # print("data samples gt_sem_seg shape \n:",sample['data_samples'].gt_sem_seg.shape)
    #visualize_sample(sample , palette, classes)
    #input('continue? {}'.format(len(dataset)))
    mean_depth = []
    std_depth = []
    for i in range(len(dataset)):
        
        meta = dataset.get_data_info(i)
        print('-----------------------------------------------------')
        #print('{}/{} {} {} {}'.format(i, len(dataset), meta['seg_map_path'], meta['img_path'], meta['depth_map_path']))
        sample = dataset.__getitem__(i)

        mean_depth.append(np.mean(sample['img'][:, :, -1]))
        std_depth.append(np.std(sample['img'][:, :, -1]))
        print('{}/{} {}% mean depth: {}, std depth: {}'.format(i, len(dataset), np.round(i / len(dataset) * 100, 2), np.mean(mean_depth), np.mean(std_depth)))
        #input()
        #sample = dataset.__getitem__(i+1)
        #try:
        #    sample = dataset.__getitem__(i)
        #except Exception as e:
        #    print('failed with {}'.format(e))
        #    input('continue')

    

    