from mmseg.registry import DATASETS
from mmengine.registry import init_default_scope
init_default_scope('mmseg')

import os
print(os.getcwd())
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from collections import defaultdict
from mmseg.registry import DATASETS
from my_mmseg.datasets.base_rgbd_mmseg_dataset import BaseRGBDSegDataset
from my_mmseg.utils.loading import *
from my_mmseg.utils.processing import *
from my_mmseg.utils.transforms import ClampLabels
from my_mmseg.utils.formatting import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

@DATASETS.register_module()
class ADE20KRGBDDataset(BaseRGBDSegDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                 'person', 'earth', 'door', 'table', 'mountain', 'plant',
                 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
                 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                 'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
                 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
                 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
                 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
                 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
                 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                 'chandelier', 'awning', 'streetlight', 'booth',
                 'television receiver', 'airplane', 'dirt track', 'apparel',
                 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
                 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
                 'conveyer belt', 'canopy', 'washer', 'plaything',
                 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
                 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
                 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
                 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                 'clock', 'flag'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 depth_map_suffix='_depth.png',
                 requires_depth=True,
                 requires_ann=True,
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            depth_map_suffix=depth_map_suffix,
            requires_depth=requires_depth, 
            requires_ann=requires_ann,
            reduce_zero_label=reduce_zero_label,
             **kwargs)

             
        print('ADE20KRGBD dataset lenth = {}, number of classes = {}'.format(len(self), len(self._metainfo['classes'])))
        #input()

def color_to_label(color_map, palette):

    color_map = np.array(color_map)
    print('color_map dtype: ',color_map.dtype)
    label_map = np.zeros(color_map.shape[:2], dtype =np.uint8)
    for label, color in enumerate(palette):
        mask = np.all(color_map == color, axis = -1)
        label_map[mask] = label

    return label_map

def label_to_color(label_map, palette):

    color_map = np.zeros((*label_map.shape,3), dtype= np.uint8)
    for label, color in enumerate(palette, start=1):
        color_map[label_map == label]= color

    return color_map


def detected_classes(label_map, class_names):
    
    unique_labels = np.unique(label_map)
    
    detected_classes =[class_names[label -1] for label in unique_labels if 0< label <= len(class_names)]
    print(f'detected classes:{detected_classes}')
    
    return detected_classes


def verify_grayscale_class_indices(grayscale_label_map , class_names):

    unique_values = np.unique(grayscale_label_map)

    print(f'unique grayscale values:{unique_values}')
    valid_values = set(range(1, len(class_names)+1))

    invalid_values = [val for val in unique_values if val not in valid_values]

    if invalid_values:
        print(f'Invalid values are: {invalid_values}')
    else:
        print('All grayscale values are valid') 


def visualize_maps(color_seg_map , grayscale_seg_map, color_label_map , grayscale_label_map, palette):
    """Visualize original and label maps side by side"""
    fig, axes = plt.subplots(2,3, figsize=(12,12))

    axes[0,0].imshow(color_seg_map)
    axes[0,0].set_title('Color Seg Map')

    axes[0,1].imshow(grayscale_seg_map, cmap='gray') 
    axes[0,1].set_title('Grayscale_seg_map')

    axes[1,0].imshow(color_label_map, cmap= 'jet')
    axes[1,0].set_title('Color_label_map')

    axes[1,1].imshow(grayscale_label_map, cmap= 'jet')
    axes[1,1].set_title('Grayscale_label_map')

    # converting grayscale to color using palette and classes
    grayscale_color_map = label_to_color(grayscale_label_map, palette)

    axes[0,2].imshow(grayscale_color_map)
    axes[0,2].set_title('Gray to Color Map') 


    plt.show()


def display_images(color_seg_path, grayscale_seg_path):
    """Display the images from paths """
    color_seg_map = Image.open(color_seg_path).convert('RGB')
    grayscale_seg_map = Image.open(grayscale_seg_path).convert('L')

    fig, axes = plt.subplots(1,2, figsize=(12,6))
    
    axes[0].imshow(color_seg_map)
    axes[0].set_title('Color Seg Image')

    axes[1].imshow(grayscale_seg_map, cmap = 'gray')
    axes[1].set_title('Grayscale Seg Image')

    plt.show()




def compare_segmentation_maps(color_seg_path, grayscale_seg_path, palette, class_names):
    


    display_images(color_seg_path, grayscale_seg_path)

    color_seg_map = Image.open(color_seg_path).convert('RGB')
    print(type(color_seg_map))
    
    color_label_map = color_to_label( color_seg_map, palette)
    print(color_label_map.dtype)
    
    grayscale_seg_map = Image.open(grayscale_seg_path).convert('L')
    grayscale_label_map = np.array(grayscale_seg_map)
    print('grayscale_label_map dtype: ',grayscale_label_map.dtype)


    verify_grayscale_class_indices(grayscale_label_map, class_names)

    print('color seg map:')
    detected_classes_color = detected_classes(color_label_map, class_names)

    print('grayscale seg map:')
    detected_classes_grayscale = detected_classes(grayscale_label_map , class_names)


    visualize_maps(color_seg_map, grayscale_seg_map, color_label_map, grayscale_label_map, palette)


    difference = np.abs(color_label_map - grayscale_label_map)

    if np.sum(difference)==0:
        print('Segmentation maps are identical')
    else: 
        print('Segmentation maps are different')
        print(f'number of differing pixels :{np.sum(difference !=0)}')


    return color_label_map, grayscale_label_map, difference


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


if __name__ == '__main__':

    data_root = '/mnt/data/publicdata/ADEChallengeData2016'
    # For RGBD
    data_prefix = dict(img_path='/mnt/data/publicdata/ADEChallengeData2016/images/validation', seg_map_path='/mnt/data/publicdata/ADEChallengeData2016/annotations/validation',
                      depth_map_path='/mnt/data/publicdata/ADEChallengeData2016/images/validation')

    classes=    ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                 'person', 'earth', 'door', 'table', 'mountain', 'plant',
                 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
                 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                 'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
                 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
                 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
                 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
                 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
                 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                 'chandelier', 'awning', 'streetlight', 'booth',
                 'television receiver', 'airplane', 'dirt track', 'apparel',
                 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
                 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
                 'conveyer belt', 'canopy', 'washer', 'plaything',
                 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
                 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
                 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
                 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                 'clock', 'flag'] 
    palette=    [ [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                [102, 255, 0], [92, 0, 255]]

    
    # For RGBD pipeline
    train_pipeline = [
        dict(type='LoadImageFromFile', to_float32= True, imdecode_backend= 'pillow'),
        dict(type='LoadDepth', depthscale2meter=1, norm_depth=256, div_nr=0.209313 * 2262.52, max_depth=150),
        dict(type='LoadAnnotations'),
        dict(type='ResizeRGBD',scale=(1366, 1024),keep_ratio=True),
        dict(type='RandomCropRGBD', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlipRGBD', prob=0.5),
        dict(type='NormalizeRGBD', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb= False),
        dict(type='Printer'),
        dict(type='DepthPositionalEncoding', depth_channels=32, encoder_hidden_dims=768),
        dict(type='Printer'),
        # dict(type='ClampLabels', max_label=18),
        # dict(type='Printer'),
        dict(type='RGBDResizePad', div_factor=14),
        dict(type='PackRGBDSegInputs')
    ]


    dataset = ADE20KRGBDDataset(data_root=data_root, data_prefix=data_prefix, test_mode=True,
                                    pipeline=train_pipeline, requires_depth=True, requires_ann=True)

    print("length of the dataset is:",len(dataset))
    print(dataset.get_data_info(0))
    print("dataset ,metainfo \n:",dataset.metainfo)
    sample = dataset[0]

    print('Total number of classes in ADE20k is : ',len(classes))
    # # print(Grayscale_values := list(range(1, len(classes)+1)))
    # color_seg_path = '/mnt/4TBSSD/datasets/ADE20K_2021_17_01/images/ADE/validation/cultural/archive/ADE_val_00000029_seg.png'
    # grayscale_seg_path = '/mnt/4TBSSD/datasets/ADEChallengeData2016/annotations/validation/ADE_val_00000029.png'
    # seg_map_path = '/home/chowanki/git/vanishing-depth-self-supervised/data/coco/annotations/train2017/000000092107_labelTrainIds.png'
    # color_label_map , grayscale_label_map, difference = compare_segmentation_maps(color_seg_path, grayscale_seg_path, palette, classes)

    # visualize_sample(sample , palette, classes)
