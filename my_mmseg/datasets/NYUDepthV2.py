
from mmseg.datasets import CityscapesDataset
from mmengine.registry import init_default_scope
init_default_scope('mmseg')

import os
print(os.getcwd())
from tqdm import tqdm
from collections import defaultdict
from mmseg.registry import DATASETS
from my_mmseg.datasets.base_rgbd_mmseg_dataset import BaseRGBDSegDataset
try:
    from my_mmseg.utils.loading import *
    from my_mmseg.utils.processing import *
    from my_mmseg.utils.transforms import ClampLabels
    from my_mmseg.utils.formatting import *
    from my_mmseg.utils.data_preprocessor import *
except:
    pass
import matplotlib.pyplot as plt
import numpy as np

@DATASETS.register_module()
class NYUv2RGBDDataset(BaseRGBDSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf',
                            'picture','counter','blinds', 'desk','shelves','curtain','dresser','pillow','mirror',
                            'floor mat','clothes','ceiling','books','refridgerator', 'television','paper','towel',
                            'shower curtain','box','whiteboard','person','night stand','toilet', 'sink','lamp',
                            'bathtub','bag','otherstructure', 'furniture','otherprop'),
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
                    [255, 255, 0], [52, 57, 131], [12, 83, 45], [119, 11, 32], 
                    [128, 64, 128],[244, 35, 232]])

    def __init__(self,
                 img_suffix='_rgb.png',
                 seg_map_suffix='_mask.png',
                 depth_map_suffix='_depth.png',
                 requires_depth=False,
                 requires_ann=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, depth_map_suffix=depth_map_suffix,
            requires_depth=requires_depth, requires_ann=requires_ann, **kwargs)
        
        print('NYU Depth V2 dataset lenth = {}, number of classes = {}'.format(len(self), len(self._metainfo['classes'])))
        #input()


def compute_rgb_mean_std(dataset):
    sum_rgb = torch.zeros(3)
    sum_rgb_sq = torch.zeros(3)
    num_pixels = 0 

    for i in tqdm(range(len(dataset)), desc = 'computing mean and std'):
        
        try:
            sample = dataset[i]
            if 'inputs' in sample:
        
                
                rgb_tensor = sample['inputs'][:3, :, :]

                sum_rgb += rgb_tensor.sum(dim=[1,2])
                sum_rgb_sq += (rgb_tensor**2).sum(dim=[1,2])
                num_pixels += rgb_tensor.size(1) * rgb_tensor.size(2)

                if i%100 == 0:
                    print(f'processesd{i} samples')
            elif 'img' in sample:
               
                img = sample['img']
                rgb_tensor = torch.from_numpy(img).permute(2,0,1).float()/255.0
                sum_rgb += rgb_tensor.sum(dim=[1,2])
                sum_rgb_sq += (rgb_tensor**2).sum(dim=[1,2])
                num_pixels += rgb_tensor.size(1) * rgb_tensor.size(2)

                if i%100 == 0:
                    print(f'processesd{i} samples')

        except Exception as e:
            print(f'Error processing{i}:{e}')

    mean_rgb = sum_rgb / num_pixels
    std_rgb = torch.sqrt((sum_rgb_sq/num_pixels) - (mean_rgb ** 2))

    return mean_rgb , std_rgb

def check_class_depth_data(dataset):
    class_depth_presence = defaultdict(bool)
    class_counts = defaultdict(int)
    total_samples = len(dataset)
    samples_with_depth = 0

    for i in range(total_samples):
        sample= dataset[i]
        seg_map = sample['data_samples'].gt_sem_seg.data
        depth_map = sample['data_samples'].gt_depth_map.data

        if torch.any(depth_map > 0):
            samples_with_depth += 1
        
        unique_classes = torch.unique(seg_map)
        for class_id in unique_classes:
            if class_id < len(dataset.METAINFO['classes']):
                class_name = dataset.METAINFO['classes'][class_id]
                class_counts[class_name] += 1

                class_mask = (seg_map == class_id)
                class_depth = depth_map[class_mask] 

                if torch.any(class_depth > 0):
                    class_depth_presence[class_name]= True

        print(f'Total samples: {total_samples}')
        print(f'Samples with depth data_:{samples_with_depth}')
        print('\n Class presence with depth data:') 
        for class_name in dataset.METAINFO['classes']:
            print(f'{class_name}: Present in {class_counts[class_name]} samples, Has depth data: {class_depth_presence[class_name]}')
        classes_without_depth =[class_name for class_name, has_depth in class_depth_presence.items() if not has_depth]
        if classes_without_depth:
            print('\n Classes without depth data:')
            for class_name in classes_without_depth:
                print(f' -{class_name}')
        else:
            print('\n All classes have depth data.')

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
    if len(depth_channels.shape) > 2:
        depth_map = depth_channels[0,:,:] 
    else:
        depth_map = depth_channels
    # depth_map = np.mean(depth_channels, axis=0)
    # depth_map = depth_channels.squeeze(0) # when using only loaddepth class and not DPE
    print('depth_map shape in viz ', depth_map.shape)

    print('depth_map min in Viz', np.min(depth_map))
    print('depth_map max in Viz', np.max(depth_map))


    #depth_map_normalized = (depth_map - np.min(depth_map))/ (np.max(depth_map) - np.min(depth_map))
    #depth_map_normalized = depth_map / 15

    gt_color = np.zeros((gt_seg_map.shape[0], gt_seg_map.shape[1], 3), dtype = np.uint8)

    for class_id, color in enumerate(class_colors):
        mask = (gt_seg_map == class_id)
        gt_color[mask] = color 

    fig, ax = plt.subplots(2,2, figsize=(20,7))

    ax[0,0].imshow(img)
    ax[0,0].set_title('Input Image')
    ax[0,0].axis('off')
    
    ax[0,1].imshow(gt_color)
    ax[0,1].set_title('Ground Truth Segmentation')
    ax[0,1].axis('off')

    ax[1,0].imshow(depth_map, cmap= 'viridis')
    ax[1,0].set_title('Depth Map')
    ax[1,0].axis('off')
    ax[1,1].imshow(depth_map, cmap= 'magma')
    ax[1,1].set_title('Depth Map')
    ax[1,1].axis('off')

    normalized_colors = normalize_colors(class_colors)

    handles =[plt.Line2D([0] ,[0], marker = 'o', color = 'w' , markerfacecolor = color, markersize = 10) for color in normalized_colors]
    ax[0,1].legend(handles, class_names, loc= 'lower right', bbox_to_anchor= (1.5,1))

    plt.show()


if __name__ == '__main__':

    #data_root = '/mnt/4TBSSD/datasets/cityscapes_segmentation/'
    
    data_root = '/mnt/data/publicdata/NYU_V2_Seg'
    # For RGBD

    mode = 'train'
    data_prefix = dict(img_path='{}/images'.format(mode), seg_map_path='{}/masks'.format(mode),
                      depth_map_path='{}/depth'.format(mode))
    
    max_depth = 20
    min_depth=0.1
    baseline=0.05
    focal_length=512
    rescale_depth=15

    # For RGBD pipeline
    train_pipeline = [
        dict(type='LoadImageFromFile', to_float32= True, imdecode_backend= 'pillow'),
        dict(type='LoadDepth', depthscale2meter=0.001, norm_depth=0, div_nr=0, max_depth=max_depth, rescale_depth=rescale_depth),
        dict(type='LoadAnnotations'),
        dict(type='ResizeRGBD',scale=(1280, 640),keep_ratio=True),
        dict(type='RandomCropRGBD', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlipRGBD', prob=0.5),
        #dict(type='NormalizeRGBD', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb= False),
        dict(type='Printer'),
        #dict(type='Printer', name='Input'),

        dict(type='DisparityAndRescale', rescale_factor=1.0, depth2disparty=False, baseline=baseline, focal_length=focal_length, min_depth=min_depth, max_depth=max_depth, clamp_max_depth=False, clamp_min_depth=False),
        dict(type='Printer', name='Disp'),
        dict(type='NormalizeRGBD', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb= False),
        
        #dict(type='NormalizeRGBD', mean=[0, 0 , 0, 0], std=[1, 1, 1, 1], to_rgb= False),    
        
        dict(type='DepthPositionalEncoding', depth_channels=32, encoder_hidden_dims=768),
        #dict(type='Printer', name='Norm'),
        # dict(type='ClampLabels', max_label=18),
         dict(type='Printer'),
        dict(type='RGBDResizePad', div_factor=16),
        dict(type='PackRGBDSegInputs')
    ]

    # # for RGB pipleine
    
    # train_pipeline = [
    #     dict(type='LoadImageFromFile'),
    #     dict(type='LoadAnnotations'),
    #     dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    #     dict(type='RandomFlip', prob=0.5),
    #     dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    #     dict(type='Printer'),
    #     dict(type='PackSegInputs'),
    #     dict(type='Printer'),] 
    

    dataset = NYUv2RGBDDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False,
                                pipeline=train_pipeline, requires_depth=True, requires_ann=True)

    
    # mean_rgb , std_rgb = compute_rgb_mean_std(dataset)
    # print('Mean RGB :', mean_rgb)
    # print('Std RGB :', std_rgb)
    
    print("length of the dataset is:",len(dataset))
    print(dataset.get_data_info(0))
    print("dataset ,metainfo \n:",dataset.metainfo)
    sample = dataset[25]
    # check_class_depth_data(dataset)
    
    for k, v in sample.items():
        print('keys in sample are:', k)
    
    # print('dataset sample 0 ', sample['gt_seg_map'].shape)
    # print('dataset sample gt_seg_map dtype is ', sample['gt_seg_map'].dtype)
    # print("sample img shape \n:",sample['img'].shape)
    # print("sample img dtype \n:",sample['img'].dtype)
    # print("sample img type \n:",type(sample['img']))
    # print("sample depth shape \n:",sample['gt_depth_map'].shape)
    # print("sample depth fields \n:",sample['depth_fields'])
    
    # print("sample input shape \n:",sample['inputs'].shape)
    
    # input()
    
    # print("data samples \n:",sample['data_samples'])
    # print("data samples seg fields \n:",sample['seg_fields'][0].dtype)
    # print("data samples label maps \n:",sample['label_map'])
    # print("data samples gt_sem_seg shape \n:",sample['data_samples'].gt_sem_seg.shape)

    visualize_sample(sample, dataset.metainfo['palette'], dataset.metainfo['classes'])
    input('continue? {}'.format(len(dataset)))
    for i in range(len(dataset)):
        
        meta = dataset.get_data_info(i)
        print('{}/{} {}'.format(i+1, len(dataset), meta['seg_map_path']))
        try:
            sample = dataset.__getitem__(i)
            visualize_sample(sample, dataset.metainfo['palette'], dataset.metainfo['classes'])
    
        except Exception as e:
            print('failed with {}'.format(e))
            input('continue')

    