# dataset settings
dataset_type = 'SunRGBDDataset'


#data_root = '/home/chowanki/git/vanishing-depth-self-supervised/data/SunRGBD/SUNRGBD'
data_root = '/mnt/data/publicdata/SunRGBD/SUNRGBD'
#ann_file_path_train = data_root + '/sunrgbd_train_segmentation_dataset_config.json'
ann_file_path_val = data_root + '/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'

crop_size = (512, 512)
resize_size_scale = (2048, 512)
max_depth = 20
min_depth=0.1
baseline=0.05
focal_length=512
depth_mul = 1.0
depth_deleted = 0.99
rescale_depth = 15
div_factor = 16

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32= True, imdecode_backend='pillow'),
    dict(type='PhotoMetricDistortion'),
    dict(type='LoadDepth', depthscale2meter=0.00025, norm_depth=0, div_nr=0, max_depth=max_depth, depth_mul=depth_mul, depth_deleted=depth_deleted, rescale_depth=rescale_depth),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='ResizeRGBD',
        scale=resize_size_scale,
        # ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCropRGBD', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlipRGBD', prob=0.5),
    dict(type='DisparityAndRescale', rescale_factor=75.0, depth2disparty=True, baseline=baseline, focal_length=focal_length, min_depth=min_depth, max_depth=max_depth, clamp_max_depth=False, clamp_min_depth=False),
    dict(type='NormalizeRGBD', mean=[123.675, 116.28, 103.53, 0.0418], std=[58.395, 57.12, 57.375, 0.0295], to_rgb= False),    
    dict(type='RGBDResizePad', div_factor=div_factor),
    dict(type='PackRGBDSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32= True, imdecode_backend='pillow'),
    dict(type='LoadDepth', depthscale2meter=0.00025, norm_depth=0, div_nr=0, max_depth=max_depth, depth_mul=depth_mul, depth_deleted=depth_deleted, rescale_depth=rescale_depth),
    dict(type='ResizeRGBD', scale=resize_size_scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='DisparityAndRescale', rescale_factor=75.0, depth2disparty=True, baseline=baseline, focal_length=focal_length, min_depth=min_depth, max_depth=max_depth, clamp_max_depth=False, clamp_min_depth=False),
    dict(type='NormalizeRGBD', mean=[123.675, 116.28, 103.53, 0.0418], std=[58.395, 57.12, 57.375, 0.0295], to_rgb= False),   
    # dict(type='ClampLabels', max_label=18),
    dict(type='RGBDResizePad', div_factor=div_factor),
    dict(type='PackRGBDSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile',to_float32= True, imdecode_backend='pillow', backend_args=None),
    dict(type='LoadDepth', depthscale2meter=0.00025, norm_depth=0, div_nr=0, max_depth=max_depth, depth_mul=depth_mul, depth_deleted=depth_deleted, rescale_depth=rescale_depth),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='ResizeRGBD', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlipRGBD', prob=0., direction='horizontal'),
                dict(type='RandomFlipRGBD', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations', reduce_zero_label=True)], 
            
            [dict(type='DisparityAndRescale', rescale_factor=75.0, depth2disparty=True, baseline=baseline, focal_length=focal_length, min_depth=min_depth, max_depth=max_depth, clamp_max_depth=False, clamp_min_depth=False)],
            [dict(type='NormalizeRGBD', mean=[123.675, 116.28, 103.53, 0.0418], std=[58.395, 57.12, 57.375, 0.0295], to_rgb= False)],   
            [dict(type='RGBDResizePad', div_factor=div_factor)],
            [dict(type='PackRGBDSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=12,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file = ann_file_path_val,
        split='train',
        data_prefix=dict(
            img_path='image', seg_map_path='seg_mask', depth_map_path='depth'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file = ann_file_path_val,
        split='val',
        data_prefix=dict(
            img_path='image',
            seg_map_path='seg_mask', 
            depth_map_path='depth'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator