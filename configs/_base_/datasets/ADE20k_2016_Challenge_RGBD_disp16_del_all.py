# dataset settings
dataset_type = 'ADE20KRGBDDataset'
#data_root = '/mnt/4TBSSD/datasets/ADEChallengeData2016'
data_root = '/mnt/data/publicdata/ADEChallengeData2016'


max_depth = 0.0
rescale_depth = 0.0
div_factor = 16
min_depth=0.0
baseline=0.05
focal_length=512
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32= True, imdecode_backend='pillow'),
    dict(type='PhotoMetricDistortion'),
    dict(type='LoadDepth', depthscale2meter=0.001, norm_depth=0, div_nr=0, max_depth=max_depth, rescale_depth=rescale_depth),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='ResizeRGBD',
        scale=(2048, 512),
        # ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCropRGBD', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlipRGBD', prob=0.5),
    #dict(type='DisparityAndRescale', rescale_factor=75.0, depth2disparty=True, baseline=baseline, focal_length=focal_length, min_depth=min_depth, max_depth=max_depth, clamp_max_depth=False, clamp_min_depth=False),
    dict(type='NormalizeRGBD', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb= False),    
    dict(type='RGBDResizePad', div_factor=div_factor),
    dict(type='PackRGBDSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32= True, imdecode_backend='pillow'),
    dict(type='LoadDepth', depthscale2meter=0.001, norm_depth=0, div_nr=0, max_depth=max_depth, rescale_depth=rescale_depth),
    dict(type='ResizeRGBD', scale=(2048, 512), keep_ratio=True),
    #dict(type='ResizeRGBD', scale=(512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    #dict(type='DisparityAndRescale', rescale_factor=75.0, depth2disparty=True, baseline=baseline, focal_length=focal_length, min_depth=min_depth, max_depth=max_depth, clamp_max_depth=False, clamp_min_depth=False),
    dict(type='NormalizeRGBD', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb= False),    
    dict(type='RGBDResizePad', div_factor=div_factor),
    dict(type='PackRGBDSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile',to_float32= True, imdecode_backend='pillow', backend_args=None),
    dict(type='LoadDepth', depthscale2meter=0.001, norm_depth=0, div_nr=0, max_depth=max_depth, rescale_depth=rescale_depth),
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
            #[dict(type='DisparityAndRescale', rescale_factor=75.0, depth2disparty=True, baseline=baseline, focal_length=focal_length, min_depth=min_depth, max_depth=max_depth, clamp_max_depth=False, clamp_min_depth=False)],
            [dict(type='NormalizeRGBD', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb= False)],   
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
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training', depth_map_path='images/training'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation', depth_map_path='images/validation'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
