# dataset settings
dataset_type = 'NYUv2RGBDDataset'
#data_root = '/home/chowanki/git/vanishing-depth-self-supervised/data/coco'
data_root = '/mnt/data/publicdata/NYU_V2_Seg'

crop_size = (512, 512)
resize_size = (5060, 512)
test_size = (2048, 480)

max_depth = None # removes values above 150m
rescale_depth = 15 # rescales the depth into the range of 0-> rescale_depth 
div_factor=32
min_depth=0.1
baseline=0.05
focal_length=512


train_pipeline = [
    dict(type='LoadImageFromFile', to_float32= True, imdecode_backend='pillow'),
    dict(type='PhotoMetricDistortion'),
    dict(type='LoadDepth', depthscale2meter=0.001, norm_depth=0, div_nr=0, max_depth=max_depth, rescale_depth=rescale_depth),
    dict(type='LoadAnnotations'),
    dict(type='ResizeRGBD',scale=resize_size, keep_ratio=True),
    dict(type='RandomCropRGBD', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlipRGBD', prob=0.5),
    dict(type='NormalizeRGBD', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb= False),
    dict(type='RGBDResizePad', div_factor=div_factor),
    dict(type='PackRGBDSegInputs')
]


test_pipeline = [
    dict(type='LoadImageFromFile', to_float32= True, imdecode_backend='pillow'),
    dict(type='LoadDepth', depthscale2meter=0.001, norm_depth=0, div_nr=0, max_depth=max_depth, rescale_depth=rescale_depth),
    dict(type='ResizeRGBD', scale=test_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='NormalizeRGBD', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb= False),   
    dict(type='RGBDResizePad', div_factor=div_factor),
    dict(type='PackRGBDSegInputs')
]


img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]


tta_pipeline = [
    dict(type='LoadImageFromFile',to_float32= True, imdecode_backend='pillow', backend_args=None),
    dict(type='LoadDepth', depthscale2meter=0.001, norm_depth=0, div_nr=0 , max_depth=max_depth, rescale_depth=rescale_depth),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='ResizeRGBD', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            #[dict(type='RandomCropRGBD', crop_size=crop_size, cat_max_ratio=0.75)],
            [
                dict(type='RandomFlipRGBD', prob=0., direction='horizontal'),
                dict(type='RandomFlipRGBD', prob=1., direction='horizontal')
            ], 
            [dict(type='LoadAnnotations')],
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
            img_path='train/images', seg_map_path='train/masks', depth_map_path='train/depth'),
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
            img_path='test/images', seg_map_path='test/masks', depth_map_path='test/depth'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

test_evaluator = val_evaluator
