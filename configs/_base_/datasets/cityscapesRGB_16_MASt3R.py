# dataset settings
dataset_type = 'CityscapesRGBDDataset'

data_root = '/mnt/wimi/publicdata/city/cityscapes_segmentation'

crop_size = (512, 512)
resize_size = (512, 1024)
test_size = (512, 1024)
#resize_size = (224, 224)

rgb_only = True
div_factor=16

'''
# RGB train pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    #dict(type= 'Resize', scale=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='RGBDResizePad', div_factor=14),
    dict(type='ClampLabels', max_label=18),
    dict(type='PackSegInputs')
]
'''
# For RGBD pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32= True, imdecode_backend='pillow'),
    dict(type='PhotoMetricDistortion'),   
]
if not rgb_only:
    train_pipeline +=[
        dict(type='LoadDepth', depthscale2meter=1, norm_depth=256, div_nr=0.209313 * 2262.52, max_depth=150),
    ] 

train_pipeline +=[
    dict(type='LoadAnnotations'),
    dict(
        type='ResizeRGBD',
        scale=resize_size,
        # ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCropRGBD', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlipRGBD', prob=0.5),
    dict(type='NormalizeRGBD', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb= False),

] 
if not rgb_only:
    train_pipeline +=[
         dict(type='DepthPositionalEncoding', depth_channels=32, encoder_hidden_dims=768),
    ] 

train_pipeline +=[
    dict(type='RGBDResizePad', div_factor=div_factor),
    #dict(type='ClampLabels', max_label=18),
    dict(type='PackRGBDSegInputs'),
] 

'''
# RGB test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='ClampLabels', max_label=18),
    dict(type='RGBDResizePad', div_factor=14),
    dict(type='PackSegInputs')
]
'''
# RGBD test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32= True, imdecode_backend='pillow'),    
]

if not rgb_only:
    test_pipeline +=[
       dict(type='LoadDepth', depthscale2meter=1, norm_depth=256, div_nr=0.209313 * 2262.52, max_depth=150),
    ] 

test_pipeline +=[
    dict(type='ResizeRGBD', scale=test_size, keep_ratio=True),
    
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    # dict(type='RandomCropRGBD', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlipRGBD', prob=0.5),^
    dict(type='NormalizeRGBD', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb= False),
    
] 

if not rgb_only:
    test_pipeline +=[
       dict(type='DepthPositionalEncoding', depth_channels=32, encoder_hidden_dims=768),
    ] 

test_pipeline +=[
    # dict(type='ClampLabels', max_label=18),
    dict(type='RGBDResizePad', div_factor=div_factor),
    dict(type='PackRGBDSegInputs')
] 


img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

'''
# RGB 
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')],
            [dict(type='RGBDResizePad', div_factor=14)],
            [dict(type='ClampLabels', max_label=18)],
            [dict(type='PackSegInputs')]
        ])
]
'''
#RGBD
tta_pipeline = [
    dict(type='LoadImageFromFile',to_float32= True, imdecode_backend='pillow', backend_args=None),
]


if not rgb_only:
    tta_pipeline.append(
       dict(type='LoadDepth', depthscale2meter=1, norm_depth=256, div_nr=0.209313 * 2262.52, max_depth=150),
    ) 

tta_aug_pipeline =[
            [
                dict(type='ResizeRGBD', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],

            [
                dict(type='RandomFlipRGBD', prob=0., direction='horizontal'),
                dict(type='RandomFlipRGBD', prob=1., direction='horizontal')
            ],   
            [dict(type='LoadAnnotations')],
            # [dict(type='RandomCropRGBD', crop_size=crop_size, cat_max_ratio=0.75)],
            # [
            #     dict(type='RandomFlipRGBD', prob=0., direction='horizontal'),
            #     dict(type='RandomFlipRGBD', prob=1., direction='horizontal')
            # ],   
            [dict(type='NormalizeRGBD', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb= False)],
    
] 

if not rgb_only:
    tta_aug_pipeline.append([
            [dict(type='DepthPositionalEncoding', depth_channels=32, encoder_hidden_dims=768)],   
])


tta_aug_pipeline.append([dict(type='RGBDResizePad', div_factor=div_factor)])
#tta_aug_pipeline.append([dict(type='ClampLabels', max_label=18)])
tta_aug_pipeline.append([dict(type='PackRGBDSegInputs')])


tta_pipeline.append(
    dict(
        type='TestTimeAug',
        transforms=tta_aug_pipeline
))



train_data_prefix = dict(
    img_path = 'leftImg8bit/train',
    seg_map_path='gtFine/train',
)

if not rgb_only : 
    train_data_prefix['depth_map_path'] = 'disparity_sequence/train'

train_dataloader = dict(
    batch_size=12,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        requires_depth= not rgb_only,
        data_prefix=train_data_prefix,
        pipeline=train_pipeline))


val_data_prefix = dict(
    img_path='leftImg8bit/val', 
    seg_map_path='gtFine/val',
)

if not rgb_only : 
    val_data_prefix['depth_map_path'] = 'disparity_sequence/val'


val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        requires_depth= not rgb_only,
        data_prefix= val_data_prefix,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator



