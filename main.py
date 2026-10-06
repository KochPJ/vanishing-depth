import random
import torch
import numpy as np
import json
import os
from datasets.representation_dataset import VanishingDepthDataset
import argparse
from model.vanishing_depth import build_model
import datasets.transforms as tfs
import torchvision.transforms as pytfs
from utils.loss import VanishingDepthLoss
import torch.nn as nn
import torch.optim.lr_scheduler as schedulers
from utils.stuff import bar_progress, mean_logs, add_logs, compute_auc, bar_progress_test, lookup, load_fitting_state_dict
import time
from datetime import datetime
import copy
import math
import sys


def get_args_parser():
    parser = argparse.ArgumentParser('Set vanishing depth', add_help=False)
    parser.add_argument('--exp_name_tag', default='EVA', type=str)  # true #
    parser.add_argument('--check_resources', default=False, type=bool) #true
    parser.add_argument('--epochs', default=100, type=int) #200
    parser.add_argument('--batch_size', default=128, type=int) #32 #96 #64
    parser.add_argument('--train_size', default=224, type=int) #480
    parser.add_argument('--num_workers', default=32, type=int) #16-8
    parser.add_argument('--lr', default=1e-5, type=float) #1e-4
    parser.add_argument('--lr_color_encoder', default=1e-5, type=float) #1-6
    parser.add_argument('--lr_decoder', default=1e-5, type=float) #1e-4
    parser.add_argument('--lr_w_fpn', default=False, type=bool) #false
    parser.add_argument('--lr_w_max', default=2.0, type=float) #2
    parser.add_argument('--lr_w_layer_decay', default=0.25, type=bool) #0.25
    parser.add_argument('--weight_decay', default=0.0, type=float) #0
    parser.add_argument('--optimizer', default='adamw', type=str) #adamw
    parser.add_argument('--normalize_loss', default=False, type=bool) #false
    parser.add_argument('--loss_with_not_mask', default=True, type=bool) #true
    parser.add_argument('--save_epoch', default=110, type=int) #true
    # loss
    parser.add_argument('--max_pow', default=1, type=float) #1
    parser.add_argument('--pow', default=0.5, type=float) #0.5
    parser.add_argument('--max_epoch', default=10, type=int) #100
    parser.add_argument('--start_epoch', default=0, type=int) #20
    parser.add_argument('--criterion_l', default=0.85, type=float) #0.85
    parser.add_argument('--criterion_final_l', default=0.85, type=float) #0.85
    parser.add_argument('--use_pow', default=False, type=bool) #true
    parser.add_argument('--sqrt_errors', default=False, type=bool) #true
    # scheduler
    parser.add_argument('--one_cycle', default=True, type=bool) #true
    parser.add_argument('--pct_start', default=0.2, type=float) #0.5
    parser.add_argument('--div_factor', default=5.0, type=float) #10.
    parser.add_argument('--final_div_factor', default=100.0, type=float) #100
    parser.add_argument('--lr_drop', default=50, type=int) #0

    # depth encoding
    parser.add_argument('--with_positional_encoding', default=True, type=bool)
    parser.add_argument('--depth_channels', default=32, type=int) #12 3d
    parser.add_argument('--temperature', default=0.0003, type=int) #1200, 14000 for 15m 3000 for 3d
    parser.add_argument('--position_offset', default=0.0, type=float)
    parser.add_argument('--scale', default=2*math.pi, type=float)
    parser.add_argument('--to_meter', default=True, type=bool)
    parser.add_argument('--random_to_m', default=0.5, type=float)#
    parser.add_argument('--zero_eps', default=1e-6, type=float)
    parser.add_argument('--set_pe_div_factors', default=False, type=bool)
    parser.add_argument('--pe_div_factors', default='1/2.5/5/25/50/150/250/500/750', type=str)

    # model
    parser.add_argument('--fpn', default=True, type=bool)
    parser.add_argument('--fpn_valid', default=False, type=bool)
    parser.add_argument('--fpn_layers', default=4, type=int)
    parser.add_argument('--model_name', default='EVAv2', type=str) # # EVAv2, Dino, OmniVore, MultiMAE, DFormer, DFormerV2, CroCoV2, SigLip2, OmniDC, DM3C
    parser.add_argument('--model_version', default='base', type=str) #  # V2-base / fit3D-base, ('base', 'small'), 'large', 
    parser.add_argument('--decoder', default='UNet', type=str) # UNet, DPT, linear
    parser.add_argument('--return_rdps', default=True, type=bool)
    parser.add_argument('--pretrained_imagenet', default=False, type=bool)
    #parser.add_argument('--pretrained_path', default='', type=str)
    #parser.add_argument('--pretrained_path', default='./data/backbones/CroCo_V2_ViTBase_BaseDecoder.pth', type=str)
    #parser.add_argument('--pretrained_path', default='./data/backbones/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', type=str)
    #parser.add_argument('--pretrained_path', default='./data/backbones/dune_vitbase14_448.pth', type=str)
    #parser.add_argument('--pretrained_path', default='./data/backbones/DFormer_Base.pth.tar', type=str)
    #parser.add_argument('--pretrained_path', default='./data/backbones/DFormerv2_Base_pretrained.pth', type=str)
    
    #parser.add_argument('--pretrained_path', default='./data/backbones/dinov2_vits14_pretrain.pth', type=str)
    #parser.add_argument('--pretrained_path', default='./data/backbones/dinov2_vitb14_pretrain.pth', type=str)
    parser.add_argument('--pretrained_path', default='./data/backbones/eva02_B_pt_in21k_p14.pt', type=str)
    #parser.add_argument('--pretrained_path', default='./data/backbones/dino_deitsmall16_pretrain.pth', type=str)
    #parser.add_argument('--pretrained_path', default='./data/backbones/dino_resnet50_pretrain.pth', type=str)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--decode_dist', default=True, type=bool)
    parser.add_argument('--decode_dist_relu', default=True, type=bool)
    parser.add_argument('--decode_dist_leaky_relu', default=False, type=bool)
    parser.add_argument('--decode_dist_sigmoid', default=False, type=bool)
    parser.add_argument('--decode_dist_softmax', default=False, type=bool)
    parser.add_argument('--decode_dist_scale_inv_log10', default=False, type=bool)
    parser.add_argument('--decode_max_dist', default=10.0, type=float)
    parser.add_argument('--decode_zero_dist', default=False, type=bool)
    parser.add_argument('--num_decode_dists', default=4, type=float)
    parser.add_argument('--decode_dist_factor', default=10, type=float)
    parser.add_argument('--decode_floor', default=1e-4, type=float)
    parser.add_argument('--decode_floor_exp', default=3, type=float)
    parser.add_argument('--freeze_color_encoder', default=True, type=float)
    parser.add_argument('--freeze_encoder', default=False, type=float)
    parser.add_argument('--with_dino_head', default=False, type=bool)
    parser.add_argument('--enable_dino_head_epoch', default=200, type=float)
    parser.add_argument('--dino_head_out_dims', default=65536, type=int)
    parser.add_argument('--dino_loss_w', default=0.5, type=float)
    parser.add_argument('--add_depth_scales_to_cls_token', default=True, type=bool)
    parser.add_argument('--encoder_hidden_dims', default=768, type=int) # dino s = 384, base = 768 , large = 1024

    # datasets
    parser.add_argument('--SUNRGBD', default=False, type=bool)
    parser.add_argument('--LineMod', default=True, type=bool)
    parser.add_argument('--T_Less', default=True, type=bool)
    parser.add_argument('--Graspnet', default=True, type=bool)
    parser.add_argument('--YCB_video', default=True, type=bool)
    parser.add_argument('--nyu_depth_v2', default=True, type=bool)
    parser.add_argument('--kitti', default=False, type=bool)
    parser.add_argument('--matterport', default=True, type=bool)
    parser.add_argument('--scannet', default=True, type=bool)
    parser.add_argument('--blendedmvs', default=True, type=bool)
    parser.add_argument('--cityscapes', default=True, type=bool)
    parser.add_argument('--SUNRGBD_sampling', default=0.25, type=float)
    parser.add_argument('--LineMod_sampling', default=1.0, type=float)
    parser.add_argument('--T_Less_sampling', default=1.0, type=float)
    parser.add_argument('--Graspnet_sampling', default=1.0, type=float)
    parser.add_argument('--YCB_video_sampling', default=1.0, type=float)
    parser.add_argument('--nyu_depth_v2_sampling', default=1.0, type=float)
    parser.add_argument('--kitti_sampling', default=1.0, type=float)
    parser.add_argument('--matterport_sampling', default=1.0, type=float)
    parser.add_argument('--scannet_sampling', default=1.0, type=float)
    parser.add_argument('--blendedmvs_sampling', default=1.0, type=float)
    parser.add_argument('--cityscapes_sampling', default=1.0, type=float)
    parser.add_argument('--SUNRGBD_depth_mul', default=1.0, type=float)
    parser.add_argument('--LineMod_depth_mul', default=1.0, type=float)
    parser.add_argument('--T_Less_depth_mul', default=1.0, type=float)
    parser.add_argument('--Graspnet_depth_mul', default=1.0, type=float)
    parser.add_argument('--YCB_video_depth_mul', default=1.0, type=float)
    parser.add_argument('--nyu_depth_v2_depth_mul', default=1.0, type=float)
    parser.add_argument('--kitti_depth_mul', default=1.0, type=float) #10 or 1
    parser.add_argument('--cityscapes_depth_mul', default=1.0, type=float) #10 or 1
    parser.add_argument('--matterport_depth_mul', default=1.0, type=float) #10 or 1
    parser.add_argument('--scannet_depth_mul', default=1.0, type=float) #10 or 1
    parser.add_argument('--blendedmvs_depth_mul', default=1.0, type=float) #10 or 1

    parser.add_argument('--samples_per_epoch', default=200000, type=int) #20000 #100000
    parser.add_argument('--valid_samples_per_dataset', default=0, type=int) # 1000
    parser.add_argument('--stack_thresholds', default=True, type=bool)
    parser.add_argument('--with_depth', default=True, type=bool)
    parser.add_argument('--with_depth_scales', default=False, type=bool)
    parser.add_argument('--with_intr', default=False, type=bool)
    parser.add_argument('--p_perlin_noise', default=0.5, type=float)
    parser.add_argument('--use_perlin', default=True, type=bool)
    #parser.add_argument('--depth_mean', default=0.8609524965286255, type=float)  # mean depth
    #parser.add_argument('--depth_std', default=1.1722136735916138, type=float)    # mean depth
    #parser.add_argument('--depth_mean', default=2.4597456455230713, type=float)  # mean depth[depth>0]
    #parser.add_argument('--depth_std', default=0.8662434220314026, type=float)    # mean depth[depth>0]
    #parser.add_argument('--depth_mean', default=4.55331632374, type=float)  # mean for disparity
    #parser.add_argument('--depth_std', default=1.60353178022, type=float)    # std for disparity
    parser.add_argument('--depth_mean', default=5.0, type=float)  # 5m-5s
    parser.add_argument('--depth_std', default=5.0, type=float)    # 5m-5s
    #parser.add_argument('--depth_mean', default=0.14909686148166656, type=float)
    #parser.add_argument('--depth_std', default=0.04623113200068474, type=float)
    #parser.add_argument('--depth_mean', default=0.0418, type=float) # omnivore
    #parser.add_argument('--depth_std', default=0.0295, type=float) # omnivore0418
    #parser.add_argument('--depth_mean', default=-1, type=float)
    #parser.add_argument('--depth_std', default=-1, type=float)
    parser.add_argument('--gt_mask_save_root', default='./data/gt_masks', type=str)
    parser.add_argument('--load_valid_gt_masks', default=True, type=bool)
    parser.add_argument('--overwrite_existing_valid_gt_masks', default=True, type=bool)
    parser.add_argument('--check_gt_masks', default=False, type=bool)
    parser.add_argument('--depth_invariant_preprocess', default=True, type=bool)
    parser.add_argument('--random_depth_start_epoch', default=0, type=int)
    parser.add_argument('--max_depth', default=15, type=float) # 15, 150
    parser.add_argument('--rescale_depth_max', default=True, type=bool)
    parser.add_argument('--min_depth', default=0.05, type=float)
    parser.add_argument('--max_gt_masks', default=2000, type=float)
    parser.add_argument('--p_disable_xy', default=0.33, type=float)

    # misc
    parser.add_argument('--visualize_samples', default=False, type=bool)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--multi_gpu', default=True, type=bool)
    parser.add_argument('--gpu_ids', default='0,1', type=str)
    parser.add_argument('--load_last_checkpoint', default=False, type=bool)
    #parser.add_argument('--load_checkpoint', default='./data/trained_models/20241012_1728722552_Dino_3D_UNet-DinoV2-3D', type=str)   
    #parser.add_argument('--load_checkpoint', default='./data/trained_models/20240920_1726824628_EVAv2_UNet-Long32c-15m-eva', type=str)
    #parser.add_argument('--load_checkpoint', default='./data/trained_models/20250104_1735985737_Dino_UNet-Long32c-No-Pre', type=str)   
    #parser.add_argument('--load_checkpoint', default='./data/trained_models/20240818_1724013182_Dino_V2-base_UNet-Long32c', type=str)      
    #parser.add_argument('--load_checkpoint', default='./data/trained_models/20240831_1725093075_Dino_UNet-Long32c-15m-fix-5m5s', type=str)   
    #parser.add_argument('--load_checkpoint', default='./data/trained_models/20250901_1756718487_Dino_DinoV3', type=str)   
    #parser.add_argument('--load_checkpoint', default='./data/trained_models/20250927_1758928924_Dino_Dinov2_PDE', type=str)   
    parser.add_argument('--load_checkpoint', default='./data/trained_models/20251115_1763217737_EVAv2_EVA', type=str)   
    #parser.add_argument('--load_checkpoint', default='./data/trained_models/20250901_1756710734_Dino_No-Perlin224', type=str)   
    
    
    parser.add_argument('--load_checkpoint_type', default='current_checkpoint', type=str)    
    parser.add_argument('--restart_from_checkpoint', default=False, type=bool)
    parser.add_argument('--output_path', default='./data/trained_models/', type=str)
    parser.add_argument('--gaussian_std', default=0.1, type=float) #0.1
    parser.add_argument('--gaussian_mean', default=0.3, type=float) #0.3
    parser.add_argument('--max_gaussian_std', default=0.3, type=float) #0.3
    parser.add_argument('--max_gaussian_mean', default=0.5, type=float) #0.5
    parser.add_argument('--max_gaussian_epoch', default=10, type=int)
    parser.add_argument('--t_min', default=0.10, type=float)
    parser.add_argument('--t_max', default=0.99, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for warm up "
                                                                      "with the start gaussian mean and std.")
    parser.add_argument('--thres_min', default=10, type=int)
    parser.add_argument('--thres_max', default=91, type=int)
    parser.add_argument('--thres_step_size', default=20, type=int)
    parser.add_argument('--eta_points', default=1000, type=int)
    parser.add_argument('--t1', default=0.6, type=float)
    parser.add_argument('--t2', default=3.0, type=float)
    parser.add_argument('--t3', default=10.0, type=float)
    return parser

#torch._dynamo.config.suppress_errors = True

def get_transforms(args, thresholds, depth_mean=None, depth_std=None):
    transforms = [tfs.ColorJitter(), tfs.RandomGray()]

    if not args.with_intr:
        transforms += [tfs.RandomHorizontalFlip(),
                       tfs.RandomVerticalFlip(),
                       tfs.RandomResizedCrop(size=args.train_size)
                       ]

    transforms.append(tfs.Depth2distance(to_meter=args.to_meter, max_dist=args.max_depth, random_to_m=args.random_to_m, rescale=args.rescale_depth_max))

    if args.depth_invariant_preprocess:
        transforms.append(tfs.RandomDepth(max_depth=args.max_depth, min_depth=args.min_depth,
                                          start_epoch=args.random_depth_start_epoch))
    transforms.append(tfs.GTDepth())

    if args.with_depth:
        transforms += [tfs.DepthNoise(min_depth=args.min_depth, max_depth=args.max_depth)]
        if args.with_intr:
            transforms += [tfs.DepthToPointCloud(p_disable_xy=args.p_disable_xy),
                           tfs.RandomHorizontalFlip(),
                           tfs.RandomVerticalFlip(),
                           tfs.RandomResizedCrop(size=args.train_size)
                           ]
        transforms += [
            tfs.VanishingDepth(fpn=args.fpn,
                               fpn_layers=args.fpn_layers,
                               perlin_nosie=args.p_perlin_noise,
                               use_perlin=args.use_perlin

        )]

    valid_transforms = [tfs.Depth2distance(to_meter=args.to_meter, max_dist=args.max_depth, random_to_m=args.random_to_m)]

    if args.with_depth:
        valid_transforms += [tfs.GTDepth()]
        if args.with_intr:
            valid_transforms += [tfs.DepthToPointCloud()]
        valid_transforms += [
            tfs.CenteredResizedCrop(size=args.train_size),
            tfs.VanishingDepthValid(thresholds=thresholds, fpn=args.fpn_valid,
                                    fpn_layers=args.fpn_layers,
                                    stack_thresholds=args.stack_thresholds,
                                    perlin_nosie=args.p_perlin_noise,
                                    use_perlin=args.use_perlin)]
                                    
    if not args.visualize_samples or True:
        preprocesses = [tfs.ToTensor(),
                        tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]

        if args.with_depth and args.with_positional_encoding:
            preprocesses += [tfs.DepthPositionalEncoding(depth_channels=args.depth_channels,
                                                         temperature=args.temperature,
                                                         position_offset=args.position_offset,
                                                         scale=args.scale,
                                                         zero_eps=args.zero_eps,
                                                         flatten_4th_dim=True if args.with_intr else False,
                                                         div_factors=args.pe_div_factors if
                                                         args.set_pe_div_factors else None,
                                                         with_dino_head=args.with_dino_head,
                                                         #with_dino_head=False,
                                                         encoder_hidden_dims=args.encoder_hidden_dims,
                                                         enable_dino_head_epoch=args.enable_dino_head_epoch,
                                                         get_depth_scales=args.with_depth_scales,
                                                         max_depth=args.max_depth)
                             ]
        elif args.with_depth and depth_mean is not None and depth_std is not None:
            preprocesses += [tfs.NormalizeDepth(mean=depth_mean, std=depth_std)]
        transforms += preprocesses
        valid_transforms += preprocesses
    return transforms, valid_transforms


def main(args):

    torch.multiprocessing.set_sharing_strategy('file_system')
    #torch.random.manual_seed(42)
    #np.random.seed(42)

    #args.enable_dino_head_epoch = 200

    if args.check_resources:
        args.enable_dino_head_epoch = 0
    if args.enable_dino_head_epoch < 1:
        args.enable_dino_head_epoch = int(args.epochs * args.enable_dino_head_epoch)
        if args.with_dino_head:
            print('Enable Dino Head at epoch: {}'.format(args.enable_dino_head_epoch))

    # loading checkpoint and set output path
    if not os.path.exists((args.output_path)):
        os.makedirs(args.output_path)
    cp = None
    path = None
    if args.load_last_checkpoint:
        print('load_last_checkpoint', args.load_last_checkpoint)
        cps = list(os.listdir(args.output_path))
        if cps:
            path = os.path.join(args.output_path, sorted(cps)[-1])
    elif args.load_checkpoint:
        print('load_checkpoint', args.load_checkpoint)
        if os.path.exists(args.load_checkpoint):
            path = args.load_checkpoint
        else:
            path = os.path.join(args.output_path, args.load_checkpoint)

    print('path', path)
    if path is not None:
        if os.path.exists(os.path.join(path, '{}.ckpt'.format(args.load_checkpoint_type))):
            print('loading checkpoint form path: {}'.format(path))
            cp = torch.load(os.path.join(path, '{}.ckpt'.format(args.load_checkpoint_type)), map_location='cpu')
        
        else:
            raise ValueError('path does not exist: {}'.format(path))


    if path is None or args.restart_from_checkpoint:
        path = os.path.join(args.output_path, '{}_{}_{}'.format(datetime.today().strftime("%Y%m%d"),
                                                                str(time.time()).split('.')[0],
                                                                args.model_name,
                                                                args.model_version))
        if args.with_intr:
            path = path + '_3D'
        if args.exp_name_tag:
            path = path + '_{}'.format(args.exp_name_tag)
    print('Saving to path: {}'.format(path))

    # Set up logs and load from checkpoint of given
    logs = {'epoch': 0,
            'epoch_logs': {'train': [], 'valid': []},
            'best_loss': -1,
            'best_auc': -1,
            'eta': {'epoch': [],
                    'train': [],
                    'valid': []},
            'args': {str(arg): getattr(args, arg) for arg in vars(args)}}
    print('cp', type(cp), args.restart_from_checkpoint)
    if cp is not None and not args.restart_from_checkpoint:
        logs = cp['logs']
        print('loading logs')
        #print(cp.keys())
        #input()
        if 'args' in cp['logs']:
            print('Loading args from pretrained checkpoint')
            for arg, v in cp['logs']['args'].items():
                if arg in ['enable_dino_head_epoch', 'scannet_sampling', 'SUNRGBD', 'kitti',  
                'restart_from_checkpoint', 't_min', 't_max', 'batch_size', 'num_workers']:
                    try:
                        print('Forcing {} to stay on {}, not loading {}'.format(arg, getattr(args, arg), v))
                    except:
                        pass
                    continue
                if v != getattr(args, arg):
                    print('Setting arg {}: from {} -> {}'.format(arg, getattr(args, arg), v))
                setattr(args, arg, v)

    # create the dataset
    datasets = {}
    valid_datasets = {}
    test_datasets = {}
    dataset_sampling = []
    load_intr = args.with_intr

    if args.SUNRGBD:
        from datasets.sunrgbd import SUNRGBD_Dataset
        print('Creating SUNRGBD dataset')
        datasets['SUNRGBD'] = SUNRGBD_Dataset(mode='train', load_from_checkpoint=True, with_intr=load_intr,
                                              p_eval=args.valid_samples_per_dataset,
                                              depth_mul=args.SUNRGBD_depth_mul)
        print('     Train SUNRGBD created with "{}" samples.'.format(len(datasets['SUNRGBD'])))
        dataset_sampling.append(args.SUNRGBD_sampling)

    if args.LineMod:
        from datasets.linemod import LineModeDataset
        print('Creating LineMod dataset')
        datasets['LineMod'] = LineModeDataset(mode='train', load_from_checkpoint=True, with_intr=load_intr,
                                              p_eval=args.valid_samples_per_dataset,
                                              depth_mul=args.LineMod_depth_mul)
        print('     Train LineMod created with "{}" samples.'.format(len(datasets['LineMod'])))
        dataset_sampling.append(args.LineMod_sampling)

    if args.T_Less:
        from datasets.t_less import TLESSDataset
        print('Creating T_Less dataset')
        datasets['T_Less'] = TLESSDataset(mode='train', load_from_checkpoint=True, with_intr=load_intr,
                                          p_eval=args.valid_samples_per_dataset,
                                          depth_mul=args.T_Less_depth_mul)
        print('     Train T_Less created with "{}" samples.'.format(len(datasets['T_Less'])))
        dataset_sampling.append(args.T_Less_sampling)

    if args.Graspnet:
        from datasets.grasp_net import GraspNetDataset
        print('Creating Graspnet dataset')
        datasets['Graspnet'] = GraspNetDataset(mode='train', load_from_checkpoint=True, with_intr=load_intr,
                                               p_eval=args.valid_samples_per_dataset,
                                               depth_mul=args.Graspnet_depth_mul)
        print('     Train Graspnet created with "{}" samples.'.format(len(datasets['Graspnet'])))
        dataset_sampling.append(args.Graspnet_sampling)

    if args.YCB_video:
        from datasets.ycb_video import YCBVDataset
        print('Creating YCB_video dataset')
        datasets['YCB_video'] = YCBVDataset(mode='train', load_from_checkpoint=True, with_intr=load_intr,
                                            p_eval=args.valid_samples_per_dataset,
                                            depth_mul=args.YCB_video_depth_mul)
        print('     Train YCB_video created with "{}" samples.'.format(len(datasets['YCB_video'])))
        dataset_sampling.append(args.YCB_video_sampling)

    if args.nyu_depth_v2:
        from datasets.nyu_depth_v2 import NyuDepthv2Dataset
        print('Creating nyu_depth_v2 dataset')
        datasets['nyu_depth_v2'] = NyuDepthv2Dataset(mode='train', load_from_checkpoint=True, with_intr=load_intr,
                                                     p_eval=args.valid_samples_per_dataset,
                                                     depth_mul=args.nyu_depth_v2_depth_mul)
        print('     Train nyu_depth_v2 created with "{}" samples.'.format(len(datasets['nyu_depth_v2'])))
        dataset_sampling.append(args.nyu_depth_v2_sampling)

    if args.kitti:
        from datasets.kitti import KittiDataset
        print('Creating kitti dataset')
        datasets['kitti'] = KittiDataset(mode='train', load_from_checkpoint=True, with_intr=load_intr,
                                         p_eval=args.valid_samples_per_dataset, depth_mul=args.kitti_depth_mul)
        print('     Train kitti created with "{}" samples.'.format(len(datasets['kitti'])))
        dataset_sampling.append(args.kitti_sampling)

    if args.cityscapes:
        from datasets.city_scapes import CityScapesDataset
        print('Creating cityscapes dataset')
        datasets['city_scapes'] = CityScapesDataset(mode='train', load_from_checkpoint=True, with_intr=load_intr,
                                                    p_eval=args.valid_samples_per_dataset,
                                                    depth_mul=args.cityscapes_depth_mul)
        print('     Train city_scapes created with "{}" samples.'.format(len(datasets['city_scapes'])))
        dataset_sampling.append(args.cityscapes_sampling)

    if args.matterport:
        from datasets.matterport3d import Matterport
        print('Creating Matterport dataset')
        datasets['Matterport'] = Matterport(load_from_checkpoint=True, with_intr=load_intr, depth_mul=args.matterport_depth_mul)
        print('     Train Matterport created with "{}" samples.'.format(len(datasets['Matterport'])))
        dataset_sampling.append(args.matterport_sampling)
    
    if args.scannet:
        from datasets.scannet import Scannet
        print('Creating Scannet dataset')
        datasets['Scannet'] = Scannet(load_from_checkpoint=True, with_intr=load_intr, depth_mul=args.scannet_depth_mul)
        print('     Train Scannet created with "{}" samples.'.format(len(datasets['Scannet'])))
        dataset_sampling.append(args.scannet_sampling)
        
    if args.blendedmvs:
        from datasets.blendedmvs import BlendedMVS
        print('Creating BlendedMVS dataset')
        datasets['BlendedMVS'] = BlendedMVS(load_from_checkpoint=True, with_intr=load_intr, depth_mul=args.blendedmvs_depth_mul)
        print('     Train BlendedMVS created with "{}" samples.'.format(len(datasets['BlendedMVS'])))
        dataset_sampling.append(args.blendedmvs_sampling)

    print('with_dino_head', args.with_dino_head, args.enable_dino_head_epoch)

    
    ds = VanishingDepthDataset(datasets, 'train', samples_per_epoch=args.samples_per_epoch,
                               with_depth=args.with_depth, 
                               with_depth_scales=args.with_depth_scales,
                               with_intr=load_intr,
                               with_dino_head=args.with_dino_head,
                               #with_dino_head=False,
                               max_gt_masks=args.max_gt_masks, dataset_sampling=dataset_sampling,
                               valid_gt_mask_save_root=args.gt_mask_save_root)
    

    thresholds = np.arange(args.thres_min, args.thres_max, args.thres_step_size) / 100.0
    print('Thresholds used in the AUC Eval Metric: {}'.format(thresholds))

    # get the depth mean and std if set
    if args.depth_mean < 0 or args.depth_std < 0:
        depth_mean = None
        depth_std = None
    else:
        depth_mean = args.depth_mean
        depth_std = args.depth_std

    # build the transforms
    transforms, valid_transforms = get_transforms(args, thresholds, depth_mean=depth_mean, depth_std=depth_std)
    ds.transforms = pytfs.Compose(transforms)
    if args.with_depth:
        # init the vanishing depth thresholds in the dataset
        ds.init_threshold(total_steps=args.max_gaussian_epoch * len(ds), gaussian_mean=args.gaussian_mean,
                          max_gaussian_mean=args.max_gaussian_mean, std=args.gaussian_std,
                          max_std=args.max_gaussian_std, t_min=args.t_min, t_max=args.t_max,
                          current_step=len(ds)*logs['epoch'], num_workers=args.num_workers,
                          warmup_steps=args.warmup_epochs * len(ds), epoch=logs['epoch'])

    if args.with_depth and not args.with_positional_encoding and (depth_mean is None or depth_std is None):
        # get the mean and std of the depth
        depth_mean = []
        depth_std = []
        print('Getting Depth mean and std from the Dataset')
        data_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)
        for j, batch in enumerate(data_loader):
            sys.stdout.write("\r" + "batch {}/{}".format(j+1, len(data_loader)))
            sys.stdout.flush()
            for depth in batch['depth']:
                try:
                    m = torch.mean(depth[depth > 0])
                    s = torch.std(depth[depth > 0])
                    if not torch.isnan(m):
                        depth_mean.append(m.unsqueeze(0))
                        depth_std.append(s.unsqueeze(0))
                except:
                    pass
        depth_mean = torch.mean(torch.cat(depth_mean))
        depth_std = torch.mean(torch.cat(depth_std))
        print('\n Using Depth mean, std: {}, {}'.format(depth_mean, depth_std))
        # log the mean and std
        logs['depth_mean'] = float(depth_mean)
        logs['depth_std'] = float(depth_std)
        # rebuild the transforms
        transforms, valid_transforms = get_transforms(args, thresholds, depth_mean=depth_mean, depth_std=depth_std)
        # reasign the transforms to the ds now including the depth and mean
        ds.transforms = pytfs.Compose(transforms)
        # reset vanishing depth thresholds
        ds.init_threshold(total_steps=args.max_gaussian_epoch * len(ds), gaussian_mean=args.gaussian_mean,
                          max_gaussian_mean=args.max_gaussian_mean, std=args.gaussian_std,
                          max_std=args.max_gaussian_std, t_min=args.t_min, t_max=args.t_max,
                          current_step=len(ds) * logs['epoch'], num_workers=args.num_workers,
                          warmup_steps=args.warmup_epochs * len(ds), epoch=logs['epoch'])

    if args.with_depth and not args.with_positional_encoding:
        logs['depth_mean'] = float(depth_mean)
        logs['depth_std'] = float(depth_std)

    if args.fpn_valid:
        args.gt_mask_save_root += '_fpn'
        args.gt_mask_save_root += '_fpn'

    if args.load_valid_gt_masks:
        if not os.path.exists(args.gt_mask_save_root):
            os.makedirs(args.gt_mask_save_root)

    ds_valid = VanishingDepthDataset(valid_datasets,
                                     'valid',
                                     pytfs.Compose(valid_transforms),
                                     samples_per_epoch=args.valid_samples_per_dataset,
                                     with_depth=args.with_depth,
                                     #with_intr=load_intr,
                                     load_valid_gt_masks=args.load_valid_gt_masks,
                                     valid_gt_mask_save_root=args.gt_mask_save_root,
                                     check_gt_masks=args.check_gt_masks,
                                     overwrite_existing_valid_gt_masks=args.overwrite_existing_valid_gt_masks,
                                     with_dino_head=args.with_dino_head,
                                     #with_dino_head=False,
                                     max_gt_masks=args.max_gt_masks)

    print('Train Dataset Length: {}'.format(len(ds)))
    print('Valid Dataset Length: {}'.format(len(ds_valid)))
    if args.visualize_samples:
        print(transforms)
        print(valid_transforms)
        # visualize samples if flag is set
        import matplotlib.pyplot as plt
        while True:
            i = int(np.random.randint(len(ds)))
            sample = ds.__getitem__(i)
            print('gt_depth', sample['gt_depth'][0].shape)
            print('gt_mask', sample['gt_mask'][0].shape)
            print('depth', sample['depth'].shape)
            print('img', sample['img'].shape)

            img = sample['img'].numpy().transpose(1, 2, 0)
            for c, (m, s) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
                img[:, :, c] = img[:, :, c] * s
                img[:, :, c] = img[:, :, c] + m
                img[:, :, c] = img[:, :, c] * 255
            img[img < 0] = 0
            img[img > 255] = 255
            img = np.array(img, dtype=np.uint8)

            depth = sample['depth']
            gt_depth = sample['gt_depth']
            if isinstance(gt_depth, list):
                gt_depth = gt_depth[0]

            gt_mask = sample['gt_mask']
            if isinstance(gt_mask, list):
                gt_mask = gt_mask[0]

            n = 4 #(img, depth, gt_depth, gt_mask)
            if args.with_intr:
                n += 2 # 2more gt_depth and all depths-1
            if len(gt_depth.shape) == 2:
                gt_depth = gt_depth.unsqueeze(0)
            if len(depth.shape) == 2:
                depth = depth.unsqueeze(0)
            else:
                n+= len(depth) - 1

            gt_mask = gt_mask.unsqueeze(0)

            gt_depth = gt_depth.numpy()
            gt_mask = gt_mask.numpy()
            depth = depth.numpy()
            x, y = lookup(n)

            plt.subplot(x, y, 1)
            plt.imshow(img)
            plt.title('in Img')

            c = 2
            for kk, d in enumerate(depth):
                plt.subplot(x, y, c)
                plt.imshow(d)
                plt.title('in depth {}'.format(kk))
                c += 1
            for kk, d in enumerate(gt_depth):
                plt.subplot(x, y, c)
                plt.imshow(d)
                plt.title('in gt_depth {}'.format(kk))
                c += 1
            for kk, d in enumerate(gt_mask):
                plt.subplot(x, y, c)
                plt.imshow(d)
                plt.title('in gt_mask {}'.format(kk))
                c += 1

            plt.show()
        return None

    print('train transforms: {}'.format(transforms))
    print('eval transforms: {}'.format(valid_transforms))


    # set device, build model, push model to device and load a checkpoint if given
    device = torch.device(args.device)
    #args.with_dino_head = True
    model = build_model(args)
    #print(model)
    #input('continue?')
    #args.with_dino_head = False

    if cp is not None:
        print('cp not none: loading model state dict')
        model = load_fitting_state_dict(model, cp['state_dict'])
        #model.load_state_dict(cp['state_dict'])

    device_ids = None
    if args.multi_gpu and torch.cuda.device_count() > 1 and args.device != 'cpu':
        if args.gpu_ids:
            device_ids = [int(gpu.replace(' ', '')) for gpu in args.gpu_ids.split(',')]
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    # init the dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=ds,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              drop_last=False)
   
    # compute lr weights per FPN layer if set
    if args.lr_w_fpn:
        lr_fpn_weights = [args.lr_w_max - ((args.fpn_layers-1 - i) * args.lr_w_layer_decay)
                          for i in range(args.fpn_layers)]
        print('Using lr weights of {} for the FPN layers (high res -> low res)'.format(lr_fpn_weights))
    else:
        lr_fpn_weights = None

    # init the loss function for train and validation
    criterion = VanishingDepthLoss(
        fpn=args.fpn, normalize=args.normalize_loss,
        loss_with_not_mask=args.loss_with_not_mask, with_point_cloud=args.with_intr,
        lw=lr_fpn_weights, t1=args.t1, t2=args.t2, t3=args.t3, use_pow=args.use_pow,
        max_pow=args.max_pow, pow=args.pow, max_epoch=args.max_epoch,
        start_epoch=args.start_epoch, current_epoch=logs['epoch'],
        l=args.criterion_l, final_l=args.criterion_final_l,
        with_dino_head=args.with_dino_head, dino_head_out_dims=args.dino_head_out_dims,
        epochs=args.epochs, dino_loss_w=args.dino_loss_w, sqrt_errors=args.sqrt_errors)

 

    if args.use_pow:
        print('criterion | pow: {},  l: {}, '.format(criterion.pow, criterion.l))

    # set the lr for the given parameter groups
    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if 'color_encoder' in n and p.requires_grad],
            "lr": args.lr_color_encoder,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if 'color_encoder' not in n and 'decoder' not in n
                       and 'depth_estimator' not in n and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if ('decoder' in n or 'depth_estimator' in n) and p.requires_grad],
            "lr": args.lr_decoder,
        }
    ]
    param_counts = [sum([np.prod([int(p_) for p_ in p.shape]) for p in d['params']]) for d in param_dicts]
    print('Total Parametergroups: {}'.format(len([p for p in model.parameters()])))
    print('Distribution of {} Paramgroups:'.format(
        sum([len(param_dicts[i]['params']) for i in range(len(param_dicts))])))
    print('     Color encoder: {} with lr: {}, frozen: {}, {}mio params = {}%'.format(len(param_dicts[0]['params']), args.lr_color_encoder,
                                                                  args.freeze_color_encoder,
                                                                  np.round(param_counts[0] / 1000000, 5),
                                                                  np.round(100*(param_counts[0] / sum(param_counts)),3)))
    print('     Depth encoder + Fusion: {} with lr: {}, {}mio params = {}%'.format(len(param_dicts[1]['params']), args.lr,
                                                                  np.round(param_counts[1] / 1000000, 5),
                                                                  np.round(100*(param_counts[1] / sum(param_counts)),3)))
    print('     Decoder: {} with lr: {}, {}mio params = {}%'.format(len(param_dicts[2]['params']), args.lr_decoder,
    
                                                                  np.round(param_counts[2] / 1000000, 5),
                                                                  np.round(100*(param_counts[2] / sum(param_counts)),3)))
    
    if args.freeze_color_encoder:
        for p in param_dicts[0]['params']:
            p.requires_grad = False

        param_dicts = param_dicts[1:]

    # init the optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=param_dicts, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=param_dicts, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_dicts, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError('the optimizer {} is not implemented'.format(args.optimizer))

    # init the scheduler
    if args.one_cycle:
        if args.freeze_color_encoder:
            lrs = [args.lr_decoder, args.lr]
        else:
            lrs = [args.lr_color_encoder, args.lr_decoder, args.lr]
        
        print('using Once Cycle with the lrs: {}'.format(lrs))
        scheduler = schedulers.OneCycleLR(optimizer=optimizer,
                                          max_lr=lrs,
                                          total_steps=args.epochs*len(data_loader),
                                          pct_start=args.pct_start, div_factor=args.div_factor,
                                          final_div_factor=args.final_div_factor)
    elif args.lr_drop > 0:
        if args.lr_drop < 1:
            args.lr_drop = int(args.lr_drop * args.epochs)
        print('Dropping LR on Epoch {}'.format(args.lr_drop))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    else:
        scheduler = None

    # load the optimizer and scheduler state dicts if we load from a checkpoint
    if cp is not None and not args.restart_from_checkpoint:
        optimizer.load_state_dict(cp['optimizer'])
        print('loading optimizer ckpt')
        if scheduler is not None and 'scheduler' in cp:
            scheduler.load_state_dict(cp['scheduler'])
            print('loading scheduler ckpt')
        if criterion.dino_loss is not None and 'dino_loss' in cp:
            criterion.dino_loss.load_state_dict(cp['dino_loss'])
            criterion_valid.dino_loss.load_state_dict(cp['dino_loss'])
            print('loading dino_loss ckpt')

    # clean the checkpoint of of the RAM
    del cp
    if args.check_resources:
        print('##########################################')
        print('### !! DEBUG MODE !! ####')
        print('### !! Checking Resources !! ####')
        print('##########################################')

    print('#### Starting Training ####')
    #input()
    print(logs['epoch'], args.epochs)
    # start the training loop
    t_epoch = time.time()
    for epoch in range(logs['epoch'], args.epochs):
        if args.check_resources:
            if epoch > 0:
                break
        logs['epoch'] = epoch + 1
        e_logs = {'loss': [],
                  'losses': {},
                  'metrics': {},
                  'dino_p': [],
                  'dino_loss': [],
                  'rdps': {str(i): [] for i in range(5)}}

        t_train = time.time()
        vanishing_depth_threshold = []
        model.train()
        for step, batch in enumerate(data_loader):
            optimizer.zero_grad()
            if args.check_resources:
                if step > args.num_workers * 2:
                    break

            # push data to device
            batch['img'] = batch['img'].to(device)

            depth, dino_depth, depth_scales, dino_depth_scales = None, None, None, None
            if args.with_depth:
                depth = batch['depth'].to(device)
                if args.with_depth_scales and 'depth_scales' in batch:
                    depth_scales = batch['depth_scales'].to(device)
                if 'dino_depth' in batch:
                    dino_depth = batch['dino_depth'].to(device)
                    if args.with_depth_scales:
                        dino_depth_scales = batch['dino_depth_scales'].to(device)
                        
            if 't' in batch:
                vanishing_depth_threshold.append(float(torch.mean(batch['t'])))

            if args.fpn:
                batch['gt_depth'] = [d.to(device) for d in batch['gt_depth']]
                batch['gt_mask'] = [m.to(device) for m in batch['gt_mask']]
            else:
                batch['gt_depth'] = batch['gt_depth'].to(device)
                batch['gt_mask'] = batch['gt_mask'].to(device)

            # forward data
            if depth is not None:
                pred = model(batch['img'], depth, depth_scales)
            else:
                pred = model(batch['img'])

            # if using dino head, compute the dino head for the teacher with the complete depth input
            if dino_depth is not None:
                model.eval()
                with torch.no_grad():
                    pred['teacher_pred'] = model(batch['img'], dino_depth, dino_depth_scales, without_decode=True)
                    print('dino teacher pred')
                model.train()

            # compute loss and metric scores
            if isinstance(pred, dict):
                if 'rdps' in pred:
                    #print(pred['rdps'])
                    if isinstance(pred['rdps'], dict):
                        pred['rdps'] = list(pred['rdps'].values())
                    for i, dp in enumerate(pred['rdps']):
                        e_logs['rdps'][str(i)].append(float(torch.mean(dp).item()))

            loss, losses, metrics = criterion(pred, batch['gt_depth'], batch['gt_mask'], epoch=epoch,
                                              depth_mul=None) #batch.get('depth_mul')
        
            if loss is None:
                # ignore a batch if by random chance a batch has only 100% of the depth disabled,
                # which is very unlikely to ever happen and only possible on spezial settings
                continue

            # backward pass and stepping
            try:
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()
                loss = float(loss.detach().item())
                optimizer.step()
            except Exception as e:
                print(e)
                print('Something went wrong, anyway, off we go')
                print('loss: {}'.format(loss))
                print('losses: {}'.format(losses))
                print('metrics: {}'.format(metrics))
                loss = 10
                for key, v in losses.items():
                    if np.isnan(v):
                        losses[key] = 10
                print('loss: {}'.format(loss))
                print('losses: {}'.format(losses))
                print('metrics: {}'.format(metrics))
                print('bounce model to avoid out of memory error')
                if args.multi_gpu and torch.cuda.device_count() > 1 and args.device != 'cpu':
                    model = model.module.cpu()
                    torch.cuda.empty_cache()
                    model = nn.DataParallel(model, device_ids=device_ids)
                else:
                    model = model.cpu()
                model = model.to(device)

            if args.one_cycle:
                scheduler.step()

            # adding logs
            if 'dino_depth' in batch:
                e_logs['dino_p'].append(losses['dino_loss'] / sum(list(losses.values())))
                e_logs['dino_loss'].append(losses['dino_loss'] / len(losses))
            e_logs = add_logs(e_logs, loss, losses, metrics)
            logs['eta']['train'].append(time.time()-t_train)
            if len(logs['eta']['train']) > args.eta_points:
                logs['eta']['train'] = logs['eta']['train'][1:]
            t_train = time.time()

            if scheduler is not None:
                current_lr = float(scheduler.get_last_lr()[-1])
            else:
                current_lr = args.lr
            bar_progress('train', epoch + 1, args.epochs, step+1, len(data_loader), 0, 0,
                         current_lr, e_logs,
                         logs['epoch_logs']['train'][-1] if len(logs['epoch_logs']['train']) > 0 else None,
                         loss, logs['eta'], np.mean(vanishing_depth_threshold))

        # step the scheduler here if not one cycle
        if not args.one_cycle and args.lr_drop > 0:
            scheduler.step()

        # log results of the training of the epoch
        e_logs = mean_logs(e_logs)
        logs['epoch_logs']['train'].append(e_logs)
      
        print('')  # next line in the progress bar

        

        if args.use_pow:
            criterion.step_epoch()
            print('\n pow: {}, l: {}'.format(criterion.pow, criterion.l))

        '''
        # end of epoch, log everything
        ev_logs = mean_logs(ev_logs)
        logs['epoch_logs']['valid'].append(ev_logs)
        current_auc = ev_logs['auc']['metrics']['layer_0']['dist']
        current_loss = ev_logs['auc']['loss']
        '''
        current_auc = e_logs['metrics']['layer_0']['dist']
        current_loss = e_logs['loss']

        # check if the current state is best loss or best AUC
        best_auc = False
        best_loss = False
        if current_auc < logs['best_auc'] or logs['best_auc'] == -1:
            logs['best_auc'] = current_auc
            best_auc = True

        if current_loss < logs['best_loss'] or logs['best_loss'] == -1:
            logs['best_loss'] = current_loss
            best_loss = True

        # make the current checkpoint
        ckpt = {'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'logs': logs}
        
        if criterion.dino_loss is not None:
            ckpt['dino_loss'] = criterion.dino_loss.state_dict()
        if scheduler is not None:
            ckpt['scheduler'] = scheduler.state_dict()

        # make the output path if it does not exist yet
        if not os.path.exists(path):
            os.makedirs(path)

        # save the current checkpoint and also if it is the best at the valid AUC or loss
        torch.save(ckpt, os.path.join(path, 'current_checkpoint.ckpt'))
        with open(os.path.join(path, 'logs.json'), 'w') as f:
            json.dump(logs, f)

        if best_auc:
            print('save best model auc')
            torch.save(ckpt, os.path.join(path, 'best_auc_checkpoint.ckpt'))

        if best_loss:
            print('save best model loss')
            torch.save(ckpt, os.path.join(path, 'best_loss_checkpoint.ckpt'))

        if args.save_epoch == logs['epoch']:
            print('save model @epoch {}'.format(logs['epoch']))
            torch.save(ckpt, os.path.join(path, 'ckeckpoint_epoch_{}.ckpt'.format(logs['epoch'])))

        # remoce the ckpt dict out of the RAM during the next epoch
        del ckpt

        # prepare the dataset for the next epoch, else it would simply restart the threshold from the start
        data_loader.dataset.init_threshold(total_steps=args.max_gaussian_epoch * len(ds),
                                           max_gaussian_mean=args.max_gaussian_mean,
                                           gaussian_mean=args.gaussian_mean, std=args.gaussian_std, t_min=args.t_min,
                                           t_max=args.t_max,
                                           current_step=len(ds) * logs['epoch'], num_workers=args.num_workers,
                                           warmup_steps=args.warmup_epochs * len(ds), epoch=logs['epoch'])
 

        # log the eta and go to next epoch
        logs['eta']['epoch'].append(time.time() - t_epoch)
        if len(logs['eta']['epoch']) > args.eta_points:
            logs['eta']['epoch'] = logs['eta']['epoch'][1:]
        t_epoch = time.time()
        print('')  # next line in the progress bar
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vanishing Depth training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)