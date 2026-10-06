import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import json
import os
from datasets.representation_dataset import VanishingDepthDataset
import argparse
from model.vanishing_depth import build_seg_model
from model.vanishing_depth import build_encoder
from model.deformable_detr.pose_estimation_transformer import build
import datasets.transforms as tfs
import torchvision.transforms as pytfs
from utils.loss import JaccardLoss
from utils.metric import IoU
import torch.nn as nn
import torch.optim.lr_scheduler as schedulers
from utils.stuff import bar_progress_pose, mean_logs, add_logs, bar_progress_test2, load_fitting_state_dict
import time
from datetime import datetime
import copy
import math
import sys
from utils.MVIP_utils import DepthNoise, RandomColor
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from model.deformable_detr.util.misc import collate_fn_pose as collate_fn
from model.deformable_detr.util.misc import LabelWrapper
from model.deformable_detr.pose_eval import build_pose_evaluator
from pathlib import Path

def get_args_parser():
    parser = argparse.ArgumentParser('Set vanishing depth', add_help=False)
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--name', default='LMO-RGBD', type=str) # LineMod, HB #YCB
    parser.add_argument('--epochs', default=10, type=int) #20
    parser.add_argument('--batch_size', default=48, type=int) #48
    parser.add_argument('--valid_batch_size', default=24, type=int)
    parser.add_argument('--batch_size_resized', default=12, type=int)
    parser.add_argument('--valid_batch_size_resized', default=12, type=int)
    parser.add_argument('--test_batch_size', default=18, type=int)
    parser.add_argument('--enable_resize_attention_maps', default=25, type=int)

    parser.add_argument('--train_size', default=(480, 640), type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--lr_encoder', default=1e-5, type=float)
    parser.add_argument('--lr_linear_project', default=5e-5, type=float)
    parser.add_argument('--lr_decoder', default=2e-4, type=float)
    #parser.add_argument('--lr_w_fpn', default=True, type=bool)
    #parser.add_argument('--lr_w_max', default=2.0, type=float)
    #parser.add_argument('--lr_w_max', default=2.0, type=float)
    #parser.add_argument('--lr_w_layer_decay', default=0.25, type=bool)f
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--optimizer', default='adam', type=str)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=30, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Matcher
    parser.add_argument('--matcher_type', default='pose', choices=['pose'], type=str)
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=1, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # datasets
    parser.add_argument('--ds_name', default='LM-O', type=str) #YCB_video, homebrew, LineMod, LM-O

    parser.add_argument('--depth_mean', default=-1, type=float)
    parser.add_argument('--depth_std', default=-1, type=float)
    parser.add_argument('--freeze_encoder', default=True, type=bool)
    parser.add_argument('--draw_num_samples', default=5000, type=int)

    # scheduler
    parser.add_argument('--one_cycle', default=False, type=bool)
    parser.add_argument('--pct_start', default=0.5, type=float)
    parser.add_argument('--div_factor', default=10.0, type=float)
    parser.add_argument('--final_div_factor', default=1000.0, type=float)
    parser.add_argument('--lr_drop', default=8, type=int)

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

    parser.add_argument('--backbone', default='ViT14', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--n_classes', default=4, type=int, help='number of feature levels')

    # * Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # Pose Estimation losses
    parser.add_argument('--translation_loss_coef', default=1, type=float, help='Loss weighing parameter for the translation')
    parser.add_argument('--rotation_loss_coef', default=1, type=float, help='Loss weighing parameter for the rotation')

    # ** PoET configs
    parser.add_argument('--bbox_mode', default='gt', type=str, choices=('gt', 'backbone', 'jitter'),
                        help='Defines which bounding boxes should be used for PoET to determine query embeddings.')
    parser.add_argument('--reference_points', default='bbox', type=str, choices=('bbox', 'learned'),
                        help='Defines whether the transformer reference points are learned or extracted from the bounding boxes')
    parser.add_argument('--query_embedding', default='bbox', type=str, choices=('bbox', 'learned'),
                        help='Defines whether the transformer query embeddings are learned or determined by the bounding boxes')
    parser.add_argument('--rotation_representation', default='6d', type=str, choices=('6d', 'quat', 'silho_quat'),
                        help="Determine the rotation representation with which PoET is trained.")
    parser.add_argument('--class_mode', default='specific', type=str, choices=('agnostic', 'specific'),
                        help="Determine whether PoET ist trained class-specific or class-agnostic")

    # model
    parser.add_argument('--fpn', default=True, type=bool)
    parser.add_argument('--fpn_valid', default=False, type=bool)
    parser.add_argument('--fpn_layers', default=4, type=int)
    parser.add_argument('--model_name', default='Dino', type=str) #ResNet_x->xd#_x->xd
    parser.add_argument('--model_version', default='V2-base', type=str) #DinoV2
    parser.add_argument('--return_rdps', default=False, type=bool)
    parser.add_argument('--with_depth', default=True, type=bool)
    parser.add_argument('--with_intr', default=False, type=bool)
    parser.add_argument('--pretrained_imagenet', default=False, type=bool)
    parser.add_argument('--pretrained_path', default='./data/backbones/dinov2_vitb14_pretrain.pth', type=str)
    #parser.add_argument('--encoder_path', default='', type=str)
    parser.add_argument('--encoder_path', default='./data/trained_models/20240908_1725793388_Dino_UNet-Long32c-15m-Fix15m32c_448_encoder.plt', type=str)
    parser.add_argument('--rgb_only', default=False, type=bool)
    parser.add_argument('--decoder', default='DPT', type=str) #DPT #linear
    parser.add_argument('--generate_depth_input', default=False, type=bool)
    parser.add_argument('--norm_return_layers', default=True, type=bool)
    parser.add_argument('--norm_return_layers_rgb', default=True, type=bool)
    parser.add_argument('--cat_outs', default=True, type=bool)
    parser.add_argument('--resize_attention_maps', default=False, type=bool)
    parser.add_argument('--encoder_hidden_dims', default=768, type=int) # dino s = 384, base = 768 , large = 1024
    parser.add_argument('--enable_dino_head_epoch', default=200, type=float)
    parser.add_argument('--with_depth_scales', default=False, type=bool)
    parser.add_argument('--max_depth', default=15, type=float) # 15, 150, 500
    #parser.add_argument('--fpn_layers', default=4, type=int)


    # misc 
    parser.add_argument('--eval_bop', default=False, type=bool)
    parser.add_argument('--visualize_samples', default=False, type=bool)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--multi_gpu', default=True, type=bool)
    parser.add_argument('--gpu_ids', default='0,1', type=str)
    parser.add_argument('--load_last_checkpoint', default=True, type=bool)
    parser.add_argument('--restart_from_checkpoint', default=False, type=bool)
    #parser.add_argument('--load_checkpoint_path', default='./data/trained_models_pose/YCBV_old/', type=str)
    #parser.add_argument('--load_checkpoint_path', default='./data/trained_models_pose/YCBV_3D/epoch_16/', type=str)
    parser.add_argument('--load_checkpoint_path', default='', type=str)
    parser.add_argument('--output_path', default='./data/trained_models_pose', type=str)
    parser.add_argument('--eta_points', default=1000, type=int)

    return parser

@torch.no_grad()
def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def eval_pose_add(args, pose_evaluator, outputs, matcher, targets, n_boxes_per_sample):
    outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
    
    #if args.ds_name == 'LM-O':
    #    obj_id_map = {1: 2, 5: 3, 6: 7, 8: 8, 9: 4, 10: 6, 11: 1, 12: 5}
    
    
    
    # Extract final predictions and store them
    indices = matcher(outputs_without_aux, targets, n_boxes_per_sample)
    idx = get_src_permutation_idx(indices)

    pred_translations = outputs_without_aux["pred_translation"][idx].detach().cpu().numpy()
    pred_rotations = outputs_without_aux["pred_rotation"][idx].detach().cpu().numpy()

    tgt_translations = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)],
                                 dim=0).detach().cpu().numpy()
    tgt_rotations = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)],
                              dim=0).detach().cpu().numpy()

    obj_classes_idx = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
    # print('obj class idx is ', obj_classes_idx)
    intrinsics = torch.cat([t['intrinsics'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
    #img_files = [data_loader.dataset.coco.loadImgs(t["image_id"].item())[0]['file_name'] for t, (_, i) in
    #             zip(targets, indices) for _ in range(0, len(i))]

    # Iterate over all predicted objects and save them in the pose evaluator
    for cls_idx, intrinsic, pred_translation, pred_rotation, tgt_translation, tgt_rotation in \
            zip(obj_classes_idx, intrinsics, pred_translations, pred_rotations, tgt_translations,
                tgt_rotations):
        # print('type of cls_idx', type(cls_idx))
        # print('cls_idx is ',cls_idx)

        #if cls_idx in obj_id_map.keys():
        #    new_cls = obj_id_map[cls_idx]


        #cls = pose_evaluator.classes[int(new_cls - 1)]
        cls = pose_evaluator.classes[int(cls_idx) - 1]
        pose_evaluator.poses_pred[cls].append(
            np.concatenate((pred_rotation, pred_translation.reshape(3, 1)), axis=1))
        pose_evaluator.poses_gt[cls].append(
            np.concatenate((tgt_rotation, tgt_translation.reshape(3, 1)), axis=1))
        #pose_evaluator.poses_img[cls].append(img_file)
        pose_evaluator.num[cls] += 1
        pose_evaluator.camera_intrinsics[cls].append(intrinsic)

    # At this point iterated over all validation images and for each object the result is fed into the pose evaluator


@torch.no_grad()
def bop_evaluate(model, matcher, data_loader, out_csv_file, n_boxes_per_sample, outputs, targets, pred_end_time, obj_id_map):
    """
    Evaluate PoET on the dataset and store the results in the BOP format
    """
    model.eval()
    matcher.eval()

    pred_end_time = -1
    n_images = len(data_loader)
    # print(data_loader)
    
    # print('target keys are')
    # for t in targets:
    #     print(t.keys())

    # input()

    # CSV format: scene_id, im_id, obj_id, score, R, t, time
    
    outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

    indices = matcher(outputs_without_aux, targets, n_boxes_per_sample)
    idx = get_src_permutation_idx(indices)

    pred_translations = outputs_without_aux["pred_translation"][idx].detach().cpu().numpy()
    pred_rotations = outputs_without_aux["pred_rotation"][idx].detach().cpu().numpy()

    

    obj_classes_idx = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)],
                                dim=0).detach().cpu().numpy()
    
    
    # intrinsics = torch.cat([t['intrinsics'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
    
    #print(targets[0].keys())
    #input()
    #print(indices)
    img_ids = [t['image_id'] for t, (_, i) in zip(targets, indices) for _ in range(len(i))] 
    targets_pos = [t['relative_position'][jj] for t, (_, i) in zip(targets, indices) for jj in range(len(i))] 
    targets_rot = [t['relative_rotation'][jj] for t, (_, i) in zip(targets, indices) for jj in range(len(i))] 
    
    #print(len(img_ids), len(pred_translations), len(pred_translations))

    for cls_idx, img_id, pred_translation, pred_rotation, tpos, trot in zip(obj_classes_idx, img_ids, pred_translations, pred_rotations, targets_pos, targets_rot):
        
        if cls_idx in obj_id_map.keys():
            cls_idx_new = obj_id_map[cls_idx] 

       
      
        
        # file_info = img_file.split("/")
        scene_id = 2 #int(file_info[1])
        img_id = img_id #int(file_info[3][:file_info[3].rfind(".")])
        obj_id = int(cls_idx_new)
        #print(tpos.shape, trot.shape)
        #print(tpos, trot)
        
        #input()
        score = 1.0
        if True:
            csv_str = "{},{},{},{},{} {} {} {} {} {} {} {} {}, {} {} {}, {}\n".format(scene_id, img_id, obj_id, score,
                                                                                    pred_rotation[0, 0], pred_rotation[0, 1], pred_rotation[0, 2],
                                                                                    pred_rotation[1, 0], pred_rotation[1, 1], pred_rotation[1, 2],
                                                                                    pred_rotation[2, 0], pred_rotation[2, 1], pred_rotation[2, 2],
                                                                                    pred_translation[0] * 1000, pred_translation[1] * 1000, pred_translation[2] * 1000,
                                                                                    pred_end_time)
        else:
            csv_str = "{},{},{},{},{} {} {} {} {} {} {} {} {}, {} {} {}, {}\n".format(scene_id, img_id, obj_id, score,
                                                                                    float(trot[0, 0]), float(trot[0, 1]), float(trot[0, 2]),
                                                                                    float(trot[1, 0]), float(trot[1, 1]), float(trot[1, 2]),
                                                                                    float(trot[2, 0]), float(trot[2, 1]), float(trot[2, 2]),
                                                                                    float(tpos[0]) * 1000, float(tpos[1]) * 1000, float(tpos[2]) * 1000,
                                                                                    pred_end_time)
        out_csv_file.write(csv_str)
        print(f'output written for : {img_id}')

    # out_csv_file.close()



def eval_pose_resutls(pose_evaluator, output_eval_dir, test=False):
    print("Start Calculating ADD")
    results_add = pose_evaluator.evaluate_pose_add(output_eval_dir)

    if test:
        print("Start Calculating ADD-S")
        results_adds = pose_evaluator.evaluate_pose_adi(output_eval_dir)
        print("Start Calculating ADD(-S)")
        pose_evaluator.evaluate_pose_adds(output_eval_dir)
        print("Start Calculating Average Translation Error")
        average_pos_err = pose_evaluator.calculate_class_avg_translation_error(output_eval_dir)
        print("Start Calculating Average Rotation Error")
        average_pos_rot = pose_evaluator.calculate_class_avg_rotation_error(output_eval_dir)

    print('Finsihed Eval')
    
    print('ADD-S results: {}'.format(results_add)) # , Pos Err: {}, Rot Err: {}, average_pos_err, average_pos_rot))


def get_dataloaders(args, ds, ds_valid, epoch=0):
    if epoch >= args.enable_resize_attention_maps:
        batch_size = args.batch_size_resized
        valid_batch_size = args.valid_batch_size_resized
    else:
        batch_size = args.batch_size
        valid_batch_size = args.valid_batch_size


    print('batch size train: {}, valid: {}'.format(batch_size, valid_batch_size))
    print('num_workers train: {}'.format(args.num_workers))

    if args.draw_num_samples > 0:
        data_loader = torch.utils.data.DataLoader(
            dataset=ds,
            sampler=torch.utils.data.RandomSampler(ds, num_samples=args.draw_num_samples * batch_size),
            batch_size=batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=collate_fn
        )
    else:
        data_loader = torch.utils.data.DataLoader(dataset=ds,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=args.num_workers,
                                                  drop_last=False,
                                                  collate_fn=collate_fn)

    valid_data_loader = torch.utils.data.DataLoader(dataset=ds_valid,
                                                    batch_size=valid_batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers,
                                                    drop_last=False,
                                                    collate_fn=collate_fn)
    return data_loader, valid_data_loader


def get_transforms(args, depth_mean=None, depth_std=None):
    transforms = [
        tfs.ColorJitter(),
        RandomColor(img_key='img')
    ]
    if args.with_intr:
        transforms.append(tfs.DepthToPointCloud(jitter=False))

    transforms += [
        #tfs.RandomHorizontalFlip(),
        #tfs.RandomVerticalFlip(),
        #tfs.RandomResizedCrop(size=args.train_size, scale=(0.5, 1.2)),
        tfs.ResizePad(),
        tfs.Depth2distance(to_meter=args.to_meter)
    ]

    if args.with_depth:
        transforms += [
            #DepthNoise(to_meter=args.to_meter, color_key='img', offset_p=0.0)
        ]

    valid_transforms = [
        tfs.ResizePad(),
        tfs.Depth2distance(to_meter=args.to_meter)
    ]

    if args.with_intr:
        valid_transforms = [tfs.DepthToPointCloud(jitter=False)] + valid_transforms
    #if args.with_depth:
    #    valid_transforms += [
    #        tfs.CenteredResizedCrop(size=args.train_size)
    #    ]

    if not args.visualize_samples:
        preprocesses = [
            tfs.ToTensor(),
            tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
        
        if args.with_depth and args.with_positional_encoding:
            preprocesses += [tfs.DepthPositionalEncoding(depth_channels=args.depth_channels,
                                                         temperature=args.temperature,
                                                         position_offset=args.position_offset,
                                                         scale=args.scale,
                                                         zero_eps=args.zero_eps,
                                                         flatten_4th_dim=True if args.with_intr else False,
                                                         div_factors=args.pe_div_factors if
                                                         args.set_pe_div_factors else None,
                                                         #with_dino_head=args.with_dino_head,
                                                         with_dino_head=False,
                                                         encoder_hidden_dims=args.encoder_hidden_dims,
                                                         enable_dino_head_epoch=args.enable_dino_head_epoch,
                                                         get_depth_scales=args.with_depth_scales,
                                                         max_depth=args.max_depth)
                             ]
        elif args.with_depth:
            if args.depth2disparty:
                preprocesses += [tfs.DisparityAndRescale(
                    rescale_factor=args.depth_rescale_div, 
                    depth2disparty=args.depth2disparty,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth
                    )]
            if depth_mean is not None and depth_std is not None:
                preprocesses += [tfs.NormalizeDepth(mean=depth_mean, std=depth_std)]


        transforms += preprocesses
        valid_transforms += preprocesses

    transforms = pytfs.Compose(transforms)
    valid_transforms = pytfs.Compose(valid_transforms)
    return transforms, valid_transforms


def main(args):

    t_s = time.time()
    hours = 0.0
    t_wait = t_s + hours * 3600

    while time.time() < t_wait:
        eta = t_wait - time.time()
        hours = 0
        if eta > 3600:
            hours = int(eta // 3600)
            eta = eta - hours * 3600
        minutes = 0
        if eta > 60:
            minutes = int(eta // 60)
            eta = eta - minutes * 60

        sec = int(eta)

        hours = str(hours).zfill(2)
        minutes = str(minutes).zfill(2)
        sec = str(sec).zfill(2)
        progress_message = 'remaining {}:{}:{}'.format(hours, minutes, sec)

        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()
        time.sleep(1)



    if not os.path.exists((args.output_path)):
        os.makedirs(args.output_path)

    cp = None
    path = os.path.join(args.output_path, args.name)

    if args.load_last_checkpoint and path is not None:
        if os.path.exists(args.load_checkpoint_path):
            print('loading checkpoint from path: {}'.format(args.load_checkpoint_path))
            cp = torch.load(os.path.join(args.load_checkpoint_path, 'current_checkpoint.ckpt'), map_location='cpu')
            args.restart_from_checkpoint = True
        elif os.path.exists(os.path.join(path, 'current_checkpoint.ckpt')):
            print('loading checkpoint form path: {}'.format(path))
            cp = torch.load(os.path.join(path, 'current_checkpoint.ckpt'), map_location='cpu')

    print('restart_from_checkpoint', args.restart_from_checkpoint)
    if cp is not None and not args.restart_from_checkpoint:
        if 'args' in cp['logs']:
            ignore_args = ['device', 'epochs', 'enable_resize_attention_maps', 'lr_drop', 'freeze_encoder']
            #print('Loading args')
            
            print('Loading args from pretrained checkpoint')
            for arg, v in cp['logs']['args'].items():
                if arg in ignore_args and arg in args:
                    print('ignore arg {}, use {} and not {}'.format(arg, getattr(args, arg), v))
                    continue

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

            '''
            for arg, v in cp['logs']['args'].items():
                if arg in ignore_args:
                    print('ignore arg {}, use {} and not {}'.format(arg, getattr(args, arg), v))
                    continue
                try:
                    #pass
                    setattr(args, arg, v)
                    print('setting', arg, v)
                except Exception as e:
                    print('help', arg, v, e)
            '''

        logs = cp['logs']
        print('loading logs')
        print('setting args again')
        logs['args'] = {str(arg): getattr(args, arg) for arg in vars(args)}


    else:
        logs = {'epoch': 0,
                'epoch_logs': {'train': [], 'valid': []},
                'epoch_logs_loss': {'train': [], 'valid': []},
                'best_loss': -1,
                'best_metric': None,
                'eta': {'epoch': [],
                        'train': [],
                        'valid': []},
                'args': {str(arg): getattr(args, arg) for arg in vars(args)}}

    #if cp is not None and not args.restart_from_checkpoint:

    sd = None
    print('encoder path', args.encoder_path)
    if os.path.exists(args.encoder_path) and not args.rgb_only:
        print('loading rgbd encoder from: {}'.format(args.encoder_path))
        sd = torch.load(args.encoder_path)
        #print(sd.keys())
        #print(sd['positional_encoding'])
        args.with_positional_encoding = sd.get('with_pe', True)
        args.depth_mean = sd.get('depth_mean')
        args.depth_std = sd.get('depth_std')
        #print('depth_mean',sd.get('depth_mean'), sd.get('depth_std') )

        if 'args' in sd:
            print('Loading args from pretrained encoder')
            for arg, v in sd['args'].items():
                if v != getattr(args, arg):
                    print('Setting arg {}: from {} -> {}'.format(arg, getattr(args, arg), v))
                setattr(args, arg, v)


    # get the depth mean and std if set
    if args.depth_mean < 0 or args.depth_std < 0:
        depth_mean = None
        depth_std = None
    else:
        depth_mean = args.depth_mean
        depth_std = args.depth_std

    print('depth_mean', depth_mean, 'depth_std', depth_std)

    transforms, valid_transforms = get_transforms(args, depth_mean, depth_std)

    
    if args.ds_name == 'photoline':
        from datasets.photoline import Photoline
        print('Creating photoline dataset')
        ds_name = 'LineMod'
        ds = Photoline(mode='train', transforms=transforms, sample_per_epoch=100)
        ds_valid = Photoline(mode='valid', transforms=transforms, sample_per_epoch=1)
        ds_test = Photoline(mode='test', transforms=transforms, sample_per_epoch=1)

    elif args.ds_name == 'LineMod':

        from datasets.linemod import LineModeDataset
        print('Creating LineMod dataset')
        ds_name = 'LineMod'
        ds = LineModeDataset(mode='train', transforms=transforms, load_from_checkpoint=True, with_intr=True,
                             with_mask=True, with_bbox=True, with_pose=True, intr_mat=True, mode_tag='pose')
        ds_valid = LineModeDataset(mode='valid', transforms=valid_transforms, load_from_checkpoint=True,  with_intr=True,
                                   with_bbox=True, with_pose=True, intr_mat=True, with_mask=True, mode_tag='pose')
        ds_test = LineModeDataset(mode='test', transforms=valid_transforms, load_from_checkpoint=True, with_intr=True,
                                  with_mask=True, with_bbox=True, with_pose=True, intr_mat=True, mode_tag='pose')

    elif args.ds_name == 'LM-O':
        from datasets.lm_o import LMODataset
        print('Creating LM-O dataset')
        ds_name = 'LM-O'
        ds = LMODataset(mode='train', transforms=transforms, load_from_checkpoint=True, with_intr=True,
                             with_mask=True, with_bbox=True, with_pose=True, intr_mat=True, mode_tag='pose')
        ds_valid = LMODataset(mode='valid', transforms=valid_transforms, load_from_checkpoint=True,  with_intr=True,
                                   with_bbox=True, with_pose=True, intr_mat=True, with_mask=True, mode_tag='pose')
        ds_test = LMODataset(mode='test', transforms=valid_transforms, load_from_checkpoint=True, with_intr=True,
                                  with_mask=True, with_bbox=True, with_pose=True, intr_mat=True, mode_tag='pose')
    elif args.ds_name == 'Graspnet':
        from datasets.grasp_net import GraspNetDataset
        print('Creating Graspnet dataset')
        ds_name = 'Graspnet'
        ds = GraspNetDataset(mode='train', transforms=transforms, load_from_checkpoint=True, with_intr=True,
                             with_mask=True)
        ds_valid = GraspNetDataset(mode='valid', transforms=valid_transforms, load_from_checkpoint=True,
                                   with_intr=True, with_mask=True)
        ds_test = GraspNetDataset(mode='test', transforms=valid_transforms, load_from_checkpoint=True, with_intr=True,
                                  with_mask=True)
                                  
    elif args.ds_name == 'YCB_video':
        from datasets.ycb_video import YCBVDataset
        print('Creating YCB_video dataset')
        ds_name = 'YCB_video'
        ds = YCBVDataset(mode='train', transforms=transforms, load_from_checkpoint=True, with_intr=True,
                         with_mask=True, p_eval=0.2, with_bbox=True, with_pose=True, intr_mat=True,
                              mode_tag='pose', with_synt_data=True)
        ds_valid = YCBVDataset(mode='valid', transforms=valid_transforms, load_from_checkpoint=True, with_intr=True,
                               with_mask=True, with_bbox=True, with_pose=True, intr_mat=True,
                              mode_tag='pose', with_synt_data=True)
        ds_test = YCBVDataset(mode='test', transforms=valid_transforms, load_from_checkpoint=True, with_intr=True,
                              with_mask=True, with_bbox=True, with_pose=True, intr_mat=True,
                              mode_tag='pose', with_synt_data=True)
    elif args.ds_name == 'homebrew':
        from datasets.homebrew import HomeBrewDataset
        print('Creating YCB_video dataset')
        ds_name = 'homebrew'
        ds = HomeBrewDataset(mode='train', transforms=transforms, load_from_checkpoint=True, with_intr=True,
                             with_mask=True, p_eval=0.2, with_bbox=True, with_pose=True, intr_mat=True,
                             mode_tag='pose', with_synt_data=True)
        ds_valid = HomeBrewDataset(mode='valid', transforms=valid_transforms, load_from_checkpoint=True, with_intr=True,
                                   with_mask=True, with_bbox=True, with_pose=True, intr_mat=True,
                                   mode_tag='pose', with_synt_data=True)
        ds_test = HomeBrewDataset(mode='test', transforms=valid_transforms, load_from_checkpoint=True, with_intr=True,
                                  with_mask=True, with_bbox=True, with_pose=True, intr_mat=True,
                                  mode_tag='pose', with_synt_data=True)

    else:
        raise ValueError('no dataset selected')

    if args.rgb_only:
        ds.with_depth = False
        ds_valid.with_depth = False
        ds_test.with_depth = False

    elif args.with_depth and not args.with_positional_encoding and (depth_mean is None or depth_std is None):
        # get the mean and std of the depth
        depth_mean = []
        depth_std = []
        print('Getting Depth mean and std from the Dataset')
        data_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)
        for j, batch in enumerate(data_loader):
            sys.stdout.write("\r" + "batch {}/{}".format(j + 1, len(data_loader)))
            sys.stdout.flush()
            depth_mean.append(torch.mean(batch['depth'].flatten(1), dim=1))
            depth_std.append(torch.std(batch['depth'].flatten(1), dim=1))
        depth_mean = torch.mean(torch.cat(depth_mean))
        depth_std = torch.mean(torch.cat(depth_std))
        print('\n Using Depth mean, std: {}, {}'.format(depth_mean, depth_std))
        # log the mean and std
        logs['depth_mean'] = float(depth_mean)
        logs['depth_std'] = float(depth_std)
        # rebuild the transforms
        transforms, valid_transforms = get_transforms(args, depth_mean=depth_mean, depth_std=depth_std)
        # reasign the transforms to the ds now including the depth and mean
        ds.transforms = transforms
        ds_valid.transforms = valid_transforms
        ds_test.transforms = valid_transforms

    print('batch_size, ', args.batch_size)
    print('Training transforms: {}'.format(transforms))
    print('Eval transforms: {}'.format(valid_transforms))

    print('Saving to path: {}'.format(path))

    print('{} classes: {}'.format(ds.num_classes, ds.class_names))
    print('Train {} created with "{}" samples.'.format(ds_name, len(ds)))
    print('Valid {} created with "{}" samples.'.format(ds_name, len(ds_valid)))
    print('Test {} created with "{}" samples.'.format(ds_name, len(ds_test)))

    if args.visualize_samples:
        class_names = ['Background'] + ds.class_names + ['Background9'] + ['Background10'] + ['Background11'] + ['Background12'] 
        print(class_names)
        import matplotlib.pyplot as plt
        while True:

            i = np.random.randint(len(ds))
            sample = ds.__getitem__(i)
            print('train', np.unique(np.array(sample['mask'])))
            # print(type(sample['depth']))
            x, y = 3, 3
            plt.subplot(x, y, 1)
            plt.title('Train {}'.format(np.unique(np.array(sample['mask']))))
            plt.imshow(sample['img'])
            plt.subplot(x, y, 2)
            # plt.imshow(np.array(sample['depth']))
            plt.subplot(x, y, 3)
            plt.imshow(sample['mask'])
            uni = np.unique(sample['mask'])
            print('uni is for masks',uni)
            uni = [(u, class_names[u]) for u in uni]
            print(uni)

            i = np.random.randint(len(ds_valid))
            sample = ds_valid.__getitem__(i)
            print('valid', np.unique(np.array(sample['mask'])))
            plt.subplot(x, y, 4)
            plt.title('Valid {}'.format( np.unique(np.array(sample['mask']))))
            plt.imshow(sample['img'])
            plt.subplot(x, y, 5)
            # plt.imshow(sample['depth'])
            plt.subplot(x, y, 6)
            plt.imshow(sample['mask'])
            uni = np.unique(sample['mask'])
            uni = [(u, class_names[u]) for u in uni]
            print(uni)

            i = np.random.randint(len(ds_test))
            sample = ds_test.__getitem__(i)
            print('test', np.unique(np.array(sample['mask'])))
            plt.subplot(x, y, 7)
            plt.title('Test {}'.format( np.unique(np.array(sample['mask']))))
            plt.imshow(sample['img'])
            plt.subplot(x, y, 8)
            # plt.imshow(sample['depth'])
            plt.subplot(x, y, 9)
            plt.imshow(sample['mask'])
            uni = np.unique(sample['mask'])
            uni = [(u, class_names[u]) for u in uni]
            print(uni)
            plt.show()

        return 0

    device = torch.device(args.device)
    #args.num_classes = ds.num_classes + 1 # (+1) adding the background class
    args.n_classes = ds.num_classes

    metric = build_pose_evaluator(args, ds)

    encoder, encoder_channels, decoder_channels = build_encoder(args,
                                                                return_layers=True,
                                                                cat_outs=args.cat_outs,
                                                                resize_layers=False,
                                                                # vit_return_layers=[11] if args.ds_name == 'MVIP' else [8, 9, 10, 11],
                                                                vit_return_layers=[2, 5, 8, 11],
                                                                with_generator=False,
                                                                norm_return_layers=args.norm_return_layers,
                                                                norm_return_layers_rgb=args.norm_return_layers_rgb)

    #args.freeze_encoder = False
    model, criterion, matcher = build(args, encoder)

    if sd is not None:
        print('loading pretrained encoder from {}'.format(args.encoder_path))
        model.backbone[0].model = load_fitting_state_dict(model.backbone[0].model, sd['state_dict'])

    print('cp', type(cp))
    if cp is not None:
        print('loading model state dict')
        model = load_fitting_state_dict(model, cp['state_dict'])
    
    device_ids = None
    if args.multi_gpu and torch.cuda.device_count() > 1 and args.device != 'cpu':
        device_ids = None  
        if args.gpu_ids:
            print('getting device ids from {}'.format(args.gpu_ids))
            device_ids = [int(gpu.replace(' ', '')) for gpu in args.gpu_ids.split(',')]
            n_gpus = len(device_ids)
        else:
            n_gpus = torch.cuda.device_count()

        print('n gpus {}, device_ids {}'.format(n_gpus, device_ids))
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        n_gpus = 1
    model = model.to(device)

    data_loader, valid_data_loader = get_dataloaders(args, ds, ds_valid, logs['epoch'])
    #metric = IoU(num_classes=args.num_classes)
    #metric_valid = IoU(num_classes=args.num_classes)
    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if 'backbone' in n and p.requires_grad],
            "lr": args.lr_encoder,
        },
        {
            "params":
                [p for n, p in model.named_parameters()
                 if 'backbone' not in n and p.requires_grad and (
                    'reference_points' in n or 'sampling_offsets' in n
                 )],
            "lr": args.lr_linear_project,
        },
        {
            "params": [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad and
                       'reference_points' not in n and 'sampling_offsets' not in n],
            "lr": args.lr_decoder,
        }
    ]

    print('Distribution of {} Params:'.format(
        sum([len(param_dicts[i]['params']) for i in range(len(param_dicts))])))
    print('     Encoder: {} with lr: {}'.format(len(param_dicts[0]['params']), args.lr_encoder))
    print('     Linear Project: {} with lr: {}'.format(len(param_dicts[1]['params']), args.lr_linear_project))
    print('     Decoder: {} with lr: {}'.format(len(param_dicts[2]['params']), args.lr_decoder))

    if args.freeze_encoder:
        param_dicts = param_dicts[1:]
    #else:
    #    print('unfreeze')
    #    pams = []
    #    for p in param_dicts[0]['params']:
    #        p.requires_grad = True
    #        pams.append(p)
    #    param_dicts[0]['params'] = pams

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=param_dicts, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=param_dicts, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params=param_dicts, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError('the optimizer {} is not implemented'.format(args.optimizer))

    if args.one_cycle and False:
        print('OneCycleLR scheduler')
        scheduler = schedulers.OneCycleLR(optimizer=optimizer,
                                          max_lr=[args.lr_encoder,  args.lr_decoder, args.lr],
                                          total_steps=args.epochs*len(data_loader),
                                          pct_start=args.pct_start, div_factor=args.div_factor,
                                          final_div_factor=args.final_div_factor)

    elif args.one_cycle:
        print('CosineAnnealingLR scheduler')
        scheduler = schedulers.CosineAnnealingLR(optimizer, args.epochs * len(data_loader))
    elif args.lr_drop > 0:
        if args.lr_drop < 1:
            args.lr_drop = int(args.epochs * args.lr_drop)
        print('lr step at epoch {}'.format(args.lr_drop))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    else:
        scheduler = None

    if args.enable_resize_attention_maps <= args.epochs:
        print('Enable Resize Attention Maps at epoch {}'.format(args.enable_resize_attention_maps))

    if cp is not None and not args.restart_from_checkpoint:
        try:
            optimizer.load_state_dict(cp['optimizer'])
            print('loading optimizer')
            if scheduler is not None and 'scheduler' in cp:
                scheduler.load_state_dict(cp['scheduler'])
                print('loading scheduler')
        except:
            pass
    del cp

    # Evaluate the model for the BOP challenge
    if args.eval_bop:
        if args.ds_name == 'LM-O':
            obj_id_map = {1: 1, 2: 5, 3: 6, 4: 8, 5: 9, 6: 10, 7: 11, 8: 12}
        else:
            obj_id_map = {}
            
        # print(args.dataset)
        print('#### loading best state ####')
        sd = torch.load(os.path.join(path, 'best_checkpoint.ckpt'))
        if isinstance(model, nn.DataParallel):
            model = model.module
            print('model from dataparalell')
        model = model.to('cpu')
        print('model to cpu')

        print('best epoch: {}'.format(sd['logs']['epoch']))
        model = load_fitting_state_dict(model, sd['state_dict'])

        if args.multi_gpu and torch.cuda.device_count() > 1 and args.device != 'cpu':
            model = nn.DataParallel(model, device_ids=device_ids)
            print('model to data paralell, device ids {}'.format(device_ids))
            #model = nn.DataParallel(model)
        print('model to device, {}'.format(device))
        model = model.to(device)

        print('##### Start Test #####')
        del valid_data_loader
        data_loader_test = torch.utils.data.DataLoader(dataset=ds_test,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                num_workers=args.num_workers,
                                                drop_last=False,
                                                collate_fn=collate_fn)

        # input()
        losses_valid = []
        
        
        output_eval_dir = args.output_path + "/bop_" + '6D' + "/"
        Path(output_eval_dir).mkdir(parents=True, exist_ok=True)

        out_csv_path = os.path.join(output_eval_dir + 'dinorgb_lmo-test.csv')
        # out_csv_file.write("scene_id,im_id,obj_id,score,R,t,time")
        with open(out_csv_path, 'w') as out_csv_file:
            out_csv_file.write("scene_id,im_id,obj_id,score,R,t,time")
            for step, batch in enumerate(data_loader_test):
                if step == 4 and args.debug:
                    break

                #if step > 50:
                #    break

                # push data to device
                batch['x'] = batch['x'].to(device)
                #print('tensors', torch.mean(batch['x'].tensors), torch.std(batch['x'].tensors))
                #print('mask', torch.mean(batch['x'].mask), torch.std(batch['x'].mask))
                #print('depth', torch.mean(batch['x'].depth), torch.std(batch['x'].depth))
                
                # print('target keys are before label wrapper')
                # for t in batch['target']:
                #     print(t.keys())
                    
                
                y = LabelWrapper(batch['target']).to(device)

                # targets = y.get_data()
                # print('target keys are after label wrapper')
                # for t in targets:
                #     print(t.keys())

                # input()
                pred_start_time= time.time()
                with torch.no_grad():
                    outputs, n_boxes_per_sample = model(batch['x'].decompose(), y.split_data(n_gpus))

                pred_end_time = time.time() - pred_start_time
                loss_dict = criterion(outputs, y.to(device).get_data(), n_boxes_per_sample)
                weight_dict = criterion.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                losses_valid.append(float(loss.item()))
                bop_evaluate(model, matcher, data_loader_test, out_csv_file, n_boxes_per_sample, outputs, y.to(device).get_data(), pred_end_time, obj_id_map)

        return


    print('##### Start Training #####')
    t_epoch = time.time()
    print(logs['epoch'], args.epochs)
    for epoch in range(logs['epoch'], args.epochs):
        if args.debug and epoch > 0:
            break
        logs['epoch'] = epoch + 1
        losses = []
        if epoch >= args.enable_resize_attention_maps:
            if isinstance(model, nn.DataParallel):
                model = model.module
            model.backbone[0].resize_attention_maps = True
            if args.multi_gpu and torch.cuda.device_count() > 1 and args.device != 'cpu':
                device_ids = None
                if args.gpu_ids:
                    device_ids = [int(gpu.replace(' ', '')) for gpu in args.gpu_ids.split(',')]
                model = nn.DataParallel(model, device_ids=device_ids)
            model = model.to(device)
            data_loader, valid_data_loader = get_dataloaders(args, ds, ds_valid, logs['epoch'])

        t_train = time.time()
        model.train()
        matcher.eval()
        metric.reset()
        for step, batch in enumerate(data_loader):
            if step == 4 and args.debug:
                break
            # push data to device
            batch['x'] = batch['x'].to(device)
            #print(batch['x'].tensors.shape)

            y = LabelWrapper(batch['target']).to(device)
            
            #print('size', batch['x'].tensors.shape, batch['x'].depth.shape)
            outputs, n_boxes_per_sample = model(batch['x'].decompose(), y.split_data(n_gpus))

            optimizer.zero_grad()
            # compute loss and metric scores

            loss_dict = criterion(outputs, y.to(device).get_data(), n_boxes_per_sample)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # backward pass and stepping
            #with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()
            if args.one_cycle:
                scheduler.step()

            # adding logs
            losses.append(float(loss.item()))
            #eval_pose_add(metric, outputs, matcher, batch['target'], n_boxes_per_sample)

            logs['eta']['train'].append(time.time()-t_train)
            if len(logs['eta']['train']) > args.eta_points:
                logs['eta']['train'] = logs['eta']['train'][1:]

            t_train = time.time()

            if scheduler is not None:
                current_lr = float(scheduler.get_last_lr()[-1])
            else:
                current_lr = args.lr
            bar_progress_pose('train',
                             epoch + 1,
                             args.epochs,
                             step+1,
                             len(data_loader),
                             0,
                             len(valid_data_loader),
                             current_lr,
                             losses,
                             logs['epoch_logs_loss']['train'][-1] if len(logs['epoch_logs_loss']['train']) > 0 else None,
                             logs['eta'])

        logs['epoch_logs_loss']['train'].append(float(np.mean(losses)))
        if not args.one_cycle and args.lr_drop > 0:
            scheduler.step()

        print('')  # next line in the progress bar
        #eval_pose_resutls(metric, path + '/train_logs/')

        t_valid = time.time()
        model.eval()
        matcher.eval()
        losses_valid = []
        metric.reset()
        for step, batch in enumerate(valid_data_loader):
            if step == 4 and args.debug:
                break

            # push data to device
            batch['x'] = batch['x'].to(device)

            y = LabelWrapper(batch['target']).to(device)
            with torch.no_grad():
                outputs, n_boxes_per_sample = model(batch['x'].decompose(), y.split_data(n_gpus))

            loss_dict = criterion(outputs, y.to(device).get_data(), n_boxes_per_sample)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            losses_valid.append(float(loss.item()))
            eval_pose_add(args, metric, outputs, matcher, y.to(device).get_data(), n_boxes_per_sample)

            logs['eta']['valid'].append(time.time() - t_valid)
            if len(logs['eta']['valid']) > args.eta_points:
                logs['eta']['valid'] = logs['eta']['valid'][1:]
            t_valid = time.time()

            bar_progress_pose('valid',
                             epoch + 1,
                             args.epochs,
                             0,
                             len(data_loader),
                             step + 1,
                             len(valid_data_loader),
                             None,
                             losses_valid,
                             logs['epoch_logs_loss']['valid'][-1] if len(logs['epoch_logs_loss']['valid']) > 0 else None,
                             logs['eta'])

        current_loss = float(np.mean(losses_valid))
        logs['epoch_logs_loss']['valid'].append(current_loss)
        best_loss = False

        if current_loss < logs['best_loss'] or logs['best_loss'] == -1:
            logs['best_loss'] = current_loss
            best_loss = True

        ckpt = {'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'logs': logs}

        if scheduler is not None:
            ckpt['scheduler'] = scheduler.state_dict()

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(ckpt, os.path.join(path, 'current_checkpoint.ckpt'))
        with open(os.path.join(path, 'logs.json'), 'w') as f:
            json.dump(logs, f)

        if best_loss:
            torch.save(ckpt, os.path.join(path, 'best_checkpoint.ckpt'))

        print('')  # next line in the progress bar
        eval_pose_resutls(metric, path + '/epoch_{}/valid_logs/'.format(logs['epoch']))
        #torch.save(ckpt, os.path.join(path, 'epoch_{}'.format(logs['epoch']), 'current_checkpoint.ckpt'))

        logs['eta']['epoch'].append(time.time() - t_epoch)
        if len(logs['eta']['epoch']) > args.eta_points:
            logs['eta']['epoch'] = logs['eta']['epoch'][1:]
        t_epoch = time.time()
        del ckpt


    print('#### loading best state ####')
    sd = torch.load(os.path.join(path, 'best_checkpoint.ckpt'))
    if isinstance(model, nn.DataParallel):
        model = model.module
        print('model from dataparalell')
    model = model.to('cpu')
    print('model to cpu')

    print('best epoch: {}'.format(sd['logs']['epoch']))
    model = load_fitting_state_dict(model, sd['state_dict'])

    if args.multi_gpu and torch.cuda.device_count() > 1 and args.device != 'cpu':
        device_ids = None
        if args.gpu_ids:
            device_ids = [int(gpu.replace(' ', '')) for gpu in args.gpu_ids.split(',')]
        model = nn.DataParallel(model, device_ids=device_ids)
        print('model to data paralell, device ids {}'.format(device_ids))

    print('model to device, {}'.format(device))
    
    model = model.to(device)

    print('##### Start Test #####')
    del valid_data_loader
    data_loader = torch.utils.data.DataLoader(dataset=ds_test,
                                              batch_size=args.test_batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              collate_fn=collate_fn)
    t_valid = time.time()
    model.eval()
    matcher.eval()
    losses_valid = []
    metric.reset()

    

    for step, batch in enumerate(data_loader):
        if step == 4 and args.debug:
            break

        #if step > 50:
        #    break

        # push data to device
        batch['x'] = batch['x'].to(device)
        #print('tensors', torch.mean(batch['x'].tensors), torch.std(batch['x'].tensors))
        #print('mask', torch.mean(batch['x'].mask), torch.std(batch['x'].mask))
        #print('depth', torch.mean(batch['x'].depth), torch.std(batch['x'].depth))
        y = LabelWrapper(batch['target']).to(device)
        with torch.no_grad():
            outputs, n_boxes_per_sample = model(batch['x'].decompose(), y.split_data(n_gpus))

        loss_dict = criterion(outputs, y.to(device).get_data(), n_boxes_per_sample)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        losses_valid.append(float(loss.item()))
        eval_pose_add(args, metric, outputs, matcher, y.to(device).get_data(), n_boxes_per_sample)

        logs['eta']['valid'].append(time.time() - t_valid)
        if len(logs['eta']['valid']) > args.eta_points:
            logs['eta']['valid'] = logs['eta']['valid'][1:]
        t_valid = time.time()
        bar_progress_test2(step+1, len(data_loader),
                           float(np.mean(logs['eta']['valid'])),
                           float(np.mean(losses_valid)), None)

    print('')  # next line in the progress bar
    eval_pose_resutls(metric, path + '/test_logs/', test=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vanishing Depth training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)