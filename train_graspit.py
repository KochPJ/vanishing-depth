
# ddp_train_mixed.py

import os
import math
import json
import time
import random
import argparse
from collections import defaultdict
import sys
import csv
import cv2
from datetime import datetime
import numpy as np 

import torch



import copy
from PIL import Image, ImageDraw
#torch.autograd.set_detect_anomaly(True)

from typing import Any, Dict, List, Optional, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
torch.set_num_threads(1)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets.multi_obj_sampler import (
    MixedMultiDataset,
    GraspItDataset,
    BOPDataset,
    GraspNetDataset,
    set_seed,
)
from model.vanishing_depth import build_encoder  # if you share the encoder building logic
from model.deformable_detr.grasp_estimation_transformer import build_grasper

from model.deformable_detr.util.misc import nested_tensor_from_tensor_list, is_main_process
from model.deformable_detr.grasping_loss import JointCriterion
from datasets.dataloader_utils import build_mixer, collate_mixed_batch, reduce_dict, _draw_gripper
import hashlib
#import torch.multiprocessing as mp
#mp.set_start_method("spawn", force=True)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _HAS_COCO = True
except Exception:
    _HAS_COCO = False

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

GRASPIT_ROOT = os.environ.get('GRASPIT_ROOT', '/mnt/kikerp/OptiSim')
GRASPIT_ASSETS = os.environ.get('GRASPIT_ASSETS', os.path.join(GRASPIT_ROOT, 'assets'))
GRASPNET_ROOT = os.environ.get('GRASPNET_ROOT', '/mnt/kikerp/publicdata/GraspNet-1Billion')
BOP_ROOT = os.environ.get('BOP_ROOT', 'data/bop')
GRASPNET_MODELS_ROOT = os.environ.get('GRASPNET_MODELS_ROOT', os.path.join(GRASPNET_ROOT, 'models'))
print('GRASPIT_ROOT', GRASPIT_ROOT)
print('GRASPIT_ASSETS', GRASPIT_ASSETS)
print('GRASPNET_ROOT', GRASPNET_ROOT)
print('BOP_ROOT', BOP_ROOT)
print('GRASPNET_MODELS_ROOT', GRASPNET_MODELS_ROOT)


def parse_args():
    p = argparse.ArgumentParser("DDP Training for Deformable DETR/Grasper on MixedMultiDataset", add_help=True)

    # DDP args
    p.add_argument("--dist_url", default="env://", type=str)
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--rank", default=0, type=int)
    p.add_argument("--local_rank", default=-1, type=int)

    # Roots
    p.add_argument("--graspit_root", default=GRASPIT_ROOT, type=str)
    p.add_argument("--graspit_assets", default=GRASPIT_ASSETS, type=str)
    p.add_argument("--bop_root", default=BOP_ROOT, type=str)
    p.add_argument("--graspnet_root", default=GRASPNET_ROOT, type=str)
    p.add_argument("--graspnet_models_root", default=GRASPNET_MODELS_ROOT, type=str)

    # What to include in datasets
    p.add_argument("--include_depth", default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--include_mask", default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--include_models", default=True, type=lambda x: str(x).lower() == 'true')

    # Mixer setup
    p.add_argument("--use_bop", default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--use_graspit", default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--use_bop_val", default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--use_graspit_val", default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--use_graspnet_val", default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--use_graspnet_val_only", default=True, type=lambda x: str(x).lower() == 'true')   

    #p.add_argument("--dataset_probs", default="bop:0.0,graspit:1.0", type=str)
    p.add_argument("--dataset_probs", default="bop:0.5,graspit:0.5", type=str)
    p.add_argument("--num_samples_per_epoch", default=100000, type=int) #100000 #200000 #200000
    p.add_argument("--num_val_samples_per_epoch", default=5000, type=int) #5000

    p.add_argument("--warmup_steps", default=20000000, type=int) #5000
    p.add_argument("--start_warmup_step", default=20000000, type=int) #5000

    p.add_argument("--val_subset_size", default=2000, type=int, help="number of validation samples per eval")
    p.add_argument("--train_height", default=490, type=int, help="Number of height pixels in training")
    p.add_argument("--train_width", default=644, type=int, help="Number of width pixels in training")
    p.add_argument("--n_images_saved", default=3, type=int, help="number of result plots saved per epoch and val")
       
    # Visualization / debugging
    p.add_argument("--viz_train", default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--viz_val", default=True, type=lambda x: str(x).lower() == 'true')

    # Training
    p.add_argument("--epochs", default=150, type=int)
    p.add_argument("--batch_size", default=2, type=int) #24
    p.add_argument("--num_workers", default=2, type=int) #16
    p.add_argument("--lr_backbone", default=None, type=float, help='The image encoder backbone')
    p.add_argument("--lr_proj", default=1e-5, type=float, help='The image encoder backbone')
    p.add_argument("--lr_tfencode", default=2e-5, type=float, help='the deformable encoder')
    p.add_argument("--lr_det", default=None, type=float, help='the object detection transformer decoder')
    p.add_argument("--lr_obj", default=None, type=float, help='The object estimation transformer decoder')
    p.add_argument("--lr_pose", default=1e-5, type=float, help='The object estimation transformer decoder')
    p.add_argument("--lr_critique", default=None, type=float, help='The object estimation transformer decoder')
    p.add_argument("--lr_other", default=None, type=float, help='Remaining untracked lernable params')
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--lr_drop", default=0.9, type=int)
    p.add_argument("--apply_vis_mask_to_shape", default=False, type=lambda x: str(x).lower() == 'true')
    
    p.add_argument("--opt", default="adamw", type=str, choices=["adamw", "adam", "sgd"])    

    # Model flags
    p.add_argument("--hidden_dim_det", default=256, type=int) 
    p.add_argument("--hidden_dim_obj", default=512, type=int) 
    p.add_argument("--nheads", default=8, type=int)
    p.add_argument("--enc_layers", default=6, type=int)
    p.add_argument("--dec_layers", default=6, type=int)
    p.add_argument("--dim_feedforward", default=1024, type=int)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--num_feature_levels", default=4, type=int)
    p.add_argument("--proj_kernel_size", default=1, type=None)  # 1 for vits you might want to use "3k,3k,1k,1k" with upsampling put "2u0k, 1u0k, 1k, 1k" 
    p.add_argument("--backbone_proj_and_upsampling", default="0p0u0k, 0p0u0k, 0p0u0k, 0p0u0k", type=None)  # 1 for vits you might want to use "3k,3k,1k,1k" with upsampling put "2u0k, 1u0k, 1k, 1k" 
    p.add_argument("--dec_n_points", default=4, type=int)
    p.add_argument("--enc_n_points", default=4, type=int)
    p.add_argument("--num_det_queries", default=300, type=int)
    p.add_argument("--num_obj_queries", default=7, type=int)
    p.add_argument("--num_grasp_queries", default=3, type=int)   
    p.add_argument("--max_objects", default=30, type=int)
    p.add_argument("--num_grasp_poses", default=5, type=int)
    p.add_argument("--n_classes", default=1, type=int, help="detection classes (object-agnostic => 1)")
    p.add_argument('--position_embedding', default='sine', type=str)
    
    # Toggle modes
    p.add_argument("--with_det", default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--with_obj", default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--with_grasp", default=True, type=lambda x: str(x).lower() == 'true') # loading of the grasps in the dataloader
    p.add_argument("--with_pose", default=True, type=lambda x: str(x).lower() == 'true') # actuall pose (6d or grasp) in the decoder
    p.add_argument("--with_critique", default=False, type=lambda x: str(x).lower() == 'true') # scoring how good the poses are. 
    p.add_argument("--add_pose_extra_query_embed", default=True, type=lambda x: str(x).lower() == 'true') # adds the same amount of queries as the obj queries to the critique if with_obj==False
    p.add_argument("--add_critique_extra_query_embed", default=True, type=lambda x: str(x).lower() == 'true') # adds the same amount of queries as the obj queries to the pose estimation if with_obj==False
    p.add_argument("--add_model_embed", default=False, type=lambda x: str(x).lower() == 'true') # if true we process 6d object pose estimation else grasping poses
    

    #backbone     
    p.add_argument('--fpn_layers', default=4, type=int)
    p.add_argument('--resize_attention_maps', default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--freeze_encoder', default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--norm_return_layers', default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--norm_return_layers_rgb', default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--cat_outs', default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--rgb_only', default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--model_name', default='Dino', type=str) # Dino; EVAv2; OmniVore, Dino , ResNet
    p.add_argument('--model_version', default='V2-base', type=str) #, # V2-base,  'base', 'small', 'large' , V2-base ,50
    p.add_argument('--return_rdps', default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--with_depth', default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--with_intr', default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--pretrained_imagenet', default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--pretrained_path', default='./data/backbones/dinov2_vitb14_pretrain.pth', type=str) 
    p.add_argument('--pretrained_deform_detr', default='./data/backbones/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth', type=str) 
    p.add_argument('--pretrained_rgbd', default='./data/trained_models/20251026_1761460596_Dino_HalfPerlin448_encoder.plt', type=str) 
    
    
    # depth encoding
    p.add_argument('--with_positional_depth_encoding', default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--depth_channels', default=32, type=int) #12 3d
    p.add_argument('--temperature', default=0.0003, type=int) #1200, 14000 for 15m 3000 for 3d
    p.add_argument('--position_offset', default=0.0, type=float)
    p.add_argument('--scale', default=2*math.pi, type=float)
    p.add_argument('--to_meter', default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--normalize_depth', default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--encode_disparity', default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument('--depth_std', default=0.0, type=float)
    p.add_argument('--depth_mean', default=0.0, type=float)
        
    # Loss weights
    p.add_argument("--loss_bbox_coef", default=5.0, type=float)
    p.add_argument("--loss_giou_coef", default=2.0, type=float)
    p.add_argument("--loss_obj_scale_coef", default=1.0, type=float)
    p.add_argument("--loss_obj_pos_coef", default=1.0, type=float)
    p.add_argument("--loss_obj_vis_coef", default=0.5, type=float)
    p.add_argument("--loss_obj_shape_coef", default=1.0, type=float)
    p.add_argument("--aux_loss", default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--aux_det_match_reuse", default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--aux_obj_match_reuse", default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--dataloader_timeout", default=60, type=int, help="seconds to wait for a batch before skipping")
    p.add_argument("--max_fetch_retries", default=5, type=int, help="consecutive fetch retries before rebuilding loader")

    # Shape loss sampling
    p.add_argument("--shape_gt_subsample", default=512, type=int, help="subsample GT points for shape loss")
    p.add_argument("--find_unused_parameters", default=True, type=lambda x: str(x).lower() == 'true')

    # Freezing / LR groups
    p.add_argument("--freeze_backbone", default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--freeze_det_heads", default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--freeze_obj_heads", default=False, type=lambda x: str(x).lower() == 'true')

    # I/O
    p.add_argument("--output_dir", default="./data/grasper/pose_no_obj_2", type=str)
    p.add_argument("--resume", default="", type=str)
    #p.add_argument("--resume", default="./data/grasper/pose_no_obj/checkpoint.pth", type=str)    
    p.add_argument("--restart", default=False, type=lambda x: str(x).lower() == 'true')
    #p.add_argument("--restart", default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--load_coco_pretrained", default=True, type=lambda x: str(x).lower() == 'true')   
    

    # Misc
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--print_freq", default=50, type=int)
    p.add_argument("--force_cpu", default=False, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--use_compile", default=False, type=lambda x: str(x).lower() == 'true')
    

    # valid and logging
    p.add_argument("--val_subset_seed", default=-1, type=int, help="seed for selecting validation subset; -1 uses args.seed")
    p.add_argument("--eval_every", default=1, type=int, help="run validation every N epochs")
    p.add_argument("--val_score_thresh", default=0.05, type=float)
    p.add_argument("--log_file", default="train_log.json", type=str, help="per-epoch JSON log in output_dir")
    p.add_argument("--csv_file", default="train_dumps.csv", type=str, help="per-epoch JSON log in output_dir")
    p.add_argument("--log_csv_every", default=50, type=int, help="write progress row to CSV every k steps")

    # WnB logging

    p.add_argument("--use_wandb", default=True, type=lambda x: str(x).lower() == 'true')
    p.add_argument("--wandb_project", default="graspit", type=str)
    p.add_argument("--wandb_run_name", default="graspit_detection_pretraining", type=str)
    p.add_argument("--wandb_entity", default="", type=str)
    p.add_argument("--wandb_mode", default="online", type=str, choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_group", default="", type=str)
    p.add_argument("--wandb_tags", default="", type=str, help="comma-separated tags")
    p.add_argument("--wandb_dir", default="./data/wandb", type=str, help="wandb_dir")

    
    return p.parse_args()


# Visualization helpers for pose-baked point clouds

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _denorm_to_pil(img_t):
    import torchvision.transforms.functional as TF
    img = img_t.detach().cpu()
    m = torch.tensor(IMAGENET_MEAN, dtype=img.dtype)[:, None, None]
    s = torch.tensor(IMAGENET_STD, dtype=img.dtype)[:, None, None]
    img = (img * s) + m
    img = img.clamp(0, 1)
    return TF.to_pil_image(img)

def _get_pred_obj_keys(outs: Dict[str, Any]):
    """
    Try to discover predicted object properties in outs.
    Returns (points_key, pos_key, scale_key) or (None, None, None) if not found.

    - points_key: [B, Q, P, 3]
    - pos_key:    [B, Q, 3]

    - scale_key:  [B, Q] or [B, Q, 1]

    """
    points_key = None
    pos_key = None
    scale_key = None

    # Points candidates
    for k, v in outs.items():
        try:
            if isinstance(v, torch.Tensor) and v.dim() == 4 and v.size(-1) == 3:
                # guess it's [B,Q,P,3]
                points_key = k
                break
        except Exception:
            pass

    # Position candidates
    for k, v in outs.items():
        try:
            if isinstance(v, torch.Tensor) and v.dim() == 3 and v.size(-1) == 3:
                pos_key = k
                break
        except Exception:
            pass

    # Scale candidates
    for k, v in outs.items():
        try:
            if isinstance(v, torch.Tensor) and v.dim() in (2, 3):
                if v.dim() == 2 and v.size(-1) >= 1:
                    scale_key = k
                    break
                if v.dim() == 3 and v.size(-1) == 1:
                    scale_key = k
                    break
        except Exception:
            pass

    return points_key, pos_key, scale_key

# Add to ddp_train_mixed.py

def _quat_wxyz_to_R_np(q):
    # q: [w,x,y,z]
    w, x, y, z = [float(v) for v in q]
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w*X, w*Y, w*Z
    xX, xY, xZ = x*X, x*Y, x*Z
    yY, yZ = y*Y, y*Z
    zZ = z*Z
    R = np.array([
        [1.0 - (yY + zZ),      xY - wZ,           xZ + wY],
        [xY + wZ,              1.0 - (xX + zZ),   yZ - wX],
        [xZ - wY,              yZ + wX,           1.0 - (xX + yY)]
    ], dtype=np.float32)
    return R

def _iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2]
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _overlay_best_object_pose_on_image(img_pil, outs, batch, bi, det_to_slot, det_xyxy, colors, intr_list):
    """
    Draw object model points under the best predicted pose per detection-slot:
    - Best pose by critique score (fallback to pose confidence).
    - slot → inst_id via det_to_inst passed in det_xyxy (or recomputed from pb/gt).
    - If models[inst_id] available, transform by predicted (R,t); else fallback to predicted obj_points.

    det_xyxy must include:
      - 'sel_indices': list[int]
      - 'sel_colors': list[(r,g,b)]
      - 'pb_xyxy': list[xyxy]    (optional if det_to_inst provided)
      - 'gt_xyxy': list[xyxy]    (optional if det_to_inst provided)
      - 'box2inst': Tensor[N]    (optional if det_to_inst provided)
      - 'det_to_inst': dict[int->inst_id]

    """
    try:
        pose_preds = outs.get("pose_preds", None)  # [L,B,K,Np,8] or [B,K,Np,8]
        if not isinstance(pose_preds, torch.Tensor):
            return None
        pose_preds = pose_preds[-1] if pose_preds.dim() == 5 else pose_preds  # [B,K,Np,8]
        K, Np = pose_preds.shape[1], pose_preds.shape[2]

        # Scores (critique preferred)
        p_good = None
        crit_logits = outs.get("critique_pred_logits", None)
        crit_mask   = outs.get("critique_pred_mask", None)
        if isinstance(crit_logits, torch.Tensor) and isinstance(crit_mask, torch.Tensor):
            crit_logits_b = crit_logits[-1, bi] if crit_logits.dim() == 4 else crit_logits[bi]  # [T,2]
            crit_mask_b   = (crit_mask[-1, bi] if crit_mask.dim() == 3 else crit_mask[bi]) != 0
            valid_logits  = crit_logits_b[crit_mask_b]  # [P,2]
            Pexp = K * Np
            if valid_logits.shape[0] >= Pexp:
                valid_logits = valid_logits[:Pexp]
                p_good = torch.softmax(valid_logits, dim=-1)[:, 1].view(K, Np)  # [K,Np]
        if p_good is None:
            p_good = torch.sigmoid(pose_preds[bi, :, :, 7])  # [K,Np]

        # Rebuild det->slot if empty
        if (not det_to_slot) or len(det_to_slot) == 0:
            obj_index = outs.get("obj_index", None)
            if isinstance(obj_index, torch.Tensor):
                det_to_slot = {}
                idx_b = obj_index[bi]
                seen = set()
                for det_q in [int(x) for x in idx_b.detach().flatten().cpu().tolist()]:
                    if det_q not in seen:
                        det_to_slot[det_q] = len(det_to_slot)
                        seen.add(det_q)

        # det->inst mapping (prefer from det_xyxy)
        det_to_inst = det_xyxy.get("det_to_inst", {}) or {}
        if not det_to_inst:
            pb_xyxy = det_xyxy.get("pb_xyxy", [])
            gt_xyxy = det_xyxy.get("gt_xyxy", [])
            box2inst = det_xyxy.get("box2inst", None)
            det_to_inst = {}
            if pb_xyxy and gt_xyxy and box2inst is not None:
                for qi, det_box in enumerate(pb_xyxy):
                    best_gt, best_iou = -1, 0.0
                    for gi, gbox in enumerate(gt_xyxy):
                        iou = _iou_xyxy(det_box, gbox)
                        if iou > best_iou:
                            best_iou, best_gt = iou, gi
                    if best_gt >= 0 and best_iou > 0.1:
                        det_to_inst[qi] = int(box2inst[best_gt].item())

        # GT models and predicted object outputs (final layer)
        models = batch[bi].get("models", {}) or {}
        obj_pts = outs.get("obj_points", None)      # [B,K,D,3]
        obj_pos = outs.get("obj_positions", None)   # [B,K,3]
        obj_sc  = outs.get("obj_scale", None)       # [B,K,1] or [B,K]

        out_img = img_pil.copy()
        sel_indices = det_xyxy.get("sel_indices", [])
        sel_colors  = det_xyxy.get("sel_colors", [])

        for j, (qi, color) in enumerate(zip(sel_indices, sel_colors)):
            slot = det_to_slot.get(int(qi), None)
            if slot is None or not (0 <= slot < K):
                continue

            # Best pose by score
            best_i = int(torch.argmax(p_good[slot]).item())
            pose_k = pose_preds[bi, slot, best_i]  # [8]
            t_pred = pose_k[:3].detach().cpu().numpy()
            q_pred = pose_k[3:7].detach().cpu().numpy().tolist()
            R_pred = _quat_wxyz_to_R_np(q_pred)

            pts_cam = None
            inst_id = det_to_inst.get(int(qi), None)
            # Prefer GT model if available
            if isinstance(inst_id, int) and (inst_id in models) and isinstance(models[inst_id], torch.Tensor):
                Pm = models[inst_id].detach().cpu().numpy().astype(np.float32)  # [N,3]
                pts_cam = (Pm @ R_pred.T) + t_pred.reshape(1, 3)
            # Fallback: predicted object points
            elif isinstance(obj_pts, torch.Tensor) and isinstance(obj_pos, torch.Tensor) and isinstance(obj_sc, torch.Tensor):
                Pn = obj_pts[bi, slot].detach().cpu().numpy().astype(np.float32)
                pos = obj_pos[bi, slot].detach().cpu().numpy().astype(np.float32)
                sc  = float(obj_sc[bi, slot].item() if obj_sc.dim() == 3 else obj_sc[bi, slot].item())
                pts_cam = Pn * sc + pos
                # If your obj_points are canonical-space, uncomment:
                # pts_cam = (Pn @ R_pred.T) + t_pred.reshape(1, 3)

            if pts_cam is not None and intr_list is not None:
                uv = _project_cam_points(pts_cam, intr_list)
                out_img = _overlay_points(out_img, uv, color=tuple(color))
        return out_img
    except Exception as e:
        print(f"[save_last_batch_plots] object pose overlay failed: {e}")
        return img_pil


def _overlay_best_grasps_on_image(img_pil, outs, bi, det_to_slot, det_xyxy, colors, intr_list):
    """
    Draw the best predicted grasp per detection-slot on img_pil.
    Best = argmax of critique score (softmax(logits)[...,1]) per slot; fallback to pose confidence.
    det_xyxy must include:
      - 'sel_indices': list[int]
      - 'sel_colors': list[(r,g,b)]
      - 'pb_xyxy': list[xyxy] (pred boxes abs)
      - 'gt_xyxy': list[xyxy] (GT boxes abs)
      - 'box2inst': Tensor[N] (GT idx -> inst_id)
      - 'det_to_inst': dict[int->inst_id] (pred det idx -> inst_id)


    """
    try:        
        pose_preds = outs.get("pose_preds", None)  # [L,B,K,Np,8] or [B,K,Np,8]
        if not isinstance(pose_preds, torch.Tensor):
            return None
        # Only take last layer if 5D
        pose_preds = pose_preds[-1] if pose_preds.dim() == 5 else pose_preds
        if pose_preds.dim() != 4:
            print(f"[save_last_batch_plots] unexpected pose_preds dim={pose_preds.dim()}, shape={tuple(pose_preds.shape)}")
            return None

        B, K, Np, _ = pose_preds.shape

        # Critique scores (preferred), else use conf logit
        #p_good = None
        #crit_logits = outs.get("critique_pred_logits", None)
        #crit_mask   = outs.get("critique_pred_mask", None)
        #if isinstance(crit_logits, torch.Tensor) and isinstance(crit_mask, torch.Tensor):
        #    crit_logits_b = crit_logits[-1, bi] if crit_logits.dim() == 4 else crit_logits[bi]  # [T,2]
        #    crit_mask_b   = (crit_mask[-1, bi] if crit_mask.dim() == 3 else crit_mask[bi]) != 0
        #    valid_logits  = crit_logits_b[crit_mask_b]  # [P,2]
        #    Pexp = K * Np
        #    if valid_logits.shape[0] >= Pexp:
        #        valid_logits = valid_logits[:Pexp]
        #        p_good = torch.softmax(valid_logits, dim=-1)[:, 1].view(K, Np)  # [K,Np]
        #if p_good is None:
        #    conf_logit = pose_preds[bi, :, :, 7]  # [K,Np]
        #    p_good = torch.sigmoid(conf_logit)

        
        # Critique scores (preferred), else use conf logit
        p_good = None
        crit_logits = outs.get("critique_pred_logits", None)
        crit_mask   = outs.get("critique_pred_mask", None)
        if isinstance(crit_logits, torch.Tensor) and isinstance(crit_mask, torch.Tensor):
            crit_logits_b = crit_logits[-1, bi] if crit_logits.dim() == 4 else crit_logits[bi]
            crit_mask_b   = (crit_mask[-1, bi] if crit_mask.dim() == 3 else crit_mask[bi]) != 0
            valid_logits  = crit_logits_b[crit_mask_b]
            Pexp = K * Np
            if valid_logits.shape[0] >= Pexp:
                valid_logits = valid_logits[:Pexp]
                p_good = torch.softmax(valid_logits, dim=-1)[:, 1].view(K, Np)
        if p_good is None:
            p_good = torch.sigmoid(pose_preds[bi, :, :, 7])  # [K,Np]

        # Rebuild det->slot if empty
        if (not det_to_slot) or len(det_to_slot) == 0:
            obj_index = outs.get("obj_index", None)
            if isinstance(obj_index, torch.Tensor):
                det_to_slot = {}
                idx_b = obj_index[bi]
                seen = set()
                for det_q in [int(x) for x in idx_b.detach().flatten().cpu().tolist()]:
                    if det_q not in seen:
                        det_to_slot[det_q] = len(det_to_slot)
                        seen.add(det_q)

        # slot -> inst_id via det_to_inst
        det_to_inst = det_xyxy.get("det_to_inst", {}) or {}
        slot_to_inst = {}
        for det_q, slot in det_to_slot.items():
            if det_q in det_to_inst:
                slot_to_inst[slot] = int(det_to_inst[det_q])

        out_img = img_pil.copy()
        sel_indices = det_xyxy.get("sel_indices", [])
        sel_colors  = det_xyxy.get("sel_colors", [])

        for j, (qi, color) in enumerate(zip(sel_indices, sel_colors)):
            slot = det_to_slot.get(int(qi), None)
            if slot is None or not (0 <= slot < K):
                if is_main_process():
                    print('slot', slot, qi, j, color, det_to_slot)
                continue
            best_i = int(torch.argmax(p_good[slot]).item())
            pose_k = pose_preds[bi, slot, best_i]  # [8]

            label_txt = None
            if slot in slot_to_inst:
                label_txt = f"id:{slot_to_inst[slot]} sc:{float(p_good[slot, best_i]):.2f}"
            else:
                label_txt = f"sc:{float(p_good[slot, best_i]):.2f}"

            grasp = {
                "xyz": pose_k[:3].detach().cpu().numpy().tolist(),
                "quat_wxyz": pose_k[3:7].detach().cpu().numpy().tolist(),
                "width": 0.05,
                "score": float(p_good[slot, best_i].detach().cpu().item()),
            }
            out_img = _draw_gripper(
                out_img, grasp, intr_list,
                finger_len_m=0.05, default_width_m=0.05, line_w=3,
                color_fingers=tuple(color), color_link=tuple(color), color_dir=(255, 0, 0),
                label_text=label_txt, label_color=tuple(color)
            )
        return out_img
    except Exception as e:
        print(f"[save_last_batch_plots] grasp overlay failed: {e}")
        return img_pil


def _draw_grasp_pose(img_pil, grasp, intr, color=(0,255,0), axis_len_m=0.03):
    """
    Draw a grasp as center dot + approach axis line in image.
    Gripper approach assumed +Z axis of grasp frame; adjust if needed.
    grasp: dict with 'xyz' [3], 'quat_wxyz' [4]
    intr: [cx, cy, fx, fy]
    """
    cx, cy, fx, fy = [float(x) for x in (intr if not isinstance(intr, torch.Tensor) else intr.cpu().tolist())]
    xyz = np.asarray(grasp['xyz'], dtype=np.float32)
    # Build rotation from [w,x,y,z]
    w, x, y, z = [float(q) for q in grasp['quat_wxyz']]
    # Convert quaternion to R if SciPy available; otherwise quick formula
    try:
        R = SciRot.from_quat([x, y, z, w]).as_matrix().astype(np.float32)
    except Exception:
        # Minimal fallback for visualization (not exact if SciPy missing)
        R = np.eye(3, dtype=np.float32)
    # Approach axis: +Z in grasp frame
    a_dir = R[:, 2]  # [3]
    p2 = xyz + a_dir * axis_len_m
    # Project
    def _proj(P):
        X, Y, Z = P
        if Z <= 1e-6: return None
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        return (int(round(u)), int(round(v)))
    p1i = _proj(xyz)
    p2i = _proj(p2)
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)
    if p1i is not None:
        draw.ellipse([(p1i[0]-2, p1i[1]-2), (p1i[0]+2, p1i[1]+2)], fill=tuple(color))
    if p1i is not None and p2i is not None:
        draw.line([p1i, p2i], fill=tuple(color), width=2)
    return out


def _draw_bbox_with_score(img_pil, xyxy, score, color):
    from PIL import ImageDraw
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    draw.rectangle([(x1, y1), (x2, y2)], outline=tuple(color), width=2)
    text = f"{float(score):.2f}"
    # draw text box
    tx, ty = x1, max(0, y1 - 12)
    draw.rectangle([(tx, ty), (tx + 40, ty + 12)], fill=tuple(color))
    draw.text((tx + 2, ty), text, fill=(0, 0, 0))
    return out


def save_last_batch_plots(mode: str, epoch: int, args, batch: List[Dict[str, Any]], outs: Dict[str, Any],
                          n_images_saved: int = 4, max_objects_per_image=30, score_thresh=0.05):
    """
    Save plots for the last batch:

      - Draw predicted boxes + scores
      - If object properties exist: overlay the object point cloud projected with intrinsics.

      - Uses obj_index (top-K) to map detection qi -> object slot.

    """
    if not is_main_process():
        return
    try:
        out_dir = os.path.join(args.output_dir,
                               "train_plots" if mode == "train" else "valid_plots",
                               str(epoch).zfill(4))
        os.makedirs(out_dir, exist_ok=True)

        pred_boxes = outs.get("pred_boxes")   # [B,Q,4] normalized
        pred_logits = outs.get("pred_logits") # [B,Q,C]
        if pred_boxes is None or pred_logits is None:
            print("[save_last_batch_plots] Missing pred_boxes/pred_logits; skipping.")
            return

        obj_index = outs.get("obj_index", None)   # [B,K] if present
        have_obj_preds = bool(args.with_obj and isinstance(obj_index, torch.Tensor))

        B, Q, _ = pred_boxes.shape
        n_save = min(int(n_images_saved), len(batch))

        for bi in range(n_save):
            meta = batch[bi]
            img_pil = _denorm_to_pil(meta["img"].cpu())
            H = int(meta["H"]); W = int(meta["W"])
            depth_pil, _ = _depth_to_colormap(meta.get("depth", None))
            intr = meta.get("intr", None)
            intr_list = intr.cpu().tolist() if isinstance(intr, torch.Tensor) else (intr if intr is not None else None)

            # Boxes to absolute xyxy
            pb = pred_boxes[bi]  # [Q,4]
            pb_xyxy = _cxcywh_norm_to_xyxy_abs(pb, W, H)

            # Scores and selection
            conf = torch.softmax(pred_logits[bi].detach().cpu(), dim=-1)  # [Q,C]
            if conf.shape[-1] < 2:
                print("[save_last_batch_plots] C<2 in pred_logits; skipping.")
                continue
            p_bg, p_obj = conf[:, 0], conf[:, 1]
            idxs = (p_obj > p_bg) & (p_obj >= score_thresh)
            idxs = torch.nonzero(idxs, as_tuple=False).squeeze(1).tolist()
            idxs.sort(key=lambda i: float(p_obj[i].item()), reverse=True)
            if len(idxs) > max_objects_per_image:
                idxs = idxs[:max_objects_per_image]

            colors = _palette(max(len(idxs), 1))
            overlay = img_pil.copy()

            # Build det->slot map for this image
            det_to_slot = {}
            pts_b = pos_b = sc_b = None
            if have_obj_preds:
                idx_b = obj_index[bi]
                idx_list = [int(x) for x in idx_b.detach().flatten().cpu().tolist()]
                seen = set()
                for det_q in idx_list:
                    if 0 <= det_q < Q and det_q not in seen:
                        det_to_slot[det_q] = len(det_to_slot)
                        seen.add(det_q)

                # Slice per-image object predictions to [K,...]
                pts_all = outs.get('obj_points', None)     # [B,K,P,3] or [L,B,K,P,3]
                pos_all = outs.get('obj_positions', None)  # [B,K,3]   or [L,B,K,3]
                sc_all  = outs.get('obj_scale', None)      # [B,K] or [B,K,1] or [L,B,K,1]

                def _slice_img(t):
                    if not isinstance(t, torch.Tensor):
                        return None
                    # [L,B,...]
                    if t.dim() >= 3 and t.shape[0] > 1 and t.shape[1] == B:
                        return t[-1, bi].detach().cpu()
                    # [B,...]
                    if t.dim() >= 2 and t.shape[0] == B:
                        return t[bi].detach().cpu()
                    return None

                pts_b = _slice_img(pts_all)
                pos_b = _slice_img(pos_all)
                sc_b  = _slice_img(sc_all)
                if isinstance(sc_b, torch.Tensor) and sc_b.dim() == 2 and sc_b.size(-1) == 1:
                    sc_b = sc_b.squeeze(-1)

            # Draw boxes and (optionally) object points
            for j, qi in enumerate(idxs):
                color = colors[j % len(colors)]
                overlay = _draw_bbox_with_score(overlay, pb_xyxy[qi], float(p_obj[qi].item()), color=color)

                if have_obj_preds and intr_list is not None and \
                   isinstance(pts_b, torch.Tensor) and isinstance(pos_b, torch.Tensor) and isinstance(sc_b, torch.Tensor):
                    slot = det_to_slot.get(int(qi), None)
                    if slot is not None and 0 <= slot < pts_b.shape[0]:
                        pn_q = pts_b[slot].numpy()     # [P,3]
                        pos_q = pos_b[slot].numpy()    # [3]
                        sc_q  = float(sc_b[slot].item())
                        pts_cam = pn_q * sc_q + pos_q  # to camera meters
                        uv = _project_cam_points(pts_cam, intr_list)
                        overlay = _overlay_points(overlay, uv, color=color)

            # Save files
            image_id = meta.get("image_id", f"{mode}_sample_{bi}")
            stem = f"{mode}__{str(image_id).replace('/','_').replace(' ','_').replace('.png','').replace('.jpg','')}"
            img_pil.save(os.path.join(out_dir, f"{stem}_rgb.png"))
            if depth_pil is not None:
                depth_pil.save(os.path.join(out_dir, f"{stem}_depth.png"))
            overlay.save(os.path.join(out_dir, f"{stem}_det.png"))

            det_xyxy_info = {
                "sel_indices": idxs,          # the selected detection query indices
                "sel_colors": [colors[j % len(colors)] for j in range(len(idxs))],
            }

            # Build det->slot from outs['obj_index'] (works with/without object head)

            det_to_slot = {}
            obj_index = outs.get("obj_index", None)
            if isinstance(obj_index, torch.Tensor):
                idx_b = obj_index[bi]
                seen = set()
                for det_q in [int(x) for x in idx_b.detach().flatten().cpu().tolist()]:
                    if det_q not in seen:
                        det_to_slot[det_q] = len(det_to_slot)
                        seen.add(det_q)

            # Map predictions (det) to GT inst via IoU
            gt_boxes_b = meta["boxes"]                          # [N,4] normalized
            gt_xyxy     = _cxcywh_norm_to_xyxy_abs(gt_boxes_b, W, H)
            box2inst_b  = meta["box2inst"]                      # [N]
            det_to_inst = {}
            for qi, det_box in enumerate(pb_xyxy):
                best_gt = -1; best_iou = 0.0
                for gi, gbox in enumerate(gt_xyxy):
                    iou = _iou_xyxy(det_box, gbox)
                    if iou > best_iou:
                        best_iou = iou; best_gt = gi
                if best_gt >= 0 and best_iou > 0.1:
                    det_to_inst[qi] = int(box2inst_b[best_gt].item())

            # Extend det_xyxy_info so overlay can resolve slot->inst correctly
            det_xyxy_info.update({
                "pb_xyxy": pb_xyxy,
                "gt_xyxy": gt_xyxy,
                "box2inst": box2inst_b,
                "det_to_inst": det_to_inst,
            })

            overlay_grasp = _overlay_best_grasps_on_image(
                img_pil, outs, bi, det_to_slot, det_xyxy_info, colors, intr_list
            )
            if overlay_grasp is not None:
                overlay_grasp.save(os.path.join(out_dir, f"{stem}_grasp.png"))

            overlay_6d = _overlay_best_object_pose_on_image(
                img_pil, outs, batch, bi, det_to_slot, det_xyxy_info, colors, intr_list
            )
            if overlay_6d is not None:
                overlay_6d.save(os.path.join(out_dir, f"{stem}_6d.png"))

    except Exception as e:
        print(f"[save_last_batch_plots] Failed: {e}")

def _to_device_nested(x, device, non_blocking=True):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: _to_device_nested(v, device, non_blocking) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device_nested(v, device, non_blocking) for v in x]
    return x

def fetch_next_batch_sync(loader, loader_iter, device, max_retries=5):
    """
    Fetch the next batch across all ranks in lockstep.

    - If any rank hits timeout/RuntimeError, all ranks skip and retry together.
    - If all ranks hit StopIteration, signal end-of-epoch (return ok=False).

    Returns: (batch, loader_iter, ok)
    """
    retries = 0
    while True:
        local_ok = 1
        local_stop = 0
        batch = None
        try:
            batch = next(loader_iter)
        except StopIteration as e:
            #print('[Warn] "/dataset"n: {}'.format(e))
            local_ok = 0
            local_stop = 1
            #raise e
        except RuntimeError as e:
            #print('[Warn] fetch batch error RuntimeError: {}'.format(e))
            # DataLoader timeout or worker error
            local_ok = 0
            local_stop = 0
            #raise e

        # Sync across ranks
        ok_tensor = torch.tensor([local_ok], device=device)
        stop_tensor = torch.tensor([local_stop], device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)   # success only if all succeeded
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MIN) # end only if all ended
        global_ok = int(ok_tensor.item())
        global_stop = int(stop_tensor.item())

        if global_ok == 1:
            # All ranks have a batch -> proceed
            return batch, loader_iter, True

        if global_stop == 1:
            # All ranks reached end-of-epoch
            return None, loader_iter, False

        # At least one rank failed (timeout/worker error) -> everyone retries
        retries += 1
        if retries >= max_retries:
            # Rebuild iterator to avoid wedged worker buffers
            loader_iter = iter(loader)
            retries = 0
        # Loop continues to retry

def _depth_to_colormap(depth_t):
    if depth_t is None:
        return None, None
    d = depth_t.squeeze().detach().cpu().numpy().astype(np.float32)
    if not np.any(d > 0):
        d8 = np.zeros_like(d, dtype=np.uint8)
    else:
        vmax = float(max(np.percentile(d[d > 0], 95.0), 1e-6))
        d8 = (np.clip(d, 0, vmax) / vmax * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    return Image.fromarray(cm[..., ::-1]), d

def _palette(n):
    rng = np.random.RandomState(123)
    return [tuple(rng.randint(0, 255, size=3).tolist()) for _ in range(max(n, 1))]

def _draw_boxes(img_pil, boxes_xyxy, colors=None):
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)
    if colors is None:
        colors = _palette(len(boxes_xyxy))
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        draw.rectangle([(x1, y1), (x2, y2)], outline=tuple(colors[i % len(colors)]), width=2)
    return out

def _overlay_points(img_pil, uv, color):
    arr = np.array(img_pil).copy()
    H, W = arr.shape[:2]
    for u, v, valid in uv:
        if valid:
            ui = int(round(u)); vi = int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                arr[vi, ui] = color
    return Image.fromarray(arr)

def _cxcywh_norm_to_xyxy_abs(boxes, W, H):
    boxes = boxes.detach().cpu()
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - 0.5 * w) * W
    y1 = (cy - 0.5 * h) * H
    x2 = (cx + 0.5 * w) * W
    y2 = (cy + 0.5 * h) * H
    return torch.stack([x1, y1, x2, y2], dim=-1).round().int().tolist()

def _project_cam_points(points_cam_m, intr):
    cx, cy, fx, fy = [float(x) for x in (intr if not isinstance(intr, torch.Tensor) else intr.cpu().tolist())]
    X = points_cam_m[:, 0]; Y = points_cam_m[:, 1]; Z = points_cam_m[:, 2]
    valid = Z > 1e-6
    u = fx * (X / (Z + 1e-12)) + cx
    v = fy * (Y / (Z + 1e-12)) + cy
    return np.stack([u, v, valid], axis=1)


def _grasps_to_list(ge):
    # ge can be list[dict] or packed dict with tensors
    if isinstance(ge, list):
        return ge
    if isinstance(ge, dict):
        xyz   = ge.get("xyz")
        quat  = ge.get("quat_wxyz")
        width = ge.get("width")
        score = ge.get("score")
        if isinstance(xyz, torch.Tensor) and xyz.ndim == 2:
            n = xyz.size(0)
            out = []
            for i in range(n):
                out.append({
                    "xyz":   xyz[i].detach().cpu().numpy(),
                    "quat_wxyz": (quat[i].detach().cpu().numpy().tolist() if isinstance(quat, torch.Tensor) else quat),
                    "width": float(width[i].item()) if isinstance(width, torch.Tensor) else float(width or 0.0),
                    "score": float(score[i].item()) if isinstance(score, torch.Tensor) else float(score or 0.0),
                })
            return out
    return []


def visualize_batch(batch, title_prefix=""):
    import matplotlib.pyplot as plt
    print('Viz Batch')
    for idx, b in enumerate(batch):
        print('{}/{}: {}'.format(idx+1, len(batch), b.keys()))
        img_pil = _denorm_to_pil(b["img"].cpu())
        H, W = int(b["H"]), int(b["W"])
        boxes_xyxy = _cxcywh_norm_to_xyxy_abs(b["boxes"], W, H)
        colors = _palette(len(boxes_xyxy))

        img_boxes = _draw_boxes(img_pil, boxes_xyxy, colors=colors)
        depth_pil, depth_meter = _depth_to_colormap(b.get("depth", None))

        img_pose = img_boxes.copy()
        intr = b.get("intr", None)
        props = b.get("props", None)

        # Overlay normalized object points (existing logic)
        if intr is not None and props is not None:
            intr_list = intr.cpu().tolist() if isinstance(intr, torch.Tensor) else intr
            for j in range(len(boxes_xyxy)):
                if not bool(props["valid"][j].item()):
                    continue
                pn = props["points_list"][j]
                if pn.numel() == 0:
                    continue
                pn = pn.cpu().numpy()
                pos = props["pos"][j].cpu().numpy()
                sc = float(props["scale"][j, 0].item())
                pts_cam = pn * sc + pos
                uv = _project_cam_points(pts_cam, intr_list)
                img_pose = _overlay_points(img_pose, uv, color=colors[j % len(colors)])

        gt_gbi = b.get("gt_grasps", {}) or {}
        if intr is not None and isinstance(gt_gbi, dict):
            intr_list = intr.cpu().tolist() if isinstance(intr, torch.Tensor) else intr
            for j in range(len(boxes_xyxy)):
                inst_id = j + 1
                packed = gt_gbi.get(inst_id, [])
                grasps_list = _grasps_to_list(packed)
                if not grasps_list:
                    continue
                sel = grasps_list if len(grasps_list) <= 2 else random.sample(grasps_list, 2)
                col = colors[j % len(colors)]
                for g in sel:
                    img_pose = _draw_gripper(
                        img_pose, g, intr_list,
                        finger_len_m=0.05, default_width_m=0.05,
                        color_link=col, label=True, label_color=col
                    )

        fig = plt.figure(figsize=(10, 4))
        fig.suptitle(f"{title_prefix} | sample {idx}: {b.get('image_id','')}")
        ax1 = fig.add_subplot(2,2,1); ax1.set_title("RGB"); ax1.imshow(img_pil); ax1.axis("off")

        ax2 = fig.add_subplot(2,2,2); ax2.set_title("Depth")
        if depth_pil is not None: ax2.imshow(depth_pil)
        else: ax2.text(0.5, 0.5, "No depth", ha="center")
        ax2.axis("off")

        ax3 = fig.add_subplot(2,2,3); ax3.set_title("Depth Ori")
        if depth_pil is not None: ax3.imshow(depth_meter) 
        else: ax3.text(0.5, 0.5, "No Ori depth", ha="center")
        ax3.axis("off")
        ax4 = fig.add_subplot(2,2,4); ax4.set_title("RGB + boxes + pose + GT grasps"); ax4.imshow(img_pose); ax4.axis("off")
        plt.tight_layout(); plt.show(block=True); plt.pause(0.001); plt.close(fig)



def select_val_indices(ds_len, n, seed):
    n = max(1, min(int(n), int(ds_len)))
    rng = np.random.RandomState(int(seed))
    idxs = rng.choice(np.arange(ds_len), size=n, replace=False)
    return sorted([int(i) for i in idxs])

def make_val_loaders(args, val_ds, collate_fn):
    """
    Returns: (val_subset_loader, val_full_loader)

    - subset loader evaluates only 'val_subset_size' random indices chosen deterministically by 'val_subset_seed'
    - full loader runs on the entire validation dataset (for final evaluation at end)

    """
    if val_ds is None:
        return None, None

    use_pin_memory = (args.device.type == "cuda")
    # FULL loader
    full_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if args.distributed else None
    val_full_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=full_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
        timeout=max(1, args.dataloader_timeout),
    )

    # SUBSET loader (used during training evals)
    subset_seed = args.val_subset_seed if args.val_subset_seed >= 0 else args.seed
    idxs = select_val_indices(len(val_ds), args.val_subset_size, subset_seed)
    val_subset = torch.utils.data.Subset(val_ds, idxs)
    subset_sampler = DistributedSampler(val_subset, shuffle=False, drop_last=False) if args.distributed else None
    val_subset_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=args.batch_size,
        sampler=subset_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
        timeout=max(1, args.dataloader_timeout),
    )
    return val_subset_loader, val_full_loader


def init_wandb(args):
    if (not args.use_wandb) or (not _HAS_WANDB) or (not is_main_process()):
        return None

    wandb_dir = getattr(args, "wandb_dir", "./data/wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir  # ensure local files go under ./data/wandb


    if args.wandb_mode == "online":
        # reads from env if key is None; or pass key explicitly
        wnb_key = os.getenv("WANDB_API_KEY")
        if wnb_key is None:
            wnadbkeypath = os.path.join(wandb_dir, 'wandb_key.json')
            print('[INFO] WandB key is None, trying to load from {}'.format(wnadbkeypath))
            with open(wnadbkeypath) as f:
                wnb_key = json.load(f).get('key')
        wandb.login(key=wnb_key, relogin=False)

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    run_name = args.wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        entity=(args.wandb_entity or None),
        mode=args.wandb_mode,
        config=vars(args),
        group=(args.wandb_group or None),
        tags=tags,
        dir=wandb_dir,
    )
    return wandb


# ddp_train_mixed.py

class ProgressLogger:
    def __init__(self, args, total_epochs, output_dir, prev_train_epoch_avg=0.0):
        self.args = args
        self.total_epochs = int(total_epochs)
        self.output_dir = output_dir
        self.prev_train_epoch_avg = float(prev_train_epoch_avg)

        # Phase timing
        self.train_step_times = []
        self.val_step_times = []

        # Per-epoch accumulators
        self.epoch_wall_start = None
        self._epoch_train_time_accum = 0.0
        self._epoch_val_time_accum = 0.0
        self._epoch_train_steps = 0
        self._epoch_val_steps = 0

        # History for averaging
        self.epoch_history = []  # [{train_steps, train_time, val_steps, val_time, overhead_time}]

        # Cached "last tick" times
        self._last_train_time = None
        self._last_val_time = None

        # Planning info
        self.eval_every = getattr(args, "eval_every", 1)
        self.train_steps_per_epoch = None
        self.val_steps_per_eval = 0  # 0 if no val loader

        # CSV
        self.csv_path = os.path.join(self.output_dir, self.args.csv_file)
        self._init_csv()

    def set_steps(self, train_steps_per_epoch, val_steps_per_eval, eval_every=None):
        self.train_steps_per_epoch = int(train_steps_per_epoch)
        self.val_steps_per_eval = int(val_steps_per_eval or 0)
        if eval_every is not None:
            self.eval_every = int(eval_every)

    def _init_csv(self):
        if not is_main_process():
            return
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "time", "mode", "epoch", "step", "steps_total", "steps_pct",
                    "epochs_done", "epochs_total", "epochs_pct",
                    "eta", "loss", "loss_avg", "loss_delta_prev_epoch", "mAP"
                ])

    def _format_eta(self, secs):
        secs = max(0, int(secs))
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _avg_train_step_time(self):
        hist_steps = sum(e["train_steps"] for e in self.epoch_history if e["train_steps"] > 0)
        hist_time  = sum(e["train_time"]  for e in self.epoch_history if e["train_steps"] > 0)
        if hist_steps > 0 and hist_time > 0:
            return hist_time / hist_steps
        if self.train_step_times:
            return sum(self.train_step_times) / len(self.train_step_times)
        return 0.0

    def _avg_val_step_time(self):
        hist_steps = sum(e["val_steps"] for e in self.epoch_history if e["val_steps"] > 0)
        hist_time  = sum(e["val_time"]  for e in self.epoch_history if e["val_steps"] > 0)
        if hist_steps > 0 and hist_time > 0:
            return hist_time / hist_steps
        if self.val_step_times:
            return sum(self.val_step_times) / len(self.val_step_times)
        return 0.0

    def _avg_overhead_time_per_epoch(self):
        if not self.epoch_history:
            return 0.0
        return sum(e["overhead_time"] for e in self.epoch_history) / max(1, len(self.epoch_history))

    def _is_eval_epoch(self, epoch_idx):
        if self.val_steps_per_eval <= 0 or self.eval_every <= 0:
            return False
        return ((epoch_idx + 1) % self.eval_every) == 0

    def _count_future_evals(self, epoch_idx):
        if self.eval_every <= 0 or self.val_steps_per_eval <= 0:
            return 0
        total_evals = self.total_epochs // self.eval_every
        evals_done_up_to_current = (epoch_idx + 1) // self.eval_every
        future_evals = total_evals - evals_done_up_to_current
        return max(0, future_evals)

    def _compute_global_eta(self, mode, epoch_idx, steps_done_current, steps_total_current):
        avg_train = self._avg_train_step_time()
        avg_val   = self._avg_val_step_time()
        avg_ovh   = self._avg_overhead_time_per_epoch()

        train_steps = steps_total_current if mode == "train" else (self.train_steps_per_epoch or steps_total_current or 0)
        val_steps   = self.val_steps_per_eval

        if mode == "train":
            train_rem = max(0, train_steps - steps_done_current) * avg_train
            val_rem   = (val_steps * avg_val) if self._is_eval_epoch(epoch_idx) else 0.0
            ovh_rem   = avg_ovh
        else:
            train_rem = 0.0
            val_rem   = max(0, val_steps - steps_done_current) * avg_val
            ovh_rem   = avg_ovh

        current_rem = train_rem + val_rem + ovh_rem

        future_epochs = max(0, self.total_epochs - (epoch_idx + 1))
        future_evals  = self._count_future_evals(epoch_idx)
        steps_per_epoch_for_future = (self.train_steps_per_epoch or train_steps)

        future_train_time = future_epochs * steps_per_epoch_for_future * avg_train
        future_val_time   = future_evals * val_steps * avg_val
        future_ovh_time   = future_epochs * avg_ovh

        eta_secs = current_rem + future_train_time + future_val_time + future_ovh_time
        return self._format_eta(eta_secs)

    def set_prev_train_epoch_avg(self, val):
        self.prev_train_epoch_avg = float(val)

    def start_train_epoch(self):
        self.epoch_wall_start = time.time()
        self._last_train_time = None
        self._last_val_time = None
        self._epoch_train_time_accum = 0.0
        self._epoch_val_time_accum = 0.0
        self._epoch_train_steps = 0
        self._epoch_val_steps = 0

    def start_val_epoch(self):
        self._last_val_time = None

    def end_epoch(self):
        if self.epoch_wall_start is None:
            return
        wall = time.time() - self.epoch_wall_start
        overhead = max(0.0, wall - self._epoch_train_time_accum - self._epoch_val_time_accum)
        self.epoch_history.append({
            "train_steps": int(self._epoch_train_steps),
            "train_time": float(self._epoch_train_time_accum),
            "val_steps": int(self._epoch_val_steps),
            "val_time": float(self._epoch_val_time_accum),
            "overhead_time": float(overhead),
        })
        self.epoch_wall_start = None

    def update_train(self, epoch_idx, total_epochs, step_idx, steps_total, batch_loss, running_mean_loss, write_csv=False, extras_avg=None):
        now = time.time()
        if self._last_train_time is not None:
            dt = now - self._last_train_time
            self.train_step_times.append(dt)
            self._epoch_train_time_accum += dt
            self._epoch_train_steps += 1
            if len(self.train_step_times) > 200:
                self.train_step_times = self.train_step_times[-100:]
        self._last_train_time = now

        steps_done = step_idx + 1
        steps_pct = 100.0 * steps_done / max(1, steps_total)
        epochs_pct = 100.0 * ((epoch_idx + (steps_done / max(1, steps_total))) / max(1, total_epochs))

        eta = self._compute_global_eta(mode="train", epoch_idx=epoch_idx, steps_done_current=steps_done, steps_total_current=steps_total)
        delta_prev = running_mean_loss - self.prev_train_epoch_avg

        extras_str = ""
        if extras_avg:   

            if self.args.with_pose:
                ps = []
                v = extras_avg.get("pose_avg_crit_score", None)
                if v is not None: ps.append(f"μ{v:.2f}")
                #v = extras_avg.get("pose_conf_err_rate", None)
                #if v is not None: ps.append(f"Δ{v:.2f}%")
                v = extras_avg.get("pose_conf_err_l1", None)
                if v is not None: ps.append(f"Δ{v:.2f}")
                v = extras_avg.get("pose_slot_acc", None)
                if v is not None: ps.append(f"sl{v:.1f}")
                v = extras_avg.get("pose_chamfer_avg", None)
                if v is not None: ps.append(f"CD{v:.3f}")                
                v = extras_avg.get("grasp_nn_trans_avg", None)
                if v is not None: ps.append(f"D{v:.3f}")
                v = extras_avg.get("grasp_nn_angle_deg_avg", None)
                if v is not None: ps.append(f"°{v:.1f}")
                v = extras_avg.get("grasp_nn_close_pct", None)
                if v is not None: ps.append(f"%{v:.1f}")

                if ps:
                    extras_str += " | P " + " ".join(ps)
            elif self.args.with_critique:
                a  = extras_avg.get("critique_acc", 0.0)
                if a > 0.0:
                    am  = extras_avg.get("critique_acc_mid", 0.0)                
                    ag = extras_avg.get("critique_acc_good", 0.0)
                    ab = extras_avg.get("critique_acc_bad", 0.0)
                    ar = extras_avg.get("critique_acc_random", 0.0)

                    parts = []
                    if a  >0.0:  parts.append(f"μ{a:.2f}")
                    if am >0.0:  parts.append(f"m{am:.2f}")
                    if ag >0.0:  parts.append(f"✓{ag:.2f}")
                    if ab >0.0:  parts.append(f"X{ab:.2f}")
                    if ar >0.0:  parts.append(f"~{ar:.2f}")
                    if parts:
                        extras_str += " | C " + " ".join(parts)

            elif self.args.with_obj:                
                vs = 0
                for k, nm in [("loss_cls","cls"),("loss_bbox","bbox"),("loss_giou","giou"),("loss_embed","emb")]:
                    vs += extras_avg.get(k, 0.0)

                extras_str = f" | O {vs:.4f} | "
                parts = []
                for k, nm in [
                    ("loss_obj_scale","sc"),
                    ("loss_obj_pos","pos"),
                    ("loss_obj_vis","vis"),
                    ("loss_obj_shape","sh"),
                    ("loss_obj_vol","vol"),
                    ("obj_slot_miss","mis") 
                ]:
                    v = extras_avg.get(k, None)
                    if v is not None:
                        if v > 1:
                            parts.append(f"{nm} {v:.2f}")
                        else:
                            parts.append(f"{nm} {v:.4f}")
                if parts: extras_str = extras_str + " ".join(parts)
            else:
                parts = []
                for k, nm in [("loss_cls","cls"),("loss_bbox","bbox"),("loss_giou","giou"),("loss_embed","emb")]:
                    v = extras_avg.get(k, None)
                    if v is not None:
                        if v > 1:
                            parts.append(f"{nm} {v:.2f}")
                        else:
                            parts.append(f"{nm} {v:.4f}")
                if parts: 
                    extras_str += " | B " + " ".join(parts)
                          

        msg = (
            f"T | e {epoch_idx+1}/{total_epochs} ({epochs_pct:5.3f}%) "
            f"s {steps_done}/{steps_total} ({steps_pct:5.3f}%) "
            f"ETA {eta} "
            f"l {batch_loss:.4f} "
            f"μ {running_mean_loss:.4f} "
            f"Δ {delta_prev:+.4f}{extras_str}"
        )

        if is_main_process():
            sys.stdout.write("\r" + msg)
            sys.stdout.flush()

            if write_csv:
                self._init_csv()
                with open(self.csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        datetime.now().isoformat(), "train",
                        epoch_idx, steps_done, steps_total, f"{steps_pct:.3f}",
                        epoch_idx, total_epochs, f"{epochs_pct:.3f}",
                        eta, f"{batch_loss:.6f}", f"{running_mean_loss:.6f}", f"{delta_prev:.6f}", ""
                    ])

    def update_val(self, epoch_idx, total_epochs, step_idx, steps_total, batch_loss, running_mean_loss, write_csv=False, extras_avg=None, final_map=None):
        now = time.time()
        if self._last_val_time is not None:
            dt = now - self._last_val_time
            self.val_step_times.append(dt)
            self._epoch_val_time_accum += dt
            self._epoch_val_steps += 1
            if len(self.val_step_times) > 200:
                self.val_step_times = self.val_step_times[-100:]
        self._last_val_time = now

        steps_done = step_idx + 1
        steps_pct = 100.0 * steps_done / max(1, steps_total)
        epochs_pct = 100.0 * ((epoch_idx + (steps_done / max(1, steps_total))) / max(1, total_epochs))

        eta = self._compute_global_eta(mode="val", epoch_idx=epoch_idx, steps_done_current=steps_done, steps_total_current=steps_total)
        

        extras_str = ""
        if extras_avg:   

            if self.args.with_pose:
                ps = []
                v = extras_avg.get("pose_avg_crit_score", None)
                if v is not None: ps.append(f"μ{v:.2f}")
                #v = extras_avg.get("pose_conf_err_rate", None)
                #if v is not None: ps.append(f"Δ{v:.2f}%")
                v = extras_avg.get("pose_conf_err_l1", None)
                if v is not None: ps.append(f"Δ{v:.2f}")
                v = extras_avg.get("pose_slot_acc", None)
                if v is not None: ps.append(f"sl{v:.1f}")
                v = extras_avg.get("pose_chamfer_avg", None)
                if v is not None: ps.append(f"CD{v:.3f}")                
                v = extras_avg.get("grasp_nn_trans_avg", None)
                if v is not None: ps.append(f"D{v:.3f}")
                v = extras_avg.get("grasp_nn_angle_deg_avg", None)
                if v is not None: ps.append(f"°{v:.1f}")
                v = extras_avg.get("grasp_nn_close_pct", None)
                if v is not None: ps.append(f"%{v:.1f}")
                if ps:
                    extras_str += " | Pose " + " ".join(ps)

            elif self.args.with_critique:
                a  = extras_avg.get("critique_acc", None)
                am  = extras_avg.get("critique_acc_mid", None)  
                ag = extras_avg.get("critique_acc_good", None)
                ab = extras_avg.get("critique_acc_bad", None)
                ar = extras_avg.get("critique_acc_random", None)

                parts = []
                if a  is not None:  parts.append(f"μ{a:.2f}")
                if am  is not None:  parts.append(f"m{am:.2f}")
                if ag is not None:  parts.append(f"✓{ag:.2f}")
                if ab is not None:  parts.append(f"X{ab:.2f}")
                if ar is not None:  parts.append(f"~{ar:.2f}")
                if parts:
                    extras_str += " | Crit " + " ".join(parts)


            elif self.args.with_obj:                
                vs = 0
                for k, nm in [("loss_cls","cls"),("loss_bbox","bbox"),("loss_giou","giou"),("loss_embed","emb")]:
                    vs += extras_avg.get(k, 0.0)

                extras_str = f" | BBox {vs:.4f} | "
                parts = []
                for k, nm in [
                    ("loss_obj_scale","sc"),
                    ("loss_obj_pos","pos"),
                    ("loss_obj_vis","vis"),
                    ("loss_obj_shape","sh"),
                    ("loss_obj_vol","vol"),
                    ("obj_slot_miss","mis") 
                ]:
                    v = extras_avg.get(k, None)
                    if v is not None:
                        if v > 1:
                            parts.append(f"{nm} {v:.2f}")
                        else:
                            parts.append(f"{nm} {v:.4f}")
                if parts: extras_str = extras_str + " ".join(parts)
            else:
                parts = []
                for k, nm in [("loss_cls","cls"),("loss_bbox","bbox"),("loss_giou","giou"),("loss_embed","emb")]:
                    v = extras_avg.get(k, None)
                    if v is not None:
                        if v > 1:
                            parts.append(f"{nm} {v:.2f}")
                        else:
                            parts.append(f"{nm} {v:.4f}")
                if parts: extras_str = extras_str + " | " + " ".join(parts)

        map_str = ""
        AP_ = ""
        if (final_map is not None) and ("AP" in final_map):
            map_str = f" mAP {final_map['AP']:.4f}"
            AP_ = final_map.get('AP', '')

        msg = (
            f"V | e {epoch_idx+1}/{total_epochs} ({epochs_pct:5.3f}%) "
            f"s {steps_done}/{steps_total} ({steps_pct:5.3f}%) "
            f"ETA {eta} "
            f"l {batch_loss:.4f} "
            f"μ {running_mean_loss:.4f}{extras_str}{map_str}"
        )

        if is_main_process():
            sys.stdout.write("\r" + msg)
            sys.stdout.flush()

            if write_csv:
                with open(self.csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        datetime.now().isoformat(), "valid",
                        epoch_idx, steps_done, steps_total, f"{steps_pct:.3f}",
                        epoch_idx, total_epochs, f"{epochs_pct:.3f}",
                        eta, f"{batch_loss:.6f}", f"{running_mean_loss:.6f}", "", f"{AP_}"
                    ])

    def newline(self):
        if is_main_process():
            sys.stdout.write("\n")
            sys.stdout.flush()

def _load_prev_train_epoch_avg(args, start_epoch):
    try:
        path = os.path.join(args.output_dir, args.log_file)
        if os.path.isfile(path):
            with open(path, "r") as f:
                hist = json.load(f)
            for r in reversed(hist.get("epochs", [])):
                if int(r.get("epoch", -1)) == int(start_epoch):  # 1-based in log
                    return float(r.get("train", {}).get("loss", 0.0))
            # fallback: last available
            if hist.get("epochs"):
                return float(hist["epochs"][-1]["train"]["loss"])
    except Exception:
        pass
    return 0.0

def _str2int_id(s: str) -> int:
    # Deterministic numeric id across ranks
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16)

def _cxcywh_norm_to_xywh_abs(boxes: torch.Tensor, W: int, H: int) -> torch.Tensor:
    # boxes [N,4] in [0,1] -> [N,4] absolute XYWH
    cx, cy, w, h = boxes.unbind(-1)
    x = (cx - 0.5 * w) * W
    y = (cy - 0.5 * h) * H
    w = w * W
    h = h * H
    return torch.stack([x, y, w, h], dim=-1)

def _all_gather_list(data: List[Dict[str, Any]], world_size: int) -> List[Dict[str, Any]]:
    if not dist.is_available() or not dist.is_initialized():
        return data
    out = [None for _ in range(world_size)]
    dist.all_gather_object(out, data)
    merged = []
    for part in out:
        if part:
            merged.extend(part)
    return merged


# ddp_train_mixed.py (replace/update update_history)

def update_history(args, epoch, train_loss_avg, val_loss_avg=None, coco_metrics=None, bests=None):
    if not is_main_process():
        return
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, args.log_file)
    if os.path.isfile(path):
        with open(path, "r") as f:
            hist = json.load(f)
    else:
        hist = {"epochs": [], "bests": {}}

    record = {
        "epoch": int(epoch),
        "train": {"loss": float(train_loss_avg)},
        "val": {"losses": val_loss_avg or {}, "coco": coco_metrics or {}},
        "time": time.time(),
    }
    hist["epochs"].append(record)

    if bests is not None:
        # snapshot of current bests
        hist["bests"] = {
            "train_loss": float(bests.get("train_loss", float("inf"))),
            "train_epoch": int(bests.get("train_epoch", -1)),
            "val_loss": float(bests.get("val_loss", float("inf"))),
            "val_epoch": int(bests.get("val_epoch", -1)),
            "val_mAP": float(bests.get("val_mAP", -1.0)),
            "val_mAP_epoch": int(bests.get("val_mAP_epoch", -1)),
        }

    with open(path, "w") as f:
        json.dump(hist, f, indent=2)

@torch.no_grad()
def evaluate_losses(val_loader, model, criterion, args):
    model.eval(); criterion.eval()
    sum_losses = {}
    n_batches = 0

    rgbd_sums = {}    # name -> (L,2)
    rgbd_counts = {}  # name -> int

    for batch in val_loader:
        imgs = [b["img"].to(args.device) for b in batch]
        if args.with_depth and not args.rgb_only:
            samples = nested_tensor_from_tensor_list(imgs, [b["depth"].to(args.device) for b in batch])
        else:
            samples = nested_tensor_from_tensor_list(imgs)

        targets = []
        for b in batch:
            t = {}
            t["boxes"] = b["boxes"].to(args.device)
            t["labels"] = torch.ones((b["boxes"].shape[0],), dtype=torch.long, device=args.device)
            t["box2inst"] = b["box2inst"].to(args.device)
            if b.get("inst_crops_images") is not None:
                t["inst_crops_images"] = b["inst_crops_images"].to(args.device) 
                t["inst_crops_inst_ids"] = b["inst_crops_inst_ids"].to(args.device)


            if criterion.with_obj:                
                t["obj_valid"] = _to_device_nested(b["props"]["valid"], args.device)
                t["obj_scale"] = _to_device_nested(b["props"]["scale"], args.device)
                t["obj_pos"]   = _to_device_nested(b["props"]["pos"], args.device)
                t["obj_vis"]   = _to_device_nested(b["props"]["vis"], args.device)
                t["obj_vis_mask"]   = _to_device_nested(b["props"]["vis_mask"], args.device)
                t["obj_points_list"] = _to_device_nested(b["props"]["points_list"], args.device)
                t["obj_vol"] = _to_device_nested(b["props"]["vol"], args.device)

            if criterion.with_pose:
                t["poses"] = _to_device_nested(b["poses"], args.device)
                t["models"] = _to_device_nested(b["models"], args.device)

            if args.with_grasp:
                t["gt_grasps"] = _to_device_nested(b["gt_grasps"], args.device)
                t["bad_grasps"] = _to_device_nested(b["bad_grasps"], args.device)
            if args.add_model_embed:
                t["pose_6d_proposals_good"] = _to_device_nested(b["pose_6d_proposals_good"], args.device)
                t["pose_6d_proposals_bad"] = _to_device_nested(b["pose_6d_proposals_bad"], args.device)
                if not criterion.with_pose:                        
                    t["models"] = _to_device_nested(b["models"], args.device)
            targets.append(t)

        outs, _ = model(samples, targets=targets)
        losses = criterion(outs, targets)
        _report_nan_counters(losses, phase="eval", epoch=-1, step=n_batches)

        # accumulate on CPU
        for k, v in losses.items():
            sum_losses[k] = sum_losses.get(k, 0.0) + float(v.item())
        n_batches += 1

        # Aggregate RGB/Depth contribution per head
        rgbd_out = outs.get('rgb_depth_pct', None)
        if isinstance(rgbd_out, dict):
            for name, p in rgbd_out.items():
                if not isinstance(p, torch.Tensor):
                    continue
                p = p.detach()                 # (L,B,2)
                batch_sum = p.sum(dim=1)       # (L,2) sum over batch
                if name not in rgbd_sums:
                    rgbd_sums[name] = torch.zeros_like(batch_sum)
                    rgbd_counts[name] = 0
                if rgbd_sums[name].shape[0] != batch_sum.shape[0]:
                    minL = min(rgbd_sums[name].shape[0], batch_sum.shape[0])
                    rgbd_sums[name][:minL] += batch_sum[:minL]
                else:
                    rgbd_sums[name] += batch_sum
                rgbd_counts[name] += p.shape[1]

    # all-reduce sums and counts for losses
    if dist.is_available() and dist.is_initialized():
        keys = sorted(sum_losses.keys())
        vals = torch.tensor([sum_losses[k] for k in keys] + [n_batches], device=args.device)
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        for i, k in enumerate(keys):
            sum_losses[k] = float(vals[i].item())
        n_batches = int(vals[-1].item())

    avg = {k: (v / max(n_batches, 1)) for k, v in sum_losses.items()}

    # All-reduce and compute means per head
    if rgbd_sums:
        for name, sum_t in rgbd_sums.items():
            sum_tensor = sum_t.to(args.device)
            cnt_tensor = torch.tensor([rgbd_counts[name]], dtype=sum_tensor.dtype, device=args.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)
            denom = max(float(cnt_tensor.item()), 1e-12)
            rgbd_mean = (sum_tensor / denom).detach().cpu()  # (L,2)
            for li in range(rgbd_mean.shape[0]):
                avg[f"rgbd/{name}/l{li}/rgb_pct"] = float(rgbd_mean[li, 0].item())
                avg[f"rgbd/{name}/l{li}/depth_pct"] = float(rgbd_mean[li, 1].item())
            # legacy keys if only one unnamed tensor was used
            if name == 'default':
                for li in range(rgbd_mean.shape[0]):
                    avg[f"rgbd/l{li}/rgb_pct"] = float(rgbd_mean[li, 0].item())
                    avg[f"rgbd/l{li}/depth_pct"] = float(rgbd_mean[li, 1].item())

    return avg


def _ann_id_from_image(image_id_int: int, gt_idx: int) -> int:
    # deterministic, unique across ranks
    import hashlib
    return int(hashlib.md5(f"{image_id_int}_{gt_idx}".encode("utf-8")).hexdigest()[:8], 16)

    
@torch.no_grad()
def evaluate_coco(val_loader, model, args, score_thresh=0.05):
    if not _HAS_COCO:
        if is_main_process():
            print("pycocotools not found. Skipping COCO evaluation.")
        return {}

    model.eval()
    gt_images = {}
    gt_annos = []
    preds = []
    ann_id = 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    val_steps_total = len(val_loader)            
    val_iter = iter(val_loader)
    val_steps_done = 0
    
    while val_steps_done < val_steps_total:
        batch, val_iter, ok = fetch_next_batch_sync(
            loader=val_loader,
            loader_iter=val_iter,
            device=args.device,
            max_retries=getattr(args, "max_fetch_retries", 5)
        )
        if not ok:
            continue

        #for batch in val_loader:
        imgs = [b["img"].to(args.device) for b in batch]
        if args.with_depth and not args.rgb_only:
            samples = nested_tensor_from_tensor_list(imgs, [b["depth"].to(args.device) for b in batch])
        else:
            samples = nested_tensor_from_tensor_list(imgs)
                                   
        outs, _ = model(samples, targets=None)
        pred_boxes = outs["pred_boxes"].detach().cpu()   # [B,Q,4] norm
        
        B, Q, _ = pred_boxes.shape
        for bi in range(B):
            meta = batch[bi]
            image_id_str = meta["image_id"]
            image_id = _str2int_id(image_id_str)
            H = int(meta["H"]); W = int(meta["W"])

            if image_id not in gt_images:
                gt_images[image_id] = dict(id=image_id, height=H, width=W, file_name=image_id_str)

            gt_boxes = meta["boxes"].cpu()
            gt_xywh = _cxcywh_norm_to_xywh_abs(gt_boxes, W, H)
            N = gt_xywh.shape[0]
            
            l_start = len(gt_annos) 
            for i in range(N):
                gt_annos.append(dict(
                    id=ann_id + (rank * 1000000),
                    image_id=image_id,
                    category_id=1 if args.n_classes <= 1 else int(1),
                    bbox=[float(x) for x in gt_xywh[i].tolist()],
                    area=float(gt_xywh[i][2] * gt_xywh[i][3]),
                    iscrowd=0,
                ))
                ann_id += 1

            pb = pred_boxes[bi]
            pred_xywh = _cxcywh_norm_to_xywh_abs(pb, W, H)

            logits = outs.get('pred_logits')
            # logits: [B,Q,C]
            conf = torch.softmax(logits[bi].detach().cpu(), dim=-1)  # [Q,C]            
            p_bg  = conf[:, 0]
            obj_scores = conf[:, 1]
            obj_cls = torch.ones_like(obj_scores, dtype=torch.long)

            # keep only non-background predictions (argmax != bg) and above threshold
            keep = (obj_scores > p_bg) & (obj_scores >= score_thresh)  # equivalently p_obj > 0.5 for binary + threshold

            sel_xywh = pred_xywh[keep]
            sel_scores = obj_scores[keep]
            sel_cls = obj_cls[keep]


            if len(sel_scores) > 100:
                topk_idx = torch.topk(sel_scores, 100).indices
                sel_xywh = sel_xywh[topk_idx]
                sel_scores = sel_scores[topk_idx]
                sel_cls = sel_cls[topk_idx]

            for j in range(sel_xywh.shape[0]):
                preds.append(dict(
                    image_id=image_id,
                    category_id=int(sel_cls[j].item()),
                    bbox=[float(x) for x in sel_xywh[j].tolist()],
                    score=float(sel_scores[j].item())
                ))
            
        val_steps_done += 1

    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    all_preds = _all_gather_list(preds, world_size)
    all_gt_annos = _all_gather_list(gt_annos, world_size)
    all_images_list = _all_gather_list(list(gt_images.values()), world_size)
    image_map = {d["id"]: d for d in all_images_list}
    images_final = list(image_map.values())

    if not is_main_process():
        return {}

    categories = [dict(id=i, name=f"class_{i}", supercategory="object")
                for i in (range(1, args.n_classes + 1) if args.n_classes > 1 else [1])]

    gt_dict = {
        "info": {
            "description": "MixedMultiDataset GT for COCO eval",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images_final,
        "annotations": all_gt_annos,
        "categories": categories,
    }

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    if len(all_preds) == 0:
        print("No predictions for COCO eval.")
        return {}

    coco_dt = coco_gt.loadRes(all_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

    names = ["AP","AP50","AP75","APs","APm","APl","AR1","AR10","AR100","ARs","ARm","ARl"]
    stats = {k: float(v) for k, v in zip(names, coco_eval.stats.tolist())}
    return stats

def save_best_loss_checkpoint(model, optimizer, scheduler, epoch, args):
    if not is_main_process():
        return
    
    def get_orig(mod):
        # unwrap DDP and compiled wrappers
        if hasattr(mod, "module"):
            mod = mod.module
        return getattr(mod, "_orig_mod", mod)

    orig = get_orig(model)
        
    ckpt = {
        "model": orig.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "args": vars(args),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(args.output_dir, "best_loss.pth"))

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
        return
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    dist.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    dist.barrier()



def load_fitting_state_dict(arch, state_dict, prefix=None):
    ori_sd = arch.state_dict()
    keys = list(ori_sd.keys())
    keys_left = list(ori_sd.keys())
    wrong_shape_keys = []
    #print('model keys', keys)
    wrong_key = 0
    wrong_shape = 0
    changed_shape = 0
    okay = 0
    #print('sd keys', state_dict.keys())

    # if prefix:
    #     state_dict ={k[len(prefix)+1: ]: v for k,v in state_dict.items() if k.startswith(prefix)} 

    for key, v in state_dict.items():
        if key not in keys:
            wrong_key += 1
            #if is_main_process():
            #    print('wrong key', key, v.shape)
            continue
        
        sd = {key: v}

        ls_ = v.shape
        os_ = ori_sd.get(key)
        if os_ is not None:
           os_ = os_.shape
        
        
        del keys_left[keys_left.index(key)]
        
        try:
            arch.load_state_dict(sd, strict=False)
        
        except Exception as e:
            current_wrong_shape_ = True
            if '.pos_embed' in key:
                pos_embed_checkpoint = state_dict[key]
                embedding_size = pos_embed_checkpoint.shape[-1]
                try:
                    num_patches = arch.encoder.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.encoder.color_encoder.pos_embed.shape[-2] - num_patches
                except:
                    num_patches = arch.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.color_encoder.pos_embed.shape[-2] - num_patches

                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    #if is_main_process():
                    #    print("Pos Embed {} interpolate from {}x{} to {}x{}".format(key, orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2).float()
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens.float(), size=(new_size, new_size), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    sd = {key: new_pos_embed}
                    arch.load_state_dict(sd, strict=False)

                okay += 1
                current_wrong_shape_ = False


            #print(state_dict.keys())
            elif 'rope' in key:              
            
                rope_embed_checkpoint = state_dict[key]
                embedding_size = rope_embed_checkpoint.shape[-1]
                try:
                    num_patches = arch.encoder.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.encoder.color_encoder.pos_embed.shape[-2] - num_patches
                except:
                    num_patches = arch.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.color_encoder.pos_embed.shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged

                
                #print('key', key, orig_size, new_size, rope_embed_checkpoint.shape)
                if orig_size != new_size:
                    if is_main_process():            
                        print("Rope {} interpolate from {}x{} to {}x{}".format(key, orig_size, orig_size, new_size, new_size))
                    
                    # only the position tokens are interpolated
                    rope_tokens = rope_embed_checkpoint.reshape(orig_size, orig_size, embedding_size).unsqueeze(0).permute(0, 3, 1, 2).float()   #.view((-1, rope_embed_checkpoint.shape[-1]))
                    
                    #pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2).float()
                    rope_tokens = torch.nn.functional.interpolate(
                        rope_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    
                    rope_tokens = rope_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
                    
                    #new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    #state_dict[key] = rope_tokens
                    sd = {key: rope_tokens}
                    arch.load_state_dict(sd, strict=False)
                okay += 1
                current_wrong_shape_ = False

            elif os_ is not None and os_ != ls_ and len(os_) == len(ls_):
                
                vs_ = copy.deepcopy(ori_sd.get(key))
                vs_s = []
                for os_s, ls_s in zip(os_, ls_):
                    vs_s.append(min(os_s, ls_s))
                
                if len(vs_s) == 1:
                    vs_[:vs_s[0]] = v[:vs_s[0]]
                if len(vs_s) == 2:
                    vs_[:vs_s[0], :vs_s[1]] = v[:vs_s[0], :vs_s[1]]
                if len(vs_s) == 3:
                    vs_[:vs_s[0], :vs_s[1], :vs_s[2]] = v[:vs_s[0], :vs_s[1], :vs_s[2]]
                if len(vs_s) == 4:
                    vs_[:vs_s[0], :vs_s[1], :vs_s[2], :vs_s[3]] = v[:vs_s[0], :vs_s[1], :vs_s[2], :vs_s[3]]
                if len(vs_s) == 5:
                    vs_[:vs_s[0], :vs_s[1], :vs_s[2], :vs_s[3], :vs_s[4]] = v[:vs_s[0], :vs_s[1], :vs_s[2], :vs_s[3], :vs_s[4]]
                if len(vs_s) == 6:
                    vs_[:vs_s[0], :vs_s[1], :vs_s[2], :vs_s[3], :vs_s[4], :vs_s[5]] = v[:vs_s[0], :vs_s[1], :vs_s[2], :vs_s[3], :vs_s[4], :vs_s[5]]


                #print('changing', key, os_, ls_, vs_s)       
                changed_shape += 1

                sd = {key: vs_}         

                arch.load_state_dict(sd, strict=False)
                okay += 1
                current_wrong_shape_ = False            

            if current_wrong_shape_:
                #if is_main_process():
                #    print('wrong shape', e)
                wrong_shape += 1
                wrong_shape_keys.append(keys)
            
            continue
        okay += 1

    msg = 'Loaded {}/{} weights, wrong key: {}, wrong shape: {}, missing: {}, changed shape: {}'.format(
        okay, len(keys), wrong_key, wrong_shape, len(keys) - (okay+wrong_key+wrong_shape), changed_shape)

    if is_main_process():    
        print(msg)
    #print('keys_left', keys_left)
    #print('wrong_shape_keys', wrong_shape_keys)
    return arch




def build_model_and_criterion(args, encoder):
    # Build Grasper model (deformable transformer with detection + object heads)
    # This uses your existing builder which returns (model, criterion_poet, matcher_poet), but we only need the model.
       
    model = build_grasper(args, encoder)  # model.deformable_detr.grasp_estimation_transformer.build
    
    if is_main_process():
        print(model)

    if args.pretrained_deform_detr and args.load_coco_pretrained:
        if is_main_process():
            print('[INFO] loading pretrained deform detr from {}'.format(args.pretrained_deform_detr))
        cp = torch.load(args.pretrained_deform_detr, map_location="cpu")['model']
        
        sd = {}
        for key, v in cp.items():
            if 'backbone.' in key:
                continue
            if 'input_proj.' in key:
                continue

            if 'transformer.encoder.' in key:
                key = key.replace('transformer.encoder.', 'transformer.det_encoder.')
            
            if 'transformer.decoder.' in key:
                key = key.replace('transformer.decoder.', 'transformer.det_decoder.')
            
            if 'transformer.level_embed.' in key:
                key = key.replace('transformer.level_embed.', 'transformer.det_level_embed.')

            if 'bbox_embed.' in key:
                if 'transformer' in key:
                    key = key.replace('bbox_embed.', 'det_bbox_embed.')
                else:
                    key = key.replace('bbox_embed.', 'transformer.det_bbox_embed.')
            if 'class_embed.' in key:
                if 'transformer' in key:
                    key = key.replace('class_embed.', 'det_class_embed.')
                else:
                    key = key.replace('class_embed.', 'transformer.det_class_embed.')
            if 'reference_points.' in key:
                key = key.replace('reference_points.', 'det_reference_points.')
            if 'query_embed.' in key:
                key = key.replace('query_embed.', 'transformer.det_query_embed.')
            
            sd[key] = v

        model = load_fitting_state_dict(model, sd)            
        del cp
        del sd

    print('criterion, resue det {}, reuse obj {}'.format(args.aux_det_match_reuse, args.aux_obj_match_reuse))
    obj2box_coef = 10

    # Enable/disable heads
    det_enabled = bool(args.with_det) and (args.lr_det is not None)
    obj_enabled = bool(args.with_obj) and (args.lr_obj  is not None) # or det_enabled or (args.with_pose and args.lr_pose is not None))
    pose_enabled = bool(args.with_pose) and (args.lr_pose is not None)
    crit_enabled = bool(args.with_critique) #and not bool(args.with_pose)

    criterion = JointCriterion(
        loss_bbox_coef=5.0,
        loss_giou_coef=2.0,
        loss_obj_scale_coef=5.0*obj2box_coef,
        loss_obj_pos_coef=10.0*obj2box_coef,
        loss_obj_vis_coef=0.1*obj2box_coef,
        loss_obj_shape_coef=1.0*obj2box_coef,
        loss_obj_vol_coef=0.1*obj2box_coef,
        with_det=det_enabled,
        with_obj=obj_enabled,
        with_pose=pose_enabled,
        with_critique=crit_enabled,
        shape_gt_subsample=512,
        aux_det_weight=1.0,
        aux_obj_weight=1.0,
        aux_det_match_reuse=args.aux_det_match_reuse,
        aux_obj_match_reuse=args.aux_obj_match_reuse,
        warmup_steps=args.warmup_steps,
        start_warmup_step=args.start_warmup_step,
        apply_vis_mask_to_shape=args.apply_vis_mask_to_shape,
    ).to(args.device)
    return model, criterion


def param_groups(model, args):
    # Build LR groups
    back_params = []
    back_params_count = 0
    det_params = []
    det_params_count = 0
    obj_params = []
    obj_params_count = 0
    pose_params = []
    pose_params_count = 0
    critique_params = []
    critique_params_count = 0
    proj_params = []
    proj_params_count = 0
    tfencoder_params = []
    tfencoder_params_count = 0
    other_params = []
    other_params_count = 0


    for n, p in model.named_parameters():
        #if not p.requires_grad:
        #    continue

        p.requires_grad = True

        def _contig_or_none(g):
            return None if g is None else g.contiguous()
        p.register_hook(_contig_or_none)

        lname = n.lower()
        if ("backbone" in lname) and ('proj_and_up' not in lname):
            back_params.append(p)
            back_params_count += int(np.prod([int(s_) for s_ in p.shape]))
        elif ("class_embed" in lname) or ("bbox_embed" in lname) or ('det_' in lname) or ('_det' in lname):
            det_params.append(p)
            det_params_count += int(np.prod([int(s_) for s_ in p.shape]))
        elif ("input_proj" in lname) or ('proj_and_up' in lname):
            if '_critique' in lname and args.lr_critique is None:                    
                critique_params.append(p)
                critique_params_count += int(np.prod([int(s_) for s_ in p.shape]))      
            elif '_pose' in lname and args.lr_pose is None:         
                pose_params.append(p)
                pose_params_count += int(np.prod([int(s_) for s_ in p.shape]))
            elif '_obj' in lname and args.lr_obj is None:
                obj_params.append(p)
                obj_params_count += int(np.prod([int(s_) for s_ in p.shape]))
            elif args.lr_det is None and '_critique' not in lname and '_pose' not in lname and '_obj' not in lname:  
                det_params.append(p)
                det_params_count += int(np.prod([int(s_) for s_ in p.shape]))
            else:
                proj_params.append(p)
                proj_params_count += int(np.prod([int(s_) for s_ in p.shape]))
            
        elif ("obj_" in lname) or ('_obj' in lname):
            obj_params.append(p)
            obj_params_count += int(np.prod([int(s_) for s_ in p.shape]))
        elif ("pose_" in lname) or ('pose_' in lname):
            pose_params.append(p)
            pose_params_count += int(np.prod([int(s_) for s_ in p.shape]))
        elif ("critique_" in lname) or ('critique_' in lname):
            critique_params.append(p)
            critique_params_count += int(np.prod([int(s_) for s_ in p.shape]))        
        elif ("transformer.encoder" in lname or ('transformer.level_embed' in lname)):
            tfencoder_params.append(p)
            tfencoder_params_count += int(np.prod([int(s_) for s_ in p.shape]))
        else:           
                
            if is_main_process():
                print('[!!WARN!!][other untracked params]', lname)
            other_params.append(p)
            other_params_count += int(np.prod([int(s_) for s_ in p.shape]))
    
    param_count = back_params_count + det_params_count + obj_params_count + other_params_count + proj_params_count+tfencoder_params_count
    
    groups = []
    
    if back_params:
        if args.lr_backbone is not None:
            for p in back_params:
                p.requires_grad = True
            if is_main_process():
                print('[INFO] Training {} backbone layers with {}mio params ({}%) and lr: {}'.format(
                    len(back_params), 
                    np.round(back_params_count / 1000000, 5),
                    np.round(100*(back_params_count / param_count),3),
                    args.lr_backbone
                    ))
            groups.append({"params": back_params, "lr": args.lr_backbone})
        else:
            if is_main_process():
                print('[WARN] Freezing {} backbone layers with {}mio params ({}%)'.format(
                    len(back_params), 
                    np.round(back_params_count / 1000000, 5),
                    np.round(100*(back_params_count / param_count),3)
                    ))
            for p in back_params:
                p.requires_grad = False
    elif is_main_process():
        print('no backbone layers')

    if proj_params:
        if args.lr_proj is not None:
            for p in proj_params:
                p.requires_grad = True
            if is_main_process():
                print('[INFO] Training {} proj layers with {}mio params ({}%) and lr: {}'.format(
                    len(proj_params), 
                    np.round(proj_params_count / 1000000, 5),
                    np.round(100*(proj_params_count / param_count),3),
                    args.lr_proj
                    ))
            groups.append({"params": proj_params, "lr": args.lr_proj})
        else:
            if is_main_process():
                print('[WARN] Freezing {} proj layers with {}mio params ({}%)'.format(
                    len(proj_params), 
                    np.round(proj_params_count / 1000000, 5),
                    np.round(100*(proj_params_count / param_count),3)
                    ))
            for p in proj_params:
                p.requires_grad = False
    elif is_main_process():
        print('no proj layers')
    
    if tfencoder_params:
        if args.lr_tfencode is not None:
            for p in tfencoder_params:
                p.requires_grad = True
            if is_main_process():
                print('[INFO] Training {} tf encoder layers with {}mio params ({}%) and lr: {}'.format(
                    len(tfencoder_params), 
                    np.round(tfencoder_params_count / 1000000, 5),
                    np.round(100*(tfencoder_params_count / param_count),3),
                    args.lr_tfencode
                    ))
            groups.append({"params": tfencoder_params, "lr": args.lr_tfencode})
        else:
            if is_main_process():
                print('[WARN] Freezing {} tf encoder layers with {}mio params ({}%)'.format(
                    len(tfencoder_params), 
                    np.round(tfencoder_params_count / 1000000, 5),
                    np.round(100*(tfencoder_params_count / param_count),3)
                    ))
            for p in tfencoder_params:
                p.requires_grad = False
    elif is_main_process():
        print('no tf encoder layers')


    if det_params:
        if args.lr_det is not None:
            for p in det_params:
                p.requires_grad = True
            if is_main_process():
                print('[INFO] Training {} det layers with {}mio params ({}%) and lr: {}'.format(
                    len(det_params), 
                    np.round(det_params_count / 1000000, 5),
                    np.round(100*(det_params_count / param_count),3),
                    args.lr_det
                    ))
            groups.append({"params": det_params, "lr": args.lr_det})
        else:
            if is_main_process():
                print('[WARN] Freezing {} det layers with {}mio params ({}%)'.format(
                    len(det_params), 
                    np.round(det_params_count / 1000000, 5),
                    np.round(100*(det_params_count / param_count),3)
                    ))
            for p in det_params:
                p.requires_grad = False
    elif is_main_process():
        print('no obj layers')
        
    if obj_params:
        if args.lr_obj is not None:
            for p in obj_params:
                p.requires_grad = True
            if is_main_process():
                print('[INFO] Training {} obj layers with {}mio params ({}%) and lr: {}'.format(
                    len(obj_params), 
                    np.round(obj_params_count / 1000000, 5),
                    np.round(100*(obj_params_count / param_count),3),
                    args.lr_obj
                    ))
            groups.append({"params": obj_params, "lr": args.lr_obj})
        else:
            if is_main_process():
                print('[WARN] Freezing {} obj layers with {}mio params ({}%)'.format(
                    len(obj_params), 
                    np.round(obj_params_count / 1000000, 5),
                    np.round(100*(obj_params_count / param_count),3)
                    ))
            for p in obj_params:
                p.requires_grad = False
    elif is_main_process():
        print('no obj layers')

    if pose_params:
        if args.lr_pose is not None:
            for p in pose_params:
                p.requires_grad = True
            if is_main_process():
                print('[INFO] Training {} pose layers with {}mio params ({}%) and lr: {}'.format(
                    len(pose_params), 
                    np.round(pose_params_count / 1000000, 5),
                    np.round(100*(pose_params_count / param_count),3),
                    args.lr_pose
                    ))
            groups.append({"params": pose_params, "lr": args.lr_pose})
        else:
            if is_main_process():
                print('[WARN] Freezing {} pose layers with {}mio params ({}%)'.format(
                    len(pose_params), 
                    np.round(pose_params_count / 1000000, 5),
                    np.round(100*(pose_params_count / param_count),3)
                    ))
            for p in pose_params:
                p.requires_grad = False
    elif is_main_process():
        print('no pose layers')

    if critique_params:
        if args.lr_critique is not None:
            for p in critique_params:
                p.requires_grad = True
            if is_main_process():
                print('[INFO] Training {} critique layers with {}mio params ({}%) and lr: {}'.format(
                    len(critique_params), 
                    np.round(critique_params_count / 1000000, 5),
                    np.round(100*(critique_params_count / param_count),3),
                    args.lr_critique
                    ))
            groups.append({"params": critique_params, "lr": args.lr_critique})
        else:
            if is_main_process():
                print('[WARN] Freezing {} critique layers with {}mio params ({}%)'.format(
                    len(critique_params), 
                    np.round(critique_params_count / 1000000, 5),
                    np.round(100*(critique_params_count / param_count),3)
                    ))
            for p in critique_params:
                p.requires_grad = False
    elif is_main_process():
        print('no critique layers')


        
    if other_params:
        if args.lr_other > 0.0:
            for p in other_params:
                p.requires_grad = True
            if is_main_process():
                print('[INFO] Training {} other layers with {}mio params ({}%) and lr: {}'.format(
                        len(other_params), 
                        np.round(other_params_count / 1000000, 5),
                        np.round(100*(other_params_count / param_count),3),
                        args.lr_other
                        ))
            groups.append({"params": other_params, "lr": args.lr_other})  # default group
        else:
            if is_main_process():
                print('[WARN] Freezing {} other layers with {}mio params ({}%)'.format(
                    len(other_params), 
                    np.round(other_params_count / 1000000, 5),
                    np.round(100*(other_params_count / param_count),3)
                    ))
            for p in other_params:
                p.requires_grad = False        
    elif is_main_process():
        print('no other layers')

    return groups


def save_checkpoint(state, args, name="checkpoint.pth"):
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(state, os.path.join(args.output_dir, name))


def _report_nan_counters(losses_dict: Dict[str, torch.Tensor], phase: str, epoch: int, step: int):
    if losses_dict is None:
        return
    bad = []
    for k, v in losses_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("pose_nan_"):
            val = float(v.detach().cpu().item())
            if val > 0:
                bad.append((k, int(val)))
    if bad and is_main_process():
        pairs = ", ".join([f"{k}={cnt}" for k, cnt in bad])
        print(f"\n[NAN] {phase} e{epoch+1} s{step+1}: {pairs}")



def main():
    args = parse_args()
    init_distributed_mode(args)
    set_seed(args.seed + (args.rank if hasattr(args, "rank") else 0))
    print('device', args.device)

    # Build data
    train_ds, val_ds = build_mixer(args)
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if (val_ds is not None and args.distributed) else None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_mixed_batch,
        pin_memory=(args.device.type == "cuda"),
        persistent_workers=False,         
        timeout=args.dataloader_timeout
    )
 
    val_subset_loader, val_full_loader = make_val_loaders(args, val_ds, collate_mixed_batch)

    # After building loaders (train_loader, val_subset_loader, val_full_loader) and before building model
    if args.viz_val and not args.viz_train:
        if val_full_loader is None and val_subset_loader is None:
            print("No validation loader available for visualization.")
            return
        vis_loader = val_subset_loader if val_subset_loader is not None else val_full_loader
        print("Visualization-only mode: showing validation batches...")
        for j, batch in enumerate(vis_loader):
            if is_main_process():
                visualize_batch(batch, title_prefix=f"VALID batch {j}")
        return

    encoder, encoder_channels, decoder_channels = build_encoder(
        args,
        return_layers=True,
        cat_outs=args.cat_outs,
        resize_layers=False,
        vit_return_layers=[2, 5, 8, 11],
        with_generator=False,
        norm_return_layers=args.norm_return_layers,
        norm_return_layers_rgb=args.norm_return_layers_rgb,
        upsample_outs=args.resize_attention_maps
    )
    
    if args.pretrained_rgbd:
        if is_main_process():
            print('[INFO] loading pretrained deform detr from {}'.format(args.pretrained_rgbd))
        sd = torch.load(args.pretrained_rgbd, map_location="cpu")['state_dict']
        encoder = load_fitting_state_dict(encoder, sd) 
        del sd  
        
    
    # Build model/criterion
    model, criterion = build_model_and_criterion(args, encoder)

    # load model checkpoint 
    ckpt = None
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        if "model" in ckpt:
            if isinstance(model, DDP):
                model.module = load_fitting_state_dict(model.module, ckpt["model"]) 
            else:
                model = load_fitting_state_dict(model, ckpt["model"])   

    det_enabled = criterion.with_det
    obj_enabled = criterion.with_obj
    pose_enabled = criterion.with_pose
    crit_enabled = criterion.with_critique

    model.to(args.device)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=args.find_unused_parameters, gradient_as_bucket_view=False)

    # Optimizer
    groups = param_groups(model, args)
    if args.opt == "adamw":
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(groups, weight_decay=args.weight_decay, momentum=0.9)

    # Scheduler
    if args.lr_drop < 1:
        lr_drop = int(args.epochs * args.lr_drop)
    else:
        lr_drop = int(args.lr_drop)

    if is_main_process():
        print('[INFO] dropping lr at {}'.format(lr_drop))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop) if lr_drop > 0 else None

    wb = init_wandb(args)
    

    best_train_loss = float("inf")
    best_train_epoch = -1

    best_val_loss = float("inf")
    best_val_epoch = -1

    best_val_map = -1.0
    best_val_map_epoch = -1

    if wb is not None:
        wb.summary["best/train_loss"] = None
        wb.summary["best/train_epoch"] = None
        wb.summary["best/val_loss"] = None
        wb.summary["best/val_epoch"] = None
        wb.summary["best/val_mAP"] = None
        wb.summary["best/val_mAP_epoch"] = None

    logger = ProgressLogger(args, total_epochs=args.epochs, output_dir=args.output_dir, prev_train_epoch_avg=0.0)

    train_steps_total = len(train_loader)
    val_steps_total = len(val_subset_loader) if val_subset_loader is not None else 0
    logger.set_steps(train_steps_total, val_steps_total, eval_every=args.eval_every)
    

    start_epoch = 0
    if ckpt is not None:
            #(model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt and not args.restart:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt and not args.restart:
            start_epoch = ckpt["epoch"] + 1
        if "scheduler" in ckpt and scheduler is not None and not args.restart:
            scheduler.load_state_dict(ckpt["scheduler"])

        if is_main_process():
            print("{} from {} at epoch {}".format('Resumed' if not args.restart else 'Restarted', args.resume, start_epoch+1))   
              
        # Load Δprev baseline from history (last completed epoch’s train loss)
        prev_epoch_train_avg = _load_prev_train_epoch_avg(args, start_epoch if 'epoch' not in ckpt else ckpt.get("epoch", 0) + 1)
        logger.set_prev_train_epoch_avg(prev_epoch_train_avg)
    else:
        prev_epoch_train_avg = 0.0
        logger.set_prev_train_epoch_avg(prev_epoch_train_avg)

    global_step = start_epoch * train_steps_total
    global_step_val = start_epoch * val_steps_total

    print(f'setting criterion step to {start_epoch}x{len(train_loader)} = {start_epoch*len(train_loader)}')
    criterion.set_step(start_epoch*len(train_loader))
    #prev_epoch_train_avg = 0.0
    for epoch in range(start_epoch, args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        model.train(); criterion.train()
        if args.lr_critique is None:
            if args.distributed:
                model.module._freeze_and_eval_critique()
            else:
                model._freeze_and_eval_critique()

        if args.lr_det is None:
            if args.distributed:
                model.module._freeze_and_eval_det()
            else:
                model._freeze_and_eval_det()

        if args.lr_obj is None:
            if args.distributed:
                model.module._freeze_and_eval_obj()
            else:
                model._freeze_and_eval_obj()   

        if args.lr_backbone is None:
            if args.distributed:
                model.module._freeze_and_eval_backbone()
            else:
                model._freeze_and_eval_backbone()         
        
        if args.lr_pose is None:
            if args.distributed:
                model.module._freeze_and_eval_pose()
            else:
                model._freeze_and_eval_pose()            

        train_loss_sum_local = 0.0
        train_steps_local = 0
        logger.set_prev_train_epoch_avg(prev_epoch_train_avg)
        logger.start_train_epoch()

        det_sums = {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "loss_embed": 0.0}
        obj_sums = {"loss_obj_scale": 0.0, "loss_obj_pos": 0.0, "loss_obj_vis": 0.0, "loss_obj_shape": 0.0,  "loss_obj_vol": 0.0, 'obj_slot_miss': 0.0}
        crit_sums = {"critique_acc": 0.0, "critique_acc_bad": 0.0, "critique_acc_random": 0.0, 'critique_acc_good': 0.0, 'critique_acc_mid': 0.0} if args.with_critique else {} 
        pose_sums = {"pose_avg_crit_score": 0.0, "pose_conf_err_rate": 0.0, "pose_conf_err_l1": 0.0, "pose_slot_acc": 0.0, "pose_chamfer_avg": 0.0,
                        "grasp_nn_trans_avg": 0.0, "grasp_nn_angle_deg_avg": 0.0, "grasp_nn_close_pct": 0.0}   

        steps_total = len(train_loader)
        train_iter = iter(train_loader)
        steps_done = 0

        extras_avg = {}
        while steps_done < steps_total:
            batch, train_iter, ok = fetch_next_batch_sync(
                loader=train_loader,
                loader_iter=train_iter,
                device=args.device,
                max_retries=getattr(args, "max_fetch_retries", 5)
            )
            if not ok:
                break  # end-of-epoch cleanly

            # Use steps_done as "i" for logging
            i = steps_done

            #for i, batch in enumerate(train_loader):
            
            if args.viz_train and is_main_process():
                visualize_batch(batch, title_prefix=f"TRAIN epoch {epoch} step {i}")

            imgs = [b["img"].to(args.device) for b in batch]
            
            if args.with_depth and not args.rgb_only:
                samples = nested_tensor_from_tensor_list(imgs, [b["depth"].to(args.device) for b in batch])  
            else:
                samples = nested_tensor_from_tensor_list(imgs)

            targets = []
            for b in batch:
                t = {}
                t["boxes"] = b["boxes"].to(args.device)
                t["labels"] = torch.ones((b["boxes"].shape[0],), dtype=torch.long, device=args.device)
                t["box2inst"] = b["box2inst"].to(args.device)
                if b.get("inst_crops_images") is not None:
                    t["inst_crops_images"] = b["inst_crops_images"].to(args.device) 
                    t["inst_crops_inst_ids"] = b["inst_crops_inst_ids"].to(args.device)

                if criterion.with_obj:
                    t["obj_valid"] = _to_device_nested(b["props"]["valid"], args.device)
                    t["obj_scale"] = _to_device_nested(b["props"]["scale"], args.device)
                    t["obj_pos"]   = _to_device_nested(b["props"]["pos"], args.device)
                    t["obj_vis"]   = _to_device_nested(b["props"]["vis"], args.device)
                    t["obj_vis_mask"]   = _to_device_nested(b["props"]["vis_mask"], args.device)
                    t["obj_points_list"] = _to_device_nested(b["props"]["points_list"], args.device)
                    t["obj_vol"] = _to_device_nested(b["props"]["vol"], args.device)

                if criterion.with_pose:
                    t["poses"] = _to_device_nested(b["poses"], args.device)
                    t["models"] = _to_device_nested(b["models"], args.device)

                if args.with_grasp:
                    t["gt_grasps"] = _to_device_nested(b["gt_grasps"], args.device)
                    t["bad_grasps"] = _to_device_nested(b["bad_grasps"], args.device)
                if args.add_model_embed:
                    t["pose_6d_proposals_good"] = _to_device_nested(b["pose_6d_proposals_good"], args.device)
                    t["pose_6d_proposals_bad"] = _to_device_nested(b["pose_6d_proposals_bad"], args.device)
                    if not criterion.with_pose:                        
                        t["models"] = _to_device_nested(b["models"], args.device)
                targets.append(t)        

            outs, _ = model(samples, targets=targets)
            
            if steps_done == steps_total -1:
                save_last_batch_plots(mode="train", epoch=epoch + 1, args=args, batch=batch, outs=outs, n_images_saved=min(args.n_images_saved, len(batch)), score_thresh=args.val_score_thresh)


            losses_dict = criterion(outs, targets)
            _report_nan_counters(losses_dict, phase="train", epoch=epoch, step=i)
            loss = [v for k, v in losses_dict.items() if k.startswith("loss_")]

            for key, v in losses_dict.items():
                if torch.isnan(v):
                    print(losses_dict)
                    input()    

            loss = sum(loss)          
            
            if torch.isnan(loss):
                print('loss', loss)
                print('losses_dict', losses_dict)
                input()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            criterion.step()

            steps_done = i + 1
            steps_pct = 100.0 * steps_done / max(1, steps_total)
            current_lr = optimizer.param_groups[-1]["lr"]

            global_step += 1

            train_loss_sum_local += float(loss.item())
            train_steps_local += 1
            running_mean = train_loss_sum_local / max(1, train_steps_local)
            det_comp = {k: float(losses_dict.get(k, torch.tensor(0.0)).item())
                        for k in ("loss_cls","loss_bbox","loss_giou", "loss_embed")}
            matcher_stats = {k: float(losses_dict.get(k, torch.tensor(0.0)).item())
                            for k in ("match_cost_class_mean","match_cost_bbox_mean","match_cost_giou_mean",
                                    "match_cost_class_std","match_cost_bbox_std","match_cost_giou_std","match_pairs")}

            

            #print('')
            #print(losses_dict.keys())

            extras_avg = {}
            if det_enabled:
                # Update running sums
                for k in det_sums.keys():
                    det_sums[k] += float(losses_dict.get(k, torch.tensor(0.0)).item())
                extras_avg.update({k: (det_sums[k] / train_steps_local) for k in det_sums.keys()})

            if obj_enabled:
                # Prepare extras running averages
                for k in obj_sums.keys():
                    obj_sums[k] += float(losses_dict.get(k, torch.tensor(0.0)).item())
                extras_avg.update({k: (obj_sums[k] / train_steps_local) for k in obj_sums.keys()})

            if pose_enabled:
                for k in pose_sums.keys():
                    v = losses_dict.get(k, None)
                    if isinstance(v, torch.Tensor):
                        pose_sums[k] += float(v.item())
                        extras_avg[k] = pose_sums[k] / max(1, train_steps_local)

            if crit_enabled:
                for k in crit_sums.keys():
                    v = float(losses_dict.get(k, torch.tensor(0.0)).item())
                    crit_sums[k] += v
                # running averages
                extras_avg.update({k: (crit_sums[k] / train_steps_local) for k in crit_sums.keys()})

            logger.update_train(
                epoch_idx=epoch,
                total_epochs=args.epochs,
                step_idx=i,
                steps_total=steps_total,
                batch_loss=float(loss.item()),
                running_mean_loss=running_mean,
                extras_avg=extras_avg,
                write_csv=(is_main_process() and ((i + 1) % args.log_csv_every == 0))
            )

            # Log per-step to W&B

            if wb is not None:
                log_items = {
                    "train/loss": float(loss.item()),
                    "train/loss_running": running_mean,
                    "train/epoch": epoch + 1,
                    "train/step": steps_done,
                    "train/steps_total": steps_total,
                    "train/steps_pct": steps_pct,
                    "train/lr": current_lr,
                    "global_step": global_step,
                }
                # add all sub-losses from criterion
                for k, v in losses_dict.items():
                    if 'aux' not in k:
                        log_items[f"train/losses/{k}"] = float(v.item())

                for k, v in extras_avg.items():                    
                    log_items[f"train/extra/{k}"] = v

                wb.log(log_items, step=global_step)

            #if is_main_process():    
            #    input()
            

        # All-reduce to get epoch train mean across ranks
        if dist.is_available() and dist.is_initialized():
            v = torch.tensor([train_loss_sum_local, train_steps_local], device=args.device)
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            train_loss_sum_global, train_steps_global = float(v[0].item()), int(v[1].item())
        else:
            train_loss_sum_global, train_steps_global = train_loss_sum_local, train_steps_local

        train_loss_avg = train_loss_sum_global / max(1, train_steps_global)
        prev_epoch_train_avg = train_loss_avg  # for next epoch delta
        logger.newline()


        if scheduler is not None:
            scheduler.step()

        # Save checkpoint

        def get_orig(mod):
            # unwrap DDP and compiled wrappers
            if hasattr(mod, "module"):
                mod = mod.module
            return getattr(mod, "_orig_mod", mod)

        orig = get_orig(model)
            
        ckpt = {
            "model": orig.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch, 
            "args": vars(args),
        }
        save_checkpoint(ckpt, args, name="checkpoint.pth")        

        #save_checkpoint(ckpt, args, name="checkpoint_ep_{}.pth".format(epoch))  

        if wb is not None:
            train_wb_logs = {
                "train/loss_epoch": train_loss_avg,
                "train/epoch": epoch + 1,
                "global_step": global_step,
            }
            for k, v in extras_avg.items():                    
                train_wb_logs[f"train/avg_metric/{k}"] = v
            wb.log(train_wb_logs, step=global_step)

            
        if train_loss_avg < best_train_loss:
            best_train_loss = train_loss_avg
            best_train_epoch = epoch + 1
            if wb is not None:
                wb.summary["best/train_loss"] = best_train_loss
                wb.summary["best/train_epoch"] = best_train_epoch

        rgbd_sums = {}    # name -> (L,2)
        rgbd_counts = {}  # name -> int

        val_loss_avg, coco_metrics = None, None
        if (val_subset_loader is not None) and ((epoch + 1) % args.eval_every == 0):
            model.eval(); criterion.eval()
            logger.start_val_epoch()

            # Step-wise val loss printing over SUBSET
            val_loader = val_subset_loader
            val_steps_total = len(val_loader)            
            val_iter = iter(val_loader)
            val_det_sums = {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "loss_embed": 0.0}
            val_obj_sums = {"loss_obj_scale": 0.0, "loss_obj_pos": 0.0, "loss_obj_vis": 0.0, "loss_obj_shape": 0.0, "loss_obj_vol": 0.0, 'obj_slot_miss': 0.0}
            val_crit_sums = {"critique_acc": 0.0, "critique_acc_bad": 0.0, "critique_acc_random": 0.0, 'critique_acc_good': 0.0, 'critique_acc_mid': 0.0} 
            val_pose_sums = {"pose_avg_crit_score": 0.0, "pose_conf_err_rate": 0.0, "pose_conf_err_l1": 0.0, "pose_slot_acc": 0.0, "pose_chamfer_avg": 0.0,
                                "grasp_nn_trans_avg": 0.0, "grasp_nn_angle_deg_avg": 0.0, "grasp_nn_close_pct": 0.0}       

            val_loss_sum_local = 0.0
            val_steps_local = 0
            val_steps_done = 0
            extras_avg_val = {}

            while val_steps_done < val_steps_total:
                batch, val_iter, ok = fetch_next_batch_sync(
                    loader=val_loader,
                    loader_iter=val_iter,
                    device=args.device,
                    max_retries=getattr(args, "max_fetch_retries", 5)
                )
                if not ok:
                    continue
                
                #for j, batch in enumerate(val_loader):
                if args.viz_val and is_main_process():
                    visualize_batch(batch, title_prefix=f"VALID epoch {epoch} step {val_steps_done}")

                imgs = [b["img"].to(args.device) for b in batch]
                
                if args.with_depth and not args.rgb_only:
                    samples = nested_tensor_from_tensor_list(imgs, [b["depth"].to(args.device) for b in batch])   
                else:
                    samples = nested_tensor_from_tensor_list(imgs)

                targets = []
                for b in batch:
                    t = {}
                    t["boxes"] = b["boxes"].to(args.device)
                    t["labels"] = torch.ones((b["boxes"].shape[0],), dtype=torch.long, device=args.device)
                    t["box2inst"] = b["box2inst"].to(args.device)
                    if b.get("inst_crops_images") is not None:
                        t["inst_crops_images"] = b["inst_crops_images"].to(args.device) 
                        t["inst_crops_inst_ids"] = b["inst_crops_inst_ids"].to(args.device)


                    if criterion.with_obj:
                        t["obj_valid"] = _to_device_nested(b["props"]["valid"], args.device)
                        t["obj_scale"] = _to_device_nested(b["props"]["scale"], args.device)
                        t["obj_pos"]   = _to_device_nested(b["props"]["pos"], args.device)
                        t["obj_vis"]   = _to_device_nested(b["props"]["vis"], args.device)
                        t["obj_vis_mask"]   = _to_device_nested(b["props"]["vis_mask"], args.device)
                        t["obj_points_list"] = _to_device_nested(b["props"]["points_list"], args.device)
                        t["obj_vol"] = _to_device_nested(b["props"]["vol"], args.device)

                    if criterion.with_pose:
                        t["poses"] = _to_device_nested(b["poses"], args.device)
                        t["models"] = _to_device_nested(b["models"], args.device)

                    if args.with_grasp:
                        t["gt_grasps"] = _to_device_nested(b["gt_grasps"], args.device)
                        t["bad_grasps"] = _to_device_nested(b["bad_grasps"], args.device)
                    if args.add_model_embed:
                        t["pose_6d_proposals_good"] = _to_device_nested(b["pose_6d_proposals_good"], args.device)
                        t["pose_6d_proposals_bad"] = _to_device_nested(b["pose_6d_proposals_bad"], args.device)
                        if not criterion.with_pose:                        
                            t["models"] = _to_device_nested(b["models"], args.device)
                    targets.append(t)

                
                with torch.no_grad():
                    outs, _ = model(samples, targets=targets)
                    losses = criterion(outs, targets)
                    _report_nan_counters(losses, phase="val", epoch=epoch, step=val_steps_done)

                if val_steps_done == val_steps_total -1:
                    save_last_batch_plots(mode="test", epoch=epoch + 1, args=args, batch=batch, outs=outs, n_images_saved=min(args.n_images_saved, len(batch)), score_thresh=args.val_score_thresh)

                #loss_val = sum(losses.values())
                loss_val = sum(v for k, v in losses.items() if k.startswith("loss_"))

                val_loss_sum_local += float(loss_val.item())
                val_steps_local += 1
                running_mean_val = val_loss_sum_local / max(1, val_steps_local)

                det_comp_val = {k: float(losses.get(k, torch.tensor(0.0)).item())
                                for k in ("loss_cls","loss_bbox","loss_giou", "loss_embed")}

                matcher_stats_val = {k: float(losses.get(k, torch.tensor(0.0)).item())
                                    for k in ("match_cost_class_mean","match_cost_bbox_mean","match_cost_giou_mean",
                                            "match_cost_class_std","match_cost_bbox_std","match_cost_giou_std","match_pairs")}

                extras_avg_val = {}
                if det_enabled:
                    # Update running sums
                    for k in val_det_sums.keys():
                        val_det_sums[k] += float(losses.get(k, torch.tensor(0.0)).item())
                    extras_avg_val.update({k: (val_det_sums[k] / val_steps_local) for k in val_det_sums.keys()})

                if obj_enabled:
                    # Prepare extras running averages
                    for k in val_obj_sums.keys():
                        val_obj_sums[k] += float(losses.get(k, torch.tensor(0.0)).item())
                    extras_avg_val.update({k: (val_obj_sums[k] / val_steps_local) for k in val_obj_sums.keys()})
                
                if pose_enabled:
                    for k in val_pose_sums.keys():
                        v = losses.get(k, None)
                        if isinstance(v, torch.Tensor):
                            val_pose_sums[k] += float(v.item())
                            extras_avg_val[k] = val_pose_sums[k] / max(1, val_steps_local)

                #print('crit_enabled', crit_enabled)
                if crit_enabled:
                    #print('losses', losses.keys())
                    for k in val_crit_sums.keys():
                        v = float(losses.get(k, torch.tensor(0.0)).item())
                        val_crit_sums[k] += v
                        #print('k', v, val_crit_sums[k])
                    # running averages
                    extras_avg_val.update({k: (val_crit_sums[k] / val_steps_local) for k in val_crit_sums.keys()})

               
                rgbd_out = outs.get('rgb_depth_pct', None)
                if isinstance(rgbd_out, dict):
                    for name, p in rgbd_out.items():
                        if not isinstance(p, torch.Tensor):
                            continue
                        p = p.detach()               # (L,B,2)
                        batch_sum = p.sum(dim=1)     # (L,2)
                        if name not in rgbd_sums:
                            rgbd_sums[name] = torch.zeros_like(batch_sum)
                            rgbd_counts[name] = 0
                        if rgbd_sums[name].shape[0] != batch_sum.shape[0]:
                            minL = min(rgbd_sums[name].shape[0], batch_sum.shape[0])
                            rgbd_sums[name][:minL] += batch_sum[:minL]
                        else:
                            rgbd_sums[name] += batch_sum
                        rgbd_counts[name] += p.shape[1]

                        

                logger.update_val(
                    epoch_idx=epoch,
                    total_epochs=args.epochs,
                    step_idx=val_steps_done,
                    steps_total=val_steps_total,
                    batch_loss=float(loss_val.item()),
                    running_mean_loss=running_mean_val,
                    extras_avg=extras_avg_val,
                    write_csv=(is_main_process() and ((val_steps_done + 1) % args.log_csv_every == 0)),
                    final_map=None
                )                
                
                val_steps_done += 1
                global_step_val += 1

                # W&B per-step validation (subset)
                '''
                if wb is not None:
                    log_items = {
                        "val_subset/loss": float(loss_val.item()),
                        "val_subset/loss_running": running_mean_val,
                        "val_subset/epoch": epoch + 1,
                        "val_subset/step": val_steps_done,
                        "val_subset/steps_total": val_steps_total,
                        "global_step_val": global_step_val,
                    }
                    for k, v in losses.items():
                        if 'aux' not in k:
                            log_items[f"val_subset/losses/{k}"] = float(v.item())

                    for k, v in extras_avg_val.items():
                        log_items[f"val_subset/extra/{k}"] = v
                    
                    wb.log(log_items, step=global_step_val)
                '''
                

            # All-reduce subset val loss
            if dist.is_available() and dist.is_initialized():
                v = torch.tensor([val_loss_sum_local, val_steps_local], device=args.device)
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
                val_loss_sum_global, val_steps_global = float(v[0].item()), int(v[1].item())
            else:
                val_loss_sum_global, val_steps_global = val_loss_sum_local, val_steps_local
            val_loss_avg = {"total": val_loss_sum_global / max(1, val_steps_global)}


            # compute global mean per-level RGB/Depth contribution across validation subset
            rgbd_means = {}
            if rgbd_sums:
                for name, sum_t in rgbd_sums.items():
                    sum_tensor = sum_t.to(args.device)
                    cnt_tensor = torch.tensor([rgbd_counts[name]], dtype=sum_tensor.dtype, device=args.device)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)
                    denom = max(float(cnt_tensor.item()), 1e-12)
                    rgbd_means[name] = (sum_tensor / denom).detach().cpu()  # (L,2)

            if rgbd_means and is_main_process():
                print('')
                print("RGB/Depth contribution (val subset mean):")
                for name, rgbd_mean in rgbd_means.items():
                    print(f"  [{name}]")
                    for li in range(rgbd_mean.shape[0]):
                        print(f"    L{li}: RGB {float(rgbd_mean[li,0]*100):.2f}% | Depth {float(rgbd_mean[li,1]*100):.2f}%")

            coco_metrics = {}
            if getattr(args, "with_det", True) and (args.lr_det is not None):
                coco_metrics = evaluate_coco(val_loader, model, args, score_thresh=args.val_score_thresh)

                
            logger.update_val(
                epoch_idx=epoch,
                total_epochs=args.epochs,
                step_idx=val_steps_total - 1,
                steps_total=val_steps_total,
                batch_loss=val_loss_avg["total"],
                running_mean_loss=val_loss_avg["total"],                
                extras_avg=extras_avg_val,
                write_csv=True,
                final_map=coco_metrics
            )
            logger.newline()

            # W&B subset summary
            if wb is not None:
                log_dict = {"val_subset/loss_epoch": val_loss_avg["total"], "val_subset/epoch": epoch + 1, "global_step": global_step}
                if coco_metrics:
                    log_dict.update({f"val_subset/{k}": v for k, v in coco_metrics.items()})
                if rgbd_means:
                    for name, rgbd_mean in rgbd_means.items():
                        for li in range(rgbd_mean.shape[0]):
                            log_dict[f"val_subset/rgbd/{name}/l{li}/rgb_pct"] = float(rgbd_mean[li, 0])
                            log_dict[f"val_subset/rgbd/{name}/l{li}/depth_pct"] = float(rgbd_mean[li, 1])
                for k, v in extras_avg_val.items():
                    log_dict[f"val_subset/avg_metric/{k}"] = v
                wb.log(log_dict, step=global_step)

            # Track best validation loss (subset) and save best-loss checkpoint
            current_val_loss = val_loss_avg.get("total", float("inf"))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_val_epoch = epoch + 1
                save_best_loss_checkpoint(model, optimizer, scheduler, epoch, args)
                if wb is not None:
                    wb.summary["best/val_loss"] = best_val_loss
                    wb.summary["best/val_epoch"] = best_val_epoch

            # Track best mAP (subset)
            current_map = coco_metrics.get("AP", -1.0) if coco_metrics else -1.0
            if current_map > best_val_map:
                best_val_map = current_map
                best_val_map_epoch = epoch + 1
                if wb is not None:
                    wb.summary["best/val_mAP"] = best_val_map
                    wb.summary["best/val_mAP_epoch"] = best_val_map_epoch

        # Persist per-epoch logs
        bests = {
            "train_loss": best_train_loss,
            "train_epoch": best_train_epoch,
            "val_loss": best_val_loss,
            "val_epoch": best_val_epoch,
            "val_mAP": best_val_map,
            "val_mAP_epoch": best_val_map_epoch,
        }

        logger.end_epoch()
        update_history(args, epoch + 1, train_loss_avg, val_loss_avg, coco_metrics, bests=bests)

    # Final save
    save_checkpoint(
        {
            "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
            "args": vars(args),
        },
        args,
        name="final.pth",
    )
        
    # ddp_train_mixed.py (end of main(), final full validation)

    if val_full_loader is not None:
        print("\nRunning FINAL full validation...")
        # Full validation losses
        final_val_losses = evaluate_losses(val_full_loader, model, criterion, args)
        # Full COCO metrics
        final_coco_metrics = {}
        if getattr(args, "with_det", True) and (args.lr_det > 0):
            final_coco_metrics = evaluate_coco(val_full_loader, model, args, score_thresh=args.val_score_thresh)

        if is_main_process():
            print("Final Val Losses:", {k: f"{v:.4f}" for k, v in final_val_losses.items()})
            print("Final COCO:", final_coco_metrics)

        # Log to W&B
        if wb is not None:
            log_dict = {f"val_final/{k}": v for k, v in final_val_losses.items()}
            if final_coco_metrics:
                log_dict.update({f"val_final/{k}": v for k, v in final_coco_metrics.items()})
            log_dict["global_step"] = global_step
            wb.log(log_dict, step=global_step)

        # Append to JSON history
        update_history(args, epoch=args.epochs, train_loss_avg=prev_epoch_train_avg,
                    val_loss_avg=final_val_losses, coco_metrics=final_coco_metrics)
    
    final_bests = {
        "train_loss": best_train_loss,
        "train_epoch": best_train_epoch,
        "val_loss": best_val_loss,
        "val_epoch": best_val_epoch,
        "val_mAP": best_val_map,
        "val_mAP_epoch": best_val_map_epoch,
    }
    update_history(args, epoch=args.epochs, train_loss_avg=prev_epoch_train_avg,
                val_loss_avg=final_val_losses, coco_metrics=final_coco_metrics, bests=final_bests)

    if wb is not None:
        wb.finish()

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
    


if __name__ == "__main__":
    main()