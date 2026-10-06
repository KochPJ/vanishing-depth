
# dataloader_utils.py

import os
import re
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import json
from PIL import ImageDraw
from model.deformable_detr.util.misc import nested_tensor_from_tensor_list, is_main_process
from scipy.spatial.transform import Rotation as SciRot
from datasets.multi_obj_sampler import (
    MixedMultiDataset,
    GraspItDataset,
    BOPDataset,
    GraspNetDataset,
    DatasetConfig
)



def _inst_crops_from_mask(img_t: torch.Tensor, mask_t: torch.Tensor, size=224, keep_aspect=True):
    """
    Build per-instance masked crops in tensor space (no PIL, no de-norm).
    - img_t: [3,H,W], ImageNet-normalized float tensor
    - mask_t: [H,W] int64 instance ids (0 = background)

    Returns:
      crops: list[Tensor(3,size,size)]  (normalized)
      inst_ids: list[int]
    """
    if img_t is None or mask_t is None:
        return [], []

    assert img_t.dim() == 3 and img_t.size(0) == 3, "img_t must be [3,H,W]"
    H, W = mask_t.shape
    inst_ids = torch.unique(mask_t)
    inst_ids = inst_ids[(inst_ids > 0)].tolist()

    crops, ids = [], []
    for inst_id in inst_ids:
        # bbox of this instance
        ys, xs = torch.where(mask_t == inst_id)
        if xs.numel() == 0:
            continue
        y1, y2 = int(ys.min().item()), int(ys.max().item())
        x1, x2 = int(xs.min().item()), int(xs.max().item())

        # crop image and mask
        img_crop = img_t[:, y1:y2+1, x1:x2+1]           # [3,hc,wc]
        m_crop = torch.zeros((img_crop.shape[-2:]), device=img_crop.device, dtype=img_crop.dtype) #[hc,wc]
        m_crop[mask_t[y1:y2+1, x1:x2+1] == inst_id] = 1.0
        if m_crop.numel() == 0:
            continue

        
        # optional: keep aspect with square pad before resize
        if keep_aspect:
            hc, wc = img_crop.shape[-2:]
            L = max(hc, wc)
            pad_top = (L - hc) // 2
            pad_bottom = L - hc - pad_top
            pad_left = (L - wc) // 2
            pad_right = L - wc - pad_left
            img_crop = F.pad(img_crop, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
            m_crop = F.pad(m_crop, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

        # resize to (size,size)
        #print('img_crop', img_crop.shape)
        #print('m_crop', m_crop.shape)
        img_crop = F.interpolate(img_crop.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
        m_crop = F.interpolate(m_crop.unsqueeze(0).unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)

        # apply mask: background -> 0 (normalized space)
        img_crop = img_crop * m_crop       # broadcast over 3 channels
        
        crops.append(img_crop)   # [3,size,size], still normalized
        ids.append(int(inst_id))
    
    if len(crops) > 0:
        crops = torch.cat(crops, dim=0)
        ids = torch.as_tensor(ids, dtype=torch.long)

        return crops, ids
    else:
        return None, None


# ddp_train_mixed.py — add near the other viz helpers

def _draw_gripper(
    img_pil, grasp, intr, finger_len_m=0.05, default_width_m=0.05,
    line_w=3, color_fingers=(0,255,0), color_link=(255,255,255),
    color_dir=(255,0,0), label=True, label_color=None, label_text=None
):
    cx, cy, fx, fy = [float(x) for x in (intr if not isinstance(intr, torch.Tensor) else intr.detach().cpu().tolist())]
    w, x, y, z = [float(q) for q in grasp.get('quat_wxyz', [1,0,0,0])]
    center = np.asarray(grasp.get('xyz', [0,0,0]), dtype=np.float32)
    width_m = float(grasp.get('width', default_width_m)) or default_width_m
    try:
        from scipy.spatial.transform import Rotation as SciRot
        R = SciRot.from_quat([x,y,z,w]).as_matrix().astype(np.float32)
    except Exception:
        R = np.eye(3, dtype=np.float32)

    # Open along local Y, approach along local Z (matches dataloader viz)
    Yv, Zv = R[:,1], R[:,2]
    half_w = 0.5 * width_m; L = float(finger_len_m)
    left_base  = center - half_w * Yv
    right_base = center + half_w * Yv
    left_tip   = left_base  + L * Zv
    right_tip  = right_base + L * Zv
    dir_end    = center - L * Zv

    def _proj(P):
        Xc, Yc, Zc = float(P[0]), float(P[1]), float(P[2])
        if Zc <= 1e-6: return None
        return (int(round(fx * (Xc/Zc) + cx)), int(round(fy * (Yc/Zc) + cy)))

    p_lb, p_rb, p_lt, p_rt, p_c, p_de = map(_proj, [left_base, right_base, left_tip, right_tip, center, dir_end])
    out = img_pil.copy(); draw = ImageDraw.Draw(out)
    if p_lb and p_lt: draw.line([p_lb, p_lt], fill=color_fingers, width=line_w)
    if p_rb and p_rt: draw.line([p_rb, p_rt], fill=color_fingers, width=line_w)
    if p_lb and p_rb: draw.line([p_lb, p_rb], fill=color_link, width=line_w)
    if p_c:
        r=2; draw.ellipse([(p_c[0]-r,p_c[1]-r),(p_c[0]+r,p_c[1]+r)], fill=(255,255,255))
    if p_c and p_de:
        draw.line([p_c, p_de], fill=color_dir, width=line_w)
        dx, dy = (p_de[0]-p_c[0]), (p_de[1]-p_c[1])
        n = (dx*dx+dy*dy)**0.5
        if n>1e-3:
            ux,uy = dx/n, dy/n; px,py = -uy, ux; ah=6
            draw.line([p_de,(int(p_de[0]-ah*ux+0.5*ah*px),int(p_de[1]-ah*uy+0.5*ah*py))], fill=color_dir, width=line_w)
            draw.line([p_de,(int(p_de[0]-ah*ux-0.5*ah*px),int(p_de[1]-ah*uy-0.5*ah*py))], fill=color_dir, width=line_w)
    if label:
        if label_text is None:
            txt = f"{float(grasp.get('score', 0.0)):.2f}"
        else:
            txt = label_text
        p_mid = (p_lb and p_rb) and (int((p_lb[0]+p_rb[0])/2), int((p_lb[1]+p_rb[1])/2)) or p_c
        if p_mid:
            pos = (p_mid[0]+6, p_mid[1]-12)
            try:
                draw.text(pos, txt, fill=(label_color or color_link), stroke_width=2, stroke_fill=(0,0,0))
            except TypeError:
                draw.text(pos, txt, fill=(label_color or color_link))
    return out



def parse_probs(s):
    # "bop:0.5,graspit:0.5"
    d = {}
    if not s:
        return d
    for tok in s.split(","):
        if not tok: continue
        k, v = tok.split(":")
        d[k.strip()] = float(v.strip())
    return d

def print_inventory_graspit(root: str, assets_root: str = None):
    inv = GraspItDataset.inventory(root)
    print("GraspIt inventory:")
    print(json.dumps(inv, indent=2))

def print_inventory_graspnet(root: str):
    inv = GraspNetDataset.inventory(root)
    print("GraspNet inventory:")
    print(json.dumps(inv, indent=2))

def print_inventory_bop(bop_root: str):
    inv = BOPDataset.inventory(bop_root)
    print("BOP inventory:")
    print(json.dumps(inv, indent=2))


def build_mixer(args):
    datasets = {}
    probs = parse_probs(args.dataset_probs)

    cfg = DatasetConfig(
        rgb_mean=[0.485, 0.456, 0.406],
        rgb_std=[0.229, 0.224, 0.225],
        out_hw=(args.train_height, args.train_width),     # (H, W)
        pc_num_points=768
    )


    bg_dirs = []
    if os.path.exists(args.graspit_root):
        bg_dirs = [os.path.join(args.graspit_root, "backgrounds")]
    
    if is_main_process():
        if os.path.exists(args.graspit_root):            
            print_inventory_graspit(args.graspit_root, args.graspit_assets)
        if os.path.exists(args.graspnet_root):
            print_inventory_graspnet(args.graspnet_root)
        if os.path.exists(args.bop_root):
            print_inventory_bop(args.bop_root)  

    # Train datasets
    if args.use_bop and os.path.isdir(args.bop_root):
        datasets["bop"] = BOPDataset(
            args.bop_root, mode="train",
            transforms=True,
            include_depth=args.include_depth,
            include_mask=args.include_mask,
            include_models=args.include_models,
            rectify_prob=0.0,
            backgrounds_dirs=bg_dirs,
            bg_replace_prob={'default': 0.8, 'ruapc': 1.0, 'ycbv': 0.0},
            min_visible_pixels=50,
            len_target_pc=args.hidden_dim_obj,
            generate_6d_pose_proposals=args.add_model_embed,
            cfg=cfg
        )

    if args.use_graspit and os.path.isdir(args.graspit_root):
        
        datasets["graspit"] = GraspItDataset(
            args.graspit_root, assets_root=args.graspit_assets,
            mode="train", transforms=True,
            rectify_prob=0.0, 
            min_visible_pixels=50,
            include_mask=args.include_mask,
            include_depth=args.include_depth,
            include_models=args.include_models,
            ignore_table=True,
            backgrounds_dirs=bg_dirs,
            bg_replace_prob=0.8,
            len_target_pc=args.hidden_dim_obj,
            with_grasp=args.with_grasp,
            fix_split=True if args.with_grasp else False,
            generate_6d_pose_proposals=args.add_model_embed,
            cfg=cfg,
            overwrite_grasps=False,
            cache_grasps=True,
        )

    # Disable GraspNet for training unless you want synthetic data merges; typically val/test only
    if args.use_graspnet_val_only and os.path.isdir(args.graspnet_root):
        pass

    if not datasets:
        raise RuntimeError("No datasets found for training. Check paths and flags.")

    # Normalize probs over present datasets
    probs_norm = {}
    s = sum([probs.get(k, 0.0) for k in datasets.keys()])
    if s <= 0:
        # equal probs
        p = 1.0 / len(datasets)
        probs_norm = {k: p for k in datasets.keys()}
    else:
        for k in datasets.keys():
            probs_norm[k] = probs.get(k, 0.0) / s

    train_ds = MixedMultiDataset(
        datasets=datasets,
        dataset_probs=probs_norm,
        num_samples_per_epoch=args.num_samples_per_epoch,
        mode="train",            
        cfg=cfg,
        with_critique=args.with_critique,
        with_pose=args.with_pose,
        with_det=args.with_det,
        with_obj=args.with_obj,
        with_grasp=args.with_grasp
    )

    # Simple validation mixer (optional): we can reuse BOP val + GraspIt val if present
    val_datasets = {}
    if args.use_bop_val and os.path.isdir(args.bop_root):
        val_datasets["bop"] = BOPDataset(
            args.bop_root, mode="val",
            transforms=False,
            include_depth=args.include_depth,
            include_mask=args.include_mask,
            include_models=args.include_models,
            rectify_prob=0.0,
            min_visible_pixels=50,
            len_target_pc=args.hidden_dim_obj,
            generate_6d_pose_proposals=args.add_model_embed,
            cfg=cfg
        )
    if args.use_graspit_val and os.path.isdir(args.graspit_root):
        val_datasets["graspit"] = GraspItDataset(
            args.graspit_root, assets_root=args.graspit_assets,
            mode="val", transforms=False,
            rectify_prob=0.0, min_visible_pixels=50,
            include_mask=args.include_mask,
            include_depth=args.include_depth,
            include_models=args.include_models,
            ignore_table=True,
            backgrounds_dirs=bg_dirs,
            bg_replace_prob=0.0,
            len_target_pc=args.hidden_dim_obj,     
            with_grasp=args.with_grasp,      
            fix_split=True if args.with_grasp else False,
            generate_6d_pose_proposals=args.add_model_embed, 
            cfg=cfg,
            overwrite_grasps=False,
            cache_grasps=True,
        )

    if args.use_graspnet_val and os.path.exists(args.graspnet_root):
        val_datasets["graspnet"] = GraspNetDataset(
            args.graspnet_root, 
            mode='val', 
            transforms=False, 
            include_grasp_labels=True, 
            models_root=args.graspnet_models_root, 
            rectify_prob=0.0, 
            min_visible_pixels=50,
            len_target_pc=args.hidden_dim_obj,
            cfg=cfg,
            with_grasp=args.with_grasp,
            generate_6d_pose_proposals=args.add_model_embed,
            overwrite_grasps=False,
            cache_grasps=True,
        )
        #print(f"GraspNet val len={len(ds_graspnet_val)} (train excluded)")


    val_ds = None
    if val_datasets:
        probs_val = {k: 1.0 / len(val_datasets) for k in val_datasets.keys()}
        val_ds = MixedMultiDataset(
            datasets=val_datasets,
            dataset_probs=probs_val,
            num_samples_per_epoch=args.num_val_samples_per_epoch,
            mode="val",            
            cfg=cfg
        )

    return train_ds, val_ds
# Replace the existing _pack_grasp_dict with this version

def _pack_grasp_dict(gdict, default_score: float = 0.0, include_width=True):
    out = {}
    if not isinstance(gdict, dict):
        return out

    for inst_id, lst in gdict.items():
        inst_id = int(inst_id)
        if not lst:
            continue

        xyz_list, quat_list, score_list, width_list = [], [], [], []

        for g in lst:
            try:
                xyz = torch.as_tensor(g.get("xyz"), dtype=torch.float32)
                quat = torch.as_tensor(g.get("quat_wxyz"), dtype=torch.float32)
                score = torch.tensor(float(g.get("score", default_score)), dtype=torch.float32)
            except Exception:
                # Skip malformed entries
                continue

            # Filter if any NaN present in xyz, quat, or score
            if torch.isnan(xyz).any() or torch.isnan(quat).any() or torch.isnan(score):
                continue

            xyz_list.append(xyz)
            quat_list.append(quat)
            score_list.append(score.clamp(0.0, 1.0))
            if include_width:
                w = torch.tensor(float(g.get("width", 0.0)), dtype=torch.float32)
                width_list.append(w)

        if not xyz_list:
            # No valid grasps for this instance id
            continue

        xyz = torch.stack(xyz_list, dim=0).view(-1, 3)
        quat = torch.stack(quat_list, dim=0).view(-1, 4)
        score = torch.stack(score_list, dim=0).view(-1, 1)

        out[inst_id] = {"xyz": xyz, "quat_wxyz": quat, "score": score}
        if include_width:
            out[inst_id]["width"] = torch.stack(width_list, dim=0).view(-1, 1)

    return out

    

def _pack_targets_cpu(sample: dict):
    # poses
    poses_in = sample.get("poses") or {}
    poses = {}
    for inst_id, p in poses_in.items():
        if p is None or "R" not in p or "t" not in p:
            continue
        poses[int(inst_id)] = {
            "obj_id": str(p.get("obj_id", "")),
            "R": (p["R"] if isinstance(p["R"], torch.Tensor)
                  else torch.as_tensor(p["R"], dtype=torch.float32)),
            "t": (p["t"] if isinstance(p["t"], torch.Tensor)
                  else torch.as_tensor(p["t"], dtype=torch.float32)),
        }

    # models
    models_in = sample.get("models") or {}
    models = {}
    for inst_id, m in models_in.items():
        if m is None:
            continue
        models[int(inst_id)] = (m if isinstance(m, torch.Tensor)
                                else torch.as_tensor(m, dtype=torch.float32))        

    gt = _pack_grasp_dict(sample.get("gt_grasps"),  default_score=1.0) 
    bad = _pack_grasp_dict(sample.get("bad_grasps"), default_score=0.0)  
    return poses, models, gt, bad


def collate_mixed_batch(samples):
    out = []
    for s in samples:
        item = {}
        item['img'] = s["img"]
        item['mask'] = s["mask"]
        item['depth'] = s["depth"] if s["depth"] is not None else None
        item["boxes"] = s["boxes_cxcywh"].float() if isinstance(s["boxes_cxcywh"], torch.Tensor) else torch.as_tensor(s["boxes_cxcywh"]).float()
        item['box2inst'] = s['box2inst']
        item["intr"] = s["intr"] if s["intr"] is not None else None

        item["image_id"] = s.get("image_id", f"sample_{len(out)}")
        item["H"] = int(s.get("H", 490))
        item["W"] = int(s.get("W", 644))

        # Instance ids for indexing
        ids = torch.unique(item['mask'])
        ids = ids[(ids > 0)]
        inst_ids_from_mask = sorted([int(i.item()) for i in ids])
        if s.get('with_det', False):
            inst_crops, inst_ids = _inst_crops_from_mask(item["img"], item["mask"], size=224)
            item["inst_crops_images"] = inst_crops           
            item["inst_crops_inst_ids"] = inst_ids

        N = item["boxes"].shape[0]

        # Baked models come in s["models"]
        baked = s.get("props", {}) or {}
        prop_valid = torch.zeros(N, dtype=torch.bool)
        prop_pos   = torch.zeros(N, 3, dtype=torch.float32)
        prop_scale = torch.zeros(N, 1, dtype=torch.float32)
        prop_vis   = torch.zeros(N, 1, dtype=torch.float32)
        prop_vis_mask   = [torch.zeros(0, 1, dtype=torch.bool) for _ in range(N)]
        prop_pts   = [torch.zeros(0, 3, dtype=torch.float32) for _ in range(N)]
        prop_sphere = [torch.zeros(0, 3, dtype=torch.float32) for _ in range(N)]
        prop_delta  = [torch.zeros(0, 3, dtype=torch.float32) for _ in range(N)]
        prop_vol   = torch.zeros(N, 1, dtype=torch.float32)

        for box_idx, inst_id in enumerate(inst_ids_from_mask):
            vm = baked.get(int(inst_id), None)
            if vm is None:
                continue
                
            prop_valid[box_idx] = True
            prop_pos[box_idx]   = vm["pos_m"] 
            prop_vis[box_idx, 0]   = vm["vis"] 
            prop_vis_mask[box_idx] = vm["vis_mask"] 
            prop_scale[box_idx, 0] = float(vm["scale_m"])
            prop_pts[box_idx]    = vm["pc_norm_cam"].float()
            prop_vol[box_idx, 0] = float(vm["vol"])

        # Expose as 'props' (no 'viz_' prefix)
        item["props"] = {
            "valid": prop_valid,
            "pos": prop_pos,           # [N,3] meters
            "vis": prop_vis,           # [N,1]
            "vis_mask": prop_vis_mask, # list of [Pi,1] bool
            "scale": prop_scale,       # [N,1] meters
            "points_list": prop_pts,   # list of [Pi,3]
            "points_sphere": prop_sphere,   # unit shere for targets
            "points_delta": prop_delta,      # deltas from the unit shere to the real pts 
            "vol": prop_vol,           # normalized/percent as you define
        }


        # adding pose (obj and grasp) data
        item["pose_6d_proposals_good"] = s.get("pose_6d_proposals_good", {}) # allready packed in torch
        item["pose_6d_proposals_bad"] = s.get("pose_6d_proposals_bad", {})  # allready packed in torch
        
        item["gt_grasps"] = s.get("gt_grasps", {})
        item["bad_grasps"] = s.get("bad_grasps", {})
        item["poses"] = s.get("poses", {})
        item["models"] = s.get("models", {})
        poses_p, models_p, gt_p, bad_p = _pack_targets_cpu(item)
        item["poses"] = poses_p
        item["models"] = models_p
        item["gt_grasps"] = gt_p
        item["bad_grasps"] = bad_p

        out.append(item)
    return out


@torch.no_grad()
def reduce_dict(input_dict, average=True):
    """
    Reduce the dict across processes. Returns a dict with the same fields.
    """
    if not dist.is_available() or not dist.is_initialized():
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack([v for v in values], dim=0)
        dist.all_reduce(values)
        if average:
            values /= dist.get_world_size()
        reduced = {k: v for k, v in zip(sorted(input_dict.keys()), values)}
        return reduced