#!/usr/bin/env python3

# convert_ruapc_to_bop_split_subscenes.py

import os
import re
import json
import glob
import shutil
import argparse
from typing import Dict, List, Tuple, Optional
import yaml
import numpy as np
from PIL import Image

def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def list_scenes(src_root: str) -> List[str]:
    return sorted([d for d in glob.glob(os.path.join(src_root, 'scene_*')) if os.path.isdir(d)], key=natural_key)

def camera_for_scene(scene_name: str) -> Dict[str, float]:
    # Scenes 1–5 use camera A; Scene 6 uses camera B
    if re.search(r'scene_6\b', scene_name):
        cfg = dict(width=640, height=480, fx=531.15, fy=531.15, cx=320.0, cy=240.0, depth_scale=1.0)
    else:
        cfg = dict(width=640, height=480, fx=572.41140, fy=573.57043, cx=325.26110, cy=242.04899, depth_scale=1.0)
    return cfg

def camK_from_cfg(cfg: Dict[str, float]) -> List[float]:
    fx, fy, cx, cy = cfg['fx'], cfg['fy'], cfg['cx'], cfg['cy']
    # Row-major flattened intrinsic
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

def read_yaml_items(yml_path: str) -> List[Dict]:
    with open(yml_path, 'r') as f:
        data = yaml.safe_load(f)
    # Accept formats: dict with numeric key (e.g., "0") -> list, or list directly
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                return v
        return []
    elif isinstance(data, list):
        return data
    else:
        return []

def copy_or_link(src: str, dst: str, mode: str):
    ensure_dir(os.path.dirname(dst))
    if mode == 'link':
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    elif mode == 'symlink':
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)

def make_zero_mask(dst_path: str, width: int, height: int):
    ensure_dir(os.path.dirname(dst_path))
    img = Image.fromarray(np.zeros((height, width), dtype=np.uint8))
    img.save(dst_path)

def find_frame_files(mod_dir: str, arr_id: str, frame_stem: str, exts=('png', 'jpg')) -> Optional[str]:
    for ext in exts:
        cand = os.path.join(mod_dir, '000', arr_id, f"{frame_stem}.{ext}")
        if os.path.exists(cand):
            return cand
    return None

def collect_instance_masks(mod_dir: str, arr_id: str, frame_stem: str) -> List[str]:
    pattern = os.path.join(mod_dir, '000', arr_id, f"{frame_stem}*.png")
    files = glob.glob(pattern)
    return sorted(files, key=natural_key)

def convert_scene_into_subscenes(src_scene_dir: str, out_split_dir: str, split: str, copy_mode: str = 'copy'):
    # Layout: <src>/scene_#/ {rgb,depth,mask,mask_visib,gt}/000/<arr_id>/<frame_id>.<ext>
    rgb_dir = os.path.join(src_scene_dir, 'rgb')
    depth_dir = os.path.join(src_scene_dir, 'depth')
    mask_dir = os.path.join(src_scene_dir, 'mask')
    mask_vis_dir = os.path.join(src_scene_dir, 'mask_visib')
    gt_dir = os.path.join(src_scene_dir, 'gt')

    # Validate presence
    for d in [rgb_dir, depth_dir, gt_dir]:
        if not os.path.isdir(d):
            print(f"[WARN] Missing modality in {src_scene_dir}: {d}")

    # Build camera config (per scene, reused for all sub-scenes)
    scene_name = os.path.basename(src_scene_dir)
    cam_cfg = camera_for_scene(scene_name)
    camK = camK_from_cfg(cam_cfg)
    width, height = int(cam_cfg['width']), int(cam_cfg['height'])
    depth_scale = float(cam_cfg['depth_scale'])

    # Iterate arrangement IDs based on gt dir
    arr_root = os.path.join(gt_dir, '000')
    if not os.path.isdir(arr_root):
        print(f"[WARN] No gt/000 directory in {src_scene_dir}, skipping scene.")
        return

    arr_ids = sorted([d for d in os.listdir(arr_root) if os.path.isdir(os.path.join(arr_root, d))], key=natural_key)

    if not arr_ids:
        print(f"[WARN] No arrangement folders in {arr_root}, skipping scene.")
        return

    for arr_id in arr_ids:
        yml_dir = os.path.join(arr_root, arr_id)
        yml_files = sorted(glob.glob(os.path.join(yml_dir, '*.yml')), key=natural_key)
        if len(yml_files) == 0:
            print(f"[WARN] No YAML in {yml_dir}, skipping sub-scene.")
            continue

        # Create output sub-scene directory: scene_<sceneid>_<subid>
        out_scene_name = f"{scene_name}_{arr_id}"
        out_scene_dir = os.path.join(out_split_dir, out_scene_name)
        ensure_dir(out_scene_dir)

        out_rgb = os.path.join(out_scene_dir, 'rgb')
        out_depth = os.path.join(out_scene_dir, 'depth')
        out_mask = os.path.join(out_scene_dir, 'mask')
        out_mask_vis = os.path.join(out_scene_dir, 'mask_visib')
        for d in [out_rgb, out_depth, out_mask, out_mask_vis]:
            ensure_dir(d)

        scene_camera = {}
        scene_gt = {}
        scene_gt_info = {}

        new_im_counter = 0

        for yml_path in yml_files:
            frame_stem = os.path.splitext(os.path.basename(yml_path))[0]  # original frame id (possibly 9-digit)
            objs = read_yaml_items(yml_path)

            # new image id (6-digit) within this sub-scene
            im_id = new_im_counter
            new_im_counter += 1
            im_key = f"{im_id}"

            # Copy/link RGB
            rgb_src = find_frame_files(rgb_dir, arr_id, frame_stem, exts=('png', 'jpg', 'jpeg'))
            if rgb_src is None:
                print(f"[WARN] Missing RGB for {frame_stem} in {src_scene_dir} (arr {arr_id})")
                continue
            rgb_dst = os.path.join(out_rgb, f"{im_id:06d}.png")
            copy_or_link(rgb_src, rgb_dst, copy_mode)

            # Copy/link depth
            depth_src = find_frame_files(depth_dir, arr_id, frame_stem, exts=('png',))
            if depth_src is None:
                print(f"[WARN] Missing depth for {frame_stem} in {src_scene_dir} (arr {arr_id})")
            else:
                depth_dst = os.path.join(out_depth, f"{im_id:06d}.png")
                copy_or_link(depth_src, depth_dst, copy_mode)

            # Collect masks and rename to BOP pattern
            masks = collect_instance_masks(mask_dir, arr_id, frame_stem) if os.path.isdir(mask_dir) else []
            masks_vis = collect_instance_masks(mask_vis_dir, arr_id, frame_stem) if os.path.isdir(mask_vis_dir) else []

            # Write camera entry
            scene_camera[im_key] = {
                'cam_K': camK,
                'depth_scale': depth_scale,
                'height': height,
                'width': width
            }

            # Write GT entries
            gt_items = []
            gt_info_items = []

            for inst_idx, od in enumerate(objs):
                # Extract pose
                R = od.get('cam_R_m2c', [])
                t = od.get('cam_t_m2c', [])
                obj_id = int(od.get('obj_id', -1))
                gt_items.append({
                    'obj_id': obj_id,
                    'cam_R_m2c': [float(x) for x in R],
                    'cam_t_m2c': [float(x) for x in t]
                })
                # Info (default values if missing)
                info = {
                    'obj_bb': [int(x) for x in od.get('obj_bb', [-1, -1, -1, -1])],
                    'obj_bb_visib': [int(x) for x in od.get('obj_bb_visib', [-1, -1, -1, -1])],
                    'px_count_all': int(od.get('px_count_all', 0)),
                    'px_count_valid': int(od.get('px_count_valid', 0)),
                    'px_count_visib': int(od.get('px_count_visib', 0)),
                    'visib_fract': float(od.get('visib_fract', 0.0)),
                }
                gt_info_items.append(info)

                # Map masks to inst index
                mask_dst = os.path.join(out_mask, f"{im_id:06d}_{inst_idx:06d}.png")
                mask_vis_dst = os.path.join(out_mask_vis, f"{im_id:06d}_{inst_idx:06d}.png")

                mask_src = masks[inst_idx] if inst_idx < len(masks) else None
                mask_vis_src = masks_vis[inst_idx] if inst_idx < len(masks_vis) else None

                if mask_src and os.path.exists(mask_src):
                    copy_or_link(mask_src, mask_dst, copy_mode)
                else:
                    make_zero_mask(mask_dst, width=width, height=height)

                if mask_vis_src and os.path.exists(mask_vis_src):
                    copy_or_link(mask_vis_src, mask_vis_dst, copy_mode)
                else:
                    make_zero_mask(mask_vis_dst, width=width, height=height)

            scene_gt[im_key] = gt_items
            scene_gt_info[im_key] = gt_info_items

        # Write JSONs for this sub-scene
        with open(os.path.join(out_scene_dir, 'scene_camera.json'), 'w') as f:
            json.dump(scene_camera, f, indent=2)
        with open(os.path.join(out_scene_dir, 'scene_gt.json'), 'w') as f:
            json.dump(scene_gt, f, indent=2)
        with open(os.path.join(out_scene_dir, 'scene_gt_info.json'), 'w') as f:
            json.dump(scene_gt_info, f, indent=2)

        print(f"[OK] Converted {src_scene_dir} -> {out_scene_dir} | frames: {len(scene_gt)}")

def main():
    ap = argparse.ArgumentParser(description="Convert RUAPC dataset to BOP format, split into sub-scenes per arrangement.")
    ap.add_argument('--src_root', default='/mnt/data/publicdata/bop/ruapc/scenes', help='Path to RUAPC root (contains scene_1..scene_6)')
    ap.add_argument('--out_root', default='/mnt/data/publicdata/bop/ruapc', help='Destination BOP root (e.g., /mnt/data/publicdata/bop/ruapc)')
    ap.add_argument('--split', default='train_synt', help='BOP split folder (train|val|test). Default: train_synt')
    ap.add_argument('--copy_mode', default='copy', choices=['copy', 'link', 'symlink'], help='copy/link/symlink files')
    args = ap.parse_args()

    scenes = list_scenes(args.src_root)
    if not scenes:
        print(f"[ERROR] No scenes found at {args.src_root}")
        return

    out_split_dir = os.path.join(args.out_root, args.split)
    ensure_dir(out_split_dir)

    for src_scene in scenes:
        print('converting', src_scene)
        convert_scene_into_subscenes(src_scene, out_split_dir, args.split, copy_mode=args.copy_mode)

    print("[DONE] RUAPC conversion complete.")

if __name__ == '__main__':
    main()
    
    
    