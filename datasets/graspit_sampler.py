
# datasets_mixed.py

import os
import re
import io
import cv2
import sys
import json
import time
import glob
import yaml
import math
import uuid
import copy
import random
import shutil
import struct
import logging
import traceback
import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

# -----------------------

# Globals and utilities

# -----------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OUT_H, OUT_W = 490, 644
PC_NUM_POINTS = 768

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def read_json(path: str, default=None):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def to_tensor_image(img_pil: Image.Image) -> torch.Tensor:
    # Convert PIL to float tensor in [0,1]
    return TF.to_tensor(img_pil)  # C,H,W float [0,1]

def normalize_rgb(img_t: torch.Tensor) -> torch.Tensor:
    return TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

def gaussian_noise(t: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(t) * sigma
    return t + noise

def gaussian_noise_depth(depth_m: torch.Tensor, sigma_m: float = 0.003) -> torch.Tensor:
    if depth_m is None:
        return None
    noise = torch.randn_like(depth_m) * sigma_m
    d = depth_m + noise
    d = torch.clamp(d, min=0.0)
    return d

def solarize_pil(img: Image.Image, threshold: int = 128) -> Image.Image:
    arr = np.array(img)
    inv = np.where(arr < threshold, arr, 255 - arr)
    return Image.fromarray(inv.astype(np.uint8))

def channel_swap(img_t: torch.Tensor) -> torch.Tensor:
    # Randomly permute channels
    C = img_t.shape[0]
    perm = torch.randperm(C)
    return img_t[perm, :, :]

def get_color_palette(n: int) -> List[Tuple[int,int,int]]:
    # Generate n distinct colors
    rng = np.random.RandomState(123)
    colors = []
    for i in range(n):
        c = rng.randint(0, 255, size=3).tolist()
        colors.append(tuple(c))
    return colors

def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
    # q is [w, x, y, z]
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X; wY = w * Y; wZ = w * Z
    xX = x * X; xY = x * Y; xZ = x * Z
    yY = y * Y; yZ = y * Z; zZ = z * Z
    R = np.array([
        [1.0 - (yY + zZ), xY - wZ, xZ + wY],
        [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
        [xZ - wY, yZ + wX, 1.0 - (xX + yY)]
    ], dtype=np.float32)
    return R

def random_rotation_matrix(max_angle_deg: float = 30.0) -> np.ndarray:
    # Uniform random rotation about random axis by angle in [-max,max]
    angle = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
    ], dtype=np.float32)
    return R

def bbox_from_mask(inst_mask: np.ndarray, inst_id: int) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(inst_mask == inst_id)
    if ys.size == 0:
        return None
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    return int(x1), int(y1), int(x2), int(y2)

def xyxy_to_cxcywh_norm(x1, y1, x2, y2, W, H):
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / W, cy / H, w / W, h / H

def project_points(points_m: np.ndarray, R: np.ndarray, t_m: np.ndarray, intr: List[float]) -> np.ndarray:
    # points_m: Nx3 obj coords in meters
    # R: 3x3, t_m: (3,)
    # intr: [cx, cy, fx, fy]
    cx, cy, fx, fy = intr
    pts = (R @ points_m.T).T + t_m.reshape(1,3)
    Z = pts[:,2:3]
    valid = Z[:,0] > 1e-6
    u = fx * (pts[:,0]/Z[:,0]) + cx
    v = fy * (pts[:,1]/Z[:,0]) + cy
    uv = np.stack([u, v, valid], axis=1)
    return uv  # N x 3 (u,v,valid)

def resize_coherent(img_pil: Image.Image, depth_np: Optional[np.ndarray], mask_np: Optional[np.ndarray], out_hw=(OUT_H, OUT_W)):
    # Resize to target size with bilinear for rgb, nearest for depth and mask
    out_h, out_w = out_hw
    img_res = img_pil.resize((out_w, out_h), Image.BILINEAR)
    depth_res = None
    mask_res = None
    if depth_np is not None:
        depth_pil = Image.fromarray(depth_np)
        depth_res = np.array(depth_pil.resize((out_w, out_h), Image.NEAREST), dtype=depth_np.dtype)
    if mask_np is not None:
        mask_pil = Image.fromarray(mask_np.astype(np.uint8))
        mask_res = np.array(mask_pil.resize((out_w, out_h), Image.NEAREST), dtype=np.uint8)
    return img_res, depth_res, mask_res

def random_affine_coherent(img_pil: Image.Image, depth_np: Optional[np.ndarray], mask_np: Optional[np.ndarray],
                           degrees=5, translate=(0.02,0.02), scale_ranges=(0.98,1.02), shear=(-2,2,-2,2)):
    # Generate same params for img/depth/mask; use PIL for image, and nearest for depth/mask
    angle = random.uniform(-degrees, degrees)
    max_dx = translate[0] * img_pil.size[0]
    max_dy = translate[1] * img_pil.size[1]
    translations = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
    scale = random.uniform(*scale_ranges)
    shear_x = random.uniform(shear[0], shear[1])
    shear_y = random.uniform(shear[2], shear[3])

    img_aff = TF.affine(img_pil, angle=angle, translate=(int(translations[0]), int(translations[1])),
                        scale=scale, shear=[shear_x, shear_y], interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    depth_aff = None
    mask_aff = None
    if depth_np is not None:
        d_pil = Image.fromarray(depth_np)
        depth_aff = np.array(
            TF.affine(d_pil, angle=angle, translate=(int(translations[0]), int(translations[1])),
                      scale=scale, shear=[shear_x, shear_y], interpolation=TF.InterpolationMode.NEAREST, fill=0)
        , dtype=depth_np.dtype)
    if mask_np is not None:
        m_pil = Image.fromarray(mask_np.astype(np.uint8))
        mask_aff = np.array(
            TF.affine(m_pil, angle=angle, translate=(int(translations[0]), int(translations[1])),
                      scale=scale, shear=[shear_x, shear_y], interpolation=TF.InterpolationMode.NEAREST, fill=0)
        , dtype=np.uint8)
    return img_aff, depth_aff, mask_aff

def build_detr_boxes_from_mask(mask_np: np.ndarray, out_w: int, out_h: int):
    ids = sorted([int(i) for i in np.unique(mask_np) if i != 0])
    boxes_xyxy = []
    boxes_cxcywh = []
    for inst_id in ids:
        bb = bbox_from_mask(mask_np, inst_id)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        boxes_xyxy.append([x1, y1, x2, y2])
        cx, cy, w, h = xyxy_to_cxcywh_norm(x1, y1, x2, y2, out_w, out_h)
        boxes_cxcywh.append([cx, cy, w, h])
    if len(boxes_xyxy) == 0:
        boxes_xyxy = np.zeros((0,4), dtype=np.float32)
        boxes_cxcywh = np.zeros((0,4), dtype=np.float32)
    else:
        boxes_xyxy = np.array(boxes_xyxy, dtype=np.float32)
        boxes_cxcywh = np.array(boxes_cxcywh, dtype=np.float32)
    return boxes_xyxy, boxes_cxcywh

def ensure_downsampled_mesh_to_pc(mesh_path: str, out_pc_path: str, num_points: int = PC_NUM_POINTS) -> Optional[str]:
    try:
        if os.path.exists(out_pc_path):
            return out_pc_path
        ensure_dir(os.path.dirname(out_pc_path))
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh is None or (not mesh.has_triangles() and not mesh.has_vertices()):
            return None
        mesh.compute_vertex_normals()
        pts = None
        if mesh.has_vertices() and not mesh.has_triangles():
            # already point cloud-like?
            pts = np.asarray(mesh.vertices)
            if pts.shape[0] > num_points:
                sel = np.random.choice(pts.shape[0], num_points, replace=False)
                pts = pts[sel]
        else:
            # sample points uniformly from mesh surface
            pcd = mesh.sample_points_uniformly(number_of_points=max(num_points * 2, num_points))
            pts = np.asarray(pcd.points)
            # downsample to exact num_points
            if pts.shape[0] > num_points:
                sel = np.random.choice(pts.shape[0], num_points, replace=False)
                pts = pts[sel]
        pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.io.write_point_cloud(out_pc_path, pcd_out, write_ascii=True)
        return out_pc_path
    except Exception as e:
        print(f"[ensure_downsampled_mesh_to_pc] Failed for {mesh_path}: {e}")
        return None

def ensure_downsampled_model_dir(src_dir: str, out_dir: str, pattern_mesh=('*.ply','*.obj','*.stl'), num_points: int = PC_NUM_POINTS):
    ensure_dir(out_dir)
    meshes = []
    for pat in pattern_mesh:
        meshes.extend(glob.glob(os.path.join(src_dir, pat)))
    for mp in meshes:
        base = os.path.splitext(os.path.basename(mp))[0]
        out_pc = os.path.join(out_dir, f"{base}_pc{num_points}.ply")
        ensure_downsampled_mesh_to_pc(mp, out_pc, num_points=num_points)

def normalize_pointcloud(pc_xyz: np.ndarray):
    # Center and scale by max extent to fit in unit cube; returns pc_normed, center, scale
    if pc_xyz.shape[0] == 0:
        return pc_xyz, np.zeros(3, dtype=np.float32), 1.0
    center = pc_xyz.mean(axis=0)
    pc_centered = pc_xyz - center
    mins = pc_centered.min(axis=0)
    maxs = pc_centered.max(axis=0)
    extents = (maxs - mins).astype(np.float32)
    scale = float(np.max(extents)) if float(np.max(extents)) > 1e-9 else 1.0
    pc_normed = pc_centered / (scale + 1e-12)
    return pc_normed.astype(np.float32), center.astype(np.float32), scale

def load_pointcloud_file(pc_path: str) -> Optional[np.ndarray]:
    try:
        pcd = o3d.io.read_point_cloud(pc_path)
        if pcd is None:
            return None
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.shape[0] > PC_NUM_POINTS:
            sel = np.random.choice(pts.shape[0], PC_NUM_POINTS, replace=False)
            pts = pts[sel]
        return pts
    except Exception as e:
        print(f"[load_pointcloud_file] Failed for {pc_path}: {e}")
        return None

def build_dummy_sphere_pc(out_dir: str, name: str, num_points: int = PC_NUM_POINTS) -> str:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{name}_pc{num_points}.ply")
    if os.path.exists(out_path):
        return out_path
    # sample unit sphere
    phi = np.random.uniform(0, 2*np.pi, size=(num_points,))
    costheta = np.random.uniform(-1, 1, size=(num_points,))
    theta = np.arccos(costheta)
    r = 1.0
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pts = np.stack([x,y,z], axis=1).astype(np.float32)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
    return out_path

def colorize_mask(mask: np.ndarray, palette: Optional[List[Tuple[int,int,int]]] = None) -> Image.Image:
    ids = sorted([i for i in np.unique(mask) if i != 0])
    if palette is None:
        palette = get_color_palette(len(ids))
    color_map = {inst_id: palette[i] for i, inst_id in enumerate(ids)}
    H, W = mask.shape
    rgb = np.zeros((H,W,3), dtype=np.uint8)
    for inst_id, col in color_map.items():
        m = (mask == inst_id)
        rgb[m] = col
    return Image.fromarray(rgb)

def draw_boxes_on_rgb(img: Image.Image, boxes_xyxy: np.ndarray, palette: Optional[List[Tuple[int,int,int]]] = None) -> Image.Image:
    import PIL.ImageDraw as ImageDraw
    out = img.copy()
    draw = ImageDraw.Draw(out)
    n = boxes_xyxy.shape[0]
    if palette is None:
        palette = get_color_palette(n)
    for i in range(n):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        col = palette[i % len(palette)]
        draw.rectangle([(x1,y1),(x2,y2)], outline=tuple(col), width=2)
    return out

def overlay_points_on_rgb(img: Image.Image, uv_valid: np.ndarray, color=(0,255,0)) -> Image.Image:
    out = np.array(img).copy()
    H, W = out.shape[0], out.shape[1]
    for i in range(uv_valid.shape[0]):
        u, v, valid = uv_valid[i]
        if not valid: 
            continue
        u_i = int(round(u))
        v_i = int(round(v))
        if 0 <= u_i < W and 0 <= v_i < H:
            out[v_i, u_i] = color
    return Image.fromarray(out)

def depth_to_colormap(depth_m: np.ndarray, vmax: Optional[float] = None) -> Image.Image:
    d = depth_m.copy()
    if vmax is None:
        vmax = np.percentile(d[d>0], 95) if np.any(d>0) else 1.0
    d = (np.clip(d, 0, vmax) / vmax) * 255.0
    d = d.astype(np.uint8)
    cm = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    return Image.fromarray(cm[..., ::-1])  # BGR->RGB

# -----------------------

# Intrinsics helpers

# -----------------------

def intr_from_graspit_camera_params(cam: Dict[str, Any]) -> Optional[List[float]]:
    # From USD-like fields: cameraAperture [W_mm, H_mm], cameraFocalLength (mm), renderProductResolution [W,H]
    # fx_pixels = (focal_mm / aperture_w_mm) * image_width_px
    # fy_pixels = (focal_mm / aperture_h_mm) * image_height_px
    try:
        apx, apy = cam['cameraAperture']
        f_mm = cam['cameraFocalLength']
        W, H = cam['renderProductResolution']
        fx = (f_mm / apx) * W
        fy = (f_mm / apy) * H
        cx = W / 2.0
        cy = H / 2.0
        return [float(cx), float(cy), float(fx), float(fy)]
    except Exception:
        return None

# -----------------------

# Base Dataset

# -----------------------

class BasePoseDataset(data.Dataset):
    def __init__(self, mode='train', transforms=True, rgb_noise_sigma=0.02, depth_noise_sigma_m=0.003,
                 do_affine=True, do_color=True, do_pose_rot=True):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.apply_transforms = transforms
        self.rgb_noise_sigma = rgb_noise_sigma
        self.depth_noise_sigma_m = depth_noise_sigma_m
        self.do_affine = do_affine
        self.do_color = do_color
        self.do_pose_rot = do_pose_rot

    def _apply_augs(self, img_pil, depth_np, mask_np, intr, poses_list):
        # 1) Resize
        img_pil, depth_np, mask_np = resize_coherent(img_pil, depth_np, mask_np, out_hw=(OUT_H, OUT_W))

        # 2) Color transforms on RGB only (train mode typically)
        if self.mode == 'train' and self.apply_transforms and self.do_color:
            if random.random() < 0.8:
                cj = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
                img_pil = cj(img_pil)
            if random.random() < 0.2:
                img_pil = ImageOps.grayscale(img_pil).convert('RGB')
            if random.random() < 0.1:
                img_pil = solarize_pil(img_pil, threshold=random.randint(64,192))

        # 3) Random channel swap
        img_t = to_tensor_image(img_pil)
        if self.mode == 'train' and self.apply_transforms and random.random() < 0.1:
            img_t = channel_swap(img_t)
            img_pil = TF.to_pil_image(img_t)

        # 4) Random affine (coherent)
        if self.mode == 'train' and self.apply_transforms and self.do_affine:
            img_pil, depth_np, mask_np = random_affine_coherent(img_pil, depth_np, mask_np,
                                                                degrees=5, translate=(0.02,0.02),
                                                                scale_ranges=(0.98,1.02), shear=(-2,2,-2,2))
        # 5) Convert rgb and normalize
        img_t = to_tensor_image(img_pil)
        img_t = normalize_rgb(img_t)

        # 6) Gaussian noise
        if self.mode == 'train' and self.apply_transforms and self.rgb_noise_sigma > 0.0:
            img_t = gaussian_noise(img_t, sigma=self.rgb_noise_sigma)

        depth_t = None
        if depth_np is not None:
            depth_t = torch.from_numpy(depth_np.astype(np.float32))
            depth_t = depth_t.unsqueeze(0)
            if self.mode == 'train' and self.apply_transforms and self.depth_noise_sigma_m > 0.0:
                depth_t = gaussian_noise_depth(depth_t, sigma_m=self.depth_noise_sigma_m)

        mask_t = None
        if mask_np is not None:
            mask_t = torch.from_numpy(mask_np.astype(np.int64))

        # 7) Pose augmentation: rotate R but keep t, store xyz
        if self.mode == 'train' and self.apply_transforms and self.do_pose_rot and poses_list is not None:
            new_poses = []
            Rrand = random_rotation_matrix(15.0)
            for p in poses_list:
                R = p['R']
                t = p['t']
                R_aug = (Rrand @ R).astype(np.float32)
                new_poses.append({'R': R_aug, 't': t.astype(np.float32), 'obj_id': p.get('obj_id'), 'xyz': t.astype(np.float32)})
            poses_list = new_poses
        else:
            if poses_list is not None:
                for p in poses_list:
                    p['xyz'] = p['t'].astype(np.float32)

        return img_t, depth_t, mask_t, intr, poses_list

    def _build_targets(self, mask_np):
        if mask_np is None:
            return np.zeros((0,4), dtype=np.float32), np.zeros((0,4), dtype=np.float32)
        boxes_xyxy, boxes_cxcywh = build_detr_boxes_from_mask(mask_np, OUT_W, OUT_H)
        return boxes_xyxy, boxes_cxcywh

    def _norm_model_pc(self, pc_xyz: np.ndarray):
        pc_normed, center, scale = normalize_pointcloud(pc_xyz)
        return pc_normed, center, scale

# -----------------------

# GraspIt Dataset

# -----------------------

class GraspItDataset(BasePoseDataset):
    def __init__(self, root: str,
                 assets_root: Optional[str] = None,
                 mode='train',
                 transforms=True,
                 depth_scale_m_per_unit=0.001,
                 invert_yaml_world2obj=True,
                 include_mask=True,
                 include_depth=True,
                 include_models=True):
        super().__init__(mode=mode, transforms=transforms)
        self.name = 'graspit'
        self.root = root
        self.assets_root = assets_root if assets_root is not None else os.path.join(root, 'assets')
        self.depth_scale_m_per_unit = depth_scale_m_per_unit
        self.invert_yaml_world2obj = invert_yaml_world2obj
        self.include_mask = include_mask
        self.include_depth = include_depth
        self.include_models = include_models

        # discover scenes
        # Support either scenes or scenes_* directories
        scene_dirs = []
        if os.path.exists(os.path.join(root, 'scenes')):
            scene_dirs.append(os.path.join(root, 'scenes'))
        scene_dirs.extend(glob.glob(os.path.join(root, 'scenes_*')))

        self.frames = []  # list of dict: {scene_dir, scene_yaml, frame_dir, frame_idx}
        for sdir in sorted(scene_dirs):
            for scene in sorted(os.listdir(sdir)):
                scene_path = os.path.join(sdir, scene)
                if not os.path.isdir(scene_path): 
                    continue
                yaml_path = os.path.join(scene_path, 'scene.yaml')
                if not os.path.exists(yaml_path):
                    continue
                frames = sorted([f for f in os.listdir(scene_path) if f.startswith('frame_')])
                for fr in frames:
                    frame_dir = os.path.join(scene_path, fr)
                    idx = None
                    # expect rgb_0000.png existence
                    pngs = glob.glob(os.path.join(frame_dir, 'rgb_*.png'))
                    if len(pngs) > 0:
                        b = os.path.basename(pngs[0])
                        idx = os.path.splitext(b)[0].split('_')[-1]
                    else:
                        # fallback scan camera_params_*.json
                        cams = glob.glob(os.path.join(frame_dir, 'camera_params_*.json'))
                        if len(cams) > 0:
                            b = os.path.basename(cams[0])
                            idx = os.path.splitext(b)[0].split('_')[-1]
                    if idx is None:
                        continue
                    self.frames.append({
                        'scene_dir': scene_path,
                        'scene_yaml': yaml_path,
                        'frame_dir': frame_dir,
                        'frame_idx': idx
                    })

        # map obj path to obj_id by filename folder
        self.assets_pc_dir = os.path.join(self.assets_root + f"_pc_{PC_NUM_POINTS}")
        ensure_dir(self.assets_pc_dir)

    def __len__(self):
        return len(self.frames)

    def _load_scene_yaml(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            sc = yaml.safe_load(f)
        # objects keys: object0, object1, ... with file_path, name, orientation, position, scale
        objs = {}
        for k,v in sc.items():
            if not isinstance(v, dict): 
                continue
            if 'file_path' in v and 'position' in v and 'orientation' in v:
                fp = v['file_path']
                # file_path might be like /share/assets/106619/106619.usd
                m = re.search(r'/assets/(\d+)/', fp)
                obj_id = None
                if m:
                    obj_id = m.group(1)
                else:
                    # fallback from name
                    n = v.get('name', '')
                    m2 = re.search(r'(\d+)', n)
                    if m2:
                        obj_id = m2.group(1)
                if obj_id is None:
                    continue
                # read orientation [w,x,y,z], position meters
                q = v['orientation']
                t = v['position']
                s = v.get('scale', [1,1,1])
                objs[obj_id] = {
                    'quat_wxyz': q,
                    't': t,
                    'scale': s,
                    'file_path': fp
                }
        sc['_objects'] = objs
        return sc

    def _poses_from_yaml(self, sc_obj_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        poses = []
        for obj_id, od in sc_obj_dict.items():
            q = od['quat_wxyz']
            t = np.array(od['t'], dtype=np.float32)  # assume meters
            R = quaternion_to_rotation_matrix(q)
            # YAML comment says world2obj; if so, invert to get obj-in-world
            if self.invert_yaml_world2obj:
                R = R.T
                t = - (R @ t)
            poses.append({'obj_id': obj_id, 'R': R.astype(np.float32), 't': t.astype(np.float32)})
        return poses

    def _load_assets_pc(self, obj_id: str) -> Optional[np.ndarray]:
        # expect original mesh at assets/<obj_id>/<obj_id>.obj (or others)
        src_obj_dir = os.path.join(self.assets_root, obj_id)
        if not os.path.isdir(src_obj_dir):
            return None
        # pick first *.obj or ply/stl
        cand = None
        for pat in ('*.obj','*.ply','*.stl'):
            g = glob.glob(os.path.join(src_obj_dir, pat))
            if g:
                cand = g[0]; break
        if cand is None:
            return None
        out_dir = os.path.join(self.assets_root + f"_pc_{PC_NUM_POINTS}", obj_id)
        ensure_dir(out_dir)
        out_pc = os.path.join(out_dir, f"model_pc{PC_NUM_POINTS}.ply")
        out_pc_path = ensure_downsampled_mesh_to_pc(cand, out_pc, num_points=PC_NUM_POINTS)
        if out_pc_path is None:
            return None
        pts = load_pointcloud_file(out_pc_path)
        return pts

    def __getitem__(self, index: int):
        meta = self.frames[index]
        scene_yaml = self._load_scene_yaml(meta['scene_yaml'])
        frame_dir = meta['frame_dir']
        frame_idx = meta['frame_idx']

        # RGB
        rgb_path = os.path.join(frame_dir, f"rgb_{frame_idx}.png")
        img = Image.open(rgb_path).convert('RGB')

        # Depth
        depth_np = None
        if self.include_depth:
            dpth_path = os.path.join(frame_dir, "depth.png")
            depth_raw = np.array(Image.open(dpth_path), dtype=np.float32)
            depth_np = depth_raw * float(self.depth_scale_m_per_unit)  # meters

        # Mask -> instance segmentation
        mask_np = None
        if self.include_mask:
            mask_info_path = os.path.join(frame_dir, f"semantic_segmentation_labels_{frame_idx}.json")
            mask_img_path = os.path.join(frame_dir, f"semantic_segmentation_{frame_idx}.png")
            mask_info = read_json(mask_info_path, default={})
            mask_rgb = np.array(Image.open(mask_img_path).convert('RGB'))
            H, W, _ = mask_rgb.shape
            mask_np = np.zeros((H,W), dtype=np.uint8)
            # assign instance ids 1..k following mask_info order
            # map RGB->inst id
            inst_id = 1
            for rgb_key, entry in mask_info.items():
                # rgb_key like "(r, g, b)"
                trip = [int(v) for v in str(rgb_key).replace('(', '').replace(')','').replace(' ','').split(',')[:3]]
                trip = np.array(trip, dtype=np.uint8)
                m = np.all(mask_rgb == trip.reshape(1,1,3), axis=2)
                if np.any(m):
                    mask_np[m] = inst_id
                    inst_id += 1
            # Others are background (0)

        # camera intr from camera_params_XXXX.json
        intr = None
        cam_p = os.path.join(frame_dir, f"camera_params_{frame_idx}.json")
        if os.path.exists(cam_p):
            cam_params = read_json(cam_p, default=None)
            if cam_params is not None:
                intr = intr_from_graspit_camera_params(cam_params)

        # 6D poses from scene.yaml (convert world2obj -> obj-in-world)
        poses = self._poses_from_yaml(scene_yaml['_objects'])

        # models (point clouds) per object
        models = {}
        if self.include_models:
            for p in poses:
                obj_id = p['obj_id']
                pts = self._load_assets_pc(obj_id)
                if pts is None:
                    continue
                pc_norm, center, scale = self._norm_model_pc(pts)
                models[obj_id] = {'pc_norm': torch.from_numpy(pc_norm), 'center': torch.from_numpy(center), 'scale': float(scale)}

        # image type/source
        # Use parent folder (e.g., scenes_3) as "source"
        image_type = os.path.basename(os.path.dirname(meta['scene_dir']))  # e.g., scenes_3

        # Apply transforms + build boxes
        img_t, depth_t, mask_t, intr, poses_aug = self._apply_augs(img, depth_np, mask_np, intr, poses)
        mask_np_resized = mask_t.numpy() if mask_t is not None else None
        boxes_xyxy, boxes_cxcywh = self._build_targets(mask_np_resized)

        sample = {
            'dataset': self.name,
            'image_id': f"{os.path.basename(meta['scene_dir'])}/{os.path.basename(meta['frame_dir'])}",
            'image_type': image_type,
            'img': img_t,                               # C,H,W normalized
            'depth': depth_t,                           # (1,H,W) meters
            'mask': mask_t,                             # (H,W) int64 with 0..k
            'boxes_xyxy': torch.from_numpy(boxes_xyxy),
            'boxes_cxcywh': torch.from_numpy(boxes_cxcywh),
            'intr': torch.tensor(intr, dtype=torch.float32) if intr is not None else None,
            'poses': poses_aug,                         # list of dicts {R,t,obj_id,xyz}
            'models': models,                           # dict obj_id->{pc_norm,center,scale}
            'H': OUT_H, 'W': OUT_W
        }
        return sample

    @staticmethod
    def inventory(root: str) -> Dict[str, int]:
        # count images under scenes/ and scenes_*
        cnt = 0
        per_source = Counter()
        scene_dirs = []
        if os.path.exists(os.path.join(root, 'scenes')):
            scene_dirs.append(os.path.join(root, 'scenes'))
        scene_dirs.extend(glob.glob(os.path.join(root, 'scenes_*')))
        for sdir in scene_dirs:
            for scene in os.listdir(sdir):
                scene_path = os.path.join(sdir, scene)
                frames = [f for f in os.listdir(scene_path) if f.startswith('frame_')]
                per_source[os.path.basename(sdir)] += len(frames)
                cnt += len(frames)
        return {'total': cnt, 'per_source': dict(per_source)}

    def visualize(self, idx: int, max_instances_overlay: int = 3, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        sample = self[idx]
        img = TF.to_pil_image(sample['img'])
        depth = sample['depth'].squeeze(0).numpy() if sample['depth'] is not None else None
        mask = sample['mask'].numpy() if sample['mask'] is not None else None
        boxes = sample['boxes_xyxy'].numpy()
        intr = sample['intr'].numpy().tolist() if sample['intr'] is not None else None
        poses = sample['poses'] or []
        models = sample['models'] or {}
        pal = get_color_palette(max(len(boxes), 1))

        img_boxes = draw_boxes_on_rgb(img, boxes, palette=pal)
        depth_vis = depth_to_colormap(depth) if depth is not None else Image.new('RGB', img.size, (0,0,0))
        mask_vis = colorize_mask(mask, pallet:=pal) if mask is not None else Image.new('RGB', img.size, (0,0,0))

        # Overlay projected points for first few instances
        img_pts = img.copy()
        if intr is not None and len(poses) > 0 and len(models) > 0:
            count = 0
            for p in poses:
                obj_id = str(p['obj_id'])
                if obj_id not in models:
                    continue
                pc_norm = models[obj_id]['pc_norm'].numpy()  # Nx3
                scale = models[obj_id]['scale']
                # metric points: R*(pc_norm*scale) + t
                pts_metric = pc_norm * scale
                uv = project_points(pts_metric, p['R'], p['t'], intr)
                col = pal[count % len(pal)]
                img_pts = overlay_points_on_rgb(img_pts, uv, color=col)
                count += 1
                if count >= max_instances_overlay:
                    break

        fig = plt.figure(figsize=(10,7))
        ax1 = fig.add_subplot(2,2,1); ax1.set_title('RGB'); ax1.imshow(img); ax1.axis('off')
        ax2 = fig.add_subplot(2,2,2); ax2.set_title('Depth'); ax2.imshow(depth_vis); ax2.axis('off')
        ax3 = fig.add_subplot(2,2,3); ax3.set_title('Mask'); ax3.imshow(mask_vis); ax3.axis('off')
        ax4 = fig.add_subplot(2,2,4); ax4.set_title('RGB + boxes/pc'); ax4.imshow(img_boxes); ax4.imshow(img_pts, alpha=0.5); ax4.axis('off')
        plt.tight_layout()
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

# -----------------------

# GraspNet Dataset

# -----------------------

class GraspNetDataset(BasePoseDataset):
    def __init__(self, root: str, mode='val',
                 transforms=True,
                 include_depth=True,
                 include_mask=True,
                 include_models=True,
                 include_grasp_labels=False,
                 mask_xml_match_policy='index_order'):
        # Per user: put all of GraspNet only in val (sampling in train = 0)
        super().__init__(mode=mode, transforms=transforms)
        self.name = 'graspnet'
        self.root = root
        self.include_depth = include_depth
        self.include_mask = include_mask
        self.include_models = include_models
        self.include_grasp_labels = include_grasp_labels
        self.mask_xml_match_policy = mask_xml_match_policy

        # build directories: train_*, test_*, val_*
        # We will put only val if mode == 'val', and no samples in train
        self.samples = []  # entries: dict with paths
        if self.mode == 'train':
            # user requirement: sampling in train = 0
            self.samples = []
            return

        # typical structure: train_1/scene_0000/realsense/{rgb,depth,label,annotations}
        # also realsense/kinect
        for split in os.listdir(root):
            if not os.path.isdir(os.path.join(root, split)):
                continue
            # for 'val' mode, accept any split; for 'test' similarly; but we'll default to scene_*/realsense
            for scene_dir in glob.glob(os.path.join(root, split, 'scene_*')):
                for sensor in ['realsense', 'kinect']:
                    base = os.path.join(scene_dir, sensor)
                    rgb_dir = os.path.join(base, 'rgb')
                    depth_dir = os.path.join(base, 'depth')
                    label_dir = os.path.join(base, 'label')
                    annot_dir = os.path.join(base, 'annotations')
                    meta_dir = os.path.join(base, 'meta')
                    if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir) or not os.path.isdir(label_dir):
                        continue
                    # gather names
                    for img_name in sorted(os.listdir(rgb_dir)):
                        if not img_name.endswith('.png'):
                            continue
                        stem = os.path.splitext(img_name)[0]
                        rgb_path = os.path.join(rgb_dir, img_name)
                        depth_path = os.path.join(depth_dir, stem + '.png')
                        label_path = os.path.join(label_dir, stem + '.png')
                        xml_path = os.path.join(annot_dir, stem + '.xml')
                        meta_path = os.path.join(meta_dir, stem + '.mat')  # if needed; alternatively camK.npy
                        # camera intr: check camK.npy
                        camK_path = os.path.join(base, 'camK.npy')
                        self.samples.append({
                            'split': split,
                            'scene': os.path.basename(scene_dir),
                            'sensor': sensor,
                            'rgb': rgb_path,
                            'depth': depth_path,
                            'label': label_path,
                            'xml': xml_path,
                            'camK': camK_path if os.path.exists(camK_path) else None
                        })

        # Dummy models cache
        self.models_dummy_dir = os.path.join(self.root, f"models_dummy_pc_{PC_NUM_POINTS}")
        ensure_dir(self.models_dummy_dir)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_intr(camK_path: Optional[str]) -> Optional[List[float]]:
        if camK_path is None or not os.path.exists(camK_path):
            return None
        K = np.load(camK_path)
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        return [float(cx), float(cy), float(fx), float(fy)]

    @staticmethod
    def _load_xml_poses(xml_path: str) -> List[Dict[str, Any]]:
        poses = []
        if not os.path.exists(xml_path):
            return poses
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('obj'):
                obj_id = obj.find('obj_id').text.strip()
                name = obj.find('obj_name').text.strip() if obj.find('obj_name') is not None else obj_id
                pos = [float(v) for v in obj.find('pos_in_world').text.strip().split()]
                ori = [float(v) for v in obj.find('ori_in_world').text.strip().split()]  # qw qx qy qz
                R = quaternion_to_rotation_matrix(ori)
                t = np.array(pos, dtype=np.float32)
                poses.append({'obj_id': obj_id, 'name': name, 'R': R.astype(np.float32), 't': t.astype(np.float32)})
        except Exception as e:
            print(f"[GraspNetDataset] XML parse failed {xml_path}: {e}")
        return poses

    def _load_dummy_model(self, obj_name: str) -> np.ndarray:
        # create or load a dummy unit sphere point cloud
        out = build_dummy_sphere_pc(self.models_dummy_dir, name=os.path.splitext(obj_name)[0], num_points=PC_NUM_POINTS)
        pts = load_pointcloud_file(out)
        return pts if pts is not None else np.zeros((0,3), dtype=np.float32)

    def __getitem__(self, index: int):
        meta = self.samples[index]
        # RGB
        img = Image.open(meta['rgb']).convert('RGB')
        # Depth (convert to meters; assume in mm)
        depth_np = np.array(Image.open(meta['depth']), dtype=np.float32) * 0.001 if self.include_depth else None
        # Mask
        mask_np = np.array(Image.open(meta['label']), dtype=np.uint8) if self.include_mask else None
        # Remap mask to 1..k instance IDs
        inst_ids = sorted([i for i in np.unique(mask_np) if i != 0]) if mask_np is not None else []
        id_map = {v: i+1 for i, v in enumerate(inst_ids)}
        if mask_np is not None:
            mask_np = np.vectorize(lambda v: id_map.get(v, 0))(mask_np).astype(np.uint8)

        # Intr
        intr = self._load_intr(meta['camK'])

        # Poses from XML (dummy match to mask instances by index order)
        poses_xml = self._load_xml_poses(meta['xml'])
        poses = []
        if len(poses_xml) > 0 and len(inst_ids) > 0:
            n = min(len(inst_ids), len(poses_xml))
            for i in range(n):
                poses.append({
                    'obj_id': poses_xml[i]['obj_id'],
                    'R': poses_xml[i]['R'],
                    't': poses_xml[i]['t']
                })

        # Models (dummy)
        models = {}
        if self.include_models and len(poses) > 0:
            for p in poses:
                name = p['obj_id']  # we don't have class mapping; use obj_id as name
                pts = self._load_dummy_model(name)
                pc_norm, center, scale = self._norm_model_pc(pts)
                models[p['obj_id']] = {'pc_norm': torch.from_numpy(pc_norm), 'center': torch.from_numpy(center), 'scale': float(scale)}

        image_type = meta['sensor']  # 'realsense' or 'kinect'

        # Apply transforms
        img_t, depth_t, mask_t, intr, poses_aug = self._apply_augs(img, depth_np, mask_np, intr, poses)
        mask_np_resized = mask_t.numpy() if mask_t is not None else None
        boxes_xyxy, boxes_cxcywh = self._build_targets(mask_np_resized)

        sample = {
            'dataset': self.name,
            'image_id': f"{meta['split']}/{meta['scene']}/{meta['sensor']}/{os.path.basename(meta['rgb'])}",
            'image_type': image_type,
            'img': img_t,
            'depth': depth_t,
            'mask': mask_t,
            'boxes_xyxy': torch.from_numpy(boxes_xyxy),
            'boxes_cxcywh': torch.from_numpy(boxes_cxcywh),
            'intr': torch.tensor(intr, dtype=torch.float32) if intr is not None else None,
            'poses': poses_aug,
            'models': models,
            'H': OUT_H, 'W': OUT_W,
        }
        # Optional grasp labels (dummy now)
        if self.include_grasp_labels:
            sample['grasp_labels'] = []  # placeholder
        return sample

    @staticmethod
    def inventory(root: str) -> Dict[str, Any]:
        per_source = Counter()
        total = 0
        for split in os.listdir(root):
            split_dir = os.path.join(root, split)
            if not os.path.isdir(split_dir):
                continue
            for scene_dir in glob.glob(os.path.join(split_dir, 'scene_*')):
                for sensor in ['realsense','kinect']:
                    rgb_dir = os.path.join(scene_dir, sensor, 'rgb')
                    if os.path.isdir(rgb_dir):
                        n = len([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
                        per_source[sensor] += n
                        total += n
        return {'total': total, 'per_source': dict(per_source)}

    def visualize(self, idx: int, max_instances_overlay: int = 3, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        sample = self[idx]
        img = TF.to_pil_image(sample['img'])
        depth = sample['depth'].squeeze(0).numpy() if sample['depth'] is not None else None
        mask = sample['mask'].numpy() if sample['mask'] is not None else None
        boxes = sample['boxes_xyxy'].numpy()
        intr = sample['intr'].numpy().tolist() if sample['intr'] is not None else None
        poses = sample['poses'] or []
        models = sample['models'] or {}
        pal = get_color_palette(max(len(boxes), 1))

        img_boxes = draw_boxes_on_rgb(img, boxes, palette=pal)
        depth_vis = depth_to_colormap(depth) if depth is not None else Image.new('RGB', img.size, (0,0,0))
        mask_vis = colorize_mask(mask, pal) if mask is not None else Image.new('RGB', img.size, (0,0,0))

        img_pts = img.copy()
        if intr is not None and len(poses) > 0 and len(models) > 0:
            count = 0
            for p in poses:
                obj_id = str(p['obj_id'])
                if obj_id not in models:
                    continue
                pc_norm = models[obj_id]['pc_norm'].numpy()
                scale = models[obj_id]['scale']
                pts_metric = pc_norm * scale
                uv = project_points(pts_metric, p['R'], p['t'], intr)
                col = pal[count % len(pal)]
                img_pts = overlay_points_on_rgb(img_pts, uv, color=col)
                count += 1
                if count >= max_instances_overlay:
                    break

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,7))
        ax1 = fig.add_subplot(2,2,1); ax1.set_title('RGB'); ax1.imshow(img); ax1.axis('off')
        ax2 = fig.add_subplot(2,2,2); ax2.set_title('Depth'); ax2.imshow(depth_vis); ax2.axis('off')
        ax3 = fig.add_subplot(2,2,3); ax3.set_title('Mask'); ax3.imshow(mask_vis); ax3.axis('off')
        ax4 = fig.add_subplot(2,2,4); ax4.set_title('RGB + boxes/pc'); ax4.imshow(img_boxes); ax4.imshow(img_pts, alpha=0.5); ax4.axis('off')
        plt.tight_layout()
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

# -----------------------

# BOP Dataset

# -----------------------

class BOPDataset(BasePoseDataset):
    def __init__(self, bop_root: str, mode='train',
                 transforms=True,
                 include_depth=True,
                 include_mask=True,
                 include_models=True,
                 interactive=False):
        super().__init__(mode=mode, transforms=transforms)
        self.name = 'bop'
        self.root = bop_root
        self.include_depth = include_depth
        self.include_mask = include_mask
        self.include_models = include_models
        self.interactive = interactive

        self.datasets = []     # dataset names under bop_root
        self.source_types = {} # dataset_name -> list of source types present
        self.source_probs = {} # dataset_name -> {source_type: prob}

        # discover datasets under bop_root
        for d in sorted(os.listdir(bop_root)):
            dpath = os.path.join(bop_root, d)
            if os.path.isdir(dpath):
                self.datasets.append(d)

        self._build_source_inventory_and_probs()

        # Build samples list (restricted to mode)
        self.samples = self._enumerate_samples()

        # Model pc cache per dataset
        # For each dataset, ensure downsampled model dir exists
        for ds_name in self.datasets:
            model_eval_dir = os.path.join(self.root, ds_name, 'models_eval')
            model_dir = os.path.join(self.root, ds_name, 'models')
            if os.path.isdir(model_eval_dir):
                out_dir = os.path.join(self.root, ds_name, f"models_eval_pc_{PC_NUM_POINTS}")
                ensure_downsampled_model_dir(model_eval_dir, out_dir, pattern_mesh=('*.ply','*.obj'), num_points=PC_NUM_POINTS)
            elif os.path.isdir(model_dir):
                out_dir = os.path.join(self.root, ds_name, f"models_pc_{PC_NUM_POINTS}")
                ensure_downsampled_model_dir(model_dir, out_dir, pattern_mesh=('*.ply','*.obj'), num_points=PC_NUM_POINTS)

    def _build_source_inventory_and_probs(self):
        # For each dataset, find source types under it (train_pbr, train_real, train_synt, val, test, test_primesense, etc)
        inv = {}
        for ds in self.datasets:
            dpath = os.path.join(self.root, ds)
            stypes = []
            for sub in os.listdir(dpath):
                subp = os.path.join(dpath, sub)
                if os.path.isdir(subp):
                    # Accept leaf dirs that contain scene dirs (000000)
                    if len(glob.glob(os.path.join(subp, '[0-9][0-9][0-9][0-9][0-9][0-9]'))) > 0:
                        stypes.append(sub)
            inv[ds] = sorted(stypes)
        self.source_types = inv

        # load or create sampling json
        samp_path = os.path.join(self.root, 'bop_sampling.json')
        old = read_json(samp_path, default=None)
        # default equal probabilities per source type
        new_probs = {}
        for ds, stypes in self.source_types.items():
            if len(stypes) == 0:
                continue
            eq = 1.0 / len(stypes)
            new_probs[ds] = {s: eq for s in stypes}

        if old is None:
            self.source_probs = new_probs
            write_json(samp_path, {'source_probs': self.source_probs})
        else:
            # if any new source types found, add them with equal prob
            changed = False
            sp = old.get('source_probs', {})
            for ds, stypes in self.source_types.items():
                if ds not in sp:
                    sp[ds] = {s: 1.0 / len(stypes) for s in stypes} if stypes else {}
                    changed = True
                else:
                    for s in stypes:
                        if s not in sp[ds]:
                            sp[ds][s] = 1.0 / len(stypes)
                            changed = True
            self.source_probs = sp
            if changed:
                if self.interactive:
                    ans = input("New BOP sources detected. Overwrite sampling JSON with defaults? [y/N]: ").strip().lower()
                    if ans == 'y':
                        write_json(samp_path, {'source_probs': self.source_probs})
                else:
                    # Update but keep existing others
                    write_json(samp_path, {'source_probs': self.source_probs})

    def _enumerate_samples(self):
        samples = []
        # Each dataset ds: each source type T (train_pbr, train_real, test, val...)
        # Mode filter: train excludes subdirs containing 'val' or 'test'; val includes 'val' and may include 'test'; test includes only 'test'.
        for ds in self.datasets:
            dpath = os.path.join(self.root, ds)
            stypes = self.source_types.get(ds, [])
            for st in stypes:
                if self.mode == 'train':
                    if ('val' in st) or ('test' in st):
                        continue
                elif self.mode == 'val':
                    if ('val' not in st) and ('test' not in st):
                        continue
                elif self.mode == 'test':
                    if ('test' not in st):
                        continue

                st_dir = os.path.join(dpath, st)
                # Scenes: 000000, 000001, ...
                scene_dirs = sorted(glob.glob(os.path.join(st_dir, '[0-9][0-9][0-9][0-9][0-9][0-9]')))
                for scene_path in scene_dirs:
                    scene_id = os.path.basename(scene_path)
                    rgb_dir = os.path.join(scene_path, 'rgb')
                    depth_dir = os.path.join(scene_path, 'depth')
                    cam_json = os.path.join(scene_path, 'scene_camera.json')
                    gt_json = os.path.join(scene_path, 'scene_gt.json')
                    gt_info_json = os.path.join(scene_path, 'scene_gt_info.json')
                    # mask directories optional
                    mask_visib_dir = os.path.join(scene_path, 'mask_visib')
                    mask_dir = os.path.join(scene_path, 'mask')
                    if not os.path.exists(rgb_dir):
                        continue
                    for img_file in sorted(os.listdir(rgb_dir)):
                        if not img_file.endswith('.png'):
                            continue
                        im_id = int(os.path.splitext(img_file)[0])
                        rgb_path = os.path.join(rgb_dir, img_file)
                        depth_path = os.path.join(depth_dir, f"{im_id:06d}.png") if os.path.isdir(depth_dir) else None
                        samples.append({
                            'ds_name': ds,
                            'source_type': st,
                            'scene_dir': scene_path,
                            'scene_id': scene_id,
                            'im_id': im_id,
                            'rgb': rgb_path,
                            'depth': depth_path if (self.include_depth and depth_path and os.path.exists(depth_path)) else None,
                            'scene_camera': cam_json,
                            'scene_gt': gt_json,
                            'scene_gt_info': gt_info_json if os.path.exists(gt_info_json) else None,
                            'mask_visib_dir': mask_visib_dir if os.path.isdir(mask_visib_dir) else None,
                            'mask_dir': mask_dir if os.path.isdir(mask_dir) else None,
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_bop_camera(scene_camera_json: str, im_id: int) -> Tuple[Optional[List[float]], float]:
        cam = read_json(scene_camera_json, default={})
        cam_im = cam.get(str(im_id), None)
        if cam_im is None:
            return None, 0.001
        K = np.array(cam_im['cam_K'], dtype=np.float32).reshape(3,3)
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        ds = float(cam_im.get('depth_scale', 0.001))  # meters per depth unit
        return [float(cx), float(cy), float(fx), float(fy)], ds

    @staticmethod
    def _load_bop_gt(scene_gt_json: str, im_id: int) -> List[Dict[str,Any]]:
        gt = read_json(scene_gt_json, default={})
        items = gt.get(str(im_id), [])
        poses = []
        for it in items:
            obj_id = str(it['obj_id'])
            R = np.array(it['cam_R_m2c'], dtype=np.float32).reshape(3,3)    # object-to-camera
            t = np.array(it['cam_t_m2c'], dtype=np.float32) * 0.001         # mm->m
            poses.append({'obj_id': obj_id, 'R': R, 't': t})
        return poses

    def _load_bop_mask_instance(self, sample: Dict[str, Any]) -> Optional[np.ndarray]:
        # Build instance map from mask_visib/ or mask/. Each object instance corresponds to an index in scene_gt[im_id].
        # Files: {im_id:06d}_{inst_id:06d}.png with inst_id indexing in order of GT list.
        if sample['mask_visib_dir'] is None and sample['mask_dir'] is None:
            return None
        # need number of gt entries
        gt = read_json(sample['scene_gt'], default={})
        gt_items = gt.get(str(sample['im_id']), [])
        H, W = None, None
        inst_map = None
        for inst_idx in range(len(gt_items)):
            fname = f"{sample['im_id']:06d}_{inst_idx:06d}.png"
            mask_path = None
            if sample['mask_visib_dir'] is not None:
                mp = os.path.join(sample['mask_visib_dir'], fname)
                if os.path.exists(mp): mask_path = mp
            if mask_path is None and sample['mask_dir'] is not None:
                mp = os.path.join(sample['mask_dir'], fname)
                if os.path.exists(mp): mask_path = mp
            if mask_path is None:
                continue
            m = np.array(Image.open(mask_path), dtype=np.uint8)
            if H is None:
                H, W = m.shape[:2]
                inst_map = np.zeros((H,W), dtype=np.uint8)
            inst_map[m > 0] = inst_idx + 1
        return inst_map

    def _load_bop_model_pc(self, ds_name: str, obj_id: str) -> Optional[np.ndarray]:
        # Prefer models_eval_pc_*, else models_pc_*
        md_eval = os.path.join(self.root, ds_name, f"models_eval_pc_{PC_NUM_POINTS}", f"obj_{int(obj_id):06d}_pc{PC_NUM_POINTS}.ply")
        md_eval_glob = glob.glob(os.path.join(self.root, ds_name, f"models_eval_pc_{PC_NUM_POINTS}", f"obj_{int(obj_id):06d}*_pc{PC_NUM_POINTS}.ply"))
        if os.path.exists(md_eval):
            return load_pointcloud_file(md_eval)
        if len(md_eval_glob) > 0 and os.path.exists(md_eval_glob[0]):
            return load_pointcloud_file(md_eval_glob[0])
        md = os.path.join(self.root, ds_name, f"models_pc_{PC_NUM_POINTS}", f"obj_{int(obj_id):06d}_pc{PC_NUM_POINTS}.ply")
        md_glob = glob.glob(os.path.join(self.root, ds_name, f"models_pc_{PC_NUM_POINTS}", f"obj_{int(obj_id):06d}*_pc{PC_NUM_POINTS}.ply"))
        if os.path.exists(md):
            return load_pointcloud_file(md)
        if len(md_glob) > 0 and os.path.exists(md_glob[0]):
            return load_pointcloud_file(md_glob[0])
        # Try making from models_eval or models directly (if not precomputed)
        model_eval_dir = os.path.join(self.root, ds_name, 'models_eval')
        model_dir = os.path.join(self.root, ds_name, 'models')
        cand_mesh = None
        for base in [model_eval_dir, model_dir]:
            g = glob.glob(os.path.join(base, f"obj_{int(obj_id):06d}.ply"))
            if g:
                cand_mesh = g[0]
                break
        if cand_mesh is None:
            return None
        out_dir = os.path.join(self.root, ds_name, f"models_eval_pc_{PC_NUM_POINTS}" if 'models_eval' in cand_mesh else f"models_pc_{PC_NUM_POINTS}")
        ensure_dir(out_dir)
        out_pc = os.path.join(out_dir, f"obj_{int(obj_id):06d}_pc{PC_NUM_POINTS}.ply")
        out_path = ensure_downsampled_mesh_to_pc(cand_mesh, out_pc, num_points=PC_NUM_POINTS)
        if out_path is None:
            return None
        return load_pointcloud_file(out_path)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        # Load rgb
        img = Image.open(sample['rgb']).convert('RGB')

        # Camera intr and depth scale
        intr, depth_unit_scale = self._load_bop_camera(sample['scene_camera'], sample['im_id'])

        # Depth meters
        depth_np = None
        if self.include_depth and sample['depth'] is not None and os.path.exists(sample['depth']):
            depth_raw = np.array(Image.open(sample['depth']), dtype=np.float32)
            depth_np = depth_raw * float(depth_unit_scale)

        # GT poses
        poses = self._load_bop_gt(sample['scene_gt'], sample['im_id'])

        # Mask (if available)
        mask_np = self._load_bop_mask_instance(sample) if self.include_mask else None

        # Models
        models = {}
        if self.include_models:
            for p in poses:
                obj_id = p['obj_id']
                pts = self._load_bop_model_pc(sample['ds_name'], obj_id)
                if pts is None:
                    continue
                pc_norm, center, scale = self._norm_model_pc(pts)
                models[obj_id] = {'pc_norm': torch.from_numpy(pc_norm), 'center': torch.from_numpy(center), 'scale': float(scale)}

        # Apply augs and build boxes
        img_t, depth_t, mask_t, intr, poses_aug = self._apply_augs(img, depth_np, mask_np, intr, poses)
        mask_np_resized = mask_t.numpy() if mask_t is not None else None
        boxes_xyxy, boxes_cxcywh = self._build_targets(mask_np_resized)

        out = {
            'dataset': self.name,
            'image_id': f"{sample['ds_name']}/{sample['source_type']}/{sample['scene_id']}/{sample['im_id']:06d}",
            'image_type': sample['source_type'],
            'img': img_t,
            'depth': depth_t,
            'mask': mask_t,
            'boxes_xyxy': torch.from_numpy(boxes_xyxy),
            'boxes_cxcywh': torch.from_numpy(boxes_cxcywh),
            'intr': torch.tensor(intr, dtype=torch.float32) if intr is not None else None,
            'poses': poses_aug,
            'models': models,
            'H': OUT_H, 'W': OUT_W,
        }
        return out

    @staticmethod
    def inventory(bop_root: str) -> Dict[str, Any]:
        # Count images per dataset and per source type
        inv = {}
        total = 0
        for ds in sorted(os.listdir(bop_root)):
            dsdir = os.path.join(bop_root, ds)
            if not os.path.isdir(dsdir): continue
            inv[ds] = {}
            for st in sorted(os.listdir(dsdir)):
                st_dir = os.path.join(dsdir, st)
                if not os.path.isdir(st_dir):
                    continue
                # scenes:
                scenes = glob.glob(os.path.join(st_dir, '[0-9][0-9][0-9][0-9][0-9][0-9]'))
                cnt = 0
                for sp in scenes:
                    rgb_dir = os.path.join(sp, 'rgb')
                    if os.path.isdir(rgb_dir):
                        cnt += len([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
                if cnt > 0:
                    inv[ds][st] = cnt
                    total += cnt
        return {'total': total, 'per_dataset': inv}

    def visualize(self, idx: int, max_instances_overlay: int = 3, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        sample = self[idx]
        img = TF.to_pil_image(sample['img'])
        depth = sample['depth'].squeeze(0).numpy() if sample['depth'] is not None else None
        mask = sample['mask'].numpy() if sample['mask'] is not None else None
        boxes = sample['boxes_xyxy'].numpy()
        intr = sample['intr'].numpy().tolist() if sample['intr'] is not None else None
        poses = sample['poses'] or []
        models = sample['models'] or {}
        pal = get_color_palette(max(len(boxes), 1))

        img_boxes = draw_boxes_on_rgb(img, boxes, palette=pal)
        depth_vis = depth_to_colormap(depth) if depth is not None else Image.new('RGB', img.size, (0,0,0))
        mask_vis = colorize_mask(mask, pal) if mask is not None else Image.new('RGB', img.size, (0,0,0))

        img_pts = img.copy()
        if intr is not None and len(poses) > 0 and len(models) > 0:
            count = 0
            for p in poses:
                obj_id = str(p['obj_id'])
                if obj_id not in models:
                    continue
                pc_norm = models[obj_id]['pc_norm'].numpy()
                scale = models[obj_id]['scale']
                pts_metric = pc_norm * scale
                uv = project_points(pts_metric, p['R'], p['t'], intr)
                col = pal[count % len(pal)]
                img_pts = overlay_points_on_rgb(img_pts, uv, color=col)
                count += 1
                if count >= max_instances_overlay:
                    break

        fig = plt.figure(figsize=(10,7))
        ax1 = fig.add_subplot(2,2,1); ax1.set_title('RGB'); ax1.imshow(img); ax1.axis('off')
        ax2 = fig.add_subplot(2,2,2); ax2.set_title('Depth'); ax2.imshow(depth_vis); ax2.axis('off')
        ax3 = fig.add_subplot(2,2,3); ax3.set_title('Mask'); ax3.imshow(mask_vis); ax3.axis('off')
        ax4 = fig.add_subplot(2,2,4); ax4.set_title('RGB + boxes/pc'); ax4.imshow(img_boxes); ax4.imshow(img_pts, alpha=0.5); ax4.axis('off')
        plt.tight_layout()
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

# -----------------------

# Mixer Dataset

# -----------------------

class MixedMultiDataset(data.Dataset):
    def __init__(self,
                 datasets: Dict[str, data.Dataset],
                 dataset_probs: Dict[str, float],
                 num_samples_per_epoch: int = 10000,
                 bop_source_probs_path: Optional[str] = None,
                 failed_log_path: str = './failed_samples.json',
                 max_retry: int = 10,
                 mode='train'):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.datasets = datasets
        # Normalize dataset_probs and enforce GraspNet->0 in train (per requirement)
        dp = dict(dataset_probs)
        if 'graspnet' in dp and self.mode == 'train':
            dp['graspnet'] = 0.0
        # Normalize
        s = sum(dp.values())
        if s <= 0:
            # fallback equally among non-zero length datasets
            nonempty = [k for k,v in datasets.items() if len(v) > 0]
            dp = {k: 1.0/len(nonempty) for k in nonempty}
        else:
            for k in dp:
                dp[k] = dp[k] / s
        self.dataset_probs = dp
        self.dataset_names = list(self.dataset_probs.keys())
        self.num_samples_per_epoch = num_samples_per_epoch
        self.max_retry = max_retry
        self.failed_log_path = failed_log_path
        self.failed = read_json(self.failed_log_path, default={'failed': []})

        # Build cumulative probabilities for sampling
        self.cum_probs = np.cumsum([self.dataset_probs[k] for k in self.dataset_names])

    def __len__(self):
        return self.num_samples_per_epoch

    def _sample_dataset_name(self) -> str:
        r = random.random()
        for name, cp in zip(self.dataset_names, self.cum_probs):
            if r <= cp:
                return name
        return self.dataset_names[-1]

    def _safe_get(self, ds_name: str):
        ds = self.datasets[ds_name]
        # Uniform random index; could be refined to sample bop sources by probabilities inside BOPDataset,
        # but here we rely on BOPDataset being prefiltered by mode and its own source_probs JSON.
        for _ in range(self.max_retry):
            try:
                idx = random.randint(0, len(ds)-1)
                return ds[idx]
            except Exception as e:
                # record failure
                self.failed['failed'].append({'dataset': ds_name, 'time': time.time(), 'exc': str(e)})
                continue
        # if we reach here, fallback to another dataset
        for name in self.dataset_names:
            if name == ds_name:
                continue
            for _ in range(self.max_retry):
                try:
                    idx = random.randint(0, len(self.datasets[name])-1)
                    return self.datasets[name][idx]
                except:
                    self.failed['failed'].append({'dataset': name, 'time': time.time(), 'exc': 'fallback_failed'})
                    continue
        raise RuntimeError("Mixer: All retries failed")

    def __getitem__(self, index: int):
        ds_name = self._sample_dataset_name()
        sample = self._safe_get(ds_name)
        return sample

    def save_failed(self):
        write_json(self.failed_log_path, self.failed)

# -----------------------

# __main__ demos

# -----------------------

def print_inventory_graspit(root: str, assets_root: Optional[str] = None):
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

def demo_visualize(dataset, n_show=2):
    for i in range(min(n_show, len(dataset))):
        try:
            dataset.visualize(i)
        except Exception as e:
            print(f"Visualization failed for idx {i}: {e}")

if __name__ == "__main__":
    set_seed(42)
    # Adjust roots as needed
    GRASPIT_ROOT = os.environ.get('GRASPIT_ROOT', '/mnt/kikerp/OptiSim')
    GRASPIT_ASSETS = os.environ.get('GRASPIT_ASSETS', os.path.join(GRASPIT_ROOT, 'assets'))
    GRASPNET_ROOT = os.environ.get('GRASPNET_ROOT', '/mnt/wimi/publicdata/GraspNet-1Billion')
    BOP_ROOT = os.environ.get('BOP_ROOT', './data/bop')

    # Print inventories
    if os.path.exists(GRASPIT_ROOT):
        print_inventory_graspit(GRASPIT_ROOT, GRASPIT_ASSETS)
    if os.path.exists(GRASPNET_ROOT):
        print_inventory_graspnet(GRASPNET_ROOT)
    if os.path.exists(BOP_ROOT):
        print_inventory_bop(BOP_ROOT)

    # Build datasets (train/val)
    # GraspIt
    if os.path.exists(GRASPIT_ROOT):
        ds_graspit_train = GraspItDataset(GRASPIT_ROOT, assets_root=GRASPIT_ASSETS, mode='train')
        ds_graspit_val = GraspItDataset(GRASPIT_ROOT, assets_root=GRASPIT_ASSETS, mode='val', transforms=False)  # less/no augs in val
        print(f"GraspIt train len={len(ds_graspit_train)}; val len={len(ds_graspit_val)}")
        # visualize a few
        if len(ds_graspit_val) > 0:
            demo_visualize(ds_graspit_val, n_show=1)

    # GraspNet (only val per requirement)
    if os.path.exists(GRASPNET_ROOT):
        ds_graspnet_val = GraspNetDataset(GRASPNET_ROOT, mode='val', transforms=False, include_grasp_labels=True)
        print(f"GraspNet val len={len(ds_graspnet_val)} (train excluded)")
        if len(ds_graspnet_val) > 0:
            demo_visualize(ds_graspnet_val, n_show=1)

    # BOP (train/val)
    if os.path.exists(BOP_ROOT):
        ds_bop_train = BOPDataset(BOP_ROOT, mode='train', transforms=True, include_mask=True)
        ds_bop_val = BOPDataset(BOP_ROOT, mode='val', transforms=False, include_mask=True)
        print(f"BOP train len={len(ds_bop_train)}; val len={len(ds_bop_val)}")
        if len(ds_bop_val) > 0:
            demo_visualize(ds_bop_val, n_show=1)

    # Mixer (train)
    datasets_train = {}
    probs_train = {}
    if os.path.exists(GRASPIT_ROOT):
        datasets_train['graspit'] = ds_graspit_train
        probs_train['graspit'] = 0.5
    if os.path.exists(BOP_ROOT):
        datasets_train['bop'] = ds_bop_train
        probs_train['bop'] = 0.5
    # GraspNet excluded in train (prob forced to 0)
    if os.path.exists(GRASPNET_ROOT):
        datasets_train['graspnet'] = GraspNetDataset(GRASPNET_ROOT, mode='train')  # empty by design
        probs_train['graspnet'] = 0.0

    if len(datasets_train) > 0:
        mixer_train = MixedMultiDataset(datasets=datasets_train, dataset_probs=probs_train, num_samples_per_epoch=1000, mode='train')
        print(f"Mixer train len={len(mixer_train)}")
        # pull a few samples
        for i in range(5):
            try:
                _ = mixer_train[i]
            except Exception as e:
                print(f"Mixer sample failed: {e}")
        mixer_train.save_failed()

    # Mixer (val): include GraspNet
    datasets_val = {}
    probs_val = {}
    if os.path.exists(GRASPIT_ROOT):
        datasets_val['graspit'] = ds_graspit_val
        probs_val['graspit'] = 0.4
    if os.path.exists(BOP_ROOT):
        datasets_val['bop'] = ds_bop_val
        probs_val['bop'] = 0.4
    if os.path.exists(GRASPNET_ROOT):
        datasets_val['graspnet'] = ds_graspnet_val
        probs_val['graspnet'] = 0.2

    if len(datasets_val) > 0:
        mixer_val = MixedMultiDataset(datasets=datasets_val, dataset_probs=probs_val, num_samples_per_epoch=200, mode='val')
        print(f"Mixer val len={len(mixer_val)}")
        for i in range(5):
            try:
                _ = mixer_val[i]
            except Exception as e:
                print(f"Mixer val sample failed: {e}")