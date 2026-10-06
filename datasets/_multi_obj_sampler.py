
# datasets_mixed.py (refactored to use per-dataset configuration)

import os
import re
import io
import cv2
import sys
import contextlib

sys.path.append(os.getcwd())
import json
import time
import glob
import yaml
import math
import uuid
import copy
import random
import shutil
import h5py
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

from scipy.spatial import ConvexHull, QhullError

import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter
from scipy.spatial.transform import Rotation as SciRot
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from graspnetAPI import GraspNet



import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont





def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> List[float]:
    """
    Convert 3x3 rotation matrix to quaternion in [w, x, y, z] convention.
    SciPy uses [x, y, z, w]; we reorder.
    """
    r = SciRot.from_matrix(R)
    q_xyzw = r.as_quat()  # [x,y,z,w]
    x, y, z, w = q_xyzw.tolist()
    return [float(w), float(x), float(y), float(z)]

# =========================

# Dataset-wide configuration

# =========================

@dataclass
class DatasetConfig:
    rgb_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    rgb_std: List[float]  = field(default_factory=lambda: [0.229, 0.224, 0.225])
    out_hw: Tuple[int, int] = (490, 644)     # (out_h, out_w)
    pc_num_points: int = 768


# =========================

# Utilities

# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def resize_depth_nearest(depth_np: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if depth_np is None:
        return None
    # Keep dtype, no smoothing
    return np.array(Image.fromarray(depth_np).resize(size, resample=Image.NEAREST), dtype=depth_np.dtype)

def resize_mask_nearest(mask_np: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask_np is None:
        return None
    return np.array(Image.fromarray(mask_np.astype(np.uint8)).resize(size, resample=Image.NEAREST), dtype=np.uint8)



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
    return TF.to_tensor(img_pil)  # C,H,W float [0,1]

def normalize_rgb(img_t: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    return TF.normalize(img_t, mean=mean, std=std)

def denorm_to_pil(img_t: torch.Tensor, mean: List[float], std: List[float]) -> Image.Image:
    m = torch.tensor(mean, dtype=img_t.dtype, device=img_t.device)[:, None, None]
    s = torch.tensor(std, dtype=img_t.dtype, device=img_t.device)[:, None, None]
    img = (img_t * s) + m
    img = img.clamp(0.0, 1.0)
    return TF.to_pil_image(img)

def gaussian_noise(t: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(t) * sigma
    return t + noise

def gaussian_noise_depth(depth_m: torch.Tensor, sigma_m: float = 0.003) -> torch.Tensor:
    if depth_m is None:
        return None
    noise = torch.normal(0, sigma_m, size=depth_m.shape)
    mask = depth_m > 0
    depth_m[mask] += noise[mask]
    #depth_m = torch.clamp(depth_m, min=0.001)
    return depth_m

def solarize_pil(img: Image.Image, threshold: int = 128) -> Image.Image:
    arr = np.array(img)
    inv = np.where(arr < threshold, arr, 255 - arr)
    return Image.fromarray(inv.astype(np.uint8))

def channel_swap(img_t: torch.Tensor) -> torch.Tensor:
    C = img_t.shape[0]
    perm = torch.randperm(C)
    return img_t[perm, :, :]

def get_color_palette(n: int) -> List[Tuple[int,int,int]]:
    rng = np.random.RandomState(123)
    colors = []
    for i in range(n):
        c = rng.randint(0, 255, size=3).tolist()
        colors.append(tuple(c))
    return colors

def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
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


# --- GraspIt camera helpers (USD/OpenGL → OpenCV) ---

def intr_from_graspit_camera_params(cam: Dict[str, Any]) -> Optional[List[float]]:
    """
    Prefer USD cameraProjection (4x4, column-major). Fallback to aperture/focal if missing.
    Returns [cx, cy, fx, fy].
    """
    try:
        W, H = cam['renderProductResolution']
        P_list = cam.get('cameraProjection', None)
        if P_list is not None and len(P_list) == 16:
            P = np.array(P_list, dtype=np.float32).reshape(4, 4, order='F')
            fx = (W * float(P[0, 0])) / 2.0
            fy = (H * float(P[1, 1])) / 2.0
            cx = (W * (1.0 + float(P[0, 2]))) / 2.0
            cy = (H * (1.0 + float(P[1, 2]))) / 2.0
            return [float(cx), float(cy), float(fx), float(fy)]
        # Fallback (older scenes)
        apx, apy = cam['cameraAperture']
        f_mm = cam['cameraFocalLength']
        fx = (f_mm / apx) * W
        fy = (f_mm / apy) * H
        cx = W / 2.0
        cy = H / 2.0
        return [float(cx), float(cy), float(fx), float(fy)]
    except Exception:
        return None

def load_camera_view_matrix(cam: Dict[str, Any]) -> np.ndarray:
    """USD cameraViewTransform is 4x4 column-major."""
    v = cam.get('cameraViewTransform', None)
    if v is None or len(v) != 16:
        return np.eye(4, dtype=np.float32)
    return np.array(v, dtype=np.float32).reshape(4, 4, order='F')

def rotation_x_pi() -> np.ndarray:
    """Rx(pi) = diag(1, -1, -1)"""
    return np.array([[1, 0, 0],
                     [0,-1, 0],
                     [0, 0,-1]], dtype=np.float32)

def random_rotation_matrix(max_angle_deg: float = 30.0) -> np.ndarray:
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
    # Use exclusive range differences (no +1)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / W, cy / H, w / W, h / H

def project_points(points_m: np.ndarray, R: np.ndarray, t_m: np.ndarray, intr: List[float]) -> np.ndarray:
    cx, cy, fx, fy = intr
    pts = (R @ points_m.T).T + t_m.reshape(1,3)
    Z = pts[:,2:3]
    valid = Z[:,0] > 1e-6
    u = fx * (pts[:,0]/Z[:,0]) + cx
    v = fy * (pts[:,1]/Z[:,0]) + cy
    uv = np.stack([u, v, valid], axis=1)
    return uv  # N x 3


# =========================

# Image geometry helpers

# =========================

def letterbox_coherent(img_pil: Image.Image,
                       depth_np: Optional[np.ndarray],
                       mask_np: Optional[np.ndarray],
                       out_hw: Tuple[int,int],
                       pad_color=(0, 0, 0)):
    """
    Keep aspect ratio: scale by s=min(OUT_W/W0, OUT_H/H0), paste centered into (OUT_H,OUT_W).


    - RGB: pad_color
    - Depth/mask: zeros

    Returns: (img_pad, depth_pad, mask_pad, meta)
    """
    out_h, out_w = out_hw
    W0, H0 = img_pil.size

    s = min(out_w / float(W0), out_h / float(H0))
    new_w = int(round(W0 * s))
    new_h = int(round(H0 * s))

    pad_left = (out_w - new_w) // 2
    pad_right = out_w - new_w - pad_left
    pad_top = (out_h - new_h) // 2
    pad_bottom = out_h - new_h - pad_top

    # RGB
    img_res = img_pil.resize((new_w, new_h), Image.BILINEAR)
    img_pad = Image.new('RGB', (out_w, out_h), pad_color)
    img_pad.paste(img_res, (pad_left, pad_top))

    # Depth
    depth_pad = None
    if depth_np is not None:
        depth_res = resize_depth_nearest(depth_np, (new_w, new_h))
        depth_pad = np.zeros((out_h, out_w), dtype=depth_np.dtype)
        depth_pad[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = depth_res
        
    
    # Mask
    mask_pad = None
    if mask_np is not None:
        mask_res = resize_mask_nearest(mask_np, (new_w, new_h))
        mask_pad = np.zeros((out_h, out_w), dtype=np.uint8)
        mask_pad[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = mask_res

    meta = {
        'scale': s,
        'pad_left': pad_left, 'pad_right': pad_right,
        'pad_top': pad_top, 'pad_bottom': pad_bottom,
        'new_w': new_w, 'new_h': new_h,
        'out_w': out_w, 'out_h': out_h
    }
    return img_pad, depth_pad, mask_pad, meta


def adjust_intr_for_letterbox(intr: Optional[List[float]], lb_meta: Dict[str, Any]) -> Optional[List[float]]:
    """
    Adjust [cx, cy, fx, fy] for keep-aspect letterbox.


    - Scale by s
    - Shift cx,cy by padding

    """
    if intr is None:
        return None
    cx, cy, fx, fy = intr
    s = float(lb_meta['scale'])
    pl = int(lb_meta['pad_left'])
    pt = int(lb_meta['pad_top'])
    cx2 = cx * s + pl
    cy2 = cy * s + pt
    fx2 = fx * s
    fy2 = fy * s
    return [float(cx2), float(cy2), float(fx2), float(fy2)]

def resize_coherent(img_pil: Image.Image, depth_np: Optional[np.ndarray], mask_np: Optional[np.ndarray], out_hw: Tuple[int,int]):
    out_h, out_w = out_hw
    img_res = img_pil.resize((out_w, out_h), Image.BILINEAR)  # RGB can stay bilinear
    depth_res = resize_depth_nearest(depth_np, (out_w, out_h)) if depth_np is not None else None
    mask_res  = resize_mask_nearest(mask_np, (out_w, out_h)) if mask_np is not None else None
    return img_res, depth_res, mask_res


def centroid(points: np.ndarray):
    points = np.asarray(points, dtype=np.float64)
    points = points[~np.isnan(points).any(axis=1)]
    if len(points) == 0:
        raise ValueError("No valid points")
    return points.mean(axis=0)

def occupancy_percent_in_cube(points: np.ndarray, resolution_edge_fraction: float = 0.01):
    """
    points: (N, 3) numpy array, coordinates assumed in [-1, 1] (some may be outside).
    resolution_edge_fraction: fraction of the cube edge length; 0.01 -> voxel_size = 0.02.
    
    Returns: (percent, occupied_voxels, total_voxels)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")

    min_bound = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    max_bound = np.array([ 1.0,  1.0,  1.0], dtype=np.float64)
    edge_length = 2.0
    voxel_size = edge_length * resolution_edge_fraction
    n = int(round(edge_length / voxel_size))
    total_voxels = n ** 3

    mask = np.all((points >= min_bound) & (points < max_bound), axis=1)
    pts_in = points[mask]
    if pts_in.size == 0:
        return 0.0, 0, total_voxels

    indices = np.floor((pts_in - min_bound) / voxel_size).astype(np.int32)
    lin_idx = indices[:, 0] + n * indices[:, 1] + (n * n) * indices[:, 2]
    occupied_voxels = np.unique(lin_idx).size
    percent = (occupied_voxels / total_voxels) * 100.0
    return percent, occupied_voxels, total_voxels

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


# --------- Point cloud I/O and voxel downsampling utilities ---------

def write_obj_point_cloud(path: str, pts: np.ndarray):
    """Write point cloud as minimal OBJ with only vertex lines."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        for p in pts:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

def read_obj_point_cloud(path: str) -> Optional[np.ndarray]:
    """Read point cloud from OBJ (only 'v' lines)."""
    pts = []
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        _, x, y, z = parts[:4]
                        pts.append([float(x), float(y), float(z)])
        return np.array(pts, dtype=np.float32) if len(pts) > 0 else None
    except Exception as e:
        print(f"[read_obj_point_cloud] Failed for {path}: {e}")
        return None

def voxel_downsample_to_target(pts: np.ndarray, target: int, max_iter: int = 12, tol: float = 0.10) -> np.ndarray:
    """Downsample via Open3D voxel grid to ~target count using only voxel subsampling."""
    if pts.shape[0] <= target:
        return pts.copy()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    min_b = pcd.get_min_bound()
    max_b = pcd.get_max_bound()
    extents = max_b - min_b
    volume = float(np.prod(extents)) if np.all(extents > 1e-9) else 1.0
    voxel_size = (volume / max(target, 1)) ** (1.0/3.0)
    voxel_size = max(voxel_size, 1e-6)
    low, high = voxel_size * 0.25, voxel_size * 4.0

    best = pts
    best_diff = float('inf')
    for _ in range(max_iter):
        ds = pcd.voxel_down_sample(voxel_size)
        cnt = len(ds.points)
        if cnt > 0:
            diff = abs(cnt - target)
            if diff < best_diff:
                best = np.asarray(ds.points, dtype=np.float32)
                best_diff = diff
            if diff / target <= tol:
                best = np.asarray(ds.points, dtype=np.float32)
                break
        if cnt == 0:
            voxel_size *= 0.5
        else:
            ratio = (cnt / float(target)) ** (1.0/3.0)
            voxel_size *= ratio
        voxel_size = float(np.clip(voxel_size, low, high))
    return best

def ensure_downsampled_mesh_to_pc(mesh_path: str, out_pc_path: str, num_points: int) -> Optional[str]:
    """Downsample mesh (or point cloud) to ~num_points using voxel grid, write to out_pc_path (.ply or .obj)."""
    try:
        if os.path.exists(out_pc_path):
            return out_pc_path
        ensure_dir(os.path.dirname(out_pc_path))

        pts_raw = None
        ext = os.path.splitext(mesh_path)[1].lower()
        mesh = None
        if ext in ['.obj', '.ply', '.stl']:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh is None or (not mesh.has_vertices() and not mesh.has_triangles()):
            pcd = o3d.io.read_point_cloud(mesh_path)
            if pcd is None or not pcd.has_points():
                return None
            pts_raw = np.asarray(pcd.points, dtype=np.float32)
        else:
            mesh.compute_vertex_normals()
            if mesh.has_triangles():
                pcd = mesh.sample_points_poisson_disk(number_of_points=max(num_points * 4, num_points * 2))
                pts_raw = np.asarray(pcd.points, dtype=np.float32)
            else:
                pts_raw = np.asarray(mesh.vertices, dtype=np.float32)

        if pts_raw.shape[0] == 0:
            return None

        pts_ds = voxel_downsample_to_target(pts_raw, num_points)

        out_ext = os.path.splitext(out_pc_path)[1].lower()
        if out_ext == '.obj':
            write_obj_point_cloud(out_pc_path, pts_ds)
        else:
            pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_ds))
            o3d.io.write_point_cloud(out_pc_path, pcd_out, write_ascii=True)
        return out_pc_path
    except Exception as e:
        print(f"[ensure_downsampled_mesh_to_pc] Failed for {mesh_path}: {e}")
        return None

def has_pc_suffix(name: str) -> bool:
    """Return True if name has a trailing _pc<digits> suffix."""
    return re.search(r'_pc\d+$', name) is not None

def strip_pc_suffix(name: str) -> str:
    """Remove one or more trailing _pc<digits> suffixes."""
    return re.sub(r'(_pc\d+)+$', '', name)

def ensure_downsampled_model_dir(src_dir: str, out_dir: str, pattern_mesh=('*.ply','*.obj','*.stl'), num_points: int = 768):
    """Create downsampled point clouds for meshes in src_dir into out_dir without duplicating _pc suffixes."""
    ensure_dir(out_dir)
    meshes = []
    for pat in pattern_mesh:
        meshes.extend(glob.glob(os.path.join(src_dir, pat)))

    processed = set()
    for mp in sorted(meshes):
        base = os.path.splitext(os.path.basename(mp))[0]
        if has_pc_suffix(base):
            continue
        base_clean = strip_pc_suffix(base)
        if base_clean in processed:
            continue
        processed.add(base_clean)
        out_pc = os.path.join(out_dir, f"{base_clean}_pc{num_points}.ply")
        ensure_downsampled_mesh_to_pc(mp, out_pc, num_points=num_points)

def normalize_pointcloud(pc_xyz: np.ndarray):
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

def load_pointcloud_file(pc_path: str, pc_num_points: int) -> Optional[np.ndarray]:
    """Load a point cloud. If too dense, voxel-downsample to pc_num_points and write next to source."""
    try:
        ext = os.path.splitext(pc_path)[1].lower()
        pts = None
        if ext == '.obj':
            pts = read_obj_point_cloud(pc_path)
        else:
            pcd = o3d.io.read_point_cloud(pc_path)
            if pcd is None or not pcd.has_points():
                return None
            pts = np.asarray(pcd.points, dtype=np.float32)
        if pts is None or pts.shape[0] == 0:
            return None

        if pts.shape[0] > pc_num_points:
            pts_ds = voxel_downsample_to_target(pts, pc_num_points)
            base = os.path.splitext(os.path.basename(pc_path))[0]
            out_obj = os.path.join(os.path.dirname(pc_path), f"{base}_{pc_num_points}.obj")
            write_obj_point_cloud(out_obj, pts_ds)
            pts = pts_ds
        return pts
    except Exception as e:
        print(f"[load_pointcloud_file] Failed for {pc_path}: {e}")
        return None

def build_dummy_sphere_pc(out_dir: str, name: str, num_points: int) -> str:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{name}_pc{num_points}.ply")
    if os.path.exists(out_path):
        return out_path
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

def draw_boxes_on_rgb(img: Image.Image,
                      boxes_xyxy: np.ndarray,
                      palette: Optional[List[Tuple[int,int,int]]] = None,
                      labels: Optional[List[str]] = None,
                      text_color: Tuple[int,int,int] = (255, 255, 255)) -> Image.Image:
    
    out = img.copy()
    draw = ImageDraw.Draw(out)
    n = boxes_xyxy.shape[0]
    if palette is None:
        palette = get_color_palette(n)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i in range(n):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        col = palette[i % len(palette)]
        draw.rectangle([(x1, y1), (x2, y2)], outline=tuple(col), width=2)

        if labels is not None and i < len(labels) and labels[i] is not None:
            text = str(labels[i])
            tx = int(x1) + 3
            ty = int(max(0, y1 - 12))
            try:
                draw.text((tx, ty), text, fill=text_color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
            except TypeError:
                draw.text((tx, ty), text, fill=text_color, font=font)
    return out

def _o3d_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float32)))


def print_mask_to_model_distance_stats(sample: dict, max_points_per_inst: Optional[int] = None) -> Dict[int, Dict[str, float]]:
    """
    For each object instance in the segmentation mask, backproject its RGB-D pixels to 3D
    and compute distances to the closest posed model point (in camera coords).
    Prints per-object [count, min, mean, max] in meters, and totals.

    Returns a dict: {inst_id: {'n': int, 'min': float, 'mean': float, 'max': float}, 'overall': {...}}
    """
    try:
        intr = sample.get('intr')
        depth_t = sample.get('depth')
        mask_t = sample.get('mask')
        props = sample.get('props', {})
        models = sample.get('models', {})
        poses = sample.get('poses', {}) 
        
        # intr as list [cx, cy, fx, fy]
        intr_list = intr.detach().cpu().tolist() if isinstance(intr, torch.Tensor) else intr
        cx, cy, fx, fy = [float(x) for x in intr_list]

        # Depth and mask aligned to output size
        depth_np = depth_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        mask_np = mask_t.detach().cpu().numpy().astype(np.int32)
        H, W = mask_np.shape

        inst_ids = sorted(list(poses.keys()))
        stats = {}
        all_dists = []
        for inst_id in inst_ids:
            vm = props.get(inst_id)
            if vm is None:
                print(f'[props-model] inst_id {inst_id}: no props')
            pn = vm['pc_norm_cam']; pos = vm['pos_m']; sc = float(vm.get('scale_m', 1.0))
            pn = pn.detach().cpu().numpy() if isinstance(pn, torch.Tensor) else np.asarray(pn)
            pos = pos.detach().cpu().numpy() if isinstance(pos, torch.Tensor) else np.asarray(pos, dtype=np.float32)
            pts_cam = pn * sc + pos
            model_pts_cam = pts_cam.astype(np.float32)      

            if model_pts_cam is None or model_pts_cam.shape[0] == 0:
                print(f"[props-model] inst_id {inst_id}: no posed model points.")
                continue

            # Gather mask pixels and depths for this instance
            ys, xs = np.where(mask_np == inst_id)
            if xs.size == 0:
                print(f"[props-model] inst_id {inst_id}: no mask pixels.")
                continue
            d = depth_np[ys, xs]
            m = d > 0
            if not np.any(m):
                print(f"[props-model] inst_id {inst_id}: no valid depth.")
                continue
            u = xs[m].astype(np.float32)
            v = ys[m].astype(np.float32)
            d = d[m].astype(np.float32)

            # Optional subsampling
            if (max_points_per_inst is not None) and (u.size > max_points_per_inst):
                sel = np.random.choice(u.size, size=int(max_points_per_inst), replace=False)
                u = u[sel]; v = v[sel]; d = d[sel]

            # Backproject to camera 3D
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d
            pts_mask_cam = np.stack([X, Y, Z], axis=1).astype(np.float32)

            # NN distances
            if model_pts_cam.shape[0] == 0 or pts_mask_cam.shape[0] == 0:
                print(f"[props-model] inst_id {inst_id}: insufficient points.")
                continue
            tree = cKDTree(model_pts_cam)
            dist, _ = tree.query(pts_mask_cam, k=1)
            if dist.size == 0:
                print(f"[props-model] inst_id {inst_id}: no distances computed.")
                continue

            dmin = float(dist.min()); dmean = float(dist.mean()); dmax = float(dist.max())
            n = int(dist.size)
            stats[inst_id] = {'n': n, 'min': dmin*1000, 'mean': dmean*1000, 'max': dmax*1000}
            all_dists.append(dist)

            print(f"[props-model] inst_id={inst_id} | n={n} pts | min={dmin*1000:.2f} mm, mean={dmean*1000:.2f} mm, max={dmax*1000:.2f} mm")

        if len(all_dists) > 0:
            ad = np.concatenate(all_dists, axis=0)
            overall = {'n': int(ad.size), 'min': float(ad.min())*1000, 'mean': float(ad.mean())*1000, 'max': float(ad.max())*1000}
            stats['overall'] = overall
            print(f"[props-model] OVERALL | n={overall['n']} pts | min={overall['min']:.2f} mm, mean={overall['mean']:.2f} mm, max={overall['max']:.2f} mm")
        else:
            print("[props-model] OVERALL | no distances.")
    
        stats = {}
        all_dists = []
        for inst_id in inst_ids:
            p = poses.get(inst_id, None)
            model_pts_cam = models.get(inst_id, None)

            if p is None or model_pts_cam is None:
                print(f"[pose-model] inst_id {inst_id}: no posed model points or pose.")
                continue

            R = p['R']; t = p['t']
            R = R.detach().cpu().numpy().astype(np.float32) if isinstance(R, torch.Tensor) else np.asarray(R, dtype=np.float32)
            t = t.detach().cpu().numpy().astype(np.float32) if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)

            model_pts_cam = model_pts_cam.detach().cpu().numpy().astype(np.float32)

            # get the real object pose in the camera frame coords
            model_pts_cam = (R @ model_pts_cam.T).T + t.reshape(1, 3)  # meters in camera

            # Gather mask pixels and depths for this instance
            ys, xs = np.where(mask_np == inst_id)
            if xs.size == 0:
                print(f"[pose-model] inst_id {inst_id}: no mask pixels.")
                continue
            d = depth_np[ys, xs]
            m = d > 0
            if not np.any(m):
                print(f"[pose-model] inst_id {inst_id}: no valid depth.")
                continue
            u = xs[m].astype(np.float32)
            v = ys[m].astype(np.float32)
            d = d[m].astype(np.float32)

            # Optional subsampling
            if (max_points_per_inst is not None) and (u.size > max_points_per_inst):
                sel = np.random.choice(u.size, size=int(max_points_per_inst), replace=False)
                u = u[sel]; v = v[sel]; d = d[sel]

            # Backproject to camera 3D
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d
            pts_mask_cam = np.stack([X, Y, Z], axis=1).astype(np.float32)

            # NN distances
            if model_pts_cam.shape[0] == 0 or pts_mask_cam.shape[0] == 0:
                print(f"[pose-model] inst_id {inst_id}: insufficient points.")
                continue
            tree = cKDTree(model_pts_cam)
            dist, _ = tree.query(pts_mask_cam, k=1)
            if dist.size == 0:
                print(f"[pose-model] inst_id {inst_id}: no distances computed.")
                continue

            dmin = float(dist.min()); dmean = float(dist.mean()); dmax = float(dist.max())
            n = int(dist.size)
            stats[inst_id] = {'n': n, 'min': dmin*1000, 'mean': dmean*1000, 'max': dmax*1000}
            all_dists.append(dist)

            print(f"[pose-model] inst_id={inst_id} | n={n} pts | min={dmin*1000:.2f} mm, mean={dmean*1000:.2f} mm, max={dmax*1000:.2f} mm")

        if len(all_dists) > 0:
            ad = np.concatenate(all_dists, axis=0)
            overall = {'n': int(ad.size), 'min': float(ad.min())*1000, 'mean': float(ad.mean())*1000, 'max': float(ad.max())*1000}
            stats['overall'] = overall
            print(f"[pose-model] OVERALL | n={overall['n']} pts | min={overall['min']:.2f} mm, mean={overall['mean']:.2f} mm, max={overall['max']:.2f} mm")
        else:
            print("[pose-model] OVERALL | no distances.")


    except Exception as e:
        #raise e
        print(f"[mask-model] Failed stats computation: {e}")
        return {}
        
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

def adjust_intr_for_resize(intr: Optional[List[float]], orig_size: Tuple[int,int], new_size: Tuple[int,int]) -> Optional[List[float]]:
    if intr is None:
        return None
    cx, cy, fx, fy = intr
    W0, H0 = orig_size
    W1, H1 = new_size
    sx = W1 / float(W0)
    sy = H1 / float(H0)
    cx2 = cx * sx
    cy2 = cy * sy
    fx2 = fx * sx
    fy2 = fy * sy
    return [float(cx2), float(cy2), float(fx2), float(fy2)]

def clamp_zero(depth, min_d=0.1, max_d=15.0):
    depth[depth < min_d] = 0
    depth[depth > max_d] = 0
    return depth
    

def _draw_gripper(
        img_pil, grasp, intr, finger_len_m: float = 0.05, default_width_m: float = 0.05,
        line_w: int = 3,
        color_fingers=(0, 255, 0), color_link=(255, 255, 255), color_dir=(255, 0, 0),
        label_text: str | None = None, label_color=(255, 255, 255)):
    
    cx, cy, fx, fy = [float(x) for x in (intr if not isinstance(intr, torch.Tensor) else intr.detach().cpu().tolist())]
    w, x, y, z = [float(q) for q in grasp.get('quat_wxyz', [1, 0, 0, 0])]
    center = np.asarray(grasp.get('xyz', [0, 0, 0]), dtype=np.float32)
    width_m = float(grasp.get('width', default_width_m)) or default_width_m

    try:
        R = SciRot.from_quat([x, y, z, w]).as_matrix().astype(np.float32)
    except Exception:
        R = np.eye(3, dtype=np.float32)

    # width along local Y, approach along local Z
    Y = R[:, 1]
    Z = R[:, 2]
    half_w = 0.5 * width_m
    L = float(finger_len_m)

    left_base  = center - half_w * Y
    right_base = center + half_w * Y
    left_tip   = left_base  + L * Z
    right_tip  = right_base + L * Z
    dir_end    = center - L * Z

    def _proj(P):
        Xc, Yc, Zc = float(P[0]), float(P[1]), float(P[2])
        if Zc <= 1e-6: return None
        u = fx * (Xc / Zc) + cx
        v = fy * (Yc / Zc) + cy
        return (int(round(u)), int(round(v)))

    p_lb = _proj(left_base);  p_lt = _proj(left_tip)
    p_rb = _proj(right_base); p_rt = _proj(right_tip)
    p_c  = _proj(center);     p_de = _proj(dir_end)

    out = img_pil.copy()
    draw = ImageDraw.Draw(out)

    if p_lb and p_lt: draw.line([p_lb, p_lt], fill=color_fingers, width=line_w)
    if p_rb and p_rt: draw.line([p_rb, p_rt], fill=color_fingers, width=line_w)
    if p_lb and p_rb: draw.line([p_lb, p_rb], fill=color_link, width=line_w)

    if p_c and p_de:
        draw.line([p_c, p_de], fill=color_dir, width=line_w)
        dx, dy = (p_de[0] - p_c[0]), (p_de[1] - p_c[1])
        norm = (dx*dx + dy*dy) ** 0.5
        if norm > 1e-3:
            ux, uy = dx / norm, dy / norm
            px, py = -uy, ux
            ah = 6
            left_ah  = (int(round(p_de[0] - ah*ux + 0.5*ah*px)), int(round(p_de[1] - ah*uy + 0.5*ah*py)))
            right_ah = (int(round(p_de[0] - ah*ux - 0.5*ah*px)), int(round(p_de[1] - ah*uy - 0.5*ah*py)))
            draw.line([p_de, left_ah],  fill=color_dir, width=line_w)
            draw.line([p_de, right_ah], fill=color_dir, width=line_w)

    if p_c:
        r = 2
        draw.ellipse([(p_c[0]-r, p_c[1]-r), (p_c[0]+r, p_c[1]+r)], fill=(255, 255, 255))

    if label_text is None and 'score' in grasp:
        label_text = f"{float(grasp.get('score', 0.0)):.2f}"
    if label_text and p_c:
        pos = (p_c[0] + 6, p_c[1] - 12)
        try:
            draw.text(pos, label_text, fill=label_color, stroke_width=2, stroke_fill=(0, 0, 0))
        except TypeError:
            draw.text(pos, label_text, fill=label_color)

    return out


# =========================

# Subsample helpers

# =========================

@torch.no_grad()
def inverse_density_indices(points: torch.Tensor, m: int = 512, k: int = 16) -> torch.Tensor:
    """
    Prefer points in sparse regions using kNN mean distance as inverse-density weight.
    points: [N,3] float tensor (CPU/GPU). Returns indices [min(N,m)].
    """
    assert points.ndim == 2 and points.size(1) == 3
    N = points.size(0)
    if N <= m:
        return torch.arange(N, device=points.device)

    D = torch.cdist(points, points)  # [N,N]
    D[torch.arange(N), torch.arange(N)] = float('inf')
    knn = D.topk(k=min(k, N-1), largest=False).values  # [N,k]
    mean_d = knn.mean(dim=1) + 1e-12
    probs = mean_d / mean_d.sum()
    idx = torch.multinomial(probs, num_samples=m, replacement=False)
    return idx

def inverse_density_subsample_np(pts_np: np.ndarray, m: int = 512, k: int = 16) -> np.ndarray:
    """
    NumPy wrapper. Returns up to m points with even coverage; if N <= m returns input.
    """
    if pts_np is None or pts_np.ndim != 2 or pts_np.shape[1] != 3:
        return pts_np
    N = pts_np.shape[0]
    if N <= m:
        return pts_np
    pts_t = torch.from_numpy(pts_np.astype(np.float32))
    idx = inverse_density_indices(pts_t, m=m, k=k).cpu().numpy()
    return pts_np[idx]


@contextlib.contextmanager
def suppress_stderr():
    fd = os.dup(2)
    try:
        with open(os.devnull, 'w') as f:
            os.dup2(f.fileno(), 2)
        yield
    finally:
        os.dup2(fd, 2)
        os.close(fd)

def _is_near_degenerate(pc, tol=1e-4):
    if pc is None or pc.shape[0] < 4:
        return True
    c = pc.mean(axis=0)
    u, s, vh = np.linalg.svd(pc - c, full_matrices=False)
    if s[0] <= 1e-12:
        return True
    return (s[-1] / s[0]) < tol  # near-planar/line

def convex_hull_volume_fraction(pc_norm_cam: np.ndarray) -> float:
    """
    pc_norm_cam in [-1,1]^3. Returns fraction in [0,1] of cube volume ‘inside’ the hull.
    Skips near-degenerate sets to avoid Qhull precision warnings.
    """
    if _is_near_degenerate(pc_norm_cam):
        return 0.0

    # Prefer SciPy (supports qhull_options)
    try:
        try:
            hull = ConvexHull(pc_norm_cam, qhull_options='QJ Pp')  # joggle + prune printing
            vol = float(hull.volume)
        except QhullError:
            return 0.0
        return float(np.clip(vol / 8.0, 0.0, 1.0))  # cube [-1,1]^3 has volume 8
    except Exception:
        pass

    # Fallback: Open3D with tiny jitter + stderr suppression
    try:
        
        pcj = pc_norm_cam + np.random.normal(scale=1e-6, size=pc_norm_cam.shape)
        with suppress_stderr():
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcj.astype(np.float32)))
            hull, _ = pcd.compute_convex_hull()
        verts = np.asarray(hull.vertices)
        tris  = np.asarray(hull.triangles)
        if verts.shape[0] < 4 or tris.shape[0] == 0:
            return 0.0
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        vol = np.abs(np.einsum('ij,ij->i', np.cross(v0, v1), v2)).sum() / 6.0
        return float(np.clip(vol / 8.0, 0.0, 1.0))
    except Exception:
        return 0.0

def gripper_mesh_thick(center, R, width_m, L_m,
                       finger_thickness_m=0.004,
                       link_thickness_m=0.002,
                       dir_len_m=None,
                       color_rgb=(1.0, 0.2, 0.2)):
    """
    Build a thick 3D gripper as TriangleMeshes.
    Convention:
      - width along local Y
      - approach along local Z
      - binormal along local X

    Geometry:
      - two finger boxes along Z
      - a base/link box along Y
      - optional direction bar along -Z

    """
    center = np.asarray(center, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    X = R[:, 0]; Y = R[:, 1]; Z = R[:, 2]

    half_w = 0.5 * float(width_m)
    L = float(L_m)
    dir_len = float(dir_len_m if dir_len_m is not None else L * 0.6)

    def _oriented_box(dim_x, dim_y, dim_z, R_world, center_world, color):
        box = o3d.geometry.TriangleMesh.create_box(width=dim_x, height=dim_y, depth=dim_z)
        # Move local center to origin, then rotate, then translate to target center
        local_center = np.array([dim_x/2.0, dim_y/2.0, dim_z/2.0], dtype=np.float32)
        box.translate(-local_center)
        box.rotate(R_world, center=(0,0,0))
        box.translate(center_world)
        box.paint_uniform_color(color)
        return box

    geoms = []
    col = np.array(color_rgb, dtype=np.float32)

    # Fingers: each finger is a box with dims [X=finger_thickness, Y=finger_thickness, Z=L]
    # centers at base + L/2 * Z
    left_base  = center - half_w * Y
    right_base = center + half_w * Y
    left_ctr   = left_base  + 0.5 * L * Z
    right_ctr  = right_base + 0.5 * L * Z

    finger_R = np.stack([X, Y, Z], axis=1).astype(np.float32)  # local axes map to world axes
    geoms.append(_oriented_box(finger_thickness_m, finger_thickness_m, L, finger_R, left_ctr,  col))
    geoms.append(_oriented_box(finger_thickness_m, finger_thickness_m, L, finger_R, right_ctr, col))

    # Base/link: spans between finger bases along Y
    # dims [X=link_thickness, Y=width_m, Z=link_thickness]
    link_dims = (max(link_thickness_m, 1e-4), float(width_m), max(link_thickness_m, 1e-4))
    link_R = np.stack([X, Y, Z], axis=1).astype(np.float32)
    geoms.append(_oriented_box(*link_dims, link_R, center, col))

    # Direction bar (optional): along -Z from center
    # dims [X=link_thickness, Y=link_thickness, Z=dir_len]
    dir_ctr = center - 0.5 * dir_len * Z
    geoms.append(_oriented_box(link_thickness_m, link_thickness_m, dir_len, link_R, dir_ctr, (1.0, 0.0, 0.0)))

    return geoms

def gripper_lineset(center, R, width_m, L, color_rgb=(1.0, 0.0, 0.0)):
    # width along Y, approach along Z
    Y = R[:, 1]; Z = R[:, 2]
    half_w = 0.5 * float(width_m)
    center = center.astype(np.float32)

    left_base  = center - half_w * Y
    right_base = center + half_w * Y
    left_tip   = left_base  + L * Z
    right_tip  = right_base + L * Z
    dir_end    = center - L * Z

    pts = np.stack([left_base, left_tip, right_base, right_tip, center, dir_end], axis=0).astype(np.float32)
    lines = np.array([[0,1],[2,3],[0,2],[4,5]], dtype=np.int32)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    lc = np.tile(np.asarray(color_rgb, dtype=np.float32).reshape(1,3), (lines.shape[0], 1))
    ls.colors = o3d.utility.Vector3dVector(lc)
    return ls


# =========================

# Base Dataset

# =========================

class BasePoseDataset(data.Dataset):
    def __init__(self, mode='train', transforms=True, rgb_noise_sigma=0.0, depth_noise_sigma_m=0.003,
                 do_affine=False, do_color=True, rectify_prob: float = 0.0,
                 min_visible_pixels: int = 50, len_target_pc: int = 512,
                 cfg: Optional[DatasetConfig] = None):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.apply_transforms = transforms
        self.rgb_noise_sigma = rgb_noise_sigma
        self.depth_noise_sigma_m = depth_noise_sigma_m
        self.do_affine = do_affine
        self.do_color = do_color
        
        self.rectify_prob = float(rectify_prob)
        self.min_visible_pixels = int(min_visible_pixels)
        self.len_target_pc = len_target_pc
        self.cfg = cfg or DatasetConfig()

    
    def _build_out_models_pose_baked(self, poses_dict, models_dict):
        """
        poses_dict: { inst_id: {pose keys and values}}
        models_dict: { inst_id: Tensor/ndarray [N,3] } where points are raw in meters AND scene-scaled already.
        Returns:
        props: { inst_id: {
            'pc_norm_cam': FloatTensor [M,3],   # centered & normalized in camera coords
            'pos_m':       FloatTensor [3],      # centroid in camera meters
            'scale_m':     float,                # normalization scale (meters)
            'cog':         FloatTensor [3],      # centroid of normalized shape (≈0,0,0)
        } }
        """
        out = {}
        if not poses_dict or not models_dict:
            return out

        for inst_id, p in poses_dict.items():
            if inst_id <= 0:
                continue

            pts_m = models_dict.get(inst_id, None)
            if pts_m is None:
                continue

            # Nx3 to numpy
            if isinstance(pts_m, torch.Tensor):
                pts_m_np = pts_m.detach().cpu().numpy().astype(np.float32)
            else:
                pts_m_np = np.asarray(pts_m, dtype=np.float32)
            if pts_m_np.ndim != 2 or pts_m_np.shape[1] != 3 or pts_m_np.shape[0] == 0:
                continue

            # Pose to camera
            R = p['R']; t = p['t']
            R = R.detach().cpu().numpy().astype(np.float32) if isinstance(R, torch.Tensor) else np.asarray(R, dtype=np.float32)
            t = t.detach().cpu().numpy().astype(np.float32) if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)

            # get the real object pose in the camera frame coords
            pts_cam = (R @ pts_m_np.T).T + t.reshape(1, 3)  # meters in camera

            # downsample
            pts_cam = inverse_density_subsample_np(pts_cam, m=self.len_target_pc, k=16)

            # size the real center of gravity is the centroid 0,0,0 or pos_m, we instead get the cog weihted 2x the distance of each point. 
            alpha = 2.0  
            r = np.linalg.norm(pts_cam, axis=1) + 1e-9
            w = (r ** alpha).astype(np.float32)
            pos_m_weighted = (w[:, None] * pts_cam).sum(axis=0) / w.sum()
            pos_m_weighted = pos_m_weighted.astype(np.float32)

            # Center and normalize
            pos_m = pts_cam.mean(axis=0).astype(np.float32)  # centroid in meters
            pts_centered = (pts_cam - pos_m).astype(np.float32)
            s_norm = float(np.max(np.abs(pts_centered))) if pts_centered.size > 0 else 1.0
            if s_norm < 1e-12:
                s_norm = 1.0
            pc_norm_cam = (pts_centered / s_norm).astype(np.float32)

            ## downsample
            #pc_norm_cam = inverse_density_subsample_np(pc_norm_cam, m=self.len_target_pc, k=16)

            # get the volume the convex hull of the obj point cloud fills in the [-1:1] cube. hence 0-1. 
            bb_vol_frac = convex_hull_volume_fraction(pc_norm_cam)  # 0..1

            out[inst_id] = {
                'pc_norm_cam': torch.from_numpy(pc_norm_cam),
                'pos_m': torch.from_numpy(pos_m),
                'scale_m': float(s_norm),
                'cog': torch.from_numpy(pos_m_weighted)-torch.from_numpy(pos_m),
                'vol': float(bb_vol_frac),
            }

        return out

    def _filter_visible_instances(self, mask_np: Optional[np.ndarray],
                                  poses_dict: Optional) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Keep only poses whose inst_id appears in mask with >= min_visible_pixels; drop unused models."""
        poses_dict = poses_dict or {}
        if mask_np is None or mask_np.size == 0 or len(poses_dict) == 0:
            return poses_dict
        counts = np.bincount(mask_np.reshape(-1))
        visible_inst = []
        for inst_id in range(1, len(counts)):
            if counts[inst_id] >= self.min_visible_pixels:
                visible_inst.append(inst_id)

        if not visible_inst:
            return {}

        poses_f = {inst_id: poses_dict[inst_id] for inst_id in visible_inst}
        return poses_f

    def _apply_augs(self, img_pil, depth_np, mask_np, intr, orig_size, distortion=None):
        intr_cur = intr
    
        # Keep-aspect letterbox to cfg.out_hw
        img_pil, depth_np, mask_np, lb_meta = letterbox_coherent(img_pil, depth_np, mask_np, out_hw=self.cfg.out_hw)
        intr_resized = adjust_intr_for_letterbox(intr_cur, lb_meta) if intr_cur is not None else None

        # Color transforms
        if self.mode == 'train' and self.apply_transforms and self.do_color:
            if random.random() < 0.8:
                cj = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
                img_pil = cj(img_pil)
            if random.random() < 0.2:
                img_pil = ImageOps.grayscale(img_pil).convert('RGB')
            if random.random() < 0.1:
                img_pil = solarize_pil(img_pil, threshold=random.randint(64,192))

        # Channel swap
        img_t = to_tensor_image(img_pil)
        if self.mode == 'train' and self.apply_transforms and random.random() < 0.1:
            img_t = channel_swap(img_t)
            img_pil = TF.to_pil_image(img_t)

        # Convert rgb and normalize
        img_t = to_tensor_image(img_pil)
        img_t = normalize_rgb(img_t, self.cfg.rgb_mean, self.cfg.rgb_std)

        # Gaussian noise
        if self.mode == 'train' and self.apply_transforms and self.rgb_noise_sigma > 0.0:
            img_t = gaussian_noise(img_t, sigma=self.rgb_noise_sigma)

        depth_t = None
        if depth_np is not None:
            depth_t = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)
            if self.mode == 'train' and self.apply_transforms and self.depth_noise_sigma_m > 0.0:
                depth_t = gaussian_noise_depth(depth_t, sigma_m=self.depth_noise_sigma_m)

        mask_t = None
        if mask_np is not None:
            mask_t = torch.from_numpy(mask_np.astype(np.int64))

        return img_t, depth_t, mask_t, intr_resized

    
    def _build_targets(self, mask_np):
        if mask_np is None:
            return np.zeros((0,4), dtype=np.float32), np.zeros((0,4), dtype=np.float32)
        out_h, out_w = self.cfg.out_hw
        boxes_xyxy, boxes_cxcywh = build_detr_boxes_from_mask(mask_np, out_w, out_h)
        return boxes_xyxy, boxes_cxcywh

    def _norm_model_pc(self, pc_xyz: np.ndarray):
        pc_normed, center, scale = normalize_pointcloud(pc_xyz)
        return pc_normed, center, scale

    def visualize3d(self, idx: int, max_instances_overlay: int = 20, max_grasps_per_inst: int = 5,
                    voxel_size_rgbd: float = 0.001, finger_len_m: float = 0.05, show_props=False):
        import open3d as o3d

        # Get sample
        sample = self[idx]
        H, W = int(sample['H']), int(sample['W'])
        intr = sample.get('intr')
        depth_t = sample.get('depth')
        img_t = sample.get('img')
        props = sample.get('props', {}) 
        models = sample.get('models', {}) 
        poses = sample.get('poses', {}) 
        gt_gbi = sample.get('gt_grasps', {}) 

        print_mask_to_model_distance_stats(sample, max_points_per_inst=20000)
        
        geoms = []
        # 1) RGB-D point cloud (if depth + intr are available)
        if (intr is not None) and (depth_t is not None):
            intr_list = intr.detach().cpu().tolist() if isinstance(intr, torch.Tensor) else intr
            cx, cy, fx, fy = [float(x) for x in intr_list]

            # Color image: denormalize to uint8
            img_pil = denorm_to_pil(img_t, mean=self.cfg.rgb_mean, std=self.cfg.rgb_std)
            color_np = np.asarray(img_pil, dtype=np.uint8)
            color_o3d = o3d.geometry.Image(color_np)

            # Depth in meters -> pass depth_scale=1.0
            depth_np = depth_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            depth_np = np.ascontiguousarray(depth_np)
            depth_o3d = o3d.geometry.Image(depth_np)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
            )

            o3d_intr = o3d.camera.PinholeCameraIntrinsic()
            o3d_intr.set_intrinsics(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy)

            pcd_rgbd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)
            if voxel_size_rgbd and voxel_size_rgbd > 0:
                pcd_rgbd = pcd_rgbd.voxel_down_sample(voxel_size=voxel_size_rgbd)
            geoms.append(pcd_rgbd)
        
       
        # Color palette per instance
        ids = []
        mask_t = sample.get('mask')
        if mask_t is not None:
            mask_np = mask_t.numpy()
            ids = sorted([i for i in np.unique(mask_np) if i != 0])
        palette = get_color_palette(max(len(ids), 1))
        color_map = {inst_id: np.array(palette[i], dtype=np.float32) / 255.0 for i, inst_id in enumerate(ids)} if ids else {}

        # 2) Object model points (baked or canonical)
        geoms_added = 0
        if len(props) > 0 and show_props:
            # models keyed by inst_id, with pc_norm_cam, pos_m, scale_m
            for inst_id, vm in props.items():
                pn = vm['pc_norm_cam'].detach().cpu().numpy() if isinstance(vm['pc_norm_cam'], torch.Tensor) else np.asarray(vm['pc_norm_cam'])
                pos = vm['pos_m'].detach().cpu().numpy() if isinstance(vm['pos_m'], torch.Tensor) else np.asarray(vm['pos_m'], dtype=np.float32)
                sc  = float(vm.get('scale_m', 1.0))
                pts_cam = pn * sc + pos
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam.astype(np.float32)))
                col = color_map.get(inst_id, np.array([0.8, 0.8, 0.2], dtype=np.float32))
                pcd.paint_uniform_color(col.tolist())
                geoms.append(pcd)
                geoms_added += 1
                if geoms_added >= max_instances_overlay:
                    break
        elif len(models) > 0:
            for inst_id, vm in models.items():                
                p = poses.get(inst_id, None)
                if p is None:
                    print('[Show Pose] {} inst_id has no gt pose'.format(inst_id))
                    continue
                R = p['R']; t = p['t']
                R = R.detach().cpu().numpy().astype(np.float32) if isinstance(R, torch.Tensor) else np.asarray(R, dtype=np.float32)
                t = t.detach().cpu().numpy().astype(np.float32) if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)

                vm = vm.detach().cpu().numpy().astype(np.float32)
                
                pts_cam = (R @ vm.T).T + t.reshape(1, 3)  # meters in camera
                
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam.astype(np.float32)))
                col = color_map.get(inst_id, np.array([0.8, 0.8, 0.2], dtype=np.float32))
                pcd.paint_uniform_color(col.tolist())
                geoms.append(pcd)
                geoms_added += 1
                if geoms_added >= max_instances_overlay:
                    break
        
        # 3) Gripper instances (3D)
        if len(gt_gbi) > 0:
            for inst_id, grasps in gt_gbi.items():
                if not grasps:
                    continue
                sel = grasps if len(grasps) <= max_grasps_per_inst else random.sample(grasps, max_grasps_per_inst)
                col = color_map.get(inst_id, np.array([1.0, 0.2, 0.2], dtype=np.float32))
                for g in sel:
                    center = np.asarray(g.get('xyz', [0,0,0]), dtype=np.float32)
                    w, x, y, z = [float(q) for q in g.get('quat_wxyz', [1,0,0,0])]
                    try:
                        R = SciRot.from_quat([x,y,z,w]).as_matrix().astype(np.float32)
                    except Exception:
                        R = np.eye(3, dtype=np.float32)
                    width_m = float(g.get('width', 0.05))
                    #ls = gripper_lineset(center, R, width_m, L=finger_len_m, color_rgb=col.tolist())
                    ls = gripper_mesh_thick(center, R, width_m, L_m=finger_len_m,
                            finger_thickness_m=0.004,  # adjust thickness here
                            link_thickness_m=0.002,
                            color_rgb=col.tolist())
                    geoms += ls
        # Visualize
        o3d.visualization.draw_geometries(geoms, window_name=f"3D frame: {sample.get('image_type','')}/{sample.get('image_id','')}")
    
    
    def visualize(self, idx: int, max_instances_overlay: int = 20, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt

        sample = self[idx]

        # Basic tensors/images
        img = denorm_to_pil(sample['img'], mean=self.cfg.rgb_mean, std=self.cfg.rgb_std)
        depth = sample['depth'].squeeze(0).numpy() if sample.get('depth') is not None else None
        mask = sample['mask'].numpy() if sample.get('mask') is not None else None
        boxes = sample['boxes_xyxy'].numpy() if hasattr(sample.get('boxes_xyxy'), 'numpy') else np.asarray(sample.get('boxes_xyxy', []), dtype=np.float32)
        intr = sample['intr'].numpy().tolist() if sample.get('intr') is not None else None

        # Unified schema
        poses = sample.get('poses', {}) or sample.get('gt_obj_poses', {}) or {}
        if isinstance(poses, list):
            poses = {int(p.get('inst_id', i + 1)): p for i, p in enumerate(poses)}

        models = sample.get('models', {}) or {}
        props = sample.get('props', {}) or {}
        gt_gbi = sample.get('gt_grasps', {}) or {}

        
        # Palette per instance in mask
        ids = sorted([i for i in np.unique(mask) if i != 0]) if mask is not None else []
        palette = get_color_palette(len(ids)) if ids else get_color_palette(max(len(boxes), 1))
        color_map = {inst_id: palette[i] for i, inst_id in enumerate(ids)} if ids else {}

        # Build labels from poses: use obj_id (e.g., "101265") per inst_id

        inst_label_by_id = {iid: str(p.get('obj_id', iid)) for iid, p in (poses or {}).items()}
        labels = [inst_label_by_id.get(i, str(i)) for i in ids] if ids else None

        # Visual layers
        img_boxes = draw_boxes_on_rgb(img, boxes, palette=palette, labels=labels)
        depth_vis = depth_to_colormap(depth) if depth is not None else Image.new('RGB', img.size, (0, 0, 0))
        mask_vis = colorize_mask(mask, palette) if mask is not None else Image.new('RGB', img.size, (0, 0, 0))

        # Overlay projected points (+ optional grasps) following visualize3d() logic
        img_pts = img.copy()

        if intr is not None:
            # 1) Prefer baked props (normalized in camera); project with R=I, t=0
            if len(props) > 0:
                count = 0
                for inst_id in sorted(props.keys()):
                    vm = props[inst_id]
                    pn = vm['pc_norm_cam'].detach().cpu().numpy() if isinstance(vm['pc_norm_cam'], torch.Tensor) else np.asarray(vm['pc_norm_cam'])
                    pos = vm['pos_m'].detach().cpu().numpy() if isinstance(vm['pos_m'], torch.Tensor) else np.asarray(vm['pos_m'], dtype=np.float32)
                    sc  = float(vm.get('scale_m', 1.0))
                    pts_cam = pn * sc + pos  # already in camera coords (meters)

                    col = color_map.get(inst_id, palette[count % len(palette)])
                    uv = project_points(pts_cam, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), intr)
                    img_pts = overlay_points_on_rgb(img_pts, uv, color=col)

                    # Optional: overlay a few grasps for this instance if available
                    grasps_list = gt_gbi.get(inst_id, [])
                    if grasps_list:
                        sel = grasps_list if len(grasps_list) <= 2 else random.sample(grasps_list, 2)
                        for g in sel:
                            inst_col = color_map.get(inst_id, palette[count % len(palette)])
                            img_pts = _draw_gripper(
                                img_pts, g, intr,
                                finger_len_m=0.05,
                                default_width_m=float(g.get('width', 0.08)),
                                color_link=inst_col,
                                label_color=None
                            )

                    count += 1
                    if count >= max_instances_overlay:
                        break

            # 2) Fallback: use raw models keyed by inst_id and poses dict keyed by inst_id
            elif len(models) > 0 and len(poses) > 0:
                count = 0
                for inst_id in sorted(models.keys()):
                    p = poses.get(inst_id, None)
                    if p is None:
                        continue

                    R = p['R']; t = p['t']
                    R = R.detach().cpu().numpy().astype(np.float32) if isinstance(R, torch.Tensor) else np.asarray(R, dtype=np.float32)
                    t = t.detach().cpu().numpy().astype(np.float32) if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)

                    vm = models[inst_id]
                    vm = vm.detach().cpu().numpy().astype(np.float32) if isinstance(vm, torch.Tensor) else np.asarray(vm, dtype=np.float32)

                    pts_cam = (R @ vm.T).T + t.reshape(1, 3)
                    col = color_map.get(inst_id, palette[count % len(palette)])
                    uv = project_points(pts_cam, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), intr)
                    img_pts = overlay_points_on_rgb(img_pts, uv, color=col)

                    # Optional: overlay a few grasps for this instance if available
                    grasps_list = gt_gbi.get(inst_id, [])
                    if grasps_list:
                        sel = grasps_list if len(grasps_list) <= 2 else random.sample(grasps_list, 2)
                        for g in sel:
                            inst_col = color_map.get(inst_id, palette[count % len(palette)])
                            img_pts = _draw_gripper(
                                img_pts, g, intr,
                                finger_len_m=0.05,
                                default_width_m=float(g.get('width', 0.08)),
                                color_link=inst_col,
                                label_color=None
                            )

                    count += 1
                    if count >= max_instances_overlay:
                        break

        # Plot
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('DS: {}, IMG: {}, type: {}'.format(sample.get('dataset'), sample.get('image_id'), sample.get('image_type')))
        ax1 = fig.add_subplot(2, 3, 1); ax1.set_title('RGB'); ax1.imshow(img); ax1.axis('off')
        ax2 = fig.add_subplot(2, 3, 2); ax2.set_title('Depth'); ax2.imshow(depth_vis); ax2.axis('off')
        ax3 = fig.add_subplot(2, 3, 3); ax3.set_title('Mask'); ax3.imshow(mask_vis); ax3.axis('off')
        ax4 = fig.add_subplot(2, 3, 4); ax4.set_title('RGB + boxes'); ax4.imshow(img_boxes); ax4.axis('off')
        ax5 = fig.add_subplot(2, 3, 5); ax5.set_title('RGB + projected points + grasps'); ax5.imshow(img_pts); ax5.axis('off')
        ax6 = fig.add_subplot(2, 3, 6); ax6.set_title('Depth Ori'); ax6.imshow(depth); ax6.axis('off')
        plt.tight_layout()
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


# =========================

# GraspIt Dataset

# =========================

class GraspItDataset(BasePoseDataset):
    def __init__(self, 
            root: str,
            assets_root: Optional[str] = None,
            mode='train',
            transforms=True,
            with_grasp=True,
            depth_scale_m_per_unit=float(1/5000),
            invert_yaml_world2obj=False,
            include_mask=True,
            include_depth=True,
            include_models=True,
            rectify_prob: float = 0.0,
            min_visible_pixels: int = 100,
            ignore_table: bool = True,
            backgrounds_dirs: Optional[List[str]] = None,
            bg_replace_prob: float = 0.0,
            len_target_pc: int = 512,
            cache_grasps: bool = True,      
            overwrite_grasps: bool = True, 
            cfg: Optional[DatasetConfig] = None):
        super().__init__(mode=mode, transforms=transforms, rectify_prob=rectify_prob, min_visible_pixels=min_visible_pixels, len_target_pc=len_target_pc, cfg=cfg)
        self.name = 'graspit'
        self.root = root
        self.assets_root = assets_root if assets_root is not None else os.path.join(root, 'assets')
        self.depth_scale_m_per_unit = depth_scale_m_per_unit
        self.invert_yaml_world2obj = invert_yaml_world2obj
        self.include_mask = include_mask
        self.include_depth = include_depth
        self.include_models = include_models
        self.ignore_table = bool(ignore_table)
        self.bg_replace_prob = float(bg_replace_prob)
        self.with_grasp = with_grasp
        self.cache_grasps = bool(cache_grasps)
        self.overwrite_grasps = bool(overwrite_grasps)


        scene_dirs = []
        if os.path.exists(os.path.join(root, 'scenes')):
            scene_dirs.append(os.path.join(root, 'scenes'))
        scene_dirs.extend(glob.glob(os.path.join(root, 'scenes_*')))
        scene_dirs = [f for f in scene_dirs if '_old' not in f and 'broken' not in f]
        print('scene_dirs', scene_dirs)

        self.frames = []
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
                    pngs = glob.glob(os.path.join(frame_dir, 'rgb_*.png'))
                    if len(pngs) > 0:
                        b = os.path.basename(pngs[0])
                        idx = os.path.splitext(b)[0].split('_')[-1]
                    else:
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

        self.assets_pc_dir = os.path.join(self.assets_root + f"_pc_{self.cfg.pc_num_points}")
        ensure_dir(self.assets_pc_dir)

        # Backgrounds
        self.backgrounds = []
        if backgrounds_dirs is None:
            backgrounds_dirs = []
        bg_default = './data/backgrounds'
        if os.path.isdir(bg_default) and bg_default not in backgrounds_dirs:
            backgrounds_dirs.append(bg_default)
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        for d in backgrounds_dirs:
            if not os.path.isdir(d):
                continue
            for broot, bdirs, bfiles in os.walk(d):
                for f in bfiles:
                    fp = os.path.join(broot, f)
                    if os.path.isfile(fp) and os.path.splitext(fp)[1].lower() in exts:
                        self.backgrounds.append(fp)
        
    def __len__(self):
        return len(self.frames)

    
    def _project_point(self, P_cam: np.ndarray, intr: List[float]) -> Optional[Tuple[int, int]]:
        if intr is None:
            return None
        cx, cy, fx, fy = intr
        X, Y, Z = float(P_cam[0]), float(P_cam[1]), float(P_cam[2])
        if Z <= 1e-6:
            return None
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        return int(round(u)), int(round(v))

    def _cache_grasps_path(self, frame_dir: str) -> Tuple[str, str]:
        # preferred, fallback
        return os.path.join(frame_dir, 'frame_grasps.npz')

    def _save_grasps_npz(self, path_npz: str,
                        good_by_inst: Dict[int, List[Dict[str, Any]]],
                        bad_by_inst: Dict[int, List[Dict[str, Any]]]) -> None:
        """
        Save grasps in a flattened, consistent schema:
        - inst_good/inst_bad: int32
        - obj_good/obj_bad: int32
        - xyz_good/xyz_bad: (N,3) float32
        - quat_good/quat_bad: (N,4) float32 [w,x,y,z]
        - width_good/width_bad: float32
        - score_good/score_bad: float32
        - version: int32 scalar (2)

        """
        os.makedirs(os.path.dirname(path_npz), exist_ok=True)

        def flatten(dct: Dict[int, List[Dict[str, Any]]]):
            inst_id, obj_id, xyz, quat, width, score = [], [], [], [], [], []
            for iid, lst in (dct or {}).items():
                for g in lst:
                    inst_id.append(int(iid))
                    obj_id.append(int(g.get('obj_id', -1)))
                    xyz.append(np.asarray(g['xyz'], dtype=np.float32))
                    quat.append(np.asarray(g['quat_wxyz'], dtype=np.float32))
                    width.append(float(g.get('width', 0.0)))
                    score.append(float(g.get('score', 0.0)))
            return (
                np.asarray(inst_id, np.int32),
                np.asarray(obj_id, np.int32),
                np.stack(xyz, axis=0).astype(np.float32) if xyz else np.empty((0, 3), np.float32),
                np.stack(quat, axis=0).astype(np.float32) if quat else np.empty((0, 4), np.float32),
                np.asarray(width, np.float32),
                np.asarray(score, np.float32),
            )

        gi = flatten(good_by_inst)
        bi = flatten(bad_by_inst)

        np.savez_compressed(
            path_npz,
            version=np.array([2], dtype=np.int32),
            inst_good=gi[0], obj_good=gi[1], xyz_good=gi[2], quat_good=gi[3], width_good=gi[4], score_good=gi[5],
            inst_bad=bi[0], obj_bad=bi[1], xyz_bad=bi[2], quat_bad=bi[3], width_bad=bi[4], score_bad=bi[5],
        )

    def _load_grasps_npz(self, path_npz: str) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
        """
        Load the new NPZ schema. If incompatible/old, return empty dicts to force recompute.
        """
        if not os.path.exists(path_npz):
            return {}, {}
        D = np.load(path_npz)
        good, bad = defaultdict(list), defaultdict(list)

        # Require new keys
        if ('inst_good' in D) and ('inst_bad' in D):
            for key_prefix in ['good', 'bad']:
                inst = D[f'inst_{key_prefix}']
                obj  = D[f'obj_{key_prefix}']
                xyz  = D[f'xyz_{key_prefix}']
                quat = D[f'quat_{key_prefix}']
                width= D[f'width_{key_prefix}']
                score= D[f'score_{key_prefix}']
                tgt = good if key_prefix == 'good' else bad
                for i in range(inst.shape[0]):
                    iid = int(inst[i])
                    tgt[iid].append({
                        'obj_id': int(obj[i]),
                        'inst_id': iid,
                        'xyz': xyz[i].astype(np.float32),
                        'quat_wxyz': quat[i].astype(np.float32).tolist(),
                        'width': float(width[i]),
                        'score': float(score[i]),
                    })
            return dict(good), dict(bad)

        # Incompatible/old cache → discard
        return {}, {}

    
    def _load_grasps_npy_pickle(self, path_npy: str) -> Optional[Dict[int, List[Dict[str, Any]]]]:
        # Backward-compat if someone wrote a dict via np.save(..., allow_pickle=True)
        try:
            data = np.load(path_npy, allow_pickle=True)
            obj = data.item() if hasattr(data, 'item') else data
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return None

    def _synthesize_bad_grasps_from_good(self, out_by_inst: Dict[int, List[Dict[str, Any]]],
                                        default_width_m: float) -> Dict[int, List[Dict[str, Any]]]:

        bad_by_inst = defaultdict(list)

        def _rand_xyz_delta():
            mag = np.random.uniform(0.1, 0.3, size=(3,)).astype(np.float32)
            sgn = np.random.choice([-1.0, 1.0], size=(3,)).astype(np.float32)
            return mag * sgn

        def _rand_rpy_delta_deg():
            mag = np.random.uniform(45.0, 90.0, size=(3,)).astype(np.float32)
            sgn = np.random.choice([-1.0, 1.0], size=(3,)).astype(np.float32)
            return mag * sgn

        for inst_id, goods in out_by_inst.items():
            if not goods:
                continue
            k = 10
            idxs = np.random.choice(len(goods), size=k, replace=(len(goods) < k))
            for base_idx in idxs:
                base = goods[base_idx]
                base_xyz = np.asarray(base['xyz'], dtype=np.float32)
                qw, qx, qy, qz = [float(v) for v in base['quat_wxyz']]
                R_base = SciRot.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)

                # combined
                dx = _rand_xyz_delta()
                rpy_deg = _rand_rpy_delta_deg()
                R_delta = SciRot.from_euler('xyz', np.deg2rad(rpy_deg)).as_matrix().astype(np.float32)
                R_new = (R_delta @ R_base).astype(np.float32)
                q_new = SciRot.from_matrix(R_new).as_quat()
                bad_by_inst[inst_id].append({
                    'obj_id': base['obj_id'],
                    'inst_id': inst_id,
                    'xyz': (base_xyz + dx).astype(np.float32),
                    'quat_wxyz': [float(q_new[3]), float(q_new[0]), float(q_new[1]), float(q_new[2])],
                    'width': float(base.get('width', default_width_m)),
                    'score': 0.0
                })

                n_xyz = 2 + int(np.random.rand() < 0.5)
                n_rpy = 2 + int(np.random.rand() < 0.5)

                # xyz-only
                for _ in range(n_xyz):
                    dx = _rand_xyz_delta()
                    bad_by_inst[inst_id].append({
                        'obj_id': base['obj_id'],
                        'inst_id': inst_id,
                        'xyz': (base_xyz + dx).astype(np.float32),
                        'quat_wxyz': base['quat_wxyz'],
                        'width': float(base.get('width', default_width_m)),
                        'score': 0.0
                    })

                # rpy-only
                for _ in range(n_rpy):
                    rpy_deg = _rand_rpy_delta_deg()
                    R_delta = SciRot.from_euler('xyz', np.deg2rad(rpy_deg)).as_matrix().astype(np.float32)
                    R_new = (R_delta @ R_base).astype(np.float32)
                    q_new = SciRot.from_matrix(R_new).as_quat()
                    bad_by_inst[inst_id].append({
                        'obj_id': base['obj_id'],
                        'inst_id': inst_id,
                        'xyz': base_xyz.astype(np.float32),
                        'quat_wxyz': [float(q_new[3]), float(q_new[0]), float(q_new[1]), float(q_new[2])],
                        'width': float(base.get('width', default_width_m)),
                        'score': 0.0
                    })
        return bad_by_inst

            
    def _canonicalize_grasp_R_t(self,
                            R_in: np.ndarray,
                            t_in: np.ndarray,
                            intr: Optional[list] = None,
                            depth_m: Optional[np.ndarray] = None,
                            use_depth_normal: bool = True,
                            finger_len_m: float = 0.05,
                            shift_tcp: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Return R_can, t_can where:
        - column 2 (Z) is the approach axis (pointing into scene/object),
        - column 0 (X) is width axis orthogonal to Z,
        - column 1 (Y) completes right-handed frame.

        If shift_tcp=True, shift TCP to palm center by -finger_len_m along local +Z.
        """
        R = np.asarray(R_in, dtype=np.float32)
        t = np.asarray(t_in, dtype=np.float32)

        # 1) Estimate desired approach direction
        Z_des = None
        if use_depth_normal and (intr is not None) and (depth_m is not None):
            # Project center pixel and estimate normal
            cx, cy, fx, fy = intr
            uv = project_points(t.reshape(1, 3), np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), intr)[0]
            u, v, valid = float(uv[0]), float(uv[1]), bool(uv[2])
            if valid:
                n = self._estimate_normal_from_depth(depth_m, u, v, intr, win=5)
                if n is not None and np.linalg.norm(n) > 1e-6:
                    Z_des = n / (np.linalg.norm(n) + 1e-9)

        if Z_des is None:
            # Fallback: use camera→object ray
            ray = t / (np.linalg.norm(t) + 1e-9)
            Z_des = ray

        # 2) Select the column of R closest to Z_des (in absolute cosine)
        cols = [R[:, 0], R[:, 1], R[:, 2]]
        dots = [abs(float(np.dot(c / (np.linalg.norm(c) + 1e-9), Z_des))) for c in cols]
        idx_z = int(np.argmax(dots))
        Z = cols[idx_z] / (np.linalg.norm(cols[idx_z]) + 1e-9)

        # Ensure Z points roughly camera→object (positive dot)
        if float(np.dot(Z, t)) < 0.0:
            Z = -Z

        # 3) Build X as "width axis" by projecting the original width candidate onto plane ⟂ Z
        idx_width = 0 if idx_z != 0 else 1  # pick another column as width candidate
        Xcand = cols[idx_width]
        X = Xcand - Z * float(np.dot(Xcand, Z))
        nX = np.linalg.norm(X)
        if nX < 1e-6:
            # Fallback to any axis orthogonal to Z
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(tmp, Z))) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            X = tmp - Z * float(np.dot(tmp, Z))
            nX = np.linalg.norm(X)
        X /= (nX + 1e-9)

        # 4) Complete right-handed frame
        Y = np.cross(Z, X)
        Y /= (np.linalg.norm(Y) + 1e-9)
        R_can = np.stack([X, Y, Z], axis=1).astype(np.float32)

        # 5) Optional TCP shift (for GraspNet: usually False)
        if shift_tcp:
            t = t - finger_len_m * Z

        return R_can, t.astype(np.float32)



    def _estimate_normal_from_depth(self, depth_m: np.ndarray, u: float, v: float, intr: list, win: int = 5):
        # Clamp to valid integer pixel
        H, W = depth_m.shape
        ui = int(round(u)); vi = int(round(v))
        ui = np.clip(ui, 0, W - 1); vi = np.clip(vi, 0, H - 1)

        # Build a window with at least 3x3 pixels
        half = max(1, int(win))
        x1 = max(0, ui - half); x2 = min(W, ui + half + 1)
        y1 = max(0, vi - half); y2 = min(H, vi + half + 1)

        # If the window collapses at edges, expand inward to ensure size
        if (x2 - x1) < 3:
            if ui <= 1: x2 = min(W, x1 + 3)
            elif ui >= W - 2: x1 = max(0, x2 - 3)
            else:
                # center case: expand both sides
                need = 3 - (x2 - x1)
                x1 = max(0, x1 - need // 2)
                x2 = min(W, x2 + need - need // 2)
        if (y2 - y1) < 3:
            if vi <= 1: y2 = min(H, y1 + 3)
            elif vi >= H - 2: y1 = max(0, y2 - 3)
            else:
                need = 3 - (y2 - y1)
                y1 = max(0, y1 - need // 2)
                y2 = min(H, y2 + need - need // 2)

        patch = depth_m[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        # Valid depth mask
        m = patch > 0
        if not np.any(m):
            return None

        # Pixel coordinates for valid points
        yy, xx = np.indices(patch.shape)
        yy = (yy + y1)[m].astype(np.float32)
        xx = (xx + x1)[m].astype(np.float32)
        ds = patch[m].astype(np.float32)

        # Backproject
        cx, cy, fx, fy = intr
        X = (xx - cx) * ds / fx
        Y = (yy - cy) * ds / fy
        Z = ds
        P = np.stack([X, Y, Z], axis=1).astype(np.float32)
        if P.shape[0] < 3:
            return None

        # Fit plane via SVD
        C = P.mean(axis=0)
        Q = P - C
        try:
            _, _, vh = np.linalg.svd(Q, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        n = vh[-1, :].astype(np.float32)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            return None
        n /= n_norm
        return n


    def _ensure_min_bad_grasps(
        self,
        gt_grasps_by_inst: Dict[int, List[Dict[str, Any]]],
        bad_grasps: Dict[int, List[Dict[str, Any]]],
        poses: Dict[int, Dict[str, Any]],
        visible_inst_ids: List[int],
        k_per_inst: int = 10,
        default_width_m: float = 0.08,
        intr: Optional[List[float]] = None,
        depth_m: Optional[np.ndarray] = None,
        finger_len_m: float = 0.05,
        models: Optional[Dict[int, Any]] = None,   # NEW
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Ensure each visible inst_id has at least some bad grasps:
        - If goods exist but < k bads: synthesize small-perturbation bad grasps from goods to reach k_per_inst.
        - If neither goods nor enough bads exist: sample random points from posed model and create random grasps.

            If model/pose missing, fallback to canonicalization around object pose.

        Random-grasp synthesis from model points:
        - pick a random model point in camera coords,
        - set approach axis Z along cam->point direction,
        - pick a random orthonormal frame around Z,
        - place TCP at point shifted by -finger_len along Z,
        - assign bad score 0.0.

        """

        def _rand_xyz_delta():
            mag = np.random.uniform(0.0, 0.05, size=(3,)).astype(np.float32)
            sgn = np.random.choice([-1.0, 1.0], size=(3,)).astype(np.float32)
            return mag * sgn

        def _rand_rpy_delta_deg():
            mag = np.random.uniform(0.0, 180.0, size=(3,)).astype(np.float32)
            sgn = np.random.choice([-1.0, 1.0], size=(3,)).astype(np.float32)
            return mag * sgn

        # Small deltas for goods->bad synthesis
        def _rand_xyz_delta_good():
            mag = np.random.uniform(0.005, 0.02, size=(3,)).astype(np.float32)
            sgn = np.random.choice([-1.0, 1.0], size=(3,)).astype(np.float32)
            if np.random.rand() > 0.5:
                mag[random.randint(0, 2)] = 0
            return mag * sgn

        def _rand_rpy_delta_deg_good():
            mag = np.random.uniform(22.5, 90.0, size=(3,)).astype(np.float32)
            sgn = np.random.choice([-1.0, 1.0], size=(3,)).astype(np.float32)
            if np.random.rand() > 0.5:
                mag[random.randint(0, 2)] = 0
            return mag * sgn

        for inst_id in visible_inst_ids:
            goods = gt_grasps_by_inst.get(inst_id, []) or []
            bads  = bad_grasps.get(inst_id, []) or []

            # If we already have enough bad grasps, skip
            if len(bads) >= k_per_inst:
                continue

            # Case B: have goods but not enough bads -> synthesize small-perturbation bads from goods
            if len(goods) > 0 and len(bads) < k_per_inst:
                missing = max(0, k_per_inst - len(bads))
                if missing > 0:
                    synth = []
                    idxs = np.random.choice(len(goods), size=missing, replace=(len(goods) < missing))
                    for base_idx in idxs:
                        base = goods[base_idx]
                        base_xyz = np.asarray(base['xyz'], dtype=np.float32)

                        # Base orientation
                        qw, qx, qy, qz = [float(v) for v in base['quat_wxyz']]
                        R_base = SciRot.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)

                        # Apply small combined xyz + rpy deltas
                        dx = _rand_xyz_delta_good()
                        rpy_deg = _rand_rpy_delta_deg_good()
                        R_delta = SciRot.from_euler('xyz', np.deg2rad(rpy_deg)).as_matrix().astype(np.float32)
                        R_new = (R_delta @ R_base).astype(np.float32)
                        q_new = SciRot.from_matrix(R_new).as_quat()  # [x,y,z,w]

                        # obj_id
                        obj_id_val = base.get('obj_id', None)
                        if obj_id_val is None:
                            p = poses.get(inst_id, None)
                            obj_id_val = p.get('obj_id', str(inst_id)) if p is not None else str(inst_id)

                        synth.append({
                            'obj_id': obj_id_val,
                            'inst_id': int(inst_id),
                            'xyz': (base_xyz + dx).astype(np.float32),
                            'quat_wxyz': [float(q_new[3]), float(q_new[0]), float(q_new[1]), float(q_new[2])],
                            'width': float(base.get('width', default_width_m)),
                            'score': 0.0
                        })
                    bad_grasps[inst_id] = bad_grasps.get(inst_id, []) + synth
                continue

            # Case C: no goods and not enough bads -> sample from posed model points
            missing = max(0, k_per_inst - len(bads))
            if missing == 0:
                continue

            p = poses.get(inst_id, None)
            vm = models.get(inst_id, None) if isinstance(models, dict) else None
            if p is not None and vm is not None:
                # Pose model into camera frame
                R_obj = p['R']; t_obj = p['t']
                R_obj = R_obj.detach().cpu().numpy().astype(np.float32) if isinstance(R_obj, torch.Tensor) else np.asarray(R_obj, dtype=np.float32)
                t_obj = t_obj.detach().cpu().numpy().astype(np.float32) if isinstance(t_obj, torch.Tensor) else np.asarray(t_obj, dtype=np.float32)
                vm_np = vm.detach().cpu().numpy().astype(np.float32) if isinstance(vm, torch.Tensor) else np.asarray(vm, dtype=np.float32)
                if vm_np.ndim == 2 and vm_np.shape[1] == 3 and vm_np.shape[0] > 0:
                    pts_cam = (R_obj @ vm_np.T).T + t_obj.reshape(1, 3)  # meters in camera

                    synth = []
                    # Sample with replacement if needed
                    idxs = np.random.choice(pts_cam.shape[0], size=missing, replace=(pts_cam.shape[0] < missing))
                    for ii in idxs:
                        p_cam = pts_cam[int(ii)].astype(np.float32)

                        # Build random orientation:
                        # Z = cam->point direction
                        Z = p_cam / (np.linalg.norm(p_cam) + 1e-9)
                        # random Y orthogonal to Z
                        u = np.random.randn(3).astype(np.float32)
                        u = u - Z * float(np.dot(u, Z))
                        nu = np.linalg.norm(u)
                        if nu < 1e-9:
                            u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                            u = u - Z * float(np.dot(u, Z))
                            nu = np.linalg.norm(u) + 1e-9
                        Y = (u / nu).astype(np.float32)
                        # X completes frame
                        X = np.cross(Y, Z).astype(np.float32)
                        X /= (np.linalg.norm(X) + 1e-9)
                        Y = np.cross(Z, X).astype(np.float32)
                        Y /= (np.linalg.norm(Y) + 1e-9)
                        Rg = np.stack([X, Y, Z], axis=1).astype(np.float32)

                        # TCP shifted back along +Z (approach) so fingertips meet the sampled point
                        tcp = (p_cam - finger_len_m * Z).astype(np.float32)

                        q_xyzw = SciRot.from_matrix(Rg).as_quat()
                        q_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]

                        synth.append({
                            'obj_id': p.get('obj_id', str(inst_id)),
                            'inst_id': int(inst_id),
                            'xyz': tcp,
                            'quat_wxyz': q_wxyz,
                            'width': float(default_width_m),
                            'score': 0.0  # bad
                        })

                    bad_grasps[inst_id] = bad_grasps.get(inst_id, []) + synth
                    continue

            # Fallback: canonicalize from object pose if model missing
            if p is not None and len(bad_grasps.get(inst_id, [])) < k_per_inst:
                R_obj = p['R']; t_obj = p['t']
                R_obj = R_obj.detach().cpu().numpy().astype(np.float32) if isinstance(R_obj, torch.Tensor) else np.asarray(R_obj, dtype=np.float32)
                t_obj = t_obj.detach().cpu().numpy().astype(np.float32) if isinstance(t_obj, torch.Tensor) else np.asarray(t_obj, dtype=np.float32)

                R_base, t_base = self._canonicalize_grasp_R_t(
                    R_in=R_obj, t_in=t_obj,
                    intr=intr, depth_m=depth_m,
                    use_depth_normal=True,
                    finger_len_m=finger_len_m,
                    shift_tcp=True
                )

                synth_list = []
                for _ in range(int(missing)):
                    dx = _rand_xyz_delta()
                    rpy_deg = _rand_rpy_delta_deg()
                    R_delta = SciRot.from_euler('xyz', np.deg2rad(rpy_deg)).as_matrix().astype(np.float32)
                    R_new = (R_delta @ R_base).astype(np.float32)
                    q_new = SciRot.from_matrix(R_new).as_quat()
                    synth_list.append({
                        'obj_id': p.get('obj_id', str(inst_id)),
                        'inst_id': int(inst_id),
                        'xyz': (t_base + dx).astype(np.float32),
                        'quat_wxyz': [float(q_new[3]), float(q_new[0]), float(q_new[1]), float(q_new[2])],
                        'width': float(default_width_m),
                        'score': 0.0
                    })
                bad_grasps[inst_id] = bad_grasps.get(inst_id, []) + synth_list

        return bad_grasps

        

    def _load_grasps_hdf5(
        self,
        scene_dir: str,
        frame_dir: str,
        R_wc_cv: np.ndarray,
        t_wc_cv: np.ndarray,
        intr: Optional[List[float]],
        inst_map: Dict[str, int],           # maps '5' (from class 'object5') -> inst_id
        visible_inst_ids: List[int],
        default_width_m: float = 0.08,
        finger_len_m: float = 0.05,
        graspit_origin_is_fingertips: bool = True,
        cache_grasps: Optional[bool] = None,
        overwrite_cache: Optional[bool] = None,
        depth_m: Optional[np.ndarray] = None,
        margin_m: float = 0.01,
        patch: int = 3,
        obj_id2key: Optional[Dict[str, str]] = None,   # map 'objectN' -> inst_id
    ) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:



        if not self.with_grasp:
            return {}, {}

        cache_grasps = self.cache_grasps if cache_grasps is None else bool(cache_grasps)
        overwrite_cache = self.overwrite_grasps if overwrite_cache is None else bool(overwrite_cache)
        path_npz = self._cache_grasps_path(frame_dir)

        def _depth_at(d_m: np.ndarray, u: float, v: float, patch_sz: int = 3) -> float:
            H, W = d_m.shape
            ui, vi = int(round(u)), int(round(v))
            x1, x2 = max(0, ui - patch_sz), min(W, ui + patch_sz + 1)
            y1, y2 = max(0, vi - patch_sz), min(H, vi + patch_sz + 1)
            win = d_m[y1:y2, x1:x2]
            win = win[win > 0]
            return float(np.median(win)) if win.size > 0 else 0.0

        def _is_visible_grasp_local(t_cam_m: np.ndarray, intr_l: list, d_m: np.ndarray, mrg: float = 0.01) -> bool:
            if intr_l is None or d_m is None or d_m.ndim != 2:
                return True
            cx, cy, fx, fy = intr_l
            X, Y, Z = float(t_cam_m[0]), float(t_cam_m[1]), float(t_cam_m[2])
            if Z <= 1e-6:
                return False
            u = fx * (X / Z) + cx
            v = fy * (Y / Z) + cy
            H, W = d_m.shape
            if u < 0 or u >= W or v < 0 or v >= H:
                return False
            d = _depth_at(d_m, u, v, patch_sz=patch)
            if d <= 0:
                return False
            return Z <= (d + mrg)

        if cache_grasps and (not overwrite_cache) and os.path.exists(path_npz):
            try:
                good_by_inst, bad_by_inst = self._load_grasps_npz(path_npz)
                if bad_by_inst or bad_by_inst:
                    return good_by_inst, bad_by_inst
            except Exception as e:
                print(f"[WARN] Failed to load grasps cache at {path_npz}: {e}")
            
        # Load successful indices from dataset.json
        ds_json = read_json(os.path.join(scene_dir, 'dataset.json'), default=None)
        
        
        good_by_inst = defaultdict(list)
        bad_by_inst  = defaultdict(list)
        for objnr in ds_json['grasp'].keys():            
            obj_num_str = str(obj_id2key.get(objnr, '-1'))

            inst_id = inst_map.get(obj_num_str, -1)

            if inst_id not in visible_inst_ids:
                continue           

            #good_idx_set = [working_traj[int(slip_idx)] for slip_idx in list(slip_map.get(objnr, {}).keys()) ]
            all_slips = ds_json['slip'].get(objnr, {})
            slip_indexes = [int(slip_idx) for slip_idx in all_slips.keys()]

            all_grasps = ds_json['grasp'].get(objnr, {})
            all_grasps_indexes = [int(grasp_idx) for grasp_idx in all_grasps.keys()] 
  
            for test_idx in all_grasps_indexes:
                #if test_idx not in working_traj:
                #    continue
                score = 1.0
                if test_idx not in slip_indexes:
                    score = 0.0
                else:
                    slip_xyz, slip_rpy = all_slips.get(str(test_idx))
                    
                    if slip_xyz[0][0] is None: # failed to close the gripper
                        score = 0.0
                    if slip_xyz[1][0] is None: # failed to lift the obj
                        score = 0.25
                    elif slip_xyz[2][0] is None: # failed to move the object horizonal left-right
                        score = 0.5
                    elif slip_xyz[3][0] is None: # failed to hold the obj in a pendulum movement
                        score = 0.75
                                        
                    if score > 0.0: # if not slipped, punish the score if we have large slip
                        slip_xyz = np.abs([s for s in np.array(slip_xyz).flatten() if s is not None])
                        slip_rpy = np.abs([s for s in np.array(slip_rpy).flatten() if s is not None])
                        # check if we have a positinal slip > 2cm or orientational > 10 deg
                        if np.max(slip_xyz) > 0.02: # pushin xyz slip > 2cm
                            score = score - (0.25/3)                        
                        if np.max(slip_rpy) > 10: # punish rpy slip > 10 deg
                            score = score - (0.25/3)                                              

                #print('str(test_idx)', str(test_idx))
                T_w = all_grasps[str(test_idx)]
                        
                T_w = np.array(T_w)                   

                R_w = T_w[:3, :3].astype(np.float32)
                t_w = T_w[:3, 3].astype(np.float32)

                R_c = (R_wc_cv @ R_w).astype(np.float32)
                t_c = (R_wc_cv @ t_w + t_wc_cv).astype(np.float32)

                if graspit_origin_is_fingertips:
                    Z = R_c[:, 2]
                    t_c = t_c - finger_len_m * Z

                # Visibility filter
                if depth_m is not None and intr is not None:
                    if not _is_visible_grasp_local(t_c, intr, depth_m, margin_m):
                        continue

                q_wxyz = rotation_matrix_to_quaternion_wxyz(R_c)
                g = {
                    'obj_id': obj_num_str,  # keep group index for traceability
                    'inst_id': int(inst_id),
                    'xyz': t_c.astype(np.float32),
                    'quat_wxyz': [float(q) for q in q_wxyz],
                    'width': float(default_width_m),
                    'score': score
                }
                if score < 0.5:
                    bad_by_inst[int(inst_id)].append(g)
                else:
                    good_by_inst[int(inst_id)].append(g)
                                            

        # Cache only the visible good grasps (maintain prior cache schema)
        if cache_grasps:
            try:
                #print('saving graps to {} with {} bad and {} good grasps'.format(path_npz, sum([len(v) for v in bad_by_inst.values()]), sum([len(v) for v in good_by_inst.values()])))
                self._save_grasps_npz(path_npz, good_by_inst, bad_by_inst)
            except Exception as e:
                print(f"[WARN] Failed to save grasps cache at {path_npz}: {e}")
        return dict(good_by_inst), dict(bad_by_inst)

    
    # Add inside class GraspItDataset(BasePoseDataset):

    def precompute_grasps(self, overwrite: bool = True, max_workers: int = 8, margin_m: float = 0.01, patch: int = 3):
        """
        Precompute and cache visible grasps for all frames using _load_grasps_hdf5.
        Overwrites existing caches when overwrite=True.

        Writes grasps_cam.npz files in each frame_dir.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _work(meta):
            try:
                scene_yaml = self._load_scene_yaml(meta['scene_yaml'])
                frame_dir  = meta['frame_dir']
                frame_idx  = meta['frame_idx']

                # Depth (meters, original resolution)
                depth_m = None
                dpth_path = os.path.join(frame_dir, "depth.png")
                if os.path.exists(dpth_path):
                    depth_raw = np.array(Image.open(dpth_path), dtype=np.float32)
                    depth_m = depth_raw * float(self.depth_scale_m_per_unit)

                # Mask and inst mapping from label ordering
                mask_np, _, label_order_obj_ids = self._build_mask_and_bg(frame_dir, frame_idx)

                inst_map = {}
                for idx, oid in enumerate(label_order_obj_ids, start=1):
                    if oid is not None:
                        inst_map[str(oid)] = idx

                # Visible instances (original size)
                if mask_np is not None and mask_np.size > 0:
                    counts = np.bincount(mask_np.reshape(-1))
                    visible_inst_ids = [i for i in range(1, len(counts)) if counts[i] >= self.min_visible_pixels]
                else:
                    # Fallback: assume all labeled objects are visible
                    visible_inst_ids = list(range(1, len(label_order_obj_ids) + 1))

                # Camera intr/extr (OpenCV-style from USD)
                intr = None
                R_wc_cv = np.eye(3, dtype=np.float32)
                t_wc_cv = np.zeros(3, dtype=np.float32)
                cam_p = os.path.join(frame_dir, f"camera_params_{frame_idx}.json")
                if os.path.exists(cam_p):
                    cam_params = read_json(cam_p, default=None)
                    if cam_params is not None:
                        intr = intr_from_graspit_camera_params(cam_params)
                        T_w2c_usd = load_camera_view_matrix(cam_params)
                        Rfix = rotation_x_pi()
                        R_wc_cv = (Rfix @ T_w2c_usd[:3, :3]).astype(np.float32)
                        t_wc_cv = (Rfix @ T_w2c_usd[:3, 3]).astype(np.float32)

                # Map YAML obj ids to objectN keys for HDF5/dataset.json
                _, obj_id2key = self._poses_from_yaml(scene_yaml['_objects'])

                # Compute and cache (overwriting if requested)
                good_by_inst, bad_by_inst = self._load_grasps_hdf5(
                    scene_dir=meta['scene_dir'],
                    frame_dir=frame_dir,
                    R_wc_cv=R_wc_cv,
                    t_wc_cv=t_wc_cv,
                    intr=intr,
                    inst_map=inst_map,
                    visible_inst_ids=visible_inst_ids,
                    default_width_m=0.08,
                    finger_len_m=0.05,
                    graspit_origin_is_fingertips=True,
                    cache_grasps=True,
                    overwrite_cache=overwrite,
                    depth_m=depth_m,
                    margin_m=margin_m,
                    patch=patch,
                    obj_id2key=obj_id2key
                )
                return sum(len(v) for v in good_by_inst.values()), sum(len(v) for v in bad_by_inst.values())
            except Exception as e:
                logging.warning(f"[precompute_grasps] Failed for {meta.get('frame_dir','?')}: {e}")
                raise e
                return 0, 0

        total_good, total_bad = 0, 0
        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_work, m) for m in self.frames]
                for f in as_completed(futs):
                    g, b = f.result()
                    total_good += g
                    total_bad += b
        else:
            for m in self.frames:
                g, b = _work(m)
                total_good += g
                total_bad += b
        print(f"[GraspItDataset] Precomputed grasps (overwrite={overwrite}) | frames={len(self.frames)} | good={total_good} | bad={total_bad}")


    def _poses_from_yaml(self, sc_obj_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        poses_w = []
        obj_id2key = {}
        for obj_id, od in sc_obj_dict.items():
            q = od['quat_wxyz']
            t = np.array(od['t'], dtype=np.float32)
            R = quaternion_to_rotation_matrix(q).astype(np.float32)  # YAML: object->world
            if self.invert_yaml_world2obj:
                R = R.T
                t = - (R @ t)
            s = od.get('scale', [1, 1, 1])
            s_vec = np.array(s if not np.isscalar(s) else [s]*3, dtype=np.float32)
            obj_id2key[od['obj_key']] = obj_id
            poses_w.append({    
                'obj_id': obj_id,
                'R_o2w': R,
                't_o2w': t,
                'scene_scale': s_vec
            })
        for i, p in enumerate(poses_w):
            p['inst_id'] = i + 1
        return poses_w, obj_id2key

    def _load_scene_yaml(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            sc = yaml.safe_load(f)
        objs = {}

        obj_idx = 0
        for k, v in sc.items():
            if not isinstance(v, dict):
                continue
            if self.ignore_table and str(v.get('name', '')).strip().lower() == 'table':
                continue
            if 'file_path' in v and 'position' in v and 'orientation' in v:
                fp = v['file_path']
                m = re.search(r'/assets/(\d+)/', fp)
                obj_id = m.group(1) if m else (re.search(r'(\d+)', v.get('name','') or '') or [None])[0]
                if obj_id is None:
                    continue 

                objs[obj_id] = {
                    'obj_key': str(obj_idx),
                    'quat_wxyz': v['orientation'],
                    't': v['position'],
                    'scale': v.get('scale', [1,1,1]),
                    'file_path': fp
                }
                obj_idx +=1
        sc['_objects'] = objs
        return sc

    def _load_assets_pc(self, obj_id: str) -> Optional[np.ndarray]:
        src_obj_dir = os.path.join(self.assets_root, obj_id)
        if not os.path.isdir(src_obj_dir):
            return None
        cand = None
        for pat in ('*.obj','*.ply','*.stl'):
            g = glob.glob(os.path.join(src_obj_dir, pat))
            if g:
                cand = g[0]; break
        if cand is None:
            return None
        out_dir = os.path.join(self.assets_root + f"_pc_{self.cfg.pc_num_points}", obj_id)
        ensure_dir(out_dir)
        out_pc = os.path.join(out_dir, f"model_pc{self.cfg.pc_num_points}.ply")
        out_pc_path = ensure_downsampled_mesh_to_pc(cand, out_pc, num_points=self.cfg.pc_num_points)
        if out_pc_path is None:
            return None
        pts = load_pointcloud_file(out_pc_path, pc_num_points=self.cfg.pc_num_points)
        return pts

    def _build_mask_and_bg(self, frame_dir: str, frame_idx: str):
        mask_info_path = os.path.join(frame_dir, f"semantic_segmentation_labels_{frame_idx}.json")
        mask_img_path = os.path.join(frame_dir, f"semantic_segmentation_{frame_idx}.png")
        mask_info = read_json(mask_info_path, default={})
        #print('frame_dir', frame_dir, frame_idx)

        # Prefer RGBA so BACKGROUND (0,0,0,0) matches exactly
        try:
            mask_rgba = np.array(Image.open(mask_img_path).convert('RGBA'))
        except Exception:
            mask_rgba = np.array(Image.open(mask_img_path).convert('RGB'))

        H, W = mask_rgba.shape[:2]
        mask_np = np.zeros((H, W), dtype=np.uint8)

        # Combined background mask (BACKGROUND ∪ plane), boolean for compositing
        bg_mask = np.zeros((H, W), dtype=bool)

        inst_id = 1
        label_order_obj_ids = []

        for rgb_key, entry in mask_info.items():
            vals = [int(v) for v in str(rgb_key).replace('(', '').replace(')', '').replace(' ', '').split(',')]
            cls = str(entry.get('class', '')).strip().lower()

            if len(vals) >= 4 and mask_rgba.shape[2] == 4:
                trip = np.array(vals[:4], dtype=np.uint8)
                m = np.all(mask_rgba == trip.reshape(1, 1, 4), axis=2)
            else:
                trip = np.array(vals[:3], dtype=np.uint8)
                m = np.all(mask_rgba[..., :3] == trip.reshape(1, 1, 3), axis=2)

            if cls in ('background', 'plane'):
                if np.any(m):
                    bg_mask |= m
                continue

            if self.ignore_table and cls == 'table':
                continue

            if cls == 'grasp_path':
                continue

            if np.any(m):
                if 'object' in cls:
                    obj_id = cls.replace('object', '')                
                    mask_np[m] = inst_id
                    label_order_obj_ids.append(obj_id)               
                    inst_id += 1

        return mask_np, bg_mask, label_order_obj_ids
    
    def _composite_background(self, img: Image.Image, depth_np: np.ndarray, bg_mask: np.ndarray, p_remove_depth: float = 1.0) -> Image.Image:
        if bg_mask is None or not np.any(bg_mask) or len(self.backgrounds) == 0:
            return img
        bg_path = random.choice(self.backgrounds)
        bg = Image.open(bg_path).convert('RGB').resize(img.size, Image.BILINEAR)
        img_np = np.array(img)
        bg_np = np.array(bg)

        out = img_np.copy()
        out[bg_mask] = bg_np[bg_mask]  # replace only BACKGROUND pixels
        
        if p_remove_depth > random.random():
            depth_np[bg_mask] = 0

        return Image.fromarray(out), depth_np
    
    def __getitem__(self, index: int):
        meta = self.frames[index]
        scene_yaml = self._load_scene_yaml(meta['scene_yaml'])
        frame_dir = meta['frame_dir']
        frame_idx = meta['frame_idx']

        # RGB
        rgb_path = os.path.join(frame_dir, f"rgb_{frame_idx}.png")
        img = Image.open(rgb_path).convert('RGB')
        W0, H0 = img.size

        # Depth
        depth_np = None
        if self.include_depth:
            dpth_path = os.path.join(frame_dir, "depth.png")
            depth_raw = np.array(Image.open(dpth_path), dtype=np.float32)
            depth_np = depth_raw * float(self.depth_scale_m_per_unit)  # meters
            #print('depth_np', np.min(depth_np[depth_np > 0]))

        # Mask + background mask
        mask_np, bg_mask, label_order_obj_ids = (None, None, [])
        if self.include_mask:
            mask_np, bg_mask, label_order_obj_ids = self._build_mask_and_bg(frame_dir, frame_idx)

        if self.bg_replace_prob > 0.0 and random.random() < self.bg_replace_prob and bg_mask is not None:
            img, depth_np = self._composite_background(img, depth_np, bg_mask)

        # Camera intr/extr with Rx(pi) fix (OpenCV-style)
        intr = None
        R_wc_cv = np.eye(3, dtype=np.float32)
        t_wc_cv = np.zeros(3, dtype=np.float32)
        cam_p = os.path.join(frame_dir, f"camera_params_{frame_idx}.json")
        if os.path.exists(cam_p):
            cam_params = read_json(cam_p, default=None)
            if cam_params is not None:
                intr = intr_from_graspit_camera_params(cam_params)
                T_w2c_usd = load_camera_view_matrix(cam_params)
                Rfix = rotation_x_pi()
                R_wc_cv = (Rfix @ T_w2c_usd[:3, :3]).astype(np.float32)
                t_wc_cv = (Rfix @ T_w2c_usd[:3, 3]).astype(np.float32)

        # Object->world from YAML, then Object->camera
        poses_w, obj_id2key = self._poses_from_yaml(scene_yaml['_objects'])
        inst_map = {}
        for idx, oid in enumerate(label_order_obj_ids, start=1):
            if oid is not None:
                inst_map[str(oid)] = idx

        poses = {}
        for p in poses_w:
            R_o2c = (R_wc_cv @ p['R_o2w']).astype(np.float32)
            t_o2c = (R_wc_cv @ p['t_o2w'] + t_wc_cv).astype(np.float32)
            inst_id = inst_map.get(str(p['obj_id']), -1)
            poses[inst_id] = {
                'obj_id': p['obj_id'],
                'R': R_o2c,
                't': t_o2c,
                'inst_id': inst_id,
                'scene_scale': p['scene_scale']
            }

        #print('obj_id2key', obj_id2key)
        #print('inst_map', inst_map)

        

        #print('depth_np before aug', np.min(depth_np[depth_np > 0]))
        img_t, depth_t, mask_t, intr_resized = self._apply_augs(img, depth_np, mask_np, intr, orig_size=(W0, H0), distortion=None)
        depth_t = clamp_zero(depth_t)
                
        #print('depth_t', torch.min(depth_t[depth_t > 0]))
        mask_np_resized = mask_t.numpy() if mask_t is not None else None
        boxes_xyxy, boxes_cxcywh = self._build_targets(mask_np_resized)
        poses = self._filter_visible_instances(mask_np_resized, poses)
        visible_inst_ids = list(poses.keys())

        # load object point clouds
        models = {}
        if self.include_models:
            for inst_id, p in poses.items():
                obj_id = p['obj_id']
                pts_m = self._load_assets_pc(obj_id)
                if pts_m is None or pts_m.shape[0] == 0:
                    continue                

                # Apply scene scale (per-axis) in meters
                s_vec = p.get('scene_scale', np.array([1.0, 1.0, 1.0], dtype=np.float32))
                if not isinstance(s_vec, np.ndarray):
                    s_vec = np.array(s_vec, dtype=np.float32)
                pts_scaled_m = pts_m * s_vec.reshape(1, 3).astype(np.float32)
                models[inst_id] = torch.from_numpy(pts_scaled_m.astype(np.float32))  # raw in meters scaled (asset-root frame)

        image_type = os.path.basename(os.path.dirname(meta['scene_dir']))
        

        props = self._build_out_models_pose_baked(poses, models)
        #print('poses_aug filter', poses_aug.keys())

        if self.with_grasp:

            
            #gt_grasps_by_inst, bad_grasps = self._load_grasps_hdf5(
            #    meta['scene_dir'],
            #    frame_dir,
            #    R_wc_cv, t_wc_cv,
            #    intr_resized if intr_resized is not None else intr,
            #    inst_map=inst_map,
            #    visible_inst_ids=visible_inst_ids,
            #    depth_m=depth_t.squeeze(0).cpu().numpy() if depth_t is not None else None
            #)

            gt_grasps_by_inst, bad_grasps = self._load_grasps_hdf5(
                scene_dir=meta['scene_dir'],
                frame_dir=frame_dir,
                R_wc_cv=R_wc_cv,
                t_wc_cv=t_wc_cv,
                intr=intr_resized if intr_resized is not None else intr,
                inst_map=inst_map,                      # asset obj_id -> inst_id (fallback)
                visible_inst_ids=visible_inst_ids,
                depth_m=depth_t.squeeze(0).cpu().numpy() if depth_t is not None else None,
                obj_id2key=obj_id2key                       # objectN -> inst_id (correct for HDF5/dataset.json)
            )

            #bad_grasps = {}
            #gt_grasps_by_inst, bad_grasps = self._load_grasps_hdf5(
            #    meta['scene_dir'],
            #    frame_dir,
            #    R_wc_cv, t_wc_cv,
            #    intr_resized if intr_resized is not None else intr,
            #    inst_map=inst_map,
            #    visible_inst_ids=visible_inst_ids
            #)  


            bad_grasps = self._ensure_min_bad_grasps(
                gt_grasps_by_inst=gt_grasps_by_inst,
                bad_grasps=bad_grasps,
                poses=poses,
                visible_inst_ids=visible_inst_ids,
                k_per_inst=10,
                default_width_m=0.08,
                intr=intr_resized if intr_resized is not None else intr,
                depth_m=depth_t.squeeze(0).cpu().numpy() if depth_t is not None else None,
                finger_len_m=0.05,
                models=models
            )            
        
            #gt_grasps_by_inst = bad_grasps
            #print('gt_grasps_by_inst', {key: len(v) for key, v in gt_grasps_by_inst.items()})
            #print('bad_grasps', {key: len(v) for key, v in bad_grasps.items()})



        #print('graspit getitem', {key: len(v) for key, v in gt_grasps_by_inst.items()}, {key: len(v) for key, v in bad_grasps.items()})

        sample = {
            'dataset': self.name,
            'image_id': f"{os.path.basename(meta['scene_dir'])}/{os.path.basename(meta['frame_dir'])}",
            'image_type': image_type,
            'img': img_t,
            'depth': depth_t,
            'mask': mask_t,
            'boxes_xyxy': torch.from_numpy(boxes_xyxy),
            'boxes_cxcywh': torch.from_numpy(boxes_cxcywh),
            'intr': torch.tensor(intr_resized, dtype=torch.float32) if intr_resized is not None else None,
            'poses': poses,
            'models': models,
            'gt_grasps': gt_grasps_by_inst,
            'bad_grasps': bad_grasps,
            'props': props,
            'H': self.cfg.out_hw[0], 'W': self.cfg.out_hw[1]
        }
        return sample 

    @staticmethod
    def inventory(root: str) -> Dict[str, int]:
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



# =========================

# GraspNet Dataset

# =========================
class GraspNetDataset(BasePoseDataset):
    def __init__(self, root: str, 
                 mode='val',
                 transforms=True,
                 include_depth=True,
                 include_mask=True,
                 include_models=True,
                 include_grasp_labels=False,
                 grasp_label_source: str = '6d',
                 with_grasp: bool = True,
                 mask_xml_match_policy='index_order',
                 models_root: Optional[str] = None,
                 rectify_prob: float = 0.0,
                 min_visible_pixels: int = 50,
                 len_target_pc: int = 512,
                 depth_noise_sigma_m: float = 0.0,
                 cache_grasps: bool = True,      
                 overwrite_grasps: bool = True, 
                 cfg: Optional[DatasetConfig] = None):
        super().__init__(mode=mode, transforms=transforms, rectify_prob=rectify_prob, min_visible_pixels=min_visible_pixels, len_target_pc=len_target_pc, cfg=cfg, depth_noise_sigma_m=depth_noise_sigma_m)
        self.name = 'graspnet'
        self.root = root
        self.rect_labels_root = os.path.join(root, 'rect_labels')
        self.grasp_labels_root = os.path.join(root, 'grasp_label')
        self.include_depth = include_depth
        self.include_mask = include_mask
        self.include_models = include_models
        self.include_grasp_labels = include_grasp_labels
        self.grasp_label_source = grasp_label_source 
        self.with_grasp = with_grasp
        self.mask_xml_match_policy = mask_xml_match_policy
        self.models_root = models_root if models_root is not None else os.path.join(root, 'models')
        self.cache_grasps = bool(cache_grasps)
        self.overwrite_grasps = bool(overwrite_grasps)


        self._gn_by_sensor = {}  
        

        self.samples = []

        # Decide if we base the split on train scenes (*0 to val) or use default test_seen

        has_labels = bool(self.grasp_label_source) and str(self.grasp_label_source).lower() != 'none' and self.with_grasp


        if has_labels:
            # Use only train_* for both train and val; the %10 rule selects which goes where
            split_dirs = [d for d in os.listdir(self.root)
                        if os.path.isdir(os.path.join(self.root, d)) and d.startswith('scenes')]
        else:
            # Default legacy behavior
            if self.mode == 'train':
                split_dirs = [d for d in os.listdir(self.root)
                            if os.path.isdir(os.path.join(self.root, d)) and d.startswith('scenes')]
            else:  # 'val' or 'test'
                split_dirs = ['test_seen'] if os.path.isdir(os.path.join(self.root, 'test_seen')) else []

        for split in split_dirs:
            split_path = os.path.join(self.root, split)
            if not os.path.isdir(split_path):
                continue
            for scene_dir in sorted(glob.glob(os.path.join(split_path, 'scene_*'))):
                scene_name = os.path.basename(scene_dir)

                if has_labels:
                    scene_num = int(scene_name.split('_')[-1])

                    if scene_num > 87:
                        continue
                    
                    if self.mode == 'val':
                        if (scene_num % 10) != 0:
                            continue
                    elif self.mode == 'train':
                        if (scene_num % 10) == 0:
                            continue

                #print('adding', split_path, scene_name)
                        

                for sensor in ['realsense', 'kinect']:
                    base = os.path.join(scene_dir, sensor)
                    rgb_dir   = os.path.join(base, 'rgb')
                    depth_dir = os.path.join(base, 'depth')
                    label_dir = os.path.join(base, 'label')
                    annot_dir = os.path.join(base, 'annotations')
                    if not (os.path.isdir(rgb_dir) and os.path.isdir(depth_dir) and os.path.isdir(label_dir)):
                        continue
                    camK_path = os.path.join(base, 'camK.npy')
                    for img_name in sorted(os.listdir(rgb_dir)):
                        if not (img_name.endswith('.png') or img_name.endswith('.jpg')):
                            continue
                        stem = os.path.splitext(img_name)[0]
                        rgb_path = os.path.join(rgb_dir, img_name)
                        depth_path = os.path.join(depth_dir, stem + '.png')
                        label_path = os.path.join(label_dir, stem + '.png')
                        xml_path   = os.path.join(annot_dir, stem + '.xml')
                        self.samples.append({
                            'split': split,
                            'scene': scene_name,
                            'sensor': sensor,
                            'rgb': rgb_path,
                            'depth': depth_path,
                            'label': label_path,
                            'xml': xml_path,
                            'camK': camK_path if os.path.exists(camK_path) else None
                        })     

    def __len__(self):
        return len(self.samples)

    def _scene_has_rect(self, scene_name: str) -> bool:
        """
        True if rect_labels/scene_xxxx has any npy for kinect or realsense.
        scene_name is like 'scene_0100'.
        """
        try:
            scene_num = int(scene_name.split('_')[-1])
        except Exception:
            return False
        for sensor in ['realsense', 'kinect']:
            sdir = os.path.join(self.rect_labels_root, f"scene_{scene_num:04d}", sensor)
            if os.path.isdir(sdir):
                for f in os.listdir(sdir):
                    if f.endswith('.npy'):
                        return True
        return False

    def _scene_has_6d(self, scene_dir: str, scene_name: str) -> bool:
        scene_num = int(scene_name.split('_')[-1])
        for sensor in ['realsense', 'kinect']:
            rgb_dir = os.path.join(scene_dir, sensor, 'rgb')
            if not os.path.isdir(rgb_dir):
                continue
            frame_ids = []
            for fn in os.listdir(rgb_dir):
                if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                stem = os.path.splitext(fn)[0]
                try:
                    fid = int(''.join([c for c in stem if c.isdigit()]))
                except Exception as e:
                    print('Error', e)
                    continue
                frame_ids.append(fid)
            if not frame_ids:
                continue
            gn = self._get_graspnet_api(sensor)
            if gn is None:
                continue
            for fid in {min(frame_ids), frame_ids[len(frame_ids)//2], max(frame_ids)}:
                try:
                    gg = gn.loadGrasp(scene_num, fid)
                    if gg is not None and len(gg) > 0:
                        return True
                except Exception as e:
                    print('Error', e)
                    pass
        return False
    
    def _get_graspnet_api(self, sensor: str):
        if sensor in self._gn_by_sensor:
            return self._gn_by_sensor[sensor]
        try:
            gn = GraspNet(self.root, camera=sensor, split='train')
            label_dir = self.grasp_labels_root
            if hasattr(gn, 'grasp_label_folder'):
                gn.grasp_label_folder = label_dir
            if hasattr(gn, 'grasp_label_dir'):
                gn.grasp_label_dir = label_dir
            if hasattr(gn, 'collision_label_folder'):
                gn.collision_label_folder = os.path.join(self.root, 'collision_label')
            self._gn_by_sensor[sensor] = gn
            return gn
        except Exception:
            print(f"no grasp api for {sensor}")
            return None

            
    def _canonicalize_grasp_R_t(self,
                            R_in: np.ndarray,
                            t_in: np.ndarray,
                            intr: Optional[list] = None,
                            depth_m: Optional[np.ndarray] = None,
                            use_depth_normal: bool = True,
                            finger_len_m: float = 0.05,
                            shift_tcp: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Return R_can, t_can where:
        - column 2 (Z) is the approach axis (pointing into scene/object),
        - column 0 (X) is width axis orthogonal to Z,
        - column 1 (Y) completes right-handed frame.

        If shift_tcp=True, shift TCP to palm center by -finger_len_m along local +Z.
        """
        R = np.asarray(R_in, dtype=np.float32)
        t = np.asarray(t_in, dtype=np.float32)

        # 1) Estimate desired approach direction
        Z_des = None
        if use_depth_normal and (intr is not None) and (depth_m is not None):
            # Project center pixel and estimate normal
            cx, cy, fx, fy = intr
            uv = project_points(t.reshape(1, 3), np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), intr)[0]
            u, v, valid = float(uv[0]), float(uv[1]), bool(uv[2])
            if valid:
                n = self._estimate_normal_from_depth(depth_m, u, v, intr, win=5)
                if n is not None and np.linalg.norm(n) > 1e-6:
                    Z_des = n / (np.linalg.norm(n) + 1e-9)

        if Z_des is None:
            # Fallback: use camera→object ray
            ray = t / (np.linalg.norm(t) + 1e-9)
            Z_des = ray

        # 2) Select the column of R closest to Z_des (in absolute cosine)
        cols = [R[:, 0], R[:, 1], R[:, 2]]
        dots = [abs(float(np.dot(c / (np.linalg.norm(c) + 1e-9), Z_des))) for c in cols]
        idx_z = int(np.argmax(dots))
        Z = cols[idx_z] / (np.linalg.norm(cols[idx_z]) + 1e-9)

        # Ensure Z points roughly camera→object (positive dot)
        if float(np.dot(Z, t)) < 0.0:
            Z = -Z

        # 3) Build X as "width axis" by projecting the original width candidate onto plane ⟂ Z
        idx_width = 0 if idx_z != 0 else 1  # pick another column as width candidate
        Xcand = cols[idx_width]
        X = Xcand - Z * float(np.dot(Xcand, Z))
        nX = np.linalg.norm(X)
        if nX < 1e-6:
            # Fallback to any axis orthogonal to Z
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(tmp, Z))) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            X = tmp - Z * float(np.dot(tmp, Z))
            nX = np.linalg.norm(X)
        X /= (nX + 1e-9)

        # 4) Complete right-handed frame
        Y = np.cross(Z, X)
        Y /= (np.linalg.norm(Y) + 1e-9)
        R_can = np.stack([X, Y, Z], axis=1).astype(np.float32)

        # 5) Optional TCP shift (for GraspNet: usually False)
        if shift_tcp:
            t = t - finger_len_m * Z

        return R_can, t.astype(np.float32)

    def _project_uv(self, xyz: np.ndarray, intr: list) -> tuple[int, int, float, bool]:
        cx, cy, fx, fy = intr
        X, Y, Z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        if Z <= 1e-6:
            return 0, 0, Z, False
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        return int(round(u)), int(round(v)), Z, True

    def _is_visible_grasp(self, t_cam_m: np.ndarray, intr: list, depth_m: np.ndarray,
                        H: int, W: int, patch: int = 3, margin_m: float = 0.01) -> bool:
        ui, vi, Zg, ok = self._project_uv(t_cam_m, intr)
        if not ok: return False
        if ui < 0 or ui >= W or vi < 0 or vi >= H: return False
        d = self._depth_at(depth_m, float(ui), float(vi), patch=patch)
        if d <= 0: return False
        # Visible if grasp depth is not significantly behind the measured surface (simple occlusion test)
        return Zg <= (d + margin_m)

    def _visible_cache_path(self, scene_num: int, sensor: str, frame_id: int) -> str:
        return os.path.join(self.root, "visible_grasps", f"scene_{scene_num:04d}", sensor, f"{frame_id:04d}.npz")

    def _save_visible_grasps_npz(self, path_npz: str, good_by_inst: dict, bad_by_inst: dict) -> None:
        os.makedirs(os.path.dirname(path_npz), exist_ok=True)
        def flatten(dct):
            inst_id, obj_id, xyz, quat, width, score, label = [], [], [], [], [], [], []
            for iid, lst in (dct or {}).items():
                for g in lst:
                    inst_id.append(int(iid))
                    obj_id.append(int(g.get('obj_id', -1)))
                    xyz.append(np.asarray(g['xyz'], dtype=np.float32))
                    quat.append(np.asarray(g['quat_wxyz'], dtype=np.float32))
                    width.append(float(g.get('width', 0.0)))
                    score.append(float(g.get('score', 0.0)))
                    label.append(1 if g.get('score', 0.0) < 0.5 else 0)  # 0=good,1=bad
            return (np.asarray(inst_id, np.int32),
                    np.asarray(obj_id, np.int32),
                    np.stack(xyz, axis=0).astype(np.float32) if xyz else np.empty((0,3), np.float32),
                    np.stack(quat, axis=0).astype(np.float32) if quat else np.empty((0,4), np.float32),
                    np.asarray(width, np.float32),
                    np.asarray(score, np.float32),
                    np.asarray(label, np.int32))
        gi = flatten(good_by_inst)
        bi = flatten(bad_by_inst)
        np.savez_compressed(path_npz,
            inst_good=gi[0], obj_good=gi[1], xyz_good=gi[2], quat_good=gi[3], width_good=gi[4], score_good=gi[5], label_good=gi[6],
            inst_bad=bi[0], obj_bad=bi[1], xyz_bad=bi[2], quat_bad=bi[3], width_bad=bi[4], score_bad=bi[5], label_bad=bi[6],
        )

    def _load_visible_grasps_npz(self, path_npz: str) -> tuple[dict, dict]:
        if not os.path.exists(path_npz): return {}, {}
        D = np.load(path_npz)
        good, bad = defaultdict(list), defaultdict(list)
        for key_prefix in ['good', 'bad']:
            inst = D[f'inst_{key_prefix}']
            obj  = D[f'obj_{key_prefix}']
            xyz  = D[f'xyz_{key_prefix}']
            quat = D[f'quat_{key_prefix}']
            width= D[f'width_{key_prefix}']
            score= D[f'score_{key_prefix}']
            tgt = good if key_prefix == 'good' else bad
            for i in range(inst.shape[0]):
                iid = int(inst[i])
                tgt[iid].append({
                    'obj_id': int(obj[i]),
                    'inst_id': iid,
                    'xyz': xyz[i].astype(np.float32),
                    'quat_wxyz': quat[i].astype(np.float32).tolist(),
                    'width': float(width[i]),
                    'score': float(score[i]),
                })
        return dict(good), dict(bad)

    def _load_6d_grasps_visible(
        self,
        scene_num: int,
        sensor: str,
        frame_id: int,
        obj2inst: dict[int, int],
        visible_inst_ids: list | None = None,
        intr: list | None = None,
        depth_m: np.ndarray | None = None,
        finger_len_m: float = 0.05,
        good_score: float = 0.95,
        margin_m: float = 0.01,
        patch: int = 3,
        cache_grasps: Optional[bool] = None,
        overwrite_cache: Optional[bool] = None,
    ) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
        """
        Load 6D grasps and keep only visible ones (via depth/intr occlusion test).
        Uses cache if enabled; saves filtered results back to cache.

        Returns: (good_by_inst, bad_by_inst) dicts keyed by inst_id.
        """

        if not self.with_grasp:
            return {}, {}cache_grasp

        # Resolve caching flags
        cache_grasps = self.cache_grasps if cache_grasps is None else bool(cache_grasps)
        overwrite_cache = self.overwrite_grasps if overwrite_cache is None else bool(overwrite_cache)

        # Cache path
        cache_path = self._visible_cache_path(scene_num, sensor, frame_id)

        # Load cache if available and not overwriting
        if cache_grasps and (not overwrite_cache) and os.path.exists(cache_path):
            try:
                good_by_inst, bad_by_inst = self._load_visible_grasps_npz(cache_path)
                
                print('load graps from {} with {} bad and {} good grasps'.format(cache_path, sum([len(v) for v in bad_by_inst.values()]), sum([len(v) for v in good_by_inst.values()])))
                return good_by_inst, bad_by_inst
            except Exception:
                # Fall through and recompute
                pass

        # Query GraspNet API
        gn = self._get_graspnet_api(sensor)
        if gn is None:
            return {}, {}

        gg = gn.loadGrasp(sceneId=scene_num, annId=frame_id, format='6d', camera=sensor, fric_coef_thresh=0.4)
        
        arr = getattr(gg, 'grasp_group_array', None)
        if arr is None:
            arr = gg.to_numpy()
        if arr is None or arr.ndim != 2 or arr.shape[1] != 17:
            return {}, {}

        # Unpack GraspNet array
        scores    = arr[:, 0].astype(np.float32)

        widths_m  = arr[:, 1].astype(np.float32)
        heights_m = arr[:, 2].astype(np.float32)
        depths_m  = arr[:, 3].astype(np.float32)
        R_api     = arr[:, 4:13].reshape(-1, 3, 3).astype(np.float32)  # cols: [X(approach), Y(width), Z(height)]
        t_cam_m   = arr[:, 13:16].astype(np.float32)
        objids    = arr[:, 16].astype(np.int32)

        # Desired axes: Y_out = Y_api, Z_out = X_api, X_out = Y_out × Z_out
        Y = R_api[:, :, 1].copy()
        Z = R_api[:, :, 0].copy()
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
        Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        X = np.cross(Y, Z)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        R_out = np.stack([X, Y, Z], axis=2).astype(np.float32)

        # Shift TCP along +Z_out by max(0, finger_len - height)
        shift = np.maximum(0.0, (finger_len_m - heights_m)).astype(np.float32)
        t_out = t_cam_m - (R_out[:, :, 2] * shift[:, None])

        if visible_inst_ids is None:
            visible_inst_ids = []

        H = depth_m.shape[0] if isinstance(depth_m, np.ndarray) and depth_m.ndim == 2 else 0
        W = depth_m.shape[1] if isinstance(depth_m, np.ndarray) and depth_m.ndim == 2 else 0
        have_visibility = (intr is not None) and isinstance(depth_m, np.ndarray) and depth_m.ndim == 2 and H > 0 and W > 0

        good_by_inst = defaultdict(list)
        bad_by_inst  = defaultdict(list)

      
        for i in range(arr.shape[0]):
            oid = int(objids[i])
            inst_id = obj2inst.get(oid, -1)
            if inst_id not in visible_inst_ids:
                continue

            # Visibility check
            is_vis = True
            if have_visibility:
                is_vis = self._is_visible_grasp(
                    t_out[i], intr, depth_m, H, W, patch=patch, margin_m=margin_m
                )
            if not is_vis:                
                continue

            q_wxyz = rotation_matrix_to_quaternion_wxyz(R_out[i])
            s = float(scores[i])
            g = {
                'obj_id': oid,
                'inst_id': inst_id,
                'xyz': t_out[i].astype(np.float32),
                'quat_wxyz': [float(q) for q in q_wxyz],
                'width': float(widths_m[i]),
                'score': s,
                'height': float(heights_m[i]),
                'depth': float(depths_m[i]),
            }

            if s >= good_score:
                good_by_inst[inst_id].append(g)
            else:
                bad_by_inst[inst_id].append(g)
                

        # Save cache
        if cache_grasps:
            try:
                print('saving graps to {} with {} bad and {} good grasps'.format(cache_path, sum([len(v) for v in bad_by_inst.values()]), sum([len(v) for v in good_by_inst.values()])))

                self._save_visible_grasps_npz(cache_path, dict(good_by_inst), dict(bad_by_inst))
            except Exception as e:
                print(f"[WARN] Failed to save visible grasps cache at {cache_path}: {e}")

        return dict(good_by_inst), dict(bad_by_inst)
        
       
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

    def _rect_labels_path(self, scene_num: int, sensor: str, frame_id: int) -> str:
        # Files are zero-padded to 4 digits, e.g., 0009.npy
        return os.path.join(self.rect_labels_root, f"scene_{int(scene_num):04d}", sensor, f"{int(frame_id):04d}.npy")

    def _depth_at(self, depth_m: np.ndarray, u: float, v: float, patch: int = 3) -> float:
        H, W = depth_m.shape
        ui = int(round(u)); vi = int(round(v))
        x1 = max(0, ui - patch); x2 = min(W, ui + patch + 1)
        y1 = max(0, vi - patch); y2 = min(H, vi + patch + 1)
        d = depth_m[y1:y2, x1:x2]
        d = d[d > 0]
        return float(np.median(d)) if d.size > 0 else 0.0

    def _backproject_uvd(self, u: float, v: float, d_m: float, intr: list) -> np.ndarray:
        cx, cy, fx, fy = intr
        if d_m <= 0:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        X = (u - cx) * d_m / fx
        Y = (v - cy) * d_m / fy
        Z = d_m
        return np.array([X, Y, Z], dtype=np.float32)

    def _compute_upper_point_2d(self, center_uv: np.ndarray, open_uv: np.ndarray, height_px: float) -> np.ndarray:
        vec = open_uv - center_uv
        n = np.linalg.norm(vec)
        if n < 1e-9 or height_px <= 1e-9:
            return center_uv.copy()
        unit = vec / n
        rot90 = np.array([[0, -1], [1, 0]], dtype=np.float32)  # CCW 90°
        up_dir = rot90 @ unit
        return center_uv + up_dir * (height_px / 2.0)

    def _rotation_from_keypoints(self, center_xyz: np.ndarray, open_xyz: np.ndarray, upper_xyz: np.ndarray) -> np.ndarray | None:
        vx = open_xyz - center_xyz
        vy = upper_xyz - center_xyz
        nx = np.linalg.norm(vx); ny = np.linalg.norm(vy)
        if nx < 1e-9 or ny < 1e-9:
            return None
        x = vx / nx
        y = vy / ny
        z = np.cross(x, y)
        nz = np.linalg.norm(z)
        if nz < 1e-9:
            return None
        z = z / nz
        y = np.cross(z, x)
        ny = np.linalg.norm(y)
        if ny < 1e-9:
            return None
        y = y / ny
        return np.stack([x, y, z], axis=1).astype(np.float32)


    def _estimate_normal_from_depth(self, depth_m: np.ndarray, u: float, v: float, intr: list, win: int = 5):
        # Clamp to valid integer pixel
        H, W = depth_m.shape
        ui = int(round(u)); vi = int(round(v))
        ui = np.clip(ui, 0, W - 1); vi = np.clip(vi, 0, H - 1)

        # Build a window with at least 3x3 pixels
        half = max(1, int(win))
        x1 = max(0, ui - half); x2 = min(W, ui + half + 1)
        y1 = max(0, vi - half); y2 = min(H, vi + half + 1)

        # If the window collapses at edges, expand inward to ensure size
        if (x2 - x1) < 3:
            if ui <= 1: x2 = min(W, x1 + 3)
            elif ui >= W - 2: x1 = max(0, x2 - 3)
            else:
                # center case: expand both sides
                need = 3 - (x2 - x1)
                x1 = max(0, x1 - need // 2)
                x2 = min(W, x2 + need - need // 2)
        if (y2 - y1) < 3:
            if vi <= 1: y2 = min(H, y1 + 3)
            elif vi >= H - 2: y1 = max(0, y2 - 3)
            else:
                need = 3 - (y2 - y1)
                y1 = max(0, y1 - need // 2)
                y2 = min(H, y2 + need - need // 2)

        patch = depth_m[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        # Valid depth mask
        m = patch > 0
        if not np.any(m):
            return None

        # Pixel coordinates for valid points
        yy, xx = np.indices(patch.shape)
        yy = (yy + y1)[m].astype(np.float32)
        xx = (xx + x1)[m].astype(np.float32)
        ds = patch[m].astype(np.float32)

        # Backproject
        cx, cy, fx, fy = intr
        X = (xx - cx) * ds / fx
        Y = (yy - cy) * ds / fy
        Z = ds
        P = np.stack([X, Y, Z], axis=1).astype(np.float32)
        if P.shape[0] < 3:
            return None

        # Fit plane via SVD
        C = P.mean(axis=0)
        Q = P - C
        try:
            _, _, vh = np.linalg.svd(Q, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        n = vh[-1, :].astype(np.float32)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            return None
        n /= n_norm
        return n
        
    def _rect_grasps_to_6d(self, rect_array: np.ndarray, depth_m: np.ndarray, intr: list,
                        mask_np: np.ndarray | None = None) -> list[dict]:
        """
        rect_array: [N,7] = [center_x, center_y, open_x, open_y, height_px, score, object_id]
        depth_m: HxW in meters; intr: [cx,cy,fx,fy]
        """
        if rect_array is None or rect_array.ndim != 2 or rect_array.shape[1] != 7:
            return []
        out = []
        for r in rect_array.astype(np.float32):
            cx, cy, ox, oy, h, score, obj_id = r
            center_uv = np.array([cx, cy], dtype=np.float32)
            open_uv   = np.array([ox, oy], dtype=np.float32)
            upper_uv  = self._compute_upper_point_2d(center_uv, open_uv, float(h))

            d_center = self._depth_at(depth_m, cx, cy, patch=3)
            if d_center <= 0:
                continue

            c_xyz = self._backproject_uvd(cx, cy, d_center, intr)
            o_xyz = self._backproject_uvd(ox, oy, d_center, intr)  # same depth as center
            u_xyz = self._backproject_uvd(upper_uv[0], upper_uv[1], d_center, intr)
            if np.any(np.isnan(c_xyz)) or np.any(np.isnan(o_xyz)) or np.any(np.isnan(u_xyz)):
                continue

            R = self._rotation_from_keypoints(c_xyz, o_xyz, u_xyz)
            if R is None:
                continue           

            n = self._estimate_normal_from_depth(depth_m, cx, cy, intr, win=5)
            if n is None:
                # Fallback: keep previous method, but this yields near-constant Z
                X = (o_xyz - c_xyz)
                X /= (np.linalg.norm(X) + 1e-9)
                # Guess Z from camera ray as last resort
                Z = c_xyz / (np.linalg.norm(c_xyz) + 1e-9)
            else:
                # Approach from surface normal, disambiguate sign to point into the scene
                Z = n
                if np.dot(Z, c_xyz) < 0:  # want Z roughly along camera->object
                    Z = -Z

            # Project width axis orthogonal to Z

            X = o_xyz - c_xyz
            X = X - Z * np.dot(X, Z)
            nx = np.linalg.norm(X)
            if nx < 1e-9:
                # Safety fallback: pick any direction orthogonal to Z
                tmp = np.array([1, 0, 0], dtype=np.float32)
                if abs(np.dot(tmp, Z)) > 0.9:
                    tmp = np.array([0, 1, 0], dtype=np.float32)
                X = tmp - Z * np.dot(tmp, Z)
                nx = np.linalg.norm(X)
            X /= (nx + 1e-9)

            # Complete right-handed frame

            Y = np.cross(Z, X)
            Y /= (np.linalg.norm(Y) + 1e-9)

            R = np.stack([X, Y, Z], axis=1).astype(np.float32)

            
            c0, o0, u0 = c_xyz.copy(), o_xyz.copy(), u_xyz.copy()

            # Half-height in meters (since upper_uv is built with height_px/2)

            h_m = float(np.linalg.norm(u0 - c0))

            # Nominal finger length along approach; reduce by half-height

            finger_len_m = 0.05  # keep consistent with visualize/_draw_gripper
            shift_m = max(0.0, finger_len_m - h_m)

            # Shift TCP (palm center) along +Z so fingertips align near the annotated upper band

            c_xyz = c0 - shift_m * Z
            o_xyz = o0 - shift_m * Z
            u_xyz = u0 - shift_m * Z

            # Quaternion from R

            q_xyzw = SciRot.from_matrix(R).as_quat()
            q_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]

            # Opening width from unshifted points

            width_m = float(2.0 * np.linalg.norm(o0 - c0))

            inst_id = -1
            if mask_np is not None:
                H, W = mask_np.shape[:2]
                ui, vi = int(round(cx)), int(round(cy))
                if 0 <= vi < H and 0 <= ui < W:
                    inst_id = int(mask_np[vi, ui])

            out.append({
                'obj_id': int(round(obj_id)),
                'inst_id': inst_id,
                'xyz': c_xyz.astype(np.float32),
                'quat_wxyz': q_wxyz,
                'width': width_m,
                'score': float(score),
                'center_uv': [float(cx), float(cy)]
            })
        return out
    
    def _project_pc_mask(self, pts_m: np.ndarray, R: np.ndarray, t: np.ndarray, intr: List[float], H: int, W: int,
                        dilate_ksize: int = 5) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Project a point cloud (object -> camera) and build a binary image mask; also returns 2D center (u,v).
        """
        uv = project_points(pts_m, R, t, intr)  # [N,3] (u,v,valid)
        uv = uv[uv[:, 2] > 0]
        if uv.shape[0] == 0:
            return np.zeros((H, W), dtype=bool), None
        u = np.round(uv[:, 0]).astype(np.int32)
        v = np.round(uv[:, 1]).astype(np.int32)
        m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[m]; v = v[m]
        if u.size == 0:
            return np.zeros((H, W), dtype=bool), None
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[v, u] = 255
        if dilate_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
            mask = cv2.dilate(mask, k, iterations=1)
        mask_bool = mask.astype(bool)
        center = np.array([u.mean(), v.mean()], dtype=np.float32) if u.size > 0 else None
        return mask_bool, center


    def _load_graspnet_model(self, obj_name: str) -> np.ndarray:
        obj_dir = os.path.join(self.models_root, str(obj_name).zfill(3))
        if not os.path.isdir(obj_dir):
            return np.zeros((0,3), dtype=np.float32)
        cand = None
        for fname in ['nontextured_simplified.ply', 'nontextured_simplified.obj', 'nontextured.ply', 'textured.obj']:
            fp = os.path.join(obj_dir, fname)
            if os.path.exists(fp):
                cand = fp
                break
        if cand is None:
            return np.zeros((0,3), dtype=np.float32)
        base = os.path.splitext(os.path.basename(cand))[0]
        out_pc = os.path.join(obj_dir, f"{base}_pc{self.cfg.pc_num_points}.obj")
        out_pc_path = ensure_downsampled_mesh_to_pc(cand, out_pc, num_points=self.cfg.pc_num_points)
        if out_pc_path is None:
            return np.zeros((0,3), dtype=np.float32)
        pts = load_pointcloud_file(out_pc_path, pc_num_points=self.cfg.pc_num_points)
        return pts if pts is not None else np.zeros((0,3), dtype=np.float32)

    def _filter_grasps_split(self, grasps: list[dict], mask_np: np.ndarray | None,
                            min_score: float = 0.5,
                            width_bounds: tuple[float, float] = (0.02, 0.10),
                            max_center_offset_px: float = 30.0) -> tuple[list[dict], list[dict]]:
        goods, bads = [], []
        if not grasps:
            return goods, bads

        inst_centroid = {}
        if mask_np is not None:
            ids = [int(i) for i in np.unique(mask_np) if i != 0]
            for inst_id in ids:
                ys, xs = np.where(mask_np == inst_id)
                if xs.size > 0:
                    inst_centroid[inst_id] = np.array([xs.mean(), ys.mean()], dtype=np.float32)

        w_min, w_max = width_bounds
        for g in grasps:
            ok = True
            s = float(g.get('score', 0.0))
            w = float(g.get('width', 0.0))
            if s < min_score:
                ok = False
            if not (w_min <= w <= w_max):
                ok = False
            if mask_np is not None and 'center_uv' in g and int(g.get('inst_id', -1)) in inst_centroid:
                ic = inst_centroid[int(g['inst_id'])]
                cuv = np.asarray(g['center_uv'], dtype=np.float32)
                #if np.linalg.norm(cuv - ic) > max_center_offset_px:
                #    ok = False
            (goods if ok else bads).append(g)
        return goods, bads

    def __getitem__(self, index: int):
        meta = self.samples[index]

        # RGB
        img = Image.open(meta['rgb']).convert('RGB')
        W0, H0 = img.size

        # Depth (meters)
        depth_np = np.array(Image.open(meta['depth']), dtype=np.float32) * 0.001 if self.include_depth else None

        # Mask (instance ids normalized to 1..N)
        mask_np = np.array(Image.open(meta['label']), dtype=np.uint8) if self.include_mask else None
        if mask_np is not None:
            inst_ids = sorted([i for i in np.unique(mask_np) if i != 0])
            id_map = {v: i + 1 for i, v in enumerate(inst_ids)}
            mask_np = np.vectorize(lambda v: id_map.get(v, 0))(mask_np).astype(np.uint8)

        # Camera intrinsics
        intr = self._load_intr(meta['camK'])

        # Load poses from XML (object->camera)
        poses_xml = self._load_xml_poses(meta['xml'])  # [{'obj_id','R','t'}...]
        
        # Convert poses list → dict keyed by inst_id
        poses = {}
        obj2inst = {}
        for i, p in enumerate(poses_xml):
            inst_id = i+1
            obj2inst[int(p['obj_id'])] = inst_id
            poses[inst_id] = {
                'obj_id': str(p['obj_id']),
                'R': p['R'].astype(np.float32),
                't': p['t'].astype(np.float32),
                'inst_id': inst_id,
                'scene_scale': np.array([1.0, 1.0, 1.0], dtype=np.float32)
            } 

        #print('obj2inst', obj2inst)

        # Filter poses to only visible inst_ids
        poses = self._filter_visible_instances(mask_np, poses)
        visible_inst_ids = list(poses.keys())
        

        # Grasp labels (use original not resized intr + mask + depth)
        gt_grasps_by_inst = {}
        # scene number and frame id from file names
        scene_num = int(str(meta['scene']).split('_')[-1])
        stem = os.path.splitext(os.path.basename(meta['rgb']))[0]
        try:
            frame_id = int(re.sub(r'\D', '', stem))
        except Exception:
            frame_id = int(stem)
        # Use ORIGINAL intr + ORIGINAL mask (before resize) for inst assignment

        gt_grasps_by_inst, bad_grasps = self._load_6d_grasps_visible(
            scene_num=scene_num,
            sensor=meta['sensor'],
            frame_id=frame_id,
            obj2inst=obj2inst,
            visible_inst_ids=visible_inst_ids,
            intr=intr,              # original intr (before resize)
            depth_m=depth_np if depth_np is not None else np.zeros((H0, W0), dtype=np.float32)
        )
        
        #print('gt_grasps_by_inst', {key: len(v) for key, v in gt_grasps_by_inst.items()})
        #for iid, gs in gt_grasps_by_inst.items():
        #    zs = [float(np.asarray(g['xyz'])[2]) for g in gs]
        #    print(f'inst {iid}: Z min/max (m) = {min(zs):.3f}/{max(zs):.3f}')

        # Optionally filter to visible inst ids
       
        # Apply augmentations (letterbox + color/noise)
        img_t, depth_t, mask_t, intr_resized = self._apply_augs(
            img, depth_np, mask_np, intr, orig_size=(W0, H0), distortion=None
        )
        depth_t = clamp_zero(depth_t)
        
        mask_np_resized = mask_t.numpy() if mask_t is not None else None

        # Targets (boxes)
        boxes_xyxy, boxes_cxcywh = self._build_targets(mask_np_resized)

        # Load models keyed by inst_id (raw points in meters)
        models = {}
        if self.include_models and len(visible_inst_ids) > 0:
            # Cache raw PCs by obj_id to avoid repeated loads
            raw_pc_cache = {}
            for inst_id in visible_inst_ids:
                p = poses.get(inst_id, None)
                if p is None:
                    continue
                obj_id = str(p['obj_id'])
                pts = raw_pc_cache.get(obj_id, None)
                if pts is None:
                    pts = self._load_graspnet_model(obj_id)  # meters
                    raw_pc_cache[obj_id] = pts
                if pts is None or pts.shape[0] == 0:
                    continue
                models[inst_id] = torch.from_numpy(pts.astype(np.float32))  # raw in meters

        # Build baked props (normalized-in-camera)
        props = self._build_out_models_pose_baked(poses, models)            

        image_type = meta['sensor']
        sample = {
            'dataset': self.name,
            'image_id': f"{meta['split']}/{meta['scene']}/{meta['sensor']}/{os.path.basename(meta['rgb'])}",
            'image_type': image_type,
            'img': img_t,
            'depth': depth_t,
            'mask': mask_t,
            'boxes_xyxy': torch.from_numpy(boxes_xyxy),
            'boxes_cxcywh': torch.from_numpy(boxes_cxcywh),
            'intr': torch.tensor(intr_resized, dtype=torch.float32) if intr_resized is not None else None,
            'poses': poses,          
            'models': models,      
            'gt_grasps': gt_grasps_by_inst,
            'bad_grasps': bad_grasps,
            'props': props,          
            'H': self.cfg.out_hw[0],
            'W': self.cfg.out_hw[1]
        }
        return sample

    def precompute_visible_grasps(self, max_workers: int = 8):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def _work(meta):
            img = Image.open(meta['rgb']).convert('RGB')
            W0, H0 = img.size
            intr = self._load_intr(meta['camK'])
            depth_np = np.array(Image.open(meta['depth']), dtype=np.float32) * 0.001 if self.include_depth else np.zeros((H0, W0), np.float32)
            poses_xml = self._load_xml_poses(meta['xml'])
            obj2inst = {int(p['obj_id']): i+1 for i, p in enumerate(poses_xml)}
            # visible_inst_ids from mask
            mask_np = np.array(Image.open(meta['label']), dtype=np.uint8) if self.include_mask else None
            if mask_np is not None:
                inst_ids = sorted([i for i in np.unique(mask_np) if i != 0])
                id_map = {v: i + 1 for i, v in enumerate(inst_ids)}
                mask_np = np.vectorize(lambda v: id_map.get(v, 0))(mask_np).astype(np.uint8)
                visible_inst_ids = list(range(1, len(inst_ids)+1))
            else:
                visible_inst_ids = list(obj2inst.values())
            # compute and cache
            scene_num = int(str(meta['scene']).split('_')[-1])
            stem = os.path.splitext(os.path.basename(meta['rgb']))[0]
            try:
                frame_id = int(re.sub(r'\D', '', stem))
            except Exception:
                frame_id = int(stem)
            self._load_6d_grasps_visible(
                scene_num, 
                meta['sensor'],
                frame_id, 
                obj2inst, 
                visible_inst_ids, 
                intr, 
                depth_np,
                cache_grasps=True,
                overwrite_cache=False
        )
        # Run
        for s in self.samples:
            _work(s)
        
        #with ThreadPoolExecutor(max_workers=max_workers) as ex:
        #    futs = [ex.submit(_work, s) for s in self.samples]
        #    for _ in as_completed(futs):
        #        pass


        #for s in self.samples:
        #    _work(s)
           

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
                        n = len([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
                        per_source[sensor] += n
                        total += n
        return {'total': total, 'per_source': dict(per_source)}

                


# =========================

# BOP Dataset

# =========================

class BOPDataset(BasePoseDataset):
    def __init__(self, bop_root: str, mode='train',
                transforms=True,
                include_depth=True,
                include_mask=True,
                include_models=True,
                interactive=False,
                use_source_probs=True,
                rectify_prob: float = 0.0,
                min_visible_pixels: int = 50,
                backgrounds_dirs: Optional[List[str]] = None,   
                bg_replace_prob: dict = {'default': 0.0},
                len_target_pc: int = 512,
                cfg: Optional[DatasetConfig] = None):                  
        super().__init__(mode=mode, transforms=transforms, rectify_prob=rectify_prob, min_visible_pixels=min_visible_pixels, len_target_pc=len_target_pc, cfg=cfg)
        self.name = 'bop'
        self.root = bop_root
        self.include_depth = include_depth
        self.include_mask = include_mask
        self.include_models = include_models
        self.interactive = interactive
        self.use_source_probs = use_source_probs

        # NEW: backgrounds
        self.bg_replace_prob = bg_replace_prob
        self.backgrounds = []
        if backgrounds_dirs is None:
            backgrounds_dirs = []
        bg_default = './data/backgrounds'
        if os.path.isdir(bg_default) and bg_default not in backgrounds_dirs:
            backgrounds_dirs.append(bg_default)
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        for d in backgrounds_dirs:
            if not os.path.isdir(d):
                continue
            for broot, bdirs, bfiles in os.walk(d):
                for f in bfiles:
                    fp = os.path.join(broot, f)
                    if os.path.isfile(fp) and os.path.splitext(fp)[1].lower() in exts:
                        self.backgrounds.append(fp)

        self.datasets = []
        self.source_types = {}
        self.source_probs = {}

        for d in sorted(os.listdir(bop_root)):
            dpath = os.path.join(bop_root, d)
            if os.path.isdir(dpath):
                self.datasets.append(d)

        self._build_source_inventory_and_probs()
        self.samples = self._enumerate_samples()
        self.samples_by_source = defaultdict(list)
        for s in self.samples:
            key = (s['ds_name'], s['source_type'])
            self.samples_by_source[key].append(s)

        for ds_name in self.datasets:
            model_eval_dir = os.path.join(self.root, ds_name, 'models_eval')
            model_dir = os.path.join(self.root, ds_name, 'models')
            if os.path.isdir(model_eval_dir):
                out_dir = os.path.join(self.root, ds_name, f"models_eval_pc_{self.cfg.pc_num_points}")
                ensure_downsampled_model_dir(model_eval_dir, out_dir, pattern_mesh=('*.ply','*.obj'), num_points=self.cfg.pc_num_points)
            elif os.path.isdir(model_dir):
                out_dir = os.path.join(self.root, ds_name, f"models_pc_{self.cfg.pc_num_points}")
                ensure_downsampled_model_dir(model_dir, out_dir, pattern_mesh=('*.ply','*.obj'), num_points=self.cfg.pc_num_points)

    def _apply_background_overlay(self, img_t: torch.Tensor, depth_t: Optional[torch.Tensor], mask_np_resized: Optional[np.ndarray], source_type: str = 'default', p_remove_depth: float = 1.0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Only apply if we have backgrounds, a mask, and probability triggers
        bg_replace_prob = self.bg_replace_prob.get(source_type, self.bg_replace_prob.get('default'))
        if bg_replace_prob is None:
            bg_replace_prob = 0.0

        if bg_replace_prob <= 0.0 or mask_np_resized is None or len(self.backgrounds) == 0:
            return img_t, depth_t
        if random.random() >= bg_replace_prob:
            return img_t, depth_t

        # Convert normalized tensor → PIL, composite, then re-normalize
        img_pil = denorm_to_pil(img_t, mean=self.cfg.rgb_mean, std=self.cfg.rgb_std)
        bg_path = random.choice(self.backgrounds)
        bg_pil = Image.open(bg_path).convert('RGB').resize(img_pil.size, Image.BILINEAR)

        img_np = np.array(img_pil)
        bg_np = np.array(bg_pil)
        bg_mask = (mask_np_resized == 0)  # background is where mask == 0

        out_np = img_np.copy()
        out_np[bg_mask] = bg_np[bg_mask]
        img_out = Image.fromarray(out_np)

        img_t_out = normalize_rgb(to_tensor_image(img_out), mean=self.cfg.rgb_mean, std=self.cfg.rgb_std)

        # Zero depth where background was overlaid
        if depth_t is not None:
            if p_remove_depth > random.random(): 
                depth_np = depth_t.squeeze(0).numpy()
                depth_np[bg_mask] = 0.0
                depth_t = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)

        return img_t_out, depth_t

    def _build_source_inventory_and_probs(self):
        inv = {}
        for ds in self.datasets:
            dpath = os.path.join(self.root, ds)
            stypes = []
            for sub in os.listdir(dpath):
                subp = os.path.join(dpath, sub)
                if os.path.isdir(subp):
                    stypes.append(sub)
            inv[ds] = sorted(stypes)
        self.source_types = inv

        samp_path = os.path.join(self.root, 'bop_sampling.json')
        old = read_json(samp_path, default=None)
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
                write_json(samp_path, {'source_probs': self.source_probs})

    def _enumerate_samples(self):
        samples = []
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
                scenes = [os.path.join(st_dir, f) for f in os.listdir(st_dir) if os.path.isdir(os.path.join(st_dir, f))]
                for scene_path in scenes:
                    scene_id = os.path.basename(scene_path)
                    rgb_dir = os.path.join(scene_path, 'rgb')
                    if not os.path.exists(rgb_dir):
                        rgb_dir = os.path.join(scene_path, 'gray')
                        if not os.path.exists(rgb_dir):
                            continue
                    mask_dir = os.path.join(scene_path, 'mask')
                    if not os.path.exists(mask_dir):
                        mask_dir = os.path.join(scene_path, 'mask_visib')
                        if not os.path.exists(mask_dir):
                            continue
                    depth_dir = os.path.join(scene_path, 'depth')
                    if not os.path.exists(depth_dir):
                        continue

                    cam_json = os.path.join(scene_path, 'scene_camera.json')
                    gt_json = os.path.join(scene_path, 'scene_gt.json')
                    gt_info_json = os.path.join(scene_path, 'scene_gt_info.json')

                    for img_file in sorted(os.listdir(rgb_dir)):
                        if not (img_file.endswith('.png') or img_file.endswith('.jpg')):
                            continue
                        im_stem = os.path.splitext(img_file)[0]
                        try:
                            im_id = int(im_stem)
                        except:
                            continue
                        rgb_path = os.path.join(rgb_dir, img_file)
                        depth_path = os.path.join(depth_dir, f"{im_id:06d}.png")
                        if not os.path.exists(depth_path):
                            continue
                        samples.append({
                            'ds_name': ds,
                            'source_type': st,
                            'scene_dir': scene_path,
                            'scene_id': scene_id,
                            'im_id': im_id,
                            'rgb': rgb_path,
                            'depth': depth_path if self.include_depth else None,
                            'scene_camera': cam_json,
                            'scene_gt': gt_json,
                            'scene_gt_info': gt_info_json if os.path.exists(gt_info_json) else None,
                            'mask_visib_dir': os.path.join(scene_path, 'mask_visib') if os.path.isdir(os.path.join(scene_path, 'mask_visib')) else None,
                            'mask_dir': os.path.join(scene_path, 'mask') if os.path.isdir(os.path.join(scene_path, 'mask')) else None,
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_bop_camera(scene_camera_json: str, im_id: int) -> Tuple[Optional[List[float]], float, Optional[List[float]]]:
        cam = read_json(scene_camera_json, default={})
        cam_im = cam.get(str(im_id), None)
        if cam_im is None:
            return None, 0.001, None
        K = np.array(cam_im['cam_K'], dtype=np.float32).reshape(3,3)
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        ds = float(cam_im.get('depth_scale', 1.0))
        ds = ds / 1000.0  # to meter
        dist = cam_im.get('cam_dist', cam_im.get('camD', cam_im.get('dist_coeffs', None)))
        if dist is not None and isinstance(dist, list):
            dist = [float(x) for x in dist]
        else:
            dist = None
        return [float(cx), float(cy), float(fx), float(fy)], ds, dist

    @staticmethod
    def _load_bop_gt(scene_gt_json: str, im_id: int) -> List[Dict[str,Any]]:
        gt = read_json(scene_gt_json, default={})
        items = gt.get(str(im_id), [])
        poses = {}
        for idx, it in enumerate(items):
            inst_id = idx + 1
            obj_id = str(it['obj_id'])
            R = np.array(it['cam_R_m2c'], dtype=np.float32).reshape(3,3)
            t = np.array(it['cam_t_m2c'], dtype=np.float32) * 0.001
            poses[inst_id] = {
                'obj_id': obj_id,
                 'R': R.astype(np.float32), 
                 't': t.astype(np.float32), 
                 'inst_id': inst_id,
                 'scene_scale': np.array([0.001, 0.001, 0.001], dtype=np.float32)
                 }
        return poses

    def _load_bop_mask_instance(self, sample: Dict[str, Any]) -> Optional[np.ndarray]:
        if sample['mask_visib_dir'] is None and sample['mask_dir'] is None:
            return None
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
        npts = self.cfg.pc_num_points
        md_eval_obj = glob.glob(os.path.join(self.root, ds_name, f"models_eval_pc_{npts}", f"obj_{int(obj_id):06d}*_pc{npts}.obj"))
        if md_eval_obj:
            return load_pointcloud_file(md_eval_obj[0], pc_num_points=npts)
        md_obj = glob.glob(os.path.join(self.root, ds_name, f"models_pc_{npts}", f"obj_{int(obj_id):06d}*_pc{npts}.obj"))
        if md_obj:
            return load_pointcloud_file(md_obj[0], pc_num_points=npts)
        md_eval = os.path.join(self.root, ds_name, f"models_eval_pc_{npts}", f"obj_{int(obj_id):06d}_pc{npts}.ply")
        md_eval_glob = glob.glob(os.path.join(self.root, ds_name, f"models_eval_pc_{npts}", f"obj_{int(obj_id):06d}*_pc{npts}.ply"))
        if os.path.exists(md_eval):
            return load_pointcloud_file(md_eval, pc_num_points=npts)
        if md_eval_glob:
            return load_pointcloud_file(md_eval_glob[0], pc_num_points=npts)
        md = os.path.join(self.root, ds_name, f"models_pc_{npts}", f"obj_{int(obj_id):06d}_pc{npts}.ply")
        md_glob = glob.glob(os.path.join(self.root, ds_name, f"models_pc_{npts}", f"obj_{int(obj_id):06d}*_pc{npts}.ply"))
        if os.path.exists(md):
            return load_pointcloud_file(md, pc_num_points=npts)
        if md_glob:
            return load_pointcloud_file(md_glob[0], pc_num_points=npts)
        # last-resort: build from mesh and save to pc dir
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
        out_dir = os.path.join(self.root, ds_name, f"models_eval_pc_{npts}" if 'models_eval' in cand_mesh else f"models_pc_{npts}")
        ensure_dir(out_dir)
        out_pc = os.path.join(out_dir, f"obj_{int(obj_id):06d}_pc{npts}.obj")
        out_path = ensure_downsampled_mesh_to_pc(cand_mesh, out_pc, num_points=npts)
        if out_path is None:
            return None
        return load_pointcloud_file(out_path, pc_num_points=npts)

    def __getitem__(self, index: int):
        # Sampling by source probs (unchanged)
        if self.use_source_probs and len(self.samples_by_source) > 0:
            ds_names = [ds for ds in self.datasets]
            keys, probs = [], []
            for ds in ds_names:
                sp = self.source_probs.get(ds, {})
                for st, p in sp.items():
                    if self.mode == 'train' and (('val' in st) or ('test' in st)):
                        continue
                    if self.mode == 'val' and ('val' not in st and 'test' not in st):
                        continue
                    if self.mode == 'test' and ('test' not in st):
                        continue
                    keys.append((ds, st))
                    probs.append(p)
            if len(keys) > 0:
                probs = np.array(probs, dtype=np.float32); probs = probs / probs.sum()
                choice = np.random.choice(np.arange(len(keys)), p=probs)
                key = keys[choice]
                pool = self.samples_by_source.get(key, [])
                sample_meta = random.choice(pool) if len(pool) > 0 else self.samples[index % len(self.samples)]
            else:
                sample_meta = self.samples[index % len(self.samples)]
        else:
            sample_meta = self.samples[index % len(self.samples)]

        # RGB
        img = Image.open(sample_meta['rgb']).convert('RGB')
        W0, H0 = img.size

        # Camera intrinsics + depth scaling + distortion
        intr, depth_unit_scale, dist = self._load_bop_camera(sample_meta['scene_camera'], sample_meta['im_id'])

        # Depth in meters
        depth_np = None
        if self.include_depth and sample_meta['depth'] is not None and os.path.exists(sample_meta['depth']):
            depth_raw = np.array(Image.open(sample_meta['depth']), dtype=np.float32)
            depth_np = depth_raw * float(depth_unit_scale)

        # Poses (object->camera, list -> dict keyed by inst_id)
        poses = self._load_bop_gt(sample_meta['scene_gt'], sample_meta['im_id'])  # [{'obj_id','R','t','inst_id'}...]
       
        # Mask
        mask_np = self._load_bop_mask_instance(sample_meta) if self.include_mask else None

        # Apply augmentations (letterbox, rectification if dist is present)
        img_t, depth_t, mask_t, intr_resized = self._apply_augs(
            img, depth_np, mask_np, intr, orig_size=(W0, H0), distortion=dist
        )
        depth_t = clamp_zero(depth_t)
        
        mask_np_resized = mask_t.numpy() if mask_t is not None else None

        # Targets
        boxes_xyxy, boxes_cxcywh = self._build_targets(mask_np_resized)

        # Filter poses to visible instances
        poses = self._filter_visible_instances(mask_np_resized, poses)
        visible_inst_ids = list(poses.keys())

        # Background overlay (optional, keep using existing helper; operates on normalized tensors)
        img_t, depth_t = self._apply_background_overlay(
            img_t, depth_t, mask_np_resized, source_type=sample_meta['ds_name']
        )

        # Load models keyed by inst_id, raw points in meters (scale 0.001)
        models = {}
        if self.include_models and len(visible_inst_ids) > 0:
            for inst_id in visible_inst_ids:
                p = poses.get(inst_id, None)
                if p is None:
                    continue
                obj_id = p['obj_id']
                pts = self._load_bop_model_pc(sample_meta['ds_name'], obj_id)  # model units (mm for BOP)
                if pts is None or pts.shape[0] == 0:
                    continue
                # Convert mm -> meters using scene_scale (0.001)
                s_vec = p.get('scene_scale', np.array([0.001, 0.001, 0.001], dtype=np.float32))
                if not isinstance(s_vec, np.ndarray):
                    s_vec = np.array(s_vec, dtype=np.float32)
                pts_m = pts * s_vec.reshape(1, 3).astype(np.float32)
                models[inst_id] = torch.from_numpy(pts_m.astype(np.float32))

        # Build baked props (normalized camera space)
        props = self._build_out_models_pose_baked(poses, models)

        out = {
            'dataset': self.name,
            'image_id': f"{sample_meta['ds_name']}/{sample_meta['source_type']}/{sample_meta['scene_id']}/{sample_meta['im_id']:06d}",
            'image_type': sample_meta['source_type'],
            'img': img_t,
            'depth': depth_t,
            'mask': mask_t,
            'boxes_xyxy': torch.from_numpy(boxes_xyxy),
            'boxes_cxcywh': torch.from_numpy(boxes_cxcywh),
            'intr': torch.tensor(intr_resized, dtype=torch.float32) if intr_resized is not None else None,
            'poses': poses,          # dict keyed by inst_id
            'models': models,        # dict keyed by inst_id (raw meters)
            'gt_grasps': {},         # BOP has no grasps
            'bad_grasps': {},         # BOP has no grasps
            'props': props,          # baked normalized camera-space props
            'H': self.cfg.out_hw[0],
            'W': self.cfg.out_hw[1],
        }
        return out


    @staticmethod
    def inventory(bop_root: str) -> Dict[str, Any]:
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
                cnt = 0
                cnt_depth = 0
                cnt_mask = 0
                scenes = [os.path.join(st_dir, f) for f in os.listdir(st_dir) if os.path.isdir(os.path.join(st_dir, f))]
                for sp in scenes:
                    rgb_dir = os.path.join(sp, 'rgb')
                    rgb_mask = os.path.join(sp, 'mask')
                    rgb_depth = os.path.join(sp, 'depth')
                    if not os.path.exists(rgb_dir):
                        rgb_dir = os.path.join(sp, 'gray')
                        if not os.path.exists(rgb_dir):
                            continue
                    if not os.path.exists(rgb_mask):
                        rgb_mask = os.path.join(sp, 'mask_visib')
                        if not os.path.exists(rgb_mask):
                            continue
                    if not os.path.exists(rgb_depth):
                        continue

                    if os.path.isdir(rgb_dir):
                        cnt += len([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
                    if os.path.isdir(rgb_mask):
                        cnt_mask += len([f for f in os.listdir(rgb_mask) if f.endswith('.png')])
                    if os.path.isdir(rgb_depth):
                        cnt_depth += len([f for f in os.listdir(rgb_depth) if f.endswith('.png')])

                if cnt > 0 and cnt_depth >= cnt and cnt_mask >= cnt:
                    inv[ds][st] = cnt
                    total += cnt
                elif cnt > 0:
                    print('ds {} broken for {} not matching rgb, d, mask for: {}, {}, {}'.format(ds, st, cnt, cnt_depth, cnt_mask))
        return {'total': total, 'per_dataset': inv}

# =========================

# Mixer Dataset

# =========================

class MixedMultiDataset(data.Dataset):
    def __init__(self,
                 datasets: Dict[str, data.Dataset],
                 dataset_probs: Dict[str, float],
                 num_samples_per_epoch: int = 10000,
                 failed_log_path: str = './failed_samples.json',
                 max_retry: int = 10,
                 mode='train',
                 cfg: Optional[DatasetConfig] = None):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.datasets = datasets
        self.cfg = cfg or DatasetConfig()

        dp = dict(dataset_probs)
        if 'graspnet' in dp and self.mode == 'train':
            dp['graspnet'] = 0.0
        s = sum(dp.values())
        if s <= 0:
            nonempty = [k for k,v in datasets.items() if len(v) > 0]
            dp = {k: 1.0/len(nonempty) for k in nonempty} if nonempty else {}
        else:
            for k in dp:
                dp[k] = dp[k] / s
        self.dataset_probs = dp
        self.dataset_names = list(self.dataset_probs.keys())
        self.num_samples_per_epoch = num_samples_per_epoch
        self.max_retry = max_retry
        self.failed_log_path = failed_log_path
        self.failed = read_json(self.failed_log_path, default={'failed': []})
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
        for _ in range(self.max_retry):
            try:
                idx = random.randint(0, max(0, len(ds)-1))
                return ds[idx]
            except Exception as e:
                #raise e
                self.failed['failed'].append({'dataset': ds_name, 'time': time.time(), 'exc': str(e)})
                continue
        for name in self.dataset_names:
            if name == ds_name:
                continue
            for _ in range(self.max_retry):
                try:
                    idx = random.randint(0, max(0, len(self.datasets[name])-1))
                    return self.datasets[name][idx]
                except Exception as e:
                    #raise e
                    self.failed['failed'].append({'dataset': name, 'time': time.time(), 'exc': 'fallback_failed'})
                    continue
        raise RuntimeError("Mixer: All retries failed")

    def __getitem__(self, index: int):
        ds_name = self._sample_dataset_name()
        sample = self._safe_get(ds_name)
        return sample

    def save_failed(self):
        write_json(self.failed_log_path, self.failed)


# =========================

# __main__ demos

# =========================

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
    idxes = list(range(len(dataset)))
    random.shuffle(idxes)
    for j, i in enumerate(idxes):
        if j == n_show:
            break
        try:
            dataset.visualize(i)
        except Exception as e:
            raise e
            print(f"Visualization failed for idx {i}: {e}")


def demo_visualize3d(dataset, n_show=2):
    idxes = list(range(len(dataset)))
    random.shuffle(idxes)
    for j, i in enumerate(idxes):
        if j == n_show:
            break
        try:
            dataset.visualize3d(i)
        except Exception as e:
            raise e
            print(f"Visualization failed for idx {i}: {e}")

if __name__ == "__main__":
    set_seed(42)

    # Configure dataset-wide parameters once, pass into all datasets and mixer
    cfg = DatasetConfig(
        rgb_mean=[0.485, 0.456, 0.406],
        rgb_std=[0.229, 0.224, 0.225],
        out_hw=(490, 644),     # (H, W)
        pc_num_points=768
    )

    GRASPIT_ROOT = os.environ.get('GRASPIT_ROOT', '/mnt/kikerp/OptiSim')
    GRASPIT_ASSETS = os.environ.get('GRASPIT_ASSETS', os.path.join(GRASPIT_ROOT, 'assets'))
    GRASPNET_ROOT = os.environ.get('GRASPNET_ROOT', '/mnt/wimi/publicdata/GraspNet-1Billion')
    BOP_ROOT = os.environ.get('BOP_ROOT', 'data/bop')
    GRASPNET_MODELS_ROOT = os.environ.get('GRASPNET_MODELS_ROOT', os.path.join(GRASPNET_ROOT, 'models'))
    print('GRASPIT_ROOT', GRASPIT_ROOT)
    print('GRASPIT_ASSETS', GRASPIT_ASSETS)
    print('GRASPNET_ROOT', GRASPNET_ROOT)
    print('BOP_ROOT', BOP_ROOT)
    print('GRASPNET_MODELS_ROOT', GRASPNET_MODELS_ROOT)

    bg_dirs = []
    if os.path.exists(GRASPIT_ROOT):
        bg_dirs = [os.path.join(GRASPIT_ROOT, 'backgrounds')]
        print_inventory_graspit(GRASPIT_ROOT, GRASPIT_ASSETS)
    if os.path.exists(GRASPNET_ROOT):
        print_inventory_graspnet(GRASPNET_ROOT)
    if os.path.exists(BOP_ROOT):
        print_inventory_bop(BOP_ROOT)

    # BOP (train/val) with optional rectification probability
    if os.path.exists(BOP_ROOT):
        ds_bop_train = BOPDataset(BOP_ROOT, mode='train', transforms=True, include_mask=True,
                                  use_source_probs=True, include_depth=True, include_models=True,
                                  rectify_prob=0.0, min_visible_pixels=50, backgrounds_dirs=bg_dirs,
                                  bg_replace_prob={'default': 1.0, 'ycbv': 0.0}, cfg=cfg)
        ds_bop_val = BOPDataset(BOP_ROOT, mode='val', transforms=False, include_mask=True,
                                use_source_probs=True, include_depth=True, include_models=True,
                                rectify_prob=0.0, min_visible_pixels=50, backgrounds_dirs=bg_dirs,
                                bg_replace_prob={'default': 0.0, 'ycbv': 0.0}, cfg=cfg)
        print(f"BOP train len={len(ds_bop_train)}; val len={len(ds_bop_val)}")
        #if len(ds_bop_train) > 0:
        #    demo_visualize3d(ds_bop_train, n_show=100)
        #    demo_visualize(ds_bop_train, n_show=100)

    # GraspIt
    if os.path.exists(GRASPIT_ROOT):
        ds_graspit_train = GraspItDataset(GRASPIT_ROOT, assets_root=GRASPIT_ASSETS,
                                        mode='train', transforms=True,
                                        rectify_prob=0.0, min_visible_pixels=100,
                                        ignore_table=True, backgrounds_dirs=bg_dirs,
                                        bg_replace_prob=1.0, cfg=cfg)
        ds_graspit_val = GraspItDataset(GRASPIT_ROOT, assets_root=GRASPIT_ASSETS,
                                        mode='val', transforms=False,
                                        rectify_prob=0.0, min_visible_pixels=100,
                                        ignore_table=True, backgrounds_dirs=bg_dirs,
                                        bg_replace_prob=0.0, cfg=cfg)
        print(f"GraspIt train len={len(ds_graspit_train)}; val len={len(ds_graspit_val)}")

        #ds_graspit_train.precompute_grasps(max_workers=32, overwrite=True)
        #ds_graspit_val.precompute_grasps(max_workers=16, overwrite=True)


        #if len(ds_graspit_train) > 0:
        #    demo_visualize3d(ds_graspit_train, n_show=100)
        #    demo_visualize(ds_graspit_train, n_show=100)

    # GraspNet (only val)
    if os.path.exists(GRASPNET_ROOT):
        ds_graspnet_val = GraspNetDataset(GRASPNET_ROOT, mode='val', transforms=False,
                                          models_root=GRASPNET_MODELS_ROOT, rectify_prob=0.0,
                                          min_visible_pixels=50, cfg=cfg)
                                          

        print(f"GraspNet val len={len(ds_graspnet_val)} (train excluded)")
        if hasattr(ds_graspnet_val, 'precompute_visible_grasps'):
            ds_graspnet_val.precompute_visible_grasps(max_workers=16)

        if len(ds_graspnet_val) > 0:
            demo_visualize3d(ds_graspnet_val, n_show=100)
            demo_visualize(ds_graspnet_val, n_show=100)


    # Mixer (train)
    datasets_train = {}
    probs_train = {}
    if os.path.exists(GRASPIT_ROOT):
        datasets_train['graspit'] = ds_graspit_train
        probs_train['graspit'] = 0.5
    if os.path.exists(BOP_ROOT):
        datasets_train['bop'] = ds_bop_train
        probs_train['bop'] = 0.5
    if os.path.exists(GRASPNET_ROOT):
        datasets_train['graspnet'] = GraspNetDataset(GRASPNET_ROOT, mode='train',
                                                     models_root=GRASPNET_MODELS_ROOT, cfg=cfg)
        probs_train['graspnet'] = 0.0

    if len(datasets_train) > 0:
        mixer_train = MixedMultiDataset(datasets=datasets_train, dataset_probs=probs_train, num_samples_per_epoch=1000, mode='train', cfg=cfg)
        print(f"Mixer train len={len(mixer_train)}")
        for i in range(5):
            try:
                _ = mixer_train[i]
            except Exception as e:
                print(f"Mixer sample failed: {e}")
        mixer_train.save_failed()

    # Mixer (val)
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
        mixer_val = MixedMultiDataset(datasets=datasets_val, dataset_probs=probs_val, num_samples_per_epoch=200, mode='val', cfg=cfg)
        print(f"Mixer val len={len(mixer_val)}")
        for i in range(5):
            try:
                _ = mixer_val[i]
            except Exception as e:
                print(f"Mixer val sample failed: {e}")