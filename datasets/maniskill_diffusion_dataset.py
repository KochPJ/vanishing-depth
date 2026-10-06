# datasets/maniskill_diffusion_dataset.py

import os
import math
import random
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import time
import h5py
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
import glob
import argparse
from scipy.spatial.transform import Rotation as SciRot
from model.pose_utils import tcp_pose_to_relative



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_STATS_SCHEMA_VERSION = 1



def _stats_cache_path(task_specs: Dict[str, Dict], action_mode: str) -> str:
    """
    Derive a JSON cache path that is:
      • placed next to the first (alphabetically sorted) h5 file
      • unique per combination of h5 files + action_mode
      • single-task: <stem>_action_stats_<mode>_v<V>.json
      • multi-task : <stem>_action_stats_<mode>_<hash8>_v<V>.json
    """
    paths = sorted(spec["path"] for spec in task_specs.values())
    first_stem, _ = os.path.splitext(paths[0])
    if len(paths) == 1:
        return (
            f"{first_stem}_action_stats_{action_mode}"
            f"_v{_STATS_SCHEMA_VERSION}.json"
        )
    h = hashlib.md5("|".join(paths).encode()).hexdigest()[:8]
    return (
        f"{first_stem}_action_stats_{action_mode}"
        f"_{h}_v{_STATS_SCHEMA_VERSION}.json"
    )


def _stats_cache_is_fresh(cache_path: str, task_specs: Dict[str, Dict]) -> bool:
    """
    Cache is fresh when:
      1. The JSON file exists.
      2. Its mtime is >= the mtime of every h5 file it was built from.
    If any h5 file was modified after the cache was written, we recompute.
    """
    if not os.path.exists(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    for spec in task_specs.values():
        if os.path.getmtime(spec["path"]) > cache_mtime:
            return False
    return True


def _load_stats_cache(
    cache_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Return (a_min, a_max, n_samples) from a valid cache, or None on any error.
    """
    try:
        with open(cache_path) as f:
            d = json.load(f)
        if d.get("schema_version") != _STATS_SCHEMA_VERSION:
            return None
        return (
            np.array(d["min"],  dtype=np.float32),
            np.array(d["max"],  dtype=np.float32),
            int(d["n_samples"]),
        )
    except Exception:
        return None


def _save_stats_cache(
    cache_path: str,
    a_min: np.ndarray,
    a_max: np.ndarray,
    n_samples: int,
    action_mode: str,
) -> None:
    """
    Atomic write (tmp → rename) so a crash during write never leaves a corrupt file.
    """
    payload = {
        "schema_version": _STATS_SCHEMA_VERSION,
        "action_mode":    action_mode,
        "n_samples":      int(n_samples),
        "min":            a_min.tolist(),
        "max":            a_max.tolist(),
    }
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, cache_path)


# ─────────────────────────────────────────────────────────────────────────────



class ManiSkillDiffusionDataset(Dataset):
    """
    Multi-task RGBD dataset for ManiSkill diffusion policy training.

    Expected HDF5 format per demo (as produced by your new converter):

      demo_x/
        rgb_base:   (T, H, W, 3) uint8
        depth_base: (T, H, W, 1) float32 (meters)
        rgb_hand:   (T, H, W, 3) [optional]
        depth_hand: (T, H, W, 1) [optional]
        actions:    (T, A) float32  # absolute PD targets
        qpos:       (T, Dq) float32 # absolute joint positions (Dq >= A)
        attrs["length"] = T

    Root attributes (computed in converter):

      action_mean:  (A,)  mean over a_rel(t) = actions_abs(t) - qpos_abs(t)
      action_std:   (A,)  std  over a_rel(t)
      action_count: scalar, total number of samples used.

    Dataset output per __getitem__:

      {
        "task":         task_name,
        "task_id":      tid,
        "rgb":          [obs_horizon * n_views, 3, H', W'],
        "depth":        [obs_horizon * n_views, 1, H', W'],
        "robot_state":  [obs_horizon * n_views, Dq],   # qpos_abs / (2π)
        "actions":      [horizon, A],                  # relative: actions_abs[t] - qpos_abs[t0]
        "demo_key":     demo_key,
        "t0":           t0,
      }

    We split demos 90%/10% per task into train/test by demo index.
    """

    def __init__(
        self,
        task_specs: Dict[str, Dict[str, any]],
        horizon: int = 16,
        obs_horizon: int = 1,
        split: str = "train",
        rgb_size: Tuple[int, int] = (224, 224),
        seed: int = 42,
        action_mode: str = 'abs_joints',
        add_goal: bool = False,
        rgb_only: bool = False,
        preload: bool = False,
        preload_fields: Optional[list] = None,
    ):
        assert split in ("train", "test")
        assert obs_horizon >= 1
        self.task_specs = task_specs
        self.horizon = int(horizon)
        self.obs_horizon = int(obs_horizon)
        self.rgb_size = rgb_size
        self.split = split

        self.files: Dict[str, h5py.File] = {}
        self.indices: List[Tuple[str, str, int]] = []   # (task_name, demo_key, t0)
        self.task_ids: Dict[str, int] = {}
        self.demo_sample_counts: Dict[Tuple[str, str], int] = {}
        self.action_mode = action_mode
        self.add_goal = add_goal
        self.rgb_only = rgb_only
        self.preload = preload
        self.preload_fields = preload_fields   # e.g. ["rgb_base","depth_base","actions","qpos"]
         # path → demo_key → field → np.ndarray
        self._ram: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        # transforms

        self.rgb_aug = T.Compose([
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            )
        ])

        self.rgb_tf = T.Compose([
            T.Resize(rgb_size, interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        random.seed(seed)
        np.random.seed(seed)

        self._build_index()

        if preload:
            self._preload_to_ram()

    # ── RAM pre-loader ────────────────────────────────────────────────
    def _estimate_disk_gb(self) -> float:
        return sum(
            os.path.getsize(s["path"])
            for s in self.task_specs.values()
            if os.path.isfile(s["path"])
        ) / 1e9

    def _preload_to_ram(self) -> None:
        est_gb = self._estimate_disk_gb()
        print(f"[Preload] Disk footprint ≈ {est_gb:.1f} GB  →  loading into RAM …")
        t0          = time.time()
        total_bytes = 0

        for task_name, spec in self.task_specs.items():
            path = spec["path"]
            self._ram[path] = {}
            with h5py.File(path, "r") as f:
                demo_keys = [k for k in f.keys() if k.startswith("demo_")]
                for dk in demo_keys:
                    grp        = f[dk]
                    fields     = self.preload_fields or list(grp.keys())
                    demo: Dict[str, np.ndarray] = {}
                    for key in fields:

                        if self.rgb_only and 'depth' in key:
                            continue

                        if key in grp:
                            arr = grp[key][:]          # ← actual RAM copy
                            demo[key] = arr
                            total_bytes += arr.nbytes
                    self._ram[path][dk] = demo

        print(
            f"[Preload] {total_bytes / 1e9:.2f} GB loaded into RAM  "
            f"({time.time() - t0:.1f}s)"
        )

    # ── Unified data accessor (RAM dict or h5py group) ────────────────
    def _get_demo(self, path: str, demo_key: str):
        """
        Returns either a plain dict[str→np.ndarray] (RAM path)
        or an h5py Group (disk path).  Both support grp[key][:] identically.
        """
        if self.preload and path in self._ram:
            return self._ram[path][demo_key]
        return self._open_file(path)[demo_key]

    # ---------- HDF5 helpers ----------
    def _open_file(self, path: str) -> h5py.File:
        # 1. Detect which PyTorch worker is executing this line
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        
        # 2. Initialize a worker-specific dictionary if it doesn't exist yet
        if not hasattr(self, '_worker_files'):
            self._worker_files = {}
            
        if worker_id not in self._worker_files:
            self._worker_files[worker_id] = {}
            
        # 3. Open the file strictly inside this worker's isolated context
        worker_dict = self._worker_files[worker_id]
        
        if path not in worker_dict:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            
            # SWMR=True enables Single-Writer Multiple-Reader mode, 
            # which drastically optimizes parallel reads in h5py
            worker_dict[path] = h5py.File(path, "r", swmr=True)
            
        return worker_dict[path]


    #def _open_file(self, path: str) -> h5py.File:
    #    if path not in self.files:
    #        if not os.path.isfile(path):
    #            raise FileNotFoundError(path)
    #        self.files[path] = h5py.File(path, "r")
    #    return self.files[path]

    def get_action_stats(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compute per-dimension p1/p99 bounds for action normalisation.

        Loading strategy (fast path first):
          1. JSON cache next to the h5 file(s) → instant load if all h5 files
             are older than the cache.
          2. Full recompute from h5 → write cache → return.

        For multi-task the cache is keyed on ALL h5 paths (via MD5 prefix) so
        adding a new task automatically invalidates the old cache.
        """
        cache_path = _stats_cache_path(self.task_specs, self.action_mode)

        # ── Fast path: load from JSON cache ───────────────────────────────
        if _stats_cache_is_fresh(cache_path, self.task_specs):
            result = _load_stats_cache(cache_path)
            if result is not None:
                a_min, a_max, n_samples = result
                # Safety gap (same as compute path)
                eps       = 1e-6
                too_close = (a_max - a_min) < eps
                a_min[too_close] -= eps
                a_max[too_close] += eps
                info_dict = {
                    "action_mode": self.action_mode,
                    "shape":       n_samples,
                    "action_min":  a_min,
                    "action_max":  a_max,
                    "range":       a_max - a_min,
                    "from_cache":  True,
                }
                print(
                    f"[action_stats] Loaded cache  {cache_path!r}"
                    f"  (n={n_samples}, mode={self.action_mode})"
                )
                return (
                    torch.from_numpy(a_min),
                    torch.from_numpy(a_max),
                    info_dict,
                )

        # ── Slow path: compute from h5 ────────────────────────────────────
        print(
            f"[action_stats] Cache miss — computing from h5"
            f"  mode={self.action_mode!r}"
            f"  tasks={list(self.task_specs.keys())}"
        )
        all_actions: List[np.ndarray] = []

        for task_name, spec in self.task_specs.items():
            path = spec["path"]
            f    = self._open_file(path)

            for demo_key in sorted(k for k in f.keys() if k.startswith("demo_")):
                grp     = f[demo_key]
                qpos_np = grp["qpos"][:]        # (T, Dq)
                T_len   = qpos_np.shape[0]

                if '_joints' in self.action_mode:
                    actions_np = grp["actions"][:]   # (T, A)
                    A = actions_np.shape[1]

                    for t0 in range(T_len):
                        idx_act       = np.clip(
                            np.arange(t0, t0 + self.horizon), 0, T_len - 1
                        )
                        future_actions = actions_np[idx_act]   # (H, A)

                        if self.action_mode == 'abs_joints':
                            act_seq = future_actions
                        else:   # relative_joints
                            anchor        = qpos_np[t0, :A]
                            act_seq       = future_actions - anchor[None, :]
                            act_seq[..., 7] = future_actions[..., 7]   # gripper abs

                        all_actions.append(act_seq)

                else:   # abs_pose / relative_pose
                    ee_pos  = grp["ee_pos"][:]          # (T, 3)
                    ee_quat = grp["ee_quat_xyzw"][:]    # (T, 4)

                    for t0 in range(T_len):
                        idx_act = np.clip(
                            np.arange(t0 + 1, t0 + 1 + self.horizon), 0, T_len - 1
                        )
                        gripper   = np.expand_dims(qpos_np[idx_act, -1], 1)   # (H,1)
                        pose_list = []

                        if self.action_mode == 'abs_pose':
                            for t in idx_act:
                                rv = SciRot.from_quat(ee_quat[t]).as_rotvec()
                                pose_list.append(
                                    np.concatenate([ee_pos[t], rv], axis=0)
                                )
                        else:   # relative_pose
                            for t in idx_act:
                                pose_list.append(
                                    tcp_pose_to_relative(
                                        ee_pos[t], ee_quat[t],
                                        ee_pos[t0], ee_quat[t0],
                                    )
                                )

                        pose_array = np.stack(pose_list, axis=0).astype(np.float32)
                        all_actions.append(
                            np.concatenate([pose_array, gripper], axis=-1)
                        )

        if not all_actions:
            raise RuntimeError(
                "[ManiSkillDiffusionDataset] No action samples found across tasks."
            )

        stacked    = np.concatenate(all_actions, axis=0)   # (N*H, A)
        n_samples  = int(stacked.shape[0])
        a_min      = np.percentile(stacked, 1,  axis=0).astype(np.float32)
        a_max      = np.percentile(stacked, 99, axis=0).astype(np.float32)

        eps       = 1e-6
        too_close = (a_max - a_min) < eps
        a_min[too_close] -= eps
        a_max[too_close] += eps

        # ── Persist to JSON ───────────────────────────────────────────────
        try:
            _save_stats_cache(cache_path, a_min, a_max, n_samples, self.action_mode)
            print(f"[action_stats] Cached → {cache_path!r}")
        except OSError as exc:
            print(f"[action_stats] Warning: could not write cache ({exc})")

        info_dict = {
            "action_mode": self.action_mode,
            "shape":       n_samples,
            "action_min":  a_min,
            "action_max":  a_max,
            "range":       a_max - a_min,
            "from_cache":  False,
        }
        return torch.from_numpy(a_min), torch.from_numpy(a_max), info_dict

    def get_action_stats_old(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-dimension min/max for action normalization to [-1, 1].
        
        Uses p1/p99 percentiles for robustness against outliers.
        Computes actions exactly as _load_seq does for each action_mode.
        
        Returns:
            action_min: (A,) tensor — lower bound per dimension
            action_max: (A,) tensor — upper bound per dimension
        """
        all_actions = []

        for task_name, spec in self.task_specs.items():
            path = spec["path"]
            f = self._open_file(path)

            for demo_key in sorted(k for k in f.keys() if k.startswith("demo_")):
                grp = f[demo_key]
                qpos_np = grp["qpos"][:]  # (T, Dq)
                T_len = qpos_np.shape[0]

                if '_joints' in self.action_mode:
                    actions_np = grp["actions"][:]  # (T, A)
                    A = actions_np.shape[1]

                    for t0 in range(T_len):
                        idx_act = np.arange(t0, t0 + self.horizon)
                        idx_act = np.clip(idx_act, 0, T_len - 1)
                        future_actions = actions_np[idx_act]  # (H, A)

                        if self.action_mode == 'abs_joints':
                            act_seq = future_actions
                        else:  # 'relative_joints'
                            anchor_qpos = qpos_np[t0, :A]
                            act_seq = future_actions - anchor_qpos[None, :]
                            act_seq[..., 7] = future_actions[..., 7]  # gripper absolute

                        all_actions.append(act_seq)  # (H, A)

                else:  # 'abs_pose' or 'relative_pose'
                    ee_pos = grp["ee_pos"][:]          # (T, 3)
                    ee_quat = grp["ee_quat_xyzw"][:]   # (T, 4)

                    for t0 in range(T_len):
                        idx_act = np.arange(t0 + 1, t0 + 1 + self.horizon)
                        idx_act = np.clip(idx_act, 0, T_len - 1)

                        gripper = qpos_np[idx_act, -1]  # (H,)
                        gripper = np.expand_dims(gripper, 1)  # (H, 1)

                        pose_list = []
                        if self.action_mode == 'abs_pose':
                            for t in idx_act:
                                pos_t = ee_pos[t]
                                quat_t = ee_quat[t]
                                R = SciRot.from_quat(quat_t)
                                rv = R.as_rotvec()
                                pose_list.append(
                                    np.concatenate([pos_t, rv], axis=0)
                                )  # (6,)
                        else:  # 'relative_pose'
                            pos_ref = ee_pos[t0]
                            quat_ref = ee_quat[t0]
                            for t in idx_act:
                                rel6 = tcp_pose_to_relative(
                                    ee_pos[t], ee_quat[t], pos_ref, quat_ref
                                )
                                pose_list.append(rel6)  # (6,)

                        pose_array = np.stack(pose_list, axis=0).astype(np.float32)  # (H, 6)
                        act_seq = np.concatenate([pose_array, gripper], axis=-1)       # (H, 7)
                        all_actions.append(act_seq)

        if not all_actions:
            raise RuntimeError(
                "[ManiSkillDiffusionDataset] No action samples found across tasks."
            )

        # Flatten: [N_total, H, A] -> [N_total * H, A]
        all_actions = np.concatenate(all_actions, axis=0)  # (N*H, A)

        # Robust percentile bounds (p1/p99)
        action_min = np.percentile(all_actions, 1, axis=0).astype(np.float32)
        action_max = np.percentile(all_actions, 99, axis=0).astype(np.float32)

        # Safety: ensure min < max everywhere
        eps = 1e-6
        too_close = (action_max - action_min) < eps
        action_min[too_close] = action_min[too_close] - eps
        action_max[too_close] = action_max[too_close] + eps


        info_dict = {
            'action_mode': self.action_mode,
            'shape': all_actions.shape[0],
            'action_min': action_min,
            'action_max': action_max,
            'range': action_max - action_min,
        }

        min_t = torch.from_numpy(action_min)
        max_t = torch.from_numpy(action_max)
        return min_t, max_t, info_dict
        

    def summarize(self):
        per_task_demo_counts = {}
        per_task_sample_counts = {}
        per_task_demo_sample_list = {}

        for (task_name, demo_key), cnt in self.demo_sample_counts.items():
            per_task_demo_counts.setdefault(task_name, 0)
            per_task_sample_counts.setdefault(task_name, 0)
            per_task_demo_sample_list.setdefault(task_name, [])
            per_task_demo_counts[task_name] += 1
            per_task_sample_counts[task_name] += cnt
            per_task_demo_sample_list[task_name].append(cnt)

        total_samples = len(self.indices)

        print(f"[ManiSkillDiffusionDataset] split='{self.split}' | total samples = {total_samples}")
        for task_name in sorted(per_task_demo_counts.keys()):
            demos = per_task_demo_counts[task_name]
            samples = per_task_sample_counts[task_name]
            arr = np.array(per_task_demo_sample_list[task_name], dtype=np.int64)
            min_v = int(arr.min()) if arr.size > 0 else 0
            max_v = int(arr.max()) if arr.size > 0 else 0
            mean_v = float(arr.mean()) if arr.size > 0 else 0.0
            std_v = float(arr.std()) if arr.size > 0 else 0.0
            print(
                f"  Task {task_name:18s} | demos={demos:4d} "
                f"| samples={samples:7d} "
                f"| per-demo: min={min_v:4d}, max={max_v:4d}, "
                f"mean={mean_v:6.1f}, std={std_v:6.1f}"
            )

    # ---------- Index building ----------

    def _build_index(self):
        """
        For each task:
          - open file
          - list demos
          - split demos 90/10 into train/test
          - build (task, demo_key, t0) for t0 in [0 .. T_len-1]
        Also fills self.demo_sample_counts[(task_name, demo_key)] = #samples.
        """
        tid = 0
        for task_name, spec in self.task_specs.items():
            h5_path = spec["path"]
            ratio = float(spec.get("ratio", 1.0))
            f = self._open_file(h5_path)

            demo_keys = sorted([k for k in f.keys() if k.startswith("demo_")])
            n_demo = len(demo_keys)
            if n_demo == 0:
                continue

            self.task_ids[task_name] = tid
            tid += 1

            n_train = max(1, int(math.floor(0.9 * n_demo)))
            train_demos = demo_keys[:n_train]
            test_demos = demo_keys[n_train:]

            use_demos = train_demos if self.split == "train" else test_demos

            for demo_key in use_demos:
                grp = f[demo_key]
                T_len = int(grp["actions"].shape[0])

                if T_len <= 0:
                    self.demo_sample_counts[(task_name, demo_key)] = 0
                    continue

                t0_min = 0
                t0_max = T_len - 1

                num_t0 = max(1, t0_max - t0_min + 1)
                rep = max(1, int(round(ratio)))
                num_samples_demo = num_t0 * rep
                self.demo_sample_counts[(task_name, demo_key)] = num_samples_demo

                for t0 in range(t0_min, t0_max + 1):
                    for _ in range(rep):
                        self.indices.append((task_name, demo_key, t0))

    # ---------- Core loading ----------

    def _load_seq(self, task_name: str, demo_key: str, t0: int):
        """
        Load obs_horizon RGBD frames ending at t0, plus:

          - robot_state_obs: [obs_horizon * n_views, Dq]
                qpos_abs[idx_obs[i]] / (2π), repeated per view

          - actions_rel: [horizon, A]
                relative actions w.r.t. qpos_abs at t0:
                actions_rel[h] = actions_abs[idx_act[h]] - qpos_abs[t0]
        """
        spec = self.task_specs[task_name]
        #spec = self.task_specs[task_name]
        grp  = self._get_demo(spec["path"], demo_key)

        #f = self._open_file(spec["path"])
        #grp = f[demo_key]

        rgb_base = grp["rgb_base"][:]      # (T, H, W, 3)
        depth_base = None
        if not self.rgb_only:
            depth_base = grp["depth_base"][:]  # (T, H, W, 1)
        has_hand = ("rgb_hand" in grp) and ("depth_hand" in grp)
        
        if has_hand:
            rgb_hand = grp["rgb_hand"][:]
            if not self.rgb_only:
                depth_hand = grp["depth_hand"][:]
        else:
            rgb_hand = None
            depth_hand = None

        rgb_base = np.asarray(rgb_base, dtype=np.uint8)
        if not self.rgb_only:
            depth_base = np.asarray(depth_base, dtype=np.float32)
        qpos_np = grp["qpos"][:]        # (T, Dq)
            


        T_len = qpos_np.shape[0]
        assert rgb_base.shape[0] == T_len
        if not self.rgb_only:
            assert depth_base.shape[0] == T_len
         
        if '_joints' in self.action_mode:
            actions_np = grp["actions"][:]  # (T, A)
            A = actions_np.shape[1]
            Dq = qpos_np.shape[1]
            assert actions_np.shape[0] == T_len
        else:
            ee_pos = grp["ee_pos"][:]           # (T,3)
            ee_quat = grp["ee_quat_xyzw"][:]    # (T,4)            
            T_len_pose = ee_pos.shape[0]
            assert T_len_pose == T_len
    
        # ---------- 1) Observation frames (idx_obs) ----------
        idx_obs = np.arange(t0 - (self.obs_horizon - 1), t0 + 1)
        idx_obs = np.clip(idx_obs, 0, T_len - 1)

        rgb_frames: List[torch.Tensor] = []
        depth_frames: List[torch.Tensor] = []
        robot_frames: List[torch.Tensor] = []

        for idx in idx_obs:
            # Base camera
            rgb_i = rgb_base[idx]      # (H, W, 3)
            rgb_pil = Image.fromarray(rgb_i.astype(np.uint8))            
            if self.split == 'train':
                rgb_pil = self.rgb_aug(rgb_pil)
            rgb_t = self.rgb_tf(rgb_pil)  # [3,H',W']            
            rgb_frames.append(rgb_t)


            if not self.rgb_only:
                depth_i = depth_base[idx]  # (H, W, 1)
                depth_pil = Image.fromarray(depth_i.squeeze(-1).astype(np.float32))
                depth_resized = depth_pil.resize(self.rgb_size, resample=Image.NEAREST)
                depth_np2 = np.array(depth_resized, dtype=np.float32)
                depth_t = torch.from_numpy(depth_np2).unsqueeze(0)  # [1,H',W']

                if self.split == 'train':
                    depth_mask = torch.rand(depth_t.shape) > float(np.random.rand())
                    depth_t[depth_mask] = 0
                depth_frames.append(depth_t)

            # Hand camera
            if has_hand:
                rgb_h = rgb_hand[idx]
                rgb_pil_h = Image.fromarray(rgb_h.astype(np.uint8))
                if self.split == 'train':
                    rgb_pil_h = self.rgb_aug(rgb_pil_h)
                rgb_t_h = self.rgb_tf(rgb_pil_h)
                rgb_frames.append(rgb_t_h)

                if not self.rgb_only:
                    depth_h = depth_hand[idx]
                    depth_pil_h = Image.fromarray(depth_h.squeeze(-1).astype(np.float32))                
                    depth_resized_h = depth_pil_h.resize(self.rgb_size, resample=Image.NEAREST)
                    depth_np2_h = np.array(depth_resized_h, dtype=np.float32)
                    depth_t_h = torch.from_numpy(depth_np2_h).unsqueeze(0)

                    if self.split == 'train':
                        depth_mask = torch.rand(depth_t_h.shape) > float(np.random.rand())
                        depth_t_h[depth_mask] = 0
                    depth_frames.append(depth_t_h)

            
            if '_joints' in self.action_mode:
                q_abs = qpos_np[idx]  # (Dq)
            else:
                pos_t = ee_pos[idx]
                quat_t = ee_quat[idx]
                q_abs = np.concatenate([pos_t, quat_t], axis=-1)
            
            if self.add_goal:
                goal = grp["ee_pos"][:][-1, :3]
                q_abs = np.concatenate([q_abs, goal], axis=-1)

            q_t = torch.from_numpy(q_abs.astype(np.float32))

            robot_frames.append(q_t)
            if has_hand:
                robot_frames.append(q_t.clone())

        rgb_seq_t = torch.stack(rgb_frames, dim=0)       # [obs_h*n_views,3,H',W']
        if not self.rgb_only:
            depth_seq_t = torch.stack(depth_frames, dim=0)   # [obs_h*n_views,1,H',W']
        else:
            depth_seq_t = None


        robot_state_t = torch.stack(robot_frames, dim=0) # [obs_h*n_views,Dq]

        
        # ---------- 2) Future actions ----------
        if self.action_mode in ('relative_joints', 'abs_joints'):
            # in action key we start with t0 which is the action taken from current qpos
            idx_act = np.arange(t0, t0 + self.horizon)
            idx_act = np.clip(idx_act, 0, T_len - 1)
            future_actions = actions_np[idx_act]        # (H,A)

            if self.action_mode == 'abs_joints':
                act_seq = future_actions                # [H,8]
            else:
                anchor_qpos = qpos_np[t0, :A]           # (A,)
                act_seq = future_actions - anchor_qpos[None, :]  # (H,A)
                # keep gripper absolute (last dim)
                act_seq[..., 7] = future_actions[..., 7]

        elif self.action_mode in ('abs_pose', 'relative_pose'):
            # in se(3) controll we use t0+1 which is the next pose of the robot 
            idx_act = np.arange(t0 + 1, t0 + 1 + self.horizon)
            idx_act = np.clip(idx_act, 0, T_len_pose - 1)

            # gripper commands from original actions at future timesteps
            gripper = qpos_np[idx_act, -1]            # (H,1)
            gripper = np.expand_dims(gripper, 1)

            pose_list = []
            if self.action_mode == 'abs_pose':
                for t in idx_act:
                    pos_t = ee_pos[t]
                    quat_t = ee_quat[t]
                    R = SciRot.from_quat(quat_t)
                    rv = R.as_rotvec()
                    pose_list.append(np.concatenate([pos_t, rv], axis=0))  # (6,)
                    
            else:  # 'relative_pose'
                pos_ref = ee_pos[t0]
                quat_ref = ee_quat[t0]
                for t in idx_act:
                    pos_t = ee_pos[t]
                    quat_t = ee_quat[t]
                    rel6 = tcp_pose_to_relative(pos_t, quat_t, pos_ref, quat_ref)
                    pose_list.append(rel6)                                    # (6,)


            pose_array = np.stack(pose_list, axis=0).astype(np.float32)       # (H,6)
            act_seq = np.concatenate([pose_array, gripper], axis=-1)          # (H,7)
        else:
            raise ValueError(f'action_mode {self.action_mode} not implemented')

        actions_t = torch.from_numpy(act_seq.astype(np.float32))  # [H, A_act]
        return rgb_seq_t, depth_seq_t, robot_state_t, actions_t


    # ---------- Dataset API ----------

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        task_name, demo_key, t0 = self.indices[idx]
        rgb_seq_t, depth_seq_t, robot_state_t, actions_t = self._load_seq(task_name, demo_key, t0)
        tid = self.task_ids[task_name]
        out_dict = {
            "task": task_name,
            "task_id": tid,
            "rgb": rgb_seq_t,            # [obs_h*n_views,3,H,W]
            "robot_state": robot_state_t,  # [obs_h*n_views,Dq] (qpos_abs / 2π)
            "actions": actions_t,        # [horizon,A] (relative: actions_abs - qpos_abs(t0))
            "demo_key": demo_key,
            "t0": t0,
        }

        if not self.rgb_only:
            out_dict["depth"] = depth_seq_t        # [obs_h*n_views,1,H,W] if not self.rgb_only

        return out_dict


# ---------- Helper to build task specs from root -----------

def _build_task_specs_from_root(train_root: str):
    """
    Scan train_root for *_train.h5 / *_train.hdf5 and build task_specs.

    Example:
      /.../PickCube_train.h5           -> "PickCube-v1"
      /.../PegInsertionSide_train.h5   -> "PegInsertionSide-v1"
    """
    patterns = [
        os.path.join(train_root, "*_train.h5"),
        os.path.join(train_root, "*_train.hdf5"),
    ]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))

    task_specs = {}
    for path in paths:
        base = os.path.basename(path)
        base_noext = base.replace(".h5", "").replace(".hdf5", "")
        prefix = base_noext.split("_train")[0]
        task_name = f"{prefix}-v1"
        task_specs[task_name] = {"path": path, "ratio": 1.0}
    return task_specs


def _unnormalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """
    Invert ImageNet normalization on a [3,H,W] tensor.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device).view(3, 1, 1)
    rgb = rgb * std + mean
    return torch.clamp(rgb, 0.0, 1.0)


def _save_sample_to_dir(sample: Dict, out_dir: str, idx: int):
    """
    Save all frames in sample["rgb"] to out_dir.

    Files:
      {idx:06d}_{task}_{demo}_t{t0}_rgb{f}.png
      {idx:06d}_{task}_{demo}_t{t0}_depth{f}.png
    """
    os.makedirs(out_dir, exist_ok=True)

    task = sample["task"]
    demo_key = sample["demo_key"]
    t0 = sample["t0"]
    rgb_seq = sample["rgb"]      # [N,3,H,W]
    depth_seq = sample["depth"]  # [N,1,H,W]
    print('depth_seq', depth_seq.shape, torch.mean(depth_seq), torch.std(depth_seq), torch.min(depth_seq), torch.max(depth_seq))

    num_frames = rgb_seq.shape[0]

    for f_idx in range(num_frames):
        # RGB
        rgb_frame = _unnormalize_rgb(rgb_seq[f_idx].cpu())
        rgb_path = os.path.join(
            out_dir,
            f"{idx:06d}_{task}_{demo_key}_t{t0}_rgb{f_idx}.png",
        )
        save_image(rgb_frame, rgb_path)

        # Depth -> grayscale
        depth_frame = depth_seq[f_idx, 0].cpu().numpy()  # [H,W]
        d = depth_frame.astype(np.float32)
        valid = np.isfinite(d) & (d > 0)
        if valid.any():
            vmax = float(np.percentile(d[valid], 95))
            if vmax <= 0:
                vmax = 1.0
            d_norm = np.clip(d, 0, vmax) / vmax
        else:
            d_norm = np.zeros_like(d, dtype=np.float32)

        d_img = (d_norm * 255.0).astype(np.uint8)
        depth_path = os.path.join(
            out_dir,
            f"{idx:06d}_{task}_{demo_key}_t{t0}_depth{f_idx}.png",
        )
        Image.fromarray(d_img).save(depth_path)


# ---------- Helper to build task specs from root -----------

def _build_task_specs_from_root(train_root: str):
    patterns = [
        os.path.join(train_root, "*_train.h5"),
        os.path.join(train_root, "*_train.hdf5"),
    ]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))

    task_specs = {}
    for path in paths:
        base = os.path.basename(path)
        base_noext = base.replace(".h5", "").replace(".hdf5", "")
        prefix = base_noext.split("_train")[0]
        task_name = f"{prefix}-v1"
        task_specs[task_name] = {"path": path, "ratio": 1.0}
    return task_specs


def _unnormalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device).view(3, 1, 1)
    rgb = rgb * std + mean
    return torch.clamp(rgb, 0.0, 1.0)


def depth_to_pil(depth: np.ndarray) -> Image.Image:
    """
    Convert a single depth map (H,W) in meters to a grayscale PIL.Image.
    """
    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.any():
        vmax = float(np.percentile(d[valid], 95))
        if vmax <= 0:
            vmax = 1.0
        d_norm = np.clip(d, 0, vmax) / vmax
    else:
        d_norm = np.zeros_like(d, dtype=np.float32)

    d_img = (d_norm * 255.0).astype(np.uint8)
    return Image.fromarray(d_img)


def make_2x2_grid(
    rgb_base: Image.Image,
    depth_base: Image.Image,
    rgb_hand: Image.Image | None,
    depth_hand: Image.Image | None,
    tile_size: Tuple[int, int],
) -> Image.Image:
    """
    Make a 2x2 grid:

      [ base RGB   | hand RGB (or black) ]
      [ base depth | hand depth (or black) ]

    All sub-images are resized to (W,H)=tile_size.
    """
    W, H = tile_size

    # Resize
    rgb_base = rgb_base.resize((W, H), resample=Image.BILINEAR).convert("RGB")
    depth_base = depth_base.resize((W, H), resample=Image.NEAREST).convert("RGB")

    if rgb_hand is not None:
        rgb_hand = rgb_hand.resize((W, H), resample=Image.BILINEAR).convert("RGB")
    else:
        rgb_hand = Image.new("RGB", (W, H), color=(0, 0, 0))

    if depth_hand is not None:
        depth_hand = depth_hand.resize((W, H), resample=Image.NEAREST).convert("RGB")
    else:
        depth_hand = Image.new("RGB", (W, H), color=(0, 0, 0))

    # Canvas
    grid = Image.new("RGB", (2 * W, 2 * H), color=(0, 0, 0))
    grid.paste(rgb_base, (0, 0))
    grid.paste(rgb_hand, (W, 0))
    grid.paste(depth_base, (0, H))
    grid.paste(depth_hand, (W, H))
    return grid
    

# ---------- Debug CLI: make GIFs of first test demo ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-root",
        type=str,
        required=True,
        help="Directory containing *_train.h5 files (ManiSkill demos).",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        required=True,
        help="Directory to save GIFs.",
    )
    parser.add_argument(
        "--rgb-size",
        type=int,
        nargs=2,
        default=(224, 224),
        help="Tile size (H W) for each sub-image in the 2x2 grid.",
    )
    args = parser.parse_args()

    os.makedirs(args.debug_dir, exist_ok=True)

    task_specs = _build_task_specs_from_root(args.train_root)
    if not task_specs:
        raise RuntimeError(f"No *_train.h5 files found under {args.train_root}")

    print("Task specs:")
    for k, v in task_specs.items():
        print(f"  {k}: {v['path']}")

    H_tile, W_tile = args.rgb_size

    for task_name, spec in task_specs.items():
        path = spec["path"]
        print(f"\n[Task] {task_name} | file={path}")
        with h5py.File(path, "r") as f:
            demo_keys = sorted([k for k in f.keys() if k.startswith("demo_")])
            n_demo = len(demo_keys)
            if n_demo == 0:
                print("  No demos; skipping.")
                continue

            n_train = max(1, int(math.floor(0.9 * n_demo)))
            test_demos = demo_keys[n_train:]
            if not test_demos:
                print("  No test demos; using last demo as fallback.")
                demo_key = demo_keys[-1]
            else:
                demo_key = test_demos[0]

            print(f"  Using demo '{demo_key}' as first test demo.")

            grp = f[demo_key]
            rgb_base = grp["rgb_base"][:]      # (T,H,W,3)
            depth_base = grp["depth_base"][:]  # (T,H,W,1)
            has_hand = ("rgb_hand" in grp) and ("depth_hand" in grp)
            if has_hand:
                rgb_hand = grp["rgb_hand"][:]      # (T,H,W,3)
                depth_hand = grp["depth_hand"][:]  # (T,H,W,1)
            else:
                rgb_hand = None
                depth_hand = None

            rgb_base = np.asarray(rgb_base, dtype=np.uint8)
            depth_base = np.asarray(depth_base, dtype=np.float32)
            if has_hand:
                rgb_hand = np.asarray(rgb_hand, dtype=np.uint8)
                depth_hand = np.asarray(depth_hand, dtype=np.float32)

            T_len = rgb_base.shape[0]
            print(f"  Demo length T={T_len}")

            frames: List[Image.Image] = []

            for t in range(T_len):
                # Base RGB / depth
                rgb_b = rgb_base[t]               # (H,W,3)
                depth_b = depth_base[t, ..., 0]   # (H,W)

                rgb_b_pil = Image.fromarray(rgb_b.astype(np.uint8))
                depth_b_pil = depth_to_pil(depth_b)

                if has_hand:
                    rgb_h = rgb_hand[t]
                    depth_h = depth_hand[t, ..., 0]
                    rgb_h_pil = Image.fromarray(rgb_h.astype(np.uint8))
                    depth_h_pil = depth_to_pil(depth_h)
                else:
                    rgb_h_pil = None
                    depth_h_pil = None

                grid = make_2x2_grid(
                    rgb_base=rgb_b_pil,
                    depth_base=depth_b_pil,
                    rgb_hand=rgb_h_pil,
                    depth_hand=depth_h_pil,
                    tile_size=(W_tile, H_tile),
                )
                frames.append(grid)

            if frames:
                gif_path = os.path.join(args.debug_dir, f"{task_name}_test_demo.gif")
                # simple fixed duration per frame
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=100,
                    loop=0,
                )
                print(f"  Saved GIF: {gif_path}")
            else:
                print("  No frames to save; skipping GIF.")


if __name__ == "__main__":
    main()

"""
Example:

python datasets/maniskill_diffusion_dataset.py \
  --train-root /mnt/wimi/publicdata/maniskill/demos/training \
  --debug-dir  /mnt/wimi/publicdata/maniskill/debug_gifs \
  --rgb-size 224 224
"""

