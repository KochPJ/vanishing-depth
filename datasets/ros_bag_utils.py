from typing import Any, List, Optional
from dataclasses import dataclass, field
import os
from icecream import ic
import numpy as np
from pathlib import Path
import yaml
from scipy.spatial.transform import Rotation as R, Slerp
from scipy import ndimage   
from PIL import Image
import torch
import random
import torchvision.transforms as T


# Function to load YAML file into a dataclass
def load_yaml_as_dataclass(yaml_file: str, dataclass_type: Any) -> Any:
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

        # Create a dataclass instance using unpacked dictionary values
        return dataclass_type(**data)


def worker_init_fn(_):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''     # no GPU in worker
    os.environ['JAX_PLATFORMS'] = 'cpu'         # jax stays on CPU if imported
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'   


@dataclass
class Config:
    """Configuration dataclass for diffusion policy training and execution."""
    
    # Diffusion Policy general parameters
    encoder_name: str
    encoder_version: str
    additional_cls_layers: tuple
    with_depth: bool

    pretrained: str
    image_resize_shape: tuple
    image_forward_shape: tuple
    image_static_crop: tuple
    action_horizon: int
    vision_feature_dim: int
    backbone_lr_mult: float
    freeze_backbone: bool
    freeze_color_encoder: bool

    whitelist_path: str
    use_whitelist: bool
    use_goal_balanced_sampling: bool
    goal_group_threshold_m: float
    max_bags_per_goal_group: int     # 0 = no cap; >0 = subsample large groups
    drop_sensor_p: float
    sensor_drop_zero_p: float


    lowdim_obs_dim: int
    num_views: int
    action_dim: int
    batch_size: int
    num_workers: int
    num_eval_workers: int
    checkpoint_path: str
    dataset_path: tuple
    test_dataset_paths: tuple
    full_test_every_epoch: int
    start_full_test_epoch: int
    ds_out_path: str
    load_pretrained_policy_path: str
    
    unpack_path: str
    load_cropped: bool
    cropped_unpack_path: tuple
    unpack_npy_paths: tuple
    
    cache_path: str
    lr: float
    max_mgpus: int
    batch_timeout_s: int
    image_fps_downsample: bool 
    image_downsample_to_fps: int 
    obs_horizon: int 
    extend_obs_horizon_agent_pos: bool
    extend_train_obs_sampling: int        
    near_threshold_pos: float
    near_threshold_deg: float  
    very_close_threshold_pos: float
    very_close_threshold_deg: float
    finish_threshold_pos: float
    finish_threshold_deg: float
    interpolate_action_speed: bool
    max_speed_threshold: float
    robot_joint_state_sampling: int
    time_dependent_sampling: bool
    time_ms_per_step: int
    stop_speed_threshold: float
    multi_view_sampling_p: float
    near_sampling_p: float
    very_close_sampling_p: float
    finish_sampling_p: float
    recovery_sampling_p: float
    far_sampling_p: float
    start_sampling_p: float
    start_max_time_s: float
    start_min_frames: int
    start_z_dominance_ratio: float
    start_min_lateral_motion_mm: float
    start_transition_window: int


    undersample_near_n_bags: int
    pred_horizon: int
    num_diffusion_iters: int
    ema_decay: float
    num_epochs: int
    save_ema: bool
    num_steps_per_epoch: int
    debug: bool
    debug_dataloader: bool
    image_random_crop_aug: bool
    image_brightness_aug: bool
    apply_image_static_crop: bool
    random_background_with_depth: bool
    background_root: str
    use_depth_completed: bool
    depth_cut_treshold: float
    depth_cut_treshold_pe: float
    
    use_depth_positional_encoding: bool
    depth_pe_channels: int          
    depth_pe_temperature: float
    depth_pe_temperature_min_band: float
    with_depth: bool
    
    # Robot specific parameters
    tcp_link_name: str
    base_link_name: str
    urdf_path: str
    num_joints: int
    joint_names_order: List[str]
    
    # Inference execution parameters
    camera_topic_name: str 
    depth_cam_topic_name: str

    joint_states_topic_name: str
    anomaly_detection_topic_name: str
    marker_publisher_topic_name: str 
    trajectory_publisher_topic_name: str 
    gripper_publisher_topic_name: str
    multi_gpu: bool
    resume_train: bool
    overwrite_data: bool
    overwrite_image_data: bool 
    dist_2_schnapp_treshold: float
    deg_2_schnapp_treshold: float
    gripper_schnapp_threshold: float
    gripper_reverse_open: bool

    tcp_offset_x: float
    tcp_offset_y: float
    tcp_offset_z: float

    # not use yet
    tcp_offset_r: float
    tcp_offset_p: float
    tcp_offset_y: float
    

    
    # bag-level split and eval settings
    num_valid_bags: int = 2
    num_test_bags: int = 2
    eval_threshold_low: float = 0.2
    eval_threshold_high: float = 0.8
    eval_ddim_steps: int = 50
    seed: int = 42

    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "diffusion-policy"
    wandb_entity: str = 'ipk'
    wandb_run_name: str = ''
    
    # Gripper integration
    use_gripper: bool = False

    gripper_state_source: str = ''   # "joint_states" or "topic"
    gripper_state_topic_name: str = ''  # topic name if using "topic" source

    gripper_joint_names: tuple[str] = ('', '')

    gripper_min_width_m: float = 0.0
    gripper_max_width_m: float = 0.0

    gripper_command_mode: str = ''  # "single" or "trajectory"
    gripper_single_command_idx: int = 10 #index from the predicted gripper pose array if using "single" mode

    base_rgb_topics: tuple[str, ...] = ()
    base_depth_topics: tuple[str, ...] = ()
    up_rgb_topics: tuple[str, ...] = ()
    up_depth_topics: tuple[str, ...] = ()

    # Max relative timestamp (ms) used for normalization of camera sync features
    cam_time_max_ms: float = 500.0

    
    def __post_init__(self):

        if not self.base_rgb_topics:
            self.base_rgb_topics = (self.camera_topic_name,)
        if not self.base_depth_topics:
            self.base_depth_topics = (self.depth_cam_topic_name,)


        if not self.image_random_crop_aug:
            self.image_forward_shape = self.image_resize_shape

        if self.vision_feature_dim != 512 and self.encoder_name.lower() == 'resnet':
            #print('config post init, forcing vision_feature_dim to 512 from', self.vision_feature_dim)
            self.vision_feature_dim = 512
        
        if len(self.additional_cls_layers) > 0 and self.encoder_name.lower() == 'vit':
            self.vision_feature_dim = 384
            self.vision_feature_dim = self.vision_feature_dim * (len(self.additional_cls_layers) + 1)
            #print('config post init additional_cls_layers', self.additional_cls_layers, self.vision_feature_dim)
        
        if self.encoder_name.lower() == 'vd':
            self.vision_feature_dim = 512 * 4 + 768

        self.vision_feature_dim = self.num_views * self.vision_feature_dim

        # adding the ms time stamp of each camera and modality to the robot input dims
        if self.with_depth:
            self.lowdim_obs_dim = self.lowdim_obs_dim + (self.num_views * 2)
        else:
            self.lowdim_obs_dim = self.lowdim_obs_dim + self.num_views

        
        lowdim_obs_dim = self.lowdim_obs_dim
        if self.extend_obs_horizon_agent_pos:
            lowdim_obs_dim = self.lowdim_obs_dim * self.pred_horizon
        else:
            lowdim_obs_dim = self.lowdim_obs_dim * self.obs_horizon

        #print('config post init, lowdim_obs_dim', lowdim_obs_dim)

        self.obs_dim = self.vision_feature_dim * self.obs_horizon + lowdim_obs_dim
        print('config post init, vision_feature_dim', self.vision_feature_dim, lowdim_obs_dim, self.obs_horizon)
        print('config post init, obs_dim', self.obs_dim)
        self.joint_names_order = tuple(self.joint_names_order)
        self.gripper_joint_names = tuple(self.gripper_joint_names)
        
        if len(self.joint_names_order) != self.num_joints:
            raise ValueError(f"num_joints ({self.num_joints}) must match length of joint_names_order {len(self.joint_names_order)}")

        if self.gripper_state_source not in ("joint_states", "topic"):
            raise ValueError("gripper_state_source must be 'joint_states' or 'topic'")

        if self.use_depth_positional_encoding:
            print(f'setting depth_cut_treshold {self.depth_cut_treshold} -> {self.depth_cut_treshold_pe}')
            self.depth_cut_treshold = self.depth_cut_treshold_pe

            self.depth_pe_temperature = self._compute_depth_pe_temperature(
                depth_cut_threshold_m=self.depth_cut_treshold,
                min_band_mm=self.depth_pe_temperature_min_band,
                depth_channels=self.depth_pe_channels,
            )
            print(f'config post init, depth_pe_temperature={self.depth_pe_temperature:.8f} '
                  f'(finest band ({self.depth_pe_channels//2}): 1 cycle per {self.depth_pe_temperature_min_band}mm, '
                  f'cut={self.depth_cut_treshold*1000:.1f}mm, channels={self.depth_pe_channels})')
                
        # Adjust total sampling probabilities
        total_p = (self.near_sampling_p + self.very_close_sampling_p + 
                self.finish_sampling_p + self.recovery_sampling_p + self.start_sampling_p + self.far_sampling_p)
        assert total_p <= 1.0, f"Sum of sampling probabilities ({total_p}) exceeds 1.0"

    @staticmethod
    def _compute_depth_pe_temperature(
        depth_cut_threshold_m: float,
        min_band_mm: float,
        depth_channels: int,
    ) -> float:
        """
        Compute the temperature so the finest frequency band completes
        exactly 1 full 2π cycle over `min_band_mm` millimeters of depth.

        Formula:
            dim_t_finest = temperature^(2*(half-1) / depth_channels)
            We want: (Δnorm * 2π) / dim_t_finest = 2π
            → dim_t_finest = Δnorm = min_band_mm / (threshold_mm)
            → temperature = dim_t ^ (depth_channels / (2*(half-1)))
        """
        half = depth_channels // 2
        if half <= 1:
            # With only 1 sin/cos pair, band 0 has dim_t=1 always
            return 1.0

        threshold_mm = depth_cut_threshold_m * 1000.0
        dim_t_target = min_band_mm / threshold_mm

        finest_exponent = 2.0 * (half - 1) / depth_channels
        temperature = dim_t_target ** (1.0 / finest_exponent)

        return temperature
        



# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # normalize to [0,1]
    denom = stats['max'] - stats['min']
    denom = np.where(denom == 0, 1.0, denom)
    ndata = (data - stats['min']) / denom
    # normalize to [-1, 1]
    return ndata * 2 - 1

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    denom = stats['max'] - stats['min']
    denom = np.where(denom == 0, 1.0, denom)
    return ndata * denom + stats['min']



def detect_start_phase(
    bag_info: dict,
    config,
) -> np.ndarray:
    """
    Detect the 'start' phase in a bag: the early orientation/positioning
    period before the robot enters sustained forward (Z-dominant) motion.

    Returns an array of frame indices belonging to the start phase.
    Returns empty array if no valid start phase is detected.
    """
    # Skip recovery bags
    if 'recoveries' in bag_info.get('bagpath', ''):
        return np.array([], dtype=np.int64)

    poses = np.array(bag_info['robot_poses'], dtype=np.float64)
    entries = int(bag_info['entries'])

    if entries < config.start_min_frames + config.start_transition_window + 2:
        return np.array([], dtype=np.int64)

    # Determine max frame index based on time limit
    img_ts = bag_info.get('img_ts', None)
    if img_ts is not None and len(img_ts) >= 2:
        img_ts = np.array(img_ts, dtype=np.float64)
        t0 = img_ts[0]
        elapsed_s = (img_ts - t0) / 1e9
        max_frame = int(np.searchsorted(elapsed_s, config.start_max_time_s))
    else:
        est_fps = config.robot_joint_state_sampling if config.time_dependent_sampling else 10
        max_frame = min(int(config.start_max_time_s * est_fps), entries - 1)

    max_frame = min(max_frame, entries - 1)
    if max_frame < config.start_min_frames + config.start_transition_window:
        return np.array([], dtype=np.int64)

    # Compute per-frame relative motion in local frame
    positions = poses[:max_frame + 1, :3]
    quats = poses[:max_frame + 1, 3:]

    local_dxyz = np.zeros((max_frame, 3), dtype=np.float64)
    rot_magnitudes = np.zeros(max_frame, dtype=np.float64)

    for i in range(max_frame):
        ref_R = R.from_quat(quats[i])
        cur_R = R.from_quat(quats[i + 1])

        dp_world = positions[i + 1] - positions[i]
        dp_local = ref_R.inv().apply(dp_world)
        local_dxyz[i] = dp_local

        R_rel = ref_R.inv() * cur_R
        rot_magnitudes[i] = np.linalg.norm(R_rel.as_rotvec())

    # Sliding window: compute z-dominance ratio
    w = config.start_transition_window
    n_windows = max_frame - w + 1
    if n_windows <= 0:
        return np.array([], dtype=np.int64)

    z_ratios = np.zeros(n_windows, dtype=np.float64)

    for i in range(n_windows):
        window = local_dxyz[i:i + w]
        abs_z = np.sum(np.abs(window[:, 2]))
        abs_xyz = np.sum(np.linalg.norm(window, axis=1))
        z_ratios[i] = abs_z / max(abs_xyz, 1e-12)

    # Find transition: first sustained window where z-dominance exceeds threshold
    transition_idx = None
    for i in range(len(z_ratios)):
        if z_ratios[i] >= config.start_z_dominance_ratio:
            if i + 1 < len(z_ratios) and z_ratios[i + 1] >= config.start_z_dominance_ratio * 0.8:
                transition_idx = i
                break
            elif i + 1 >= len(z_ratios):
                transition_idx = i
                break

    if transition_idx is None:
        # No clear transition — check if beginning has lateral/rotational motion
        early_lateral = np.sum(np.linalg.norm(local_dxyz[:min(w, max_frame), :2], axis=1)) * 1000
        early_rot = np.sum(rot_magnitudes[:min(w, max_frame)])

        if early_lateral < config.start_min_lateral_motion_mm and early_rot < np.deg2rad(2.0):
            return np.array([], dtype=np.int64)
        else:
            transition_idx = min(max_frame // 2, max_frame - w)

    # Start phase = frames from 0 to transition point
    start_end_frame = transition_idx + w // 2

    # Validate: check cumulative lateral + rotation in the start phase
    cum_lateral_mm = np.sum(np.linalg.norm(local_dxyz[:start_end_frame, :2], axis=1)) * 1000
    cum_rot_deg = np.degrees(np.sum(rot_magnitudes[:start_end_frame]))

    if cum_lateral_mm < config.start_min_lateral_motion_mm and cum_rot_deg < 1.0:
        return np.array([], dtype=np.int64)

    if start_end_frame < config.start_min_frames:
        return np.array([], dtype=np.int64)

    # Return valid start indices (must leave room for full sample)
    sample_len = config.pred_horizon + config.obs_horizon
    valid_start_indices = list(range(0, min(start_end_frame, entries - sample_len)))

    return np.array(valid_start_indices, dtype=np.int64)

    

def tcp_pose_to_relative(pos, quat_xyzw, ref_pos, ref_quat_xyzw):
    """
    Returns 6D relative pose [p_rel(3), rotvec_rel(3)] where translation is in ref EE frame.
    """
    ref_R = R.from_quat(ref_quat_xyzw)
    cur_R = R.from_quat(quat_xyzw)

    p_rel = ref_R.inv().apply(np.asarray(pos) - np.asarray(ref_pos))
    R_rel = ref_R.inv() * cur_R
    rotvec_rel = R_rel.as_rotvec()
    return np.concatenate([p_rel, rotvec_rel], axis=0).astype(np.float32)

def tcp_pose_from_relative(rel_6d, ref_pos, ref_quat_xyzw):
    """
    Returns absolute (pos, quat_xyzw).
    """
    ref_R = R.from_quat(ref_quat_xyzw)

    p_rel = np.asarray(rel_6d[:3])
    rv_rel = np.asarray(rel_6d[3:6])
    R_rel = R.from_rotvec(rv_rel)

    pos_abs = np.asarray(ref_pos) + ref_R.apply(p_rel)
    R_abs = ref_R * R_rel
    quat_abs = R_abs.as_quat().astype(np.float32)
    return pos_abs.astype(np.float32), quat_abs



class LoadAndCropSample:
    """
    Unified image loader that:
    1. Tries to load from the pre-cropped directory (fast path: smaller files).
    2. Falls back to loading the original + applying the crop box on the fly.
    
    Returns a dict with 'image' (PIL), 'depth' (np.float32 array in meters),
    'depth_completed_path', and 'is_cropped' flag.
    
    This replaces the separate Image.open() calls scattered in __getitem__
    and ensures consistent loading + cropping in one place.
    """

    def __init__(
        self,
        crop_box: tuple,
        original_root: str,
        cropped_root: tuple,
        load_cropped: bool = True,
        with_depth: bool = True,
        depth_cut_threshold: float = 0.5,
    ):
        """
        Args:
            crop_box: (left, top, right, bottom) for PIL crop
            original_root: path to original unpacked images (e.g. /data/rosbags_unpacked)
            cropped_root: list of paths to pre-cropped images (e.g. /data/rosbags_unpacked_crop)
            load_cropped: if True, try cropped_root first
            with_depth: whether to load depth images
            depth_cut_threshold: depth values above this (in meters) are zeroed
        """
        self.crop_box = crop_box
        self.original_root = original_root
        self.cropped_root = cropped_root
        self.load_cropped = load_cropped
        self.with_depth = with_depth
        self.depth_cut_threshold = depth_cut_threshold

    def _to_cropped_path(self, original_path: str) -> str | None:
        """Map an original-root path to its cropped-root equivalent."""
        if not original_path or not self.cropped_root:
            return None
        if self.original_root and original_path.startswith(self.original_root):
            cropped_root = random.sample(self.cropped_root, 1)[0]
            new_path = original_path.replace(self.original_root, cropped_root, 1)
            #print(f'replacing {original_path} with {cropped_root} to {new_path}')
            return new_path
        return None

    def _load_image(self, path: str, debug=False) -> Image.Image | None:
        """
        Load an image, preferring the pre-cropped version.
        If cropped version doesn't exist, load original and crop.
        """
        if not path:
            return None


        # Fast path: load pre-cropped
        if self.load_cropped:
            cropped_path = self._to_cropped_path(path)
            
            if debug:
                print(f'[LoadAndCropSample] load crop image {cropped_path}')
            
            if cropped_path and os.path.exists(cropped_path):
                img = Image.open(cropped_path)
                return img
        
        if debug:
            print(f'[LoadAndCropSample] load image {path}, cropped {self.load_cropped}, crop box {self.crop_box}')

        # Slow path: load original, apply crop
        if not os.path.exists(path):            
            if debug:
                print(f'[LoadAndCropSample] load image does not exist {path}')
            return None

        if debug:
            print(f'[LoadAndCropSample] load image and crop {path} to {self.crop_box}')
        img = Image.open(path)
        img = img.crop(self.crop_box)
        return img

    def load_rgb(self, path: str, debug=False) -> Image.Image:
        """Load and crop an RGB image. Raises if not found."""
        img = self._load_image(path, debug=debug)
        if img is None:
            raise FileNotFoundError(f"RGB image not found: {path}")
        return img

    def load_depth(self, path: str, debug=False) -> np.ndarray | None:
        """Load and crop a depth image. Returns float32 array in meters, thresholded."""
        if not self.with_depth or not path:
            return None
        img = self._load_image(path, debug=debug)
        if img is None:
            return None
        depth = np.array(img, dtype=np.float32) / 1000.0
        depth[depth > self.depth_cut_threshold] = 0.0
        return depth

    def load_completed_depth(self, path: str, debug=False) -> np.ndarray | None:
        """Load and crop a completed-depth image (same logic as depth)."""
        if not path:
            return None
        img = self._load_image(path, debug=debug)
        if img is None:
            return None
        depth = np.array(img, dtype=np.float32) / 1000.0
        return depth

    def __call__(
        self,
        rgb_path: str,
        depth_path: str | None = None,
        completed_path: str | None = None,
        debug: bool = False,
    ) -> dict:
        """
        Load all images for one frame/view.

        Returns:
            {
                'image': PIL.Image (RGB, already cropped),
                'depth': np.ndarray (H, W) float32 meters (or absent),
                'depth_completed_path': str (for lazy aug loading),
                'is_cropped': True,
                'debug': bool,
            }
        """
        sample = {
            'is_cropped': True,
            'debug': debug,
        }

        # RGB (mandatory)
        sample['image'] = self.load_rgb(rgb_path, debug=debug)

        # Depth (optional)
        if self.with_depth and depth_path:
            depth = self.load_depth(depth_path, debug=debug)
            if depth is not None:
                sample['depth'] = depth

        # Completed depth path for lazy loading in augmentations
        if completed_path:
            sample['depth_completed'] = self.load_completed_depth(completed_path)
        return sample


class SensorDropoutBackground(object):
    """
    Simulate full-view sensor dropouts:
    - with probability `zero_prob`: set the whole view to zero (image + depth).
    - otherwise: replace the whole view image with a random background image
      sampled like DepthCompletedBackgroundExchange, and synthesize depth
      noise based on the statistics of the current depth tensor.

    This works on a *whole view* tensor: shape (T, H, W, 3) and (T, H, W, 1).
    It can be applied to either base or up view, controlled by the call site.
    """
    def __init__(self, background_root='', with_depth=False, zero_prob=0.5):
        self.with_depth = with_depth
        self.zero_prob = zero_prob

        self.folder_images: dict[str, list[str]] = {}
        self.folders: list[str] = []

        if background_root and os.path.isdir(background_root):
            for root, dirs, files in os.walk(background_root):
                imgs = [
                    os.path.join(root, f) for f in files
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                if imgs:
                    self.folder_images[root] = imgs

            self.folders = list(self.folder_images.keys())
            total = sum(len(v) for v in self.folder_images.values())
            print(f"[SensorDropoutBackground] Found {total} background images "
                  f"across {len(self.folders)} folders in {background_root}")
        else:
            print(f"[SensorDropoutBackground] No backgrounds found in '{background_root}'")

    def _sample_background_path(self) -> str | None:
        if not self.folders:
            return None
        folder = random.choice(self.folders)
        return random.choice(self.folder_images[folder])

    def __call__(self, images: np.ndarray, depths: np.ndarray, base_depths: np.ndarray | None, view_name: str = '', debug=False):
        """
        images: (T, H, W, 3) float32
        depths: (T, H, W, 1) float32 or None
        view_name: 'base'/'up' for debugging (optional)
        """
        if images is None or images.size == 0:
            return images, depths

        if debug:
            print('applying SensorDropoutBackground with {} background images'.format(len(self.folders)))

        # Case A: zero out everything
        if random.random() < self.zero_prob or not self.folders:
            images[...] = 0.0
            if self.with_depth and depths is not None:
                depths[...] = 0.0
            if debug:
                print('SensorDropoutBackground zero everything')
            return images, depths

        # Case B: replace image with random background
        T_frames, H, W, _ = images.shape
        bg_path = self._sample_background_path()
        if bg_path is None:
            images[...] = 0.0
            if self.with_depth and depths is not None:
                depths[...] = 0.0
            if debug:
                print('SensorDropoutBackground broken background path, dropping to zeros')
            return images, depths

        try:
            bg_img = Image.open(bg_path).convert('RGB')
        except Exception:
            images[...] = 0.0
            if self.with_depth and depths is not None:
                depths[...] = 0.0
            if debug:
                print('SensorDropoutBackground broken background image, dropping to zeros')
            return images, depths

        # Use a torchvision RandomResizedCrop to get matching size
        tf_crop = T.RandomResizedCrop(size=(H, W), antialias=True)
        for t in range(T_frames):
            bg_cropped = np.array(tf_crop(bg_img), dtype=np.float32)
            images[t] = bg_cropped

        # Depth: synthesize noise based on current depth stats
        if self.with_depth and base_depths is not None:
            valid = base_depths > 0
            if np.any(valid):
                mean_d = float(base_depths[valid].mean())
                std_d  = float(base_depths[valid].std())
                noise = np.random.normal(mean_d, std_d, depths.shape).astype(np.float32)
                noise[noise < 0] = 0.0
                thres = float(np.random.rand())
                mask = torch.rand(noise.shape) < thres
                noise[mask] = 0.0
                depths[...] = noise
                
                if debug:
                    print(f'SensorDropoutBackground setting random background and depth from {bg_path} and depth mean {mean_d} std {std_d} and t < {thres}')
                
            else:
                depths[...] = 0.0
                if debug:
                    print(f'SensorDropoutBackground setting random setting depth to 0 no valid depth inputs provided')

        return images, depths



def _advance_to_nearest(cur, prev, it, target_ts):
    """
    cur, prev: current / previous (conn, ts, raw) tuples or None
    it: iterator over messages
    target_ts: target ns timestamp
    returns (best_msg, new_prev, new_cur)
    """
    if it is None or cur is None:
        return None, prev, cur
    # Move forward while cur.ts < target_ts
    while True:
        conn_c, ts_c, raw_c = cur
        if ts_c >= target_ts:
            break
        prev = cur
        try:
            cur = next(it)
        except StopIteration:
            cur = None
            break
    # choose closer of prev/cur
    best = None
    if prev is not None and cur is not None:
        _, ts_p, _ = prev
        _, ts_c, _ = cur
        best = prev if abs(ts_p - target_ts) <= abs(ts_c - target_ts) else cur
    elif prev is not None:
        best = prev
    elif cur is not None:
        best = cur
    return best, prev, cur

class UseCompletedDepth(object):
    """
    During training, with probability p, replace the regular depth with
    the completed depth (if available). If swapped, optionally cut large
    holes into it to simulate real sensor failures.
    """
    def __init__(self, p=0.5, hole_cut_p=0.5, max_holes=5, max_hole_ratio=0.2):
        self.p = p
        self.hole_cut_p = hole_cut_p
        self.max_holes = max_holes
        self.max_hole_ratio = max_hole_ratio

    def __call__(self, sample):
        if 'depth_completed_path' not in sample:
            return sample

        if random.random() > self.p:
            return sample
            
        
        if sample.get('debug', False):
            print('UseCompletedDepth applied')

        if 'depth_completed' not in sample:
            completed_path = sample['depth_completed_path']
            if not os.path.exists(completed_path):
                return sample

            try:
                depth_completed = Image.open(completed_path)
            except Exception:
                return sample

            depth_completed = np.array(depth_completed, dtype=np.float32) / 1000
            sample['depth_completed'] = copy.deepcopy(depth_completed)

        else:
            depth_completed = copy.deepcopy(sample['depth_completed'])

        # With p=hole_cut_p, cut large holes into the completed depth
        if random.random() < self.hole_cut_p:
            h, w = depth_completed.shape[:2]
            n_holes = random.randint(1, self.max_holes)
            for _ in range(n_holes):
                hole_h = random.randint(h // 8, int(h * self.max_hole_ratio))
                hole_w = random.randint(w // 8, int(w * self.max_hole_ratio))
                y = random.randint(0, max(1, h - hole_h))
                x = random.randint(0, max(1, w - hole_w))
                depth_completed[y:y + hole_h, x:x + hole_w] = 0

        # Replace depth with completed version
        sample['depth'] = depth_completed
        return sample

class DepthCompletedBackgroundExchange(object):
    """
    Background exchange using depth-completed images for foreground/background
    segmentation. Loads a completed depth map, determines background pixels
    (depth > max_range or invalid), and replaces those pixels in the RGB image
    with a random crop from a random background image.

    Sampling: first picks a folder uniformly, then picks an image uniformly
    within that folder — ensures equal representation across folders regardless
    of how many images each contains.
    """
    def __init__(self, background_root='', max_range=0.5, p=0.5,
                 min_valid_m=0.01):
        self.max_range = max_range
        self.p = p
        self.min_valid_m = min_valid_m

        # Group images by their immediate parent folder
        self.folder_images = {}  # {folder_path: [image_path, ...]}
        if background_root and os.path.isdir(background_root):
            for root, dirs, files in os.walk(background_root):
                imgs_in_folder = [
                    os.path.join(root, f) for f in files
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                if imgs_in_folder:
                    self.folder_images[root] = imgs_in_folder

            self.folders = list(self.folder_images.keys())
            total = sum(len(v) for v in self.folder_images.values())
            print(f'[DepthCompletedBackgroundExchange] Found {total} background images '
                  f'across {len(self.folders)} folders in {background_root}')
        else:
            self.folders = []
            print(f'[WARN] DepthCompletedBackgroundExchange: no backgrounds in "{background_root}"')

    def _sample_background_path(self):
        """Uniform over folders, then uniform within the chosen folder."""
        folder = random.choice(self.folders)
        return random.choice(self.folder_images[folder])

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        if not self.folders:            
            return sample
            
        if sample.get('debug', False):
            print('DepthCompletedBackgroundExchange applied')


        if 'depth_completed' not in sample:
            # Need a completed depth to create the mask
            if 'depth_completed_path' not in sample:
                if sample.get('debug', False):
                    print('DepthCompletedBackgroundExchange depth_completed_path not in sample')
                return sample
                
            completed_path = sample['depth_completed_path']
            if not os.path.exists(completed_path):
                return sample

            try:
                depth_completed = Image.open(completed_path)
            except Exception as e:
                if sample.get('debug', False):
                    print(f'DepthCompletedBackgroundExchange depth_completed load err {e}')
                return sample

            # Convert to meters
            depth_m = np.array(depth_completed, dtype=np.float32) / 1000.0

            sample['depth_completed'] = copy.deepcopy(depth_m)
        else:
            depth_m = copy.deepcopy(sample['depth_completed'])

        # Background mask: too far OR invalid
        bg_mask = (depth_m > self.max_range) | (depth_m < self.min_valid_m)

        if not np.any(bg_mask):        
            if sample.get('debug', False):
                print(f'DepthCompletedBackgroundExchange no bg_mask')
            return sample  # entire image is foreground

        # Load and crop random background (folder-balanced sampling)
        bg_path = self._sample_background_path()
        try:
            bg_img = Image.open(bg_path).convert('RGB')
        except Exception as e:
            if sample.get('debug', False):
                print(f'DepthCompletedBackgroundExchange bg_img load err {e}')
            return sample

        # Get current image size (PIL: width, height)
        img = sample['image']
        w_img, h_img = img.size

        tf = T.RandomResizedCrop(size=(h_img, w_img), antialias=True)
        bg_cropped = np.array(tf(bg_img), dtype=np.uint8)

        # Resize mask to match image if sizes differ
        h_mask, w_mask = bg_mask.shape[:2]
        if (h_mask, w_mask) != (h_img, w_img):
            mask_pil = Image.fromarray((bg_mask.astype(np.uint8) * 255))
            mask_pil = mask_pil.resize((w_img, h_img), Image.NEAREST)
            bg_mask = np.array(mask_pil) > 127

        # Replace background pixels in the RGB image
        img_arr = np.array(img, dtype=np.uint8)

        if bg_cropped.shape[:2] != img_arr.shape[:2]:
            bg_cropped = np.array(
                Image.fromarray(bg_cropped).resize((w_img, h_img), Image.BILINEAR),
                dtype=np.uint8
            )

        img_arr[bg_mask] = bg_cropped[bg_mask]
        sample['image'] = Image.fromarray(img_arr)
        if sample.get('debug', False):
            print(f'DepthCompletedBackgroundExchange bg exchanged')

        return sample

class Depth2Pil(object):
    def __call__(self, sample):
        if 'depth' in sample:
            sample['depth'] = Image.fromarray(np.array(sample['depth']*1000, dtype=np.uint16))
        return sample


class CropSample(object):
    """Crop image and depth. Skips if sample is already cropped."""
    def __init__(self, box):
        self.box = box

    def __call__(self, sample):
        if sample.get('is_cropped', False):
            return sample

        sample['image'] = sample['image'].crop(self.box)
        if 'depth' in sample and isinstance(sample['depth'], Image.Image):
            sample['depth'] = sample['depth'].crop(self.box)
        elif 'depth' in sample and isinstance(sample['depth'], np.ndarray):
            l, t, r, b = self.box
            sample['depth'] = sample['depth'][t:b, l:r]

        sample['is_cropped'] = True
        return sample
    

class ResizeSample(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        sample['image'] = sample['image'].resize(self.size)
        if 'depth' in sample:
            sample['depth'] = sample['depth'].resize(self.size)
        return sample
    

class ColorJitter(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.tf = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, sample):
        sample['image'] = self.tf(sample['image'])
        return sample
    


class RandomResizedCrop(object):
    def __init__(self, size=(224, 224), antialias=True):
        self.tf = T.RandomResizedCrop(size=size, antialias=antialias)

    def __call__(self, sample):
        sample['image'] = self.tf(sample['image'])
        return sample


class RandomCutout(object):
    """
    Randomly mask rectangular patches in the image.
    Forces the model to not rely on any single image region.
    Preserves spatial layout — only removes information, doesn't warp it.
    """
    def __init__(self, n_holes=3, max_h_ratio=0.15, max_w_ratio=0.15, fill_value=0, p=0.5):
        self.n_holes = n_holes
        self.max_h_ratio = max_h_ratio
        self.max_w_ratio = max_w_ratio
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        img = np.array(sample['image'], dtype=np.uint8)
        h, w, c = img.shape

        for _ in range(self.n_holes):
            if random.random() > self.p:
                continue

            hole_h = random.randint(1, int(h * self.max_h_ratio))
            hole_w = random.randint(1, int(w * self.max_w_ratio))

            y = random.randint(0, h - hole_h)
            x = random.randint(0, w - hole_w)

            img[y:y+hole_h, x:x+hole_w, :] = self.fill_value

        sample['image'] = Image.fromarray(img)
        return sample
    
class GaussianNoise(object):
    """Simulate camera sensor noise."""
    def __init__(self, mean=0, std=5, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        img = np.array(sample['image'], dtype=np.float32)
        noise = np.random.normal(self.mean, self.std, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        sample['image'] = Image.fromarray(img)
        return sample

class RandomGaussianBlur(object):
    """Simulate slight defocus / motion blur."""
    def __init__(self, kernel_range=(3, 7), p=0.3):
        self.kernel_range = kernel_range
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        from PIL import ImageFilter
        k = random.choice(range(self.kernel_range[0], self.kernel_range[1]+1, 2))
        sample['image'] = sample['image'].filter(ImageFilter.GaussianBlur(radius=k//2))
        return sample

class RandomShadow(object):
    """Add random dark/bright elliptical patches simulating lighting changes."""
    def __init__(self, n_spots=2, intensity_range=(0.5, 1.5), max_radius_ratio=0.3, p=0.3):
        self.n_spots = n_spots
        self.intensity_range = intensity_range
        self.max_radius_ratio = max_radius_ratio
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        img = np.array(sample['image'], dtype=np.float32)
        h, w, _ = img.shape

        for _ in range(self.n_spots):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            rx = random.randint(w // 10, int(w * self.max_radius_ratio))
            ry = random.randint(h // 10, int(h * self.max_radius_ratio))
            intensity = random.uniform(*self.intensity_range)

            Y, X = np.ogrid[:h, :w]
            mask = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0

            img[mask] = np.clip(img[mask] * intensity, 0, 255)

        sample['image'] = Image.fromarray(img.astype(np.uint8))
        return sample

class DepthAwareCutout(object):
    """
    Randomly mask either foreground or background patches
    using depth to determine what's what.
    More realistic than random rectangular cutout.
    """
    def __init__(self, depth_threshold=0.25, fg_p=0.2, bg_p=0.3, 
                 max_patch_ratio=0.2, n_patches=3):
        self.depth_threshold = depth_threshold
        self.fg_p = fg_p
        self.bg_p = bg_p
        self.max_patch_ratio = max_patch_ratio
        self.n_patches = n_patches

    def __call__(self, sample):
        if 'depth' not in sample:
            return sample

        img = np.array(sample['image'], dtype=np.uint8)
        depth = np.array(sample['depth'], dtype=np.float32) / 1000.0
        h, w = img.shape[:2]

        fg_mask = (depth > 0.05) & (depth < self.depth_threshold)
        bg_mask = ~fg_mask

        r = random.random()

        if r < self.fg_p:
            target_mask = fg_mask
        elif r < self.fg_p + self.bg_p:
            target_mask = bg_mask
        else:
            return sample

        # find random patches within target region
        ys, xs = np.where(target_mask)
        if len(ys) < 10:
            return sample

        for _ in range(self.n_patches):
            idx = random.randint(0, len(ys) - 1)
            cy, cx = ys[idx], xs[idx]
            ph = random.randint(5, int(h * self.max_patch_ratio))
            pw = random.randint(5, int(w * self.max_patch_ratio))

            y1 = max(0, cy - ph // 2)
            y2 = min(h, cy + ph // 2)
            x1 = max(0, cx - pw // 2)
            x2 = min(w, cx + pw // 2)

            # only mask pixels that belong to target region
            patch_mask = target_mask[y1:y2, x1:x2]
            img[y1:y2, x1:x2][patch_mask] = random.randint(0, 30)

        sample['image'] = Image.fromarray(img)
        return sample

class DepthDropout(object):
    """
    Simulate random depth pixel failures.
    Real depth sensors frequently return 0 for:
    - reflective surfaces
    - edges between objects
    - out-of-range pixels
    - sensor noise

    """
    def __init__(self, dropout_p_min=0.05, dropout_p_max=1.0, p=0.5):
        self.dropout_p_min = dropout_p_min
        self.dropout_p_max = dropout_p_max
        self.p = p

    def __call__(self, sample):
        if 'depth' not in sample or random.random() > self.p:
            return sample

        if sample.get('debug', False):
            print('DepthDropout applied')

        depth = sample['depth']
        h, w = depth.shape[:2]

        # random individual pixel dropout
        dropout_p = self.dropout_p_min + random.random() * (self.dropout_p_max - self.dropout_p_min)

        mask = np.random.random((h, w)) < dropout_p
        depth[mask] = 0

        sample['depth'] = depth
        return sample


class DepthGaussianNoise(object):
    """
    Simulate depth sensor measurement noise.
    Real sensors: noise scales with distance² (quadratic).
    Typical: ±2mm at 0.5m, ±5mm at 1m.
    """
    def __init__(self, base_std_mm=1.5, p=0.5, max_off_mm=3.0):
        self.base_std_m = base_std_mm / 1000
        self.max_off_m = max_off_mm / 1000

        self.p = p

    def __call__(self, sample):
        if 'depth' not in sample or random.random() > self.p:
            return sample

        if sample.get('debug', False):
            print('DepthGaussianNoise applied')

        depth = sample['depth']
        valid = depth > 0

        if not np.any(valid):
            return sample

        #print('depth', np.mean(depth[valid]), np.std(depth[valid]), np.min(depth[valid]), np.max(depth[valid]), np.sum(depth[valid]) / (480*640))
        noise = np.random.randn(*depth.shape).astype(np.float32) * self.base_std_m
        #print('noise', np.mean(noise), np.std(noise), np.min(noise), np.max(noise))
        noise = np.clip(noise, -self.max_off_m, self.max_off_m)
        #print('clipped', np.mean(noise), np.std(noise), np.min(noise), np.max(noise))

        depth[valid] = depth[valid] + noise[valid]
        depth[depth < 0] = 0
        
        sample['depth'] = depth
        return sample

class DepthEdgeNoise(object):
    """
    Simulate depth bleeding at object edges.
    Real depth sensors produce noisy/invalid readings
    at depth discontinuities.
    """
    def __init__(self, edge_width=3, dropout_p=0.5, p=0.3):
        self.edge_width = edge_width
        self.dropout_p = dropout_p
        self.p = p

    def __call__(self, sample):
        if 'depth' not in sample or random.random() > self.p:
            return sample

        
        if sample.get('debug', False):
            print('DepthEdgeNoise applied')


        depth = sample['depth']

        # find edges via gradient magnitude
        grad_x = ndimage.sobel(depth, axis=1)
        grad_y = ndimage.sobel(depth, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # threshold for "significant depth edge"
        threshold = np.percentile(magnitude[magnitude > 0], 90)
        edge_mask = magnitude > threshold

        # dilate edges
        edge_mask = ndimage.binary_dilation(
            edge_mask, iterations=self.edge_width
        )

        # randomly drop or corrupt edge pixels
        drop = np.random.random(depth.shape) < self.dropout_p
        depth[edge_mask & drop] = 0

        sample['depth'] = depth
        return sample



class DepthGlobalOffset(object):
    """
    Simulate small global depth bias from calibration drift
    or temperature changes. Real sensors shift ±3-5mm over time.
    """
    def __init__(self, max_offset_mm=3.0, p=0.3):
        self.max_offset_mm = max_offset_mm
        self.p = p

    def __call__(self, sample):
        if 'depth' not in sample or random.random() > self.p:
            return sample

        
        if sample.get('debug', False):
            print('DepthGlobalOffset applied')

        depth = sample['depth']
        valid = depth > 0

        offset = random.uniform(-self.max_offset_mm, self.max_offset_mm)
        depth[valid] += offset
        depth[depth < 0] = 0

        sample['depth'] = depth
        return sample

