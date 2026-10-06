# dataset/ros_bag_dataset.py

import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
import os
import copy
import random
import json
import time
from io import BytesIO
from icecream import ic
from datasets.ros_bag_utils import *
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp



class ROSBagDataset(Dataset):
    def __init__(self, config: Config, transform=None, train=True, time_crit=None, log=True):
        self.config = config
        self.fwd_kin = None #load_urdf(config.urdf_path, config.tcp_link_name, config.base_link_name)
        self.train = train
        self.log = log

        self.data = {}
        self.time_crit = {} if time_crit is None else time_crit

        #self.time_crit['maxdate'] = 20260623
        #self.time_crit['maxdate_time'] = 74142

        # ─── Pre-cropped image loader ───!
        self.unpack_path = config.unpack_path
        self.unpack_npy_paths = config.unpack_npy_paths
        cropped_unpack_path = config.cropped_unpack_path
        if isinstance(cropped_unpack_path, str):
            cropped_unpack_path = (cropped_unpack_path,)
        self.load_cropped = config.load_cropped 
        self.cropped_unpack_path = cropped_unpack_path

        self.image_loader = LoadAndCropSample(
            crop_box=config.image_static_crop,
            original_root=config.unpack_path,
            cropped_root=self.cropped_unpack_path if self.load_cropped else None,
            load_cropped=self.load_cropped,
            with_depth=config.with_depth,
            depth_cut_threshold=config.depth_cut_treshold,
        )

        tfs = []
        pre = [Depth2Pil()]

        if config.random_background_with_depth and config.with_depth and config.background_root and os.path.isdir(config.background_root):
            tfs.append(DepthCompletedBackgroundExchange(
                background_root=config.background_root,
                max_range=config.depth_cut_treshold,  
                p=0.25  # amount of smaple views which get this exchange
            ))

        if config.with_depth and config.use_depth_completed:
            tfs.append(UseCompletedDepth(
                p=0.25,
                hole_cut_p=0.5,
                max_holes=5,
                max_hole_ratio=0.2
            ))

            
        if config.image_brightness_aug:
            if log:
                ic("brightness augmentation enabled")
            tfs.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

        #tfs.append(RandomCutout(n_holes=3, max_h_ratio=0.12, max_w_ratio=0.12, p=0.5))
        #tfs.append(GaussianNoise(mean=0, std=3, p=0.5))
        #tfs.append(RandomGaussianBlur(kernel_range=(3, 5), p=0.2))
        #tfs.append(RandomShadow(n_spots=2, p=0.2))

        #tfs.append(DepthGaussianNoise(p=0.25))
        #tfs.append(DepthDropout(p=0.25))
        #tfs.append(DepthEdgeNoise(edge_width=2, p=0.25))
        #tfs.append(DepthGlobalOffset(max_offset_mm=2.0, p=0.25))
        


        '''
        if config.image_random_crop_aug:
            tfs.append(RandomResizedCrop(size=(self.config.image_forward_shape), antialias=True))
        elif config.apply_image_static_crop:
            pre.append(CropSample(box=self.config.image_static_crop))

        if not config.apply_image_static_crop:
            pre.append(ResizeSample(size=self.config.image_forward_shape))
        '''

        self.augmentation_transforms = T.Compose(tfs)          
        self.preprocesses = T.Compose(pre)     

        # Sensor dropout / background replacement for whole views (base/up)
        self.sensor_dropout_bg = None
        if config.background_root and os.path.isdir(config.background_root):
            self.sensor_dropout_bg = SensorDropoutBackground(
                background_root=config.background_root,
                with_depth=config.with_depth,
                zero_prob=self.config.sensor_drop_zero_p,   # 50% zeros, 50% background+noise
            )
        else:
            if log:
                print('no backgouroudn rooot!!!!!!!!!!!!!!')

        if log:    
            print('applying train augs of', self.augmentation_transforms)
            ic(config.dataset_path)
            
        self.read_messages_js_at_img_timestamp()

        # --- Whitelist filtering (before anything else) ---
        if self.config.use_whitelist and self.config.whitelist_path and self.train:
            if log:
                print('[INFO] applying whitelist, use /dataset/whitelist.py to generate a new whitelist once new data is unpacked to filter bad from good bags')
            self._apply_whitelist()

        self.bag_names = sorted(list(self.data.keys()))
        self.sample_len = self.config.pred_horizon + self.config.obs_horizon

         # --- Up-view availability (bag-level) ---
        self.bags_with_up: list[str] = []
        self.bags_with_up_dates: list[str] = []
        self.bags_without_up: list[str] = []
        for bn in self.bag_names:
            bag_dir = os.path.join(self.unpack_path, bn)
            has_up = False
            if os.path.isdir(bag_dir):
                files = os.listdir(bag_dir)
                # consider bag "with up" if it has *any* RGB up-view image
                has_up = any(f.endswith("_rgb_up.png") for f in files)

            if has_up:
                self.bags_with_up.append(bn)
                bdate = bn.split('_')[1]
                if bdate not in self.bags_with_up_dates:
                    self.bags_with_up_dates.append(bdate)
            else:
                self.bags_without_up.append(bn)
        if log:
            print(f'up view bag dates: {len(self.bags_with_up_dates)}')
        for bi, bdate in enumerate(self.bags_with_up_dates):
            print(bi+1, bdate)
        if log:
            print(f"[up-balance] bags with up-view: {len(self.bags_with_up)} | " f"without up-view: {len(self.bags_without_up)}")


        # --- Goal-group clustering ---
        if self.config.use_goal_balanced_sampling:
            self.goal_groups, self.goal_centroids, self.bag_to_goal_group = self._build_goal_groups()
        else:
            self.goal_groups = [[bn] for bn in self.bag_names]
            self.bag_to_goal_group = {bn: i for i, bn in enumerate(self.bag_names)}

        # ─────────────────────────────────────────────────────
        # Progress classification (far / near / very_close / finish / recovery)
        #   (this block is UNCHANGED from the original)
        # ─────────────────────────────────────────────────────
        has_nears = []
        has_very_close = []
        has_finishes = []
        has_fars = []
        has_recoveries = []
        has_starts = []

        for bag_name, bag in self.data.items():
            far_idx = []
            near_idx = []
            very_close_idx = []
            finish_idx = []
            recoveries_idx = []
            start_idx = []
            for idx in range(0, bag['entries'] - self.sample_len):
                if 'recoveries' in bag.get('bagpath', ''):
                    recoveries_idx.append(idx)
                else:

                    if 'start_idx' not in bag:
                        # Compute on the fly if not in bag_info (backwards compatibility)
                        start_indices = detect_start_phase(bag, self.config)
                        self.data[bag_name]['start_idx'] = start_indices
                    else:
                        start_indices = np.array(bag.get('start_idx', []))

                    if idx in set(start_indices):
                        start_idx.append(idx)
                    else:
                        pos = bag['rel_schnapp_pos'][idx + self.config.obs_horizon - 1]
                        rot = bag['rel_schnapp_rot'][idx + self.config.obs_horizon - 1]
                        if pos < self.config.finish_threshold_pos and rot < self.config.finish_threshold_deg:
                            finish_idx.append(idx)
                        elif pos < self.config.very_close_threshold_pos and rot < self.config.very_close_threshold_deg:
                            very_close_idx.append(idx)
                        elif pos < self.config.near_threshold_pos and rot < self.config.near_threshold_deg:
                            near_idx.append(idx)
                        else:
                            far_idx.append(idx)

            self.data[bag_name]['far_idx'] = np.array(far_idx)
            self.data[bag_name]['near_idx'] = np.array(near_idx)
            self.data[bag_name]['very_close_idx'] = np.array(very_close_idx)
            self.data[bag_name]['finish_idx'] = np.array(finish_idx)
            self.data[bag_name]['recoveries_idx'] = np.array(recoveries_idx)
            self.data[bag_name]['start_idx'] = np.array(start_idx)

            if len(far_idx) > 0:
                has_fars.append((len(far_idx), bag_name, far_idx))
            if len(near_idx) > 0:
                has_nears.append((len(near_idx), bag_name, near_idx))
            if len(very_close_idx) > 0:
                has_very_close.append((len(very_close_idx), bag_name, very_close_idx))
            if len(finish_idx) > 0:
                has_finishes.append((len(finish_idx), bag_name, finish_idx))
            if len(recoveries_idx) > 0:
                has_recoveries.append((len(recoveries_idx), bag_name, recoveries_idx))
            if len(start_idx) > 0:
                has_starts.append((len(start_idx), bag_name, start_idx))

        self.has_fars = has_fars
        self.has_nears = has_nears
        self.has_very_close = has_very_close
        self.has_finishes = has_finishes
        self.has_recoveries = has_recoveries
        self.has_starts = has_starts

        self.stage_map = {
            'start':      self.has_starts,
            "near":       self.has_nears,
            "very_close": self.has_very_close,
            "finish":     self.has_finishes,
            "recovery":   self.has_recoveries,
            "far":        self.has_fars,
        }
        self.progress_names = {0: 'start', 1: 'far', 2: 'near', 3: 'very_close', 4: 'finish'}


        
        self.stage_valid_groups_up = {}
        self.stage_valid_groups_base = {}

        # --- Build goal-balanced sampling tables ---
        if self.config.use_goal_balanced_sampling:
            self._build_stage_goal_structure()     
    

        self._length = sum([d['entries'] - self.sample_len for d in self.data.values()])


        self._n_fars = sum([n[0] for n in self.has_fars])
        self._n_nears = sum([n[0] for n in self.has_nears])
        self._n_very_close = sum([n[0] for n in self.has_very_close])
        self._n_finishes = sum([n[0] for n in self.has_finishes])
        self._n_recoveries = sum([n[0] for n in self.has_recoveries])
        self._n_starts = sum([n[0] for n in self.has_starts])

        self.near_sampling_p = self.config.near_sampling_p
        self.very_close_sampling_p = self.near_sampling_p + self.config.very_close_sampling_p
        self.finish_sampling_p = self.very_close_sampling_p + self.config.finish_sampling_p
        self.recovery_sampling_p = self.finish_sampling_p + self.config.recovery_sampling_p
        self.start_sampling_p = self.recovery_sampling_p + self.config.start_sampling_p
        self.far_sampling_p = self.start_sampling_p + self.config.far_sampling_p
        # remainng p is for the rest (aka far)

        # Adjust total sampling probabilities
        
        assert self.far_sampling_p == 1.0, f"Sum of sampling probabilities ({self.far_sampling_p}) do not sum up to 1.0"
        if not (0.0 <= self.config.multi_view_sampling_p <= 1.0):
            raise ValueError(f"multi_view_sampling_p must be in [0,1], got {self.config.multi_view_sampling_p}")
        
        
        self.mode_name = 'training' if self.train else 'testing'

        if log:
            print('#########################')
            print('dataset n bags', len(self.data))
            print('dataset len', self._length)
            print('{} with {} start samples ({}% of all samples), sampling p={}'.format(self.mode_name, self._n_starts, np.round(100*self._n_starts / self._length, 4), self.config.start_sampling_p))
            print('{} with {} far samples ({}% of all samples), sampling p={} '.format(self.mode_name, self._n_fars, np.round(100*self._n_fars / self._length, 4), self.config.far_sampling_p))
            print('{} with {} near samples ({}% of all samples), sampling p={}'.format(self.mode_name, self._n_nears, np.round(100*self._n_nears / self._length, 4), self.config.near_sampling_p))
            print('{} with {} very close samples ({}% of all samples), sampling p={}'.format(self.mode_name, self._n_very_close, np.round(100*self._n_very_close / self._length, 4), self.config.very_close_sampling_p))
            print('{} with {} finish samples ({}% of all samples), sampling p={}'.format(self.mode_name, self._n_finishes, np.round(100*self._n_finishes / self._length, 4), self.config.finish_sampling_p))
            print('{} with {} recovery samples ({}% of all samples), sampling p={}'.format(self.mode_name, self._n_recoveries, np.round(100*self._n_recoveries / self._length, 4), self.config.recovery_sampling_p))
            
            print('sampling thresholds')
            print('near between {}->{}'.format(0, self.near_sampling_p))
            print('very close between {}->{}'.format(self.near_sampling_p, self.very_close_sampling_p))
            print('finish between {}->{}'.format(self.very_close_sampling_p, self.finish_sampling_p))
            print('recovery between {}->{}'.format(self.finish_sampling_p, self.recovery_sampling_p))
            print('start between {}->{}'.format(self.recovery_sampling_p, self.start_sampling_p))
            print('far between {}->{}'.format(self.start_sampling_p, 1.0))
            print('#########################')

        # Approximate sample counts by up-view vs base-only bags
        self._n_up_bag_samples = 0
        self._n_no_up_bag_samples = 0
        for bn in self.bag_names:
            num = self.data[bn]['entries'] - self.sample_len
            if num <= 0:
                continue
            if bn in self.bags_with_up:
                self._n_up_bag_samples += num
            else:
                self._n_no_up_bag_samples += num

        total_up = self._n_up_bag_samples + self._n_no_up_bag_samples
        if total_up > 0:
            if log:
                print('[up-balance] approx train samples from up-view bags: '
                    f'{self._n_up_bag_samples} '
                    f'({100*self._n_up_bag_samples/total_up:.1f}%), '
                    f'from base-only bags: {self._n_no_up_bag_samples} '
                    f'({100*self._n_no_up_bag_samples/total_up:.1f}%), '
                    f'config.p_up={self.config.multi_view_sampling_p}')
        else:
            if log:
                print('[up-balance] no samples found when counting up-view vs base-only bags')


        self._indexes = []        
        self._indexes_dict = {}

        counter = 0
        for key in self.bag_names:
            train_samples = self.data[key]['entries'] - self.sample_len
            counter = counter + train_samples
            self._indexes.append((counter, train_samples, key))
            self._indexes_dict[key] = (counter, train_samples)

        max_ms = self.config.cam_time_max_ms
        num_views = self.config.num_views
        if self.config.with_depth:
            num_views = num_views * 2


        self.stats = {
            'action': {
                'min': np.array([-0.01, -0.01, 0.0, -0.03, -0.03, -0.15, 0.0, 0.0], dtype=np.float32),
                'max': np.array([ 0.01,  0.01, 0.032, 0.03, 0.03, 0.15, 0.50, 0.50], dtype=np.float32)
            },
            'agent_pose': {
                # 6 pose dims as before:
                'min': np.array(
                    [-0.01, -0.01, 0.0, -0.03, -0.03, -0.15] + [-0.1 for _ in range(num_views)],
                    dtype=np.float32
                ),
                'max': np.array(
                    [ 0.01,  0.01, 0.032, 0.03, 0.03, 0.15] + [max_ms for _ in range(num_views)],
                    dtype=np.float32
                )
            }
        }



    @staticmethod
    def get_offset(cur, ref):
        
        cur_pos = np.asarray(cur[:3], dtype=np.float32)
        ref_pos = np.asarray(ref[:3], dtype=np.float32)

        dist = float(np.linalg.norm(cur_pos - ref_pos))
        
        q1 = np.asarray(cur[3:], dtype=np.float32)
        q2 = np.asarray(ref[3:], dtype=np.float32)

        dot = float(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
        rot = float(2.0 * np.arccos(dot) * 180.0 / np.pi)
        #rot = np.rad2deg(rot)
        return dist, rot



    def read_messages_js_at_img_timestamp(self):
        if self.log:
            ic(self.config.dataset_path)

        for dataset_path in self.config.dataset_path:
            
            frame_interval = 1
            
            if self.log:
                ic(dataset_path)
                ic(frame_interval)
                ic("Reading all bag files / ROS2 bag directories in the directory")
            bag_paths = []

            for entry in os.listdir(dataset_path):
                full_path = os.path.join(dataset_path, entry)
                if os.path.isdir(full_path) and any(
                    name.startswith('metadata') and name.endswith('.yaml')
                    for name in os.listdir(full_path)
                ):
                    bag_paths.append(full_path)

            bag_paths.extend(
                os.path.join(dataset_path, f)
                for f in os.listdir(dataset_path)
                if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith('.bag')
            )

            if not bag_paths:
                raise FileNotFoundError(
                    "No ROS1 .bag files or ROS2 bag directories found in dataset_path."
                )

            bag_paths = sorted(bag_paths)
            #self.bag_paths = bag_paths
            #print('loaded rosbags', len(self.bag_paths))
            #for ib, bag_path in enumerate(self.bag_paths):
            #    print('bag {} | {}'.format(ib+1, bag_path))

            if self.log:
                ic(len(bag_paths))

            i = 0
            N_bags = len(bag_paths)
            if self.log:
                print('[WARM] hardcoded date section of bags')
            for bid, bag_path in enumerate(bag_paths):
                #if 'bag_20260212_065344' not in bag_path: # or '' not in bag_path:
                #    continue

                #if 'bag_20260203_100130' not in bag_path:
                #    continue

                if self.time_crit.get('fixbag'):
                    if self.time_crit.get('fixbag') not in bag_path:
                        continue

                #
                bag_date, bag_time = bag_path.split('_')[-2:]
                bag_date, bag_time = int(bag_date), int(bag_time)

                if self.time_crit.get('mindate') is not None:
                    if bag_date < self.time_crit.get('mindate'):
                        continue

                if self.time_crit.get('mindate_time') is not None:
                    if bag_time < self.time_crit.get('mindate_time') and bag_date <= self.time_crit.get('mindate'):
                        continue

                if self.time_crit.get('maxdate') is not None:
                    if bag_date > self.time_crit.get('maxdate'):
                        continue
                if self.time_crit.get('maxdate_time') is not None:
                    if bag_time > self.time_crit.get('maxdate_time') and bag_date >= self.time_crit.get('maxdate'):
                        continue
                    

                #if bag_date >= 20260203 and bag_time >= 100130:
                #    continue
                #if bag_date < 20260204:
                #    continue
                #if bag_date == 20260204 and bag_time < 105957:
                #    continue 
                #elif bag_date == 20260212 and bag_time =< 80148: # 82538:
                #if bag_date >= 20260210 and bag_time >= 151747:
                #    continue      
                      
                #if bag_date < 20260215:
                #    continue
                #elif bag_date == 20260210 and bag_time < 143928:
                #    continue
                
                good = False
                try:
                    good = self._read_single_bag_rosbags((i, bid+1), bag_path, frame_interval, N_bags)
                except Exception as e:
                    if self.log:
                        print('bag {} failed with {}'.format(bag_path, e))
                    raise e
                    
                if good:
                    i += 1

    def _get_camera_conns(self, reader):
        """Return (base_img_conns, base_depth_conns, up_img_conns, up_depth_conns)."""
        # Base view: try configured list in order
        base_rgb_conns = []
        base_depth_conns = []
        for topic in getattr(self.config, 'base_rgb_topics', (self.config.camera_topic_name,)):
            base_rgb_conns = [c for c in reader.connections if c.topic == topic]
            if base_rgb_conns:
                break

        for topic in getattr(self.config, 'base_depth_topics', (self.config.depth_cam_topic_name,)):
            base_depth_conns = [c for c in reader.connections if c.topic == topic]
            if base_depth_conns:
                break

        if not base_rgb_conns:
            raise ValueError(f"No base RGB camera topic found in bag. "
                             f"Tried: {getattr(self.config, 'base_rgb_topics', ())}")
        if not base_depth_conns:
            raise ValueError(f"No base depth topic found in bag. "
                             f"Tried: {getattr(self.config, 'base_depth_topics', ())}")

        # Up view (optional)
        up_rgb_conns = []
        up_depth_conns = []

        for topic in getattr(self.config, 'up_rgb_topics', ()):
            up_rgb_conns = [c for c in reader.connections if c.topic == topic]
            if up_rgb_conns:
                break

        for topic in getattr(self.config, 'up_depth_topics', ()):
            up_depth_conns = [c for c in reader.connections if c.topic == topic]
            if up_depth_conns:
                break

        return base_rgb_conns, base_depth_conns, up_rgb_conns, up_depth_conns
  

    def _read_single_bag_rosbags(self, bid, bag_path, frame_interval: int, N_bags=0):
        bagpath = Path(bag_path)
        bag_name = os.path.basename(str(bag_path).rstrip('/'))

        #print('bagpath', bagpath)
        #print('bag_name', bag_name)
        
        bag_dir = os.path.join(self.unpack_path, bag_name)
        info_dir = os.path.join(self.unpack_path, bag_name, 'bag_info.npy')
        if os.path.exists(bag_dir):
            
            imgs = sorted([f for f in os.listdir(bag_dir) if '_rgb.png' in f])
            depths = sorted([f for f in os.listdir(bag_dir) if '_depth.png' in f])
            robot_states = sorted([f for f in os.listdir(bag_dir) if '_robot_state.npy' in f])
            #print('imgs', imgs)
            #print('depths', depths)
            #print('robot_states', robot_states)
            info = os.path.exists(info_dir)

            corr_len = len(imgs) == len(depths) == len(robot_states)            
            num_samples = len(imgs)
            corr_idx = True
            for idx in range(len(imgs)):
                if '{}_rgb.png'.format(str(idx).zfill(6)) not in imgs:
                    corr_idx = False
                
                if '{}_depth.png'.format(str(idx).zfill(6)) not in depths:
                    corr_idx = False
                    
                if '{}_robot_state.npy'.format(str(idx).zfill(6)) not in robot_states:
                    corr_idx = False
                
        else:
            info = False
            corr_len = False
            corr_idx = False
            num_samples = 0
            
            os.makedirs(bag_dir)

        #print('corr_len', corr_len)
        #print('corr_idx', corr_idx)
        #print('num_samples', num_samples)

        if corr_len and corr_idx and num_samples > self.config.pred_horizon + self.config.obs_horizon and not self.config.overwrite_data and info:
            bag_info = np.load(info_dir, allow_pickle=True).item()
            bag_info['bagpath'] = str(bagpath)

            if bag_info['valid'] and 'recoveries' not in str(bagpath) and 'start_idx' not in bag_info:
                start_indices = detect_start_phase(bag_info, self.config)
                bag_info['start_idx'] = start_indices
                bag_info['has_start_phase'] = len(start_indices) > 0
                bag_info['start_end_frame'] = int(start_indices[-1]) if len(start_indices) > 0 else 0
            elif 'start_idx' not in bag_info:
                bag_info['start_idx'] = np.array([], dtype=np.int64)
                bag_info['has_start_phase'] = False
                bag_info['start_end_frame'] = 0


            #print('bag_info', bag_info.keys())
            if bag_info['valid']:
                self.data[bag_name] = bag_info
                if self.log:
                    print(f"\r{bid[0]+1} valid ({bid[1]-bid[0]+1} not valid) of {bid[1]}/{N_bags} loaded {bag_info['entries']} entries for {bag_name}", end='', flush=True)
                return True
            else:
                if self.log:
                    print('loaded bag {} not valid'.format(bag_name))
                return False
        return False 

            
    def __len__(self):
        return self._length

    # ─────────────────────────────────────────────────────────
    #  Whitelist filtering
    # ─────────────────────────────────────────────────────────

    def _apply_whitelist(self):
        """Remove bags not present in the whitelist JSON."""
        wl_path = self.config.whitelist_path
        if not os.path.isfile(wl_path):
            if self.log:
                print(f"[whitelist] WARNING: file not found: {wl_path}  → skipping filter")
            return

        with open(wl_path, "r") as f:
            wl_data = json.load(f)

        wl_set = set(wl_data.get("whitelist", []))
        before = len(self.data)
        removed_names = [k for k in self.data if k not in wl_set]
        for k in removed_names:
            del self.data[k]
        after = len(self.data)

        summary = wl_data.get("summary", {})
        if self.log:
            print(f"[whitelist] Loaded {len(wl_set)} accepted bags from {wl_path}")
        if self.log:
            print(f"[whitelist] Filtered {before} → {after} bags "
              f"({before - after} removed, {len(wl_set) - after} in whitelist but not loaded)")
        if summary:
            if self.log:
                print(f"[whitelist] Whitelist stats: "
                  f"total={summary.get('total')}, "
                  f"accepted={summary.get('accepted')}, "
                  f"rejected={summary.get('rejected')}")

    # ─────────────────────────────────────────────────────────
    #  Goal-group clustering
    # ─────────────────────────────────────────────────────────

    def _build_goal_groups(self):
        """
        Cluster bags by schnapp_pose xyz (greedy, ±threshold_m), but only within
        the SAME recording day.

        Recovery bags each get their own singleton group (as before).

        Returns
        -------
        groups    : list[list[str]]   – groups[g] = [bag_name, …]
        centroids : list[np.ndarray]  – xyz centroid per group
        bag2group : dict[str, int]    – bag_name → global group index
        """
        threshold = self.config.goal_group_threshold_m
        cap = self.config.max_bags_per_goal_group

        # Helper: extract date from bag name or bagpath (e.g. bag_20260204_092536 → 20260204)
        def _extract_date(bag_name: str, bag_info: dict) -> str:
            src = bag_info.get("bagpath", bag_name)
            base = os.path.basename(str(src).rstrip("/"))
            parts = base.split("_")
            if len(parts) >= 3 and parts[-2].isdigit():
                return parts[-2]          # e.g. "20260204"
            return "unknown"

        # Per-date clusters
        groups_by_date: dict[str, list[list[str]]] = {}
        centroids_by_date: dict[str, list[np.ndarray]] = {}

        # Global outputs
        groups: list[list[str]] = []
        centroids: list[np.ndarray] = []
        bag2group: dict[str, int] = {}

        for bag_name in sorted(self.data.keys()):
            bag = self.data[bag_name]

            # Recovery bags → own global group (no schnapp_pose clustering)
            if "recoveries" in bag.get("bagpath", ""):
                g_idx = len(groups)
                groups.append([bag_name])
                centroids.append(np.array([float("inf"), float("inf"), float("inf")]))
                bag2group[bag_name] = g_idx
                continue

            # Normal bags: cluster by schnapp_pose xyz within same date
            pos = np.array(bag["schnapp_pose"][:3], dtype=np.float64)
            date_key = _extract_date(bag_name, bag)

            d_groups = groups_by_date.setdefault(date_key, [])
            d_centroids = centroids_by_date.setdefault(date_key, [])

            assigned = False
            for gi, c in enumerate(d_centroids):
                if np.linalg.norm(pos - c) <= threshold:
                    d_groups[gi].append(bag_name)
                    n = len(d_groups[gi])
                    d_centroids[gi] = c + (pos - c) / n  # running mean
                    assigned = True
                    break

            if not assigned:
                d_groups.append([bag_name])
                d_centroids.append(pos.copy())

        # Optional cap on bags per group (per date)
        if cap > 0:
            for date_key, d_groups in groups_by_date.items():
                for gi in range(len(d_groups)):
                    if len(d_groups[gi]) > cap:
                        d_groups[gi] = random.sample(d_groups[gi], cap)

        # Flatten per-date groups into global group list
        for date_key in sorted(groups_by_date.keys()):
            d_groups = groups_by_date[date_key]
            d_centroids = centroids_by_date[date_key]
            for gi, members in enumerate(d_groups):
                g_idx = len(groups)
                groups.append(members)
                centroids.append(d_centroids[gi])
                for bn in members:
                    bag2group[bn] = g_idx

        return groups, centroids, bag2group

    def _build_stage_goal_structure(self):
        """
        Build the lookup tables for goal-balanced sampling.

        For each (stage, goal_group) pair, store the list of
        (bag_name, index_array) entries.

        Populates
        ---------
        self.stage_goal_samples     : dict[str, dict[int, list[tuple[str, np.ndarray]]]]
        self.stage_valid_groups     : dict[str, list[int]]
        self.stage_valid_groups_up  : dict[str, list[int]]
        self.stage_valid_groups_base: dict[str, list[int]]
        """
        stage_key_map = {
            "start":      "start_idx",
            "far":        "far_idx",
            "near":       "near_idx",
            "very_close": "very_close_idx",
            "finish":     "finish_idx",
            "recovery":   "recoveries_idx",
        }

        # stage → group_id → [(bag_name, indices), ...]
        sgs: dict[str, dict[int, list]] = {s: {} for s in stage_key_map}

        for bag_name in self.bag_names:
            gi = self.bag_to_goal_group.get(bag_name)
            if gi is None:
                continue
            bag = self.data[bag_name]
            for stage, key in stage_key_map.items():
                indices = bag.get(key, np.array([]))
                if len(indices) > 0:
                    sgs[stage].setdefault(gi, []).append((bag_name, indices))

        # All group IDs per stage
        svg: dict[str, list[int]] = {}
        for stage in stage_key_map:
            svg[stage] = sorted(sgs[stage].keys())

        self.stage_goal_samples = sgs           # {stage: {group_id: [(bag_name, idx_array), ...]}}
        self.stage_valid_groups = svg           # {stage: [group_id, ...]}

        # ── Multi-view vs single-view group lists per stage ──
        self.stage_valid_groups_up = {}         # {stage: [group_id, ...]}
        self.stage_valid_groups_base = {}       # {stage: [group_id, ...]}

        for stage, group_dict in sgs.items():
            up_groups = []
            base_groups = []
            for gi, entries in group_dict.items():
                bag_names = [bn for (bn, _) in entries]
                has_up   = any(bn in self.bags_with_up     for bn in bag_names)
                has_base = any(bn in self.bags_without_up for bn in bag_names)
                if has_up:
                    up_groups.append(gi)
                if has_base:
                    base_groups.append(gi)
            self.stage_valid_groups_up[stage]   = sorted(up_groups)
            self.stage_valid_groups_base[stage] = sorted(base_groups)

        # ── Summary for all goal groups (existing) ──
        n_groups = len(self.goal_groups)
        sizes = [len(g) for g in self.goal_groups]
        if self.log:
            print(f"\n[goal-groups] {n_groups} goal state groups from "
                f"{sum(sizes)} bags  (threshold={self.config.goal_group_threshold_m*100:.1f}cm)")
            print(f"[goal-groups] Group sizes: "
                f"min={min(sizes)}, max={max(sizes)}, "
                f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.0f}")

        unique, counts = np.unique(sizes, return_counts=True)
        top = min(10, len(unique))
        if self.log:
            print(f"[goal-groups] Size distribution (top {top}): "
              + ", ".join(f"{s}×{c}" for s, c in
                          sorted(zip(unique, counts), key=lambda x: -x[1])[:top]))

        for stage in stage_key_map:
            n_g = len(svg[stage])
            n_samples = sum(
                sum(len(idx) for _, idx in entries)
                for entries in sgs[stage].values()
            )
            if self.log:
                print(f"[goal-groups]   {stage:<12s}: {n_g:>4d} groups, "
                  f"{n_samples:>7d} samples")
        if self.log:
            print()

        # ── NEW: per-stage up/base goal stats ──
        if self.log:
            print("\n[up/base goal-groups per stage]")
        for stage in stage_key_map:
            g_up   = self.stage_valid_groups_up.get(stage, [])
            g_base = self.stage_valid_groups_base.get(stage, [])

            # count samples belonging to up/base groups
            n_samples_up = sum(
                sum(len(idx) for _, idx in sgs[stage][gi])
                for gi in g_up
            )
            n_samples_base = sum(
                sum(len(idx) for _, idx in sgs[stage][gi])
                for gi in g_base
            )

            if self.log:
                print(
                    f"  stage={stage:<10s} | "
                    f"up-groups={len(g_up):4d}, up-samples={n_samples_up:7d} | "
                    f"base-groups={len(g_base):4d}, base-samples={n_samples_base:7d}"
                )
        if self.log:
            print()


    def quat_geodesic_angle_xyzw_np(self, q, r, eps=1e-9):
        # q, r: (...,4) xyzw
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        r = r / np.linalg.norm(r, axis=-1, keepdims=True)
        dot = np.clip(np.abs(np.sum(q * r, axis=-1)), -1.0, 1.0)
        return 2.0 * np.arccos(dot)  # radians in [0, pi]

    def select_indices_progress_with_rot_safe(
        self,
        tcp_horizon,            # nparray of N, 7 wiht of (pos(3,), quat(4, xyzw))
        ref_xyz, ref_quat_xyzw, # (3,), (4,)
        pred_horizon,           # int
        max_speed_threshold,    # meters (your computed cap from pos_diffs)
        rot_equiv_radius_m=0.05,# meters per rad (tool/TCP lever arm)
        rot_weight=1.0,         # extra weight on rotation term
        search_radius=10        # local refinement window
    ):
        # Stack arrays
        P = np.array([p[:3] for p in tcp_horizon], dtype=np.float32)   # (N,3)
        Q = np.array([q[3:] for q in tcp_horizon], dtype=np.float32)   # (N,4)
        N = P.shape[0]
        if N == 0:
            return [0] * pred_horizon

        ref_quat = np.broadcast_to(ref_quat_xyzw, (N, 4))

        # Offsets to reference
        pos_diffs = np.linalg.norm(P - ref_xyz[None, :], axis=1)      # (N,)
        rot_rad   = self.quat_geodesic_angle_xyzw_np(Q, ref_quat)          # (N,)
        rot_m     = rot_equiv_radius_m * rot_rad                      # meters

        # Valid indices under speed cap
        valid = pos_diffs <= (max_speed_threshold + 1e-9)
        if not np.any(valid):
            # If nothing passes the cap, relax by allowing all but still follow sels
            valid = np.ones_like(valid, dtype=bool)

        # Targets (your exact schedule)
        sels = (np.arange(1, pred_horizon + 1) / pred_horizon) * max_speed_threshold

        indices = []
        last = 0
        for s in sels:
            # Candidates forward from 'last' that are valid
            J = np.where(valid & (np.arange(N) >= last))[0]
            if len(J) == 0:
                # No forward valid candidates: pad with 'last'
                indices.append(last)
                continue

            # First pick closest by position to keep the speed schedule
            j0 = J[int(np.argmin(np.abs(pos_diffs[J] - s)))]

            # Refine around j0 (clamped forward to 'last')
            w_start = max(j0 - search_radius, last)
            w_end   = min(j0 + search_radius + 1, N)
            W = np.arange(w_start, w_end)
            W = W[valid[W]]  # still respect speed cap
            if len(W) == 0:
                # If window is empty, fall back to j0 or last
                j_best = max(j0, last)
            else:
                # Combined cost (meters): |pos_diffs - s| + rot_weight * rot_m
                comb_cost = np.abs(pos_diffs[W] - s) + rot_weight * rot_m[W]
                j_best = int(W[int(np.argmin(comb_cost))])

            indices.append(j_best)
            last = j_best  # monotonic non-decreasing

        # Ensure exactly pred_horizon length (pad if anything went wrong)
        while len(indices) < pred_horizon:
            indices.append(indices[-1] if indices else 0)

        return indices

    def __getitem__(self, idx):
        """Retry wrapper — on failure, pick a random new sample."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return self._getitem_impl(idx)
            except Exception as e:
                raise e
                if attempt < max_retries - 1:
                    #idx = np.random.randint(0, max(1, self._length))
                    if attempt > 0:  # only print on 2nd+ retry to reduce noise
                        print(f'\n[Dataset] __getitem__ retry {attempt+1}: {type(e).__name__}')
                else:
                    raise e

    def _getitem_impl(self, idx):

        if self.train:
            # ─── Choose which progress stage to sample from ───
            r = np.random.rand()
            if r < self.near_sampling_p and self._n_nears > 0:
                stage = "near"
            elif r < self.very_close_sampling_p and self._n_very_close > 0:
                stage = "very_close"
            elif r < self.finish_sampling_p and self._n_finishes > 0:
                stage = "finish"
            elif r < self.recovery_sampling_p and self._n_recoveries > 0:
                stage = "recovery"
            elif r < self.start_sampling_p and self._n_starts > 0:
                stage = "start"
            else:
                stage = "far"

            p_up = self.config.multi_view_sampling_p
            use_up = (np.random.rand() < p_up)
            if use_up:
                stage_valid_groups = self.stage_valid_groups_up
            else:
                stage_valid_groups = self.stage_valid_groups_base


            if self.config.use_goal_balanced_sampling:
                # ─── Goal-balanced: stage → goal group → bag → sample ───
                valid_groups = stage_valid_groups.get(stage, [])
                if not valid_groups:
                    # Fallback: try 'far' which should always have groups
                    stage = "far"
                    valid_groups = stage_valid_groups.get("far", [])

                if valid_groups:
                    # 1) Uniform over goal groups
                    group_id = random.choice(valid_groups)
                    candidates = self.stage_goal_samples[stage][group_id]  # list[(bag_name, indices)]
                    bag_name, sel_indexes = random.choice(candidates)
                else:
                    # Ultimate fallback: sample from "far" pool with up‑view balancing
                    pool = self.stage_map.get("far", self.has_fars)
                    if not pool:
                        pool = self.has_fars

                    if use_up:
                        pool_f = [e for e in pool if e[1] in self.bags_with_up]
                        if not pool_f:
                            pool_f = pool
                    else:
                        pool_f = [e for e in pool if e[1] in self.bags_without_up]
                        if not pool_f:
                            pool_f = pool

                    _, bag_name, sel_indexes = random.choice(pool_f)

            else:
                # ─── No goal balancing: stage → bag → sample ───
                pool = self.stage_map.get(stage, self.has_fars)
                if not pool:
                    pool = self.has_fars

                if use_up:
                    pool_f = [e for e in pool if e[1] in self.bags_with_up]
                    if not pool_f:
                        pool_f = pool
                else:
                    pool_f = [e for e in pool if e[1] in self.bags_without_up]
                    if not pool_f:
                        pool_f = pool

                _, bag_name, sel_indexes = random.choice(pool_f)

            # ─── Choose sample index within the bag ───
            counter, train_samples = self._indexes_dict[bag_name]
            idx = int(np.random.choice(sel_indexes, size=(1,)))

        else:
            # ─── Eval: sequential iteration (UNCHANGED) ───
            if idx >= self._length:
                idx = idx % self._length
            for (counter, train_samples, bag_name) in self._indexes:
                if idx < counter:
                    break
            idx = train_samples - (counter - idx)

        # the idx is the index of the sample we want to load. each sample starts with the obs horizon and then continues with the pred_horizon.
        # thereby our idx+self.config.obs_horizon-1 is our current robot pose (aka reference pose)

        
        bag_dir = os.path.join(self.unpack_path, bag_name)        
        bag_info = self.data[bag_name]
        bag_id = self.bag_names.index(bag_name)

        rel_schnapp_pos = bag_info['rel_schnapp_pos'][idx+self.config.obs_horizon-1]
        rel_schnapp_rot = bag_info['rel_schnapp_rot'][idx+self.config.obs_horizon-1]
    
        # compute the state at which we are in the trajectory

        start_index = bag_info.get('start_idx', [])
        if idx in start_index:
            progress = 0
        elif rel_schnapp_pos < self.config.finish_threshold_pos and rel_schnapp_rot < self.config.finish_threshold_deg:
            progress = 4 #finished
        elif rel_schnapp_pos < self.config.very_close_threshold_pos and rel_schnapp_rot < self.config.very_close_threshold_deg:
            progress = 3 # very close
        elif rel_schnapp_pos < self.config.near_threshold_pos and rel_schnapp_rot < self.config.near_threshold_deg:
            progress = 2 # close
        else:
            progress = 1 # far away
            
        # collect the inputs (observations)
        # we sample the past observations until the current robot pose (ref pose is the last in the obs_indexes)
        if self.train and self.config.extend_train_obs_sampling > 0:
            obs_indexs = [self.config.obs_horizon-1]
            for _ in range(self.config.obs_horizon-1):
                obs_indexs.append(int(random.randint(-self.config.extend_train_obs_sampling, self.config.obs_horizon-1)))
            obs_indexs = sorted(obs_indexs)
        else:
            obs_indexs = list(range(self.config.obs_horizon))

        images_base = []
        images_up = []
        depths_base = []
        depths_up = []

        img_up_times = []
        img_base_times = []
        depth_up_times = []
        depth_base_times = []        
        agent_traj = []
        robot_state = None
        has_real_up = False
        nr_last = -1000

        # loading images of the objs
        for i in obs_indexs:
            nr_ = idx + i
            if nr_ < 0:
                nr_ = 0
            nr = str(nr_).zfill(6)

            # robot state
            if robot_state is None or nr_ != nr_last:                        
                unpack_npy_path = random.sample(self.unpack_npy_paths, 1)[0]                        
                bag_dir_npy = os.path.join(unpack_npy_path, bag_name)
                #print(f'loading robot state {nr} from {bag_dir_npy}')
                robot_state = np.load(
                    os.path.join(bag_dir_npy, f"{nr}_robot_state.npy"), allow_pickle=True
                ).item()
                nr_last = nr_

            agent_traj.append(np.array(robot_state['robot_pose'], dtype=np.float32))

            # ─── Paths ───
            rgb_base_path = os.path.join(bag_dir, f"{nr}_rgb.png")
            depth_base_path = os.path.join(bag_dir, f"{nr}_depth.png")
            rgb_up_path = os.path.join(bag_dir, f"{nr}_rgb_up.png")
            depth_up_path = os.path.join(bag_dir, f"{nr}_depth_up.png")

            # Completed depth paths (for augmentation)
            completed_base_path = os.path.join(bag_dir, f"{nr}_completed.png")
            completed_up_path = os.path.join(bag_dir, f"{nr}_completed_up.png")
            if not os.path.exists(completed_base_path):
                completed_base_path = None
            if not os.path.exists(completed_up_path):
                completed_up_path = None

            # ─── Load base view (via unified loader) ───
            base_sample = self.image_loader(
                rgb_path=rgb_base_path,
                depth_path=depth_base_path if self.config.with_depth else None,
                completed_path=completed_base_path if self.train and self.config.with_depth and self.config.use_depth_completed else None,
                debug=self.config.debug_dataloader,
            )

            img_base_times.append(robot_state['img_ts'] * 1e-6)
            if self.config.with_depth:
                depth_base_times.append(robot_state['depth_ts'] * 1e-6)

            # ─── Load up view (via unified loader, if exists) ───
            have_up = os.path.exists(rgb_up_path) and (
                not self.config.with_depth or os.path.exists(depth_up_path)
            )

            if have_up:
                up_sample = self.image_loader(
                    rgb_path=rgb_up_path,
                    depth_path=depth_up_path if self.config.with_depth else None,
                    completed_path=completed_up_path if self.train and self.config.with_depth and self.config.use_depth_completed else None,
                    debug=self.config.debug_dataloader,
                )
                has_real_up = True
                img_up_times.append(robot_state['img_ts_up'] * 1e-6)
                if self.config.with_depth:
                    depth_up_times.append(robot_state['depth_ts_up'] * 1e-6)
            else:
                up_sample = None

            # ─── Augmentations (train-time) ───
            if self.train:
                if base_sample is not None:
                    base_sample = self.augmentation_transforms(base_sample)
                if have_up and up_sample is not None:
                    up_sample = self.augmentation_transforms(up_sample)

            # ─── Resize (crop already done by loader) ───
            base_sample = self.preprocesses(base_sample)
            if have_up and up_sample is not None:
                up_sample = self.preprocesses(up_sample)
            else:
                # Build empty up_sample matching base view size
                w, h = base_sample['image'].size
                empty_img = Image.new('RGB', (w, h), color=(0, 0, 0))
                up_sample = {'image': empty_img}
                if self.config.with_depth:
                    up_sample['depth'] = Image.fromarray(
                        np.zeros((h, w), dtype=np.uint16)
                    )

            # ─── Convert to numpy ───
            images_base.append(np.array(base_sample['image'], dtype=np.float32))
            images_up.append(np.array(up_sample['image'], dtype=np.float32))

            if self.config.with_depth:
                depth_base = np.array(base_sample['depth'], dtype=np.float32) / 1000.0
                depth_base[depth_base > self.config.depth_cut_treshold] = 0
                depths_base.append(np.expand_dims(depth_base, axis=-1))

                depth_up = np.array(up_sample['depth'], dtype=np.float32) / 1000.0
                depth_up[depth_up > self.config.depth_cut_treshold] = 0
                depths_up.append(np.expand_dims(depth_up, axis=-1))

        
        images_base = np.stack(images_base, axis=0)          # (T,H,W,3)
        images_up   = np.stack(images_up, axis=0)            # (T,H,W,3)
        if self.config.with_depth:
            depths_base = np.stack(depths_base, axis=0)      # (T,H,W,1)
            depths_up   = np.stack(depths_up, axis=0)        # (T,H,W,1)

        if self.train and self.sensor_dropout_bg is not None:
            # A bag has a real up view if any obs frame has an actual *_rgb_up.png file
            
            # 1) No real up view: always synthesize the up view (zeros or background)
            if not has_real_up:
                images_up, depths_up = self.sensor_dropout_bg(
                    images_up, depths_up, depths_base, view_name='up', debug=self.config.debug_dataloader
                )
            else:
                # 2) Real two-view bag: with 25% probability, kill one view (base or up)
                if np.random.rand() < self.config.drop_sensor_p:
                    if np.random.rand() < 0.75:
                        # drop / perturb base
                        images_base, depths_base = self.sensor_dropout_bg(
                            images_base, depths_base, depths_base, view_name='base', debug=self.config.debug_dataloader
                        )
                    else:
                        # drop / perturb up
                        images_up, depths_up = self.sensor_dropout_bg(
                            images_up, depths_up, depths_base, view_name='up', debug=self.config.debug_dataloader
                        )

        
        # robot sate keys: joints, robot_pose, gripper_width, g_open 
        ref_pose = np.array(robot_state['robot_pose'], dtype=np.float32) # ref pose here is the agent pos at index idx.
        ref_xyz = ref_pose[:3]
        ref_quat = ref_pose[3:]  
        ref_time = robot_state['img_ts'] * 1e-6 # ns -> ms

        #print('ref time', ref_time)
        #print('img_base_times', img_base_times)
        #print('depth_base_times', depth_base_times)
        #print('img_up_times', img_up_times)
        #print('depth_up_times', depth_up_times)
        
        # compute the relative action based on the current aka ref pose 
        # relative poses for each obs step
        rel_agent_poses = []
        for i, s in enumerate(agent_traj):
            rel6d = tcp_pose_to_relative(
                pos=s[:3], quat_xyzw=s[3:], ref_pos=ref_xyz, ref_quat_xyzw=ref_quat
            )

            time_feats = [ref_time - img_base_times[i]]
            if self.config.with_depth:
                time_feats.append(ref_time - depth_base_times[i])

            if has_real_up:
                time_feats.append(ref_time - img_up_times[i])
                if self.config.with_depth:
                    time_feats.append(ref_time - depth_up_times[i])
            else:
                time_feats.append(ref_time - img_base_times[i] + 2/32 * (np.random.rand() - 0.5))
                if self.config.with_depth:
                    time_feats.append(ref_time - depth_base_times[i] + 2/32 * (np.random.rand() - 0.5))
            
            #print('time_feats', time_feats)
            # clip to [0, cam_time_max_ms]
            time_feats = np.array(time_feats, dtype=np.float32) 
            rel_agent_poses.append(np.concatenate([rel6d, time_feats], axis=0))
        rel_agent_poses = np.stack(rel_agent_poses, axis=0).astype(np.float32)
       
        # collect the target (predidcition horizon)
        # get the future horizon from the current ref pose
        tcp_horion = robot_state['tcp_horion']        
            
        if self.config.time_dependent_sampling:
            l_horizon = len(tcp_horion)
            if l_horizon < self.config.pred_horizon:
                sel_idx = list(range(l_horizon))
                for lmiss in range(self.config.pred_horizon-l_horizon):
                    sel_idx.append(l_horizon-1)
            else:
                # we sample with a constant hz from the tcp horizon
                step_size = l_horizon // self.config.pred_horizon
                sel_idx = list(range(step_size-1, l_horizon, step_size))            
                if len(sel_idx) != self.config.pred_horizon:
                    step_size = l_horizon // (self.config.pred_horizon+1)
                    sel_idx = list(range(step_size-1, l_horizon, step_size))[:self.config.pred_horizon]      
                
            #print('robot_state', robot_state.keys() )
            #deltas = []
            #joint_ts = []
            #for si in sel_idx:
            #    ts = robot_state['joint_ts'][si]
            #    if joint_ts:
            #        deltas.append((ts-joint_ts[-1])/1e6)                    
            #    joint_ts.append(ts)
            #print('deltas', np.mean(deltas), np.std(deltas), np.min(deltas), np.max(deltas))
            #print('sel_idx', sel_idx)
            #print('time elpsed', (joint_ts[-1]-robot_state['first_joint_ts'])/1e9)

        else:
            # if we want to sample based on distance increments between each pred step we sample based on the max speed (full range in the pred horzion) we want to allow
            P = np.array([p[:3] for p in tcp_horion], dtype=np.float32)
            pos_diffs = np.linalg.norm(P - ref_xyz[None, :], axis=1)
            max_speed_threshold = float(min(pos_diffs.max(), self.config.max_speed_threshold))
            sel_idx = self.select_indices_progress_with_rot_safe(
                tcp_horion,
                ref_xyz, ref_quat,
                pred_horizon=self.config.pred_horizon,
                max_speed_threshold=max_speed_threshold,
                rot_equiv_radius_m=getattr(self.config, 'rot_equiv_radius_m', 0.05),
                rot_weight=getattr(self.config, 'rot_weight', 1.0),
                search_radius=getattr(self.config, 'rot_search_radius', 10)
            )

        # sample the action horzion (future steps) from the selected indexes
        action_traj = [tcp_horion[i] for i in sel_idx]  # guaranteed len == pred_horizon
        origin = np.concatenate([ref_xyz, ref_quat], axis=0).astype(np.float32)
        action_traj = np.array([origin] + action_traj)    

        # we remove the first index which is the origin and only used in the agent pose traj and not in the action traj
        rel_actions = []   
        for a_pose in action_traj[1:]:
            rel6d = tcp_pose_to_relative(pos=a_pose[:3], quat_xyzw=a_pose[3:], ref_pos=ref_xyz, ref_quat_xyzw=ref_quat) 
            s_pos, s_rot = self.get_offset(bag_info['schnapp_pose'], a_pose)
            #print('s_pos, s_rot', s_pos, s_rot)
            s_pos = min(s_pos, self.config.dist_2_schnapp_treshold) / self.config.dist_2_schnapp_treshold
            s_rot = min(s_rot, self.config.deg_2_schnapp_treshold) / self.config.deg_2_schnapp_treshold
            #print('after', s_pos, s_rot)
            rel_actions.append(
                np.concatenate([rel6d, np.array([s_pos, s_rot])], axis=0).flatten()
            )

        rel_actions = np.stack(rel_actions, axis=0).astype(np.float32)
   
               
        if self.config.debug_dataloader:
            print('saveing')
            self.save_images(images_base, depths_base, images_up, depths_up, bag_id, bag_name, idx)

            self.save_traj(rel_agent_poses, rel_actions, ref_pose, rel_schnapp_pos, rel_schnapp_rot, obs_indexs, bag_id, bag_name, idx, progress, has_real_up)
            print('progress', progress, rel_actions[0,6], rel_actions[0,7], rel_schnapp_pos, rel_schnapp_rot, float(np.linalg.norm(rel_actions[0,:3])), float(np.linalg.norm(rel_actions[-1,:3])))
                    
        #rel_actions = normalize_data(rel_actions, self.stats['action']) # we normalize in the training forward
        rel_agent_poses = normalize_data(rel_agent_poses, self.stats['agent_pose']) # we encode the normalized

        sample = {
            'image':       images_base,         # (T,H,W,3)
            'image_up':    images_up,           # (T,H,W,3)
            'agent_pose':  rel_agent_poses,     # (T,10)
            'action':      rel_actions,         # (pred_horizon, 8)
            'ref_poses':   ref_pose,
            'progress':    np.array([progress]),
            'rel_schnapp_pos': rel_schnapp_pos,
            'rel_schnapp_rot': rel_schnapp_rot,
            'bag_id':      np.array([bag_id]),
            'bag_name':    bag_name
        }

        if self.config.with_depth:
            sample['depth']    = depths_base      # (T,H,W,1)
            sample['depth_up'] = depths_up        # (T,H,W,1)
        elif self.config.with_depth:
            print('no depth')
            raise ValueError('no depth')
                        
             
        if self.config.debug_dataloader:
            print('saved and ready for review, enter to continue:')
            time.sleep(2)

        return sample
    
    
    def tcp_to_flange(self, tcp_pos, tcp_quat):
        # flange_T_tcp: pure translation along local +Z by tcp_z (same as in add_tcp)
        flange_T_tcp = np.eye(4, dtype=np.float64)
        flange_T_tcp[:3, 3] = np.array([self.config.tcp_offset_x, self.config.tcp_offset_y, self.config.tcp_offset_z], dtype=np.float64)

        # base_T_tcp from input pose
        base_T_tcp = self.pose_to_hmat(tcp_pos, tcp_quat)

        # tcp_T_flange is the inverse of flange_T_tcp
        tcp_T_flange = self.hmat_inv(flange_T_tcp)

        # base_T_flange = base_T_tcp · tcp_T_flange
        base_T_flange = base_T_tcp @ tcp_T_flange

        pos_fl = base_T_flange[:3, 3].astype(np.float32)
        quat_fl = R.from_matrix(base_T_flange[:3, :3]).as_quat().astype(np.float32)
        return pos_fl, quat_fl
    
    
    def add_tcp(self, pos, rot):
        
        flange2tcp = np.eye(4, dtype=np.float32)
        flange2tcp[:3, 3] = np.array([self.config.tcp_offset_x, self.config.tcp_offset_y, self.config.tcp_offset_z], dtype=np.float32)

        #print('flange2tcp', flange2tcp)
        
        robot2flange = np.eye(4, dtype=np.float32)
        robot2flange[:3, :3] = R.from_quat(np.array(rot)).as_matrix()
        robot2flange[:3, 3] = np.array(pos)

        #print('robot2flange', robot2flange)

        tcp = robot2flange @ flange2tcp #.T
        #print('tcp', tcp)

        pos = tcp[:3, 3]

        rot = R.from_matrix(tcp[:3, :3]).as_quat()
                        
        return pos, rot
    

    def to_float_vec(self, vec):
        try:
            return [float(v) for v in vec]
        except:
            return [[float(v) for v in vec_] for vec_ in vec]
        
    def save_traj(self, agent_pose, action, ref_pose, rel_schnapp_pos, rel_schnapp_rot, obs_indexs, bag_id, bag_name, idx, progress, has_real_up):
        
        action = self.to_float_vec(action)
        agent_pose = self.to_float_vec(agent_pose)
        
        action_norm = self.to_float_vec(normalize_data(action, self.stats['action']))
        agent_pose_norm = self.to_float_vec(normalize_data(agent_pose, self.stats['agent_pose']))

        ref_pose = self.to_float_vec(ref_pose)
        rel_schnapp_pos = float(rel_schnapp_pos)
        rel_schnapp_rot = float(rel_schnapp_rot)        
                
        out_path = '/data/debug'
        out_traj = os.path.join(out_path, 'traj.json')
        with open(out_traj, 'w') as f:
            json.dump({
                'agent_pose': agent_pose,
                'agent_pose_norm': agent_pose_norm,
                'action': action,
                'action_norm': action_norm,
                'ref_pose': ref_pose,
                'rel_schnapp_pos': rel_schnapp_pos,
                'rel_schnapp_rot': rel_schnapp_rot,
                'obs_indexs': obs_indexs,
                'has_real_up': has_real_up,
                'progress': self.progress_names.get(progress, 'None')
            }, f)            



    def save_images(self, images_base, depths_base, images_up, depths_up, bag_id, bag_name, idx):
        out_path = '/data/debug'
        print('out_path', out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            print('make out dir', out_path)
        else:
            # clean outpath
            print('clear old')
            for f in os.listdir(out_path):
                os.remove(os.path.join(out_path, f))

        def _save_one_view(images, depths, view_suffix):
            print('save imgs', len(images), len(depths))
            for i, (im, d) in enumerate(zip(images, depths)):
                img = np.array(im, dtype=np.uint8)
                depth = np.array(d[:, :, 0] * 1000, dtype=np.uint16)  # m → mm
                img = Image.fromarray(img)
                depth = Image.fromarray(depth)

                vs = f"_{view_suffix}" if view_suffix else ""
                out_rgb = os.path.join(
                    out_path,
                    f"b{vs}_obs{i:02d}_rgb.png"
                )
                out_depth = os.path.join(
                    out_path,
                    f"b{vs}_obs{i:02d}_depth.png"
                )
                print('save debug rgb', out_rgb)
                print('save debug depth', out_depth)
                img.save(out_rgb)
                depth.save(out_depth)

        _save_one_view(images_base, depths_base, 'base')
        _save_one_view(images_up,   depths_up,   'up')
        
    def get_indices_for_bags(self, bag_ids):
        indices = []
        for bid in bag_ids:
            counter, train_samples, key = self._indexes[bid]
            indices.extend(list(range(counter-train_samples, counter)))
        return indices
    

    def get_num_bags(self):
        return len(self.data)
    
    def set_train(self):
        print('setting dataset to train')
        self.train = True

    def set_eval(self):
        print('setting dataset to eval')
        self.train = False


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    from dataset.helper_fnc import numpy_collate, worker_init_fn
    worker_init_fn(None)

    config = load_yaml_as_dataclass('config/config.yaml', Config)
    config.debug_dataloader = False
    config.overwrite_data = False
    config.overwrite_image_data = False
    config.use_whitelist = False

    time_crit = {
        'fixbag': None, #'bag_20260519_104239',
        'mindate': None, #20260400,
        'mindate_time': None, # 152012,
        'maxdate': None, #20260521,
        'maxdate_time': None, #83752
    }

    ds = ROSBagDataset(config, train=True, time_crit=time_crit)

    N = len(ds)
    idx = list(range(N))
    random.shuffle(idx)

    # Collect per-step and per-progress statistics
    progress_names = {0: 'start', 1: 'far', 2: 'near', 3: 'very_close', 4: 'finish'}

    # Action collectors
    last_actions_by_progress = {0: [], 1: [], 2: [], 3: [], 4: []}
    all_actions = []
    all_last_actions = []
    all_norms_last = []

    # Agent pose collectors
    all_agent_poses = []          # list of (obs_horizon, 6)
    all_last_agent_poses = []     # last obs step only
    last_agent_by_progress = {0: [], 1: [], 2: [], 3: [], 4: []}
    all_agent_norms_last = []

    for i, nr in enumerate(idx):
        try:
            sample = ds[nr]
        except Exception as e:
            print(f'[ERROR] at {i+1}: {e}')
            raise e
            continue

        a = unnormalize_data(sample['action'], ds.stats['action'])          # (pred_horizon, 8)
        ap = unnormalize_data(sample['agent_pose'], ds.stats['agent_pose']) # (obs_horizon, 6)

        progress = int(sample['progress'][0])

        if config.debug_dataloader:
            input('next?')

        # Action
        all_actions.append(a)
        all_last_actions.append(a[-1])
        all_norms_last.append(np.linalg.norm(a[-1, :3]))
        last_actions_by_progress[progress].append(a[-1])

        # Agent pose
        all_agent_poses.append(ap)
        all_last_agent_poses.append(ap[-1])
        all_agent_norms_last.append(np.linalg.norm(ap[-1, :3]))
        last_agent_by_progress[progress].append(ap[-1])

        print(f"\r {i+1}/{N} {nr} collected", end='', flush=True)

        #print('sample', sample.keys())
        #print('a', a[-1][:3]*1000)
        #print('ap', ap[0][:3]*1000)
        #print('bag_name', sample['bag_name'])
        #print('progress', sample['progress'])
        #input()

        #if i > 100000:
        #    break


    print('\n\n========== ACTION RESULTS ==========\n')

    # ─── Global stats for the LAST action step ───
    A_last = np.array(all_last_actions)
    dim_names_action = ['Δx', 'Δy', 'Δz', 'Δrx', 'Δry', 'Δrz', 'dist2snap', 'deg2snap']

    print(f'Total samples: {len(A_last)}')
    print(f'\n--- Last action step (step {config.pred_horizon}) global stats ---')
    print(f'{"dim":>10s} {"mean":>10s} {"std":>10s} {"min":>10s} {"p01":>10s} '
          f'{"p50":>10s} {"p99":>10s} {"max":>10s}')
    for d in range(A_last.shape[1]):
        col = A_last[:, d]
        print(f'{dim_names_action[d]:>10s} {np.mean(col):10.6f} {np.std(col):10.6f} '
              f'{np.min(col):10.6f} {np.percentile(col, 1):10.6f} '
              f'{np.percentile(col, 50):10.6f} {np.percentile(col, 99):10.6f} '
              f'{np.max(col):10.6f}')

    norms = np.array(all_norms_last)
    print(f'\n{"pos_norm":>10s} {np.mean(norms):10.6f} {np.std(norms):10.6f} '
          f'{np.min(norms):10.6f} {np.percentile(norms, 1):10.6f} '
          f'{np.percentile(norms, 50):10.6f} {np.percentile(norms, 99):10.6f} '
          f'{np.max(norms):10.6f}')

    # ─── Per-progress breakdown (action) ───
    for prog_id in sorted(progress_names.keys()):
        arr = np.array(last_actions_by_progress[prog_id])
        if len(arr) == 0:
            continue
        name = progress_names[prog_id]
        print(f'\n--- Last action step | progress={prog_id} ({name}), n={len(arr)} ---')
        print(f'{"dim":>10s} {"mean":>10s} {"std":>10s} {"min":>10s} '
              f'{"p01":>10s} {"p50":>10s} {"p99":>10s} {"max":>10s}')
        for d in range(arr.shape[1]):
            col = arr[:, d]
            print(f'{dim_names_action[d]:>10s} {np.mean(col):10.6f} {np.std(col):10.6f} '
                  f'{np.min(col):10.6f} {np.percentile(col, 1):10.6f} '
                  f'{np.percentile(col, 50):10.6f} {np.percentile(col, 99):10.6f} '
                  f'{np.max(col):10.6f}')
        pn = np.linalg.norm(arr[:, :3], axis=1)
        print(f'{"pos_norm":>10s} {np.mean(pn):10.6f} {np.std(pn):10.6f} '
              f'{np.min(pn):10.6f} {np.percentile(pn, 1):10.6f} '
              f'{np.percentile(pn, 50):10.6f} {np.percentile(pn, 99):10.6f} '
              f'{np.max(pn):10.6f}')

    # ─── Per-step stats (action, all pred_horizon steps) ───
    A_all = np.array(all_actions)
    print(f'\n--- Per-step action positional norm (mm) ---')
    print(f'{"step":>6s} {"mean":>10s} {"std":>10s} {"p50":>10s} {"p99":>10s} {"max":>10s}')
    for step in range(config.pred_horizon):
        pn = np.linalg.norm(A_all[:, step, :3], axis=1) * 1000
        print(f'{step+1:>6d} {np.mean(pn):10.3f} {np.std(pn):10.3f} '
              f'{np.percentile(pn, 50):10.3f} {np.percentile(pn, 99):10.3f} '
              f'{np.max(pn):10.3f}')

    print('\n\n========== AGENT POSE RESULTS ==========\n')

    # ─── Global stats for the LAST agent pose obs step ───
    AP_last = np.array(all_last_agent_poses)
    dim_names_agent = ['Δx', 'Δy', 'Δz', 'Δrx', 'Δry', 'Δrz']
    n_agent_dims = AP_last.shape[1]

    print(f'Total samples: {len(AP_last)}')
    print(f'\n--- Last agent pose obs (obs step {config.obs_horizon}) global stats ---')
    print(f'{"dim":>10s} {"mean":>10s} {"std":>10s} {"min":>10s} {"p01":>10s} '
          f'{"p50":>10s} {"p99":>10s} {"max":>10s}')
    for d in range(n_agent_dims):
        col = AP_last[:, d]
        label = dim_names_agent[d] if d < len(dim_names_agent) else f'dim{d}'
        print(f'{label:>10s} {np.mean(col):10.6f} {np.std(col):10.6f} '
              f'{np.min(col):10.6f} {np.percentile(col, 1):10.6f} '
              f'{np.percentile(col, 50):10.6f} {np.percentile(col, 99):10.6f} '
              f'{np.max(col):10.6f}')

    ap_norms = np.array(all_agent_norms_last)
    print(f'\n{"pos_norm":>10s} {np.mean(ap_norms):10.6f} {np.std(ap_norms):10.6f} '
          f'{np.min(ap_norms):10.6f} {np.percentile(ap_norms, 1):10.6f} '
          f'{np.percentile(ap_norms, 50):10.6f} {np.percentile(ap_norms, 99):10.6f} '
          f'{np.max(ap_norms):10.6f}')

    # ─── Per-progress breakdown (agent pose) ───
    for prog_id in sorted(progress_names.keys()):
        arr = np.array(last_agent_by_progress[prog_id])
        if len(arr) == 0:
            continue
        name = progress_names[prog_id]
        print(f'\n--- Last agent pose obs | progress={prog_id} ({name}), n={len(arr)} ---')
        print(f'{"dim":>10s} {"mean":>10s} {"std":>10s} {"min":>10s} '
              f'{"p01":>10s} {"p50":>10s} {"p99":>10s} {"max":>10s}')
        for d in range(arr.shape[1]):
            col = arr[:, d]
            label = dim_names_agent[d] if d < len(dim_names_agent) else f'dim{d}'
            print(f'{label:>10s} {np.mean(col):10.6f} {np.std(col):10.6f} '
                  f'{np.min(col):10.6f} {np.percentile(col, 1):10.6f} '
                  f'{np.percentile(col, 50):10.6f} {np.percentile(col, 99):10.6f} '
                  f'{np.max(col):10.6f}')
        pn = np.linalg.norm(arr[:, :3], axis=1)
        print(f'{"pos_norm":>10s} {np.mean(pn):10.6f} {np.std(pn):10.6f} '
              f'{np.min(pn):10.6f} {np.percentile(pn, 1):10.6f} '
              f'{np.percentile(pn, 50):10.6f} {np.percentile(pn, 99):10.6f} '
              f'{np.max(pn):10.6f}')

    # ─── Per-obs-step stats (agent pose) ───
    AP_all = np.array(all_agent_poses)
    print(f'\n--- Per-obs-step agent pose positional norm (mm) ---')
    print(f'{"step":>6s} {"mean":>10s} {"std":>10s} {"p50":>10s} {"p99":>10s} {"max":>10s}')
    n_obs_steps = AP_all.shape[1]
    for step in range(n_obs_steps):
        pn = np.linalg.norm(AP_all[:, step, :3], axis=1) * 1000
        print(f'{step+1:>6d} {np.mean(pn):10.3f} {np.std(pn):10.3f} '
              f'{np.percentile(pn, 50):10.3f} {np.percentile(pn, 99):10.3f} '
              f'{np.max(pn):10.3f}')

    # ─── Per-obs-step rotation norm (agent pose, degrees) ───
    print(f'\n--- Per-obs-step agent pose rotation norm (deg) ---')
    print(f'{"step":>6s} {"mean":>10s} {"std":>10s} {"p50":>10s} {"p99":>10s} {"max":>10s}')
    for step in range(n_obs_steps):
        rn = np.linalg.norm(AP_all[:, step, 3:6], axis=1) * (180.0 / np.pi)
        print(f'{step+1:>6d} {np.mean(rn):10.3f} {np.std(rn):10.3f} '
              f'{np.percentile(rn, 50):10.3f} {np.percentile(rn, 99):10.3f} '
              f'{np.max(rn):10.3f}')

    # ═══════════════════════════════════════════════════════
    # Suggested normalization bounds
    # ═══════════════════════════════════════════════════════
    print('\n\n========== SUGGESTED NORMALIZATION STATS ==========\n')

    # --- Action ---
    A_flat = A_all.reshape(-1, A_all.shape[-1])
    a_p01 = np.percentile(A_flat, 1, axis=0)
    a_p99 = np.percentile(A_flat, 99, axis=0)
    a_p05 = np.percentile(A_flat, 5, axis=0)
    a_p95 = np.percentile(A_flat, 95, axis=0)

    print('--- Action stats (p1/p99) ---')
    print(f"'min': np.array({np.array2string(a_p01, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32)")
    print(f"'max': np.array({np.array2string(a_p99, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32)")

    print('\n--- Action stats (p5/p95) ---')
    print(f"'min': np.array({np.array2string(a_p05, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32)")
    print(f"'max': np.array({np.array2string(a_p95, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32)")

    # --- Agent pose ---
    AP_flat = AP_all.reshape(-1, AP_all.shape[-1])
    ap_p01 = np.percentile(AP_flat, 1, axis=0)
    ap_p99 = np.percentile(AP_flat, 99, axis=0)
    ap_p05 = np.percentile(AP_flat, 5, axis=0)
    ap_p95 = np.percentile(AP_flat, 95, axis=0)

    print('\n--- Agent pose stats (p1/p99) ---')
    print(f"'min': np.array({np.array2string(ap_p01, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32)")
    print(f"'max': np.array({np.array2string(ap_p99, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32)")

    print('\n--- Agent pose stats (p5/p95) ---')
    print(f"'min': np.array({np.array2string(ap_p05, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32)")
    print(f"'max': np.array({np.array2string(ap_p95, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32)")

    # --- Cross-check: are action[:6] and agent_pose sharing the same space? ---
    print('\n--- Cross-check: action[:6] vs agent_pose range comparison ---')
    print(f'{"dim":>10s} {"action p1":>12s} {"action p99":>12s} {"agent p1":>12s} {"agent p99":>12s} {"same range":>12s}')
    for d in range(min(6, n_agent_dims)):
        a1, a99 = a_p01[d], a_p99[d]
        p1, p99 = ap_p01[d], ap_p99[d]
        # Check if ranges overlap substantially
        overlap = min(a99, p99) - max(a1, p1)
        range_a = a99 - a1
        range_p = p99 - p1
        ratio = overlap / max(min(range_a, range_p), 1e-12)
        same = '✓' if ratio > 0.7 else '⚠ differ'
        label = dim_names_agent[d] if d < len(dim_names_agent) else f'dim{d}'
        print(f'{label:>10s} {a1:12.8f} {a99:12.8f} {p1:12.8f} {p99:12.8f} {same:>12s}')

    print('\n--- Recommended config stats block (copy-paste ready) ---')
    print("""
        self.stats = {
            'action': {""")
    print(f"                'min': np.array({np.array2string(a_p01, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32),")
    print(f"                'max': np.array({np.array2string(a_p99, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32)")
    print("""            },
            'agent_pose': {""")
    print(f"                'min': np.array({np.array2string(ap_p01, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32),")
    print(f"                'max': np.array({np.array2string(ap_p99, separator=', ', formatter={'float_kind':lambda x: f'{x:.8f}'})}, dtype=np.float32)")
    print("""            }
        }""")