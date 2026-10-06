"""
PyTorch multi-GPU inference manager for dual-camera diffusion policy.

Drop-in replacement for the JAX MultiGPUManager.

Worker payload (same keys as JAX version):
  images_base: (T, H, W, 3) uint8
  images_up:   (T, H, W, 3) uint8  — zeros if no up-cam
  depths_base: (T, H, W, 1) float32 m or None
  depths_up:   (T, H, W, 1) float32 m or None
  agent_pose:  (T, lowdim_obs_dim) float32  ALREADY normalised [-1,1]
  ts, ref:     metadata

Result (same keys as JAX version, one CRITICAL difference):
  naction: (pred_horizon, act_dim) float32 in PHYSICAL units.
  *** Do NOT call unnormalize_data() on this in the inference node. ***
  JAX workers returned normalised actions; here sample_actions() already
  calls _unnormalize() internally.
"""

import os
import time
import argparse
import warnings
import queue as pyqueue
import multiprocessing as mp
from typing import Optional

import numpy as np
import torch

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Policy reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _build_policy_from_checkpoint(ckpt_path: str, device: torch.device):
    """
    Reload a DiffusionPolicy from a train_policy_ros_bag.py checkpoint.

    Returns
    -------
    policy          : DiffusionPolicy   on *device*, eval mode
    config          : Config | None     (from checkpoint's config_path)
    pred_horizon    : int
    act_dim         : int
    n_views         : int
    obs_horizon     : int
    robot_state_dim : int
    depth_cut       : float             (metres, post-init value)
    with_depth      : bool
    """
    from model.diffusion_policy import DiffusionPolicy
    from datasets.ros_bag_utils import Config, load_yaml_as_dataclass

    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw  = ckpt.get('args', {})
    train_args = argparse.Namespace(**(raw if isinstance(raw, dict) else vars(raw)))

    # ── Load YAML config used at training time ────────────────────────────
    config_path = ckpt.get('config_path',
                            getattr(train_args, 'config_path', None))
    config: Optional[Config] = None
    if config_path and os.path.isfile(config_path):
        try:
            config = load_yaml_as_dataclass(config_path, Config)
            # __post_init__ already overwrites depth_cut_treshold with
            # depth_cut_treshold_pe when use_depth_positional_encoding=True
        except Exception as e:
            print(f'[_build_policy] config load failed ({config_path}): {e}')

    def _cfg(attr: str, default):
        return getattr(config, attr, default) if config is not None else default

    # ── Backbone namespace (mirrors build_backbone_args in train script) ──
    enc_type  = getattr(train_args, 'encoder_type', 'dino_vd_rgbd')
    with_det  = 'det'   in enc_type
    with_obj  = 'obj'   in enc_type
    with_pose = 'grasp' in enc_type

    bb = argparse.Namespace(
        device                 = str(device),
        model_name             = _cfg('encoder_name',  'Dino'),
        model_version          = _cfg('encoder_version', 'V2-small'),
        with_depth             = _cfg('with_depth', True),
        rgb_only               = not _cfg('with_depth', True),
        freeze_encoder         = False,
        freeze_color_encoder   = _cfg('freeze_color_encoder', True),
        max_objects            = getattr(train_args, 'max_objects', 30),
        num_feature_levels     = 4,
        fpn_layers             = 4,
        cat_outs               = True,
        norm_return_layers     = True,
        norm_return_layers_rgb = True,
        resize_attention_maps  = False,
        return_rdps            = False,
        with_intr              = False,
        pretrained_imagenet    = True,
        pretrained_path        = './data/backbones/dinov2_vits14_pretrain.pth',
        aux_loss               = True,
        proj_kernel_size       = 1,
        with_det               = with_det,
        with_obj               = with_obj,
        with_pose              = with_pose,
        with_critique          = False,
        # positional embedding
        position_embedding     = 'sine',
        backbone_proj_and_upsampling = '0p0u0k, 0p0u0k, 0p0u0k, 0p0u0k',
        # projection dims
        hidden_dim             = _cfg('vision_feature_dim', 384),
        hidden_dim_det         = _cfg('vision_feature_dim', 384),
        # depth encoding (handled inside backbone's DepthEncoder)
        with_positional_depth_encoding = _cfg('use_depth_positional_encoding', True),
        depth_channels         = _cfg('depth_pe_channels', 8),
        temperature            = _cfg('depth_pe_temperature', 0.00574),
        max_depth              = _cfg('depth_cut_treshold_pe', 0.5),
        normalize_depth        = False,
        encode_disparity       = False,
    )

    # Extra args needed only when the grasper transformer is used
    if with_det or with_obj or with_pose:
        bb.hidden_dim_obj    = _cfg('vision_feature_dim', 384)
        bb.nheads            = 8
        bb.enc_layers        = 6
        bb.dec_layers        = 6
        bb.dim_feedforward   = 1024
        bb.dropout           = 0.1
        bb.dec_n_points      = 4
        bb.enc_n_points      = 4
        bb.num_det_queries   = 100
        bb.num_obj_queries   = 8
        bb.num_grasp_queries = 3
        bb.num_grasp_poses   = 5
        bb.n_classes         = 1
        bb.add_model_embed               = False
        bb.add_critique_extra_query_embed = False
        bb.add_pose_extra_query_embed     = False

    # ── Policy dims ───────────────────────────────────────────────────────
    n_views         = _cfg('num_views',         2)
    vis_feat        = _cfg('vision_feature_dim', 384)
    token_dim       = vis_feat // n_views          # 192 for V2-small + 2 views
    act_dim         = _cfg('action_dim',         8)
    robot_state_dim = _cfg('lowdim_obs_dim',    10)
    pred_horizon    = _cfg('pred_horizon',      16)
    obs_horizon     = _cfg('obs_horizon',        2)
    n_diffusion     = _cfg('num_diffusion_iters', 100)
    n_eval          = _cfg('eval_ddim_steps',    10)
    img_kernel      = _cfg('joiner_feature_map_kernel', 4)

    action_min = getattr(train_args, 'action_min',
                         [-0.01, -0.01, 0.0, -0.03, -0.03, -0.15, 0.0, 0.0])
    action_max = getattr(train_args, 'action_max',
                         [0.01, 0.01, 0.032, 0.03, 0.03, 0.15, 0.5, 0.5])

    policy = DiffusionPolicy(
        args_backbone         = bb,
        encoder_type          = enc_type,
        diffusion_max_objs    = min(5, bb.max_objects),
        token_dim             = token_dim,
        act_dim               = act_dim,
        robot_state_dim       = robot_state_dim,
        horizon               = pred_horizon,
        obs_horizon           = obs_horizon,
        n_views               = n_views,
        model_dim             = token_dim,
        n_layers              = getattr(train_args, 'tf_layers', 8),
        n_heads               = getattr(train_args, 'tf_heads',  8),
        diffusion_steps_train = n_diffusion,
        diffusion_steps_eval  = n_eval,
        image_kernel          = img_kernel,
        action_min            = action_min,
        action_max            = action_max,
    )

    state_dict = ckpt.get('eval_policy', ckpt.get('model', {}))
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    print(f'[_build_policy] missing={len(missing)}  unexpected={len(unexpected)}')
    if missing:
        print(f'  first 5 missing: {missing[:5]}')
    policy.eval().to(device)

    depth_cut  = _cfg('depth_cut_treshold', 0.5)   # post-init value
    with_depth = _cfg('with_depth', True)

    return (policy, config, pred_horizon, act_dim,
            n_views, obs_horizon, robot_state_dim,
            depth_cut, with_depth)


# ─────────────────────────────────────────────────────────────────────────────
#  Worker process
# ─────────────────────────────────────────────────────────────────────────────

def inference_worker_pt(
    gpu_id: int,
    ckpt_path: str,
    result_q,
    task_q,
    log: bool = False,
):
    """
    Subprocess: loads PyTorch DiffusionPolicy, runs DDIM sampling forever.

    Image preprocessing matches rosbag_collate exactly:
      • ImageNet-normalise RGB
      • Interleave base/up: [base₀, up₀, base₁, up₁, ...]
      • repeat_interleave(2) on agent_pose to match
    """
    # ── Device setup ─────────────────────────────────────────────────────
    if gpu_id is None or gpu_id < 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device('cuda:0')

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='Couldn\'t find sharding')
    torch.set_num_threads(4)

    if log:
        print(f'[Worker {gpu_id}] starting on {device}')

    # ── Load model ────────────────────────────────────────────────────────
    (policy, config, pred_horizon, act_dim,
     n_views, obs_horizon, robot_state_dim,
     depth_cut, with_depth) = _build_policy_from_checkpoint(ckpt_path, device)

    if log:
        print(f'[Worker {gpu_id}] policy ready  '
              f'pred_h={pred_horizon}  act_dim={act_dim}  '
              f'n_views={n_views}  obs_h={obs_horizon}  '
              f'with_depth={with_depth}  depth_cut={depth_cut}')

    # ── Warmup forward pass ───────────────────────────────────────────────
    fwd_h = getattr(config, 'image_forward_shape', [448, 448])[0] if config else 448
    fwd_w = getattr(config, 'image_forward_shape', [448, 448])[1] if config else 448
    T_mdl = obs_horizon * n_views
    try:
        dummy_rgb   = torch.zeros(1, T_mdl, 3, fwd_h, fwd_w,   device=device)
        dummy_dep   = torch.zeros(1, T_mdl, 1, fwd_h, fwd_w,   device=device) if with_depth else None
        dummy_state = torch.zeros(1, T_mdl, robot_state_dim,    device=device)
        with torch.no_grad():
            _ = policy.sample_actions(dummy_rgb, dummy_dep, dummy_state,
                                      horizon=pred_horizon)
        if log:
            print(f'[Worker {gpu_id}] warmup OK')
    except Exception as e:
        print(f'[Worker {gpu_id}] warmup failed: {e}')

    # ── Preprocessing helpers ─────────────────────────────────────────────
    _mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    _std  = torch.tensor(_IMAGENET_STD,  device=device).view(1, 3, 1, 1)

    def _prep_rgb(arr: np.ndarray) -> torch.Tensor:
        """(T,H,W,3) uint8 → (T,3,H,W) ImageNet-normalised on device."""
        t = torch.from_numpy(arr).float() / 255.0   # (T,H,W,3)
        t = t.permute(0, 3, 1, 2).to(device)         # (T,3,H,W)
        return (t - _mean) / _std

    def _prep_depth(arr: Optional[np.ndarray],
                    T: int, H: int, W: int) -> torch.Tensor:
        """(T,H,W,[1]) float32 m → (T,1,H,W) thresholded on device."""
        if arr is None:
            return torch.zeros(T, 1, H, W, device=device)
        a = arr.copy()
        if a.ndim == 3:                     # (T,H,W) → (T,H,W,1)
            a = a[:, :, :, np.newaxis]
        a[a > depth_cut] = 0.0
        return torch.from_numpy(a).float().permute(0, 3, 1, 2).to(device)

    def _interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """[base₀,…,baseT] + [up₀,…,upT] → [base₀,up₀,…,baseT,upT]."""
        T = a.shape[0]
        return torch.stack([a, b], dim=1).reshape(T * 2, *a.shape[1:])

    run_first = True

    try:
        while True:
            item = task_q.get()
            if item is None:
                break

            t0 = time.perf_counter()

            images_base = item['images_base']                           # (T,H,W,3) uint8
            images_up   = item.get('images_up', np.zeros_like(images_base))
            depths_base = item.get('depths_base')                       # (T,H,W,1) m or None
            depths_up   = item.get('depths_up')
            agent_pos_t = item['agent_pose']                            # (T,D) normalised
            ts          = item.get('ts')
            ref         = item.get('ref')
            T           = images_base.shape[0]      # = obs_horizon
            H_img, W_img = images_base.shape[1:3]

            # ── Preprocess ────────────────────────────────────────────
            rgb_base  = _prep_rgb(images_base)
            rgb_up    = _prep_rgb(images_up)
            dep_base  = _prep_depth(depths_base, T, H_img, W_img)
            dep_up    = _prep_depth(depths_up,   T, H_img, W_img)

            # ── Interleave base+up, add batch dim ─────────────────────
            rgb_seq   = _interleave(rgb_base, rgb_up).unsqueeze(0)   # (1,T*2,3,H,W)
            depth_seq = _interleave(dep_base, dep_up).unsqueeze(0)   # (1,T*2,1,H,W)

            # agent_pose: (T,D) → repeat_interleave(2,0) → (T*2,D)
            state_t   = (torch.from_numpy(agent_pos_t).float()
                         .to(device)
                         .repeat_interleave(2, dim=0))
            state_seq = state_t.unsqueeze(0)                         # (1,T*2,D)

            enc_ms = (time.perf_counter() - t0) * 1e3

            if log and run_first:
                print(f'[Worker {gpu_id}] shapes: '
                      f'rgb={tuple(rgb_seq.shape)}  '
                      f'dep={tuple(depth_seq.shape)}  '
                      f'state={tuple(state_seq.shape)}')

            # ── DDIM sampling ─────────────────────────────────────────
            t1 = time.perf_counter()
            with torch.no_grad():
                actions = policy.sample_actions(
                    rgb_seq,
                    depth_seq if with_depth else None,
                    state_seq,
                    horizon=pred_horizon,
                )   # (1, pred_horizon, act_dim)  ← already unnormalised
            if device.type == 'cuda':
                torch.cuda.synchronize()

            denoise_ms = (time.perf_counter() - t1) * 1e3
            pred = actions[0].cpu().numpy()  # (pred_horizon, act_dim)

            if log and run_first:
                print(f'[Worker {gpu_id}] pred={pred.shape}  '
                      f'enc={enc_ms:.1f}ms  denoise={denoise_ms:.1f}ms')
                run_first = False

            result_q.put({
                'gpu_id':      gpu_id,
                'naction':     pred,          # PHYSICAL units — no unnorm needed
                'enc_ms':      enc_ms,
                'denoise_ms':  denoise_ms,
                'duration_ms': enc_ms + denoise_ms,
                'ts':          ts,
                'ref':         ref,
                'images_base': images_base,
                'images_up':   images_up,
                'depths_base': depths_base,
                'depths_up':   depths_up,
                'agent_pos_t': agent_pos_t,
            })

    except Exception as e:
        import traceback
        print(f'[Worker {gpu_id}] FATAL: {e}')
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
#  Manager
# ─────────────────────────────────────────────────────────────────────────────

class MultiGPUManagerMultiviewPT:
    """
    One PyTorch inference subprocess per GPU (or one CPU fallback).

    Same dispatch / try_get_result / get_result_blocking interface as the
    original JAX MultiGPUManager.  The only breaking difference is that
    result['naction'] is already in physical units — do NOT unnormalize.
    """

    def __init__(
        self,
        ckpt_path: str,
        max_gpus: Optional[int] = None,
        queue_size: int = 1,
        return_latest: bool = True,
        log: bool = False,
    ):
        self.ckpt_path     = ckpt_path
        self.return_latest = return_latest

        # GPU detection
        n_total = 0
        try:
            if torch.cuda.is_available():
                n_total = torch.cuda.device_count()
        except Exception:
            pass
        if max_gpus is not None:
            n_total = min(max_gpus, n_total)
        if n_total == 0:
            print('[MultiGPUManagerMultiviewPT] No CUDA GPUs — using CPU worker.')
            self.gpu_ids = [None]
        else:
            self.gpu_ids = list(range(n_total))

        self.n   = len(self.gpu_ids)
        ctx      = mp.get_context('spawn')

        self.task_queues  = []
        self.result_queue = ctx.Queue()
        self.procs        = []

        for gid in self.gpu_ids:
            tq = ctx.Queue(maxsize=max(1, queue_size))
            _gid = -1 if gid is None else gid
            p = ctx.Process(
                target   = inference_worker_pt,
                args     = (_gid, ckpt_path, self.result_queue, tq, log),
                daemon   = True,
            )
            p.start()
            self.task_queues.append(tq)
            self.procs.append(p)

        self.next_gpu_idx = 0
        self._dur_hist: list = []

        print(f'[MultiGPUManagerMultiviewPT] '
              f'{self.n} worker(s) spawned — waiting for init …')
        time.sleep(5)   # enough for model load + warmup

    # ── Dispatch ──────────────────────────────────────────────────────────

    def dispatch(
        self,
        images_base: np.ndarray,
        agent_pos_t: np.ndarray,
        images_up:   Optional[np.ndarray] = None,
        depths_base: Optional[np.ndarray] = None,
        depths_up:   Optional[np.ndarray] = None,
        ts=None,
        ref=None,
    ) -> bool:
        gid = self.next_gpu_idx
        self.next_gpu_idx = (self.next_gpu_idx + 1) % self.n

        if images_up is None:
            images_up = np.zeros_like(images_base)

        payload = dict(
            images_base=images_base, images_up=images_up,
            depths_base=depths_base, depths_up=depths_up,
            agent_pose=agent_pos_t, ts=ts, ref=ref,
        )
        tq = self.task_queues[gid]
        for _ in range(2):
            try:
                tq.put_nowait(payload)
                return True
            except pyqueue.Full:
                try:
                    tq.get_nowait()    # evict stale task
                except pyqueue.Empty:
                    pass
        return False   # silent drop; next tick has a fresher frame

    # ── Result collection ─────────────────────────────────────────────────

    def try_get_result(self):
        """Non-blocking — returns latest result or None."""
        latest = None
        while True:
            try:
                latest = self.result_queue.get_nowait()
            except (pyqueue.Empty, Exception):
                break
        if latest is not None:
            self._dur_hist.append(latest.get('duration_ms', 0.0))
            if len(self._dur_hist) > 50:
                self._dur_hist.pop(0)
        return latest

    def get_result_blocking(self, timeout: Optional[float] = None):
        """Blocking — waits up to *timeout* s, returns latest if return_latest."""
        try:
            res = self.result_queue.get(timeout=timeout)
        except pyqueue.Empty:
            return None
        if self.return_latest:
            while True:
                try:
                    res = self.result_queue.get_nowait()
                except (pyqueue.Empty, Exception):
                    break
        return res

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def shutdown(self):
        for tq in self.task_queues:
            try:
                tq.put(None)
            except Exception:
                pass
        for p in self.procs:
            try:
                p.join(timeout=5.0)
            except Exception:
                pass