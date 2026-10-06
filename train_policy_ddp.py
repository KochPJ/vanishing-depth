import os
import copy
import argparse
import time
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from datasets.maniskill_diffusion_dataset import ManiSkillDiffusionDataset
from model.diffusion_policy import DiffusionPolicy
from datasets.multi_obj_sampler import set_seed
from model.deformable_detr.util.misc import is_main_process
from policy_training.ema import EMAModel
from policy_training.lr_schedule import get_cosine_schedule_with_warmup
from datasets.fixed_steps_sampler import FixedStepsSampler
# ─────────────────────────────────────────────────────────────────────────────

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

import json
from datetime import datetime
import math


# ---------------------------------------------------------------------------
# Training logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """
    Persists per-epoch train/eval metrics to JSON.
    """

    MILESTONES = [0.90, 0.95, 0.99]

    def __init__(self, log_path: str, args=None):
        args_dict: dict = {}
        if args is not None:
            _raw = vars(args) if not isinstance(args, dict) else args
            for key, v in _raw.items():
                if key == "device":
                    continue
                if isinstance(v, np.ndarray):
                    continue
                if isinstance(v, torch.Tensor):  
                    continue
                args_dict[key] = v

        self.log_path = log_path
        self.data = {
            "epochs": [],
            "best": {
                "train_loss": {"value": float("inf"), "epoch": -1},
                "test_mse":   {"value": float("inf"), "epoch": -1},
            },
            "milestones": {
                "train_loss": {},
                "test_mse":   {},
            },
            "args": args_dict,
        }

        if os.path.isfile(log_path):
            try:
                with open(log_path, "r") as f:
                    existing = json.load(f)
                self.data["epochs"]     = existing.get("epochs",     [])
                self.data["best"]       = existing.get("best",       self.data["best"])
                self.data["milestones"] = existing.get("milestones", self.data["milestones"])
                if args is not None:
                    self.data["args"] = args_dict
                print(f"[TrainingLogger] Resumed {log_path} "
                      f"({len(self.data['epochs'])} epochs found)")
            except Exception as e:
                print(f"[TrainingLogger] Could not load {log_path}: {e} — starting fresh")

    def log_epoch(self, epoch, train_loss, test_mse, attn_mean=None):
        record = {
            "epoch":      epoch + 1,
            "train_loss": round(float(train_loss), 6),
            "test_mse":   round(float(test_mse),   6),
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if attn_mean:
            record["attn_mean"] = {k: round(float(v), 4) for k, v in attn_mean.items()}

        idx_map = {r["epoch"]: i for i, r in enumerate(self.data["epochs"])}
        if record["epoch"] in idx_map:
            self.data["epochs"][idx_map[record["epoch"]]] = record
        else:
            self.data["epochs"].append(record)
        self.data["epochs"].sort(key=lambda r: r["epoch"])

        changed = False
        for key, val in [("train_loss", train_loss), ("test_mse", test_mse)]:
            if val < self.data["best"][key]["value"]:
                self.data["best"][key] = {"value": round(float(val), 6), "epoch": epoch + 1}
                changed = True
        if changed:
            self._recompute_milestones()
        self._save()

    def _recompute_milestones(self):
        for metric in ["train_loss", "test_mse"]:
            best_val = self.data["best"][metric]["value"]
            if not math.isfinite(best_val) or best_val <= 0:
                continue
            milestones = {}
            for frac in self.MILESTONES:
                label     = f"{int(frac * 100)}pct"
                threshold = best_val * (2.0 - frac)
                for record in self.data["epochs"]:
                    val = record.get(metric)
                    if val is not None and val <= threshold:
                        milestones[label] = {
                            "epoch":     record["epoch"],
                            "value":     round(float(val), 6),
                            "threshold": round(float(threshold), 6),
                        }
                        break
            self.data["milestones"][metric] = milestones

    def _save(self):
        try:
            with open(self.log_path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[TrainingLogger] save failed: {e}")

    def print_summary(self):
        if not is_main_process():
            return
        print("\n[TrainingLogger] ======= Training Summary =======")
        for metric in ["train_loss", "test_mse"]:
            best = self.data["best"][metric]
            print(f"  Best {metric}: {best['value']:.6f}  @ epoch {best['epoch']}")
            ms = self.data["milestones"].get(metric, {})
            for pct in ["90pct", "95pct", "99pct"]:
                if pct in ms:
                    m = ms[pct]
                    print(f"    {pct} of best (≤{m['threshold']:.6f}): "
                          f"epoch {m['epoch']} (value={m['value']:.6f})")
                else:
                    print(f"    {pct} of best: not reached")
        print("[TrainingLogger] =====================================\n")


# ---------------------------------------------------------------------------
# LR parsing + optimizer
# ---------------------------------------------------------------------------

def str_to_lr(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("none", "null"):
        return None
    return float(v)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_optimizer(policy, args):
    if is_main_process():
        print("################ build_optimizer ###############")
    param_groups = []
    seen = set()

    def add_group(name, lr, predicate):
        matched = [(n, p) for n, p in policy.named_parameters() if predicate(n, p)]
        p_sum   = sum(np.prod([int(s) for s in p.shape]) for _, p in matched) / 1e6
        if lr is None:
            for _, p in matched:
                p.requires_grad_(False)
                seen.add(id(p))
            if is_main_process():
                print(f"Optimizer group '{name}': frozen {len(matched)} layers "
                      f"and {p_sum:.4f} mio params (lr=None)")
            return
        params = []
        for n, p in matched:
            if id(p) in seen:
                continue
            p.requires_grad_(True)
            params.append(p)
            seen.add(id(p))
        if params:
            param_groups.append({"params": params, "lr": lr, 'initial_lr': float(lr),
                                  "weight_decay": args.weight_decay})
            if is_main_process():
                print(f"Optimizer group '{name}': {len(params)} layers "
                      f"and {p_sum:.4f} mio params, lr={lr}")

    add_group("backbone", args.lr_backbone,
              lambda n, p: (n.startswith("perception.encoder")
                            or "perception.grasper.backbone" in n
                            or ".backbone." in n))
    add_group("det", args.lr_det,
              lambda n, p: ("perception.grasper" in n)
                           and ("det_" in n or "_det" in n or ".input_proj." in n))
    add_group("obj", args.lr_obj,
              lambda n, p: ("perception.grasper" in n)
                           and ("obj_" in n or "_obj" in n))
    add_group("pose", args.lr_pose,
              lambda n, p: ("perception.grasper" in n)
                           and ("pose_" in n or "_pose" in n))
    add_group("perception_proj", args.lr_proj,
              lambda n, p: (n.startswith("perception")
                            and "encoder" not in n and "grasper" not in n)
                           or n.startswith("perc_to_model"))
    add_group("diffusion_head", args.lr_diff,
              lambda n, p: n.startswith("head"))

    leftovers = [(n, p) for n, p in policy.named_parameters() if id(p) not in seen]
    if leftovers:
        if is_main_process():
            print("[build_optimizer] WARNING: unassigned params (will be frozen):")
            for n, _ in leftovers:
                print("   ", n)
        for _, p in leftovers:
            p.requires_grad_(False)

    if is_main_process():
        total_frozen = total_trained = 0
        for _, p in policy.named_parameters():
            n = p.numel()
            if p.requires_grad:
                total_trained += n
            else:
                total_frozen += n
        print(f"[INFO] trainable: {total_trained/1e6:.4f} M  "
              f"frozen: {total_frozen/1e6:.4f} M")
        print("##############################")

    if not param_groups:
        raise RuntimeError("No param groups with lr>0. Check LR settings.")

    return torch.optim.AdamW(param_groups, lr=1e-4, weight_decay=args.weight_decay)


def load_fitting_state_dict(arch, state_dict, prefix=None):
    ori_sd = arch.state_dict()
    keys   = list(ori_sd.keys())
    keys_left = list(ori_sd.keys())
    wrong_key = wrong_shape = changed_shape = okay = 0

    for key, v in state_dict.items():
        if key not in keys:
            wrong_key += 1
            continue
        sd   = {key: v}
        ls_  = v.shape
        os_  = ori_sd.get(key)
        if os_ is not None:
            os_ = os_.shape
        del keys_left[keys_left.index(key)]
        try:
            arch.load_state_dict(sd, strict=False)
        except Exception as e:
            current_wrong_shape_ = True
            if ".pos_embed" in key:
                pos_embed_checkpoint = state_dict[key]
                embedding_size = pos_embed_checkpoint.shape[-1]
                try:
                    num_patches = arch.encoder.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.encoder.color_encoder.pos_embed.shape[-2] - num_patches
                except Exception:
                    num_patches = arch.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.color_encoder.pos_embed.shape[-2] - num_patches
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                new_size  = int(num_patches ** 0.5)
                if orig_size != new_size:
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    pos_tokens   = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens   = pos_tokens.reshape(-1, orig_size, orig_size,
                                                      embedding_size).permute(0, 3, 1, 2).float()
                    pos_tokens   = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size),
                        mode="bicubic", align_corners=False)
                    pos_tokens   = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    arch.load_state_dict({key: new_pos_embed}, strict=False)
                okay += 1
                current_wrong_shape_ = False
            elif "rope" in key:
                rope_embed_checkpoint = state_dict[key]
                embedding_size = rope_embed_checkpoint.shape[-1]
                try:
                    num_patches = arch.encoder.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.encoder.color_encoder.pos_embed.shape[-2] - num_patches
                except Exception:
                    num_patches = arch.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.color_encoder.pos_embed.shape[-2] - num_patches
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                new_size  = int(num_patches ** 0.5)
                if orig_size != new_size:
                    if is_main_process():
                        print(f"Rope {key} interpolate {orig_size}→{new_size}")
                    rope_tokens = rope_embed_checkpoint.reshape(
                        orig_size, orig_size, embedding_size
                    ).unsqueeze(0).permute(0, 3, 1, 2).float()
                    rope_tokens = torch.nn.functional.interpolate(
                        rope_tokens, size=(new_size, new_size),
                        mode="bicubic", align_corners=False)
                    rope_tokens = rope_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
                    arch.load_state_dict({key: rope_tokens}, strict=False)
                okay += 1
                current_wrong_shape_ = False
            elif os_ is not None and os_ != ls_ and len(os_) == len(ls_):
                vs_ = copy.deepcopy(ori_sd.get(key))
                vs_s = [min(a, b) for a, b in zip(os_, ls_)]
                slices = tuple(slice(0, s) for s in vs_s)
                vs_[slices] = v[slices]
                arch.load_state_dict({key: vs_}, strict=False)
                changed_shape += 1
                okay += 1
                current_wrong_shape_ = False
            if current_wrong_shape_:
                wrong_shape += 1
            continue
        okay += 1

    if is_main_process():
        print(f"Loaded {okay}/{len(keys)} weights, wrong_key={wrong_key}, "
              f"wrong_shape={wrong_shape}, missing={len(keys)-(okay+wrong_key+wrong_shape)}, "
              f"changed_shape={changed_shape}")
    return arch


def filter_joiner(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if "backbone." not in k:
            continue
        idx = k.index("backbone.")
        out[k[idx + len("backbone."):]] = v
    return out


def load_grasper_checkpoint_into_policy(policy, ckpt, args):
    if not args.block_grasper_pretrained:
        if isinstance(ckpt, str):
            ckpt = torch.load(ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        if policy.perception.grasper is not None:
            policy.perception.grasper = load_fitting_state_dict(
                policy.perception.grasper, state
            )
        else:
            policy.perception.encoder = load_fitting_state_dict(
                policy.perception.encoder, filter_joiner(state)
            )
    if args.load_policy:
        if is_main_process():
            print(f"[INFO] loading pretrained policy from {args.load_policy}")
        ckpt2 = torch.load(args.load_policy, map_location="cpu")
        policy = load_fitting_state_dict(policy, ckpt2["model"])
    return policy


# ---------------------------------------------------------------------------
# DDP utils
# ---------------------------------------------------------------------------

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank       = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        args.rank       = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
        )
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    dist.init_process_group(
        backend="nccl", init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank,
    )
    dist.barrier()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        "Diffusion policy DDP training for ManiSkill", add_help=True
    )

    # DDP
    p.add_argument("--dist_url",    default="env://", type=str)
    p.add_argument("--world_size",  default=1,         type=int)
    p.add_argument("--rank",        default=0,         type=int)
    p.add_argument("--local_rank",  default=-1,        type=int)
    p.add_argument("--force_cpu",   default=False,     type=str2bool)

    # Data
    p.add_argument("--horizon",      default=16,    type=int)
    p.add_argument("--obs_horizon",  default=2,     type=int)
    p.add_argument("--action_mode",  default="abs_joints", type=str)
    p.add_argument("--rgb_size",     default=224,   type=int)
    p.add_argument("--data_root",
                   default="/mnt/wimi/publicdata/maniskill/demos/training", type=str)

    # Encoders
    p.add_argument("--encoder_type", default="dino_vd_rgbd_det_obj_grasp", type=str,
                   choices=["ResNet@50", "dino_rgb", "dino_vd_rgbd",
                            "dino_vd_rgbd_det", "dino_vd_rgbd_det_obj",
                            "dino_vd_rgbd_det_obj_grasp"])
    p.add_argument("--max_objects",  default=30, type=int)
    p.add_argument("--grasper_ckpt",
                   default="./data/grasper/pose_with_obj/checkpoint.pth", type=str)
    p.add_argument("--task_filter",  default="StackCube-v1", type=str)

    # Per-module LRs
    p.add_argument("--lr_backbone", default="none", type=str)
    p.add_argument("--lr_det",      default="none", type=str)
    p.add_argument("--lr_obj",      default="none", type=str)
    p.add_argument("--lr_pose",     default="none", type=str)
    p.add_argument("--lr_proj",     default="1e-4", type=str)
    p.add_argument("--lr_diff",     default="1e-4", type=str)

    # Diffusion
    p.add_argument("--act_dim",                default=8,   type=int)
    p.add_argument("--diffusion_steps_train",  default=100, type=int)
    p.add_argument("--diffusion_steps_eval",   default=10,  type=int)
    p.add_argument("--model_dim",              default=256, type=int)
    p.add_argument("--tf_layers",              default=8,   type=int)
    p.add_argument("--tf_heads",               default=8,   type=int)
    p.add_argument("--joiner_feature_map_kernel", default=4, type=int)

    # Training
    p.add_argument("--epochs",        default=50,  type=int)
    p.add_argument(
        "--steps_per_epoch", default=-1, type=int,
        help=(
            "Fixed number of gradient steps per epoch. "
            "The DataLoader will yield exactly this many batches regardless of "
            "dataset size or batch size.  "
            "Pass -1 to fall back to the standard ceil(len(dataset)/batch_size) "
            "behaviour."),)
    p.add_argument("--batch_size",    default=64,   type=int)
    p.add_argument("--num_workers",   default=16,    type=int)
    p.add_argument("--weight_decay",  default=1e-3, type=float)
    p.add_argument("--seed",          default=42,   type=int)
    p.add_argument("--output_dir",    default="./data/grasper_policy_runs", type=str)
    p.add_argument("--special",       default="", type=str)
    p.add_argument("--pretrained_policy_path", default="", type=str)
    p.add_argument("--add_goal",      default=False, type=str2bool)
    p.add_argument("--resume",        default=True,  type=str2bool)
    p.add_argument("--debug",         default=False, type=str2bool)
    p.add_argument("--rgb_only",      default=False, type=str2bool)
    p.add_argument("--block_grasper_pretrained", default=False, type=str2bool)
    p.add_argument("--load_policy",   default="", type=str)

    # ── EMA + LR schedule ─────────────────────────────────────────────
    p.add_argument("--ema_decay",      default=0.9999, type=float,
                   help="EMA shadow decay (0 = no EMA, 0.9999 = recommended)")
    p.add_argument("--warmup_steps",   default=500,    type=int,
                   help="Number of linear warm-up steps for the cosine LR schedule")
    p.add_argument("--min_lr_ratio",   default=0.01,   type=float,
                   help="LR floor = initial_lr * min_lr_ratio (end of cosine)")
    # ──────────────────────────────────────────────────────────────────────

    # WandB
    p.add_argument("--use_wandb",      default=False, type=str2bool)
    p.add_argument("--wandb_project",  default="maniskill_diffusion", type=str)
    p.add_argument("--wandb_run_name", default="debug_run", type=str)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Task specs
# ---------------------------------------------------------------------------

def build_task_specs(root: str, img_size: int) -> Dict:
    suffix = "_224" if img_size <= 224 else ""
    return {
        "PickCube-v1":         {"path": os.path.join(root, f"PickCube_train{suffix}.h5"),         "ratio": 1.0},
        "PushCube-v1":         {"path": os.path.join(root, f"PushCube_train{suffix}.h5"),         "ratio": 1.0},
        "StackCube-v1":        {"path": os.path.join(root, f"StackCube_train{suffix}.h5"),        "ratio": 1.5},
        "PlaceSphere-v1":      {"path": os.path.join(root, f"PlaceSphere_train{suffix}.h5"),      "ratio": 1.0},
        "PegInsertionSide-v1": {"path": os.path.join(root, f"PegInsertionSide_train{suffix}.h5"), "ratio": 2.0},
        "PlugCharger-v1":      {"path": os.path.join(root, f"PlugCharger_train{suffix}.h5"),      "ratio": 2.0},
    }


# ---------------------------------------------------------------------------
# Progress logger (unchanged)
# ---------------------------------------------------------------------------

class PolicyProgressLogger:
    def __init__(self, total_epochs, steps_per_epoch):
        self.total_epochs     = int(total_epochs)
        self.steps_per_epoch  = int(steps_per_epoch)
        self.start_time       = time.time()
        self.last_train_time  = None
        self.last_val_time    = None
        self.train_step_times = []

    @staticmethod
    def _fmt_eta(seconds):
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def start_epoch(self):
        self.last_train_time = None
        self.last_val_time   = None

    def update_train(self, epoch_idx, step_idx, steps_total, loss, running_mean, lr):
        now = time.time()
        if self.last_train_time is not None:
            self.train_step_times.append(now - self.last_train_time)
            if len(self.train_step_times) > 200:
                self.train_step_times = self.train_step_times[-100:]
        self.last_train_time = now

        steps_done   = epoch_idx * steps_total + (step_idx + 1)
        steps_total_ = self.total_epochs * steps_total
        avg_dt       = (sum(self.train_step_times) / len(self.train_step_times)
                        if self.train_step_times else 0.0)
        eta = self._fmt_eta((steps_total_ - steps_done) * avg_dt)

        steps_pct  = 100.0 * (step_idx + 1) / max(1, steps_total)
        epochs_pct = 100.0 * (epoch_idx + (step_idx + 1) / max(1, steps_total)) / max(1, self.total_epochs)

        if is_main_process():
            print(
                f"\rT | e {epoch_idx+1}/{self.total_epochs} ({epochs_pct:5.2f}%) "
                f"s {step_idx+1}/{steps_total} ({steps_pct:5.2f}%) "
                f"ETA {eta} l {loss:.4f} μ {running_mean:.4f} lr {lr:.2e}",
                end="", flush=True,
            )

    def update_eval(self, epoch_idx, step_idx, steps_total, loss, running_mean):
        now = time.time()
        self.last_val_time = now
        steps_pct  = 100.0 * (step_idx + 1) / max(1, steps_total)
        epochs_pct = 100.0 * (epoch_idx + (step_idx + 1) / max(1, steps_total)) / max(1, self.total_epochs)
        if is_main_process():
            print(
                f"\rV | e {epoch_idx+1}/{self.total_epochs} ({epochs_pct:5.2f}%) "
                f"s {step_idx+1}/{steps_total} ({steps_pct:5.2f}%) "
                f"l {loss:.4f} μ {running_mean:.4f}",
                end="", flush=True,
            )

    def newline(self):
        if is_main_process():
            print("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    
    add_scheduler = True
    args.lr_backbone = str_to_lr(args.lr_backbone)
    args.lr_det      = str_to_lr(args.lr_det)
    args.lr_obj      = str_to_lr(args.lr_obj)
    args.lr_pose     = str_to_lr(args.lr_pose)
    args.lr_proj     = str_to_lr(args.lr_proj)
    args.lr_diff     = str_to_lr(args.lr_diff)

    args.rgb_only = args.encoder_type in ("dino_rgb", "ResNet@50")

    init_distributed_mode(args)
    set_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # ── Run dir ────────────────────────────────────────────────────────────
    special = []
    if args.special:
        special.append(args.special)
        if args.special.startswith('ft'):
            print(f'[INFO] finetuning with no lr scheduler and setting lrs with min_lr_ratio: {args.min_lr_ratio}')
            add_scheduler = False
            print(f'[INFO] finetuning with lr_proj {args.lr_proj}->{args.lr_proj * args.min_lr_ratio} via min_lr_ratio: {args.min_lr_ratio}')
            print(f'[INFO] finetuning with lr_diff {args.lr_diff}-{args.lr_diff * args.min_lr_ratio} via min_lr_ratio: {args.min_lr_ratio}')
            args.lr_proj = args.lr_proj * args.min_lr_ratio
            args.lr_diff = args.lr_diff * args.min_lr_ratio

    if args.lr_backbone is not None: 
        special.append("backbone")
    if args.lr_det      is not None: 
        special.append("det")
    if args.lr_obj      is not None: 
        special.append("obj")
    if args.lr_pose     is not None: 
        special.append("pose")
    if args.rgb_only:                
        special.append("rgb")
    if args.add_goal:                
        special.append("goal")
    if not special:                  
        special.append("default")

    run_dir = os.path.join(
        args.output_dir, args.encoder_type, args.task_filter, "_".join(special)
    )
    if is_main_process():
        print(f"[INFO] saving to run_dir: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)

    # ── WandB ──────────────────────────────────────────────────────────────
    if is_main_process() and args.use_wandb and _HAS_WANDB:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))

    # ── Task specs ─────────────────────────────────────────────────────────
    all_specs = build_task_specs(args.data_root, args.rgb_size)
    if args.task_filter == "All":
        train_specs = test_specs = all_specs
    else:
        if args.task_filter not in all_specs:
            raise ValueError(f"Unknown task_filter={args.task_filter}")
        train_specs = test_specs = {args.task_filter: all_specs[args.task_filter]}

    # ── Datasets ───────────────────────────────────────────────────────────
    rgb_size = (args.rgb_size, args.rgb_size)
    train_ds = ManiSkillDiffusionDataset(
        train_specs, horizon=args.horizon, obs_horizon=args.obs_horizon,
        split="train", rgb_size=rgb_size, seed=args.seed,
        action_mode=args.action_mode, add_goal=args.add_goal, rgb_only=args.rgb_only
    )
    test_ds = ManiSkillDiffusionDataset(
        test_specs, horizon=args.horizon, obs_horizon=args.obs_horizon,
        split="test", rgb_size=rgb_size, seed=args.seed,
        action_mode=args.action_mode, add_goal=args.add_goal, rgb_only=args.rgb_only
    )

    act_min_t, act_max_t, act_stats_info = train_ds.get_action_stats()
    if is_main_process():
        print("[train_policy_ddp] action_min:", act_min_t.tolist())
        print("[train_policy_ddp] action_max:", act_max_t.tolist())
        print(f"[ActionStats] action_mode={act_stats_info['action_mode']}")
        print(f"[ActionStats] total samples: {act_stats_info['shape']}")

    args.action_min = act_min_t.tolist()
    args.action_max = act_max_t.tolist()
    

    if is_main_process():
        print("=== TRAIN DATASET STATS ==="); train_ds.summarize()
        print("=== TEST DATASET STATS ===");  test_ds.summarize()

    if args.steps_per_epoch > 0:
        effective_steps = args.steps_per_epoch
        use_fixed_sampler = True
    else:
        # Fallback: all samples / (batch_size * world_size), rounded up
        ws = args.world_size if args.distributed else 1
        effective_steps = math.ceil(len(train_ds) / (args.batch_size * ws))
        use_fixed_sampler = False

    if is_main_process():
        ws = args.world_size if args.distributed else 1
        print(
            f"[Sampler] steps_per_epoch={effective_steps}  "
            f"batch_size={args.batch_size}  world_size={ws}  "
            f"effective_global_batch={args.batch_size * ws}  "
            f"fixed={use_fixed_sampler}"
        )

    # ── Train sampler ──────────────────────────────────────────────────────
    if use_fixed_sampler:
        train_sampler = FixedStepsSampler(
            dataset_size    = len(train_ds),
            steps_per_epoch = effective_steps,
            batch_size      = args.batch_size,
            rank            = args.rank       if args.distributed else 0,
            world_size      = args.world_size if args.distributed else 1,
            shuffle         = True,
            seed            = args.seed,
        )
    else:
        # Original behaviour
        train_sampler = (
            DistributedSampler(train_ds, shuffle=True, drop_last=True)
            if args.distributed else None
        )

    test_sampler  = DistributedSampler(test_ds,  shuffle=False, drop_last=False) \
        if args.distributed else None

    if is_main_process():
        print(f"[DataLoader] num_workers={args.num_workers}  "
              f"batch_size={args.batch_size}")

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        sampler     = train_sampler,
        shuffle     = False,            
        num_workers = args.num_workers,
        drop_last   = True,
        pin_memory  = True,
        persistent_workers = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        sampler     = test_sampler,
        shuffle     = False,
        num_workers = args.num_workers,
        drop_last   = False,
        pin_memory  = True,
        persistent_workers = True,
    )

    # ── Backbone args ──────────────────────────────────────────────────────
    ckpt_meta = torch.load(args.grasper_ckpt, map_location="cpu")
    if "args" in ckpt_meta:
        bb = (argparse.Namespace(**ckpt_meta["args"])
              if isinstance(ckpt_meta["args"], dict) else ckpt_meta["args"])
        bb.device = str(args.device)
        print(f"loading args from {args.grasper_ckpt}")
    else:
        class _BB: pass
        bb = _BB()
        bb.device = str(args.device)
        bb.model_name = "Dino"; bb.model_version = "V2-base"
        bb.fpn_layers = 4; bb.cat_outs = True
        bb.norm_return_layers = bb.norm_return_layers_rgb = True
        bb.resize_attention_maps = bb.return_rdps = False
        bb.with_depth = True; bb.with_intr = False
        bb.pretrained_imagenet = True
        bb.pretrained_path = "./data/backbones/dinov2_vitb14_pretrain.pth"
        bb.num_feature_levels = 4; bb.aux_loss = True
        bb.proj_kernel_size = 1; bb.max_objects = args.max_objects

    if "det"   not in args.encoder_type: bb.with_det = bb.with_obj = bb.with_pose = False
    elif "obj" not in args.encoder_type: bb.with_obj = bb.with_pose = False
    elif "grasp" not in args.encoder_type: bb.with_pose = False

    if args.lr_backbone is not None:
        bb.freeze_encoder = False
    bb.rgb_only = args.rgb_only

    if args.action_mode in ("abs_joints", "relative_joints"):
        args.act_dim = 8
    elif args.action_mode in ("abs_pose", "relative_pose"):
        args.act_dim = 7
    else:
        raise ValueError(f"Unknown action_mode {args.action_mode}")

    robot_state_dim = args.act_dim + (3 if getattr(args, "add_goal", False) else 0)

    if is_main_process():
        print(f"[info] act_dim={args.act_dim}  robot_state_dim={robot_state_dim}")

    # ── Build policy ───────────────────────────────────────────────────────
    device = args.device
    policy = DiffusionPolicy(
        args_backbone=bb,
        encoder_type=args.encoder_type,
        diffusion_max_objs=min(5, args.max_objects),
        token_dim=args.model_dim,
        act_dim=args.act_dim,
        robot_state_dim=robot_state_dim,
        horizon=args.horizon,
        obs_horizon=args.obs_horizon,
        n_views=2,
        model_dim=args.model_dim,
        n_layers=args.tf_layers,
        n_heads=args.tf_heads,
        diffusion_steps_train=args.diffusion_steps_train,
        diffusion_steps_eval=args.diffusion_steps_eval,
        image_kernel=args.joiner_feature_map_kernel,
        action_min=args.action_min,
        action_max=args.action_max,
    )

    # ── Load weights ───────────────────────────────────────────────────────
    resume_path = None
    resumed     = False
    ckpt_resume = None

    if args.pretrained_policy_path:
        resume_path = args.pretrained_policy_path
    elif args.resume:
        cand = os.path.join(run_dir, "checkpoint.pth")
        if os.path.isfile(cand):
            resume_path = cand
            resumed     = True

    start_epoch     = 0
    best_train_loss = float("inf")
    best_train_epoch = -1
    best_eval_mse   = float("inf")
    best_eval_epoch = -1

    if resume_path is not None:
        if is_main_process():
            print(f"[INFO] Resuming from {resume_path}")
        ckpt_resume = torch.load(resume_path, map_location="cpu")
        state = ckpt_resume.get("model", ckpt_resume)
        policy.load_state_dict(state, strict=False)
        if resumed:
            start_epoch      = ckpt_resume.get("epoch", -1) + 1
            best_train_loss  = ckpt_resume.get("best_train_loss",  float("inf"))
            best_train_epoch = ckpt_resume.get("best_train_epoch", -1)
            best_eval_mse    = ckpt_resume.get("best_eval_mse",    float("inf"))
            best_eval_epoch  = ckpt_resume.get("best_eval_epoch",  -1)
    else:
        policy = load_grasper_checkpoint_into_policy(policy, ckpt_meta, args)

    policy = policy.to(device)

    # ── Eval policy — non-DDP copy populated from EMA before each eval
    eval_policy = copy.deepcopy(policy).to(device)
    eval_policy.eval()

    # ── EMA — shadow initialised from current weights ────────────────
    ema = EMAModel(eval_policy, decay=args.ema_decay)
    
    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = build_optimizer(policy, args)
    policy._print_param_summary()

    # ── DDP ────────────────────────────────────────────────────────────────
    if args.distributed:
        
        find_unused_parameters = False if add_scheduler else True
        if is_main_process():
            print(f"[INFO] ddp find_unused_parameters {find_unused_parameters}")
            
        policy = DDP(policy, device_ids=[args.local_rank],
                     find_unused_parameters=find_unused_parameters)


    # ── Cosine LR schedule with linear warmup ────────────────────────
    total_steps  = args.epochs * effective_steps
    
    if add_scheduler:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps = args.warmup_steps,
            total_steps  = total_steps,
            min_lr_ratio = args.min_lr_ratio,
            last_epoch   = start_epoch * effective_steps,
        )
    else:
        lr_scheduler = None

    # ── Restore optimizer / EMA / scheduler from checkpoint ───────────────
    global_step = 0
    if resumed and ckpt_resume is not None:
        if "optimizer" in ckpt_resume:
            try:
                optimizer.load_state_dict(ckpt_resume["optimizer"])
                if is_main_process():
                    print("[INFO] Restored optimizer state")
            except Exception as exc:
                if is_main_process():
                    print(f"[WARN] optimizer restore failed: {exc}")
        if "ema" in ckpt_resume:
            ema.load_state_dict(ckpt_resume["ema"])
            if is_main_process():
                print("[INFO] Restored EMA state")
        if "eval_policy" in ckpt_resume:
            eval_policy.load_state_dict(ckpt_resume["eval_policy"])
            if is_main_process():
                print("[INFO] Restored eval_policy state")
        if "scheduler" in ckpt_resume and lr_scheduler is not None:
            #if isinstance(ckpt_resume["scheduler"], tuple):
            #    ckpt_resume["scheduler"] = ckpt_resume["scheduler"][0]

            print('current scheduler', lr_scheduler.state_dict())
            print('loaded scheduler', ckpt_resume["scheduler"])
            #lr_scheduler.load_state_dict(ckpt_resume["scheduler"])
            #if is_main_process():
            #    print("[INFO] Restored LR scheduler state")
        global_step = ckpt_resume.get(
            "global_step", start_epoch * len(train_loader)
        )

    if is_main_process():
        print(f'[INFO] ema decay {ema.decay}')

    # ── Logger ─────────────────────────────────────────────────────────────
    logger      = PolicyProgressLogger(args.epochs, effective_steps)
    log_path    = os.path.join(run_dir, "train_log.json")
    train_logger = TrainingLogger(log_path, args=args) if is_main_process() else None

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        policy.train()
        p_model = policy.module if isinstance(policy, DDP) else policy
        g = getattr(p_model.perception, "grasper", None)

        if g is None:
            if args.lr_backbone is None:
                enc = getattr(p_model.perception, "encoder", None)
                if enc is not None:
                    for _, param in enc.named_parameters():
                        param.requires_grad_(False)
                    enc.eval()
        else:
            if args.lr_backbone is None: g._freeze_and_eval_backbone()
            if args.lr_det      is None: g._freeze_and_eval_det()
            if args.lr_obj      is None: g._freeze_and_eval_obj()
            if args.lr_pose     is None: g._freeze_and_eval_pose()

        train_loss_sum = 0.0
        n_train_steps  = 0
        t0 = time.time()
        logger.start_epoch()

        for i, batch in enumerate(train_loader):
            if args.debug and i > 10:
                break

            rgb_seq = batch["rgb"].to(device)
            if not args.rgb_only:
                depth_seq = batch["depth"].to(device)
            else:
                depth_seq = None
            robot_state_seq = batch["robot_state"].to(device)
            actions = batch["actions"].to(device)

            loss = policy(rgb_seq, depth_seq, robot_state_seq, actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            # ── EMA shadow update + LR step ─────────────────────────
            ema.step(policy.module if isinstance(policy, DDP) else policy)
            if lr_scheduler is not None:
                lr_scheduler.step()
            global_step += 1

            train_loss_sum += float(loss.item())
            n_train_steps  += 1
            running_mean    = train_loss_sum / max(1, n_train_steps)
            current_lr      = optimizer.param_groups[-1]["lr"]   # updated by scheduler

            logger.update_train(
                epoch_idx=epoch, step_idx=i, steps_total=len(train_loader),
                loss=float(loss.item()), running_mean=running_mean, lr=current_lr,
            )

        # All-reduce train loss across ranks
        if dist.is_available() and dist.is_initialized():
            v = torch.tensor([train_loss_sum, n_train_steps], device=device)
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            train_loss_sum, n_train_steps = float(v[0]), int(v[1])

        train_loss_avg = train_loss_sum / max(1, n_train_steps)
        logger.newline()
        if is_main_process():
            print(f"[Epoch {epoch+1}] train_loss={train_loss_avg:.6f} "
                  f"({time.time()-t0:.1f}s)  lr={optimizer.param_groups[-1]['lr']:.2e}")

        # ── Evaluation ─────────────────────────────────────────────────────
        # copy EMA shadow into eval_policy before running inference
        eval_policy = ema.copy_to(eval_policy)
        #eval_policy = copy.deepcopy(policy.module if isinstance(policy, DDP) else policy)
        eval_policy.eval()
        #olicy.eval()          # keeps grasper BN layers in eval mode

        attn_acc   = {}
        attn_count = {}

        with torch.no_grad():
            mse_sum   = 0.0
            n_samples = 0
            logger.start_epoch()

            for j, batch in enumerate(test_loader):
                if args.debug and j > 10:
                    break

                rgb = batch["rgb"].to(device)
                if not args.rgb_only:
                    depth = batch["depth"].to(device)
                else:
                    depth = None

                robot_state = batch["robot_state"].to(device)
                actions_gt  = batch["actions"].to(device)
                B           = actions_gt.shape[0]


                #loss = eval_policy(rgb, depth, robot_state, copy.deepcopy(actions_gt))
                #print('loss', loss)

                # ── always use EMA eval_policy for inference ─────────
                actions_pred = eval_policy.sample_actions(
                    rgb, depth, robot_state, horizon=args.horizon
                )

                actions_gt_norm = eval_policy._normalize(actions_gt)
                actions_pred_norm = eval_policy._normalize(actions_pred)
                mse = ((actions_pred_norm - actions_gt_norm) ** 2).mean(dim=(1, 2))


                #diff = actions_pred - actions_gt
                #print('diff shape', diff.shape, actions_pred.shape, actions_gt.shape)
                #mse = (diff ** 2).mean(dim=(1, 2))
                #print('mse', mse.shape)
                #print('val mse', float(mse.sum().item()))
                #print('mse mean', float(mse.mean().item()), float(mse.sum().item()), B)

                mse_sum   += float(mse.sum().item())
                n_samples += B

                logger.update_eval(
                    epoch_idx=epoch, step_idx=j, steps_total=len(test_loader),
                    loss=float(mse.mean().item()), running_mean=mse_sum / max(1, n_samples),
                )

                # Attention stats (rank 0 only)
                if is_main_process():
                    attn_stats_batch = eval_policy.get_attention_stats(
                        rgb, depth, robot_state, actions_gt,
                        only_action_queries=True,
                    )
                    for g_name, val in attn_stats_batch.items():
                        attn_acc.setdefault(g_name, 0.0)
                        attn_count.setdefault(g_name, 0)
                        attn_acc[g_name]   += float(val)
                        attn_count[g_name] += 1

            logger.newline()

            if dist.is_available() and dist.is_initialized():
                v = torch.tensor([mse_sum, n_samples], device=device)
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
                mse_sum, n_samples = float(v[0]), int(v[1])

            test_mse = mse_sum / max(1, n_samples)

        if is_main_process():
            print(f"[Epoch {epoch+1}] test_offset_mse={test_mse:.6f}")

            attn_mean = None
            if attn_acc:
                attn_mean = {
                    g: attn_acc[g] / attn_count[g]
                    for g in attn_acc if attn_count[g] > 0
                }
                print("[Attention stats] mean over eval batches:")
                for g_name in sorted(attn_mean):
                    print(f"    {g_name}: {attn_mean[g_name]:.3f}")

            if args.use_wandb and _HAS_WANDB:
                log_dict = {
                    "train/loss":     train_loss_avg,
                    "test/mse":       test_mse,
                    "train/lr":       optimizer.param_groups[-1]["lr"],
                    "epoch":          epoch + 1,
                    "global_step":    global_step,
                }
                if attn_mean:
                    for g_name, val in attn_mean.items():
                        log_dict[f"attn/{g_name}"] = val
                wandb.log(log_dict, step=epoch + 1)

            # ── Save checkpoints ───────────────────────────────────────────
            # checkpoint now includes EMA state, scheduler state,
            #      global_step, and the optimizer state for clean resuming.
            ckpt_common = {
                "epoch":            epoch,
                "global_step":      global_step,
                "model":            (policy.module.state_dict()
                                     if isinstance(policy, DDP)
                                     else policy.state_dict()),
                "eval_policy":      eval_policy.state_dict(),       
                "ema":              ema.state_dict(),         
                "optimizer":        optimizer.state_dict(),   
                "args":             vars(args),
                "best_train_loss":  best_train_loss,
                "best_train_epoch": best_train_epoch,
                "best_eval_mse":    best_eval_mse,
                "best_eval_epoch":  best_eval_epoch,
                "attn_mean":        attn_mean,
            }
            if lr_scheduler is not None:
                ckpt_common["scheduler"] = lr_scheduler.state_dict()

            if args.debug:
                input("[DEBUG] working?")

            # 1) Always-current checkpoint
            torch.save(ckpt_common, os.path.join(run_dir, "checkpoint.pth"))

            # 2) Best training loss
            if train_loss_avg < best_train_loss:
                best_train_loss  = train_loss_avg
                best_train_epoch = epoch + 1
                ckpt_common["best_train_loss"]  = best_train_loss
                ckpt_common["best_train_epoch"] = best_train_epoch
                torch.save(ckpt_common, os.path.join(run_dir, "best_train.pth"))

            # 3) Best eval MSE
            if test_mse < best_eval_mse:
                best_eval_mse   = test_mse
                best_eval_epoch = epoch + 1
                ckpt_common["best_eval_mse"]   = best_eval_mse
                ckpt_common["best_eval_epoch"] = best_eval_epoch
                torch.save(ckpt_common, os.path.join(run_dir, "best_eval.pth"))

            train_logger.log_epoch(
                epoch=epoch, 
                train_loss=train_loss_avg,
                test_mse=test_mse, 
                attn_mean=attn_mean,
            )

        if args.distributed:
            dist.barrier()

    if is_main_process():
        train_logger.print_summary()

    if is_main_process() and args.use_wandb and _HAS_WANDB:
        wandb.finish()

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


'''
CUDA_VISIBLE_DEVICES=1 torchrun \
--nproc_per_node=1 \
--master_port=29502 train_policy_ddp.py   \
--data_root /mnt/data/publicdata/maniskill/demos/training   \
--task_filter  PickCube-v1  \
--encoder_type dino_vd_rgbd_det_obj_grasp   \
--grasper_ckpt ./data/grasper/pose_with_obj/checkpoint.pth   \
--batch_size 64 \
--epochs 50 \
--add_goal True \
--seed 46 \
--special t5


PlaceSphere-v1
PickCube-v1 
StackCube-v1
PushCube-v1



add_goal True


CUDA_VISIBLE_DEVICES=2 torchrun \
--nproc_per_node=1 \
--master_port=29504 train_policy_ddp.py   \
--data_root /mnt/data/publicdata/maniskill/demos/training   \
--task_filter StackCube-v1   \
--encoder_type dino_vd_rgbd_det_obj_grasp   \
--grasper_ckpt ./data/grasper/pose_with_obj/checkpoint.pth   \
--batch_size 128 \
--epochs 200 \
--special t2_200 \
--seed 43 \
--num_workers 16


CUDA_VISIBLE_DEVICES=2 torchrun \
--nproc_per_node=1 \
--master_port=29505 train_policy_ddp.py   \
--data_root /mnt/data/publicdata/maniskill/demos/training   \
--task_filter StackCube-v1   \
--encoder_type dino_rgb   \
--grasper_ckpt ./data/grasper/pose_with_obj/checkpoint.pth   \
--batch_size 128 \
--epochs 150 \
--special t2 \
--seed 43 \
--num_workers 32



CUDA_VISIBLE_DEVICES=3 torchrun \
--nproc_per_node=1 \
--master_port=29506 train_policy_ddp.py   \
--data_root /mnt/wimi/publicdata/maniskill/demos/training   \
--task_filter StackCube-v1   \
--encoder_type dino_rgb   \
--grasper_ckpt ./data/grasper/pose_with_obj/checkpoint.pth   \
--batch_size 128 \
--epochs 200 \
--special t2_200 \
--seed 43 \
--num_workers 32







--horizon 16   \
--obs_horizon 2 \
--joiner_feature_map_kernel 1 \
--diffusion_steps_train 100   \
--diffusion_steps_eval 10 \
--lr_backbone 1e-5   \
--lr_det 1e-5   \
--lr_obj 1e-5  \
--lr_pose 1e-5   \
--lr_proj 1e-4   \
--lr_diff 1e-4   \
--ema_decay 0.9999 \
--warmup_steps 500 \
--min_lr_ratio 0.01 \
--use_wandb false \
--resume True




CUDA_VISIBLE_DEVICES=2,3 torchrun \
--nproc_per_node=2 \
--master_port=29504 train_policy_ddp.py   \
--task_filter PickCube-v1   \
--encoder_type dino_vd_rgbd   \
--batch_size 64 \
--epochs 50   \
--lr_backbone None   \
--lr_det None   \
--lr_obj None  \
--lr_pose None   \
--lr_proj 1e-4   \
--lr_diff 1e-4   \
--ema_decay 0.9999 \
--warmup_steps 500 \
--min_lr_ratio 0.01
'''