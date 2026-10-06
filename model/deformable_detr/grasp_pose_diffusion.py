from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------

def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Quaternion order: [w, x, y, z].
    """
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)

    # Same SO(3) rotation for q and -q. Use one hemisphere consistently.
    return torch.where(q[..., :1] < 0.0, -q, q)


def quaternion_geodesic(
    q1: torch.Tensor,
    q2: torch.Tensor,
) -> torch.Tensor:
    """
    q1/q2: [..., 4], wxyz.
    Returns angular distance in radians, in [0, pi].
    """
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    dot = (q1 * q2).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def quat_wxyz_to_rotvec(
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    q: [...,4], [w,x,y,z].
    Returns [...,3] rotation-vector / axis-angle representation in radians.
    """
    q = normalize_quaternion(q)

    w = q[..., 0].clamp(-1.0, 1.0)
    xyz = q[..., 1:]

    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w.unsqueeze(-1))
    axis = xyz / sin_half.clamp_min(eps)

    rvec = axis * angle

    return torch.where(
        sin_half > eps,
        rvec,
        torch.zeros_like(rvec),
    )


def rotvec_to_quat_wxyz(
    rvec: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    rvec: [...,3], radians.
    Returns [...,4] quaternion in [w,x,y,z].
    """
    angle = rvec.norm(dim=-1, keepdim=True)
    axis = rvec / angle.clamp_min(eps)

    half_angle = 0.5 * angle

    q = torch.cat(
        [
            torch.cos(half_angle),
            axis * torch.sin(half_angle),
        ],
        dim=-1,
    )

    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0

    q = torch.where(angle > eps, q, identity)
    return normalize_quaternion(q)


# ---------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------

def cosine_beta_schedule(
    num_steps: int,
    s: float = 0.008,
) -> torch.Tensor:
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float32)

    alpha_bar = torch.cos(
        ((x / num_steps) + s) / (1.0 + s) * math.pi * 0.5
    ) ** 2

    alpha_bar = alpha_bar / alpha_bar[0]

    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(min=1e-5, max=0.999)


# ---------------------------------------------------------------------
# Confidence target
# ---------------------------------------------------------------------

def grasp_confidence_target(
    gt_score: torch.Tensor,
    translation_error_m: torch.Tensor,
    rotation_error_deg: torch.Tensor,
    translation_threshold_m: float = 0.020,
    rotation_threshold_deg: float = 15.0,
    full_score_fraction: float = 0.10,
) -> torch.Tensor:
    """
    Soft target in [0,1].

    Full GT score until:
      translation <= 2 mm
      rotation <= 1.5 deg

    Linear decay:
      translation from 2 mm to 20 mm
      rotation from 1.5 deg to 15 deg

    Score is exactly zero if either:
      translation >= 20 mm
      rotation >= 15 deg
    """
    trans_start = translation_threshold_m * full_score_fraction
    rot_start = rotation_threshold_deg * full_score_fraction

    trans_penalty = (
        (translation_error_m - trans_start)
        / max(translation_threshold_m - trans_start, 1e-8)
    ).clamp(0.0, 1.0)

    rot_penalty = (
        (rotation_error_deg - rot_start)
        / max(rotation_threshold_deg - rot_start, 1e-8)
    ).clamp(0.0, 1.0)

    # Strict: poor translation OR poor orientation lowers the score.
    combined_penalty = torch.maximum(trans_penalty, rot_penalty)

    geometric_quality = (1.0 - combined_penalty).clamp(0.0, 1.0)

    return gt_score.clamp(0.0, 1.0) * geometric_quality


# ---------------------------------------------------------------------
# Diffusion head
# ---------------------------------------------------------------------

class GraspPoseDiffusionHead(nn.Module):
    """
    Conditional DDPM/DDIM grasp generator.

    Diffused pose representation:
        [x, y, z, rx, ry, rz]

    x/y/z are in meters.
    rvec components are in radians.

    All metric values are normalized to [-1, 1] before diffusion.
    """

    def __init__(
        self,
        cond_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_train_steps: int = 100,
        num_eval_steps: int = 10,
        dropout: float = 0.1,
        pose_min=None,
        pose_max=None,
    ):
        super().__init__()

        self.pose_dim = 6
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_train_steps = int(num_train_steps)
        self.num_eval_steps = int(num_eval_steps)

        if pose_min is None:
            pose_min = [
                -3.0,
                -3.0,
                0.0,
                -math.pi,
                -math.pi,
                -math.pi,
            ]

        if pose_max is None:
            pose_max = [
                3.0,
                3.0,
                3.0,
                math.pi,
                math.pi,
                math.pi,
            ]

        pose_min = torch.as_tensor(pose_min, dtype=torch.float32)
        pose_max = torch.as_tensor(pose_max, dtype=torch.float32)

        if pose_min.numel() != self.pose_dim:
            raise ValueError("pose_min must contain exactly 6 values.")
        if pose_max.numel() != self.pose_dim:
            raise ValueError("pose_max must contain exactly 6 values.")
        if torch.any(pose_max <= pose_min):
            raise ValueError("Every pose_max entry must be > pose_min.")

        # Saved in checkpoints, transferred automatically with .to(device).
        self.register_buffer(
            "pose_min",
            pose_min.view(1, 1, 1, self.pose_dim),
        )
        self.register_buffer(
            "pose_max",
            pose_max.view(1, 1, 1, self.pose_dim),
        )

        self.pose_in = nn.Linear(self.pose_dim, hidden_dim)
        self.cond_in = nn.Linear(cond_dim, hidden_dim)

        self.time_in = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.noise_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.pose_dim),
        )

        # Predicts a score for each generated pose-query.
        self.confidence_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        betas = cosine_beta_schedule(num_train_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer(
            "sqrt_alpha_bar",
            torch.sqrt(alpha_bar),
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_bar",
            torch.sqrt(1.0 - alpha_bar),
        )

    def normalize_pose(self, pose_metric: torch.Tensor) -> torch.Tensor:
        """
        pose_metric: [B,K,P,6], meter/radian.
        """
        denom = (self.pose_max - self.pose_min).clamp_min(1e-6)

        pose_norm = (
            2.0 * (pose_metric - self.pose_min) / denom - 1.0
        )

        return pose_norm.clamp(-1.0, 1.0)

    def unnormalize_pose(self, pose_norm: torch.Tensor) -> torch.Tensor:
        """
        pose_norm: [B,K,P,6], approximately [-1,1].
        """
        pose_norm = pose_norm.clamp(-1.0, 1.0)

        pose_metric = 0.5 * (pose_norm + 1.0) * (
            self.pose_max - self.pose_min
        ) + self.pose_min

        # rvec must correspond to a canonical SO(3) rotation with angle <= pi.
        rvec = pose_metric[..., 3:6]
        angle = rvec.norm(dim=-1, keepdim=True)

        scale = torch.where(
            angle > math.pi,
            math.pi / angle.clamp_min(1e-8),
            torch.ones_like(angle),
        )

        rvec = rvec * scale

        return torch.cat(
            [
                pose_metric[..., :3],
                rvec,
            ],
            dim=-1,
        )

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2

        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(10000.0),
                half,
                device=t.device,
                dtype=torch.float32,
            )
        )

        x = t.float().unsqueeze(-1) / freqs.unsqueeze(0)

        emb = torch.cat(
            [
                torch.sin(x),
                torch.cos(x),
            ],
            dim=-1,
        )

        if emb.shape[-1] < self.hidden_dim:
            emb = F.pad(emb, (0, self.hidden_dim - emb.shape[-1]))

        return emb

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        x0/noise: [B,K,P,6]
        t:        [B]
        """
        alpha = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

        return alpha * x0 + sigma * noise

    def predict_noise(
        self,
        noisy_pose: torch.Tensor,
        pose_hidden: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        noisy_pose: [B,K,P,6]
        pose_hidden: [B,K,P,C]
        t: [B]

        Returns:
          predicted_noise: [B,K,P,6]
          confidence_logits: [B,K,P]
        """
        B, K, P, _ = noisy_pose.shape

        pose_tokens = self.pose_in(
            noisy_pose.reshape(B * K, P, self.pose_dim)
        )

        cond_tokens = self.cond_in(
            pose_hidden.reshape(B * K, P, pose_hidden.shape[-1])
        )

        time_tokens = self.time_in(self._time_embedding(t))
        time_tokens = time_tokens[:, None, None, :].expand(B, K, P, -1)
        time_tokens = time_tokens.reshape(B * K, P, self.hidden_dim)

        hidden = pose_tokens + cond_tokens + time_tokens
        hidden = self.transformer(hidden)

        noise_pred = self.noise_out(hidden)
        confidence_logits = self.confidence_out(hidden).squeeze(-1)

        return (
            noise_pred.reshape(B, K, P, self.pose_dim),
            confidence_logits.reshape(B, K, P),
        )

    @torch.no_grad()
    def ddim_sample(
        self,
        pose_hidden: torch.Tensor,
        steps: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          pose_metric: [B,K,P,6] = xyz + rvec
          confidence:  [B,K,P] in [0,1]
        """
        if steps is None:
            steps = self.num_eval_steps

        B, K, P, _ = pose_hidden.shape
        device = pose_hidden.device

        x = torch.randn(B, K, P, self.pose_dim, device=device)

        timesteps = torch.linspace(
            self.num_train_steps - 1,
            0,
            steps,
            device=device,
        ).long()

        final_conf_logits = None

        for i, t_scalar in enumerate(timesteps):
            t = torch.full(
                (B,),
                int(t_scalar.item()),
                dtype=torch.long,
                device=device,
            )

            eps, conf_logits = self.predict_noise(
                noisy_pose=x,
                pose_hidden=pose_hidden,
                t=t,
            )

            final_conf_logits = conf_logits

            alpha_t = self.alpha_bar[t_scalar].view(1, 1, 1, 1)

            x0 = (
                x - torch.sqrt(1.0 - alpha_t) * eps
            ) / torch.sqrt(alpha_t)

            x0 = x0.clamp(-1.0, 1.0)

            if i == len(timesteps) - 1:
                x = x0
                break

            prev_t = timesteps[i + 1]
            alpha_prev = self.alpha_bar[prev_t].view(1, 1, 1, 1)

            # Deterministic DDIM, eta=0.
            x = (
                torch.sqrt(alpha_prev) * x0
                + torch.sqrt(1.0 - alpha_prev) * eps
            )

        pose_metric = self.unnormalize_pose(x)
        confidence = torch.sigmoid(final_conf_logits)

        return pose_metric, confidence


# ---------------------------------------------------------------------
# Hungarian assignment: detection -> object slot -> good grasp
# ---------------------------------------------------------------------

def build_slot_grasp_targets(
    outs: Dict[str, torch.Tensor],
    targets: List[Dict],
    translation_cost_scale_m: float = 0.05,
    rotation_cost_weight: float = 0.25,
    confidence_translation_threshold_m: float = 0.020,
    confidence_rotation_threshold_deg: float = 15.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      target_pose_rvec: [B,K,P,6]
      target_conf:      [B,K,P]
      valid_mask:       [B,K,P]

    Important:
      Only actual Hungarian matched pose-queries are valid.
      Unmatched P pose-queries are excluded from diffusion and confidence loss.
    """
    from model.deformable_detr.grasping_loss import HungarianMatcher

    pred_boxes = outs["pred_boxes"].detach()
    pred_logits = outs["pred_logits"].detach()
    obj_index = outs["obj_index"].detach()

    # Direct pretrained pose head output is used solely as detached matching anchor.
    anchor_pose = outs["pose_preds"].detach()
    if anchor_pose.dim() == 5:
        anchor_pose = anchor_pose[-1]

    B, K, P, _ = anchor_pose.shape
    device = anchor_pose.device
    dtype = anchor_pose.dtype

    target_pose_rvec = torch.zeros(
        B,
        K,
        P,
        6,
        device=device,
        dtype=dtype,
    )

    target_conf = torch.zeros(
        B,
        K,
        P,
        device=device,
        dtype=dtype,
    )

    valid_mask = torch.zeros(
        B,
        K,
        P,
        device=device,
        dtype=torch.bool,
    )

    det_matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=1.0,
        cost_giou=2.0,
        cost_embed=0.0,
    )

    for b in range(B):
        src_idx, tgt_idx = det_matcher(
            pred_boxes[b],
            pred_logits[b],
            targets[b]["boxes"].to(device),
            targets[b]["labels"].to(device),
        )

        det_to_gt = {
            int(det_q.item()): int(gt_q.item())
            for det_q, gt_q in zip(src_idx, tgt_idx)
        }

        box2inst = targets[b]["box2inst"].to(device)

        for k in range(K):
            det_query_idx = int(obj_index[b, k].item())

            if det_query_idx not in det_to_gt:
                continue

            gt_box_idx = det_to_gt[det_query_idx]
            inst_id = int(box2inst[gt_box_idx].item())

            packed = (targets[b].get("gt_grasps", {}) or {}).get(inst_id)
            if packed is None:
                continue

            xyz = packed.get("xyz")
            quat = packed.get("quat_wxyz")
            score = packed.get("score")

            if not isinstance(xyz, torch.Tensor):
                continue
            if not isinstance(quat, torch.Tensor):
                continue
            if xyz.numel() == 0 or quat.numel() == 0:
                continue

            xyz = xyz.to(device=device, dtype=dtype)
            quat = normalize_quaternion(
                quat.to(device=device, dtype=dtype)
            )

            if isinstance(score, torch.Tensor):
                score = score.to(device=device, dtype=dtype).view(-1)
            else:
                score = torch.ones(
                    xyz.shape[0],
                    device=device,
                    dtype=dtype,
                )

            G = min(
                xyz.shape[0],
                quat.shape[0],
                score.shape[0],
            )

            if G <= 0:
                continue

            xyz = xyz[:G]
            quat = quat[:G]
            score = score[:G].clamp(0.0, 1.0)

            # Existing pretrained direct grasp predictions are matching anchors.
            proposal_xyz = anchor_pose[b, k, :, :3]
            proposal_quat = normalize_quaternion(
                anchor_pose[b, k, :, 3:7]
            )

            trans_cost = torch.cdist(proposal_xyz, xyz)

            q1 = proposal_quat[:, None, :].expand(P, G, 4)
            q2 = quat[None, :, :].expand(P, G, 4)

            rot_cost = quaternion_geodesic(q1, q2) / math.pi

            cost = (
                trans_cost / translation_cost_scale_m
                + rotation_cost_weight * rot_cost
            )

            row_np, col_np = linear_sum_assignment(
                cost.detach().float().cpu().numpy()
            )

            if len(row_np) == 0:
                continue

            row = torch.as_tensor(
                row_np,
                device=device,
                dtype=torch.long,
            )

            col = torch.as_tensor(
                col_np,
                device=device,
                dtype=torch.long,
            )

            matched_xyz = xyz[col]
            matched_quat = quat[col]
            matched_score = score[col]

            target_pose_rvec[b, k, row] = torch.cat(
                [
                    matched_xyz,
                    quat_wxyz_to_rotvec(matched_quat),
                ],
                dim=-1,
            )

            # Confidence target uses the pretrained direct pose-anchor error.
            translation_error_m = torch.norm(
                proposal_xyz[row] - matched_xyz,
                dim=-1,
            )

            rotation_error_deg = quaternion_geodesic(
                proposal_quat[row],
                matched_quat,
            ) * (180.0 / math.pi)

            target_conf[b, k, row] = grasp_confidence_target(
                gt_score=matched_score,
                translation_error_m=translation_error_m,
                rotation_error_deg=rotation_error_deg,
                translation_threshold_m=confidence_translation_threshold_m,
                rotation_threshold_deg=confidence_rotation_threshold_deg,
                full_score_fraction=0.10,
            )

            valid_mask[b, k, row] = True

    return target_pose_rvec, target_conf, valid_mask


# ---------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------

def grasp_diffusion_loss(
    diffusion_head: GraspPoseDiffusionHead,
    outs: Dict[str, torch.Tensor],
    targets: List[Dict],
    confidence_loss_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    DDPM epsilon-prediction loss plus confidence BCE loss.
    """
    pose_hidden = outs["pose_hidden"]

    target_pose, target_conf, valid_mask = build_slot_grasp_targets(
        outs=outs,
        targets=targets,
    )

    if not valid_mask.any():
        zero = pose_hidden.sum() * 0.0

        return zero, {
            "loss_grasp_diffusion_eps": zero.detach(),
            "loss_grasp_diffusion_conf": zero.detach(),
            "grasp_diffusion_valid_queries": zero.detach(),
            "grasp_diffusion_valid_slots": zero.detach(),
            "grasp_diffusion_conf_target_mean": zero.detach(),
        }

    B = target_pose.shape[0]

    t = torch.randint(
        low=0,
        high=diffusion_head.num_train_steps,
        size=(B,),
        device=target_pose.device,
        dtype=torch.long,
    )

    x0 = diffusion_head.normalize_pose(target_pose)
    noise = torch.randn_like(x0)

    x_t = diffusion_head.q_sample(
        x0=x0,
        t=t,
        noise=noise,
    )

    noise_pred, conf_logits = diffusion_head.predict_noise(
        noisy_pose=x_t,
        pose_hidden=pose_hidden,
        t=t,
    )

    pose_mask = valid_mask.unsqueeze(-1).expand_as(noise)

    loss_eps = F.mse_loss(
        noise_pred[pose_mask],
        noise[pose_mask],
    )

    loss_conf = F.binary_cross_entropy_with_logits(
        conf_logits[valid_mask],
        target_conf[valid_mask],
    )

    total_loss = loss_eps + confidence_loss_weight * loss_conf

    return total_loss, {
        "loss_grasp_diffusion_eps": loss_eps.detach(),
        "loss_grasp_diffusion_conf": loss_conf.detach(),
        "grasp_diffusion_valid_queries": valid_mask.float().sum().detach(),
        "grasp_diffusion_valid_slots": (
            valid_mask.any(dim=-1).float().sum().detach()
        ),
        "grasp_diffusion_conf_target_mean": (
            target_conf[valid_mask].mean().detach()
        ),
        "grasp_diffusion_mean_timestep": t.float().mean().detach(),
    }