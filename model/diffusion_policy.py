# model/diffusion_policy.py

import math
from typing import List, Dict, Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vanishing_depth import build_encoder as build_vd_encoder
from model.deformable_detr.grasp_estimation_transformer import build_grasper
from model.deformable_detr.backbone import DepthEncoder, DepthEncodingConfig, build_backbone
from model.deformable_detr.grasp_deform_transformer import ModelPointNetEncoder
from model.deformable_detr.util.misc import nested_tensor_from_tensor_list, is_main_process


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_params(mod: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in mod.parameters(recurse=True):
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _fmt_m(n: int) -> str:
    return f"{n/1e6:.2f}M"


# ---------------------------------------------------------------------------
# Time embedding for diffusion
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] int/float → [B, dim]"""
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device)
        )
        args = t.float().unsqueeze(1) / freqs.unsqueeze(0)   # [B, half]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb   # [B, dim]


# ---------------------------------------------------------------------------
# Cosine beta schedule (replaces linear linspace)
# ---------------------------------------------------------------------------

def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Squared-cosine noise schedule (Nichol & Dhariwal, Improved DDPM, 2021).

    Compared to the original linear schedule the cosine schedule injects less
    noise at both extremes, which helps the model learn fine-grained action
    details near t=0 and coarse structure near t=T.
    """
    steps      = T + 1
    t          = torch.linspace(0.0, T, steps) / T
    alphas_bar = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]          # normalise so ab[0]==1
    betas      = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(min=1e-5, max=0.999)


# ---------------------------------------------------------------------------
# Gaussian diffusion (DDPM / DDIM) on action sequences
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    def __init__(self, num_train_steps: int = 100, num_eval_steps: int = 10,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.num_eval_steps  = num_eval_steps

        # ── Cosine schedule ────────────────────────────────────────────────
        # beta_start / beta_end kept as kwargs for API compat, but ignored.
        betas = _cosine_beta_schedule(num_train_steps)

        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]])

        self.register_buffer("betas",               betas)
        self.register_buffer("alphas",              alphas)
        self.register_buffer("alphas_cumprod",      alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)  
        self.register_buffer("sqrt_alphas_cumprod",
                             torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """q(x_t | x_0).  x0: [B,T,A],  t: [B] int."""
        if noise is None:
            noise = torch.randn_like(x0)
        t = t.to(x0.device)
        sqrt_alpha  = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_m  = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_m * noise

    @torch.no_grad()
    def ddim_sample(self, model, x_T: torch.Tensor,
                    cond_tokens: torch.Tensor,
                    steps: Optional[int] = None) -> torch.Tensor:
        """
        Deterministic DDIM sampling.
        The DDIM reverse step only requires alpha_bar_t and alpha_bar_{t-1}
        (both available from alphas_cumprod).  The per-step non-cumulative
        alpha_t appears only in the DDPM posterior mean, not in DDIM.
        """
        device = x_T.device
        if steps is None:
            steps = self.num_eval_steps

        step_indices = torch.linspace(
            0, self.num_train_steps - 1, steps,
            dtype=torch.long, device=device
        ).flip(0)   # descending: T-1 → 0

        x = x_T
        for i, t_idx in enumerate(step_indices):
            t   = torch.full((x.shape[0],), int(t_idx.item()),
                             device=device, dtype=torch.long)
            eps = model(x, cond_tokens, t)

            # ── only alpha_bar quantities are needed for DDIM ──────────────
            alpha_bar_t    = self.alphas_cumprod[t_idx]
            sqrt_ab        = alpha_bar_t.sqrt()
            sqrt_one_m_ab  = (1.0 - alpha_bar_t).sqrt()

            x0_pred = (x - sqrt_one_m_ab * eps) / sqrt_ab
            x0_pred = x0_pred.clamp(-1.0, 1.0)  # clamp to the normalized range


            if i == steps - 1:
                x = x0_pred
            else:
                alpha_bar_prev = self.alphas_cumprod[step_indices[i + 1]]
                x = alpha_bar_prev.sqrt() * x0_pred + (1.0 - alpha_bar_prev).sqrt() * eps

        return x


# ---------------------------------------------------------------------------
# Temporal Positional Encoding for stacked observation windows
# ---------------------------------------------------------------------------

class TemporalPositionalEncoding(nn.Module):
    """
    Adds a distinct learnable embedding vector to every time-slice of
    observation tokens before the slices are concatenated into the full
    condition sequence.

    Without this the Transformer cannot distinguish 'now' from '2 steps ago'
    when obs_horizon > 1.

    Parameters
    ──────────
    token_dim       : must match PerceptionTokens.token_dim
    max_obs_horizon : Embedding table size; must be ≥ obs_horizon at runtime

    Forward
    ───────
    tokens_per_step : list of T tensors [B, N_t, token_dim], oldest → newest
    Returns         : [B, T * N_t, token_dim]
    """

    def __init__(self, token_dim: int, max_obs_horizon: int = 2):
        super().__init__()
        self.embed = nn.Embedding(max_obs_horizon, token_dim)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, tok: torch.Tensor, i) -> torch.Tensor:
        idx = torch.tensor(i, dtype=torch.long, device=tok.device)
        # Shape shifts smoothly from [D] -> [1, 1, D] for broadcasting
        t_emb = self.embed(idx).unsqueeze(0).unsqueeze(0)
        return tok + t_emb

# ---------------------------------------------------------------------------
# Perception: RGBD backbone + Grasper + projections -> tokens
# ---------------------------------------------------------------------------

import math
from typing import List, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from model.deformable_detr.grasp_estimation_transformer import build_grasper
from model.deformable_detr.backbone import DepthEncoder, DepthEncodingConfig
from model.deformable_detr.util.misc import (
    nested_tensor_from_tensor_list,
    is_main_process,
)


class PerceptionTokens(nn.Module):
    def __init__(
        self,
        args_backbone,
        encoder_type: str = "dino_vd_rgbd_det_obj_grasp",
        max_objects_pretrained: int = 30,
        diffusion_max_objs: int = 5,
        token_dim: int = 256,
        last_image_fmaps: int = 2,
        image_kernel: int = 3,
        robot_state_dim: int = 8,
        obs_horizon: int = 2,
        n_views: int = 2,
    ):
        super().__init__()

        self.is_resnet = False
        if 'dino' not in encoder_type:
            args_backbone.model_name, args_backbone.model_version = encoder_type.split('@')
            last_image_fmaps = 1
            self.is_resnet   = True
            image_kernel     = 1

        self.encoder_type             = encoder_type
        self.max_objects_pretrained   = max_objects_pretrained
        self.diffusion_max_objs       = diffusion_max_objs
        self.token_dim                = token_dim
        self.robot_state_dim          = robot_state_dim
        self.image_kernel             = image_kernel
        self.image_stride             = image_kernel
        self._printed_token_summary        = False
        self._printed_observation_summary  = False
        self.n_views = n_views

        # ── 1) Build DINO(+VD) encoder ────────────────────────────────────
        encoder, encoder_channels, _ = build_vd_encoder(
            args_backbone,
            return_layers=True,
            cat_outs=args_backbone.cat_outs,
            resize_layers=False,
            vit_return_layers=[2, 5, 8, 11],
            with_generator=False,
            norm_return_layers=args_backbone.norm_return_layers,
            norm_return_layers_rgb=args_backbone.norm_return_layers_rgb,
            upsample_outs=args_backbone.resize_attention_maps,
        )

        # ── 2) Image feature projection ───────────────────────────────────
        self.last_image_fmaps        = last_image_fmaps
        self.with_depth = True if 'rgbd' in encoder_type.lower() else False
        
        self.encoder_dim = int(encoder_channels[-1])
        if self.with_depth:
            self.encoder_dim = self.encoder_dim // 2




        self.image_token_projection  = nn.ModuleList([
            nn.Conv2d(
                self.encoder_dim, self.token_dim,
                kernel_size=self.image_kernel,
                stride=self.image_stride,
                padding=0,
            )
            for _ in range(last_image_fmaps)
        ])
        

        for img_proj in self.image_token_projection:
            nn.init.xavier_uniform_(img_proj.weight, gain=1.0)
            if img_proj.bias is not None:
                nn.init.zeros_(img_proj.bias)



        self.image_token_projection_depth = None
        if self.with_depth:
            self.image_token_projection_depth  = nn.ModuleList([
                nn.Conv2d(
                    self.encoder_dim, self.token_dim,
                    kernel_size=self.image_kernel,
                    stride=self.image_stride,
                    padding=0,
                    )
                for _ in range(last_image_fmaps)
            ])
                
            for img_proj in self.image_token_projection_depth:
                nn.init.xavier_uniform_(img_proj.weight, gain=1.0)
                if img_proj.bias is not None:
                    nn.init.zeros_(img_proj.bias)



        # ── 3) Optional Grasper wrapper ───────────────────────────────────
        self.grasper: Optional[nn.Module] = None
        self.det_cls_token_proj: Optional[nn.Linear] = None
        self.det_boxlogit_proj:  Optional[nn.Linear] = None
        self.obj_attr_proj:      Optional[nn.Linear] = None
        self.obj_shape_proj:     Optional[nn.Linear] = None
        self.grasp_pose_proj:    Optional[nn.Linear] = None
        self.encoder:            Optional[nn.Module] = None

        if "det" in encoder_type:
            self.grasper       = build_grasper(args_backbone, encoder)
            self.det_hidden_dim = self.grasper.hidden_dim_det
            self.obj_hidden_dim = self.grasper.hidden_dim_obj
            self.num_grasp_poses = args_backbone.num_grasp_poses

            det_cls_dim = 768
            self.det_cls_proj   = nn.Linear(self.det_hidden_dim + det_cls_dim,
                                            self.token_dim, bias=False)
            self.det_box_proj   = nn.Linear(self.det_hidden_dim + 4,
                                            self.token_dim, bias=False)
            self.det_logit_proj = nn.Linear(self.det_hidden_dim + 2,
                                            self.token_dim, bias=False)
        else:
            encoder, is_rgbd = build_backbone(args_backbone, encoder)
            self.encoder         = encoder
            self.det_hidden_dim  = 0
            self.obj_hidden_dim  = 0
            self.num_grasp_poses = getattr(args_backbone, "num_grasp_poses", 0)

        if "obj" in encoder_type and "det" in encoder_type:
            C_obj = self.obj_hidden_dim
            self.obj_pos_proj   = nn.Linear(C_obj + 3, self.token_dim, bias=False)
            self.obj_vis_proj   = nn.Linear(C_obj + 1, self.token_dim, bias=False)
            self.obj_scale_proj = nn.Linear(C_obj + 1, self.token_dim, bias=False)
            self.obj_vol_proj   = nn.Linear(C_obj + 1, self.token_dim, bias=False)
            self.obj_shape_projs = nn.ModuleList(
                [nn.Linear(C_obj, self.token_dim, bias=False) for _ in range(3)]
            )
        else:
            self.obj_pos_proj    = self.obj_vis_proj  = None
            self.obj_scale_proj  = self.obj_vol_proj  = None
            self.obj_shape_projs = None

        if "grasp" in encoder_type and "obj" in encoder_type and "det" in encoder_type:
            C_obj = self.obj_hidden_dim
            self.grasp_pos_proj  = nn.Linear(C_obj + 3, self.token_dim, bias=False)
            self.grasp_rot_proj  = nn.Linear(C_obj + 4, self.token_dim, bias=False)
            self.grasp_conf_proj = nn.Linear(C_obj + 1, self.token_dim, bias=False)
        else:
            self.grasp_pos_proj = self.grasp_rot_proj = self.grasp_conf_proj = None

        # ── Robot-state encoding ──────────────────────────────────────────
        self.robot_state_pe   = ScalarPositionalEncoding(
            in_dim=robot_state_dim, num_frequencies=32
        )
        self.robot_state_proj = nn.Linear(
            self.robot_state_pe.out_dim, self.token_dim, bias=False
        )
        nn.init.xavier_uniform_(self.robot_state_proj.weight, gain=1.0)

        # ── Temporal positional encoding ───────────────────────
        # Adds a distinct embedding to each time-slice so the Transformer
        # can distinguish 'now' from 'obs_horizon-1 steps ago'.
        self.temporal_pe = TemporalPositionalEncoding(
            token_dim=token_dim, max_obs_horizon=obs_horizon # 2 vieww
        )

    # ------------------------------------------------------------------
    # Grasper-based helpers (unchanged)
    # ------------------------------------------------------------------
    def _run_grasper(self, samples) -> Dict:
        assert self.grasper is not None
        outs, _ = self.grasper(samples, targets=None)
        return outs

    def _image_tokens_from_joiner_backbone(self, features) -> torch.Tensor:
        assert isinstance(features, list) and len(features) > 0
        image_tokens = []
        depth_tokens = []
        for l in range(self.last_image_fmaps):
            feat = features[-(l + 1)]
            src, mask, _ = feat.decompose()
            if self.is_resnet:
                src = src.flatten(-2).unsqueeze(-1)
                

            if self.with_depth:
                x_enc = self.image_token_projection[l](src[:, :self.encoder_dim])
                xd_enc = self.image_token_projection_depth[l](src[:, self.encoder_dim:])
                image_tokens.append(x_enc.unsqueeze(-1))
                depth_tokens.append(xd_enc.unsqueeze(-1))

            else:
                x_enc = self.image_token_projection[l](src)
                image_tokens.append(x_enc.unsqueeze(-1))
        
        
        image_tokens = torch.cat(image_tokens, dim=-1)
        image_tokens = torch.flatten(image_tokens, 2)
        image_tokens = image_tokens.transpose(1, 2)
        if self.with_depth:
            
            depth_tokens = torch.cat(depth_tokens, dim=-1)
            depth_tokens = torch.flatten(depth_tokens, 2)
            depth_tokens = depth_tokens.transpose(1, 2)


        return image_tokens, depth_tokens

    def _object_tokens_from_grasper(self, outs: Dict) -> torch.Tensor:
        B              = outs["pred_boxes"].shape[0]
        device         = outs["pred_boxes"].device
        det_indices    = outs["obj_index"][:, :min(outs["obj_index"].shape[1],
                                                   self.diffusion_max_objs)]
        K_sel          = det_indices.shape[1]

        def gather_det(x: torch.Tensor) -> torch.Tensor:
            idx = det_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            return torch.gather(x, 1, idx)

        det_hidden  = outs.get("det_hidden", None)
        obj_hidden_all  = outs.get("obj_hidden", None)
        pose_hidden_all = outs.get("pose_hidden", None)

        tokens_list = []

        # Detection tokens
        det_cls_sel   = gather_det(outs["det_cls_token"])
        boxes_sel     = gather_det(outs["pred_boxes"])
        logits_sel    = gather_det(outs["pred_logits"])

        if det_hidden is not None:
            det_hidden_sel = gather_det(det_hidden)
            det_cls_in     = torch.cat([det_hidden_sel, det_cls_sel], dim=-1)
            det_box_in     = torch.cat([det_hidden_sel, boxes_sel],   dim=-1)
            det_logit_in   = torch.cat([det_hidden_sel, logits_sel],  dim=-1)
        else:
            det_cls_in, det_box_in, det_logit_in = det_cls_sel, boxes_sel, logits_sel

        tokens_list.append(self.det_cls_proj(det_cls_in).unsqueeze(2))
        tokens_list.append(self.det_box_proj(det_box_in).unsqueeze(2))
        tokens_list.append(self.det_logit_proj(det_logit_in).unsqueeze(2))

        # Object tokens
        has_obj = ("obj" in self.encoder_type) and (obj_hidden_all is not None)
        if has_obj:
            def _last(x):
                return x[-1] if x.dim() == 4 else x

            obj_pos_sel   = _last(outs["obj_positions"])[:, :K_sel]
            obj_vis_sel   = _last(outs["obj_vis"])[:, :K_sel]
            obj_scale_sel = _last(outs["obj_scale"])[:, :K_sel]
            obj_vol_sel   = _last(outs["obj_vol"])[:, :K_sel]

            obj_hidden_sel = obj_hidden_all[:, :K_sel]
            pos_h   = obj_hidden_sel[:, :, 0, :]
            vis_h   = obj_hidden_sel[:, :, 1, :]
            scale_h = obj_hidden_sel[:, :, 2, :]
            shape_h = obj_hidden_sel[:, :, 3:6, :]
            vol_h   = obj_hidden_sel[:, :, 6, :]

            tokens_list.append(
                self.obj_pos_proj(torch.cat([pos_h, obj_pos_sel], -1)).unsqueeze(2))
            tokens_list.append(
                self.obj_vis_proj(torch.cat([vis_h, obj_vis_sel], -1)).unsqueeze(2))
            tokens_list.append(
                self.obj_scale_proj(torch.cat([scale_h, obj_scale_sel], -1)).unsqueeze(2))
            tokens_list.append(
                self.obj_vol_proj(torch.cat([vol_h, obj_vol_sel], -1)).unsqueeze(2))
            for i in range(3):
                tokens_list.append(
                    self.obj_shape_projs[i](shape_h[:, :, i, :]).unsqueeze(2))

        # Grasp tokens
        has_grasp = ("grasp" in self.encoder_type) and (pose_hidden_all is not None)
        if has_grasp and self.num_grasp_poses > 0:
            pose_preds = outs["pose_preds"]
            if pose_preds.dim() == 5:
                pose_preds = pose_preds[-1]
            pose_sel        = pose_preds[:, :K_sel]
            pose_hidden_sel = pose_hidden_all[:, :K_sel]
            B_, K_, P, D8   = pose_sel.shape

            def _reshape_proj(proj, inp):
                B_, K_, P_, Din = inp.shape
                out = proj(inp.view(B_ * K_ * P_, Din))
                return out.view(B_, K_, P_, self.token_dim)

            tokens_list.append(
                _reshape_proj(self.grasp_pos_proj,
                              torch.cat([pose_hidden_sel, pose_sel[..., :3]], -1)))
            tokens_list.append(
                _reshape_proj(self.grasp_rot_proj,
                              torch.cat([pose_hidden_sel, pose_sel[..., 3:7]], -1)))
            tokens_list.append(
                _reshape_proj(self.grasp_conf_proj,
                              torch.cat([pose_hidden_sel, pose_sel[..., 7:]], -1)))

        if not tokens_list:
            return torch.zeros(B, 0, self.token_dim, device=device)

        tokens_per_det = torch.cat(tokens_list, dim=2)   # [B,K_sel,n_types,D]
        B_, K_, n_types, D = tokens_per_det.shape
        return tokens_per_det.view(B_, K_ * n_types, D)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward_single_frame(
        self,
        rgb:         torch.Tensor,
        depth:       Optional[torch.Tensor],
        robot_state: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> torch.Tensor:
        rgb_list   = list(rgb)
        depth_list = list(depth) if depth is not None else None
        samples    = nested_tensor_from_tensor_list(rgb_list, depth_list)

        if self.grasper is not None:
            outs       = self._run_grasper(samples)
            img_tokens, depth_tokens = self._image_tokens_from_joiner_backbone(outs['features'])
            obj_tokens = self._object_tokens_from_grasper(outs)
        else:
            features, _, _ = self.encoder(samples)
            img_tokens, depth_tokens = self._image_tokens_from_joiner_backbone(features)
            obj_tokens     = torch.zeros(
                img_tokens.shape[0], 0, self.token_dim, device=img_tokens.device
            )

        if self.with_depth:
            tokens = torch.cat([img_tokens, depth_tokens, obj_tokens], dim=1)   # [B, N_img+N_obj, D]
        else:
            tokens = torch.cat([img_tokens, obj_tokens], dim=1)   # [B, N_img+N_obj, D]

        if robot_state is not None:
            rs_enc = self.robot_state_pe(robot_state)
            rs_tok = self.robot_state_proj(rs_enc).unsqueeze(1)
            tokens = torch.cat([tokens, rs_tok], dim=1)

        if (not self._printed_token_summary) and is_main_process():
            print("\n[PerceptionTokens] token summary (single frame):")
            print(f"  image tokens   : {img_tokens.shape[1]}")
            if self.with_depth:
                print(f"  depth tokens   : {depth_tokens.shape[1]}")
            print(f"  obj/det tokens : {obj_tokens.shape[1]}")
            if robot_state is not None:
                print("  robot tokens   : 1")
            print(f"  total tokens   : {tokens.shape[1]}")
            print(f"  token_dim      : {self.token_dim}")
            self._printed_token_summary = True

        if not return_info:
            return tokens

        N_img   = int(img_tokens.shape[1])
        if self.with_depth:
            N_depth   = int(depth_tokens.shape[1])
        else:
            N_depth = 0
        N_obj   = int(obj_tokens.shape[1])
        N_robot = 1 if robot_state is not None else 0
        has_det   = "det"   in self.encoder_type
        has_obj   = "obj"   in self.encoder_type and has_det
        has_grasp = "grasp" in self.encoder_type and has_obj

        info = {
            "N_img": N_img, "N_depth": N_depth, "N_obj": N_obj, "N_robot": N_robot,
            "last_image_fmaps": int(self.last_image_fmaps),
            "has_det": has_det, "has_obj": has_obj, "has_grasp": has_grasp,
        }

        if has_det and N_obj > 0:
            n_types = 3
            if has_obj:
                n_types += 7
            if has_grasp and self.num_grasp_poses > 0:
                n_types += 3 * self.num_grasp_poses
            info["n_types"] = n_types
            info["K_sel"]   = N_obj // n_types
        else:
            info["n_types"] = 0
            info["K_sel"]   = 0

        return tokens, info

    def forward(
        self,
        rgb_seq:         torch.Tensor,
        depth_seq:       Optional[torch.Tensor] = None,
        robot_state_seq: Optional[torch.Tensor] = None,
        return_info:     bool = False,
    ) -> torch.Tensor:
        """Returns [B, N_obs, token_dim] with temporal PE applied."""
        assert rgb_seq.dim() == 5, f"Expected [B,T,3,H,W], got {rgb_seq.shape}"
        B, T, C, H, W = rgb_seq.shape

        tokens_all: List[torch.Tensor] = []
        info_list:  List[Dict]         = []

        for t in range(T):
            rgb_t         = rgb_seq[:, t]
            depth_t       = depth_seq[:, t]       if depth_seq       is not None else None
            robot_state_t = robot_state_seq[:, t] if robot_state_seq is not None else None

            if return_info:
                tok_t, info_t = self.forward_single_frame(
                    rgb_t, depth_t, robot_state_t, return_info=True
                )
                info_list.append(info_t)
            else:
                tok_t = self.forward_single_frame(rgb_t, depth_t, robot_state_t)

                
            # ── Apply temporal PE before concatenation ────────────
            # temporal_pe adds a distinct embedding to each time-slice so the            
            tok_t = self.temporal_pe(tok_t, t // self.n_views)   # [B, T*N_per_frame, D]
            tokens_all.append(tok_t)

        # Transformer can distinguish 'now' from 'obs_horizon-1 steps ago'.
        tokens = torch.cat(tokens_all, dim=1) 

        if (not self._printed_observation_summary) and is_main_process():
            print("\n[PerceptionTokens] observation summary:")
            print(f"  image shape      : {rgb_seq.shape}")
            print(f"  total tokens     : {tokens.shape[1]}")
            print(f"  with depth       : {depth_seq is not None}")
            print(f"  with robot state : {robot_state_seq is not None}")
            print(f"  temporal_pe      : enabled (obs_horizon={T//self.n_views} and n_views={self.n_views})")
            self._printed_observation_summary = True

        if return_info:
            return tokens, info_list
        return tokens


# ---------------------------------------------------------------------------
# Transformer Decoder with cross-attention capture
# ---------------------------------------------------------------------------

class TransformerDecoderLayerWithAttn(nn.Module):
    """
    Standard post-norm Transformer decoder layer (consistent with encoder).

    Self-attention  : action tokens attend to each other.
    Cross-attention : action tokens (q) attend to condition memory (k, v).
                      Cross-attention weights [B,H,T_act,N_ctx] are returned
                      when return_attn=True; these feed get_attention_stats().
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1,
                 activation="gelu"):
        super().__init__()
        # Self-attention on action tokens
        self.self_attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Cross-attention: actions → condition memory
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout  = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model)

        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.norm3    = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu
        self.register_buffer('_causal_mask', torch.zeros(1, 1, dtype=torch.bool), persistent=False)


    def forward(self, tgt, memory, tgt_key_padding_mask=None,
            memory_key_padding_mask=None, return_attn=False):

        T_act = tgt.shape[1]
        if self._causal_mask.shape[0] != T_act:
            self._causal_mask = torch.triu(
                torch.ones(T_act, T_act, dtype=torch.bool, device=tgt.device), diagonal=1
            )
        causal_mask = self._causal_mask.to(tgt.device)

        # 1) Self-attention — pre-norm 
        tgt_n = self.norm1(tgt)
        sa_out, _ = self.self_attn(
            tgt_n, tgt_n, tgt_n,
            attn_mask=causal_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.dropout1(sa_out)          # clean residual

        # 2) Cross-attention — pre-norm 
        tgt_n = self.norm2(tgt)                    # normalize BEFORE attention
        ca_out, ca_weights = self.cross_attn(
            tgt_n, memory, memory,                 # normed query, raw memory K/V
            key_padding_mask=memory_key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        tgt = tgt + self.dropout2(ca_out)          # clean residual

        # 3) FFN — pre-norm 
        tgt_n = self.norm3(tgt)                    # normalize BEFORE FFN
        ff    = self.linear2(self.dropout(self.activation(self.linear1(tgt_n))))
        tgt   = tgt + self.dropout3(ff)            # clean residual

        return tgt, ca_weights


class TransformerDecoderWithAttn(nn.Module):
    def __init__(self, layer: TransformerDecoderLayerWithAttn, num_layers: int):
        super().__init__()
        self.layers     = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm       = nn.LayerNorm(layer.self_attn.embed_dim)

    def forward(
        self,
        tgt:    torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask:    Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:

        cross_attn_list: Optional[List[torch.Tensor]] = [] if return_attn else None
        output = tgt

        for layer in self.layers:
            output, ca_w = layer(
                output, memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                return_attn=return_attn,
            )
            if return_attn and ca_w is not None:
                cross_attn_list.append(ca_w)   # [B,H,T_act,N_ctx]

        output = self.norm(output)
        return output, cross_attn_list


# ---------------------------------------------------------------------------
# Decoder Diffusion Head (replaces encoder-only)
# ---------------------------------------------------------------------------

class TransformerDiffusionHead(nn.Module):
    """
    Decoder Transformer noise predictor.

    Architecture change from original
    ──────────────────────────────────

               TransformerDecoder  uses noisy action tokens as queries,
                                   cross-attends to memory (no leakage)

    Note on layer count
    ───────────────────
    n_layers is used for BOTH encoder and decoder.  The total parameter count
    is higher than the original encoder-only design.  Reduce --tf_layers by
    half if you need an iso-parameter comparison.

    Inputs
    ──────
    x_t         : [B, T, A]       noisy actions at diffusion step t
    cond_tokens : [B, N_ctx, D]   perception tokens (already in model_dim)
    t           : [B]             diffusion timestep indices

    Output
    ──────
    eps_pred    : [B, T, A]       predicted noise ε_θ
    """

    def __init__(
        self,
        act_dim: int,
        horizon: int,
        model_dim: int = 256,
        n_layers: int  = 8,
        n_heads: int   = 8,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.act_dim   = int(act_dim)
        self.horizon   = int(horizon)
        self.model_dim = int(model_dim)

        if dim_feedforward is None:
            dim_feedforward = 4 * self.model_dim

        # ── Action input projection ────────────────────────────────────────
        self.action_in = nn.Linear(self.act_dim, self.model_dim)

        # ── Diffusion timestep embedding ───────────────────────────────────
        self.time_embed = SinusoidalTimeEmbedding(self.model_dim)

        # ── Learned positional encoding over the action horizon ───────────
        # +8 headroom in case horizon is slightly exceeded at inference
        # action_pos needs +1 for time token at position 0
        self.action_pos = nn.Embedding(horizon + 8 + 1, self.model_dim)  # +1
        nn.init.trunc_normal_(self.action_pos.weight, std=0.02)

        # ── Action decoder ────────────────────────────────────────────────
        # Action queries self-attend, then cross-attend to condition memory.
        dec_layer = TransformerDecoderLayerWithAttn(
            d_model=self.model_dim, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu",
        )
        self.action_decoder = TransformerDecoderWithAttn(
            dec_layer, num_layers=n_layers
        )

        # ── Output projection ──────────────────────────────────────────────
        self.action_out = nn.Linear(self.model_dim, self.act_dim)


        
    def forward(self, x_t, cond_tokens, t, return_attn=False):
        B, T, A = x_t.shape
        

        # 1. Project actions + horizon PE (positions 1..T, leaving 0 for t_tok)
        act = self.action_in(x_t)                                   # [B, T, D]
        pos = torch.arange(1, T + 1, device=x_t.device)            # 1-indexed ← changed
        act = act + self.action_pos(pos).unsqueeze(0)               # [B, T, D]

        # 2. Prepend time token at position 0
        t_tok = self.time_embed(t).unsqueeze(1)                     # [B, 1, D]
        t_tok = t_tok + self.action_pos(
            torch.zeros(1, dtype=torch.long, device=x_t.device)    # position 0
        ).unsqueeze(0)
        act = torch.cat([t_tok, act], dim=1)                        # [B, T+1, D]

        # 3. Decode (causal mask is auto-resized to T+1 — no change needed)
        out, cross_attn_list = self.action_decoder(
            act, cond_tokens, return_attn=return_attn
        )                                                           # [B, T+1, D]

        # 4. Skip time token output at position 0
        eps_pred = self.action_out(out[:, 1:, :])                  # [B, T, A]

        if return_attn:
            # cross_attn_list: list of [B,H,T_act,N_ctx] (one per decoder layer)
            # cond_len returned for compatibility with get_attention_stats
            return eps_pred, cross_attn_list, cond_tokens.shape[1] 
        return eps_pred


# ---------------------------------------------------------------------------
# ScalarPositionalEncoding (unchanged)
# ---------------------------------------------------------------------------

class ScalarPositionalEncoding(nn.Module):
    """
    Sin/cos frequency encoding for robot joint values.
    freq_bands = 2^k so that small gripper-meter values get the same
    frequency resolution as radian joint angles.
    """

    def __init__(self, in_dim: int, num_frequencies: int = 16):
        super().__init__()
        self.in_dim          = in_dim
        self.num_frequencies = num_frequencies
        freq_bands = math.pi *(2.0 ** torch.arange(num_frequencies))
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.out_dim = in_dim * num_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim] → [B, in_dim * num_frequencies * 2]"""
        if x.dim() != 2:
            x = x.view(x.shape[0], -1)
        x      = x.unsqueeze(-1)                              # [B, in_dim, 1]
        angles = x * self.freq_bands.view(1, 1, -1)          # [B, in_dim, F]
        pe     = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return pe.view(pe.shape[0], -1)                       # [B, in_dim*2F]


# ---------------------------------------------------------------------------
# Full policy: perception + diffusion
# ---------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """
    Diffusion policy:
      PerceptionTokens  : RGBD backbone (+ optional Grasper) → obs tokens
      GaussianDiffusion : cosine-schedule forward process
      TransformerDiffusionHead : encoder-decoder noise predictor
    """

    def __init__(
        self,
        args_backbone,
        encoder_type: str,
        diffusion_max_objs: int,
        token_dim: int,
        act_dim: int,
        robot_state_dim: int,
        horizon: int,
        obs_horizon: int,
        n_views: int,
        action_min,
        action_max,
        model_dim: int = 256,
        n_layers: int  = 8,
        n_heads: int   = 8,
        diffusion_steps_train: int = 100,
        diffusion_steps_eval:  int = 10,
        image_kernel: int = 1,
    ):
        super().__init__()

        self.perception = PerceptionTokens(
            args_backbone=args_backbone,
            encoder_type=encoder_type,
            max_objects_pretrained=args_backbone.max_objects,
            diffusion_max_objs=diffusion_max_objs,
            token_dim=token_dim,
            image_kernel=image_kernel,
            robot_state_dim=robot_state_dim,
            obs_horizon=obs_horizon,
            n_views=n_views,
        )

        self.diffusion = GaussianDiffusion(
            num_train_steps=diffusion_steps_train,
            num_eval_steps=diffusion_steps_eval,
        )

        self.head = TransformerDiffusionHead(
            act_dim=act_dim,
            horizon=horizon,
            model_dim=model_dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )

        self.perc_to_model = None
        if token_dim != model_dim:
            self.perc_to_model = nn.Linear(token_dim, model_dim)

        a_min = torch.as_tensor(action_min, dtype=torch.float32)
        a_max = torch.as_tensor(action_max, dtype=torch.float32)
        self.register_buffer("action_min", a_min.view(1, 1, -1))
        self.register_buffer("action_max", a_max.view(1, 1, -1))
            

    def _normalize(self, actions):
        denom = (self.action_max - self.action_min).clamp(min=1e-6)
        return (actions - self.action_min) / denom * 2.0 - 1.0

    def _unnormalize(self, ndata):
        denom = (self.action_max - self.action_min).clamp(min=1e-6)
        return (ndata + 1.0) / 2.0 * denom + self.action_min

    def _print_param_summary(self):
        enc = self.perception.encoder
        gr  = self.perception.grasper
        gr_total,  gr_train  = _count_params(gr)  if isinstance(gr,  nn.Module) else (0, 0)
        enc_total, enc_train = _count_params(enc) if isinstance(enc, nn.Module) else (0, 0)

        if enc is None:
            enc = self.perception.grasper.backbone
            enc_total, enc_train = _count_params(enc)
            gr_total -= enc_total
            gr_train -= enc_train

        proj_params = [p for n, p in self.perception.named_parameters()
                       if "encoder" not in n and "grasper" not in n]
        proj_mod       = nn.Module()
        proj_mod._params = nn.ParameterList(proj_params)
        proj_total, proj_train = _count_params(proj_mod)

        head_total, head_train = _count_params(self.head)
        pol_total,  pol_train  = _count_params(self)

        if is_main_process():
            print("\n[DiffusionPolicy] Parameter breakdown:")
            print(f"  Backbone encoder : total={_fmt_m(enc_total)}, trainable={_fmt_m(enc_train)}")
            print(f"  Grasper          : total={_fmt_m(gr_total)},  trainable={_fmt_m(gr_train)}")
            print(f"  Perception proj  : total={_fmt_m(proj_total)},trainable={_fmt_m(proj_train)}")
            print(f"  Diffusion head   : total={_fmt_m(head_total)},trainable={_fmt_m(head_train)}")
            print(f"  TOTAL policy     : total={_fmt_m(pol_total)}, trainable={_fmt_m(pol_train)}")

    # ── Training forward ──────────────────────────────────────────────────
    def forward(
        self,
        rgb_seq:         torch.Tensor,   # [B,Hobs,3,H,W]
        depth_seq:       torch.Tensor,   # [B,Hobs,1,H,W]
        robot_state_seq: torch.Tensor,   # [B,Hobs,Dq]
        actions:         torch.Tensor,   # [B,T,A]
    ) -> torch.Tensor:
        B, T, A = actions.shape
        device  = actions.device

        with torch.set_grad_enabled(True):
            tokens = self.perception(rgb_seq, depth_seq, robot_state_seq)
        if tokens is None or tokens.numel() == 0:
            tokens = torch.zeros(B, 1, self.head.model_dim, device=device)
        if self.perc_to_model is not None:
            tokens = self.perc_to_model(tokens)

        actions_norm = self._normalize(actions)

        t     = torch.randint(0, self.diffusion.num_train_steps, (B,),
                              device=device, dtype=torch.long)
        noise = torch.randn_like(actions_norm)
        x_t   = self.diffusion.q_sample(actions_norm, t, noise)

        eps_pred = self.head(x_t, tokens, t)
        return F.mse_loss(eps_pred, noise)

    # ── Sampling ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_actions(
        self,
        rgb_seq:         torch.Tensor,
        depth_seq:       torch.Tensor,
        robot_state_seq: torch.Tensor,
        horizon:         int,
    ) -> torch.Tensor:
        B      = rgb_seq.shape[0]
        device = rgb_seq.device

        tokens = self.perception(rgb_seq, depth_seq, robot_state_seq)
        if tokens is None or tokens.numel() == 0:
            tokens = torch.zeros(B, 1, self.head.model_dim, device=device)
        if self.perc_to_model is not None:
            tokens = self.perc_to_model(tokens)

        x_T    = torch.randn(B, horizon, self.head.act_dim, device=device)
        x0_norm = self.diffusion.ddim_sample(self.head, x_T, tokens)
        return self._unnormalize(x0_norm)

    # ── Attention stats ────────────────────────────────────────────────────
    @torch.no_grad()
    def get_attention_stats(
        self,
        rgb_seq:         torch.Tensor,
        depth_seq:       torch.Tensor,
        robot_state_seq: torch.Tensor,
        actions:         torch.Tensor,
        only_action_queries: bool = True,   # kept for API compat; always True now
    ) -> Dict[str, float]:
        """
        Returns mean cross-attention fractions per token group, averaged over
        all decoder layers.

        With the encoder-decoder design the decoder's cross-attention is
        action→condition by construction, so only_action_queries is always True
        and no start_idx splitting is required.
        """
        self.eval()
        device  = rgb_seq.device
        B       = rgb_seq.shape[0]
        T_act   = actions.shape[1]

        tokens, frame_infos = self.perception(
            rgb_seq, depth_seq, robot_state_seq, return_info=True
        )
        if self.perc_to_model is not None:
            tokens = self.perc_to_model(tokens)
        N_ctx = tokens.shape[1]

        t = torch.zeros(B, device=device, dtype=torch.long)
        eps_pred, cross_attn_list, cond_len = self.head(
            actions, tokens, t, return_attn=True
        )
        assert cond_len == N_ctx, f"cond_len={cond_len} != N_ctx={N_ctx}"

        if not cross_attn_list:
            return {}

        groups = self._build_token_groups(frame_infos, N_ctx)
        return self._aggregate_attention(cross_attn_list, groups, N_ctx)

    def _build_token_groups(
        self, frame_infos: List[Dict], cond_len: int
    ) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        offset = 0

        for info in frame_infos:
            N_img   = info["N_img"]
            N_depth   = info["N_depth"]
            N_obj   = info["N_obj"]
            N_robot = info.get("N_robot", 0)
            L       = info["last_image_fmaps"]

            if N_img > 0 and L > 0:
                for i in range(N_img):
                    groups.setdefault("img", []).append(offset + i)
            
            if N_depth > 0 and L > 0:
                for i in range(N_depth):
                    groups.setdefault("depth", []).append(offset + N_img + i)

            has_det   = info["has_det"]
            has_obj   = info["has_obj"]
            has_grasp = info["has_grasp"]
            n_types   = info.get("n_types", 0)
            K_sel     = info.get("K_sel", 0)

            if has_det and N_obj > 0 and n_types > 0 and K_sel > 0:
                for j in range(K_sel):
                    base = N_img + N_depth + j * n_types
                    groups.setdefault("det", []).extend(
                        [offset + base, offset + base + 1, offset + base + 2]
                    )
                    if has_obj:
                        for k, name in enumerate(
                            ["obj_pos", "obj_vis", "obj_scale", "obj_vol", "obj_shape", "obj_shape", "obj_shape"]
                        ):
                            groups.setdefault(name, []).append(offset + base + 3 + k)
                    if has_grasp:
                        for k in range(10, n_types):
                            groups.setdefault("grasp", []).append(offset + base + k)

            if N_robot > 0:
                groups.setdefault("robot_state", []).append(
                    offset + N_img + N_depth + N_obj
                )

            offset += N_img + N_depth + N_obj + N_robot

        assert offset == cond_len, (
            f"Token group construction mismatch: offset={offset}, cond_len={cond_len}"
        )
        return groups

    def _aggregate_attention(
        self,
        cross_attn_list: List[torch.Tensor],   # list of [B,H,T_act,N_ctx]
        groups:          Dict[str, List[int]],
        cond_len:        int,
        only_action_queries: bool = True,       # kept for API compat
    ) -> Dict[str, float]:
        """
        For each decoder layer compute the fraction of cross-attention directed
        at each token group, then average over layers.

        cross_attn_list[l] shape: [B, H, T_act, N_ctx]
          - rows  = action queries (always all of them in cross-attn)
          - cols  = condition keys (indexed by groups)
        """
        if not cross_attn_list:
            return {}

        stats: Dict[str, List[float]] = {}
        eps       = 1e-8
        obj_total: List[float] = []

        for attn in cross_attn_list:
            if attn is None:
                continue
            B, H, T_act, N_ctx_k = attn.shape

            total_cond = attn.sum(dim=-1)   # [B,H,T_act]

            for g_name, idx_list in groups.items():
                if not idx_list:
                    continue
                idx = torch.tensor(idx_list, device=attn.device, dtype=torch.long)
                idx = idx[idx < N_ctx_k]   # safety clamp
                if idx.numel() == 0:
                    continue
                attn_g    = attn[:, :, :, idx]             # [B,H,T_act,Ng]
                sum_g     = attn_g.sum(dim=-1)             # [B,H,T_act]
                frac      = sum_g / (total_cond + eps)     # [B,H,T_act]
                stats.setdefault(g_name, []).append(float(frac.mean().item()))

        for g_name in list(stats.keys()):
            vals          = stats[g_name]
            stats[g_name] = sum(vals) / len(vals)
            if "obj_" in g_name:
                obj_total.append(stats[g_name])

        if obj_total:
            stats["obj_sum"] = sum(obj_total)

        return stats