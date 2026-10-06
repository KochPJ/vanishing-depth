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
        """
        t: [B] int or float in [0, num_steps-1]
        returns [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device)
        )
        args = t.float().unsqueeze(1) / freqs.unsqueeze(0)  # [B,half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # [B,dim]


# ---------------------------------------------------------------------------
# Gaussian diffusion (DDPM / DDIM) on action sequences
# ---------------------------------------------------------------------------
def _cosine_beta_schedule(T, s=0.008):
    t = torch.linspace(0, T, T + 1) / T
    alphas_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(max=0.999)



class GaussianDiffusion(nn.Module):
    """
    Simple Gaussian diffusion on action sequences.
    Implements q(x_t|x_0) and DDIM sampling.
    """


    def __init__(self, num_train_steps: int = 100, num_eval_steps: int = 10,
                 beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.num_eval_steps = num_eval_steps

        betas = torch.linspace(beta_start, beta_end, num_train_steps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        q(x_t | x_0)
        x0:   [B,T,A]
        t:    [B] int timesteps
        noise:[B,T,A] or None
        """
        if noise is None:
            noise = torch.randn_like(x0)
        device = x0.device
        t = t.to(device)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1).to(device)  # [B,1,1]
        sqrt_one_m = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1).to(device)

        return sqrt_alpha * x0 + sqrt_one_m * noise

    @torch.no_grad()
    def ddim_sample(self, model, x_T: torch.Tensor,
                    cond_tokens: torch.Tensor,
                    steps: Optional[int] = None) -> torch.Tensor:
        """
        Deterministic DDIM sampling from pure noise:
          model: eps = model(x_t, cond_tokens, t)
        x_T: [B,T,A]
        cond_tokens: [B, N_obs, D]
        """
        device = x_T.device
        if steps is None:
            steps = self.num_eval_steps

        # Choose timesteps
        step_indices = torch.linspace(
            0, self.num_train_steps - 1, steps,
            dtype=torch.long, device=device
        )
        step_indices = step_indices.flip(0)  # reverse

        x = x_T
        for i, t_idx in enumerate(step_indices):
            t = torch.full((x.shape[0],), int(t_idx.item()),
                           device=device, dtype=torch.long)
            eps = model(x, cond_tokens, t)

            alpha_t = self.alphas[t_idx].to(device)
            alpha_bar_t = self.alphas_cumprod[t_idx].to(device)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_1m_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

            # predicted x0
            x0_pred = (x - sqrt_1m_alpha_bar_t * eps) / sqrt_alpha_bar_t

            if i == steps - 1:
                x = x0_pred
            else:
                alpha_bar_prev = self.alphas_cumprod[step_indices[i + 1]].to(device)
                # Deterministic DDIM: no additional noise
                x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps
        return x


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

# Import your VD+DINO builder
# from model.vd_encoder import build_vd_encoder


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
        robot_state_dim: int = 8,  # k=3, stride=3 as you described
    ):
        super().__init__()

        self.is_resnet = False
        if 'dino' not in encoder_type:
            args_backbone.model_name, args_backbone.model_version =  encoder_type.split('@')
            last_image_fmaps = 1
            self.is_resnet = True
            image_kernel = 1

        self.encoder_type = encoder_type
        self.max_objects_pretrained = max_objects_pretrained
        self.diffusion_max_objs = diffusion_max_objs
        self.token_dim = token_dim
        self.robot_state_dim = robot_state_dim
        self.image_kernel = image_kernel
        self.image_stride = image_kernel
        self._printed_token_summary = False
        self._printed_observation_summary = False

        # ---------------------------------------------------------
        # 1) Build DINO(+VD) encoder; get encoder_dim
        # ---------------------------------------------------------


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
        # final feature channel dim (e.g. 768 for RGB, ~2*768 for RGBD)
        self.encoder_dim = int(encoder_channels[-1])

        # ---------------------------------------------------------
        # 3) Image feature projection (always used)
        #    Final feature map [B, encoder_dim, H, W] ->
        #    Conv(k=3,s=3) -> [B, token_dim, H', W'] -> flatten
        # ---------------------------------------------------------

        self.last_image_fmaps = last_image_fmaps
        

        self.image_token_projection = nn.ModuleList([nn.Conv2d(
            self.encoder_dim,
            self.token_dim,
            kernel_size=self.image_kernel,
            stride=self.image_stride,
            padding=0,
        ) for _ in range(last_image_fmaps)])
        for img_proj in self.image_token_projection:
            nn.init.xavier_uniform_(img_proj.weight, gain=1.0)
            if img_proj.bias is not None:
                nn.init.zeros_(img_proj.bias)

        # ---------------------------------------------------------
        # 4) Optional Grasper wrapper (DETR+obj+pose)
        # ---------------------------------------------------------
        self.grasper: Optional[nn.Module] = None

        # Object-related projection layers (defined in __init__)
        #  - det_cls_token_proj: det cls embedding -> token_dim
        #  - det_boxlogit_proj:  6 scalars (4 box + 2 logits) -> token_dim
        #  - obj_attr_proj:      6 scalars (pos3 + vis + scale + vol) -> token_dim
        #  - obj_shape_proj:     D*3 scalars (shape points) -> token_dim
        #  - grasp_pose_proj:    8 scalars (pos3 + rot4 + conf1) -> token_dim
        self.det_cls_token_proj: Optional[nn.Linear] = None
        self.det_boxlogit_proj: Optional[nn.Linear] = None
        self.obj_attr_proj: Optional[nn.Linear] = None
        self.obj_shape_proj: Optional[nn.Linear] = None
        self.grasp_pose_proj: Optional[nn.Linear] = None
        self.encoder: Optional[nn.Linear] = None

        if "det" in encoder_type:
            # Wrap encoder with Grasper (Joiner + transformer + heads)
            self.grasper = build_grasper(args_backbone, encoder)

            # cache hidden dims from transformer
            self.det_hidden_dim = self.grasper.hidden_dim_det   # d_model_det
            self.obj_hidden_dim = self.grasper.hidden_dim_obj   # d_model_obj
            self.num_grasp_poses = args_backbone.num_grasp_poses

            # Detection tokens:
            #  - DET-CLS   : [det_hidden ; det_cls_token]
            #  - DET-BOX   : [det_hidden ; bbox]
            #  - DET-LOGIT : [det_hidden ; logits]
            det_cls_dim = 768  # encoder_dim used for det_cls_token

            self.det_cls_proj = nn.Linear(
                self.det_hidden_dim + det_cls_dim,
                self.token_dim,
                bias=False,
            )
            self.det_box_proj = nn.Linear(
                self.det_hidden_dim + 4,  # 4 box coords
                self.token_dim,
                bias=False,
            )
            self.det_logit_proj = nn.Linear(
                self.det_hidden_dim + 2,  # 2 logits for foreground/background
                self.token_dim,
                bias=False,
            )
        else:
            encoder, is_rgbd = build_backbone(args_backbone, encoder)
            self.encoder = encoder
            self.det_hidden_dim = 0
            self.obj_hidden_dim = 0
            self.num_grasp_poses = getattr(args_backbone, "num_grasp_poses", 0)


        if "obj" in encoder_type and "det" in encoder_type:
            C_obj = self.obj_hidden_dim

            # Object tokens:
            #  - OBJ-POS   : [obj_hidden_pos ; pos(3)]
            #  - OBJ-VIS   : [obj_hidden_vis ; vis(1)]
            #  - OBJ-SCALE : [obj_hidden_scale ; scale(1)]
            #  - OBJ-VOL   : [obj_hidden_vol ; vol(1)]
            #  - OBJ-SHAPE1/2/3 : each from its own shape-query hidden vector

            self.obj_pos_proj   = nn.Linear(C_obj + 3, self.token_dim, bias=False)
            self.obj_vis_proj   = nn.Linear(C_obj + 1, self.token_dim, bias=False)
            self.obj_scale_proj = nn.Linear(C_obj + 1, self.token_dim, bias=False)
            self.obj_vol_proj   = nn.Linear(C_obj + 1, self.token_dim, bias=False)

            # Shape tokens use only hidden vectors of 3 "shape" queries
            self.obj_shape_projs = nn.ModuleList(
                [nn.Linear(C_obj, self.token_dim, bias=False) for _ in range(3)]
            )
        else:
            self.obj_pos_proj = self.obj_vis_proj = None
            self.obj_scale_proj = self.obj_vol_proj = None
            self.obj_shape_projs = None

        if "grasp" in encoder_type and "obj" in encoder_type and "det" in encoder_type:
            C_obj = self.obj_hidden_dim
            P = self.num_grasp_poses

            # Grasp tokens per pose:
            #  - GRASP-POS  : [pose_hidden ; pos(3)]
            #  - GRASP-ROT  : [pose_hidden ; quat(4)]
            #  - GRASP-CONF : [pose_hidden ; conf(1)]
            self.grasp_pos_proj  = nn.Linear(C_obj + 3, self.token_dim, bias=False)
            self.grasp_rot_proj  = nn.Linear(C_obj + 4, self.token_dim, bias=False)
            self.grasp_conf_proj = nn.Linear(C_obj + 1, self.token_dim, bias=False)
        else:
            self.grasp_pos_proj = self.grasp_rot_proj = self.grasp_conf_proj = None

        
        # --- Robot-state encoding (initialized lazily when first used) ---
        self.robot_state_pe = ScalarPositionalEncoding(in_dim=robot_state_dim, num_frequencies=32)
        self.robot_state_proj = nn.Linear(self.robot_state_pe.out_dim, self.token_dim, bias=False)
        nn.init.xavier_uniform_(self.robot_state_proj.weight, gain=1.0)


    # ------------------------------------------------------------------
    # Grasper-based path: image + object tokens
    # ------------------------------------------------------------------
    def _run_grasper(
        self,
        samples,
    ) -> Dict:
        assert self.grasper is not None, "_run_grasper called but grasper is None"
        outs, _ = self.grasper(samples, targets=None)
        return outs

    def _image_tokens_from_joiner_backbone(self, features) -> torch.Tensor:
        """
        Use Joiner backbone outputs from Grasper to build final feature map.

        features : list[NestedTensor], each has .tensors [B,C_l,H_l,W_l].

        output : image_tokens B, token_dim, H*W*last_image_fmaps

        """
        assert isinstance(features, list) and len(features) > 0

        ref_H, ref_W = features[0].tensors.shape[-2:]
        image_tokens = []
        for l in range(self.last_image_fmaps):
            feat = features[-(l+1)]
            src, mask, _ = feat.decompose()
            if self.is_resnet:
                src = src.flatten(-2)
                #src = src.mean(-1)
                #print('src1', src.shape)
                src = src.unsqueeze(-1)
                #print('src2', src.shape)
            
            x_enc = self.image_token_projection[l](src)
            #print('x_enc', x_enc.shape)
            
            image_tokens.append(x_enc.unsqueeze(-1))

        image_tokens = torch.cat(image_tokens, dim=-1)           # [B,token_dim,H,W,last_image_fmaps]
        image_tokens = torch.flatten(image_tokens, 2) # [B, token_dim, H*W*last_image_fmaps]
        image_tokens = image_tokens.transpose(1, 2)     # [B, N_tokens, D]
        return image_tokens


    def _object_tokens_from_grasper(self, outs: Dict) -> torch.Tensor:
        """
        Build object-related tokens for diffusion.

        Per object j (after top-K selection) we create:

          Detection (3 tokens):
            - DET-CLS   : [det_hidden_j ; det_cls_token_j]
            - DET-BOX   : [det_hidden_j ; bbox_j]
            - DET-LOGIT : [det_hidden_j ; logits_j]

          Object (7 tokens):
            - OBJ-POS   : [obj_hidden_pos_j ; pos(3)]
            - OBJ-VIS   : [obj_hidden_vis_j ; vis(1)]
            - OBJ-SCALE : [obj_hidden_scale_j ; scale(1)]
            - OBJ-VOL   : [obj_hidden_vol_j ; vol(1)]
            - OBJ-SHAPE1: from shape_hidden_1
            - OBJ-SHAPE2: from shape_hidden_2
            - OBJ-SHAPE3: from shape_hidden_3

          Grasp (3 * num_grasp_poses tokens):
            For each pose p in {0..P-1}:
              - GRASP-POS  : [pose_hidden_jp ; pos(3)]
              - GRASP-ROT  : [pose_hidden_jp ; quat(4)]
              - GRASP-CONF : [pose_hidden_jp ; conf(1)]

        So per object we have:
          n_types = 3 (det) + 7 (obj) + 3 * P (grasp) tokens.
        """
        B = outs["pred_boxes"].shape[0]
        device = outs["pred_boxes"].device

        # --- indices of selected detections in DETR space: [B,Kmax] ---
        det_indices_full = outs["obj_index"]           # [B,Kmax]
        Kmax = det_indices_full.shape[1]
        K_sel = min(Kmax, self.diffusion_max_objs)
        det_indices = det_indices_full[:, :K_sel]      # [B,K_sel]

        # Helper: gather along detection dimension Q using det_indices
        def gather_det(x: torch.Tensor) -> torch.Tensor:
            # x: [B,Q,...]  -> [B,K_sel,...]
            assert x.shape[0] == B
            idx = det_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # [B,K_sel,last]
            return torch.gather(x, 1, idx)

        # Optional hidden states
        det_hidden  = outs.get("det_hidden", None)            # [B,Q,C_det]
        obj_hidden_all  = outs.get("obj_hidden", None)        # [B,Kmax,Qobj,C_obj]
        pose_hidden_all = outs.get("pose_hidden", None)       # [B,Kmax,num_poses,C_obj]

        tokens_list = []

        # --------------------------------------------------------------
        # 1) Detection tokens
        # --------------------------------------------------------------
        det_cls = outs["det_cls_token"]              # [B,Q,dim_encoder]
        det_cls_sel = gather_det(det_cls)            # [B,K_sel,dim_encoder]
        boxes  = outs["pred_boxes"]                  # [B,Q,4]
        logits = outs["pred_logits"]                 # [B,Q,C]
        boxes_sel  = gather_det(boxes)               # [B,K_sel,4]
        logits_sel = gather_det(logits)              # [B,K_sel,2] (for C=2)

        if det_hidden is not None:
            det_hidden_sel = gather_det(det_hidden)  # [B,K_sel,C_det]

            det_cls_in   = torch.cat([det_hidden_sel, det_cls_sel], dim=-1)  # [B,K_sel,C_det+enc]
            det_box_in   = torch.cat([det_hidden_sel, boxes_sel],   dim=-1)  # [B,K_sel,C_det+4]
            det_logit_in = torch.cat([det_hidden_sel, logits_sel],  dim=-1)  # [B,K_sel,C_det+2]
        else:
            det_cls_in, det_box_in, det_logit_in = det_cls_sel, boxes_sel, logits_sel

        det_cls_tok   = self.det_cls_proj(det_cls_in)     # [B,K_sel,D]
        det_box_tok   = self.det_box_proj(det_box_in)     # [B,K_sel,D]
        det_logit_tok = self.det_logit_proj(det_logit_in) # [B,K_sel,D]

        tokens_list.append(det_cls_tok.unsqueeze(2))    # -> [B,K_sel,1,D]
        tokens_list.append(det_box_tok.unsqueeze(2))    # -> [B,K_sel,1,D]
        tokens_list.append(det_logit_tok.unsqueeze(2))  # -> [B,K_sel,1,D]

        # --------------------------------------------------------------
        # 2) Object tokens (pos/vis/scale/vol/shape)
        # --------------------------------------------------------------
        has_obj = ("obj" in self.encoder_type) and (obj_hidden_all is not None)
        if has_obj:
            obj_pos   = outs["obj_positions"]   # [L,B,Kmax,3] or [B,Kmax,3]
            if obj_pos.dim() == 4:
                obj_pos = obj_pos[-1]
            obj_vis   = outs["obj_vis"]
            if obj_vis.dim() == 4:
                obj_vis = obj_vis[-1]
            obj_scale = outs["obj_scale"]
            if obj_scale.dim() == 4:
                obj_scale = obj_scale[-1]
            obj_vol   = outs["obj_vol"]
            if obj_vol.dim() == 4:
                obj_vol = obj_vol[-1]

            obj_pos_sel   = obj_pos[:, :K_sel]    # [B,K_sel,3]
            obj_vis_sel   = obj_vis[:, :K_sel]    # [B,K_sel,1]
            obj_scale_sel = obj_scale[:, :K_sel]  # [B,K_sel,1]
            obj_vol_sel   = obj_vol[:, :K_sel]    # [B,K_sel,1]

            # obj_hidden_all: [B,Kmax,Qobj,C_obj], Qobj >= 7
            obj_hidden_sel = obj_hidden_all[:, :K_sel]           # [B,K_sel,Qobj,C_obj]
            B_, K_, Qobj, C_obj = obj_hidden_sel.shape
            assert Qobj >= 7, f"Expected at least 7 object queries, got {Qobj}"

            # Query index mapping (0-based within obj_hidden_sel):
            #  0 -> pos, 1 -> vis, 2 -> scale, 3,4,5 -> shape, 6 -> vol

            pos_hidden   = obj_hidden_sel[:, :, 0, :]  # [B,K_sel,C_obj]
            vis_hidden   = obj_hidden_sel[:, :, 1, :]
            scale_hidden = obj_hidden_sel[:, :, 2, :]
            shape_hidden = obj_hidden_sel[:, :, 3:6, :]  # [B,K_sel,3,C_obj]
            vol_hidden   = obj_hidden_sel[:, :, 6, :]

            # POS token
            pos_in = torch.cat([pos_hidden, obj_pos_sel], dim=-1)   # [B,K_sel,C_obj+3]
            pos_tok = self.obj_pos_proj(pos_in)                     # [B,K_sel,D]
            tokens_list.append(pos_tok.unsqueeze(2))

            # VIS token
            vis_in = torch.cat([vis_hidden, obj_vis_sel], dim=-1)   # [B,K_sel,C_obj+1]
            vis_tok = self.obj_vis_proj(vis_in)                     # [B,K_sel,D]
            tokens_list.append(vis_tok.unsqueeze(2))

            # SCALE token
            scale_in = torch.cat([scale_hidden, obj_scale_sel], dim=-1)  # [B,K_sel,C_obj+1]
            scale_tok = self.obj_scale_proj(scale_in)                    # [B,K_sel,D]
            tokens_list.append(scale_tok.unsqueeze(2))

            # VOL token
            vol_in = torch.cat([vol_hidden, obj_vol_sel], dim=-1)   # [B,K_sel,C_obj+1]
            vol_tok = self.obj_vol_proj(vol_in)                     # [B,K_sel,D]
            tokens_list.append(vol_tok.unsqueeze(2))

            # SHAPE tokens: three hidden shape queries, each its own projection
            for i in range(3):
                shape_h = shape_hidden[:, :, i, :]                     # [B,K_sel,C_obj]
                shape_tok_i = self.obj_shape_projs[i](shape_h)         # [B,K_sel,D]
                tokens_list.append(shape_tok_i.unsqueeze(2))

        # --------------------------------------------------------------
        # 3) Grasp tokens (per pose)
        # --------------------------------------------------------------
        has_grasp = ("grasp" in self.encoder_type) and (pose_hidden_all is not None)
        if has_grasp and self.num_grasp_poses > 0:
            pose_preds = outs["pose_preds"]          # [L,B,Kmax,num_poses,8] or [B,Kmax,num_poses,8]
            if pose_preds.dim() == 5:
                pose_preds = pose_preds[-1]          # last layer: [B,Kmax,num_poses,8]
            pose_sel = pose_preds[:, :K_sel]         # [B,K_sel,P,8]
            Bp, Kp, P, D8 = pose_sel.shape
            assert P == self.num_grasp_poses and D8 == 8

            pose_hidden_sel = pose_hidden_all[:, :K_sel]  # [B,K_sel,P,C_obj]

            pose_pos  = pose_sel[..., :3]    # [B,K_sel,P,3]
            pose_rot  = pose_sel[..., 3:7]   # [B,K_sel,P,4]
            pose_conf = pose_sel[..., 7:]    # [B,K_sel,P,1]

            # GRASP-POS tokens
            pos_in = torch.cat([pose_hidden_sel, pose_pos], dim=-1)         # [B,K_sel,P,C_obj+3]
            B_,K_,P_,Din = pos_in.shape
            pos_tok = self.grasp_pos_proj(pos_in.view(B_*K_*P_, Din))       # [B*K*P,D]
            pos_tok = pos_tok.view(B_, K_, P_, self.token_dim)              # [B,K_sel,P,D]
            tokens_list.append(pos_tok)   # note: already has "P" in the 3rd dim

            # GRASP-ROT tokens
            rot_in = torch.cat([pose_hidden_sel, pose_rot], dim=-1)         # [B,K_sel,P,C_obj+4]
            B_,K_,P_,Din = rot_in.shape
            rot_tok = self.grasp_rot_proj(rot_in.view(B_*K_*P_, Din))       # [B*K*P,D]
            rot_tok = rot_tok.view(B_, K_, P_, self.token_dim)              # [B,K_sel,P,D]
            tokens_list.append(rot_tok)

            # GRASP-CONF tokens
            conf_in = torch.cat([pose_hidden_sel, pose_conf], dim=-1)       # [B,K_sel,P,C_obj+1]
            B_,K_,P_,Din = conf_in.shape
            conf_tok = self.grasp_conf_proj(conf_in.view(B_*K_*P_, Din))    # [B*K*P,D]
            conf_tok = conf_tok.view(B_, K_, P_, self.token_dim)            # [B,K_sel,P,D]
            tokens_list.append(conf_tok)

        # --------------------------------------------------------------
        # Pack tokens: [B,K_sel,*,D] -> [B,K_sel*n_types,D]
        # --------------------------------------------------------------
        if not tokens_list:
            return torch.zeros(B, 0, self.token_dim, device=device)

        # tokens_list has entries of shape:
        #   [B,K_sel,1,D] for det & object, and [B,K_sel,P,D] for grasps
        tokens_per_det = torch.cat(tokens_list, dim=2)  # [B,K_sel,n_types,D]
        Bx, Kx, n_types, D = tokens_per_det.shape
        assert Bx == B and Kx == K_sel and D == self.token_dim

        obj_tokens = tokens_per_det.view(B, K_sel * n_types, D)  # [B,K_sel*n_types,D]
        return obj_tokens


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def forward_single_frame(
        self,
        rgb: torch.Tensor,          # [B,3,H,W]
        depth: Optional[torch.Tensor],  # [B,1,H,W] or None
        robot_state: Optional[torch.Tensor] = None,  # [B,Dq] or None 
        return_info: bool = False,
    ) -> torch.Tensor:
        """
        Tokens for one frame + optional layout info.
        """
        rgb_list = list(rgb)                   # B x [3,H,W]
        depth_list = list(depth) if depth is not None else None
        samples = nested_tensor_from_tensor_list(rgb_list, depth_list)
        if self.grasper is not None:
            outs = self._run_grasper(samples)
            img_tokens = self._image_tokens_from_joiner_backbone(outs['features'])
            obj_tokens = self._object_tokens_from_grasper(outs)
        else:
            features, _, _ = self.encoder(samples)
            img_tokens = self._image_tokens_from_joiner_backbone(features)
            B = img_tokens.shape[0]
            obj_tokens = torch.zeros(B, 0, self.token_dim, device=img_tokens.device)

        tokens = torch.cat([img_tokens, obj_tokens], dim=1)  # [B, N_img+N_obj, D]

        # ---- Robot-state token (one per frame) ----
        if robot_state is not None:
            rs_enc = self.robot_state_pe(robot_state)              # [B, D_enc]
            rs_tok = self.robot_state_proj(rs_enc).unsqueeze(1)    # [B,1,D]
            tokens = torch.cat([tokens, rs_tok], dim=1)            # [B,N_img+N_obj+1,D]

        if (not self._printed_token_summary) and is_main_process():
            print("\n[PerceptionTokens] token summary (single frame):")
            print(f"  image tokens   : {img_tokens.shape[1]}")
            print(f"  obj/det tokens : {obj_tokens.shape[1]} "
                  f"(= num_considered_best_detec_objects * number_of_object_tokens from det, obj, or grasp)")
            if robot_state is not None:
                print("  robot tokens   : 1")
            print(f"  total tokens   : {tokens.shape[1]}")
            print(f"  token_dim      : {self.token_dim}")
            self._printed_token_summary = True

        if not return_info:
            return tokens

        # ---- build layout info for this frame ----
        N_img = int(img_tokens.shape[1])
        N_obj = int(obj_tokens.shape[1])
        N_robot = 1 if robot_state is not None else 0

        info = {
            "N_img": N_img,
            "N_obj": N_obj,
            "N_robot": N_robot,   
            "last_image_fmaps": int(self.last_image_fmaps),
        }


        has_det = ("det" in self.encoder_type)
        has_obj = ("obj" in self.encoder_type) and has_det
        has_grasp = ("grasp" in self.encoder_type) and has_obj

        info["has_det"] = has_det
        info["has_obj"] = has_obj
        info["has_grasp"] = has_grasp

        if has_det and N_obj > 0:
            # base det tokens
            n_types = 3  # DET-CLS, DET-BOX, DET-LOGIT

            if has_obj:
                # OBJ-POS, OBJ-VIS, OBJ-SCALE, OBJ-VOL, OBJ-SHAPE1/2/3
                n_types += 7

            if has_grasp and self.num_grasp_poses > 0:
                # per pose: GRASP-POS, GRASP-ROT, GRASP-CONF
                n_types += 3 * self.num_grasp_poses

            info["n_types"] = n_types
            info["K_sel"] = N_obj // n_types
        else:
            info["n_types"] = 0
            info["K_sel"] = 0

        return tokens, info

    def forward(
        self,
        rgb_seq: torch.Tensor,                  # [B,T,3,H,W]
        depth_seq: Optional[torch.Tensor] = None,        # [B,T,1,H,W]
        robot_state_seq: Optional[torch.Tensor] = None,  # [B,T,Dq] 
        return_info: bool = False,
    ) -> torch.Tensor:
        """
        Returns:
            tokens: [B, N_obs, token_dim]
            info_list (if return_info): list of dicts, length T
        """
        assert rgb_seq.dim() == 5, f"Expected [B,T,3,H,W], got {rgb_seq.shape}"
        B, T, C, H, W = rgb_seq.shape
        tokens_all: List[torch.Tensor] = []
        info_list: List[Dict] = []

        for t in range(T):
            rgb_t = rgb_seq[:, t]
            depth_t = depth_seq[:, t] if depth_seq is not None else None
            robot_state_t = (
                robot_state_seq[:, t] if robot_state_seq is not None else None
            )
            if return_info:
                tok_t, info_t = self.forward_single_frame(
                    rgb_t, depth_t, robot_state_t, return_info=True
                )
                info_list.append(info_t)
            else:
                tok_t = self.forward_single_frame(rgb_t, depth_t, robot_state_t)
            tokens_all.append(tok_t)

        tokens = torch.cat(tokens_all, dim=1)  # [B, sum_t N_t, D]

        if (not self._printed_observation_summary) and is_main_process():
            print("\n[PerceptionTokens] observation summary):")
            print(f"  image shape   : {rgb_seq.shape}")
            print(f"  tokens : {tokens.shape[1]} ")
            print(f"  with depth    : {depth_seq is not None}")
            print(f"  with robot state: {robot_state_seq is not None}")
            print(f"  return_info: {return_info}")
            self._printed_observation_summary = True



        if return_info:
            return tokens, info_list
        return tokens



# ---------------------------------------------------------------------------
# TransformerDiffusionHead: action head
# ---------------------------------------------------------------------------
class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,  # [B, L, D]
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        src: [B, L, D]
        returns:
            src_out: [B, L, D]
            attn: [B, num_heads, L, L] or None
        """
        attn_output, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)

        if return_attn:
            return src, attn_weights  # [B, H, L, L]
        return src, None


class TransformerEncoderWithAttn(nn.Module):
    def __init__(self, layer: TransformerEncoderLayerWithAttn, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(layer.self_attn.embed_dim)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        src: [B, L, D]
        returns:
            src_out: [B, L, D]
            attn_list: list of [B,H,L,L] (one per layer) if return_attn
        """
        attn_list: Optional[List[torch.Tensor]] = [] if return_attn else None
        output = src
        for layer in self.layers:
            output, attn = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attn=return_attn,
            )
            if return_attn:
                attn_list.append(attn)
        output = self.norm(output)
        if return_attn:
            return output, attn_list
        return output, None



class TransformerDiffusionHead(nn.Module):
    """
    Transformer-based noise predictor for diffusion on action sequences.

    Inputs
    ------
    x_t          : [B, T, A]      noisy actions at diffusion step t
    cond_tokens  : [B, N_ctx, D]  perception tokens (already in model_dim)
    t            : [B]            diffusion timestep indices

    Output
    ------
    eps_pred     : [B, T, A]      predicted noise for each action element
    """
    def __init__(
        self,
        act_dim: int,
        horizon: int,
        model_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.act_dim = int(act_dim)
        self.horizon = int(horizon)
        self.model_dim = int(model_dim)

        if dim_feedforward is None:
            dim_feedforward = 4 * self.model_dim

        # Project actions to model space
        self.action_in = nn.Linear(self.act_dim, self.model_dim)

        # Time embedding (same dim as model_dim)
        self.time_embed = SinusoidalTimeEmbedding(self.model_dim)

        # Transformer encoder over [cond_tokens; action_tokens]

        enc_layer = TransformerEncoderLayerWithAttn(
            d_model=self.model_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = TransformerEncoderWithAttn(enc_layer, num_layers=n_layers)
        

        # Project model space back to action space
        self.action_out = nn.Linear(self.model_dim, self.act_dim)

    def forward(
        self,
        x_t: torch.Tensor,         # [B, T, A]
        cond_tokens: torch.Tensor, # [B, N_ctx, D]
        t: torch.Tensor,           # [B]
        return_attn: bool = False,
    ):
        B, T, A = x_t.shape
        assert A == self.act_dim, f"Expected act_dim={self.act_dim}, got {A}"

        # Embed actions
        act_tokens = self.action_in(x_t)    # [B, T, D]

        # Add time embedding to action tokens
        t_emb = self.time_embed(t)          # [B, D]
        t_emb = t_emb.unsqueeze(1)          # [B, 1, D]
        act_tokens = act_tokens + t_emb     # broadcast over T

        # Handle condition tokens (may be empty)
        if (cond_tokens is None) or (cond_tokens.numel() == 0):
            seq = act_tokens                 # [B, T, D]
            start_idx = 0
        else:
            assert cond_tokens.shape[0] == B, "Batch size mismatch in cond_tokens"
            assert cond_tokens.shape[-1] == self.model_dim, (
                f"cond_tokens dim {cond_tokens.shape[-1]} != model_dim {self.model_dim}"
            )
            seq = torch.cat([cond_tokens, act_tokens], dim=1)  # [B, N_ctx+T, D]
            start_idx = cond_tokens.shape[1]

        # Transformer encoder mixes cond + action tokens
        enc_out, attn_list = self.encoder(
            seq,
            src_mask=None,
            src_key_padding_mask=None,
            return_attn=return_attn,
        )                                   # enc_out: [B, N_ctx+T, D]

        # Take only the action positions
        act_out = enc_out[:, start_idx:, :]  # [B, T, D]

        # Predict noise in action space
        eps_pred = self.action_out(act_out)  # [B, T, A]

        if return_attn:
            return eps_pred, attn_list, start_idx  # start_idx for cond/action split
        return eps_pred






# ---------------------------------------------------------------------------
# Full encode low dim robot state with positional encoding into high dim frequencies. 
# ---------------------------------------------------------------------------
class ScalarPositionalEncoding(nn.Module):
    """
    Depth-style positional encoding for 1D scalars (e.g., joint positions).

    For input x ∈ ℝ^(B,D), produces sin/cos features at multiple frequencies:
        PE(x) = [sin(ω_k x), cos(ω_k x)]_k
    Output is in [-1,1].
    """
    def __init__(self, in_dim: int, num_frequencies: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        freq_bands = 2.0 ** torch.arange(num_frequencies)  # [F]
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.out_dim = in_dim * num_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim]
        returns: [B, in_dim * num_frequencies * 2] in [-1,1]
        """
        if x.dim() != 2:
            x = x.view(x.shape[0], -1)
        # [B, in_dim, 1]
        x = x.unsqueeze(-1)
        # [B, in_dim, F]
        angles = x * self.freq_bands.view(1, 1, -1)
        sin_feat = torch.sin(angles)
        cos_feat = torch.cos(angles)
        # [B, in_dim, 2F]
        pe = torch.cat([sin_feat, cos_feat], dim=-1)
        # [B, in_dim * 2F]
        pe = pe.view(pe.shape[0], -1)
        return pe

# ---------------------------------------------------------------------------
# Full policy: perception + diffusion
# ---------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """
    Diffusion policy:
      - PerceptionTokens: RGBD backbone (+ optional Grasper) -> obs tokens
      - GaussianDiffusion + TransformerDiffusionHead on actions
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
        model_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        diffusion_steps_train: int = 100,
        diffusion_steps_eval: int = 10,
        image_kernel: int = 1,
        action_min: Optional[torch.Tensor | List[float]] = None,
        action_max: Optional[torch.Tensor | List[float]] = None,
    ):
        super().__init__()

        self.perception = PerceptionTokens(
            args_backbone=args_backbone,
            encoder_type=encoder_type,
            max_objects_pretrained=args_backbone.max_objects,
            diffusion_max_objs=diffusion_max_objs,
            token_dim=token_dim,
            image_kernel=image_kernel,
            robot_state_dim=robot_state_dim
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

        # If perception token_dim != model_dim, add a projection
        self.perc_to_model = None
        if token_dim != model_dim:
            self.perc_to_model = nn.Linear(token_dim, model_dim)

        # ---- action normalization stats ----
        if action_min is not None and action_max is not None:
            a_min = torch.as_tensor(action_min, dtype=torch.float32)
            a_max = torch.as_tensor(action_max, dtype=torch.float32)
            self.register_buffer("action_min", a_min.view(1, 1, -1))
            self.register_buffer("action_max", a_max.view(1, 1, -1))
        else:
            self.action_max = None
            self.action_min  = None

    def _normalize(self, actions):
        denom = (self.action_max - self.action_min).clamp(min=1e-6)
        return (actions - self.action_min) / denom * 2.0 - 1.0

    def _unnormalize(self, ndata):
        denom = (self.action_max - self.action_min).clamp(min=1e-6)
        return (ndata + 1.0) / 2.0 * denom + self.action_min


    def _print_param_summary(self):
        # ---------------- Parameter summary ----------------    
        enc = self.perception.encoder
        gr = self.perception.grasper
        gr_total, gr_train = _count_params(gr) if isinstance(gr, nn.Module) else (0, 0)
        enc_total, enc_train = _count_params(enc) if isinstance(enc, nn.Module) else (0, 0)

        if enc is None: 
            enc = self.perception.grasper.backbone
            enc_total, enc_train = _count_params(enc) if isinstance(enc, nn.Module) else (0, 0)
            gr_total = gr_total - enc_total
            gr_train = gr_train - enc_train      

        # Perception projection layers (everything in perception except encoder+grasper)
        proj_params = []
        for n, p in self.perception.named_parameters():
            if ("encoder" not in n) and ("grasper" not in n):
                proj_params.append(p)
        proj_mod = nn.Module()
        proj_mod._params = nn.ParameterList(proj_params)  # hack module for counting
        proj_total, proj_train = _count_params(proj_mod)

        # Diffusion head
        head_total, head_train = _count_params(self.head)

        # All
        pol_total, pol_train = _count_params(self)

        if is_main_process():
            print("\n[DiffusionPolicy] Parameter breakdown:")
            print(f"  Backbone encoder: total={_fmt_m(enc_total)}, trainable={_fmt_m(enc_train)}, frozen={_fmt_m(enc_total - enc_train)}")
            print(f"  Grasper       :   total={_fmt_m(gr_total)}, trainable={_fmt_m(gr_train)}, frozen={_fmt_m(gr_total - gr_train)}")
            print(f"  Perception proj:  total={_fmt_m(proj_total)}, trainable={_fmt_m(proj_train)}, frozen={_fmt_m(proj_total - proj_train)}")
            print(f"  Diffusion head:   total={_fmt_m(head_total)}, trainable={_fmt_m(head_train)}, frozen={_fmt_m(head_total - head_train)}")
            print(f"  TOTAL policy:     total={_fmt_m(pol_total)}, trainable={_fmt_m(pol_train)}, frozen={_fmt_m(pol_total - pol_train)}")

    # ---------------- Training forward ----------------
    def forward(
        self,
        rgb_seq: torch.Tensor,          # [B,Hobs,3,H,W]
        depth_seq: torch.Tensor,        # [B,Hobs,1,H,W]
        robot_state_seq: torch.Tensor,  # [B,Hobs,Dq]  
        actions: torch.Tensor,          # [B,T,A]       (RELATIVE expert actions)
    ) -> torch.Tensor:
        B, T, A = actions.shape
        device = actions.device
        
        # Observation tokens (vision + object + robot state)
        with torch.set_grad_enabled(True):
            tokens = self.perception(rgb_seq, depth_seq, robot_state_seq)
        if tokens is None or tokens.numel() == 0:
            tokens = torch.zeros(B, 1, self.head.model_dim, device=device)
        if self.perc_to_model is not None:
            tokens = self.perc_to_model(tokens)  # [B,N_obs,model_dim]

        # Normalize actions in RELATIVE space
        actions_norm = self._normalize(actions)

        t = torch.randint(
            0, self.diffusion.num_train_steps, (B,), device=device, dtype=torch.long
        )
        noise = torch.randn_like(actions_norm)
        x_t = self.diffusion.q_sample(actions_norm, t, noise)
        eps_pred = self.head(x_t, tokens, t)
        loss = F.mse_loss(eps_pred, noise)

        #w = torch.ones_like(eps_pred)
        #w[:, 0] = 5.0
        #loss = F.mse_loss(eps_pred * w, noise * w)


        return loss

    
    # ---------------- Sampling for evaluation ----------------
    @torch.no_grad()
    def sample_actions(
        self,
        rgb_seq: torch.Tensor,          # [B,Hobs,3,H,W]
        depth_seq: torch.Tensor,        # [B,Hobs,1,H,W]
        robot_state_seq: torch.Tensor,  # [B,Hobs,Dq]
        horizon: int,
    ) -> torch.Tensor:
        """
        DDIM sampling of RELATIVE actions [B,T,A] given obs tokens.
        Returns actions in RELATIVE units (de-normalized if stats present).
        """
        B = rgb_seq.shape[0]
        device = rgb_seq.device
        
        tokens = self.perception(rgb_seq, depth_seq, robot_state_seq)
        if tokens is None or tokens.numel() == 0:
            tokens = torch.zeros(B, 1, self.head.model_dim, device=device)
        if self.perc_to_model is not None:
            tokens = self.perc_to_model(tokens)

        x_T = torch.randn(B, horizon, self.head.act_dim, device=device)  # normalized space
        x0_norm = self.diffusion.ddim_sample(self.head, x_T, tokens)

        actions = self._unnormalize(x0_norm)

        return actions

    
    @torch.no_grad()
    def get_attention_stats(
        self,
        rgb_seq: torch.Tensor,          # [B,T,3,H,W]
        depth_seq: torch.Tensor,        # [B,T,1,H,W]
        robot_state_seq: torch.Tensor,  # [B,T,Dq]    
        actions: torch.Tensor,          # [B,T_act,A]
        only_action_queries: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute average attention from queries (default: action tokens)
        to different groups of condition tokens (image lvl-*, det_cls, det_geo,
        obj, grasp), per layer.

        Returns:
            stats[layer_name][group_name] = mean attention fraction.
        """
        self.eval()
        device = rgb_seq.device
        B, T_obs = rgb_seq.shape[:2]
        B_act, T_act, _ = actions.shape
        assert B == B_act, "Batch mismatch"

        # 1) Build tokens + layout info
        tokens, frame_infos = self.perception(
            rgb_seq, depth_seq, robot_state_seq, return_info=True
        )
        if self.perc_to_model is not None:
            tokens = self.perc_to_model(tokens) # [B,N_ctx,model_dim]

        N_ctx = tokens.shape[1]

        # 2) Run diffusion head once with attention recording
        # we don't care much which t we pick here; use zeros
        t = torch.zeros(B, device=device, dtype=torch.long)
        x_t = actions  # you can also use q_sample if you prefer

        eps_pred, attn_list, start_idx = self.head(
            x_t, tokens, t, return_attn=True
        )
        # start_idx must equal N_ctx (cond tokens come first)
        assert start_idx == N_ctx, f"Unexpected start_idx={start_idx}, N_ctx={N_ctx}"

        # attn_list: list of [B,H,L,L], where L = N_ctx + T_act
        L_total = attn_list[0].shape[-1]
        assert L_total == N_ctx + T_act

        # 3) Build token groups (indices into cond tokens [0..N_ctx-1])
        groups = self._build_token_groups(frame_infos, N_ctx)

        # 4) Aggregate attention per group, per layer
        stats = self._aggregate_attention(attn_list, groups, N_ctx, only_action_queries)
        return stats

    def _build_token_groups(
        self,
        frame_infos: List[Dict],
        cond_len: int,
    ) -> Dict[str, List[int]]:
        """
        Build mapping group_name -> list of cond-token indices.

        With new layout per object:
          0          : DET-CLS
          1          : DET-BOX
          2          : DET-LOGIT
          3..9       : OBJ-POS,OBJ-VIS,OBJ-SCALE,OBJ-VOL,OBJ-SHAPE1/2/3
          10..n_types-1 : GRASP-* (if any)
        """
        groups: Dict[str, List[int]] = {}

        offset = 0
        for f, info in enumerate(frame_infos):
            N_img = info["N_img"]
            N_obj = info["N_obj"]
            N_robot = info.get("N_robot", 0)   
            L = info["last_image_fmaps"]
            frame_len = N_img + N_obj + N_robot  

            # Image tokens by FPN level
            if N_img > 0 and L > 0:
                for lvl in range(L):
                    idxs = []
                    for i in range(N_img):
                        if (i % L) == lvl:
                            idxs.append(offset + i)
                    if "img" in groups:
                        groups["img"] += idxs
                    else:
                        groups["img"] = idxs

            has_det = info["has_det"]
            has_obj = info["has_obj"]
            has_grasp = info["has_grasp"]
            n_types = info.get("n_types", 0)
            K_sel = info.get("K_sel", 0)

            if has_det and N_obj > 0 and n_types > 0 and K_sel > 0:
                for j in range(K_sel):
                    base = N_img + j * n_types

                    # DET-CLS
                    groups.setdefault("det", []).append(offset + base + 0)

                    # DET-GEO = DET-BOX + DET-LOGIT
                    groups.setdefault("det", []).append(offset + base + 1)
                    groups.setdefault("det", []).append(offset + base + 2)

                    if has_obj:
                        # OBJ = all 7 object tokens
                        groups.setdefault("obj_pos", []).append(offset + base + 3)
                        groups.setdefault("obj_vis", []).append(offset + base + 4)
                        groups.setdefault("obj_scale", []).append(offset + base + 5)
                        groups.setdefault("obj_shape", []).append(offset + base + 6)
                        groups.setdefault("obj_shape", []).append(offset + base + 7)
                        groups.setdefault("obj_shape", []).append(offset + base + 8)
                        groups.setdefault("obj_vol", []).append(offset + base + 9)

                    if has_grasp:
                        # GRASP = all grasp tokens (indices 10..n_types-1)
                        for k in range(10, n_types):
                            groups.setdefault("grasp", []).append(offset + base + k)

            # Robot-state token (last token in frame)
            if N_robot > 0:
                robot_idx = offset + N_img + N_obj
                groups.setdefault("robot_state", []).append(robot_idx)

            offset += frame_len

        assert offset == cond_len, (
            f"Token group construction mismatch: offset={offset}, cond_len={cond_len}"
        )
        return groups

    def _aggregate_attention(
        self,
        attn_list: List[torch.Tensor],    # list of [B,H,L,L]
        groups: Dict[str, List[int]],
        cond_len: int,
        only_action_queries: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        For each layer and group, compute mean attention fraction from queries to
        that group (over cond tokens only).
        """
        stats = {}
        eps = 1e-8
        
        obj_total = []

        for layer_idx, attn in enumerate(attn_list):
            # attn: [B,H,L,L]
            B, H, Lq, Lk = attn.shape
            assert Lk == Lq

            # queries: either all tokens or only action tokens (indices >= cond_len)
            if only_action_queries:
                q_idx = torch.arange(cond_len, Lq, device=attn.device, dtype=torch.long)
            else:
                q_idx = torch.arange(0, Lq, device=attn.device, dtype=torch.long)

            attn_q = attn[:, :, q_idx, :]              # [B,H,Q,Lk]
            # Total attention to condition tokens (keys 0..cond_len-1)
            total_cond = attn_q[:, :, :, :cond_len].sum(dim=-1)  # [B,H,Q]

            for g_name, idx_list in groups.items():
                if not idx_list:
                    continue
                idx = torch.tensor(idx_list, device=attn.device, dtype=torch.long)
                attn_g = attn_q[:, :, :, idx]          # [B,H,Q,Ng]
                sum_g = attn_g.sum(dim=-1)             # [B,H,Q]
                frac = sum_g / (total_cond + eps)      # [B,H,Q]
                mean_frac = float(frac.mean().item())

                if g_name not in stats:
                    stats[g_name] = [mean_frac]
                else:
                    stats[g_name].append(mean_frac)
        
        for g_name in stats.keys():
            stats[g_name] = sum(stats[g_name]) / len(stats[g_name])
            if 'obj_' in g_name:
                obj_total.append(stats[g_name])
        
        
        if len(obj_total) > 0:
            stats['obj_sum'] = sum(obj_total)
        return stats
