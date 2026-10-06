
# model/deformable_detr/grasp_estimation_transformer.py (excerpt)

import torch
import torch.nn.functional as F
from torch import nn
from model.deformable_detr.util.misc import (NestedTensor)
from model.deformable_detr.backbone import build_backbone as wrap_backbone
from model.deformable_detr.grasp_deform_transformer import build_deforamble_transformer
from model.deformable_detr.grasp_pose_diffusion import (
    GraspPoseDiffusionHead,
    grasp_diffusion_loss,
)


import copy
import re
from collections import defaultdict

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _parse_proj_token(token: str):
    """
    Parses a token like '2u', '1u', '3k', '0k', or '2u3k'.
    Returns (up_count, k_val), where:

      - up_count: number of 2x ConvTranspose upsamples to apply
      - k_val:    J from 'Jk' (both kernel size and repeat count). If 0 => no post-upsampling convs.

    """
    t = token.strip()
    m_u = re.search(r'(\d+)u', t)
    m_k = re.search(r'(\d+)k', t)
    up_count = int(m_u.group(1)) if m_u else 0
    k_val = int(m_k.group(1)) if m_k else 0
    return up_count, k_val

import torch
from torch import nn

@torch.no_grad()
def _rgb_depth_contrib_percent_1x1(x_bchw: torch.Tensor, conv1x1: nn.Conv2d, eps: float = 1e-12):
    """
    x_bchw: (B, C, H, W) input to the 1x1 conv (pre-projection).
    conv1x1: nn.Conv2d with kernel_size=1.
    split_idx: channel index separating RGB ([:split_idx]) and Depth ([split_idx:]).
    Returns:
        pct_rgb, pct_depth: each (B,) with per-image fractions of squared pre-activation energy.
    """

    
    # Use half split by default; set to 768 if you want fixed split
    split_idx = conv1x1.in_channels // 2
    assert isinstance(conv1x1, nn.Conv2d) and conv1x1.kernel_size == (1, 1), "Expected a 1x1 Conv2d"

    B, C, H, W_ = x_bchw.shape
    assert split_idx > 0 and split_idx < C, f"Invalid split_idx={split_idx} for C={C}"

    # Prepare tensors
    x = x_bchw.permute(0, 2, 3, 1).reshape(B, H * W_, C)  # (B, N, C)
    Wmat = conv1x1.weight.view(conv1x1.out_channels, conv1x1.in_channels)  # (O, C)

    # Split inputs and weights
    x_rgb = x[:, :, :split_idx]                 # (B, N, split_idx)
    x_depth = x[:, :, split_idx:]               # (B, N, C - split_idx)
    W_rgb = Wmat[:, :split_idx]                 # (O, split_idx)
    W_depth = Wmat[:, split_idx:]               # (O, C - split_idx)

    # Find the how much each modalitiy contributes to the output 
    y_rgb   = F.linear(x_rgb,   W_rgb,   bias=None)  # (B, N, O)
    y_depth = F.linear(x_depth, W_depth, bias=None)  # (B, N, O)

    # compute the squared energy (everything positive) and sum over the O output nodes given their N activations to get a overall contribution per input image in each batch B
    e_rgb = (y_rgb ** 2).sum(dim=(1, 2))       # (B,)
    e_depth = (y_depth ** 2).sum(dim=(1, 2))   # (B,)

    # compute the percentage
    total = e_rgb + e_depth + eps
    pct_rgb = e_rgb / total
    pct_depth = e_depth / total
    return pct_rgb, pct_depth


class Grasper(nn.Module):
    def __init__(self, backbone, transformer, num_feature_levels=4, aux_loss=True, proj_kernel_size=1, det_rgb_only=True, is_rgbd=False, max_encoded_img_per_stack=512):
        super().__init__()
        self.transformer = transformer
        self.hidden_dim_det = transformer.d_model_det
        self.hidden_dim_obj = transformer.d_model_obj
        self.with_obj = transformer.with_obj
        self.with_pose = transformer.with_pose
        self.with_critique = transformer.with_critique
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.num_feature_levels = num_feature_levels
        self.proj_kernel_size = proj_kernel_size
        self.det_rgb_only = det_rgb_only
        self.is_rgbd = is_rgbd
        self.max_encoded_img_per_stack = max_encoded_img_per_stack
        print('proj_kernel_size', self.proj_kernel_size)
        
        # Project backbone feature maps to transformer d_model
        num_backbone_outs = len(backbone.num_channels)
        
        input_proj_list = []
            
        # Original behavior: one conv per backbone output
        for n in range(num_backbone_outs):
            
            in_channels = backbone.num_channels[n]
            if self.det_rgb_only and is_rgbd:
                in_channels = in_channels // 2

            if proj_kernel_size > 0 and in_channels != self.hidden_dim_det:        
                seq = nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim_det, kernel_size=1, padding=0),
                    nn.GroupNorm(32, self.hidden_dim_det),
                    nn.GELU(),
                )
                input_proj_list.append(seq)
            else:
                input_proj_list.append(None)
                     
        self.input_proj = nn.ModuleList(input_proj_list)    

        
        for proj in self.input_proj:
            if proj is None:
                continue
            for m in proj.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)   # or kaiming_normal_ for ReLU-like
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        self.input_proj_obj = None
        if self.with_obj:
            input_proj_list = []                
            # Original behavior: one conv per backbone output
            for n in range(num_backbone_outs):            
                in_channels = backbone.num_channels[n]
                if proj_kernel_size > 0 and in_channels != self.hidden_dim_obj:            
                    seq = nn.Sequential(
                        nn.Conv2d(in_channels, self.hidden_dim_obj, kernel_size=1, padding=0),
                        nn.GroupNorm(32, self.hidden_dim_obj),
                        nn.GELU(),
                    )
                    input_proj_list.append(seq)
                else:
                    input_proj_list.append(None)
            self.input_proj_obj = nn.ModuleList(input_proj_list)            

            for proj in self.input_proj_obj:
                if proj is None:
                    continue
                for m in proj.modules():
                    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                        nn.init.xavier_uniform_(m.weight, gain=1.0)   # or kaiming_normal_ for ReLU-like
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)


        self.input_proj_pose = None
        if self.with_pose:
            input_proj_list = []                
            # Original behavior: one conv per backbone output
            for n in range(num_backbone_outs):            
                in_channels = backbone.num_channels[n]
                if proj_kernel_size > 0 and in_channels != self.hidden_dim_obj:            
                    seq = nn.Sequential(
                        nn.Conv2d(in_channels, self.hidden_dim_obj, kernel_size=1, padding=0),
                        nn.GroupNorm(32, self.hidden_dim_obj),
                        nn.GELU(),
                    )
                    input_proj_list.append(seq)
                else:
                    input_proj_list.append(None)
            self.input_proj_pose = nn.ModuleList(input_proj_list)            

            for proj in self.input_proj_pose:
                if proj is None:
                    continue
                for m in proj.modules():
                    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                        nn.init.xavier_uniform_(m.weight, gain=1.0)   # or kaiming_normal_ for ReLU-like
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        self.input_proj_critique = None
        if self.with_critique:
            input_proj_list = []                
            # Original behavior: one conv per backbone output
            for n in range(num_backbone_outs):            
                in_channels = backbone.num_channels[n]
                if proj_kernel_size > 0 and in_channels != self.hidden_dim_obj:            
                    seq = nn.Sequential(
                        nn.Conv2d(in_channels, self.hidden_dim_obj, kernel_size=1, padding=0),
                        nn.GroupNorm(32, self.hidden_dim_obj),
                        nn.GELU(),
                    )
                    input_proj_list.append(seq)
                else:
                    input_proj_list.append(None)
            self.input_proj_critique = nn.ModuleList(input_proj_list)            

            for proj in self.input_proj_critique:
                if proj is None:
                    continue
                for m in proj.modules():
                    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                        nn.init.xavier_uniform_(m.weight, gain=1.0)   # or kaiming_normal_ for ReLU-like
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)


        # Optional diffusion head for grasp-pose proposal generation.
        self.use_grasp_diffusion = bool(
            getattr(transformer, "use_grasp_diffusion", False)
        )

        self.grasp_diffusion = None

        if self.use_grasp_diffusion:
            if not self.with_pose:
                raise RuntimeError(
                    "use_grasp_diffusion=True requires with_pose=True."
                )

            self.grasp_diffusion = GraspPoseDiffusionHead(
                cond_dim=self.hidden_dim_obj,
                hidden_dim=transformer.grasp_diffusion_hidden_dim,
                num_layers=transformer.grasp_diffusion_layers,
                num_heads=transformer.grasp_diffusion_heads,
                num_train_steps=transformer.grasp_diffusion_train_steps,
                num_eval_steps=transformer.grasp_diffusion_eval_steps,
                pose_min=transformer.grasp_diffusion_pose_min,
                pose_max=transformer.grasp_diffusion_pose_max,
            )




    def _freeze_and_eval_backbone(self):
        # 1) freeze parameters
        for n, p in self.named_parameters():
            if 'backbone' in n:
                p.requires_grad_(False)
        # 2) set all critique submodules to eval
        for n, m in self.named_modules():
            if 'backbone' in n:
                m.eval()                

    def _freeze_and_eval_critique(self):
        # 1) freeze parameters
        for n, p in self.named_parameters():
            if 'critique_' in n or '_critique' in n:
                p.requires_grad_(False)
        # 2) set all critique submodules to eval
        for n, m in self.named_modules():
            if 'critique_' in n or '_critique' in n:
                m.eval()                

    def _freeze_and_eval_obj(self):
        # 1) freeze parameters
        for n, p in self.named_parameters():
            if 'obj_' in n or '_obj' in n:
                p.requires_grad_(False)
        # 2) set all obj submodules to eval
        for n, m in self.named_modules():
            if 'obj_' in n or '_obj' in n:
                m.eval()                

    def _freeze_and_eval_det(self):
        # 1) freeze parameters
        for n, p in self.named_parameters():
            if 'det_' in n or '_det' in n:
                p.requires_grad_(False)
        # 2) set all det submodules to eval
        for n, m in self.named_modules():
            if 'det_' in n or '_det' in n:
                m.eval()  
            
    def _freeze_and_eval_pose(self):
        # 1) freeze parameters
        for n, p in self.named_parameters():
            if 'pose_' in n or '_pose' in n:
                p.requires_grad_(False)
        # 2) set all pose submodules to eval
        for n, m in self.named_modules():
            if 'pose_' in n or '_pose' in n:
                m.eval()  
            
    def _set_aux_loss(self, out_dict):

        num_layers = out_dict['det_coord'].shape[0]
        aux = []
        for l in range(num_layers - 1):
            aux_ = {
                'pred_boxes': out_dict['det_coord'][l],# [B,Q,4]
                'pred_logits': out_dict['det_logits'][l],# [B,Q,C=2]
                'det_cls_token': out_dict['det_cls_token'][l] # [B,Q,dEnc]
            }
            # include object aux stacks when present
            for k in out_dict.keys():
                if k.startswith('obj_') or k.startswith('critique_') or k.startswith('pose_'):
                    aux_[k] = out_dict[k][l]
            aux.append(aux_)
        return aux


    #def forward(self, samples: NestedTensor, targets=None):
    def forward(
        self,
        samples: NestedTensor,
        targets=None,
        diffusion_targets=None,
        diffusion_sample_steps: int = 0,
    ):

        if not isinstance(samples, NestedTensor):
            samples = NestedTensor(*samples)
        device = samples.tensors.device
        out = {}

        if targets is not None:
            crops = []
            batch_ids = []
            for i, t in enumerate(targets):
                if "inst_crops_images" not in t:
                    continue
                crops.append(t["inst_crops_images"])
                batch_ids.append(torch.full((len(crops[-1]),), i, device=crops[-1].device, dtype=crops[-1].dtype))

            if len(crops) > 0:
                crops = torch.cat(crops, dim=0)
                batch_ids = torch.cat(batch_ids, dim=0)

                if len(crops) > self.max_encoded_img_per_stack:
                    inst_crops_embeds = []
                    s = len(crops) // self.max_encoded_img_per_stack
                    for i in range(s):
                        inst_crops_embeds.append(
                            self.backbone.extract_cls_token(crops[i*self.max_encoded_img_per_stack: (i+1)*self.max_encoded_img_per_stack])
                        )
                    if s*self.max_encoded_img_per_stack < len(crops):
                        inst_crops_embeds.append(
                            self.backbone.extract_cls_token(crops[s*self.max_encoded_img_per_stack:])
                        )
                    inst_crops_embeds = torch.cat(inst_crops_embeds, dim=0).detach()
                else:
                    inst_crops_embeds = self.backbone.extract_cls_token(crops).detach()
                out['inst_crops_embeds'] = [inst_crops_embeds[batch_ids == i] for i in range(len(targets))]
            
        # Backbone forward
        features, pos, pos_obj = self.backbone(samples)

        out["features"] = features      
        out["pos"] = pos                
        out["pos_obj"] = pos_obj   

        # Multi-scale projection
        in_dict = defaultdict(list)
        in_dict['pos_embeds'] = pos
        in_dict['targets'] = targets

        if self.with_obj or self.with_critique or self.with_pose:
            in_dict['pos_obj'] = pos_obj

        rgb_depth_pct_per_level = defaultdict(list)  # will store (B,2) per level or None
        for lvl, feat in enumerate(features):
            src, mask, _ = feat.decompose()
            if self.input_proj[lvl] is not None:
                if self.is_rgbd and self.det_rgb_only:
                    l = src.shape[1]//2
                    in_dict['srcs'].append(self.input_proj[lvl](src[:, :l, :, :]))
                else:
                    in_dict['srcs'].append(self.input_proj[lvl](src))

            else:
                if self.is_rgbd and self.det_rgb_only:
                    l = src.shape[1]//2
                    in_dict['srcs'].append(src[:, :l, :, :])
                else:
                    in_dict['srcs'].append(src)

            if self.with_obj:
                if self.input_proj_obj[lvl] is not None:
                    if not self.training:                    
                        pct_rgb, pct_depth = _rgb_depth_contrib_percent_1x1(src, self.input_proj_obj[lvl][0])
                        rgb_depth_pct_per_level['obj'].append(torch.stack((pct_rgb, pct_depth), dim=-1))  # (B,2)        
                    in_dict['srcs_obj'].append(self.input_proj_obj[lvl](src))
                else:
                    in_dict['srcs_obj'].append(src)

            if self.with_pose:
                if self.input_proj_pose[lvl] is not None:
                    if not self.training:                    
                        pct_rgb, pct_depth = _rgb_depth_contrib_percent_1x1(src, self.input_proj_pose[lvl][0])
                        rgb_depth_pct_per_level['pose'].append(torch.stack((pct_rgb, pct_depth), dim=-1))  # (B,2)        
                    in_dict['srcs_pose'].append(self.input_proj_pose[lvl](src))
                else:
                    in_dict['srcs_pose'].append(src)

            if self.with_critique:
                if self.input_proj_critique[lvl] is not None:
                    if not self.training:                    
                        pct_rgb, pct_depth = _rgb_depth_contrib_percent_1x1(src, self.input_proj_critique[lvl][0])
                        rgb_depth_pct_per_level['critique'].append(torch.stack((pct_rgb, pct_depth), dim=-1))  # (B,2)        
                    in_dict['srcs_critique'].append(self.input_proj_critique[lvl](src))
                else:
                    in_dict['srcs_critique'].append(src)


            in_dict['masks'].append(mask)
            assert mask is not None
        # Transformer forward (hs_out: intermediates; out_dict: stacked outputs)

        #print('in_dict', {key: type(v) for key, v in in_dict.items()})

        hs_out, out_dict, outs_shared = self.transformer(**in_dict)
        
  
        # Final-layer outputs (criterion expects 'pred_boxes', 'pred_classes')
        out.update({
            'pred_boxes': out_dict['det_coord'] [-1],      # [B,Q,4]
            'pred_logits': out_dict['det_logits'] [-1],  # [B,Q,2]
            'det_cls_token': out_dict['det_cls_token'][-1]  # [B,Q,dEnc]
            #'pred_conf': conf_stack[-1] 
        })

        # Object stacks (kept as [L,B,...] for aux)
        for k in out_dict.keys():
            if k.startswith('obj_') or k.startswith('critique_') or k.startswith('pose_'):
                out[k] = out_dict[k][-1]
        
        for k in outs_shared.keys():
            out[k] = outs_shared[k]

        # Aux detection
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_dict)

        # Pack RGB/Depth percentages as (L, B, 2): [:,:,0]=RGB, [:,:,1]=Depth
        if not self.training:
            rgbd_dict = {}
            for name, vals in rgb_depth_pct_per_level.items():
                if not vals:
                    continue
                packed = torch.stack([p.to(device) for p in vals], dim=0)  # (L,B,C)
                rgbd_dict[name] = packed
                
            if rgbd_dict:
                out['rgb_depth_pct'] = rgbd_dict


        # n_boxes per sample (same for all in the batch)
        B = out['pred_boxes'].shape[0]
        n_boxes_per_sample = torch.full(
            (B,), fill_value=self.transformer.num_det_queries, dtype=torch.long, device=device
        )


        # -------------------------------------------------------------
        # Grasp diffusion training / DDIM inference
        # -------------------------------------------------------------
        if self.use_grasp_diffusion and self.grasp_diffusion is not None:
            # Training:
            # Hungarian matching per detected object slot against
            # targets[b]["gt_grasps"][inst_id], then DDPM epsilon loss.
            if diffusion_targets is not None:

                diffusion_loss, diffusion_metrics = grasp_diffusion_loss(
                    diffusion_head=self.grasp_diffusion,
                    outs=out,
                    targets=diffusion_targets,
                    confidence_loss_weight=(
                        self.transformer.grasp_diffusion_confidence_loss_weight
                    ),
                )

                out["loss_grasp_diffusion"] = diffusion_loss
                out.update(diffusion_metrics)

            # Validation / inference:
            # Generate new grasp proposals using deterministic DDIM.
            if diffusion_sample_steps > 0:
                pose_hidden = out.get("pose_hidden", None)

                if pose_hidden is None:
                    raise RuntimeError(
                        "pose_hidden is missing. "
                        "Diffusion grasp generation requires with_pose=True."
                    )

                sampled_pose, sampled_confidence = self.grasp_diffusion.ddim_sample(
                    pose_hidden=pose_hidden,
                    steps=diffusion_sample_steps,
                )

                out["grasp_diffusion_pose_preds"] = sampled_pose
                out["grasp_diffusion_confidence"] = sampled_confidence


        return out, n_boxes_per_sample


def build_grasper(args, backbone):
    device = torch.device(args.device)
    backbone, is_rgbd = wrap_backbone(args, backbone)
    transformer = build_deforamble_transformer(args)
    model = Grasper(
        backbone,
        transformer,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        proj_kernel_size=args.proj_kernel_size,
        is_rgbd=is_rgbd
    )
    return model