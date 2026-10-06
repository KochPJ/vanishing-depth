
# loss_joint_with_cls.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from model.deformable_detr.util import box_ops
import numpy as np


def chamfer_symmetric(pred_pts: torch.Tensor, gt_pts: torch.Tensor) -> torch.Tensor:
    """
    CD = mean(min_{gt} ||pred-gt||^2) + mean(min_{pred} ||gt-pred||^2).
    pred_pts: [P,3], gt_pts: [G,3]
    """
    D2 = torch.cdist(pred_pts, gt_pts) ** 2
    return D2.min(dim=1).values.mean() + D2.min(dim=0).values.mean()

def soft_diversity_penalty(pred_pts: torch.Tensor, gt_pts: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """
    Soft GT->pred assignment; penalize deviation from uniform counts (G/P per pred).
    """
    P, G = pred_pts.shape[0], gt_pts.shape[0]
    D2 = torch.cdist(pred_pts, gt_pts) ** 2
    assign = F.softmax(-D2 / tau, dim=0)  # columns (GT) sum to 1 over preds
    counts = assign.sum(dim=1)            # [P]
    target = float(G) / max(P, 1)
    return ((counts - target) ** 2).mean()

def repulsion_loss(pred_pts: torch.Tensor, r: float = 0.02) -> torch.Tensor:
    """
    Hinge repulsion for pred-pred pairs within radius r.
    """
    P = pred_pts.shape[0]
    if P <= 1:
        return pred_pts.new_zeros(())
    Dpp = torch.cdist(pred_pts, pred_pts)
    Dpp = Dpp + torch.eye(P, device=pred_pts.device) * 1e6
    return F.relu(r - Dpp).mean()


class HungarianMatcher:
    def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=2.0):
        self.cost_class = float(cost_class)
        self.cost_bbox  = float(cost_bbox)
        self.cost_giou  = float(cost_giou)

    @torch.no_grad()
    def __call__(self, pred_boxes, pred_logits, tgt_boxes, tgt_labels):
        dev = pred_boxes.device
        Q, C = pred_logits.shape
        M = tgt_boxes.shape[0]
        if Q == 0 or M == 0:
            return (torch.as_tensor([], dtype=torch.long, device=dev),
                    torch.as_tensor([], dtype=torch.long, device=dev))

        # Basic input sanitation
        if not torch.isfinite(pred_boxes).all():
            pred_boxes = torch.where(torch.isfinite(pred_boxes), pred_boxes, torch.zeros_like(pred_boxes))
        if not torch.isfinite(pred_logits).all():
            pred_logits = torch.where(torch.isfinite(pred_logits), pred_logits, torch.zeros_like(pred_logits))
        if not torch.isfinite(tgt_boxes).all():
            tgt_boxes = torch.where(torch.isfinite(tgt_boxes), tgt_boxes, torch.zeros_like(tgt_boxes))

        out_prob = pred_logits.softmax(-1).clamp_min(1e-9)  # avoid log(0)
        # guard labels
        if tgt_labels.numel() > 0:
            assert (tgt_labels >= 0).all() and (tgt_labels < C).all(), f"labels out of range [0,{C-1}]"
        tgt_labels_exp = tgt_labels.view(1, M).expand(Q, M)

        cost_class = -torch.log(out_prob.gather(1, tgt_labels_exp))            # [Q,M]
        cost_bbox  = torch.cdist(pred_boxes, tgt_boxes, p=1)                   # [Q,M]
        giou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(pred_boxes),
            box_ops.box_cxcywh_to_xyxy(tgt_boxes)
        )
        cost_giou = 1.0 - giou

        # Replace non-finite with large finite number to keep LAP stable
        big = torch.tensor(1e6, device=dev)
        for t in (cost_class, cost_bbox, cost_giou):
            t.masked_fill_(~torch.isfinite(t), big)

        C_total = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C_np = C_total.detach().cpu().numpy()
        if not np.isfinite(C_np).all():
            C_np = np.where(np.isfinite(C_np), C_np, 1e6)

        i_idx, j_idx = linear_sum_assignment(C_np)
        return (torch.as_tensor(i_idx, dtype=torch.long, device=dev),
                torch.as_tensor(j_idx, dtype=torch.long, device=dev))


class JointCriterion(nn.Module):
    """
    Detection + classification + object-property losses, with aux on both detection and object heads.
    """
    def __init__(self,
                 loss_bbox_coef=5.0,
                 loss_giou_coef=2.0,
                 loss_obj_scale_coef=1.0,
                 loss_obj_pos_coef=1.0,
                 loss_obj_cog_coef=0.5,
                 loss_obj_shape_coef=1.0,           
                 loss_obj_shape_cd_coef=1.0,
                 loss_obj_shape_div_coef=0.1,
                 loss_obj_shape_rep_coef=0.05,
                 loss_obj_vol_coef=1.0,
                 shape_tau=0.05,
                 shape_r=0.02,
                 w_cls=1.0,
                 eos_coef=0.1,
                 set_cost_class=1.0,
                 set_cost_bbox=1.0,
                 set_cost_giou=2.0,
                 with_obj=True,
                 with_grasp=False,
                 shape_gt_subsample=512,
                 aux_det_weight=1.0,
                 aux_obj_weight=1.0,
                 aux_det_match_reuse=True,
                 aux_obj_match_reuse=True):
        super().__init__()
        self.matcher = HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou)

        self.w_cls   = float(w_cls)
        self.eos_coef = float(eos_coef)
        self.w_bbox  = float(loss_bbox_coef)
        self.w_giou  = float(loss_giou_coef)
        self.w_scale = float(loss_obj_scale_coef)
        self.w_pos   = float(loss_obj_pos_coef)
        self.w_cog   = float(loss_obj_cog_coef)

        # shape composite weights
        self.w_shape     = float(loss_obj_shape_coef)
        self.w_shape_cd  = float(loss_obj_shape_cd_coef)
        self.w_shape_div = float(loss_obj_shape_div_coef)
        self.w_shape_rep = float(loss_obj_shape_rep_coef)
        self.shape_tau   = float(shape_tau)
        self.shape_r     = float(shape_r)

        self.w_vol   = float(loss_obj_vol_coef)

        self.with_obj = bool(with_obj)
        self.with_grasp = bool(with_grasp)
        self.shape_gt_subsample = int(shape_gt_subsample)

        self.aux_det_weight = float(aux_det_weight)
        self.aux_obj_weight = float(aux_obj_weight)

        self.aux_det_match_reuse = bool(aux_det_match_reuse)
        self.aux_obj_match_reuse = bool(aux_obj_match_reuse)
        self._warned_obj_recompute = False


    def forward(self, outs, targets):
        device = outs["pred_boxes"].device
        pred_boxes_all  = outs["pred_boxes"]   # [B,Q,4]
        pred_logits_all = outs["pred_logits"]  # [B,Q,C]
        B, Q, C = pred_logits_all.shape

        # Final-layer object tensors (no stacks)
        obj_pos_final   = outs.get("obj_positions")   # [B,K,3]
        obj_cog_final   = outs.get("obj_cogs")        # [B,K,3]
        obj_scale_final = outs.get("obj_scale")       # [B,K,1]
        obj_pts_final   = outs.get("obj_points")      # [B,K,D,3]
        obj_vol_final   = outs.get("obj_vol")         # [B,K,1]
        obj_index       = outs.get("obj_index")       # [B,K]
        
        eps = 1e-6
        class_weights = torch.ones((C,), device=device)
        class_weights[0] = self.eos_coef

        loss_bbox = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)
        loss_cls  = torch.tensor(0.0, device=device)
        n_pairs   = 0

        loss_scale = torch.tensor(0.0, device=device)
        loss_pos   = torch.tensor(0.0, device=device)
        loss_cog   = torch.tensor(0.0, device=device)
        loss_shape = torch.tensor(0.0, device=device)
        loss_vol   = torch.tensor(0.0, device=device)
        n_obj_pairs = 0

        final_indices = []

        # Final detection + classification + object losses
        for b in range(B):
            pred_boxes  = pred_boxes_all[b]
            pred_logits = pred_logits_all[b]
            tgt_boxes   = targets[b]["boxes"]
            tgt_labels  = targets[b]["labels"]

            idx_pred, idx_tgt = self.matcher(pred_boxes, pred_logits, tgt_boxes, tgt_labels)
            final_indices.append((idx_pred, idx_tgt))

            if idx_pred.numel() > 0:
                n_pairs += idx_pred.numel()
                pb = pred_boxes[idx_pred]
                tb = tgt_boxes[idx_tgt]
                loss_bbox += F.l1_loss(pb, tb, reduction="sum")
                giou = box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(pb),
                    box_ops.box_cxcywh_to_xyxy(tb)
                )
                loss_giou += (1.0 - torch.diag(giou)).sum()

            # CE over all queries (bg=0 unmatched)
            tgt_full = torch.zeros((Q,), dtype=torch.long, device=device)
            if idx_pred.numel() > 0:
                tgt_full[idx_pred] = tgt_labels[idx_tgt]
            loss_cls += F.cross_entropy(pred_logits, tgt_full, weight=class_weights, reduction="sum")

            # Object-property losses (matched pairs only) using final-layer tensors
            if self.with_obj and idx_pred.numel() > 0:
                det_to_slot = {int(det_q): slot for slot, det_q in enumerate(obj_index[b].tolist())}
                valid_mask = targets[b]["obj_valid"]

                for p_i, t_j in zip(idx_pred.tolist(), idx_tgt.tolist()):
                    if not valid_mask[t_j]:
                        continue
                    if p_i not in det_to_slot:
                        continue
                    slot = det_to_slot[p_i]

                    pred_scale  = obj_scale_final[b, slot]
                    pred_pos    = obj_pos_final[b, slot]
                    pred_cog    = obj_cog_final[b, slot]
                    pred_points = obj_pts_final[b, slot]
                    pred_vol    = obj_vol_final[b, slot]

                    gt_scale = targets[b]["obj_scale"][t_j]
                    gt_pos   = targets[b]["obj_pos"][t_j]
                    gt_cog   = targets[b]["obj_cog"][t_j]
                    gt_list  = targets[b]["obj_points_list"][t_j]
                    gt_vol   = targets[b]["obj_vol"][t_j]

                    # Optional GT subsample for speed
                    if self.shape_gt_subsample > 0 and gt_list.shape[0] > self.shape_gt_subsample:
                        idxs = torch.randperm(gt_list.shape[0], device=device)[:self.shape_gt_subsample]
                        gt_list = gt_list[idxs]

                    loss_scale += F.mse_loss(pred_scale, gt_scale, reduction="sum")
                    loss_pos   += F.mse_loss(pred_pos,   gt_pos,   reduction="sum")
                    loss_cog   += F.mse_loss(pred_cog,   gt_cog,   reduction="sum")

                    cd  = chamfer_symmetric(pred_points, gt_list)
                    div = soft_diversity_penalty(pred_points, gt_list, tau=self.shape_tau)
                    rep = repulsion_loss(pred_points, r=self.shape_r)
                    loss_shape += (self.w_shape_cd * cd + self.w_shape_div * div + self.w_shape_rep * rep)

                    loss_vol   += F.mse_loss(pred_vol, gt_vol, reduction="sum")
                    n_obj_pairs += 1

        # Final aggregated losses
        losses = {}
        losses["loss_cls"]  = self.w_cls  * (loss_cls  / (B * Q))
        losses["loss_bbox"] = self.w_bbox * (loss_bbox / (n_pairs + eps)) if n_pairs > 0 else torch.tensor(0.0, device=device)
        losses["loss_giou"] = self.w_giou * (loss_giou / (n_pairs + eps)) if n_pairs > 0 else torch.tensor(0.0, device=device)

        if self.with_obj and n_obj_pairs > 0:
            losses["loss_obj_scale"] = self.w_scale * (loss_scale / (n_obj_pairs + eps))
            losses["loss_obj_pos"]   = self.w_pos   * (loss_pos   / (n_obj_pairs + eps))
            losses["loss_obj_cog"]   = self.w_cog   * (loss_cog   / (n_obj_pairs + eps))
            losses["loss_obj_shape"] = self.w_shape * (loss_shape / (n_obj_pairs + eps))
            losses["loss_obj_vol"]   = self.w_vol   * (loss_vol   / (n_obj_pairs + eps))
        else:
            losses["loss_obj_scale"] = torch.tensor(0.0, device=device)
            losses["loss_obj_pos"]   = torch.tensor(0.0, device=device)
            losses["loss_obj_cog"]   = torch.tensor(0.0, device=device)
            losses["loss_obj_shape"] = torch.tensor(0.0, device=device)
            losses["loss_obj_vol"]   = torch.tensor(0.0, device=device)

        # Aux detection losses
        aux_list = outs.get("aux_outputs", [])
        if aux_list:
            for i, aux in enumerate(aux_list):
                aux_boxes  = aux["pred_boxes"]   # [B,Q,4]
                aux_logits = aux["pred_logits"]  # [B,Q,C]
                aux_lb = torch.tensor(0.0, device=device)
                aux_lg = torch.tensor(0.0, device=device)
                aux_lc = torch.tensor(0.0, device=device)
                aux_n  = 0
                for b in range(B):
                    pb = aux_boxes[b]
                    pl = aux_logits[b]
                    tb = targets[b]["boxes"]
                    tl = targets[b]["labels"]

                    if self.aux_det_match_reuse:
                        idx_pred_i, idx_tgt_i = final_indices[b]
                    else:
                        idx_pred_i, idx_tgt_i = self.matcher(pb, pl, tb, tl)

                    if idx_pred_i.numel() > 0:
                        aux_n += idx_pred_i.numel()
                        psel = pb[idx_pred_i]
                        tsel = tb[idx_tgt_i]
                        aux_lb += F.l1_loss(psel, tsel, reduction="sum")
                        giou = box_ops.generalized_box_iou(
                            box_ops.box_cxcywh_to_xyxy(psel),
                            box_ops.box_cxcywh_to_xyxy(tsel)
                        )
                        aux_lg += (1.0 - torch.diag(giou)).sum()

                    Q_ = pl.shape[0]
                    tgt_full = torch.zeros((Q_,), dtype=torch.long, device=device)
                    if idx_pred_i.numel() > 0:
                        tgt_full[idx_pred_i] = tl[idx_tgt_i]
                    aux_lc += F.cross_entropy(pl, tgt_full, weight=class_weights, reduction="sum")

                losses[f"loss_bbox_{i}"] = self.aux_det_weight * self.w_bbox * (aux_lb / (aux_n + eps)) if aux_n > 0 else torch.tensor(0.0, device=device)
                losses[f"loss_giou_{i}"] = self.aux_det_weight * self.w_giou * (aux_lg / (aux_n + eps)) if aux_n > 0 else torch.tensor(0.0, device=device)
                losses[f"loss_cls_{i}"]  = self.aux_det_weight * self.w_cls  * (aux_lc / (B * Q))

        # Aux object losses: use aux_list per-layer object preds, not stacks
        if self.with_obj and aux_list:
            for l, aux in enumerate(aux_list):
                
                aux_ls = torch.tensor(0.0, device=device)
                aux_lp = torch.tensor(0.0, device=device)
                aux_lcog = torch.tensor(0.0, device=device)
                aux_lsh = torch.tensor(0.0, device=device)
                aux_lvol = torch.tensor(0.0, device=device)
                aux_n = 0
                for b in range(B):
                    # detection matching per layer (reuse or recompute via aux detections)
                    if self.aux_obj_match_reuse:
                        idx_pred_i, idx_tgt_i = final_indices[b]
                    else:
                        pb = aux["pred_boxes"][b]
                        pl = aux["pred_logits"][b]
                        tb = targets[b]["boxes"]
                        tl = targets[b]["labels"]
                        idx_pred_i, idx_tgt_i = self.matcher(pb, pl, tb, tl)

                    if idx_pred_i.numel() == 0 or obj_index is None:
                        continue

                    det_to_slot = {int(det_q): slot for slot, det_q in enumerate(obj_index[b].tolist())}
                    valid_mask = targets[b]["obj_valid"]

                    for p_i, t_j in zip(idx_pred_i.tolist(), idx_tgt_i.tolist()):
                        if not valid_mask[t_j]:
                            continue
                        if p_i not in det_to_slot:
                            continue
                        slot = det_to_slot[p_i]

                        pred_scale  = aux["obj_scale"][b, slot]
                        pred_pos    = aux["obj_positions"][b, slot]
                        pred_cog    = aux["obj_cogs"][b, slot]
                        pred_points = aux["obj_points"][b, slot]
                        pred_vol    = aux["obj_vol"][b, slot]

                        gt_scale = targets[b]["obj_scale"][t_j]
                        gt_pos   = targets[b]["obj_pos"][t_j]
                        gt_cog   = targets[b]["obj_cog"][t_j]
                        gt_list  = targets[b]["obj_points_list"][t_j]
                        gt_vol   = targets[b]["obj_vol"][t_j]

                        if self.shape_gt_subsample > 0 and gt_list.shape[0] > self.shape_gt_subsample:
                            idxs = torch.randperm(gt_list.shape[0], device=device)[:self.shape_gt_subsample]
                            gt_list = gt_list[idxs]

                        aux_ls  += F.mse_loss(pred_scale, gt_scale, reduction="sum")
                        aux_lp  += F.mse_loss(pred_pos,   gt_pos,   reduction="sum")
                        aux_lcog+= F.mse_loss(pred_cog,   gt_cog,   reduction="sum")

                        cd  = chamfer_symmetric(pred_points, gt_list)
                        div = soft_diversity_penalty(pred_points, gt_list, tau=self.shape_tau)
                        rep = repulsion_loss(pred_points, r=self.shape_r)
                        aux_lsh += (self.w_shape_cd * cd + self.w_shape_div * div + self.w_shape_rep * rep)

                        aux_lvol += F.mse_loss(pred_vol, gt_vol, reduction="sum")
                        aux_n   += 1

                if aux_n > 0:
                    losses[f"loss_obj_scale_{l}"] = self.aux_obj_weight * self.w_scale * (aux_ls  / (aux_n + eps))
                    losses[f"loss_obj_pos_{l}"]   = self.aux_obj_weight * self.w_pos   * (aux_lp  / (aux_n + eps))
                    losses[f"loss_obj_cog_{l}"]   = self.aux_obj_weight * self.w_cog   * (aux_lcog/ (aux_n + eps))
                    losses[f"loss_obj_shape_{l}"] = self.aux_obj_weight * (aux_lsh / (aux_n + eps))
                    losses[f"loss_obj_vol_{l}"]   = self.aux_obj_weight * self.w_vol   * (aux_lvol/ (aux_n + eps))
                else:
                    losses[f"loss_obj_scale_{l}"] = torch.tensor(0.0, device=device)
                    losses[f"loss_obj_pos_{l}"]   = torch.tensor(0.0, device=device)
                    losses[f"loss_obj_cog_{l}"]   = torch.tensor(0.0, device=device)
                    losses[f"loss_obj_shape_{l}"] = torch.tensor(0.0, device=device)
                    losses[f"loss_obj_vol_{l}"]   = torch.tensor(0.0, device=device)

        if self.with_grasp:
            losses["loss_grasp"] = torch.tensor(0.0, device=device)

        return losses

    def forward_old(self, outs, targets):
        device = outs["pred_boxes"].device
        pred_boxes_all  = outs["pred_boxes"]   # [B,Q,4]
        pred_logits_all = outs["pred_logits"]  # [B,Q,C]
        B, Q, C = pred_logits_all.shape

        if self.with_obj:
            # Object stacks and mapping
            obj_pos_stack   = outs["obj_positions"]   # [L,B,K,3]
            obj_cog_stack   = outs["obj_cogs"]        # [L,B,K,3]
            obj_scale_stack = outs["obj_scale"]       # [L,B,K,1]
            obj_pts_stack   = outs["obj_points"]      # [L,B,K,D,3]
            obj_vol_stack   = outs["obj_vol"]         # [L,B,K,1]
            obj_index       = outs["obj_index"]       # [B,K]

            # Final object tensors
            if isinstance(obj_pos_stack, list):
                obj_pos_final   = obj_pos_stack[-1]
                obj_cog_final   = obj_cog_stack[-1]
                obj_scale_final = obj_scale_stack[-1]
                obj_pts_final   = obj_pts_stack[-1]
                obj_vol_final   = obj_vol_stack[-1]
            else:
                obj_pos_final   = obj_pos_stack[-1]
                obj_cog_final   = obj_cog_stack[-1]
                obj_scale_final = obj_scale_stack[-1]
                obj_pts_final   = obj_pts_stack[-1]
                obj_vol_final   = obj_vol_stack[-1]

        eps = 1e-6
        class_weights = torch.ones((C,), device=device)
        class_weights[0] = self.eos_coef

        loss_bbox = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)
        loss_cls  = torch.tensor(0.0, device=device)
        n_pairs   = 0

        loss_scale = torch.tensor(0.0, device=device)
        loss_pos   = torch.tensor(0.0, device=device)
        loss_cog   = torch.tensor(0.0, device=device)
        loss_shape = torch.tensor(0.0, device=device)
        loss_vol   = torch.tensor(0.0, device=device)
        n_obj_pairs = 0

        final_indices = []

        # Final detection + classification + object losses
        for b in range(B):
            pred_boxes  = pred_boxes_all[b]
            pred_logits = pred_logits_all[b]
            tgt_boxes   = targets[b]["boxes"]
            tgt_labels  = targets[b]["labels"]

            idx_pred, idx_tgt = self.matcher(pred_boxes, pred_logits, tgt_boxes, tgt_labels)
            final_indices.append((idx_pred, idx_tgt))

            if idx_pred.numel() > 0:
                n_pairs += idx_pred.numel()
                pb = pred_boxes[idx_pred]
                tb = tgt_boxes[idx_tgt]
                loss_bbox += F.l1_loss(pb, tb, reduction="sum")
                giou = box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(pb),
                    box_ops.box_cxcywh_to_xyxy(tb)
                )
                loss_giou += (1.0 - torch.diag(giou)).sum()

            # CE over all queries (bg=0 unmatched)
            tgt_full = torch.zeros((Q,), dtype=torch.long, device=device)
            if idx_pred.numel() > 0:
                tgt_full[idx_pred] = tgt_labels[idx_tgt]
            loss_cls += F.cross_entropy(pred_logits, tgt_full, weight=class_weights, reduction="sum")

            # Object-property losses (matched pairs only)
            if idx_pred.numel() > 0:
                det_to_slot = {int(det_q): slot for slot, det_q in enumerate(obj_index[b].tolist())}
                valid_mask = targets[b]["obj_valid"]
                for p_i, t_j in zip(idx_pred.tolist(), idx_tgt.tolist()):
                    if not valid_mask[t_j]:
                        continue
                    if p_i not in det_to_slot:
                        continue
                    slot = det_to_slot[p_i]

                    pred_scale  = obj_scale_final[b, slot]
                    pred_pos    = obj_pos_final[b, slot]
                    pred_cog    = obj_cog_final[b, slot]
                    pred_points = obj_pts_final[b, slot]
                    pred_vol    = obj_vol_final[b, slot]

                    gt_scale = targets[b]["obj_scale"][t_j]
                    gt_pos   = targets[b]["obj_pos"][t_j]
                    gt_cog   = targets[b]["obj_cog"][t_j]
                    gt_list  = targets[b]["obj_points_list"][t_j]
                    gt_vol   = targets[b]["obj_vol"][t_j]

                    # Optional GT subsample for speed
                    if self.shape_gt_subsample > 0 and gt_list.shape[0] > self.shape_gt_subsample:
                        idxs = torch.randperm(gt_list.shape[0], device=device)[:self.shape_gt_subsample]
                        gt_list = gt_list[idxs]

                    loss_scale += F.mse_loss(pred_scale, gt_scale, reduction="sum")
                    loss_pos   += F.mse_loss(pred_pos,   gt_pos,   reduction="sum")
                    loss_cog   += F.mse_loss(pred_cog,   gt_cog,   reduction="sum")

                    cd  = chamfer_symmetric(pred_points, gt_list)
                    div = soft_diversity_penalty(pred_points, gt_list, tau=self.shape_tau)
                    rep = repulsion_loss(pred_points, r=self.shape_r)
                    loss_shape += (self.w_shape_cd * cd + self.w_shape_div * div + self.w_shape_rep * rep)

                    loss_vol   += F.mse_loss(pred_vol, gt_vol, reduction="sum")
                    n_obj_pairs += 1

        # Final aggregated losses
        losses = {}
        losses["loss_cls"]  = self.w_cls  * (loss_cls  / (B * Q))
        losses["loss_bbox"] = self.w_bbox * (loss_bbox / (n_pairs + eps)) if n_pairs > 0 else torch.tensor(0.0, device=device)
        losses["loss_giou"] = self.w_giou * (loss_giou / (n_pairs + eps)) if n_pairs > 0 else torch.tensor(0.0, device=device)

        if self.with_obj and n_obj_pairs > 0:
            losses["loss_obj_scale"] = self.w_scale * (loss_scale / (n_obj_pairs + eps))
            losses["loss_obj_pos"]   = self.w_pos   * (loss_pos   / (n_obj_pairs + eps))
            losses["loss_obj_cog"]   = self.w_cog   * (loss_cog   / (n_obj_pairs + eps))
            losses["loss_obj_shape"] = self.w_shape * (loss_shape / (n_obj_pairs + eps))
            losses["loss_obj_vol"]   = self.w_vol   * (loss_vol   / (n_obj_pairs + eps))
        else:
            losses["loss_obj_scale"] = torch.tensor(0.0, device=device)
            losses["loss_obj_pos"]   = torch.tensor(0.0, device=device)
            losses["loss_obj_cog"]   = torch.tensor(0.0, device=device)
            losses["loss_obj_shape"] = torch.tensor(0.0, device=device)
            losses["loss_obj_vol"]   = torch.tensor(0.0, device=device)

        # Aux detection losses (per layer), reuse optional
        aux_list = outs.get("aux_outputs", [])
        if aux_list:
            for i, aux in enumerate(aux_list):
                aux_boxes  = aux["pred_boxes"]   # [B,Q,4]
                aux_logits = aux["pred_logits"]  # [B,Q,C]
                aux_lb = torch.tensor(0.0, device=device)
                aux_lg = torch.tensor(0.0, device=device)
                aux_lc = torch.tensor(0.0, device=device)
                aux_n  = 0
                for b in range(B):
                    pb = aux_boxes[b]
                    pl = aux_logits[b]
                    tb = targets[b]["boxes"]
                    tl = targets[b]["labels"]

                    if self.aux_det_match_reuse:
                        idx_pred_i, idx_tgt_i = final_indices[b]
                    else:
                        idx_pred_i, idx_tgt_i = self.matcher(pb, pl, tb, tl)

                    if idx_pred_i.numel() > 0:
                        aux_n += idx_pred_i.numel()
                        psel = pb[idx_pred_i]
                        tsel = tb[idx_tgt_i]
                        aux_lb += F.l1_loss(psel, tsel, reduction="sum")
                        giou = box_ops.generalized_box_iou(
                            box_ops.box_cxcywh_to_xyxy(psel),
                            box_ops.box_cxcywh_to_xyxy(tsel)
                        )
                        aux_lg += (1.0 - torch.diag(giou)).sum()

                    Q_ = pl.shape[0]
                    tgt_full = torch.zeros((Q_,), dtype=torch.long, device=device)
                    if idx_pred_i.numel() > 0:
                        tgt_full[idx_pred_i] = tl[idx_tgt_i]
                    aux_lc += F.cross_entropy(pl, tgt_full, weight=class_weights, reduction="sum")

                losses[f"loss_bbox_{i}"] = self.aux_det_weight * self.w_bbox * (aux_lb / (aux_n + eps)) if aux_n > 0 else torch.tensor(0.0, device=device)
                losses[f"loss_giou_{i}"] = self.aux_det_weight * self.w_giou * (aux_lg / (aux_n + eps)) if aux_n > 0 else torch.tensor(0.0, device=device)
                losses[f"loss_cls_{i}"]  = self.aux_det_weight * self.w_cls  * (aux_lc / (B * Q))

        # Aux object losses (reuse matching or recompute per layer)
        if self.with_obj and (isinstance(obj_pos_stack, torch.Tensor) or isinstance(obj_pos_stack, list)):
            # number of object decoder layers in stacks
            L_obj = obj_pos_stack.shape[0] if hasattr(obj_pos_stack, "shape") else len(obj_pos_stack)
            for l in range(max(L_obj - 1, 0)):  # layers 0..L-2
                aux_ls = torch.tensor(0.0, device=device)
                aux_lp = torch.tensor(0.0, device=device)
                aux_lcog = torch.tensor(0.0, device=device)
                aux_lsh = torch.tensor(0.0, device=device)
                aux_lvol = torch.tensor(0.0, device=device)
                aux_n = 0
                for b in range(B):
                    # detection matching per layer (reuse or recompute via aux detections if provided)
                    if self.aux_obj_match_reuse or not aux_list:
                        idx_pred_i, idx_tgt_i = final_indices[b]
                    else:
                        # recompute using the detection aux outputs at the same layer index
                        pb = aux_list[l]["pred_boxes"][b]
                        pl = aux_list[l]["pred_logits"][b]
                        tb = targets[b]["boxes"]
                        tl = targets[b]["labels"]
                        idx_pred_i, idx_tgt_i = self.matcher(pb, pl, tb, tl)

                    if idx_pred_i.numel() == 0:
                        continue

                    det_to_slot = {int(det_q): slot for slot, det_q in enumerate(obj_index[b].tolist())}
                    valid_mask = targets[b]["obj_valid"]

                    for p_i, t_j in zip(idx_pred_i.tolist(), idx_tgt_i.tolist()):
                        if not valid_mask[t_j]: 
                            continue
                        if p_i not in det_to_slot: 
                            continue
                        slot = det_to_slot[p_i]

                        pred_scale  = obj_scale_stack[l][b, slot]
                        pred_pos    = obj_pos_stack[l][b, slot]
                        pred_cog    = obj_cog_stack[l][b, slot]
                        pred_points = obj_pts_stack[l][b, slot]
                        pred_vol    = obj_vol_stack[l][b, slot]

                        gt_scale = targets[b]["obj_scale"][t_j]
                        gt_pos   = targets[b]["obj_pos"][t_j]
                        gt_cog   = targets[b]["obj_cog"][t_j]
                        gt_list  = targets[b]["obj_points_list"][t_j]
                        gt_vol   = targets[b]["obj_vol"][t_j]

                        if self.shape_gt_subsample > 0 and gt_list.shape[0] > self.shape_gt_subsample:
                            idxs = torch.randperm(gt_list.shape[0], device=device)[:self.shape_gt_subsample]
                            gt_list = gt_list[idxs]

                        aux_ls  += F.mse_loss(pred_scale, gt_scale, reduction="sum")
                        aux_lp  += F.mse_loss(pred_pos,   gt_pos,   reduction="sum")
                        aux_lcog+= F.mse_loss(pred_cog,   gt_cog,   reduction="sum")

                        cd  = chamfer_symmetric(pred_points, gt_list)
                        div = soft_diversity_penalty(pred_points, gt_list, tau=self.shape_tau)
                        rep = repulsion_loss(pred_points, r=self.shape_r)
                        aux_lsh += (self.w_shape_cd * cd + self.w_shape_div * div + self.w_shape_rep * rep)

                        aux_lvol += F.mse_loss(pred_vol, gt_vol, reduction="sum")
                        aux_n   += 1

                if aux_n > 0:
                    losses[f"loss_obj_scale_{l}"] = self.aux_obj_weight * self.w_scale * (aux_ls  / (aux_n + eps))
                    losses[f"loss_obj_pos_{l}"]   = self.aux_obj_weight * self.w_pos   * (aux_lp  / (aux_n + eps))
                    losses[f"loss_obj_cog_{l}"]   = self.aux_obj_weight * self.w_cog   * (aux_lcog/ (aux_n + eps))
                    losses[f"loss_obj_shape_{l}"] = self.aux_obj_weight * (aux_lsh / (aux_n + eps))
                    losses[f"loss_obj_vol_{l}"]   = self.aux_obj_weight * self.w_vol   * (aux_lvol/ (aux_n + eps))
                else:
                    losses[f"loss_obj_scale_{l}"] = torch.tensor(0.0, device=device)
                    losses[f"loss_obj_pos_{l}"]   = torch.tensor(0.0, device=device)
                    losses[f"loss_obj_cog_{l}"]   = torch.tensor(0.0, device=device)
                    losses[f"loss_obj_shape_{l}"] = torch.tensor(0.0, device=device)
                    losses[f"loss_obj_vol_{l}"]   = torch.tensor(0.0, device=device)

        if self.with_grasp:
            losses["loss_grasp"] = torch.tensor(0.0, device=device)

        return losses