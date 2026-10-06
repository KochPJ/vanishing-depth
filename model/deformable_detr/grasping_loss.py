
# loss_joint_with_cls.py (replace JointCriterion with this focal version)

import torch._dynamo as dynamo
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import box_ops
from scipy.optimize import linear_sum_assignment
import numpy as np
import math 


def soft_best_scores(p_b: torch.Tensor, tau: float = 0.2, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    # Optional masking per-slot proposal validity
    if valid_mask is not None:
        # invalid proposals get -inf so they don't get weight
        p_b = p_b.masked_fill(~valid_mask, float('-inf'))
    # subtract rowwise max for numerical stability
    z = p_b.max(dim=1, keepdim=True).values
    w = torch.softmax((p_b - z) / tau, dim=1)            # [K,Np]
    best_scores = (w * p_b).sum(dim=1)                   # [K]
    return best_scores

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1); w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def quat_geodesic_minimal(q1, q2, eps=1e-6):
    q1 = F.normalize(q1, dim=-1); q2 = F.normalize(q2, dim=-1)
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    q2a = torch.where(dot < 0, -q2, q2)
    qinv = torch.stack([q1[...,0], -q1[...,1], -q1[...,2], -q1[...,3]], dim=-1)
    qrel = F.normalize(quat_mul(qinv, q2a), dim=-1)
    w = qrel[...,0].abs().clamp_min(eps)
    v = qrel[...,1:]
    return 2.0 * torch.atan2(v.norm(dim=-1), w)  # radians in [0,π]

def select_gt_prototypes(t_gt, q_gt, s_gt, K_req, s_t=0.10, w_q=0.05):
    G = t_gt.shape[0]
    Kp = min(int(K_req), int(G))
    if Kp <= 0:
        return torch.empty((0,), dtype=torch.long, device=t_gt.device)
    if G <= Kp:
        return torch.arange(G, device=t_gt.device)

    # FPS in joint space
    idxs = [int(torch.argmax(s_gt).item())]
    Dt = torch.cdist(t_gt, t_gt) / s_t                       # [G,G]
    q1 = q_gt.unsqueeze(1).expand(G, G, 4)
    q2 = q_gt.unsqueeze(0).expand(G, G, 4)
    Dq = quat_geodesic_minimal(q1, q2) / math.pi             # [G,G]
    D = Dt + w_q * Dq

    for _ in range(1, Kp):
        cand = D[idxs].min(dim=0).values
        # prefer higher-score GTs (use +s_gt, not +(1-s_gt))
        cand = cand + 0.05 * s_gt
        nxt = int(torch.argmax(cand).item())
        idxs.append(nxt)
    return torch.as_tensor(idxs, device=t_gt.device)

def build_gt_clusters(t_gt, q_gt, s_gt, K_req, s_t=0.10, w_q=0.05):
    """
    Return:
      proto_idx [Kp], t_cent [Kp,3], q_cent [Kp,4],
      clusters: list of length Kp with each entry a 1D LongTensor of GT indices.
    """
    G = t_gt.shape[0]
    proto_idx = select_gt_prototypes(t_gt, q_gt, s_gt, K_req, s_t=s_t, w_q=w_q)  # [Kp]
    Kp = int(proto_idx.numel())
    if Kp == 0:
        return proto_idx, t_gt.new_zeros((0,3)), q_gt.new_zeros((0,4)), []

    t_cent = t_gt[proto_idx]  # [Kp,3]
    q_cent = q_gt[proto_idx]  # [Kp,4]

    # Assign GTs to nearest prototype
    Dt = torch.cdist(t_gt, t_cent) / s_t                        # [G,Kp]
    q1 = q_gt.unsqueeze(1).expand(G, Kp, 4)
    q2 = q_cent.unsqueeze(0).expand(G, Kp, 4)
    Dq = quat_geodesic_minimal(q1, q2) / math.pi                # [G,Kp]
    Dj = Dt + w_q * Dq                                          # [G,Kp]

    k_nn = Dj.argmin(dim=1)                                     # [G]
    clusters = [torch.nonzero(k_nn == k, as_tuple=False).squeeze(1) for k in range(Kp)]
    return proto_idx, t_cent, q_cent, clusters

def pred_cluster_softmin(t_pred, q_pred, t_gt, q_gt, s_gt, clusters,
                         s_t=0.10, w_t=1.0, w_q=0.05, t_gate=0.05, tau=0.2, score_lambda=0.05):
    Np = t_pred.shape[0]; Kp = len(clusters)
    if Kp == 0:
        return t_pred.new_zeros((Np, 0))
    L_ik = []
    for k in range(Kp):
        idx = clusters[k]
        if idx.numel() == 0:
            L_ik.append(t_pred.new_zeros((Np,)))
            continue
        tg = t_gt[idx]; qg = q_gt[idx]; sg = s_gt[idx]
        Dt = torch.cdist(t_pred, tg) / s_t                      # [Np,Mk]
        q1 = q_pred.unsqueeze(1).expand(Np, tg.size(0), 4)
        q2 = qg.unsqueeze(0).expand(Np, tg.size(0), 4)
        Dq = quat_geodesic_minimal(q1, q2) / math.pi            # [Np,Mk]
        gate = torch.sigmoid((t_gate - Dt))
        C = w_t*Dt + (w_q*gate)*Dq + score_lambda*(1.0 - sg.unsqueeze(0))
        m = C.min(dim=1, keepdim=True).values
        w = torch.softmax(-(C - m) / tau, dim=1)                # [Np,Mk]
        L_ik.append((w * C).sum(dim=1))                         # [Np]
    return torch.stack(L_ik, dim=1)                             # [Np,Kp]

def sinkhorn_log(K_log: torch.Tensor,
                 a: torch.Tensor | None = None,
                 b: torch.Tensor | None = None,
                 n_iter: int = 30,
                 eps: float = 1e-9) -> torch.Tensor:
    """
    Log-domain Sinkhorn (stable), supports 2D [N,M] or 3D [B,N,M].

    Args:
        K_log: log-kernel, e.g. K_log = -(C / tau); shape [N,M] or [B,N,M]
        a:     row marginals; if None => uniform; [N] or [B,N]
        b:     col marginals; if None => uniform; [M] or [B,M]
        n_iter: iterations
        eps:   clamp for logs

    Returns:
        P: transport plan, same batch shape as K_log (i.e., [N,M] or [B,N,M])
    """
    batched = (K_log.dim() == 3)
    if not batched:
        K_log = K_log.unsqueeze(0)  # [1,N,M]

    B, N, M = K_log.shape

    if a is None:
        a = torch.full((B, N), 1.0 / max(N, 1), device=K_log.device, dtype=K_log.dtype)
    else:
        a = a if a.dim() == 2 else a.unsqueeze(0).expand(B, -1)  # [B,N]

    if b is None:
        b = torch.full((B, M), 1.0 / max(M, 1), device=K_log.device, dtype=K_log.dtype)
    else:
        b = b if b.dim() == 2 else b.unsqueeze(0).expand(B, -1)  # [B,M]

    # log scalings
    u = torch.zeros_like(a)  # [B,N]
    v = torch.zeros_like(b)  # [B,M]

    for _ in range(n_iter):
        # u = log a - logsumexp(K + v, axis=cols)
        u = torch.log(a.clamp_min(eps)) - torch.logsumexp(K_log + v[:, None, :], dim=2)
        # v = log b - logsumexp(K + u, axis=rows)
        v = torch.log(b.clamp_min(eps)) - torch.logsumexp(K_log + u[:, :, None], dim=1)

    P_log = K_log + u[:, :, None] + v[:, None, :]           # [B,N,M]
    P = torch.exp(P_log).clamp_min(eps)

    return P if batched else P.squeeze(0)


def se3_repulsion(t_pred: torch.Tensor, q_pred: torch.Tensor,
                  r_t: float = 0.02, r_q_deg: float = 15.0,
                  w_t: float = 1.0, w_q: float = 1.0) -> torch.Tensor:
    # t_pred: [Np,3], q_pred: [Np,4]
    Np = t_pred.size(0)
    if Np <= 1:
        return t_pred.new_zeros(())

    # Translation
    Dt = torch.cdist(t_pred, t_pred)  # [Np,Np]
    Dt = Dt + torch.eye(Np, device=t_pred.device) * 1e6
    rep_t = F.relu(r_t - Dt).mean()

    # Orientation (geodesic in radians)
    q1 = q_pred.unsqueeze(1).expand(Np, Np, 4)
    q2 = q_pred.unsqueeze(0).expand(Np, Np, 4)
    ang = quat_geodesic_minimal(q1, q2)  # [Np,Np] radians
    ang = ang + torch.eye(Np, device=ang.device) * 1e6
    r_q = (r_q_deg * math.pi / 180.0)
    rep_q = F.relu(r_q - ang).mean()

    return w_t * rep_t + w_q * rep_q

def ot_loss(t_pred: torch.Tensor,
            q_pred: torch.Tensor,
            t_gt: torch.Tensor,
            q_gt: torch.Tensor,
            s_gt: torch.Tensor,
            s_t: float = 0.05,
            w_t: float = 1.0,
            w_q: float = 2.0,
            score_lambda: float = 0.05,
            tau: float = 0.2,
            n_iter: int = 30,
            use_score_as_mass: bool = False,
            reduction: str = "mean") -> torch.Tensor:
    """
    Single entropic OT over all GTs (no clustering).
    Computes cost C = w_t*Dt + w_q*Dq + score_lambda*(1-s_gt),
    solves OT with Sinkhorn, returns <P, C>.

    Shapes:
        t_pred: [Np,3], q_pred: [Np,4]
        t_gt:   [G,3], q_gt:   [G,4], s_gt: [G] in [0,1]

    Returns:
        scalar tensor loss (per-slot); set reduction=None to get raw sum.
    """
    device = t_pred.device
    Np, G = t_pred.size(0), t_gt.size(0)
    if Np == 0 or G == 0:
        return t_pred.new_zeros(())

    Dt = torch.cdist(t_pred, t_gt) / s_t                                # [Np,G]
    q1 = q_pred.unsqueeze(1).expand(Np, G, 4)
    q2 = q_gt.unsqueeze(0).expand(Np, G, 4)
    # Use your repository implementation
    # from .your_module import quat_geodesic_minimal
    Dq = quat_geodesic_minimal(q1, q2) / math.pi                        # [Np,G] in [0,1]

    C = w_t*Dt + w_q*Dq + score_lambda*(1.0 - s_gt.unsqueeze(0)) # [Np,G]

    #print('Dt', w_t, Dt.detach().mean(), Dt.detach().min(), Dt.detach().max())
    #print('Dq', w_q, gate.detach().mean(), gate.detach().min(), gate.detach().max(), Dq.detach().mean(), Dq.detach().min(), Dq.detach().max())
    #print('s_gt', score_lambda, s_gt.detach().mean(), s_gt.detach().min(), s_gt.detach().max())
    #input()

    # Entropic kernel
    K_log = -(C / tau)                                                  # [Np,G]

    # Marginals
    a = torch.full((Np,), 1.0 / max(Np, 1), device=device, dtype=C.dtype)  # [Np]
    if use_score_as_mass:
        mass = s_gt.clamp_min(1e-9)
        b = mass / mass.sum()
    else:
        b = torch.full((G,), 1.0 / max(G, 1), device=device, dtype=C.dtype)

    P = sinkhorn_log(K_log, a, b, n_iter=n_iter)              # [Np,G]
    loss = (P * C).sum()                                      # scalar

    if reduction == "mean":
        return loss / max(Np, 1)
    elif reduction == "sum" or reduction is None:
        return loss
    else:
        raise ValueError(f"reduction {reduction} not in [mean,sum,None]")


def pred_cluster_softmin_vectorized(
    t_pred: torch.Tensor,   # [Np,3]
    q_pred: torch.Tensor,   # [Np,4]
    t_gt: torch.Tensor,     # [G,3]
    q_gt: torch.Tensor,     # [G,4]
    s_gt: torch.Tensor,     # [G]
    cluster_ids: torch.Tensor,  # [G] long in [0..Kp-1]
    Kp: int,
    s_t: float = 0.10,
    w_t: float = 1.0,
    w_q: float = 0.05,
    t_gate: float = 0.05,
    tau: float = 0.2,
    score_lambda: float = 0.05,
) -> torch.Tensor:
    """
    Vectorized version of per-cluster soft-min. No Python loops.

    Returns:
        L_ik: [Np,Kp] where each column k is a soft-min over GTs in cluster k.
        Clusters with no GTs receive zeros.

    Notes:
        Uses your quat_geodesic_minimal.
    """
    device = t_pred.device
    Np = t_pred.size(0)
    G = t_gt.size(0)
    if Np == 0 or G == 0 or Kp == 0:
        return t_pred.new_zeros((Np, Kp))

    # Full cost to all GT once
    Dt = torch.cdist(t_pred, t_gt) / s_t                      # [Np,G]
    q1 = q_pred.unsqueeze(1).expand(Np, G, 4)
    q2 = q_gt.unsqueeze(0).expand(Np, G, 4)
    Dq = quat_geodesic_minimal(q1, q2) / math.pi              # [Np,G]
    gate = torch.sigmoid(t_gate - Dt)                         # [Np,G]
    C_full = w_t*Dt + (w_q*gate)*Dq + score_lambda*(1.0 - s_gt.unsqueeze(0))  # [Np,G]

    # One-hot masks for each cluster (Kp,G)
    cluster_ids = cluster_ids.to(device)
    counts = torch.bincount(cluster_ids, minlength=Kp)        # [Kp]
    has_gt = counts > 0
    if has_gt.any():
        # Construct mask [Kp,G]
        M_k = (cluster_ids[None, :] == torch.arange(Kp, device=device)[:, None])  # [Kp,G]
        # Only select non-empty clusters to avoid NaNs
        idx_k = torch.nonzero(has_gt, as_tuple=False).squeeze(1)                  # [Ksel]
        M_sel = M_k.index_select(0, idx_k)                                        # [Ksel,G]

        # Broadcast cost & mask
        C_b = C_full.unsqueeze(1)                     # [Np,1,G]
        C_b = C_b.expand(-1, M_sel.shape[0], -1)      # [Np,Ksel,G]
        M_b = M_sel.unsqueeze(0).expand(Np, -1, -1)   # [Np,Ksel,G]

        # Masked min for stabilization: set outside cluster to +inf
        C_masked = C_b.masked_fill(~M_b, float('inf'))            # [Np,Ksel,G]
        m = C_masked.min(dim=2, keepdim=True).values              # [Np,Ksel,1]

        # Softmax over masked entries: set masked logits to -inf
        logits = -(C_b - m) / tau                                 # [Np,Ksel,G]
        logits = logits.masked_fill(~M_b, float('-inf'))
        w = torch.softmax(logits, dim=2)                          # [Np,Ksel,G]

        L_sel = (w * C_b).sum(dim=2)                              # [Np,Ksel]

        # Scatter back to full Kp columns, zeros for empty clusters
        L_full = C_full.new_zeros((Np, Kp))
        L_full.index_copy_(1, idx_k, L_sel)
    else:
        L_full = C_full.new_zeros((Np, Kp))

    return L_full

def sinkhorn_assign_to_clusters(t_pred, q_pred, t_cent, q_cent, s_t=0.10, w_q=0.05, tau_a=0.2):
    Np = t_pred.shape[0]; Kp = t_cent.shape[0]
    if Kp == 0:
        return t_pred.new_zeros((Np, 0))
    Dt = torch.cdist(t_pred, t_cent) / s_t                      # [Np,Kp]
    q1 = q_pred.unsqueeze(1).expand(Np, Kp, 4)
    q2 = q_cent.unsqueeze(0).expand(Np, Kp, 4)
    Dq = quat_geodesic_minimal(q1, q2) / math.pi                # [Np,Kp]
    Cpc = Dt + w_q * Dq
    m = Cpc.min(dim=1, keepdim=True).values
    K0 = torch.exp(-(Cpc - m) / tau_a)                          # [Np,Kp]
    a = torch.full((Np,), 1.0 / Np, device=K0.device)
    b = torch.full((Kp,), 1.0 / Kp, device=K0.device)
    # Sinkhorn
    u = torch.ones_like(a); v = torch.ones_like(b)
    eps = 1e-6
    for _ in range(30):
        u = a / (K0 @ v).clamp_min(eps)
        v = b / (K0.T @ u).clamp_min(eps)
    P = (u.unsqueeze(1) * K0) * v.unsqueeze(0)                  # [Np,Kp]
    return P
    

def _quat_wxyz_to_R(q):
    # q: (...,4) [w,x,y,z]
    w, x, y, z = q.unbind(-1)
    # normalization safe-guard
    s = torch.clamp((w*w + x*x + y*y + z*z), min=1e-12).sqrt()
    w = w / s; x = x / s; y = y / s; z = z / s
    # rotation matrix (vectorized)
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
    ], dim=-1)
    return R.view(*q.shape[:-1], 3, 3)

def chamfer_symmetric_torch(P, Q):
    # P: (N,3), Q:(M,3)
    if P.numel() == 0 or Q.numel() == 0:
        return P.new_zeros(())
    D2 = torch.cdist(P, Q) ** 2
    return D2.min(dim=1).values.mean() + D2.min(dim=0).values.mean()


def _gather_valid_critique(b_logits: torch.Tensor, b_mask: torch.Tensor):
    """
    b_logits: [T,2]
    b_mask:   [T] (bool or int-coded); may come as [T,1] or mismatched dtype/shape.
    Returns: (logits_valid [N,2], idx [N] long)
    """
    # to bool 1D
    if b_mask.dtype != torch.bool:
        b_mask = b_mask != 0
    b_mask = b_mask.view(-1)

    # align lengths
    T = b_logits.shape[0]
    if b_mask.numel() != T:
        b_mask = b_mask[:T]

    # 1D long index
    idx = torch.nonzero(b_mask, as_tuple=True)[0]  # [N]
    if idx.numel() == 0:
        return b_logits.new_zeros((0, b_logits.shape[1])), idx
    return b_logits.index_select(0, idx), idx


def _count_objects_in_image(slot_inst_ids_b):
    # Accepts tensor [K], list/tuple of ints, or dict {slot_idx: inst_id}
    if slot_inst_ids_b is None:
        return 1
    if isinstance(slot_inst_ids_b, dict):
        ids = torch.as_tensor(list(slot_inst_ids_b.values()), dtype=torch.long)
    elif isinstance(slot_inst_ids_b, (list, tuple)):
        ids = torch.as_tensor(slot_inst_ids_b, dtype=torch.long)
    elif isinstance(slot_inst_ids_b, torch.Tensor):
        ids = slot_inst_ids_b.to(torch.long)
    else:
        return 1
    ids = ids[ids >= 0]
    return max(int(torch.unique(ids).numel()), 1)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

# Add near the other helpers

def chamfer_symmetric(pred_pts: torch.Tensor, gt_subset: torch.Tensor) -> torch.Tensor:
    """
    Chamfer distance between pred_pts [P,3] and a subset of GT points [M,3].
    Returns 0 if gt_subset is empty.
    """
    D2 = torch.cdist(pred_pts, gt_subset) ** 2
    return D2.min(dim=1).values.mean() + D2.min(dim=0).values.mean()

def soft_diversity_penalty(pred_pts: torch.Tensor, gt_pts: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    P, G = pred_pts.shape[0], gt_pts.shape[0]
    D2 = torch.cdist(pred_pts, gt_pts) ** 2
    assign = F.softmax(-D2 / tau, dim=0)
    counts = assign.sum(dim=1)
    target = float(G) / max(P, 1)
    return ((counts - target) ** 2).mean()

def repulsion_loss(pred_pts: torch.Tensor, r: float = 0.02) -> torch.Tensor:
    P = pred_pts.shape[0]
    if P <= 1:
        return pred_pts.new_zeros(())
    Dpp = torch.cdist(pred_pts, pred_pts)
    Dpp = Dpp + torch.eye(P, device=pred_pts.device) * 1e6
    return F.relu(r - Dpp).mean()
    # In loss_joint_with_cls.py



class HungarianMatcher:
    def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=2.0, cost_embed=0.0):
        self.cost_class = float(cost_class)
        self.cost_bbox  = float(cost_bbox)
        self.cost_giou  = float(cost_giou)
        self.cost_embed = float(cost_embed)

    @dynamo.disable
    @torch.no_grad()
    def __call__(self, pred_boxes, pred_logits, tgt_boxes, tgt_labels,
                 pred_cls_tokens=None, tgt_embeds=None):
        dev = pred_boxes.device
        Q, C = pred_logits.shape
        M = tgt_boxes.shape[0]

        if Q == 0 or M == 0:
            return (torch.as_tensor([], dtype=torch.long, device=dev),
                    torch.as_tensor([], dtype=torch.long, device=dev))

        out_prob = pred_logits.softmax(-1).clamp_min(1e-9)
        tgt_labels_exp = tgt_labels.view(1, M).expand(Q, M)
        cost_class = -torch.log(out_prob.gather(1, tgt_labels_exp))
        cost_bbox  = torch.cdist(pred_boxes, tgt_boxes, p=1)
        giou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(pred_boxes),
            box_ops.box_cxcywh_to_xyxy(tgt_boxes)
        )
        cost_giou = 1.0 - giou

        if (pred_cls_tokens is not None) and (tgt_embeds is not None):
            a = F.normalize(pred_cls_tokens, dim=-1)  # [Q,D]
            b = F.normalize(tgt_embeds, dim=-1)       # [M,D]
            cost_embed = 1.0 - (a @ b.T)              # [Q,M]
        else:
            cost_embed = torch.zeros_like(cost_bbox)

        big = torch.tensor(1e6, device=dev)
        for t in (cost_class, cost_bbox, cost_giou, cost_embed):
            t.masked_fill_(~torch.isfinite(t), big)

        C_total = (self.cost_class * cost_class
                   + self.cost_bbox * cost_bbox
                   + self.cost_giou * cost_giou
                   + self.cost_embed * cost_embed)

        C_np = np.where(np.isfinite(C_total.detach().cpu().numpy()), 
                        C_total.detach().cpu().numpy(), 1e6)

        i_idx, j_idx = linear_sum_assignment(C_np)
        return (torch.as_tensor(i_idx, dtype=torch.long, device=dev),
                torch.as_tensor(j_idx, dtype=torch.long, device=dev))

def _count_nonfinite(t: torch.Tensor) -> int:
    try:
        return int((~torch.isfinite(t)).sum().item())
    except Exception:
        return 0

def _nan_to_num(t: torch.Tensor, name: str = "", nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        return t
    nf = _count_nonfinite(t)
    if nf > 0:
        # Optional: print once to see the source
        # print(f"[nan_guard] {name}: replaced {nf} non-finite entries")
        t = torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf)
    return t


class JointCriterion(nn.Module):
    """
    Detection + classification (Deformable DETR-style focal) + object-property losses.
    """
    def __init__(
        self,
        loss_bbox_coef=5.0,
        loss_giou_coef=2.0,
        loss_obj_scale_coef=1.0,
        loss_obj_pos_coef=1.0,
        loss_obj_vis_coef=0.5,
        loss_obj_shape_coef=1.0,
        loss_obj_shape_cd_coef=1.0,
        
        loss_obj_shape_cd_vis_coef=1.0,
        loss_obj_shape_cd_inv_coef=0.25,

        loss_obj_shape_div_coef=0.1,
        loss_obj_shape_rep_coef=0.05,
        loss_obj_vol_coef=1.0,
        loss_critique_coef_cls=1.0,           
        loss_critique_coef_score=1.0,
        loss_det_embed_coef=1.0,         

        shape_tau=0.05,
        shape_r=0.02,
        w_cls=1.0,
        focal_alpha=0.25,           
        set_cost_class=1.0,
        set_cost_bbox=1.0,
        set_cost_giou=2.0,
        set_cost_embed=0.5,
        with_det=True,
        with_obj=True,
        with_critique=False,             
        with_pose=False,
        shape_gt_subsample=512,
        aux_det_weight=1.0,
        aux_obj_weight=1.0,
        aux_critique_weight=1.0,
        loss_pose_conf_coef=0.0,
        loss_pose_push_coef=0.0,
        loss_grasp_reg_coef=5.0,
        loss_pose_chamfer_coef=1.0,
        loss_pose_conf_gt_coef=1.0,
        grasp_nn_t_thr_m: float = 0.02,
        grasp_nn_ang_thr_deg: float = 15.0,
        warmup_steps = 1000,
        start_warmup_step=1000,
        aux_det_match_reuse=True,
        aux_obj_match_reuse=True,
        apply_vis_mask_to_shape=False,
        ):
        super().__init__()
        # matcher uses softmax (original-style)
        self.matcher = HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou, cost_embed=set_cost_embed)

        self.w_cls   = float(w_cls)
        self.w_bbox  = float(loss_bbox_coef)
        self.w_giou  = float(loss_giou_coef)

        self.focal_alpha = float(focal_alpha)

        # object properties
        self.w_scale = float(loss_obj_scale_coef)
        self.w_pos   = float(loss_obj_pos_coef)
        self.w_vis   = float(loss_obj_vis_coef)
        self.w_vol   = float(loss_obj_vol_coef)

        self.w_shape     = float(loss_obj_shape_coef)
        self.w_shape_cd  = float(loss_obj_shape_cd_coef)
        self.w_shape_cd_vis = float(loss_obj_shape_cd_vis_coef)
        self.w_shape_cd_inv = float(loss_obj_shape_cd_inv_coef)
        self.w_shape_div = float(loss_obj_shape_div_coef)
        self.w_shape_rep = float(loss_obj_shape_rep_coef)
        self.shape_tau   = float(shape_tau)
        self.shape_r     = float(shape_r)


        self.with_det = bool(with_det)
        self.with_obj = bool(with_obj)
        self.with_pose = bool(with_pose)
        self.with_critique = bool(with_critique)        

        self.shape_gt_subsample = int(shape_gt_subsample)

        self.aux_det_weight = float(aux_det_weight)
        self.aux_obj_weight = float(aux_obj_weight)
        self.aux_critique_weight = float(aux_critique_weight)

        self.aux_det_match_reuse = bool(aux_det_match_reuse)
        self.aux_obj_match_reuse = bool(aux_obj_match_reuse)
        
        self.loss_det_embed_coef = float(loss_det_embed_coef)
        self.loss_critique_coef_cls = float(loss_critique_coef_cls)   
        self.loss_critique_coef_score = float(loss_critique_coef_score)

        self.loss_pose_push_coef = float(loss_pose_push_coef)
        self.loss_pose_chamfer_coef = float(loss_pose_chamfer_coef)
        self.loss_pose_conf_coef = float(loss_pose_conf_coef)
        self.loss_grasp_reg_coef = float(loss_grasp_reg_coef)
        self.loss_pose_conf_gt_coef = float(loss_pose_conf_gt_coef)

        self.grasp_nn_t_thr_m = float(grasp_nn_t_thr_m) 
        self.grasp_nn_ang_thr_deg = float(grasp_nn_ang_thr_deg)
        self._step = 0
        self._warmup_steps = warmup_steps
        self._start_warmup_step = start_warmup_step

        self.score_gamma = 2.0            # nonlinearity for score weight
        self.score_softmin_tau = 0.10     # temperature for score-aware softmin (same scale as tau_reg)
        self.score_bias_lambda = 0.05      # optional additive bias toward high-score GT in distance: D' = D + λ*(1-s)
        self.apply_vis_mask_to_shape = apply_vis_mask_to_shape


    def _remap_labels_zero_based(self, tgt_labels: torch.Tensor, C: int) -> torch.Tensor:
        """
        Ensure labels in [0, C-1]. If a common object-agnostic target uses '1' for object with C=1,
        remap to 0. Otherwise clamp into range.
        """
        if tgt_labels.numel() == 0:
            return tgt_labels
        if int(tgt_labels.max().item()) >= C or int(tgt_labels.min().item()) < 0:
            # try a simple remap: shift by -1 then clamp
            tgt_labels = torch.clamp(tgt_labels - 1, 0, C - 1)
        return tgt_labels

    def _build_target_inst_embeds(self, outs, targets, b, d_enc, device):
        # outs['inst_crops_embeds'][b]: [Ni, dEnc]
        # targets[b]['inst_crops_inst_ids']: [Ni]
        # targets[b]['box2inst']: [N]

        if 'inst_crops_inst_ids' not in targets[b]:
            return None, None
            
        emb_b = outs['inst_crops_embeds'][b]
        
        inst_ids_b = targets[b]['inst_crops_inst_ids'].to(device).view(-1)
        id2emb = {int(inst_ids_b[i].item()): emb_b[i].to(device) for i in range(inst_ids_b.shape[0])}

        box2inst_b = targets[b]['box2inst'].to(device).view(-1)
        N = box2inst_b.shape[0]
        tgt_embeds_b = torch.zeros(N, d_enc, device=device, dtype=emb_b.dtype)
        has_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for j in range(N):
            iid = int(box2inst_b[j].item())
            if iid in id2emb:
                tgt_embeds_b[j] = id2emb[iid]
                has_mask[j] = True

        return tgt_embeds_b, has_mask  # [N,dEnc], [N]


    def _build_focal_targets(self, pred_logits: torch.Tensor, targets, indices, num_boxes):
        B, Q, C = pred_logits.shape  # here C should be 2
        device = pred_logits.device
        assert C == 2, "Expected two-class logits [bg, fg]"

        target_classes = torch.zeros((B, Q), dtype=torch.long, device=device)  # default bg=0
        for b, (src, tgt) in enumerate(indices):
            if src.numel() == 0:
                continue
            tgt_labels = targets[b]["labels"].to(device)
            # Ensure labels are {0(bg),1(obj)}; typically gt boxes are 1
            tgt_labels = torch.clamp(tgt_labels, 0, C - 1)
            target_classes[b, src] = tgt_labels[tgt]

        target_onehot = F.one_hot(target_classes, num_classes=C).to(pred_logits.dtype)  # [B,Q,2]
        return target_onehot
    
    def _compute_num_boxes(self, targets, device) -> float:
        num_boxes = sum([len(t["labels"]) for t in targets])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        world = torch.distributed.get_world_size() if (torch.distributed.is_available() and torch.distributed.is_initialized()) else 1
        num_boxes = torch.clamp(num_boxes / world, min=1).item()
        return num_boxes


    def forward(self, outs, targets):

        losses = {}
        slot_inst_ids = None
        inst_id_2_slot_id = outs.get('inst_id_2_slot_id', None)
        add_pose_extra_query_embed = bool(outs.get("pose_mode_is_object_6d", False))

        # FINAL matching (softmax-based costs like original)       
        device = outs["pred_boxes"].device
        pred_boxes_all  = outs["pred_boxes"]   # [B,Q,4]
        pred_logits_all = outs["pred_logits"]  # [B,Q,C]
        det_tokens = outs["det_cls_token"]  # [B,Q,dEnc]
        d_enc = det_tokens.shape[-1]
        box2inst = [t['box2inst'] for t in targets]  

        B, Q, C = pred_logits_all.shape

        final_indices = []        
        for b in range(B):
            idx_pred, idx_tgt = self.matcher(pred_boxes_all[b], pred_logits_all[b], targets[b]["boxes"], targets[b]["labels"])
            # guard: remap labels later if needed, matcher itself will assert otherwise
            final_indices.append((idx_pred, idx_tgt))

        final_indices = []
        for b in range(B):
            tgt_emb_b, tgt_has_b = self._build_target_inst_embeds(outs, targets, b, d_enc, device)          
                 
            idx_pred, idx_tgt = self.matcher(
                pred_boxes_all[b], pred_logits_all[b],
                targets[b]["boxes"], targets[b]["labels"],
                pred_cls_tokens=det_tokens[b], tgt_embeds=tgt_emb_b
            )            
            final_indices.append((idx_pred, idx_tgt))

        
        # Build per-image slot_inst_ids [B,K] from det->GT and obj_index

        obj_index = outs.get("obj_index", None)  # [B,K]
        if isinstance(obj_index, torch.Tensor):
            B, K = obj_index.shape
            slot_inst_ids = torch.full((B, K), -1, dtype=torch.long, device=device)
            for b in range(B):
                srcs, tgts = final_indices[b]  # matched pairs
                srcs = srcs.tolist()
                tgts = tgts.tolist()
                # build det_q -> target_idx map
                #print('src', srcs)
                #print('tgt', tgts)
                #print('idx', obj_index[b].tolist())
                #print('box2inst', box2inst[b])
                

                #det2tgt = {int(src[i].item()): int(tgt[i].item()) for i in range(src.numel())}
                for k, idx in enumerate(obj_index[b].tolist()):

                    #print('it', k, idx)
                    if idx not in srcs:
                        continue

                    src_idx = srcs.index(idx)
                    tgt = tgts[src_idx]
                    #print('src_idx', src_idx, tgt)


                    inst_id = box2inst[b][tgt]
                    slot_inst_ids[b, k] = inst_id

        if self.with_det:        
            # Compute num_boxes across processes (original normalization)
            num_boxes = max(self._compute_num_boxes(targets, device), 1.0)

            # Classification focal loss (original style)
            #pred_logits_all_fg = pred_logits_all[..., 1:]
            #print(pred_logits_all.shape, pred_logits_all_fg.shape)
            #target_onehot = self._build_focal_targets(pred_logits_all, targets, final_indices, num_boxes)
            target_onehot = self._build_focal_targets(pred_logits_all, targets, final_indices, num_boxes)


            # focal returns sum normalized by num_boxes; scale by Q like original
            loss_cls = sigmoid_focal_loss(pred_logits_all, target_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * Q
            loss_cls = self.w_cls * loss_cls

            # BBox + GIoU normalized by num_boxes (original)
            loss_bbox_sum = torch.tensor(0.0, device=device)
            loss_giou_sum = torch.tensor(0.0, device=device)
            loss_embed_sum = torch.tensor(0.0, device=device)
            for b in range(B):
                src, tgt = final_indices[b]
                if src.numel() == 0:
                    continue
                pb = pred_boxes_all[b][src]
                tb = targets[b]["boxes"][tgt].to(device)
                loss_bbox_sum += F.l1_loss(pb, tb, reduction="sum")
                giou = box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(pb),
                    box_ops.box_cxcywh_to_xyxy(tb)
                )
                loss_giou_sum += (1.0 - torch.diag(giou)).sum()
                
                tgt_emb_b, tgt_has_b = self._build_target_inst_embeds(outs, targets, b, d_enc, device)
                if tgt_has_b is None:
                    continue

                keep = tgt_has_b[tgt]
                if keep.numel() == 0 or not keep.any():
                    continue
                pred_tok = F.normalize(det_tokens[b][src][keep], dim=-1)   # [P',dEnc]
                tgt_tok  = F.normalize(tgt_emb_b[tgt][keep], dim=-1)       # [P',dEnc]
                cos = (pred_tok * tgt_tok).sum(dim=-1)
                loss_embed_sum += (1.0 - cos).sum()
                

            loss_bbox = self.w_bbox * (loss_bbox_sum / num_boxes)
            loss_giou = self.w_giou * (loss_giou_sum / num_boxes)
            loss_embed = self.loss_det_embed_coef * (loss_embed_sum / num_boxes)

            losses.update({
                "loss_cls": loss_cls,
                "loss_bbox": loss_bbox,
                "loss_giou": loss_giou,
                "loss_embed": loss_embed
            })  
            

        # Object-property losses (keep your normalization by matched pairs)
        det_to_slot = None
        if self.with_obj:
            obj_pos_final   = outs.get("obj_positions")
            obj_vis_final   = outs.get("obj_vis")
            obj_scale_final = outs.get("obj_scale")
            obj_pts_final   = outs.get("obj_points")
            obj_vol_final   = outs.get("obj_vol")
            obj_index       = outs.get("obj_index")
            
    
            det_to_slot = [{int(det_q): slot for slot, det_q in enumerate(obj_index[b].tolist())} for b in range(B)]
                
            loss_scale = torch.tensor(0.0, device=device)
            loss_pos   = torch.tensor(0.0, device=device)
            loss_vis   = torch.tensor(0.0, device=device)
            loss_shape = torch.tensor(0.0, device=device)
            loss_vol   = torch.tensor(0.0, device=device)
            n_obj_pairs = 0
            slot_considered = 0
            slot_misses = 0

            for b in range(B):
                src, tgt = final_indices[b]
                if src.numel() == 0 or obj_index is None:
                    continue
            
                valid_mask = targets[b]["obj_valid"]
                for p_i, t_j in zip(src.tolist(), tgt.tolist()):
                    if not bool(valid_mask[t_j].item()):
                        continue

                    # count a considered pair
                    slot_considered += 1
                    # if the matched detection is not part of obj_index (no slot for it)
                    if p_i not in det_to_slot[b]:
                        slot_misses += 1
                        continue
                    slot = det_to_slot[b][p_i]
                    pred_scale  = obj_scale_final[b, slot]
                    pred_pos    = obj_pos_final[b, slot]
                    pred_vis    = obj_vis_final[b, slot]
                    pred_points = obj_pts_final[b, slot]
                    pred_vol    = obj_vol_final[b, slot]

                    gt_scale = targets[b]["obj_scale"][t_j].to(device)
                    gt_pos   = targets[b]["obj_pos"][t_j].to(device)
                    gt_vis   = targets[b]["obj_vis"][t_j].to(device)
                    gt_list  = targets[b]["obj_points_list"][t_j].to(device)
                    gt_vol   = targets[b]["obj_vol"][t_j].to(device)

                    if self.apply_vis_mask_to_shape:
                        m = targets[b]["obj_vis_mask"][t_j].to(device)

                        gt_list_vis = gt_list[m]
                        gt_list_inv = gt_list[~m]

                        if len(gt_list_vis) > 0:
                            cd_vis = chamfer_symmetric(pred_points, gt_list_vis)  # 0 if empty
                        else:
                            cd_vis = torch.tensor(0.0, dtype=gt_list.dtype, device=gt_list.device)
                        
                        if len(gt_list_inv) > 0:
                            cd_inv = chamfer_symmetric(pred_points, gt_list_inv)  # 0 if empty
                        else:
                            cd_inv = torch.tensor(0.0, dtype=gt_list.dtype, device=gt_list.device)

                        cd = self.w_shape_cd_vis * cd_vis + self.w_shape_cd_inv * cd_inv
                    else:
                        cd = chamfer_symmetric(pred_points, gt_list)

                    div = soft_diversity_penalty(pred_points, gt_list, tau=self.shape_tau)
                    rep = repulsion_loss(pred_points, r=self.shape_r)
                    loss_shape += (cd + self.w_shape_div * div + self.w_shape_rep * rep)

                    loss_scale += F.mse_loss(pred_scale, gt_scale, reduction="sum")
                    loss_pos   += F.mse_loss(pred_pos,   gt_pos,   reduction="sum")

                    #print('pred_vis', pred_vis)
                    #print('gt_vis', gt_vis)

                    
                    
                    loss_vis += F.binary_cross_entropy_with_logits(pred_vis, gt_vis, reduction="mean")
                    #print('loss_vis', loss_vis)
                    loss_vol += F.binary_cross_entropy_with_logits(pred_vol, gt_vol, reduction="mean")
                    n_obj_pairs += 1

            if n_obj_pairs > 0:
                losses["loss_obj_scale"] = self.w_scale * (loss_scale / n_obj_pairs)
                losses["loss_obj_pos"]   = self.w_pos   * (loss_pos   / n_obj_pairs)
                losses["loss_obj_vis"]   = self.w_vis   * (loss_vis   / n_obj_pairs)
                #print('loss vis', loss_vis)
                #print('loss_obj_vis', losses["loss_obj_vis"])

                losses["loss_obj_shape"] = self.w_shape * (loss_shape / n_obj_pairs)
                losses["loss_obj_vol"]   = self.w_vol   * (loss_vol   / n_obj_pairs)
            else:
                losses["loss_obj_scale"] = torch.tensor(0.0, device=device)
                losses["loss_obj_pos"]   = torch.tensor(0.0, device=device)
                losses["loss_obj_vis"]   = torch.tensor(0.0, device=device)
                losses["loss_obj_shape"] = torch.tensor(0.0, device=device)
                losses["loss_obj_vol"]   = torch.tensor(0.0, device=device)

            miss_pct = (100.0 * slot_misses / max(1, slot_considered)) if slot_considered > 0 else 0.0
            losses["obj_slot_miss"] = torch.as_tensor(miss_pct, device=device)
        #input()

    
        if self.with_pose:
            # Decide mode from model config you pass in via outs (or args); here we infer from a flag on outs
            pose_preds = outs['pose_preds']            
            pose_slot_logits = outs.get('pose_slot_logits', None)
            crit_pred_logits = outs.get('critique_pred_logits', None)
            crit_pred_mask = outs.get('critique_pred_mask', None)           
            pose_losses = self._pose_losses(pose_preds, pose_slot_logits, crit_pred_logits, crit_pred_mask, slot_inst_ids, inst_id_2_slot_id, 
                                            targets, add_pose_extra_query_embed=add_pose_extra_query_embed, return_metric=True)
            losses.update(pose_losses)        
        
        #print('critique', self.with_critique)
        if self.with_critique:
            crit_logits = outs.get('critique_gt_logits', None)     # [L,B,N,2] or [B,N,2]
            crit_masks  = outs.get('critique_gt_pose_mask', None)  # [B,N]
            crit_scores = outs.get('critique_gt_score', None)      # [B,N]

            #print('crit_logits', type(crit_logits))
            if crit_logits is not None:
                if device is None:
                    device = crit_logits.device
                #print('in')

                # last layer only
                if crit_logits.dim() == 4:
                    crit_logits = crit_logits[-1]   # [B,N,2]

                B, N, _ = crit_logits.shape
                logits_flat = crit_logits.reshape(B * N, 2)
                masks_flat  = crit_masks.reshape(-1).long()                 # 0=good,1=bad,2=random
                y_cls = (masks_flat == 0).long()                            # good->1; bad+random->0
                
                # classification (good=1 else=0)
                y_onehot = F.one_hot(y_cls, num_classes=2).to(logits_flat.dtype)

                denom = max(int(y_cls.numel()), 1)
                loss_crit = sigmoid_focal_loss(logits_flat, y_onehot, denom, alpha=self.focal_alpha, gamma=2)
                losses["loss_critique"] = self.loss_critique_coef_cls * loss_crit

                # score calibration (BCE with logits) on "good" logit difference
                scores_flat = crit_scores.reshape(-1).clamp(0.0, 1.0).to(logits_flat.dtype)   # soft target in [0,1]
                # logit for class 'good' from 2-way logits = l1 - l0
                z_good = logits_flat[:, 1] - logits_flat[:, 0]
                bce = F.binary_cross_entropy_with_logits(z_good, scores_flat, reduction="mean")
                losses["loss_critique_score"] = self.loss_critique_coef_score * bce

                # Metrics (unchanged): argmax over logits => 1=good, 0=bad/random
                with torch.no_grad():                  
                    pred_good = (logits_flat[:, 1] > logits_flat[:, 0]).long()
                    mid_mask = (scores_flat > 0.3) & (scores_flat < 0.7)
                    if int(mid_mask.sum().item()) > 0:
                        y_mid = (scores_flat[mid_mask] >= 0.5).long()     # target: s>=0.5 -> good
                        pred_mid = pred_good[mid_mask]
                        acc_mid = (pred_mid == y_mid).float().mean().item()
                    else:
                        acc_mid = 0.0

                    pred = (logits_flat[:, 1] > logits_flat[:, 0]).long()
                    tp = ((pred == 1) & (y_cls == 1)).sum().item()
                    tn = ((pred == 0) & (y_cls == 0)).sum().item()
                    fp = ((pred == 1) & (y_cls == 0)).sum().item()
                    fn = ((pred == 0) & (y_cls == 1)).sum().item()
                    total = max(tp + tn + fp + fn, 1)

                    acc = (tp + tn) / float(total)
                    prec = tp / float(max(tp + fp, 1))
                    rec = tp / float(max(tp + fn, 1))
                    f1 = (2.0 * prec * rec) / float(max(prec + rec, 1e-12))

                    # Per-group accuracies
                    idx_good   = (masks_flat == 0)
                    idx_bad    = (masks_flat == 1)
                    idx_random = (masks_flat == 2)

                    ng = int(idx_good.sum().item())
                    nb = int(idx_bad.sum().item())
                    nr = int(idx_random.sum().item())
                    acc_good    = ((pred[idx_good]    == y_cls[idx_good]).sum().item() / float(max(ng, 1))) if ng > 0 else 0.0
                    acc_bad    = ((pred[idx_bad]    == y_cls[idx_bad]).sum().item() / float(max(nb, 1))) if nb > 0 else 0.0
                    acc_random = ((pred[idx_random] == y_cls[idx_random]).sum().item() / float(max(nr, 1))) if nr > 0 else 0.0

                    
                    losses["critique_acc_mid"] = torch.as_tensor(acc_mid, device=device)*100
                    losses["critique_acc"]         = torch.as_tensor(acc, device=device)*100
                    losses["critique_precision"]   = torch.as_tensor(prec, device=device)*100
                    losses["critique_recall"]      = torch.as_tensor(rec, device=device)*100
                    losses["critique_f1"]          = torch.as_tensor(f1, device=device)*100
                    losses["critique_acc_good"]     = torch.as_tensor(acc_good, device=device)*100
                    losses["critique_acc_bad"]     = torch.as_tensor(acc_bad, device=device)*100
                    losses["critique_acc_random"]  = torch.as_tensor(acc_random, device=device)*100
                    losses["critique_n"]           = torch.as_tensor(total, device=device)
                    losses["critique_n_good"]      = torch.as_tensor(ng, device=device)
                    losses["critique_n_bad"]       = torch.as_tensor(nb, device=device)
                    losses["critique_n_random"]    = torch.as_tensor(nr, device=device)
                    #print('?', losses["critique_acc"])

        # Aux losses
        aux_list = outs.get("aux_outputs", [])
        if aux_list:
            # Aux detection
            if self.with_det:
                for i, aux in enumerate(aux_list):
                    aux_boxes  = aux["pred_boxes"]   # [B,Q,4]
                    aux_logits = aux["pred_logits"]  # [B,Q,C]
                    aux_tok = aux.get('det_cls_token', None)

                    Q_aux = aux_logits.shape[1]

                    # reuse or recompute matching
                    if self.aux_det_match_reuse:
                        aux_indices = final_indices
                    else:
                        aux_indices = []
                        for b in range(B):
                            ip, it = self.matcher(aux_boxes[b], aux_logits[b], targets[b]["boxes"], targets[b]["labels"])
                            aux_indices.append((ip, it))

                    aux_tgt_onehot = self._build_focal_targets(aux_logits, targets, aux_indices, num_boxes)
                    aux_lc = sigmoid_focal_loss(aux_logits, aux_tgt_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * Q_aux
                    losses[f"loss_cls_aux{i}"] = self.aux_det_weight * self.w_cls * aux_lc

                    lb_sum = torch.tensor(0.0, device=device)
                    lg_sum = torch.tensor(0.0, device=device)
                    le_sum = torch.tensor(0.0, device=device)
                    for b in range(B):
                        src, tgt = aux_indices[b]
                        if src.numel() == 0:
                            continue
                        pb = aux_boxes[b][src]
                        tb = targets[b]["boxes"][tgt].to(device)
                        lb_sum += F.l1_loss(pb, tb, reduction="sum")
                        giou = box_ops.generalized_box_iou(
                            box_ops.box_cxcywh_to_xyxy(pb),
                            box_ops.box_cxcywh_to_xyxy(tb)
                        )
                        lg_sum += (1.0 - torch.diag(giou)).sum()

                        tgt_emb_b, tgt_has_b = self._build_target_inst_embeds(outs, targets, b, aux_tok.shape[-1], device)
                        if tgt_has_b is None: 
                            continue

                        keep = tgt_has_b[tgt]
                        if keep.numel() == 0 or not keep.any():
                            continue
                        pred_tok = F.normalize(aux_tok[b][src][keep], dim=-1)
                        tgt_tok  = F.normalize(tgt_emb_b[tgt][keep], dim=-1)
                        cos = (pred_tok * tgt_tok).sum(dim=-1)
                        le_sum += (1.0 - cos).sum()
                        

                    losses[f"loss_bbox_aux{i}"] = self.aux_det_weight * self.w_bbox * (lb_sum / num_boxes)
                    losses[f"loss_giou_aux{i}"] = self.aux_det_weight * self.w_giou * (lg_sum / num_boxes)
                    losses[f"loss_embed_aux{i}"] = self.aux_det_weight * self.loss_det_embed_coef * (le_sum / num_boxes)
                
            # Aux object-property
            if self.with_obj:
                obj_index = outs.get("obj_index", None)  # [B,K]
                det_to_slot = None
                if isinstance(obj_index, torch.Tensor):
                    det_to_slot = [{int(det_q): slot for slot, det_q in enumerate(obj_index[b].tolist())}
                                for b in range(B)]

                L_total = len(aux_list) + 1
                for l, aux in enumerate(aux_list):
                    aux_ls = torch.tensor(0.0, device=device)
                    aux_lp = torch.tensor(0.0, device=device)
                    aux_lvis = torch.tensor(0.0, device=device)
                    aux_lsh = torch.tensor(0.0, device=device)
                    aux_lvol = torch.tensor(0.0, device=device)
                    aux_n = 0

                    # reuse or recompute matching for each aux layer
                    if self.aux_obj_match_reuse:
                        aux_indices_obj = final_indices
                    else:
                        aux_indices_obj = []
                        for b in range(B):
                            ip, it = self.matcher(aux["pred_boxes"][b], aux["pred_logits"][b], targets[b]["boxes"], targets[b]["labels"])
                            aux_indices_obj.append((ip, it))

                    for b in range(B):
                        src, tgt = aux_indices_obj[b]
                        if src.numel() == 0 or (obj_index is None) or (det_to_slot is None):
                            continue

                        valid_mask = targets[b]["obj_valid"]

                        for p_i, t_j in zip(src.tolist(), tgt.tolist()):
                            if not bool(valid_mask[t_j].item()):
                                continue
                            if p_i not in det_to_slot[b]:
                                continue
                            slot = det_to_slot[b][p_i]

                            pred_scale  = aux["obj_scale"][b, slot]
                            pred_pos    = aux["obj_positions"][b, slot]
                            pred_vis    = aux["obj_vis"][b, slot]
                            pred_points = aux["obj_points"][b, slot]
                            pred_vol    = aux["obj_vol"][b, slot]

                            gt_scale = targets[b]["obj_scale"][t_j].to(device)
                            gt_pos   = targets[b]["obj_pos"][t_j].to(device)
                            gt_vis   = targets[b]["obj_vis"][t_j].to(device)
                            gt_vol   = targets[b]["obj_vol"][t_j].to(device)
                            gt_list  = targets[b]["obj_points_list"][t_j].to(device)

                            if self.apply_vis_mask_to_shape:
                                m = targets[b]["obj_vis_mask"][t_j].to(device)
                                gt_list_vis = gt_list[m]
                                gt_list_inv = gt_list[~m]

                                if len(gt_list_vis) > 0:
                                    cd_vis = chamfer_symmetric(pred_points, gt_list_vis)  # 0 if empty
                                else:
                                    cd_vis = torch.tensor(0.0, dtype=gt_list.dtype, device=gt_list.device)
                                
                                if len(gt_list_inv) > 0:
                                    cd_inv = chamfer_symmetric(pred_points, gt_list_inv)  # 0 if empty
                                else:
                                    cd_inv = torch.tensor(0.0, dtype=gt_list.dtype, device=gt_list.device)
            
                                cd = self.w_shape_cd_vis * cd_vis + self.w_shape_cd_inv * cd_inv
                            else:
                                cd = chamfer_symmetric(pred_points, gt_list)
                            
                            div = soft_diversity_penalty(pred_points, gt_list, tau=self.shape_tau)
                            rep = repulsion_loss(pred_points, r=self.shape_r)
                            aux_lsh += (cd + self.w_shape_div * div + self.w_shape_rep * rep)
                            aux_ls += F.mse_loss(pred_scale, gt_scale, reduction="sum")
                            aux_lp   += F.mse_loss(pred_pos,   gt_pos,   reduction="sum")
                            aux_lvis += F.binary_cross_entropy_with_logits(pred_vis, gt_vis, reduction="sum")
                            aux_lvol += F.binary_cross_entropy_with_logits(pred_vol, gt_vol, reduction="sum")

                            #aux_ls  += F.mse_loss(pred_scale, gt_scale, reduction="sum")
                            #aux_lp  += F.mse_loss(pred_pos,   gt_pos,   reduction="sum")
                            #aux_lvis+= F.mse_loss(pred_vis,   gt_vis,   reduction="sum")
                            #cd  = chamfer_symmetric(pred_points, gt_list)
                            #div = soft_diversity_penalty(pred_points, gt_list, tau=self.shape_tau)
                            #rep = repulsion_loss(pred_points, r=self.shape_r)
                            #aux_lsh += (self.w_shape_cd * cd + self.w_shape_div * div + self.w_shape_rep * rep)
                            #aux_lvol += F.mse_loss(pred_vol, gt_vol, reduction="sum")

                            aux_n   += 1

                    if aux_n > 0:
                        losses[f"loss_obj_scale_aux{l}"] = self.aux_obj_weight * self.w_scale * (aux_ls  / aux_n)
                        losses[f"loss_obj_pos_aux{l}"]   = self.aux_obj_weight * self.w_pos   * (aux_lp  / aux_n)
                        losses[f"loss_obj_vis_aux{l}"]   = self.aux_obj_weight * self.w_vis   * (aux_lvis/ aux_n)
                        losses[f"loss_obj_shape_aux{l}"] = self.aux_obj_weight * (aux_lsh / aux_n)
                        losses[f"loss_obj_vol_aux{l}"]   = self.aux_obj_weight * self.w_vol   * (aux_lvol/ aux_n)
                    else:
                        losses[f"loss_obj_scale_aux{l}"] = torch.tensor(0.0, device=device)
                        losses[f"loss_obj_pos_aux{l}"]   = torch.tensor(0.0, device=device)
                        losses[f"loss_obj_vis_aux{l}"]   = torch.tensor(0.0, device=device)
                        losses[f"loss_obj_shape_aux{l}"] = torch.tensor(0.0, device=device)
                        losses[f"loss_obj_vol_aux{l}"]   = torch.tensor(0.0, device=device)

            if self.with_pose:                    
                for i, aux in enumerate(aux_list):                
                    pose_preds = aux['pose_preds']
                    pose_slot_logits = aux.get('pose_slot_logits', None)
                    crit_pred_logits = aux.get('critique_pred_logits', None)
                    crit_pred_mask = aux.get('critique_pred_mask', None)
                    
                    pose_losses = self._pose_losses(pose_preds, pose_slot_logits, crit_pred_logits, crit_pred_mask, slot_inst_ids, inst_id_2_slot_id, targets, add_pose_extra_query_embed=add_pose_extra_query_embed)
                    for pkey, pl in pose_losses.items():
                        losses[f'{pkey}_aux{i}'] = pl                        

            # Aux critique
            if aux_list and self.with_critique:
                crit_masks = outs.get('critique_gt_pose_mask', None)  # [B,N]
                crit_scores = outs.get('critique_gt_score', None)     # [B,N]
                for i, aux in enumerate(aux_list):
                    aux_crit_logits = aux.get('critique_gt_logits', None)  # [B,N,2]
                    if not isinstance(aux_crit_logits, torch.Tensor):
                        continue

                    B_aux, N_aux, _ = aux_crit_logits.shape
                    logits_flat = aux_crit_logits.reshape(B_aux * N_aux, 2)

                    if isinstance(crit_masks, torch.Tensor):
                        masks_flat = crit_masks.reshape(-1).long()
                        y_cls = (masks_flat == 0).long()
                        y_onehot = F.one_hot(y_cls, num_classes=2).to(logits_flat.dtype)
                        denom = max(int(y_cls.numel()), 1)

                        aux_loss_cls = sigmoid_focal_loss(logits_flat, y_onehot, denom, alpha=self.focal_alpha, gamma=2)
                        losses[f"loss_critique_aux{i}"] = self.aux_critique_weight * self.loss_critique_coef_cls * aux_loss_cls

                        if isinstance(crit_scores, torch.Tensor):
                            scores_flat = crit_scores.reshape(-1).clamp(0.0, 1.0).to(logits_flat.dtype)
                            z_good = logits_flat[:, 1] - logits_flat[:, 0]
                            aux_bce = F.binary_cross_entropy_with_logits(z_good, scores_flat, reduction="mean")
                            losses[f"loss_critique_score_aux{i}"] = self.aux_critique_weight * self.loss_critique_coef_score * aux_bce
        return losses



    def _build_slot_targets(self, slot_inst_ids: torch.Tensor, inst_id_2_slot_id, K: int, Np: int, M: int, device: torch.device):
        # slot_inst_ids: [B,K] long; inst_id per slot (or -1)
        B = slot_inst_ids.shape[0]
        none_id = M - 1  # reserve last class for “none”
        tgt_k = torch.full((B, K), none_id, dtype=torch.long, device=device)
        for b in range(B):
            mp = inst_id_2_slot_id[b] if isinstance(inst_id_2_slot_id, list) else {}
            for k in range(K):
                inst_id = int(slot_inst_ids[b, k].item())
                if inst_id < 0:
                    continue
                tgt_k[b, k] = int(mp.get(inst_id, none_id))
        return tgt_k.unsqueeze(-1).expand(B, K, Np)  # [B,K,Np]

    def set_step(self, step: int):
        self._step = int(step)
    
    def step(self):
        if self._step < self._warmup_steps + self._start_warmup_step:
            self._step += 1

    def _conf_warmup_scale(self) -> float:
        #print('')
        #print('step_', self._step)
        #print('_start_warmup_step', self._start_warmup_step)
        #print('_warmup_steps', self._warmup_steps)
        if self._step > self._start_warmup_step:
            t = (self._start_warmup_step + self._warmup_steps - self._step) / float(self._warmup_steps)
            #print('t', t)
            return float(max(0.0, t))
        else:
            return float(1.0)


    def _pose_losses(self, pose_preds, pose_slot_logits, crit_pred_logits, crit_pred_mask,
                    slot_inst_ids, inst_id_2_slot_id, targets,
                    add_pose_extra_query_embed: bool = False, return_metric=False):
        """
        Returns dict with:
        - losses: loss_pose_conf, loss_pose_push, loss_pose_chamfer (6D), loss_grasp_reg (grasp), loss_pose_conf_gt (grasp), loss_pose_slot_cls
        - metrics when return_metric=True:

                pose_avg_crit_score, pose_conf_err_l1, pose_conf_err_rate,
                pose_slot_acc, pose_chamfer_avg (6D),
                grasp_nn_trans_avg, grasp_nn_angle_deg_avg, grasp_nn_combined_avg, grasp_nn_close_pct (grasp)
        """
        losses = {}

        nan_stats = {
            "pose_pred_conf": 0,
            "pose_pred_pos": 0,
            "pose_pred_quat": 0,
            "crit_logits": 0,
            "crit_mask": 0,
            "targets_models": 0,
            "targets_poses": 0,
            "targets_gt_grasps": 0,
        }

        # Sanitize predicted confidence logits (only the conf channel)

        
        B, K, Np, _ = pose_preds.shape
        device = pose_preds.device
        conf_logit = pose_preds[..., 7] # [B,K,Np]
        nan_stats["pose_pred_conf"] = _count_nonfinite(conf_logit)
        conf_flat  = conf_logit.reshape(B, K * Np)      # [B, K*Np]
        
        # Sanitize critique logits before softmax (prevents NaN softmax)

        
        wu = self._conf_warmup_scale()  # warmup scale in [0,1]

        # Helper metrics accumulators (for return_metric)
        avg_crit_score_sum = 0.0
        avg_crit_score_cnt = 0
        conf_abs_err_sum = 0.0
        conf_err_bin_sum = 0.0
        conf_total_cnt = 0.0

        if crit_pred_logits is not None:
            # Safe softmax
            p_good_all_soft = torch.softmax(crit_pred_logits, dim=-1)[..., 1].clamp(1e-4, 1 - 1e-4)
            p_good_all_det  = p_good_all_soft.detach()  # for conf calibration ONLY
            
            # 1) Confidence calibration to critique (always-on), the pred conf should be equal to the critique score
            loss_conf_sum = 0.0
            norm_obj_sum = 0
            for b in range(B):
                idx_b = torch.nonzero(crit_pred_mask[b].to(torch.bool), as_tuple=False).squeeze(1)  # [Tv]
                #print('idx_b', sum(idx_b), len(idx_b))
                if idx_b.numel() == 0:
                    continue
                p_good = p_good_all_det[b, idx_b] # [Tv] we deteach to give no gradients back to the pose rot and pos prediction here, only to the conf pred part 
                conf_b = conf_flat[b, idx_b]
                
                #print('p_good', n_valid, conf_b.shape)

                loss_b = F.binary_cross_entropy_with_logits(conf_b, p_good, reduction="mean")
                n_obj_b = _count_objects_in_image(slot_inst_ids[b] if slot_inst_ids is not None else None)
                loss_conf_sum += loss_b / float(n_obj_b)
                norm_obj_sum += 1

                # metrics wrt critique
                with torch.no_grad():
                    conf_prob = torch.sigmoid(conf_b)
                    conf_abs_err_sum += torch.abs(conf_prob - p_good).sum().item()
                    conf_err_bin_sum += ((conf_prob >= 0.5) != (p_good >= 0.5)).float().sum().item()
                    conf_total_cnt += float(p_good.numel())

            if norm_obj_sum > 0 and self.loss_pose_conf_coef > 0:
                losses["loss_pose_conf"] = self.loss_pose_conf_coef * (loss_conf_sum / norm_obj_sum)
            
            # 2) Push scores of slots with good grasps to 1 (conf == criique score)
            loss_push_sum = 0.0
            n_push_slots = 0


            for b in range(B):        
                idx_b = torch.nonzero(crit_pred_mask[b].to(torch.bool), as_tuple=False).squeeze(1)  # [Tv]
                if idx_b.numel() == 0:
                    continue
                p_b = p_good_all_soft[b, idx_b]  # [K*Np] # no detach to give gradients back into the pose pred of pos and rot outs
                #print('p_b', p_b.shape)
                p_b = p_b.view(K, Np) # [K*Np]
                #print('p_b 2', p_b.shape)
                
                # eligibility mask per slot k
                elig_k = torch.zeros(K, dtype=torch.bool, device=device)
                if add_pose_extra_query_embed:
                    if slot_inst_ids is not None:
                        elig_k = slot_inst_ids[b] >= 0
                else:
                    if slot_inst_ids is not None:
                        #print('slot_inst_ids[b]', slot_inst_ids[b])
                        inst_ids_b = slot_inst_ids[b].tolist()
                        gt_g = targets[b].get("gt_grasps", {}) or {}
                        #print('gt_g', gt_g.keys(), len(gt_g))
                        #has_good = torch.tensor([int(inst_id) in gt_g and len(gt_g.get(int(inst_id), [])) > 0
                        #                        for inst_id in inst_ids_b], device=device, dtype=torch.bool)

                        #print('slot2isnt', np.unique(inst_ids_b, return_counts=True))
                        elig_k = torch.tensor([True if int(inst_id) in gt_g else False for inst_id in inst_ids_b], device=device, dtype=torch.bool)
                
                #print('elig_k', elig_k.shape)
                if elig_k.any():

                    if False:
                        #best_scores, _ = p_b.max(dim=1)  # [K]
                        scores = soft_best_scores(p_b, tau=0.2)             # [K]
                        scores = scores[elig_k]
                    else:
                        scores = p_b[elig_k]
                    
                    #print('scores', scores.shape)

                    #input()

                    #soft_best_scores
                    if scores.numel() > 0:
                        loss_push_sum += (1.0 - scores).mean()
                        n_push_slots += 1
                        if return_metric:
                            avg_crit_score_sum += float(scores.sum().item())
                            avg_crit_score_cnt += int(scores.numel())
                    
            if n_push_slots > 0 and self.loss_pose_push_coef > 0:
                losses["loss_pose_push"] = self.loss_pose_push_coef * (loss_push_sum / n_push_slots)

        
        # 3) Grasp mode extras (add_pose_extra_query_embed == False)
        if not add_pose_extra_query_embed and (wu > 1e-8 or return_metric):
            # Warmup scale (already computed earlier): wu in [0,1]
            # Helpers
            
            # Coefs and temps
            tau_reg = 0.10; w_t_reg, w_q_reg = 1.0, 1.0     # regression weights
            w_t_m,   w_q_rad               = 1.0, 1.0       # metric combined weights
            t_thr = getattr(self, "grasp_nn_t_thr_m", 0.02)
            ang_thr_deg = getattr(self, "grasp_nn_ang_thr_deg", 15.0)

            # Accumulators
            reg_sum, se3_sum, reg_cnt = 0.0, 0.0, 0
            nn_t_sum, nn_ang_sum, nn_comb_sum, nn_cnt, n_close_slots = 0.0, 0.0, 0.0, 0, 0

            for b in range(B):
                if slot_inst_ids is None:
                    continue

                Lis = []
                se3s = []

                for k in range(K):
                    inst_id = int(slot_inst_ids[b, k].item())
                    if inst_id < 0:
                        continue

                    grasps  = targets[b].get("gt_grasps", {}) or {}
                    if inst_id not in grasps:
                        continue

                    G = torch.cat([grasps[inst_id]['xyz'], grasps[inst_id]['quat_wxyz'], grasps[inst_id]['score']], dim=-1)

                    #print('G', G.shape)

                    # Preds and GT
                    P = pose_preds[b, k]         # [Np,7]
                    
                    #print('P', P.shape)
                    t_pred, q_pred, s_pred = P[:, :3], P[:, 3:7], P[:, 7]
                    t_gt, q_gt, s_gt   = G[:, :3], G[:, 3:7], G[:, 7]
                    #print('s_gt', s_gt.shape, s_gt.mean(), s_gt.min(), s_gt.max())

                    # Distances computed once
                    Dt_l2 = torch.cdist(t_pred, t_gt, p=2)        # [Np,G] L2 (for metrics)

                    #print('Dt_l2', Dt_l2.shape, torch.mean(Dt_l2), torch.min(Dt_l2), torch.max(Dt_l2))

                    q1 = q_pred.unsqueeze(1).expand(-1, t_gt.size(0), -1)
                    q2 = q_gt.unsqueeze(0).expand(t_pred.size(0), -1, -1)
                    Dq = quat_geodesic_minimal(q1, q2)             # [Np,G] mormalized radians (shared) for better balance controll
                    Dq_n = Dq / math.pi
                    #print('Dq', Dq.shape, torch.mean(Dq), torch.min(Dq), torch.max(Dq))
                    #input()

                    Ds = 1.0 - (s_gt.unsqueeze(0) - 0.5) * 2
                    #print('Ds', Ds.shape)

                    #Dq = torch.nan_to_num(Dq, nan=math.pi, posinf=math.pi, neginf=math.pi)                    
                    # 3a) Loss: regression (warmup only)
                    D_reg = None
                    if wu > 1e-8:
                        #Li = unique_reg_loss(
                        #    t_pred=t_pred, q_pred=q_pred,
                        #    t_gt=t_gt, q_gt=q_gt, s_gt=s_gt,
                        #    s_t=0.10, w_t=1.0, w_q=0.05, t_gate=0.05,
                        #    tau=0.1, use_sinkhorn=True  # or False for Hungarian
                        #)  # [Np]

                        if False:
                            Np = t_pred.shape[0]
                            proto_idx, t_cent, q_cent, clusters = build_gt_clusters(t_gt, q_gt, s_gt, K_req=Np, s_t=0.10, w_q=0.05)
                            L_ik = pred_cluster_softmin_vectorized(t_pred, q_pred, t_gt, q_gt, s_gt, clusters,
                                                        s_t=0.10, w_t=1.0, w_q=0.05, t_gate=0.05, tau=0.1,
                                                        score_lambda=self.score_bias_lambda)                      # [Np,Kp]
                            P_pc = sinkhorn_assign_to_clusters(t_pred, q_pred, t_cent, q_cent,
                                                            s_t=0.10, w_q=0.05, tau_a=0.1)                    # [Np,Kp]
                            Li = (P_pc * L_ik).sum(dim=1)                                                        # [Np]                        
                            Lis.append(Li.mean())
                        else:
                            Li = ot_loss(t_pred, q_pred, t_gt, q_gt, s_gt, s_t=0.05, w_t=1.0, w_q=2.0,
                                        score_lambda=0.05, tau=0.1, n_iter=30,
                                            use_score_as_mass=False, reduction="mean")

                                    
                        Lis.append(Li)

                        se3 = se3_repulsion(t_pred, q_pred, r_t=0.01, r_q_deg=10.0, w_t=1.0, w_q= 1.0)
                        se3s.append(se3)                      

                        '''
                        D_reg = w_t_reg * Dt_l2 + w_q_reg * Dq / torch.pi
                        if self.score_bias_lambda > 0.0:
                            D_reg = D_reg + self.score_bias_lambda * Ds # broadcast

                        
                        Knn = min(16, D_reg.shape[1])
                        Dk, idxk = torch.topk(D_reg, k=Knn, dim=1, largest=False)  # [Np,Knn]
                        
                        # Soft-min with subtract-min

                        tau = 0.3  # start 0.3–0.5, anneal to 0.1–0.2
                        m = Dk.min(dim=1, keepdim=True).values
                        weights = torch.softmax(-(Dk - m) / tau, dim=1)            # [Np,Knn]
                        Li = (weights * Dk).mean()                             # [Np]
                        
                        #tau_reg = 0.10
                        #Li = (-tau_reg * torch.logsumexp(-D_reg / tau_reg, dim=1)).mean()
                        Lis.append(Li)
                        '''
                        
                    # 3c) Metrics: nearest-GT proximity (only if requested)
                    if return_metric:
                        if D_reg is None:
                            D_reg = w_t_reg * Dt_l2 + w_q_reg * Dq_n
                            if self.score_bias_lambda > 0.0:
                                D_reg = D_reg + self.score_bias_lambda * (1.0 - s_gt.unsqueeze(0))  # broadcast

                        
                        # argmin index over GT for each predicted grasp

                        idx = D_reg.argmin(dim=1)              # [Np]
                        row = torch.arange(idx.shape[0], device=idx.device)

                        dt_min_joint = Dt_l2[row, idx]           # [Np] meters
                        dq_min_joint = Dq[row, idx]              # [Np] radians

                        #print('min', dt_min_joint.detach().cpu(), dq_min_joint.detach().cpu() * (180.0 / torch.pi), idx)
                        d_comb_min  = (w_t_m * dt_min_joint + w_q_rad * dq_min_joint)
                        

                        nn_t_sum   += float(dt_min_joint.mean().item())
                        nn_ang_sum += float(dq_min_joint.mean().item() * (180.0 / torch.pi))
                        nn_comb_sum+= float(d_comb_min.mean().item())
                        nn_cnt     += 1

                        close_mask = (dt_min_joint <= t_thr) & (dq_min_joint * (180.0 / torch.pi) <= ang_thr_deg)
                        n_close_slots += int(close_mask.float().mean().item() > 0.0)
                
                if len(Lis) > 0:
                    reg_sum += sum(Lis) / len(Lis)
                    se3_sum += sum(se3s) / len(se3s)
                    reg_cnt += 1

                #print('############')
                #input()
                        
                    

            # Emit losses
            if wu > 1e-8 and reg_cnt > 0:
                losses["loss_grasp_reg"] = (self.loss_grasp_reg_coef * wu * (reg_sum / reg_cnt))
                losses["loss_grasp_se3"] = (1.0 * wu * (se3_sum / reg_cnt))
                #print('loss_grasp_reg', losses["loss_grasp_reg"])
                #print('loss_grasp_se3', losses["loss_grasp_se3"])

            # Emit metrics
            if return_metric and nn_cnt > 0:
                losses["grasp_nn_trans_avg"]      = torch.as_tensor(nn_t_sum / nn_cnt, device=device)
                losses["grasp_nn_angle_deg_avg"]  = torch.as_tensor(nn_ang_sum / nn_cnt, device=device)
                losses["grasp_nn_combined_avg"]   = torch.as_tensor(nn_comb_sum / nn_cnt, device=device)
                losses["grasp_nn_close_pct"]      = torch.as_tensor(100.0 * (n_close_slots / max(nn_cnt, 1)), device=device)

        
        # 4) Object 6D geometry (Chamfer on point clouds) – only in 6D mode
        chamfer_sum, chamfer_cnt = 0.0, 0
        if add_pose_extra_query_embed and self.loss_pose_chamfer_coef > 0:
            p_good_last = torch.softmax(crit_pred_logits, dim=-1)[..., 1]  # [B,T]
            for b in range(B):
                if slot_inst_ids is None:
                    continue
                for k in range(K):
                    inst_id = int(slot_inst_ids[b, k].item()) if slot_inst_ids[b, k].item() >= 0 else -1
                    if inst_id < 0:
                        continue
                    gt_poses = targets[b].get("poses", {}) or {}
                    gt_models = targets[b].get("models", {}) or {}
                    p_gt = gt_poses.get(inst_id, None)
                    M_gt = gt_models.get(inst_id, None)
                    if (p_gt is None) or (M_gt is None) or (M_gt.numel() == 0):
                        continue

                    idx_b = torch.nonzero(crit_pred_mask[b], as_tuple=False).squeeze(1)
                    if idx_b.numel() < K * Np:
                        continue
                    p_slot = p_good_last[b, idx_b].view(K, Np)[k]  # [Np]
                    best_i = torch.argmax(p_slot).item()
                    pose_k = pose_preds[b, k, best_i]  # [8]

                    t_pred = pose_k[:3]
                    q_pred = pose_k[3:7]
                    R_pred = _quat_wxyz_to_R(q_pred.unsqueeze(0)).squeeze(0)

                    R_gt = torch.tensor(p_gt['R'], dtype=R_pred.dtype, device=R_pred.device) if not isinstance(p_gt['R'], torch.Tensor) else p_gt['R'].to(R_pred)
                    t_gt = torch.tensor(p_gt['t'], dtype=t_pred.dtype, device=t_pred.device) if not isinstance(p_gt['t'], torch.Tensor) else p_gt['t'].to(t_pred)

                    Pm = M_gt.to(R_pred).float()
                    P_pred = (Pm @ R_pred.T) + t_pred.view(1, 3)
                    P_true = (Pm @ R_gt.T) + t_gt.view(1, 3)

                    chamfer = chamfer_symmetric_torch(P_pred, P_true)
                    chamfer_sum += float(chamfer.item())
                    chamfer_cnt += 1

            if chamfer_cnt > 0:
                losses["loss_pose_chamfer"] = self.loss_pose_chamfer_coef * (torch.as_tensor(chamfer_sum / chamfer_cnt, device=device))

        # 5) Pose slot classification CE + optional accuracy
        if isinstance(pose_slot_logits, torch.Tensor) and (slot_inst_ids is not None) and (inst_id_2_slot_id is not None):
            Bp, Kp, Npp, M = pose_slot_logits.shape
            targets_bkn = self._build_slot_targets(slot_inst_ids, inst_id_2_slot_id, Kp, Npp, M, pose_slot_logits.device)
            logits_flat = pose_slot_logits.reshape(Bp * Kp * Npp, M)
            targets_flat = targets_bkn.reshape(-1)
            losses['loss_pose_slot_cls'] = F.cross_entropy(logits_flat, targets_flat, reduction='mean')

            if return_metric:
                with torch.no_grad():
                    pred_cls = logits_flat.argmax(dim=-1)
                    acc = (pred_cls == targets_flat).float().mean()
                    losses['pose_slot_acc'] = torch.as_tensor(float(acc.item()) * 100.0, device=device)
                    
        # 6) Metrics summary wrt critique (already collected)
        if return_metric:
            if avg_crit_score_cnt > 0:
                losses["pose_avg_crit_score"] = torch.as_tensor(avg_crit_score_sum / avg_crit_score_cnt, device=device)
            # else: omit; printing every time is noisy
            if conf_total_cnt > 0:
                losses["pose_conf_err_l1"] = torch.as_tensor(conf_abs_err_sum / conf_total_cnt, device=device)
                losses["pose_conf_err_rate"] = torch.as_tensor(100.0 * (conf_err_bin_sum / conf_total_cnt), device=device)
            if chamfer_cnt > 0:
                losses["pose_chamfer_avg"] = torch.as_tensor(chamfer_sum / chamfer_cnt, device=device)

        nan_encountered = False
        l = 0
        for key, v in losses.items():
            if torch.isnan(v):
                print(f'WARN {key} is {v}')
                nan_encountered = True
                losses[key] = losses[key].new_zeros(())
                
            if key.startswith('loss_'):
                l += 1

        if nan_encountered or l == 0:
            print('wu', wu)
            print('l', l)
            print('nan_encountered', nan_encountered)
            print('losses', losses)
            input()

        return losses
        
    
    