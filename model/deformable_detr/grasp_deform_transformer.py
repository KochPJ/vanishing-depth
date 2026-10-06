# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE_DEFORMABLE_DETR in the LICENSES folder for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch._dynamo as dynamo
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from model.deformable_detr.util.misc import inverse_sigmoid
from model.deformable_detr.ops.modules import MSDeformAttn
from typing import List, Optional, Tuple
import random
from model.deformable_detr.util.misc import nested_tensor_from_tensor_list, is_main_process


# Balanced, mixed good/bad/random chunk builder for critique head

import math
import torch
from typing import Dict, List, Tuple, Optional

def _grasps_list_to_pose_tensor(grasps: List[dict], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = []
    for g in grasps or []:
        xyz = g.get('xyz', None)
        q = g.get('quat_wxyz', None)
        if xyz is None or q is None:
            continue
        t = torch.tensor([float(x) for x in (*xyz, *q)], device=device, dtype=dtype)
        if t.numel() == 7:
            rows.append(t)
    if len(rows) == 0:
        return torch.empty(0, 7, device=device, dtype=dtype)
    return torch.stack(rows, dim=0)


def _compute_bounds_from_maps(good_map: Dict[int, torch.Tensor], bad_map: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    device = None
    dtype  = None
    pools = []
    for mp in (good_map, bad_map):
        for _, t in mp.items():
            if t.numel() > 0:
                if device is None: device = t.device
                if dtype  is None: dtype  = t.dtype
                pools.append(t[:, :3])
    if len(pools) == 0:
        device = device or torch.device('cpu')
        dtype  = dtype or torch.float32
        return (torch.tensor([-0.2, -0.2, 0.2], device=device, dtype=dtype),
                torch.tensor([+0.2, +0.2, 1.0], device=device, dtype=dtype))
    t_all = torch.cat(pools, dim=0)
    tmin = t_all.min(dim=0).values
    tmax = t_all.max(dim=0).values
    extent = torch.clamp(tmax - tmin, min=1e-6)
    pad = 0.5 * extent
    return (tmin - pad), (tmax + pad)

def _dbg_print_nonfinite(name, t):
    # Only check floating tensors
    if not is_main_process:
        return None

    if not (torch.is_tensor(t) and t.dtype.is_floating_point):
        return None
    numel = t.numel()
    if numel == 0:
        return None
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    if n_nan or n_inf:
        pct_nan = 100.0 * n_nan / numel
        pct_inf = 100.0 * n_inf / numel
        print(
            f"[decoder] {name}: "
            f"NaN={n_nan} ({pct_nan:.3f}%), "
            f"Inf={n_inf} ({pct_inf:.3f}%), "
            f"numel={numel}, shape={tuple(t.shape)}, device={t.device}, dtype={t.dtype}"
        )
        input()
    return None


def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def _sample_noise_translations(n: int, tmin: torch.Tensor, tmax: torch.Tensor) -> torch.Tensor:
    if n <= 0:
        return torch.empty(0, 3, device=tmin.device, dtype=tmin.dtype)
    u = torch.rand(n, 3, device=tmin.device, dtype=tmin.dtype)
    return tmin[None, :] + u * (tmax - tmin)[None, :]

def _sample_balanced_from_map(inst_map: Dict[int, torch.Tensor], quota: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Fair-share across inst_id, then remainder distributed round-robin; pad by replacement if needed.
    Returns [quota,7] (or fewer if no data at all).
    """
    if quota <= 0 or len(inst_map) == 0:
        return torch.empty(0, 7, device=device, dtype=dtype)

    inst_ids = list(inst_map.keys())
    n_inst = len(inst_ids)

    # Shuffle order within each inst
    perms = {}
    for iid in inst_ids:
        M = inst_map[iid].size(0)
        if M > 0:
            perms[iid] = torch.randperm(M, device=device)
        else:
            perms[iid] = torch.empty(0, dtype=torch.long, device=device)

    per = quota // n_inst
    rem = quota - per * n_inst

    sel_list = []
    taken = {iid: 0 for iid in inst_ids}

    # Base fair share
    for iid in inst_ids:
        M = inst_map[iid].size(0)
        take = min(per, M)
        if take > 0:
            inds = perms[iid][:take]
            sel_list.append(inst_map[iid][inds])
            taken[iid] = take

    # Remainder round-robin
    if rem > 0:
        order = torch.randperm(n_inst, device=device)
        for j in range(rem):
            iid = inst_ids[int(order[j % n_inst])]
            M = inst_map[iid].size(0)
            cur = taken[iid]
            if M > cur:
                idx = perms[iid][cur:cur+1]
                sel_list.append(inst_map[iid][idx])
                taken[iid] = cur + 1
            elif M > 0:
                ridx = torch.randint(0, M, (1,), device=device)
                sel_list.append(inst_map[iid][ridx])

    if len(sel_list) == 0:
        # nothing available
        return torch.empty(0, 7, device=device, dtype=dtype)

    out = torch.cat(sel_list, dim=0)

    # If still short: pad with replacement from all available
    if out.size(0) < quota:
        cat_all = torch.cat([inst_map[iid] for iid in inst_ids if inst_map[iid].size(0) > 0], dim=0)
        if cat_all.size(0) > 0:
            deficit = quota - out.size(0)
            ridx = torch.randint(0, cat_all.size(0), (deficit,), device=device)
            out = torch.cat([out, cat_all[ridx]], dim=0)

    # If overflow (unlikely), trim randomly
    if out.size(0) > quota:
        perm = torch.randperm(out.size(0), device=device)
        out = out[perm[:quota]]

    # Final shuffle for randomness
    perm = torch.randperm(out.size(0), device=device)
    out = out[perm]
    return out


def _to_maps_packed_with_scores(gdict):
    poses_map, scores_map = {}, {}

    if isinstance(gdict, dict) and len(gdict) > 0:
        for iid, pack in gdict.items():
            xyz  = pack.get('xyz', None)
            quat = pack.get('quat_wxyz', None)
            sco  = pack.get('score', None)
            if not (isinstance(xyz, torch.Tensor) and isinstance(quat, torch.Tensor) and xyz.ndim == 2 and quat.ndim == 2):
                continue
            Kx, Kq = xyz.size(0), quat.size(0)
            if isinstance(sco, torch.Tensor) and sco.ndim >= 1:
                sc = sco.view(-1, 1)
                Ks = sc.size(0)
            else:
                Ks = min(Kx, Kq)
                sc = torch.zeros(Ks, 1, device=xyz.device, dtype=xyz.dtype)

            K = min(Kx, Kq, Ks)
            if K <= 0:
                continue

            poses_map[int(iid)]  = torch.cat([xyz[:K].to(torch.float32), quat[:K].to(torch.float32)], dim=-1)
            scores_map[int(iid)] = sc[:K].to(torch.float32).view(-1)
        return poses_map, scores_map

    if isinstance(gdict, list) and len(gdict) > 0:
        rows, scores = [], []
        dev = None
        for g in gdict:
            xyz = g.get('xyz'); q = g.get('quat_wxyz'); s = float(g.get('score', 0.0))
            if xyz is None or q is None: continue
            t = torch.tensor([*map(float, xyz + q)], dtype=torch.float32, device=dev or 'cuda')
            dev = t.device
            if t.numel() == 7:
                rows.append(t); scores.append(s)
        if rows:
            poses_map[1]  = torch.stack(rows, dim=0)
            scores_map[1] = torch.tensor(scores, dtype=torch.float32, device=poses_map[1].device)
    return poses_map, scores_map


@torch.no_grad()
def _sample_balanced_from_maps_with_ids(
    poses_map: Dict[int, torch.Tensor],
    scores_map: Dict[int, torch.Tensor],
    quota: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Balanced sampler across inst_id with inst-id tracking.
    Returns (poses [quota,7], scores [quota], inst_ids [quota]), possibly empty if no data.
    """
    if quota <= 0 or len(poses_map) == 0:
        return (torch.empty(0, 7, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=torch.long))

    inst_ids = list(poses_map.keys())
    n_inst = len(inst_ids)

    # Trim per inst to common K and keep on target device/dtype
    P_map, S_map, lengths = {}, {}, {}
    for iid in inst_ids:
        P = poses_map[iid].to(device=device, dtype=dtype)
        S = scores_map[iid].to(device=device, dtype=dtype)
        Kp = int(P.size(0))
        Ks = int(S.size(0))
        K = min(Kp, Ks)
        if K <= 0:
            P_map[iid] = P[:0]
            S_map[iid] = S[:0]
            lengths[iid] = 0
        else:
            P_map[iid] = P[:K]
            S_map[iid] = S[:K]
            lengths[iid] = K

    if n_inst == 0:
        return (torch.empty(0, 7, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=torch.long))

    per = quota // n_inst
    rem = quota - per * n_inst

    sel_pose, sel_score, sel_iid = [], [], []
    taken = {iid: 0 for iid in inst_ids}
    perms = {}
    for iid in inst_ids:
        M = lengths[iid]
        perms[iid] = torch.randperm(M, device=device, dtype=torch.long) if M > 0 else torch.empty(0, dtype=torch.long, device=device)

    # Base fair share
    for iid in inst_ids:
        M = lengths[iid]
        if M <= 0:
            continue
        take = min(per, M)
        if take > 0:
            idx = perms[iid][:take]
            sel_pose.append(P_map[iid].index_select(0, idx))
            sel_score.append(S_map[iid].index_select(0, idx))
            sel_iid.append(torch.full((take,), int(iid), device=device, dtype=torch.long))
            taken[iid] = take

    # Remainder round-robin
    if rem > 0 and n_inst > 0:
        order = torch.randperm(n_inst).tolist()
        for j in range(rem):
            iid = inst_ids[order[j % n_inst]]
            M = lengths[iid]
            if M <= 0:
                continue
            cur = taken[iid]
            if cur < M:
                idx = torch.tensor([cur], device=device, dtype=torch.long)
            else:
                idx = torch.randint(0, M, (1,), device=device, dtype=torch.long)
            sel_pose.append(P_map[iid].index_select(0, idx))
            sel_score.append(S_map[iid].index_select(0, idx))
            sel_iid.append(torch.full((1,), int(iid), device=device, dtype=torch.long))
            taken[iid] = min(cur + 1, M)

    if len(sel_pose) == 0:
        return (torch.empty(0, 7, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=torch.long))

    poses_out  = torch.cat(sel_pose,  dim=0)
    scores_out = torch.cat(sel_score, dim=0)
    iids_out   = torch.cat(sel_iid,   dim=0)

    # Pad/trim to quota
    N = poses_out.size(0)
    if N < quota:
        deficit = quota - N
        if N > 0:
            ridx = torch.randint(0, N, (deficit,), device=device, dtype=torch.long)
            poses_out  = torch.cat([poses_out,  poses_out.index_select(0, ridx)],  dim=0)
            scores_out = torch.cat([scores_out, scores_out.index_select(0, ridx)], dim=0)
            iids_out   = torch.cat([iids_out,   iids_out.index_select(0, ridx)],   dim=0)
        else:
            poses_out  = torch.zeros(deficit, 7, device=device, dtype=dtype)
            scores_out = torch.zeros(deficit,    device=device, dtype=dtype)
            iids_out   = torch.full((deficit,), -1, device=device, dtype=torch.long)
    elif N > quota:
        perm = torch.randperm(N, device=device, dtype=torch.long)[:quota]
        poses_out  = poses_out[perm]
        scores_out = scores_out[perm]
        iids_out   = iids_out[perm]

    # Final shuffle
    perm = torch.randperm(quota, device=device, dtype=torch.long)
    poses_out  = poses_out[perm]
    scores_out = scores_out[perm]
    iids_out   = iids_out[perm]
    return poses_out, scores_out, iids_out


@torch.no_grad()
def chunk_and_pad_poses_mixed_balanced(
    good_grasps_by_img: List[dict],
    bad_grasps_by_img:  List[dict],
    num_critiques: int,
    max_chunks: int = 3,
    good_frac_max: float = 0.34,
    bad_frac_max:  float = 0.33,
    device: torch.device = torch.device('cuda'),
    dtype: torch.dtype = torch.float32,
) -> Tuple[
    List[torch.Tensor],  # poses_chunks:  [B, NCrit, 7]
    List[torch.Tensor],  # masks_chunks:  [B, NCrit] with {0:good,1:bad,2:random}
    List[torch.Tensor],  # scores_chunks: [B, NCrit]
    List[torch.Tensor],  # instid_chunks: [B, NCrit] long (inst_id; random entries sampled from visible insts, or -1 if none)
]:
    """
    Return four lists (length=max_chunks):
      - poses_chunks[b]:   [B, NCrit, 7]    (tx,ty,tz,qw,qx,qy,qz)
      - masks_chunks[b]:   [B, NCrit]       (0=good, 1=bad, 2=random)
      - scores_chunks[b]:  [B, NCrit]       ([0,1] for good/bad; 0 for random)
      - instid_chunks[b]:  [B, NCrit] long  (inst_id for each pose; random draws from union of visible insts; -1 if none)

    """
    assert num_critiques > 0
    B = len(good_grasps_by_img)

    k_good = int(round(num_critiques * good_frac_max))
    k_bad  = int(round(num_critiques * bad_frac_max))
    k_good = max(0, min(k_good, num_critiques))
    k_bad  = max(0, min(k_bad,  num_critiques - k_good))

    poses_chunks:  List[torch.Tensor] = []
    masks_chunks:  List[torch.Tensor] = []
    scores_chunks: List[torch.Tensor] = []
    instid_chunks: List[torch.Tensor] = []

    def _bounds_from_maps(gpose: Dict[int, torch.Tensor], bpose: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pools = []
        for mp in (gpose, bpose):
            for _, t in mp.items():
                if t.numel() > 0:
                    pools.append(t[:, :3])
        if len(pools) == 0:
            tmin = torch.tensor([-0.2, -0.2, 0.2], device=device, dtype=dtype)
            tmax = torch.tensor([+0.2, +0.2, 1.0], device=device, dtype=dtype)
            return tmin, tmax
        t_all = torch.cat(pools, dim=0)
        tmin = t_all.min(dim=0).values
        tmax = t_all.max(dim=0).values
        extent = torch.clamp(tmax - tmin, min=1e-6)
        pad = 0.5 * extent
        return (tmin - pad), (tmax + pad)

    def _random_quaternions(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        u1 = torch.rand(n, device=device, dtype=dtype)
        u2 = torch.rand(n, device=device, dtype=dtype)
        u3 = torch.rand(n, device=device, dtype=dtype)
        sqrt1 = torch.sqrt(1.0 - u1)
        sqrt2 = torch.sqrt(u1)
        two_pi = 2.0 * torch.pi
        theta1 = two_pi * u2
        theta2 = two_pi * u3
        qx = sqrt1 * torch.sin(theta1)
        qy = sqrt1 * torch.cos(theta1)
        qz = sqrt2 * torch.sin(theta2)
        qw = sqrt2 * torch.cos(theta2)
        q = torch.stack([qw, qx, qy, qz], dim=-1)
        q = q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        return q

    def _sample_noise_translations(n: int, tmin: torch.Tensor, tmax: torch.Tensor) -> torch.Tensor:
        if n <= 0:
            return torch.empty(0, 3, device=device, dtype=dtype)
        u = torch.rand(n, 3, device=device, dtype=dtype)
        return tmin[None, :] + u * (tmax - tmin)[None, :]

    for _ in range(max_chunks):
        poses_c   = torch.zeros(B, num_critiques, 7, device=device, dtype=dtype)
        mask_c    = torch.full((B, num_critiques), 2, device=device, dtype=torch.long)  # 2=random
        score_c   = torch.zeros(B, num_critiques, device=device, dtype=dtype)           # 0 for random
        instid_c  = torch.full((B, num_critiques), -1, device=device, dtype=torch.long)

        for b in range(B):
            gpose_map, gscore_map = _to_maps_packed_with_scores(good_grasps_by_img[b] or {})
            bpose_map, bscore_map = _to_maps_packed_with_scores(bad_grasps_by_img[b]  or {})

            # Union of visible inst ids in this image (for assigning randoms)
            inst_pool = list(set(list(gpose_map.keys()) + list(bpose_map.keys())))
            if len(inst_pool) == 0:
                inst_pool = [-1]

            # Bounds for random translations
            tmin, tmax = _bounds_from_maps(gpose_map, bpose_map)

            sel_good_pose, sel_good_score, sel_good_iid = _sample_balanced_from_maps_with_ids(
                gpose_map, gscore_map, k_good, device, dtype
            )
            sel_bad_pose,  sel_bad_score,  sel_bad_iid  = _sample_balanced_from_maps_with_ids(
                bpose_map, bscore_map, k_bad,  device, dtype
            )

            need_rand = max(0, num_critiques - (sel_good_pose.size(0) + sel_bad_pose.size(0)))
            if need_rand > 0:
                rand_t = _sample_noise_translations(need_rand, tmin, tmax)                 # [R,3]
                rand_q = _random_quaternions(need_rand, device=device, dtype=dtype)        # [R,4]
                sel_rand_pose  = torch.cat([rand_t, rand_q], dim=-1)                        # [R,7]
                sel_rand_score = torch.zeros(need_rand, device=device, dtype=dtype)         # 0.0
                # Assign a random inst_id from pool for each random proposal
                if len(inst_pool) == 1:
                    sel_rand_iid = torch.full((need_rand,), int(inst_pool[0]), device=device, dtype=torch.long)
                else:
                    idx = torch.randint(0, len(inst_pool), (need_rand,), device=device)
                    sel_rand_iid = torch.tensor([int(inst_pool[i.item()]) for i in idx], device=device, dtype=torch.long)
            else:
                sel_rand_pose  = torch.empty(0, 7, device=device, dtype=dtype)
                sel_rand_score = torch.empty(0,    device=device, dtype=dtype)
                sel_rand_iid   = torch.empty(0,    device=device, dtype=torch.long)

            merged_pose   = torch.cat([sel_good_pose,  sel_bad_pose,  sel_rand_pose],  dim=0)
            merged_score  = torch.cat([sel_good_score, sel_bad_score, sel_rand_score], dim=0)
            merged_iid    = torch.cat([sel_good_iid,   sel_bad_iid,   sel_rand_iid],   dim=0)
            labels = torch.cat([
                torch.zeros(sel_good_pose.size(0), device=device, dtype=torch.long),     # 0=good
                torch.ones(sel_bad_pose.size(0),  device=device, dtype=torch.long),      # 1=bad
                torch.full((sel_rand_pose.size(0),), 2, device=device, dtype=torch.long) # 2=random
            ], dim=0)

            # pad/trim to NCrit
            Ncur = merged_pose.size(0)
            if Ncur < num_critiques:
                deficit = num_critiques - Ncur
                # pad with random
                rand_t = _sample_noise_translations(deficit, tmin, tmax)
                rand_q = _random_quaternions(deficit, device=device, dtype=dtype)
                pad_pose  = torch.cat([rand_t, rand_q], dim=-1)
                pad_score = torch.zeros(deficit, device=device, dtype=dtype)
                if len(inst_pool) == 1:
                    pad_iid = torch.full((deficit,), int(inst_pool[0]), device=device, dtype=torch.long)
                else:
                    idx = torch.randint(0, len(inst_pool), (deficit,), device=device)
                    pad_iid = torch.tensor([int(inst_pool[i.item()]) for i in idx], device=device, dtype=torch.long)

                merged_pose   = torch.cat([merged_pose,  pad_pose],  dim=0)
                merged_score  = torch.cat([merged_score, pad_score], dim=0)
                merged_iid    = torch.cat([merged_iid,   pad_iid],   dim=0)
                labels        = torch.cat([labels, torch.full((deficit,), 2, device=device, dtype=torch.long)], dim=0)

            if merged_pose.size(0) > num_critiques:
                idx = torch.randperm(merged_pose.size(0), device=device)[:num_critiques]
                merged_pose   = merged_pose[idx]
                merged_score  = merged_score[idx]
                merged_iid    = merged_iid[idx]
                labels        = labels[idx]
            else:
                idx = torch.randperm(num_critiques, device=device)
                merged_pose   = merged_pose[idx]
                merged_score  = merged_score[idx]
                merged_iid    = merged_iid[idx]
                labels        = labels[idx]

            poses_c[b]  = merged_pose
            mask_c[b]   = labels
            score_c[b]  = merged_score
            instid_c[b] = merged_iid

        poses_chunks.append(poses_c)
        masks_chunks.append(mask_c)
        scores_chunks.append(score_c)
        instid_chunks.append(instid_c)

    return poses_chunks, masks_chunks, scores_chunks, instid_chunks


def chunk_predicted_poses_with_ids(poses_bxN7: torch.Tensor,
                                   model_embed_flat: torch.Tensor,  
                                   num_critiques: int,
                                   max_chunks: int):
    """
    Returns:
      - chunks:     list of [B, NCrit, 7]
      - masks           list of [B, NCrit] bool
      - model_chunks:      list of [B, NCrit, C]

    Robust to s >= N by padding whole chunk.
    """
    B, N, D = poses_bxN7.shape
    B, N, C = model_embed_flat.shape
    
    chunks, masks, model_chunks = [], [], []
    for c in range(max_chunks):
        s = c * num_critiques
        if s > N:
            break
        e = min(N, s + num_critiques)


        # Safe slices (OK even if s >= N -> empty slice)
        slice_poses = poses_bxN7[:, s:e]       # [B, k, D]
        slice_model   = model_embed_flat[:, s:e]     # [B, k, C]
        k = slice_poses.shape[1]               # actual number of real items in this chunk

        if k < num_critiques:
            pad = num_critiques - k
            pad_poses = poses_bxN7.new_zeros(B, pad, D)
            pad_model   = model_embed_flat.new_full((B, pad, C), 0)
            chunk     = torch.cat([slice_poses, pad_poses], dim=1)            # [B, NCrit, D]
            model_c     = torch.cat([slice_model,   pad_model],   dim=1)            # [B, NCrit, C]
            mask_c    = torch.cat([
                torch.ones(B, k,   dtype=torch.bool, device=poses_bxN7.device),
                torch.zeros(B, pad, dtype=torch.bool, device=poses_bxN7.device)
            ], dim=1)                                                         # [B, NCrit]
        else:
            chunk, model_c = slice_poses, slice_model
            mask_c = torch.ones(B, num_critiques, dtype=torch.bool, device=poses_bxN7.device)

        chunks.append(chunk)
        masks.append(mask_c)
        model_chunks.append(model_c)

    return chunks, masks, model_chunks




class ModelPointNetEncoder(nn.Module):
    def __init__(self, out_dim: int, hidden=(64, 128, 256), use_meanmax=True):
        super().__init__()
        layers = []
        c = 3
        for h in hidden:
            layers += [nn.Linear(c, h), nn.ReLU(), nn.LayerNorm(h)]
            c = h
        self.per_point = nn.Sequential(*layers)
        self.use_meanmax = use_meanmax
        agg_in = c * 2 if use_meanmax else c
        self.proj = nn.Sequential(
            nn.Linear(agg_in, 2 * out_dim),
            nn.ReLU(),
            nn.Linear(2 * out_dim, out_dim),
        )

    @staticmethod
    def _normalize(pc):  # pc: (B,N,3)
        c = pc.mean(dim=1, keepdim=True)
        x = pc - c
        s = x.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6)
        return x / s

    def forward(self, pc):  # (B,N,3)
        x = self._normalize(pc)
        f = self.per_point(x)            # (B,N,C)
        if self.use_meanmax:
            g = torch.cat([f.mean(dim=1), f.max(dim=1).values], dim=-1)
        else:
            g = f.max(dim=1).values
        return self.proj(g)              # (B,out_dim)


class PosePositionalEncoder(nn.Module):
    """
    Upsamples 7D poses (tx,ty,tz,qw,qx,qy,qz) to C-dim with sinusoidal posenc:
      - normalize xyz by max_xyz (default 15.0) and clamp to [-1, 1]
      - multiply all 7 dims by scale (default 2*pi)
      - use temperature (default 0.0003)
      - per feature dim_t length = 2 * (C // (2*7)), so total used = 7 * 2 * f
      - zero-pad to reach exactly C

    """
    def __init__(self, out_dim: int, temperature: float = 0.0003, scale: float = 2.0 * math.pi, max_xyz: float = 15.0):
        super().__init__()
        assert out_dim > 0, "out_dim must be > 0"
        self.out_dim = int(out_dim)
        self.temperature = float(temperature)
        self.scale = float(scale)
        self.max_xyz = float(max_xyz)

        # channels per feature: C // (2*7)
        self.freqs_per_feat = max(int(self.out_dim // (2 * 7)), 0)
        self.per_feat_dim = 2 * self.freqs_per_feat  # length of dim_t for each scalar


        # Precompute dim_t for per-feature encoding (same for all 7)
        if self.per_feat_dim > 0:
            dim_t = torch.arange(self.per_feat_dim, dtype=torch.float32)
            # match your depth pos-encoding scheme
            dim_t = (2 * torch.div(dim_t, 2, rounding_mode='trunc')) / float(self.per_feat_dim)
            dim_t = self.temperature ** dim_t
            self.register_buffer("dim_t", dim_t, persistent=False)
        else:
            self.register_buffer("dim_t", torch.empty(0), persistent=False)


    def _encode_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,N] scalar
        returns [B,N,per_feat_dim] after sin/cos packing
        """
        if self.per_feat_dim == 0:
            # no capacity -> return zeros
            return torch.zeros(x.shape[0], x.shape[1], 0, dtype=x.dtype, device=x.device)
        # Expand and divide by dim_t
        v = (x.unsqueeze(-1) * self.scale)  # [B,N,1] -> scale
        v = v / self.dim_t.to(v.device)  # [B,N,per_feat_dim]
        # sin on even, cos on odd indices, then flatten back to per_feat_dim
        enc = torch.stack((v[..., 0::2].sin(), v[..., 1::2].cos()), dim=-1).flatten(-2)  # [B,N,per_feat_dim]
        return enc

    def forward(self, poses_bnx7: torch.Tensor) -> torch.Tensor:
        """
        poses_bnx7: [B, N, 7] in order [tx,ty,tz,qw,qx,qy,qz]
        returns: [B, N, C]
        """
        B, N, D = poses_bnx7.shape
        assert D == 7, f"Expected 7D poses, got {D}"


        # Normalize xyz by max_xyz -> [-1,1], keep quaternions as-is (already in [-1,1])
        t = poses_bnx7[..., :3].clamp_min(-self.max_xyz).clamp_max(self.max_xyz) / self.max_xyz  # [B,N,3]
        q = poses_bnx7[..., 3:]  # [B,N,4]

        # Encode each of the 7 scalars
        enc_parts = []
        for i in range(3):
            enc_parts.append(self._encode_scalar(t[..., i]))  # [B,N,per_feat_dim]
        for i in range(4):
            enc_parts.append(self._encode_scalar(q[..., i]))  # [B,N,per_feat_dim]

        if len(enc_parts) == 0:
            used = 0
            enc_cat = torch.zeros(B, N, 0, dtype=poses_bnx7.dtype, device=poses_bnx7.device)
        else:
            enc_cat = torch.cat(enc_parts, dim=-1)  # [B,N, 7*per_feat_dim]
            used = enc_cat.shape[-1]  # 7 * per_feat_dim = 7 * 2 * (C // (2*7)) <= C

        # Zero-pad to exactly C
        if used < self.out_dim:
            pad = torch.zeros(B, N, self.out_dim - used, dtype=enc_cat.dtype, device=enc_cat.device)
            enc_full = torch.cat([enc_cat, pad], dim=-1)  # [B,N,C]
        else:
            # Truncate if over (shouldn't happen with integer division, but safe-guard)
            enc_full = enc_cat[..., :self.out_dim]
        return enc_full

def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    """
    Normalize quaternion with eps and canonicalize hemisphere; replace non-finite by identity.
    q: (...,4) [w,x,y,z]
    """
    if not isinstance(q, torch.Tensor) or q.numel() == 0:
        return q
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = q / norm
    # hemisphere: w >= 0
    q = torch.where(q[..., :1] < 0, -q, q)
    return q

def _random_quaternions(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Uniform random unit quaternions (Shoemake method).
    Returns [n,4] in [qw,qx,qy,qz].
    """
    u1 = torch.rand(n, device=device, dtype=dtype)
    u2 = torch.rand(n, device=device, dtype=dtype)
    u3 = torch.rand(n, device=device, dtype=dtype)

    sqrt1 = torch.sqrt(1.0 - u1)
    sqrt2 = torch.sqrt(u1)
    two_pi = 2.0 * math.pi

    theta1 = two_pi * u2
    theta2 = two_pi * u3

    qx = sqrt1 * torch.sin(theta1)
    qy = sqrt1 * torch.cos(theta1)
    qz = sqrt2 * torch.sin(theta2)
    qw = sqrt2 * torch.cos(theta2)
    q = torch.stack([qw, qx, qy, qz], dim=-1)
    return _normalize_quat(q)

def _compute_translation_bounds(gt_poses_b: torch.Tensor, margin_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    gt_poses_b: [M,7] (tx,ty,tz,qw,qx,qy,qz)
    Returns (min_t, max_t) expanded by margin_ratio on the range.
    """
    if gt_poses_b.numel() == 0:
        # default bounds if no GT
        min_t = torch.tensor([-0.2, -0.2, 0.2], dtype=gt_poses_b.dtype, device=gt_poses_b.device)  # e.g., z positive
        max_t = torch.tensor([+0.2, +0.2, 1.0], dtype=gt_poses_b.dtype, device=gt_poses_b.device)
        return min_t, max_t

    t = gt_poses_b[:, :3]  # [M,3]
    tmin = t.min(dim=0).values
    tmax = t.max(dim=0).values
    extent = torch.clamp(tmax - tmin, min=1e-6)
    pad = margin_ratio * extent
    return tmin - pad, tmax + pad

def _sample_noise_translations(n: int, tmin: torch.Tensor, tmax: torch.Tensor) -> torch.Tensor:
    """
    Uniform sample in [tmin, tmax], returns [n,3].
    """
    u = torch.rand(n, 3, device=tmin.device, dtype=tmin.dtype)
    return tmin[None, :] + u * (tmax - tmin)[None, :]

def _balance_gt_per_object(gt_poses_b: torch.Tensor,
                           gt_obj_ids_b: Optional[torch.Tensor],
                           max_per_obj: Optional[int]) -> torch.Tensor:
    """
    Select up to max_per_obj GT poses per object id (if provided).
    gt_poses_b: [M,7]
    gt_obj_ids_b: [M] or None
    Returns filtered GT poses [M',7].
    """
    if max_per_obj is None or gt_obj_ids_b is None or gt_obj_ids_b.numel() == 0:
        return gt_poses_b
    sel_idxs = []
    unique_ids = torch.unique(gt_obj_ids_b)
    for oid in unique_ids.tolist():
        mask = (gt_obj_ids_b == oid)
        idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if idxs.numel() <= max_per_obj:
            sel_idxs.append(idxs)
        else:
            # deterministically or randomly choose
            choose = idxs[:max_per_obj]  # take first max_per_obj; switch to random for randomness
            sel_idxs.append(choose)
    if len(sel_idxs) == 0:
        return gt_poses_b[:0]
    sel = torch.cat(sel_idxs, dim=0)
    return gt_poses_b[sel]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def get_activation(name: str) -> nn.Module:
    name = name.lower()
    act_map = {
        "relu":       lambda: nn.ReLU(inplace=True),
        "gelu":       lambda: nn.GELU(),
        "silu":       lambda: nn.SiLU(inplace=True),   # also known as swish
        "leaky_relu": lambda: nn.LeakyReLU(0.01, inplace=True),
        "elu":        lambda: nn.ELU(inplace=True),
        "tanh":       lambda: nn.Tanh(),
        "sigmoid":    lambda: nn.Sigmoid(),
        "softplus":   lambda: nn.Softplus(),
        "selu":       lambda: nn.SELU(),
        "identity":   lambda: nn.Identity(),
    }
    if name not in act_map:
        raise ValueError(f"Unsupported activation '{name}'. "
                         f"Choose from {list(act_map.keys())}.")
    return act_map[name]()


class MLP(nn.Module):
    """
    Simple MLP with the same activation for all layers.
    If act_last=False, the last layer has no activation.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: str = "relu",
        act_last: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        sizes = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1], bias=bias) for i in range(num_layers)
        )
        self.num_layers = num_layers
        self.act_last = act_last
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # apply activation on all but last layer; apply on last only if act_last=True
            if i < self.num_layers - 1 or self.act_last:
                x = self.act(x)
        return x

def _fibonacci_sphere(n: int) -> torch.Tensor:
    """
    Returns [n,3] points roughly uniformly distributed on unit sphere [-1,1]^3.
    """
    # Golden angle
    ga = math.pi * (3.0 - math.sqrt(5.0))
    pts = []
    for i in range(n):
        y = 1.0 - (2.0 * (i + 0.5) / float(n))       # in [-1,1]
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = ga * i
        x = r * math.cos(theta)
        z = r * math.sin(theta)
        pts.append([x, y, z])
    return torch.tensor(pts, dtype=torch.float32)

    

class DeformableTransformer(nn.Module):
    def __init__(self, d_model_det=256, d_model_obj=512, shape_dims=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers_det=6, num_decoder_layers_obj=6, num_decoder_layers_grasp=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=True,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 num_det_queries=100, num_obj_queries=8, num_grasp_queries=3, max_objects=10,
                 num_classes=2, hidden_dim=1024, with_obj=True, obj_refine_mul=0.01, with_critique=False, with_pose=False, num_poses=5,
                 #add_critique_extra_query_embed=False, add_pose_extra_query_embed=False, add_model_embed=False, dim_encoder=768
                 #):
                add_critique_extra_query_embed=False,
                add_pose_extra_query_embed=False,
                add_model_embed=False,
                dim_encoder=768,
                use_grasp_diffusion=False,
                 grasp_diffusion_hidden_dim=512,
                 grasp_diffusion_layers=4,
                 grasp_diffusion_heads=8,
                 grasp_diffusion_train_steps=100,
                 grasp_diffusion_eval_steps=10,
                 grasp_diffusion_pose_min=None,
                 grasp_diffusion_pose_max=None,
                 grasp_diffusion_confidence_loss_weight=1.0):


        super().__init__()

        self.d_model_det = d_model_det
        self.d_model_obj = d_model_obj
        self.nhead = nhead
        self.num_det_queries = num_det_queries
        self.num_obj_queries = num_obj_queries
        self.num_grasp_queries = num_grasp_queries

        self.max_objects = max_objects
        self.obj_refine_mul = obj_refine_mul
        
        self.with_obj = with_obj
        self.with_pose = with_pose
        self.with_critique = with_critique
        self.num_poses = num_poses
           
        self.add_critique_extra_query_embed = add_critique_extra_query_embed
        self.add_pose_extra_query_embed = add_pose_extra_query_embed
        self.add_model_embed = add_model_embed
        self.num_object_tokens_per_det = 8 
        self.num_critiques = self.num_extra_query_embed = self.max_objects * self.num_object_tokens_per_det  


        self.use_grasp_diffusion = bool(use_grasp_diffusion)
        self.grasp_diffusion_hidden_dim = int(grasp_diffusion_hidden_dim)
        self.grasp_diffusion_layers = int(grasp_diffusion_layers)
        self.grasp_diffusion_heads = int(grasp_diffusion_heads)
        self.grasp_diffusion_train_steps = int(grasp_diffusion_train_steps)
        self.grasp_diffusion_eval_steps = int(grasp_diffusion_eval_steps)

        self.grasp_diffusion_pose_min = grasp_diffusion_pose_min
        self.grasp_diffusion_pose_max = grasp_diffusion_pose_max

        self.grasp_diffusion_confidence_loss_weight = float(
            grasp_diffusion_confidence_loss_weight
        )


        # detects objects
        encoder_layer = DeformableTransformerEncoderLayer(d_model_det, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        decoder_layer = DeformableTransformerDecoderLayer(d_model_det, dim_feedforward,
                                                        dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.det_encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        self.det_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers_det, return_intermediate_dec)
        
        # decodes grasping poses per object
        #self.decoder_grasp = DeformableTransformerDecoder(decoder_layer, num_decoder_layers_grasp, return_intermediate_dec)

        self.det_level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model_det))
        
        self.det_reference_points = nn.Linear(d_model_det, 2)
        #self.grasp_reference_points = nn.Linear(d_model, 2)

        self.det_query_embed = nn.Embedding(num_det_queries, d_model_det * 2)
        #self.grasp_query_embed = nn.Embedding(num_grasp_queries, d_model * 2)

        self.t_dim = 3 # xyz position 
        self.rot_dim = 4 # quat
        self.scale_dim = 1
        self.shape_dims = d_model_obj
        
        # Estimation Heads
        self.num_classes = num_classes
        # for object detection
        self.det_class_embed = nn.Linear(d_model_det, num_classes)
        self.det_bbox_embed = MLP(d_model_det, d_model_det, 4, 3)
        self.det_cls_token = MLP(d_model_det, d_model_det, dim_encoder, 3)
        self.det_delta_cls_gain = nn.Parameter(torch.full((num_decoder_layers_obj-1,), obj_refine_mul))

        self._init_parameters()

        self.det_class_embed = nn.ModuleList([copy.deepcopy(self.det_class_embed) for _ in range(num_decoder_layers_det)])
        self.det_bbox_embed = nn.ModuleList([copy.deepcopy(self.det_bbox_embed) for _ in range(num_decoder_layers_det)])
        self.det_cls_token = nn.ModuleList([copy.deepcopy(self.det_cls_token) for _ in range(num_decoder_layers_det)])
        self.det_decoder.det_bbox_embed = self.det_bbox_embed

               
        if self.with_obj:
            self.register_buffer("unit_sphere", _fibonacci_sphere(self.shape_dims) * 0.75)            # [D,3], float32 stays on device, saved/restored with model

            self.obj_query_embed = nn.Embedding(num_obj_queries, d_model_obj * 2)
            self.obj_query_proj =  nn.Linear(d_model_det, d_model_obj)
            self.obj_logits_proj =  nn.Linear(d_model_det, d_model_obj)
            
            self.obj_reference_points = nn.Linear(d_model_obj, 4) # deltas
        
            self.obj_level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model_obj))

            # estimates properties per object
            encoder_layer = DeformableTransformerEncoderLayer(d_model_obj, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
            decoder_layer = DeformableTransformerDecoderLayer(d_model_obj, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, dec_n_points)
            self.obj_encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
            self.obj_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers_obj, return_intermediate_dec)

            # for the object shape
            self.obj_scale_head = MLP(d_model_obj, d_model_obj, output_dim=self.scale_dim, num_layers=3, activation='gelu') # scales the object shape output activation in the loss = None -> real unbound linear 3d mapping
            self.obj_vol_head = MLP(d_model_obj, d_model_obj, output_dim=self.scale_dim, num_layers=3, activation='gelu') # % of the [-1, 1] cube covered by the output normalized point cloud (shape), activation in the loss = sigmoid (0-1 ratio)
            self.obj_pos_head = MLP(d_model_obj, d_model_obj, output_dim=self.t_dim, num_layers=3, activation='gelu') # positions the object shape from the origin into the 3d space, activation in the loss = None  -> real unbound linear 3d mapping
            self.obj_vis_head = MLP(d_model_obj, d_model_obj, output_dim=self.scale_dim, num_layers=3, activation='gelu') # visiblity 0-1 % of the point cloud 
            self.obj_shape_head = MLP(d_model_obj*3, self.shape_dims*3, output_dim=self.shape_dims*3, num_layers=3, activation='gelu') # defines the normalized shape ([-1: 1] x coord) in the origin (thus, incl the orientation), activation in the loss = None -> real unbound linear 3d mapping
            
            # learnable residules in the refinement layers of the output whcih help the model to produce small output deltas (the model can output large logits which are downscaled to small deltas)
            self.obj_delta_scale_gain = nn.Parameter(torch.full((num_decoder_layers_obj-1,), obj_refine_mul))
            self.obj_delta_vol_gain = nn.Parameter(torch.full((num_decoder_layers_obj-1,), obj_refine_mul))
            self.obj_delta_pos_gain = nn.Parameter(torch.full((num_decoder_layers_obj-1,), obj_refine_mul))
            self.obj_delta_vis_gain = nn.Parameter(torch.full((num_decoder_layers_obj-1,), obj_refine_mul))
            self.obj_delta_shape_gain = nn.Parameter(torch.full((num_decoder_layers_obj,), obj_refine_mul))
            self.obj_scale_head = nn.ModuleList([copy.deepcopy(self.obj_scale_head) for _ in range(num_decoder_layers_obj)])
            self.obj_vol_head = nn.ModuleList([copy.deepcopy(self.obj_vol_head) for _ in range(num_decoder_layers_obj)])
            self.obj_pos_head = nn.ModuleList([copy.deepcopy(self.obj_pos_head) for _ in range(num_decoder_layers_obj)])
            self.obj_vis_head = nn.ModuleList([copy.deepcopy(self.obj_vis_head) for _ in range(num_decoder_layers_obj)])
            self.obj_shape_head = nn.ModuleList([copy.deepcopy(self.obj_shape_head) for _ in range(num_decoder_layers_obj)])
        else:
            self.obj_query_embed, self.obj_query_proj, self.obj_logits_proj, self.obj_reference_points, self.obj_level_embed = None, None, None, None, None
            self.obj_encoder, self.obj_decoder = None, None,
            self.obj_scale_head, self.obj_vol_head, self.obj_pos_head, self.obj_vis_head, self.obj_shape_head = None, None, None, None, None 
            self.obj_delta_scale_gain, self.obj_delta_vol_gain, self.obj_delta_pos_gain, self.obj_delta_vis_gain, self.obj_delta_shape_gain = None, None, None, None, None

        self.pose_extra_query_embed, self.pose_extra_reference_points = None, None
        self.pose_model_embed, self.pose_model_query_embed, self.pose_model_reference_points, pose_model_query_2_qpos = None, None, None, None
        self.pose_logits_proj, self.pose_qpos_proj, self.pose_slot_head, self.pose_slot_delta_gain = None, None, None, None
        if self.with_pose:
            self.pose_query_embed = nn.Embedding(self.num_poses, d_model_obj * 2)
            self.pose_reference_points = nn.Linear(d_model_obj, 4)       
            self.pose_level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model_obj))
            encoder_layer = DeformableTransformerEncoderLayer(d_model_obj, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
            decoder_layer = DeformableTransformerDecoderLayer(d_model_obj, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, dec_n_points)
            self.pose_encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
            self.pose_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers_obj, return_intermediate_dec)
            
            self.pose_translation_delta_gain = nn.Parameter(torch.full((num_decoder_layers_obj-1,), obj_refine_mul))
            self.pose_orientation_delta_gain = nn.Parameter(torch.full((num_decoder_layers_obj-1,), obj_refine_mul))
            self.pose_confidence_delta_gain = nn.Parameter(torch.full((num_decoder_layers_obj-1,), obj_refine_mul))
            
            self.pose_translation_head = MLP(d_model_obj, d_model_obj, output_dim=3, num_layers=3, activation='gelu') # scales the object shape output activation in the loss = None -> real unbound linear 3d mapping
            self.pose_orientation_head = MLP(d_model_obj, d_model_obj, output_dim=4, num_layers=3, activation='gelu') # % of the [-1, 1] cube covered by the output normalized point cloud (shape), activation in the loss = sigmoid (0-1 ratio)
            self.pose_confidence_head = MLP(d_model_obj, d_model_obj, output_dim=1, num_layers=3, activation='gelu') # confidence how good the pose is
            self.pose_translation_head = nn.ModuleList([copy.deepcopy(self.pose_translation_head) for _ in range(num_decoder_layers_obj)])
            self.pose_orientation_head = nn.ModuleList([copy.deepcopy(self.pose_orientation_head) for _ in range(num_decoder_layers_obj)])
            self.pose_confidence_head = nn.ModuleList([copy.deepcopy(self.pose_confidence_head) for _ in range(num_decoder_layers_obj)])

            if self.add_model_embed:
                self.pose_model_embed = ModelPointNetEncoder(out_dim=d_model_obj)
                self.pose_model_query_embed = nn.Embedding(self.max_objects, d_model_obj * 2)
                self.pose_model_query_2_qpos = nn.Linear(d_model_obj, d_model_obj)
                self.pose_model_reference_points = nn.Linear(d_model_obj, 4)
                self.pose_slot_head = MLP(d_model_obj, d_model_obj, output_dim=self.max_objects+1, num_layers=3, activation='gelu') # adding one as no match
                self.pose_slot_head = nn.ModuleList([copy.deepcopy(self.pose_slot_head) for _ in range(num_decoder_layers_obj)])
                self.pose_slot_delta_gain = nn.Parameter(torch.full((num_decoder_layers_obj-1,), obj_refine_mul))
            

            if self.add_pose_extra_query_embed:
                self.pose_extra_query_embed = nn.Embedding(self.num_extra_query_embed, d_model_obj * 2)
                self.pose_extra_reference_points = nn.Linear(d_model_obj, 4) 
            
            if not self.with_obj:
                self.pose_logits_proj = nn.Linear(d_model_det, d_model_obj)
                self.pose_qpos_proj   = nn.Linear(d_model_det, d_model_obj)
            
        else: 
            self.pose_query_embed, self.pose_reference_points, self.pose_level_embed = None, None, None
            self.pose_encoder, self.pose_decoder = None, None
            self.pose_translation_head, self.pose_orientation_head, self.pose_translation_delta_gain, self.pose_orientation_delta_gain, self.pose_confidence_delta_gain  = None, None, None, None, None
            

        self.critique_extra_query_embed, self.critique_extra_reference_points = None, None
        if self.with_critique:
            self.critique_query_embed = nn.Embedding(self.num_critiques, d_model_obj * 2)
            self.critique_reference_points = nn.Linear(d_model_obj, 2)       
            self.critique_level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model_obj))
            encoder_layer = DeformableTransformerEncoderLayer(d_model_obj, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
            decoder_layer = DeformableTransformerDecoderLayer(d_model_obj, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, dec_n_points)
            self.critique_delta_gain = nn.Parameter(torch.full((num_decoder_layers_obj-1,), obj_refine_mul))
            self.critique_encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
            self.critique_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers_obj, return_intermediate_dec)
            
            self.critique_confidence_head = MLP(d_model_obj, d_model_obj, output_dim=2, num_layers=3, activation='gelu') # % of the [-1, 1] cube covered by the output normalized point cloud (shape), activation in the loss = sigmoid (0-1 ratio)
            self.critique_confidence_head = nn.ModuleList([copy.deepcopy(self.critique_confidence_head) for _ in range(num_decoder_layers_obj)])

            self.critique_tokenzizer = PosePositionalEncoder(out_dim=d_model_obj, temperature=0.0003, scale=2.0*math.pi, max_xyz=5.0)
            if self.add_critique_extra_query_embed:
                self.critique_extra_query_embed = nn.Embedding(self.num_extra_query_embed, d_model_obj * 2)
                self.critique_extra_reference_points = nn.Linear(d_model_obj, 2) 

            if self.add_model_embed:
                self.critique_model_embed = ModelPointNetEncoder(out_dim=d_model_obj)
                            
        else: 
            self.critique_query_embed, self.critique_reference_points, self.critique_level_embed = None, None, None
            self.critique_encoder, self.critique_decoder = None, None
            self.critique_translation_head, critique_delta_gain = None, None
            self.critique_tokenzizer = None
            self.critique_model_embed = None

        self._reset_parameters()

    def _init_parameters(self):        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.det_class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.det_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.det_bbox_embed.layers[-1].bias.data, 0)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.det_reference_points.weight.data, gain=1.0)
        constant_(self.det_reference_points.bias.data, 0.)
        normal_(self.det_level_embed)
        
        if self.with_obj:
            xavier_uniform_(self.obj_reference_points.weight.data, gain=1.0)
            constant_(self.obj_reference_points.bias.data, 0.)
            normal_(self.obj_level_embed)

        if self.with_pose:
            xavier_uniform_(self.pose_reference_points.weight.data, gain=1.0)
            constant_(self.pose_reference_points.bias.data, 0.)
            normal_(self.pose_level_embed)
            if self.add_pose_extra_query_embed:
                xavier_uniform_(self.pose_extra_reference_points.weight.data, gain=1.0)
                constant_(self.pose_extra_reference_points.bias.data, 0.)
        
        
        if self.with_critique:
            xavier_uniform_(self.critique_reference_points.weight.data, gain=1.0)
            constant_(self.critique_reference_points.bias.data, 0.)
            normal_(self.critique_level_embed)

            if self.add_critique_extra_query_embed:
                xavier_uniform_(self.critique_extra_reference_points.weight.data, gain=1.0)
                constant_(self.critique_extra_reference_points.bias.data, 0.)




    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, srcs_obj=None, pos_obj=None, srcs_pose=None, srcs_critique=None, targets=None):
        # prepare input for encoder
        src_flatten = []
        src_obj_flatten = []
        src_pose_flatten = []
        src_critique_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        lvl_pos_embed_obj_flatten = []
        lvl_pos_embed_pose_flatten = []
        lvl_pos_embed_critique_flatten = []
        spatial_shapes = []
        for lvl in range(len(srcs)):
            src, mask, pos_embed = srcs[lvl], masks[lvl], pos_embeds[lvl]
            h, w = src.shape[-2:]
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.det_level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

            if pos_obj is not None:
                pos_obj_embed = pos_obj[lvl].flatten(2).transpose(1, 2)    
            
            if srcs_obj is not None:
                src_obj = srcs_obj[lvl]
                src_obj = src_obj.flatten(2).transpose(1, 2)
                src_obj_flatten.append(src_obj)
                lvl_pos_embed_obj = pos_obj_embed + self.obj_level_embed[lvl].view(1, 1, -1)
                lvl_pos_embed_obj_flatten.append(lvl_pos_embed_obj)
                
            if srcs_pose is not None:
                src_pose = srcs_pose[lvl]
                src_pose = src_pose.flatten(2).transpose(1, 2)
                src_pose_flatten.append(src_pose)
                lvl_pos_embed_pose = pos_obj_embed + self.pose_level_embed[lvl].view(1, 1, -1)
                lvl_pos_embed_pose_flatten.append(lvl_pos_embed_pose)

            if srcs_critique is not None:
                src_critique = srcs_critique[lvl]
                src_critique = src_critique.flatten(2).transpose(1, 2)
                src_critique_flatten.append(src_critique)                
                lvl_pos_embed_critique = pos_obj_embed + self.critique_level_embed[lvl].view(1, 1, -1)
                lvl_pos_embed_critique_flatten.append(lvl_pos_embed_critique)

            
        src_flatten = torch.cat(src_flatten, 1)            
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder det
        memory = self.det_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        memory_obj = None
        if srcs_obj is not None and self.with_obj:
            src_obj_flatten = torch.cat(src_obj_flatten, 1)
            lvl_pos_embed_obj_flatten = torch.cat(lvl_pos_embed_obj_flatten, 1)
            memory_obj = self.obj_encoder(src_obj_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_obj_flatten, mask_flatten)
        
        if srcs_pose is not None and self.with_pose:
            src_pose_flatten = torch.cat(src_pose_flatten, 1)
            lvl_pos_embed_pose_flatten = torch.cat(lvl_pos_embed_pose_flatten, 1)
            memory_pose = self.pose_encoder(src_pose_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_pose_flatten, mask_flatten)
        
        if srcs_critique is not None and self.with_critique:
            src_critique_flatten = torch.cat(src_critique_flatten, 1)
            lvl_pos_embed_critique_flatten = torch.cat(lvl_pos_embed_critique_flatten, 1)
            memory_critique = self.critique_encoder(src_critique_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_critique_flatten, mask_flatten)
        
        det_query_embed = self.det_query_embed.weight
        #grasp_query_embed = self.grasp_query_embed.weight

        # prepare input for decoder
        B, _, C = memory.shape
        if len(det_query_embed.shape) == 2:
            det_query_embed, det_tgt = torch.split(det_query_embed, C, dim=1)
            det_query_embed = det_query_embed.unsqueeze(0).expand(B, -1, -1)
            det_tgt = det_tgt.unsqueeze(0).expand(B, -1, -1)
        else:
            det_query_embed, det_tgt = torch.split(det_query_embed, C, dim=2)

        det_reference_points = self.det_reference_points(det_query_embed).sigmoid()
        det_reference_points = det_reference_points.clamp(1e-4, 1 - 1e-4)
        
        init_reference_out = det_reference_points

        # decoder
        det_hs, det_inter_references = self.det_decoder(
            det_tgt,  # decoding queries
            det_reference_points, 
            memory, # encoder embeddings
            spatial_shapes, 
            level_start_index, 
            valid_ratios, 
            det_query_embed,  # positional encoding
            mask_flatten
            )

        #print('det', det_hs.shape, det_inter_references.shape, det_tgt.shape, memory.shape, det_query_embed.shape)

        outputs_logits, outputs_coords, outputs_cls_tokens = [], [], []

        for lvl in range(det_hs.shape[0]):
            reference = det_inter_references[lvl - 1].clamp(1e-4, 1 - 1e-4)
            reference = inverse_sigmoid(reference)

            logits = self.det_class_embed[lvl](det_hs[lvl])
            tmp = self.det_bbox_embed[lvl](det_hs[lvl])
            cls_token = self.det_cls_token[lvl](det_hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                tmp[..., :2] += reference

            if lvl > 0:
                cls_token = outputs_cls_tokens[-1] + cls_token.mul(self.det_delta_cls_gain[lvl-1])

            coord = tmp.sigmoid()  # no clamp here
            outputs_coords.append(coord)
            outputs_logits.append(logits)
            outputs_cls_tokens.append(cls_token)
      
        outputs_logits = torch.stack(outputs_logits)          # [L,B,Q,C]
        outputs_coords  = torch.stack(outputs_coords)           # [L,B,Q,4]
        outputs_cls_tokens = torch.stack(outputs_cls_tokens)     # [L,B,Q,dEnc]

        #print(outputs_conf.shape)
        #outputs_classes = outputs_conf.max(dim=-1).indices

        outs = {
            'det_logits': outputs_logits,
            'det_coord': outputs_coords,
            'det_cls_token': outputs_cls_tokens
        }

        hs_out = {
            'det_hs': det_hs,
            'det_inter_references': det_inter_references
        }

        outs_shared = {
            'det_hidden': det_hs[-1],
        }

        
        # Shapes
        B = det_hs.shape[1]                 # batch
        Q = det_hs.shape[2]                 # det queries
        C = det_hs.shape[3]                 # channel dim det
        K = self.max_objects                # max_objects

        outputs_confs = F.softmax(outputs_logits, dim=-1)     # softmax over classes
        conf_last = outputs_confs[-1]          # [B,Q,C]
        C_cls = conf_last.shape[-1]
        if C_cls == 2:
            objness = conf_last[:, :, 1]      # prob(object)
        else:
            objness = conf_last[:, :, 1:].max(dim=-1).values  # best foreground prob

        index = objness.topk(K, dim=-1).indices  # [B,K]
        outs_shared['obj_index'] = index  # [B, K]

        if not self.with_obj and not self.with_pose and not self.with_critique:
            return hs_out, outs, outs_shared


        if self.with_obj or self.with_pose:

            #print('B, Q, C, K:', B, Q, C, K)
            # Final layer detection features & refs

            det_hs_last = det_hs[-1].detach()            # [B, Q, C]
            det_ref_last = det_inter_references[-1].detach()  # [B, Q, 4]
            det_qpos_last = det_query_embed.detach()      # [B, Q, C] (pos encoding from queries)
            #print('det_hs_last', det_hs_last.shape)
            #print('det_ref_last', det_ref_last.shape)
            #print('det_qpos_last', det_qpos_last.shape)

            # Batched gather with index [B,K]

            idx_exp_C = index.unsqueeze(-1).expand(B, K, C)
            idx_exp_4 = index.unsqueeze(-1).expand(B, K, det_ref_last.size(-1))
            #print('idx_exp_C', idx_exp_C.shape)
            #print('idx_exp_4', idx_exp_4.shape)

            selected_hs   = torch.gather(det_hs_last,   dim=1, index=idx_exp_C)  # [B, K, C]
            selected_ref  = torch.gather(det_ref_last,  dim=1, index=idx_exp_4)  # [B, K, 4]
            selected_qpos = torch.gather(det_qpos_last, dim=1, index=idx_exp_C)  # [B, K, C]           
            
            #print('selected_hs', selected_hs.shape)
            #print('selected_ref', selected_ref.shape)
            #print('selected_qpos', selected_qpos.shape)

        C = self.d_model_obj

        if self.with_obj:           
            # projection det to obj dim

            selected_hs = self.obj_logits_proj(selected_hs)
            selected_qpos = self.obj_query_proj(selected_qpos)
            det_hs_slot   = selected_hs.unsqueeze(2)           # [B, K, 1, C]
            det_ref_slot  = selected_ref.unsqueeze(2)          # [B, K, 1, 4]
            det_qpos_slot = selected_qpos.unsqueeze(2)         # [B, K, 1, C]  # positinal encoding during decoding
            
            #print('selected_hs', selected_hs.shape)
            #print('selected_ref', selected_ref.shape)
            #print('selected_qpos', selected_qpos.shape)

            # get object query embeddings
            obj_query_embed = self.obj_query_embed.weight # [Qobj, 2C]
            #print('obj_query_embed weight', obj_query_embed.shape)       
            obj_query_embed = obj_query_embed.unsqueeze(0).unsqueeze(0).expand(B, K, self.num_obj_queries, 2*C)  # [B, K, Qobj, 2C]
            
            obj_qpos, obj_tgt = torch.split(obj_query_embed, C, dim=-1) #  [B, K, Qobj, C],  [B, K, Qobj, C]
            
            # Prepare object queries replicated per detection
            det_hs_rep = selected_hs.unsqueeze(2).expand(B, K, self.num_obj_queries, C)            # [B, K, Qobj, C]            
            det_hs_qpos = selected_qpos.unsqueeze(2).expand(B, K, self.num_obj_queries, C)        

            obj_tgt = obj_tgt+det_hs_rep                       
            obj_reference_points_deltas = 0.05 * torch.tanh(self.obj_reference_points(obj_qpos+det_hs_qpos))  # [B, K, Qobj, 4]
            obj_ref = (det_ref_slot + obj_reference_points_deltas).clamp(1e-4, 1-1e-4)
            obj_qpos = obj_qpos + det_hs_qpos   

    
            
            #print('det_hs_slot', det_hs_slot.shape)
            #print('det_ref_slot', det_ref_slot.shape)
            #print('det_qpos_slot', det_qpos_slot.shape)        
            #print('obj_tgt_rep', obj_tgt_rep.shape)
            #print('obj_reference_points_deltas', obj_reference_points_deltas.shape)
            #print('obj_qpos_rep', obj_qpos_rep.shape)

            obj_tgt_cat = torch.cat([det_hs_slot, obj_tgt], dim=2)                                                # [B, K, 1+Qobj, C]
            obj_ref_cat = torch.cat([det_ref_slot, obj_ref], dim=2)     # [B, K, 1+Qobj, 4]
            obj_qpos_cat = torch.cat([det_qpos_slot, obj_qpos], dim=2)                                                       # [B, K, 1+Qobj, C]
            
            #print('obj_tgt_cat', obj_tgt_cat.shape)
            #print('obj_ref_cat', obj_ref_cat.shape)
            #print('obj_qpos_cat', obj_qpos_cat.shape)

            # Flatten (objects * queries-per-object) for the decoder

            N_total = (1 + self.num_obj_queries) * K
            obj_tgt_flat = obj_tgt_cat.view(B, N_total, C).contiguous()
            obj_ref_flat = obj_ref_cat.view(B, N_total, obj_ref_cat.shape[-1]).contiguous()
            obj_qpos_flat = obj_qpos_cat.view(B, N_total, C).contiguous()
            
            #print('obj_tgt_flat', obj_tgt_flat.shape)
            #print('obj_ref_flat', obj_ref_flat.shape)
            #print('obj_qpos_flat', obj_qpos_flat.shape)

            # Decode object properties
            obj_hs, obj_inter_references = self.obj_decoder(
                obj_tgt_flat,           # [B, N_total, C]
                obj_ref_flat,           # [B, N_total, 5]
                memory_obj,
                spatial_shapes,
                level_start_index,
                valid_ratios,
                obj_qpos_flat,
                mask_flatten
            )

            # Reshape back to [L, B, K, (1+Qobj), C] and refs to [..., 4]
            L = obj_hs.shape[0]
            obj_hs = obj_hs.view(L, B, K, (1 + self.num_obj_queries), C).contiguous()
            obj_inter_references = obj_inter_references.view(L, B, K, (1 + self.num_obj_queries), obj_inter_references.shape[-1]).contiguous()

            # Accumulate heads per layer (like before)
            outputs_points, outputs_scale, output_positions, output_vis, output_vol = [], [], [], [], []

            for lvl in range(obj_hs.shape[0]):
                # index 1.. to use object queries (0 is detection slot)
                obj_pos = self.obj_pos_head[lvl](obj_hs[lvl][:, :, 1])          # [B, K, 3]
                obj_vis = self.obj_vis_head[lvl](obj_hs[lvl][:, :, 2])          # [B, K, 1]
                obj_scale = self.obj_scale_head[lvl](obj_hs[lvl][:, :, 3])      # [B, K, 1]
                #...
                obj_vol = self.obj_vol_head[lvl](obj_hs[lvl][:, :, 7])          # [B, K, 1]


                # combining the xyz points into a point cloud
                fused_xyz_pc = torch.cat([obj_hs[lvl][:, :, 4], obj_hs[lvl][:, :, 5], obj_hs[lvl][:, :, 6]], dim=-1)
                #obj_points = self.obj_shape_head[lvl](fused_xyz_pc).view(B, K, self.shape_dims, 3)
                delta = self.obj_shape_head[lvl](fused_xyz_pc).view(B, K, self.shape_dims, 3)            

                # iterative refinement of the predictions
                if lvl > 0: # refining the initial guess, we down scale the outputs to make the decoding esier for the model to produce small numbers
                    obj_pos = obj_pos.mul(self.obj_delta_pos_gain[lvl - 1]) + output_positions[-1]
                    obj_vis = obj_vis.mul(self.obj_delta_vis_gain[lvl - 1]) + output_vis[-1]
                    obj_scale = obj_scale.mul(self.obj_delta_scale_gain[lvl - 1]) + outputs_scale[-1]
                    obj_vol = obj_vol.mul(self.obj_delta_vol_gain[lvl - 1])  + output_vol[-1]                 
                    obj_points = outputs_points[-1] + delta.mul(self.obj_delta_shape_gain[lvl])
                else:
                    obj_points = self.unit_sphere.to(delta.dtype).to(delta.device)          # [D,3]
                    obj_points = obj_points.unsqueeze(0).unsqueeze(0).expand(B, K, -1, -1)        # [B,K,D,3]
                    obj_points = obj_points + delta.mul(self.obj_delta_shape_gain[lvl])
                    
                output_positions.append(obj_pos)
                output_vis.append(obj_vis)
                outputs_scale.append(obj_scale)
                outputs_points.append(obj_points)
                output_vol.append(obj_vol)


            # Stack across layers

            output_positions = torch.stack(output_positions)  # [L, B, K, 3]
            output_vis      = torch.stack(output_vis)         # [L, B, K, 1]
            outputs_scale    = torch.stack(outputs_scale)     # [L, B, K, 1]
            outputs_points   = torch.stack(outputs_points)    # [L, B, K, D, 3]
            output_vol   = torch.stack(output_vol)            # [L, B, K, D, 1]
                    
            #print('output_positions', output_positions.shape)
            #print('output_vis', output_vis.shape)
            #print('outputs_scale', outputs_scale.shape)
            #print('outputs_points', outputs_points.shape)
            #print('output_vol', output_vol.shape)
        
            outs.update({
                'obj_positions': output_positions,      # [L,B,K,3]
                'obj_vis': output_vis,                # [L,B,K,3]
                'obj_scale': outputs_scale,             # [L,B,K,1]
                'obj_points': outputs_points,           # [L,B,K,D,3]
                'obj_vol': output_vol                   # [L,B,K,1]
            })

            hs_out.update({
                'obj_hs': obj_hs,
                'obj_inter_references': obj_inter_references
            })
            outs_shared['obj_hidden'] = obj_hs[-1][:, :, 1:]


        if self.with_pose:           
            if self.with_obj:
                # Object-context (already in obj space and with references)
                ctx_tgt   = obj_hs[-1].detach() #.view(B, -1, C)               # [B, N_ctx, C]
                ctx_ref  = obj_inter_references[-1].detach() #.view(B, -1, 4) # [B, N_ctx, 4]
                ctx_qpos = obj_qpos_cat.detach() #.view(B, -1, C)             # [B, N_ctx, C]
            
            else:
                # Projected detection context (pose side)
                ctx_tgt   = self.pose_logits_proj(selected_hs.detach())  # [B,K,C]
                ctx_qpos = self.pose_qpos_proj(selected_qpos.detach())  # [B,K,C]
                ctx_ref  = selected_ref.detach()                             # [B,K,4]

                ctx_tgt = ctx_tgt.unsqueeze(2).expand(B, K, self.num_object_tokens_per_det, C).contiguous() #.view(B, self.num_extra_query_embed, C)
                ctx_qpos = ctx_qpos.unsqueeze(2).expand(B, K, self.num_object_tokens_per_det, C).contiguous() #.view(B, self.num_extra_query_embed, C)
                ctx_ref = ctx_ref.unsqueeze(2).expand(B, K, self.num_object_tokens_per_det, 4).contiguous() #.view(B, self.num_extra_query_embed, 4)


            if self.add_pose_extra_query_embed:
                extra_query = self.pose_extra_query_embed.weight            # [N_ctx, 2C]
                extra_query = extra_query.unsqueeze(0).expand(B, self.num_extra_query_embed, 2*C).view(
                    B, K, self.num_object_tokens_per_det, 2*C) # [B, K, 1+Qobj, 2C]
                extra_qpos, extra_tgt = torch.split(extra_query, C, dim=-1)  # [B, K, 1+Qobj,C] each
                
                extra_ref_d = 0.05 * torch.tanh(self.pose_extra_reference_points(extra_qpos + ctx_qpos)) #[B,K, 1+Qobj,4]   
                ctx_tgt = ctx_tgt + extra_tgt #[B,N_ctx,C]
                ctx_qpos = ctx_qpos + extra_qpos #[B,N_ctx,C]
                ctx_ref = (ctx_ref + extra_ref_d).clamp(1e-4, 1-1e-4) #[B,N_ctx,4]
                       

            # get object query embeddings
            pose_query_embed = self.pose_query_embed.weight # [Qobj, 2C]   
            _dbg_print_nonfinite("pose_query_embed", pose_query_embed)
            pose_query_embed = pose_query_embed.unsqueeze(0).unsqueeze(0).expand(
                B, K, self.num_poses, 2*C)#.contiguous().view(B, K*self.num_poses, 2*C) # [B, K*Qposes, 2C]
            
            #print('pose_query_embed', pose_query_embed.shape)
            pose_qpos, pose_tgt = torch.split(pose_query_embed, C, dim=-1) # [B, K, Qposes, C], [B, K, Qposes, C]

            pose_tgt_flat = (pose_tgt + ctx_tgt[:, :, 0].unsqueeze(2)).contiguous().view(B, -1, C) # [B, K*Qposes, C] 
            pose_qpos_flat = (pose_qpos + ctx_qpos[:, :, 0].unsqueeze(2)).contiguous().view(B, -1, C) # [B, K*Qposes, C] 
            pose_reference_points_deltas = 0.05 * torch.tanh(self.pose_reference_points(pose_qpos_flat)) # [B, K*Qposes, 4] 
            _dbg_print_nonfinite("pose_reference_points_deltas", pose_reference_points_deltas)
            #print('pose_reference_points_deltas', pose_reference_points_deltas.shape)
            ref = ctx_ref[:, :, 0].unsqueeze(2).expand(B, K, self.num_poses, 4).contiguous().view(B, -1, 4)
            #print('ref', ref.shape)
            pose_ref_flat = (pose_reference_points_deltas + ref).clamp(1e-4, 1-1e-4) # [B, K*Qposes, 4] 
            Qposes = pose_tgt_flat.shape[1]           

            ctx_tgt = ctx_tgt.contiguous().view(B, -1, C)
            ctx_ref = ctx_ref.contiguous().view(B, -1, 4)
            ctx_qpos = ctx_qpos.contiguous().view(B, -1, C)
            
            
            _dbg_print_nonfinite("pose_tgt_flat", pose_tgt_flat)
            _dbg_print_nonfinite("ctx_tgt", ctx_tgt)
            _dbg_print_nonfinite("pose_ref_flat", pose_ref_flat)
            _dbg_print_nonfinite("ctx_ref", ctx_ref)
            _dbg_print_nonfinite("pose_qpos_flat", pose_qpos_flat)
            _dbg_print_nonfinite("ctx_qpos", ctx_qpos)

            pose_tgt_flat        = torch.cat([pose_tgt_flat, ctx_tgt], dim=1)
            pose_ref_flat        = torch.cat([pose_ref_flat, ctx_ref], dim=1)
            pose_qpos_flat        = torch.cat([pose_qpos_flat, ctx_qpos], dim=1)
                           

            if self.add_model_embed:
                inst_id_2_slot_id = []
                model_embed = torch.zeros(B, self.max_objects, C, device=pose_tgt_flat.device, dtype=pose_tgt_flat.dtype)
                #print('model_embed', model_embed.shape)
                #print('pose_tgt_flat', pose_tgt_flat.shape)
                #print('BMC', B, self.max_objects, C)
                for b in range(B):
                    t_b = targets[b]
                    models_b = t_b.get('models', {}) or {}
                    inst2slot, model_v, obj_ids = {}, {}, []
                    slot_ts = list(range(self.max_objects))
                    if self.training:
                        random.shuffle(slot_ts)

                    for inst_id in sorted(models_b.keys()):
                        obj_id = t_b['poses'][inst_id]['obj_id']
                        if obj_id not in obj_ids:
                            obj_ids.append(obj_id)
                        obj_idx = obj_ids.index(obj_id)
                        if obj_idx >= self.max_objects:
                            continue
                        if obj_idx not in model_v:
                            pc = models_b[inst_id].to(pose_tgt_flat.device, dtype=pose_tgt_flat.dtype)
                            model_v[obj_idx] = self.pose_model_embed(pc.unsqueeze(0)).squeeze(0)  # (C,)
                            #print('model pc', pc.shape, model_v[obj_idx].shape, obj_idx, inst_id)

                        slot_idx = slot_ts[obj_idx]
                        model_embed[b, slot_idx] = model_v[obj_idx]
                        inst2slot[int(inst_id)] = slot_idx

                    inst_id_2_slot_id.append(inst2slot)

                outs_shared['inst_id_2_slot_id'] = inst_id_2_slot_id  # list[dict{inst_id->slot_id}]
                #print('inst_id_2_slot_id', inst_id_2_slot_id)
                outs_shared['pose_mode_is_object_6d'] = True
                
                # 3) Build model query embeddings for exactly max_objects tokens
                base = self.pose_model_query_embed.weight  # (max_objects, 2C)
                qpos_base, tgt_base = torch.split(base, C, dim=1)  # (max_objects,C)

                qpos_model = qpos_base.unsqueeze(0).expand(B, -1, -1)      # (B,max_objects,C)
                tgt_model  = tgt_base.unsqueeze(0).expand(B, -1, -1)       # (B,max_objects,C)

                qpos_model = qpos_model + self.pose_model_query_2_qpos(model_embed)  # (B,max_objects,C)
                ref_model  = self.pose_model_reference_points(qpos_model)            # (B,max_objects,4)
                tgt_model  = tgt_model + model_embed                                  # (B,max_objects,C)

                # place before context and the num_poses slots
                pose_tgt_flat = torch.cat([pose_tgt_flat, tgt_model],        dim=1)
                pose_ref_flat = torch.cat([pose_ref_flat, ref_model],        dim=1)
                pose_qpos_flat = torch.cat([pose_qpos_flat, qpos_model],     dim=1)

            else:
                outs_shared['pose_mode_is_object_6d'] = False
                outs_shared['inst_id_2_slot_id'] = None
            
                                  
            _dbg_print_nonfinite("tgt_in", pose_tgt_flat)
            _dbg_print_nonfinite("pose_ref_flat", pose_ref_flat)
            _dbg_print_nonfinite("memory_pose", memory_pose)
            _dbg_print_nonfinite("spatial_shapes", spatial_shapes)
            _dbg_print_nonfinite("pose_qpos_flat", pose_qpos_flat)

            pose_hs, _ = self.pose_decoder(
                pose_tgt_flat, 
                pose_ref_flat, 
                memory_pose, 
                spatial_shapes,
                level_start_index, 
                valid_ratios, 
                pose_qpos_flat,
                mask_flatten
            )
            _dbg_print_nonfinite("pose_hs", pose_hs)
            
            # Heads (only the last num_poses slots per object)
            L = pose_hs.shape[0]
            pose_pos_stack, pose_rot_stack, pose_conf_stack, pose_slot_stack = [], [], [], []
            for lvl in range(L):
                pos = self.pose_translation_head[lvl](pose_hs[lvl][:, :Qposes])  # [B,Qposes,3]
                rot = self.pose_orientation_head[lvl](pose_hs[lvl][:, :Qposes])  # [B,Qposes,4]
                conf = self.pose_confidence_head[lvl](torch.cat([pose_hs[lvl][:, :Qposes]], dim=-1))  # [B,Qposes,1]

                # if add_model_embed : index at K addresses object encoded at model_index_mapping[idx] which is the inst id 
                pos = pos.view(B, K, self.num_poses, 3) # [B,K, self.num_poses, 3]
                rot = rot.view(B, K, self.num_poses, 4) # [B,K, self.num_poses, 4]
                conf = conf.view(B, K, self.num_poses, 1) # [B,K, self.num_poses, 1]
                if lvl > 0:
                    pos  = pos.mul(self.pose_translation_delta_gain[lvl-1]) + pose_pos_stack[-1]
                    conf = conf.mul(self.pose_confidence_delta_gain[lvl-1]) + pose_conf_stack[-1]
                    
                    #pos = 0.1 * self.pose_translation_delta_gain[lvl-1] + pose_pos_stack[-1]
                    #rot = rot.mul(self.pose_orientation_delta_gain[lvl-1]) + pose_rot_stack[-1]
                    #conf = 0.1 * self.pose_confidence_delta_gain[lvl-1] + pose_conf_stack[-1]

                # refinement instead of: rot = rot.mul(gain) + prev

                #if is_main_process() and lvl == L-1:
                #    print('torch mean', float(rot.detach().mean().item()), float(rot.detach().max().item()), float(rot.detach().min().item()))
            
                #delta_raw = self.pose_orientation_head[lvl](pose_hs[lvl][:, :Qposes])       # [B,Q,4]
                rot = rot.tanh()                                                # [-1,1] bounded
                rot   = _normalize_quat(rot)
                prev_q    = pose_rot_stack[-1] if lvl > 0 else torch.tensor([1,0,0,0], device=rot.device).view(1,1,1,4).expand_as(rot)
                rot = quat_mul(prev_q, rot)
                rot = _normalize_quat(rot)

                pose_pos_stack.append(pos)
                pose_rot_stack.append(rot)
                pose_conf_stack.append(conf)

                if self.add_model_embed:
                    slot_logits = self.pose_slot_head[lvl](pose_hs[lvl][:, :Qposes])  # [B,Qposes,M]
                    slot_logits = slot_logits.view(B, K, self.num_poses, self.max_objects+1)  # [B,K,Np,M]
                    if lvl > 0:
                        slot_logits = slot_logits.mul(self.pose_slot_delta_gain[lvl-1]) + pose_slot_stack[-1]
                    pose_slot_stack.append(slot_logits)


            pose_pos_stack = torch.stack(pose_pos_stack, dim=0)     # [L,B,K,num_poses,3]
            pose_rot_stack = torch.stack(pose_rot_stack, dim=0)     # [L,B,K,num_poses,4]
            #if is_main_process():
            #    print('')
            #    print('pos', float(pose_pos_stack.detach().mean().item()), float(pose_pos_stack.detach().max().item()), float(pose_pos_stack.detach().min().item()))
            #if is_main_process():
            #    print('rot', float(pose_rot_stack.detach().mean().item()), float(pose_rot_stack.detach().max().item()), float(pose_rot_stack.detach().min().item()))
            
            pose_rot_stack = _normalize_quat(pose_rot_stack)     # [L,B,K,num_poses,4]
            #pose_rot_stack = torch.tanh(torch.stack(pose_rot_stack, dim=0))     # [L,B,K,num_poses,4]
            
            pose_conf_stack = torch.stack(pose_conf_stack, dim=0)     # [L,B,K,num_poses,1]
            #if is_main_process():
            #    print('conf', float(pose_conf_stack.detach().mean().item()), float(pose_conf_stack.detach().max().item()), float(pose_conf_stack.detach().min().item()))
            
            outs['pose_preds'] = torch.cat([pose_pos_stack, pose_rot_stack, pose_conf_stack], dim=-1)  # [L,B,K,num_poses,8]
            if self.add_model_embed:
                outs['pose_slot_logits'] = torch.stack(pose_slot_stack, dim=0)  

            pose_hidden_last = pose_hs[-1][:, :Qposes, :]  # [B, K*num_poses, C_obj]
            outs_shared['pose_hidden'] = pose_hidden_last.view(B, K, self.num_poses, C)
              

        if not self.with_critique:
            return hs_out, outs, outs_shared

        NCrit = self.num_critiques
        # Build base critique queries from embedding
        crit_query_w = self.critique_query_embed.weight            # [NCrit, 2C]
        crit_qpos_w, crit_tgt_w = torch.split(crit_query_w, C, dim=1)
        crit_ref  = self.critique_reference_points(crit_qpos_w).sigmoid()  # [NCrit,2]

        crit_tgt_flat  = crit_tgt_w.unsqueeze(0).expand(B, -1, -1)   # [B,NCrit,C]
        crit_ref_flat  = crit_ref.unsqueeze(0).expand(B, -1, -1)   # [B,NCrit,C]
        crit_qpos_flat = crit_qpos_w.unsqueeze(0).expand(B, -1, -1)  # [B,NCrit,C]
            

        # Optional object context

        if self.with_obj:
            # adding object context = ctx
            ctx_tgt = obj_hs[-1].view(B, -1, C).detach()           # [B,K*(1+Qobj),C]
            ctx_ref = obj_inter_references[-1].view(B, -1, 4).detach()[..., :2]  # [B,K*(1+Qobj),2]
            ctx_qpos= obj_qpos_flat.view(B, -1, C).detach()        # [B,K*(1+Qobj),C]
            critique_tgt_flat = torch.cat([ctx_tgt, crit_tgt_flat], dim=1)
            critique_ref_flat = torch.cat([ctx_ref, crit_ref_flat], dim=1)
            critique_qpos_flat= torch.cat([ctx_qpos, crit_qpos_flat], dim=1)
            N_context = ctx_tgt.shape[1]
        elif self.add_critique_extra_query_embed:
            extra_crit_query_w = self.critique_extra_query_embed.weight            # [NCrit, 2C]
            extra_crit_qpos_w, extra_crit_tgt_w = torch.split(extra_crit_query_w, C, dim=1)
            extra_crit_ref  = self.critique_extra_reference_points(extra_crit_qpos_w).sigmoid()  # [NCrit,2]

            extra_crit_tgt_flat  = extra_crit_tgt_w.unsqueeze(0).expand(B, -1, -1)   # [B,NCrit,C]
            extra_crit_ref_flat  = extra_crit_ref.unsqueeze(0).expand(B, -1, -1)   # [B,NCrit,C]
            extra_crit_qpos_flat = extra_crit_qpos_w.unsqueeze(0).expand(B, -1, -1)  # [B,NCrit,C]
            
            critique_tgt_flat = torch.cat([extra_crit_tgt_flat, crit_tgt_flat], dim=1)
            critique_ref_flat = torch.cat([extra_crit_ref_flat, crit_ref_flat], dim=1)
            critique_qpos_flat= torch.cat([extra_crit_qpos_flat, crit_qpos_flat], dim=1)
            N_context = extra_crit_qpos_flat.shape[1]
        else:
            critique_tgt_flat, critique_ref_flat, critique_qpos_flat = crit_tgt_flat, crit_ref_flat, crit_qpos_flat
            N_context = 0


        if outs.get('pose_preds') is not None:
            NCrit = self.num_critiques
            outputs_critiques = []
            outputs_pred_masks = []

            Lp, Bp, Kp, Np, _ = outs['pose_preds'].shape
            # last layer slot logits: [B,K,Np,M]
            pose_slot_logits = outs.get('pose_slot_logits', None)
            inst_id_2_slot_id = outs_shared.get('inst_id_2_slot_id', None)
            model_embed_per_image = None

            # Build unique model embedding and store them into the slot index allocated by the previous inst id and the inst2slot mapping
            if self.add_model_embed:
                model_embed_per_image = torch.zeros(B, self.max_objects+1, C, device=critique_tgt_flat.device, dtype=critique_tgt_flat.dtype)
                for b in range(B):
                    t_b = targets[b] if targets is not None else {}
                    models_b = t_b.get('models', {}) or {}
                    inst2slot = inst_id_2_slot_id[b]
                    # obj_id -> embedding (first occurrence)
                    for inst_id, pc in models_b.items():                        
                        if  (pc is None) or (pc.numel() == 0):
                            continue
                        if inst_id >= self.max_objects:
                            continue
                        slot_idx = inst2slot[inst_id]
                        model_embed_per_image[b, slot_idx] = self.critique_model_embed(pc.unsqueeze(0)).squeeze(0)  # (C,)

            for lvl_crit in range(Lp):
                poses_lvl = outs['pose_preds'][lvl_crit]  # [B,K,Np,8]
                #print('poses_lvl', poses_lvl.shape)
                poses_b   = poses_lvl.reshape(B, -1, 8)[:, :, :7]  # [B,K*Np,7]
                #print('poses_lvl2', poses_b.shape)
                
                model_embed_flat = torch.zeros(B, poses_b.shape[1], C, device=poses_b.device, dtype=poses_b.dtype)
                if self.add_model_embed:
                    slot_logits = pose_slot_logits[lvl_crit]  # [B,K,Np,M]
                    slot_pred = slot_logits.argmax(dim=-1)  # [B,K,Np] in [0..M-1]
                    slot_pred_flat = slot_pred.reshape(B, -1)  # [B,K*Np]
                    # gather model embed per predicted slot
                    # model_embed_per_image: [B,M,C]
                    for b in range(B):
                        if model_embed_per_image is None:
                            continue                        
                        model_embed_flat[b] = model_embed_per_image[b].index_select(0, slot_pred_flat[b]).contiguous()

                # chunk
                pred_chunks, valid_masks, model_chunks = chunk_predicted_poses_with_ids(
                    poses_bxN7=poses_b,
                    model_embed_flat=model_embed_flat,
                    num_critiques=NCrit, max_chunks=3
                )
                # predicted slot indices (last layer)
                
                # build masks and run critique
                pred_mask_lvl = torch.cat(valid_masks, dim=-1)    # [B,total_NCrit]
                outputs_pred_masks.append(pred_mask_lvl)

                chunk_logits_final = []
                for chunk_idx, poses_chunk in enumerate(pred_chunks):  # [B,NCrit,7]
                    pose_embed = self.critique_tokenzizer(poses_chunk)         # [B,NCrit,C]
                    model_embed_chunk = model_chunks[chunk_idx]
                    #print('criti, pose_preds', chunk_idx, pose_embed.shape, float(model_embed_chunk.detach().mean().item()))
                  
                    tgt_in = torch.cat(
                        [
                            critique_tgt_flat[:, :N_context],                             # [B, N_ctx, C]
                            critique_tgt_flat[:, N_context:] + pose_embed + model_embed_chunk  # [B, NCrit, C]
                        ],
                        dim=1
                    )

                    critique_hs, _ = self.critique_decoder(
                        tgt_in,
                        critique_ref_flat,
                        memory_critique,
                        spatial_shapes,
                        level_start_index,
                        valid_ratios,
                        critique_qpos_flat,
                        mask_flatten
                    )
                    # last layer logits
                    logits_stack = []
                    for l in range(critique_hs.shape[0]):
                        logits = self.critique_confidence_head[l](critique_hs[l][:, N_context:]) # [B,NCrit,2]
                        if l > 0:
                            logits = logits_stack[-1] + logits.mul(self.critique_delta_gain[l-1])
                        logits_stack.append(logits)
                    chunk_logits_final.append(logits_stack[-1])

                outputs_critiques.append(torch.cat(chunk_logits_final, dim=1))

            # Stack levels
            outputs_critiques = torch.stack(outputs_critiques, dim=0)   # [L,B,total_NCrit,2]
            outputs_pred_masks = torch.stack(outputs_pred_masks, dim=0) # [L,B,total_NCrit] (bool)
            outs.update({
                'critique_pred_logits': outputs_critiques,
                'critique_pred_mask':   outputs_pred_masks,
            })
        
        if targets is not None and not (self.with_pose and self.training):
            # Build per-image dicts (inst_id -> list[grasp dict]) from targets
            #print('running crits')
            if self.add_model_embed:
                good_by_img = [t.get('pose_6d_proposals_good', {}) for t in targets]
                bad_by_img  = [t.get('pose_6d_proposals_bad', {}) for t in targets]
            else:
                good_by_img = [t.get('gt_grasps', {}) for t in targets]
                bad_by_img  = [t.get('bad_grasps', {}) for t in targets]

            # Exactly max_chunks chunks; per-chunk balanced across inst_id; 34% good, 33% bad, rest random
            poses_chunks, pose_masks, pose_scores, inst_ids_chunks = chunk_and_pad_poses_mixed_balanced(
                good_grasps_by_img=good_by_img,
                bad_grasps_by_img=bad_by_img,
                num_critiques=NCrit,
                max_chunks=3,
                good_frac_max=0.34,
                bad_frac_max=0.33,
                device=critique_tgt_flat.device,
                dtype=critique_tgt_flat.dtype
            )
            pose_mask_total = torch.cat(pose_masks, dim=-1)  # [B, total_NCrit]
            pose_score_total = torch.cat(pose_scores, dim=-1) # [B, total_NCrit]
            pose_inst_total  = torch.cat(inst_ids_chunks, dim=-1)  # [B, total_NCrit] (long)

            #print('pose_mask_total', pose_mask_total.shape, torch.unique(pose_mask_total, return_counts=True))
            total_NCrit = pose_mask_total.shape[-1]
            
            # encod the model point clouds and add to each pose query addressing that pose by inst_id
            model_vec_maps = None
            if self.add_model_embed:
                model_vec_maps = []
                for b in range(B):
                    t_b = targets[b]
                    inst2vec = {}
                    models_b = t_b.get('models', {}) or {}
                    # models_b: {inst_id: Tensor [Ni,3] in meters}
                    for inst_id, pc in models_b.items():
                        pc = pc.to(critique_tgt_flat.device, dtype=critique_tgt_flat.dtype)
                        v = self.critique_model_embed(pc.unsqueeze(0))  # (1,C)
                        inst2vec[int(inst_id)] = v.squeeze(0)            # (C,)
                    model_vec_maps.append(inst2vec)

            crit_logits_layers = []
            for chunk_id, poses_chunk in enumerate(poses_chunks):  # [B,NCrit,7]
                
                        
                pose_embed = self.critique_tokenzizer(poses_chunk)  # [B,NCrit,C]
                if model_vec_maps is not None:
                    inst_ids_chunk = inst_ids_chunks[chunk_id]  # [B,NCrit] long
                    zero_vec = torch.zeros(self.d_model_obj, device=pose_embed.device, dtype=pose_embed.dtype)
                    # Gather shape vecs per (b,i)
                    model_embed = []
                    for b in range(B):
                        inst2vec = model_vec_maps[b]
                        ids_b = inst_ids_chunk[b]
                        vecs_b = torch.stack([inst2vec.get(int(ids_b[i].item()), zero_vec) for i in range(NCrit)], dim=0)
                        model_embed.append(vecs_b)
                    model_embed = torch.stack(model_embed, dim=0)  # (B,NCrit,C)
                else:
                    model_embed = 0.0

                tgt_in = torch.cat(
                    [
                        critique_tgt_flat[:, :N_context],                              # [B, N_ctx, C]
                        critique_tgt_flat[:, N_context:] + pose_embed + model_embed    # [B, NCrit, C]
                    ],
                    dim=1
                )


                crit_hs, _ = self.critique_decoder(
                    tgt_in,
                    critique_ref_flat,
                    memory_critique,
                    spatial_shapes,
                    level_start_index,
                    valid_ratios,
                    critique_qpos_flat,
                    mask_flatten
                )

                #_dbg_print_nonfinite("tgt_in", tgt_in)
                #_dbg_print_nonfinite("critique_ref_flat", critique_ref_flat)
                #_dbg_print_nonfinite("memory_critique", memory_critique)
                #_dbg_print_nonfinite("critique_qpos_flat", critique_qpos_flat)

                #_dbg_print_nonfinite("crit_hs", crit_hs)
                #_dbg_print_nonfinite("critique_tgt_flat", critique_tgt_flat)
                #_dbg_print_nonfinite("pose_embed", pose_embed)

                #_dbg_print_nonfinite("model_embed", model_embed)

                logits_stack = []
                for lvl in range(crit_hs.shape[0]):
                    logits = self.critique_confidence_head[lvl](crit_hs[lvl][:, N_context:])
                    if lvl > 0:
                        logits = logits_stack[-1] + logits.mul(self.critique_delta_gain[lvl-1])
                    logits_stack.append(logits)  # [B,NCrit,2]
                crit_logits_layers.append(torch.stack(logits_stack, dim=0))  # [L,B,NCrit,2]

            outs['critique_gt_logits'] = torch.cat(crit_logits_layers, dim=2)   # [L,B,total_NCrit,2]
            outs_shared['critique_gt_pose_mask'] = pose_mask_total              # [B,total_NCrit], int-coded 0/1/2
            outs_shared['critique_gt_score'] = pose_score_total                # [B,total_NCrit]  in [0,1


            
        return hs_out, outs, outs_shared


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src



class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.det_bbox_embed = None
        self.det_class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        #print('tgt', tgt.shape, src.shape)

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            #print('lid', lid, reference_points.shape)
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.det_bbox_embed is not None:
                tmp = self.det_bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()

                new_reference_points = new_reference_points.clamp(1e-4, 1 - 1e-4)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model_det=args.hidden_dim_det,
        d_model_obj=args.hidden_dim_obj,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers_obj=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        with_obj=args.with_obj,
        num_det_queries=args.num_det_queries, 
        num_obj_queries=args.num_obj_queries, 
        num_grasp_queries=args.num_grasp_queries, 
        max_objects=args.max_objects,
        with_critique=args.with_critique, 
        with_pose=args.with_pose,
        add_critique_extra_query_embed=args.add_critique_extra_query_embed,
        add_pose_extra_query_embed=args.add_pose_extra_query_embed,
        add_model_embed=args.add_model_embed,
        num_poses=args.num_grasp_poses if not args.add_model_embed else 1,
        num_classes=args.n_classes + 1, # adding the backround class id 0

        use_grasp_diffusion=getattr(
            args,
            "use_grasp_diffusion",
            False,
        ),
        grasp_diffusion_hidden_dim=getattr(
            args,
            "grasp_diffusion_hidden_dim",
            512,
        ),
        grasp_diffusion_layers=getattr(
            args,
            "grasp_diffusion_layers",
            4,
        ),
        grasp_diffusion_heads=getattr(
            args,
            "grasp_diffusion_heads",
            8,
        ),
        grasp_diffusion_train_steps=getattr(
            args,
            "grasp_diffusion_train_steps",
            100,
        ),
        grasp_diffusion_eval_steps=getattr(
            args,
            "grasp_diffusion_eval_steps",
            10,
        ),
        grasp_diffusion_pose_min=getattr(
            args,
            "grasp_diffusion_pose_min",
            [-3.0, -3.0, 0.0, -math.pi, -math.pi, -math.pi],
        ),
        grasp_diffusion_pose_max=getattr(
            args,
            "grasp_diffusion_pose_max",
            [3.0, 3.0, 3.0, math.pi, math.pi, math.pi],
        ),
        grasp_diffusion_confidence_loss_weight=getattr(
            args,
            "grasp_diffusion_confidence_loss_weight",
            1.0,
        )
        )


