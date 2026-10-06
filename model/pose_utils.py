# model/pose_utils.py
import torch
import numpy as np
from scipy.spatial.transform import Rotation as SciRot

def tcp_pose_to_relative(pos, quat_xyzw, ref_pos, ref_quat_xyzw):
    """
    All inputs: array-like, pos in R^3, quats in [x,y,z,w].
    Returns 6D relative pose [p_rel(3), rotvec_rel(3)],
    with translation expressed in the reference EE frame.

    This matches your original function.
    """
    ref_R = SciRot.from_quat(ref_quat_xyzw)
    cur_R = SciRot.from_quat(quat_xyzw)

    pos = np.asarray(pos, dtype=np.float64)
    ref_pos = np.asarray(ref_pos, dtype=np.float64)

    p_rel = ref_R.inv().apply(pos - ref_pos)
    R_rel = ref_R.inv() * cur_R
    rotvec_rel = R_rel.as_rotvec()
    rel_6d = np.concatenate([p_rel, rotvec_rel], axis=0).astype(np.float32)
    return rel_6d


def tcp_pose_from_relative(rel_6d, ref_pos, ref_quat_xyzw):
    """
    rel_6d : [p_rel(3), rotvec_rel(3)], translation in ref EE frame.
    Returns (pos_abs, quat_xyzw_abs).
    """
    ref_R = SciRot.from_quat(ref_quat_xyzw)

    rel_6d = np.asarray(rel_6d, dtype=np.float64)
    p_rel = rel_6d[:3]
    rv_rel = rel_6d[3:6]
    R_rel = SciRot.from_rotvec(rv_rel)

    pos_abs = np.asarray(ref_pos, dtype=np.float64) + ref_R.apply(p_rel)
    R_abs = ref_R * R_rel
    quat_abs_xyzw = R_abs.as_quat().astype(np.float32)
    return pos_abs.astype(np.float32), quat_abs_xyzw

def tcp_pose_to_relative_torch(pos, quat_xyzw, ref_pos, ref_quat_xyzw):
    """
    pos, ref_pos: [...,3]
    quat_xyzw, ref_quat_xyzw: [...,4] in [x,y,z,w]
    Returns rel_6d [...,6] on same device.
    """
    from scipy.spatial.transform import Rotation as SciRot
    # convert per batch element via numpy; keep this for occasional use, not inner loops
    pos_np = pos.detach().cpu().numpy()
    quat_np = quat_xyzw.detach().cpu().numpy()
    ref_pos_np = ref_pos.detach().cpu().numpy()
    ref_quat_np = ref_quat_xyzw.detach().cpu().numpy()
    out = []
    for p, q, pr, qr in zip(pos_np, quat_np, ref_pos_np, ref_quat_np):
        out.append(tcp_pose_to_relative(p, q, pr, qr))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out).to(pos.device, dtype=pos.dtype)