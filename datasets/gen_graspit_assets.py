
# generate_graspit_assets_pc.py

import os
import sys
import glob
import time
import json
import argparse
import numpy as np

# Minimize thread contention for OMP/MKL/OpenBLAS when Open3D runs

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import open3d as o3d


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def write_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def voxel_downsample_to_target(pts: np.ndarray, target: int, max_iter: int = 12, tol: float = 0.10) -> np.ndarray:
    """Downsample via Open3D voxel grid to ~target count using only voxel subsampling."""
    if pts.shape[0] <= target:
        return pts.copy()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    min_b = pcd.get_min_bound()
    max_b = pcd.get_max_bound()
    extents = max_b - min_b
    volume = float(np.prod(extents)) if np.all(extents > 1e-9) else 1.0
    voxel_size = (volume / max(target, 1)) ** (1.0 / 3.0)
    voxel_size = max(voxel_size, 1e-6)
    low, high = voxel_size * 0.25, voxel_size * 4.0

    best = pts
    best_diff = float("inf")
    for _ in range(max_iter):
        ds = pcd.voxel_down_sample(voxel_size)
        cnt = len(ds.points)
        if cnt > 0:
            diff = abs(cnt - target)
            if diff < best_diff:
                best = np.asarray(ds.points, dtype=np.float32)
                best_diff = diff
            if diff / target <= tol:
                best = np.asarray(ds.points, dtype=np.float32)
                break
        if cnt == 0:
            voxel_size *= 0.5
        else:
            ratio = (cnt / float(target)) ** (1.0 / 3.0)
            voxel_size *= ratio
        voxel_size = float(np.clip(voxel_size, low, high))
    return best


def read_first_mesh(obj_dir: str):
    """Find a candidate mesh/pc file inside an object directory."""
    for pat in ("*.obj", "*.ply", "*.stl"):
        g = glob.glob(os.path.join(obj_dir, pat))
        if g:
            return g[0]
    return None


def load_points_from_mesh_or_pc(src_path: str, num_points: int) -> np.ndarray:
    """Read mesh or point cloud and return downsampled points (~num_points)."""
    ext = os.path.splitext(src_path)[1].lower()
    pts_raw = None
    mesh = None
    try:
        if ext in [".obj", ".ply", ".stl"]:
            # Try mesh
            mesh = o3d.io.read_triangle_mesh(src_path)
        if mesh is None or (not mesh.has_vertices() and not mesh.has_triangles()):
            # fallback: point cloud
            pcd = o3d.io.read_point_cloud(src_path)
            if pcd is None or not pcd.has_points():
                return None
            pts_raw = np.asarray(pcd.points, dtype=np.float32)
        else:
            mesh.compute_vertex_normals()
            if mesh.has_triangles():
                # sample a dense set, then voxel downsample
                pcd = mesh.sample_points_poisson_disk(number_of_points=max(num_points * 4, num_points * 2))
                pts_raw = np.asarray(pcd.points, dtype=np.float32)
            else:
                # vertices only
                pts_raw = np.asarray(mesh.vertices, dtype=np.float32)
        if pts_raw is None or pts_raw.shape[0] == 0:
            return None
        pts_ds = voxel_downsample_to_target(pts_raw, num_points)
        return pts_ds
    except Exception as e:
        print(f"[load_points] failed for {src_path}: {e}")
        return None


def save_points_ply(out_pc_path: str, pts: np.ndarray):
    ensure_dir(os.path.dirname(out_pc_path))
    pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    o3d.io.write_point_cloud(out_pc_path, pcd_out, write_ascii=True)


def process_assets(assets_root: str, num_points: int = 768, overwrite: bool = False, dry_run: bool = False):
    """
    For each <obj_id> in assets_root, find a mesh, downsample to ~num_points, and write:
      assets_root_pc_<num_points>/<obj_id>/model_pc<num_points>.ply
    """
    out_root = assets_root + f"_pc_{num_points}"
    ensure_dir(out_root)

    obj_dirs = sorted([d for d in os.listdir(assets_root)
                       if os.path.isdir(os.path.join(assets_root, d)) and d.isdigit()])

    total = len(obj_dirs)
    ok = 0
    failed = 0
    failures = []
    t0 = time.time()

    print(f"Found {total} objects under {assets_root}")
    print(f"Target points: {num_points}")
    print(f"Output: {out_root}")
    print(f"Overwrite: {overwrite} | Dry-run: {dry_run}")

    for i, obj_id in enumerate(obj_dirs, start=1):
        src_dir = os.path.join(assets_root, obj_id)
        out_dir = os.path.join(out_root, obj_id)
        out_pc_path = os.path.join(out_dir, f"model_pc{num_points}.ply")

        # progress
        sys.stdout.write(f"\r[{i}/{total}] obj {obj_id} -> {out_pc_path}")
        sys.stdout.flush()

        # Skip existing unless overwrite
        if os.path.exists(out_pc_path) and not overwrite:
            ok += 1
            continue

        cand = read_first_mesh(src_dir)
        if cand is None:
            failed += 1
            failures.append({"obj_id": obj_id, "reason": "no mesh found"})
            continue

        if dry_run:
            ok += 1
            continue

        pts = load_points_from_mesh_or_pc(cand, num_points)
        if pts is None or pts.shape[0] == 0:
            failed += 1
            failures.append({"obj_id": obj_id, "src": cand, "reason": "load_points returned None/empty"})
            continue

        try:
            save_points_ply(out_pc_path, pts)
            ok += 1
        except Exception as e:
            failed += 1
            failures.append({"obj_id": obj_id, "src": cand, "exc": str(e)})

    sys.stdout.write("\n")
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s. Success: {ok}/{total}, Failed: {failed}")

    if failures:
        fail_log = os.path.join(out_root, "failures.json")
        write_json(fail_log, {"failures": failures})
        print(f"Wrote failure log with {len(failures)} entries: {fail_log}")


def main():
    ap = argparse.ArgumentParser("Generate GraspIt assets point clouds (preprocessing)")
    ap.add_argument("--assets_root", type=str, default='/mnt/kikerp/OptiSim/assets', help="Path to GraspIt assets (root with object ID folders)")
    ap.add_argument("--num_points", type=int, default=768, help="Target number of points per asset")
    ap.add_argument("--overwrite", default=False, action="store_true", help="Recompute even if output exists")
    ap.add_argument("--dry_run", default=False, action="store_true", help="Only check and print, no writing")
    args = ap.parse_args()

    # Safety: run single-process to avoid Open3D fork issues
    process_assets(args.assets_root, num_points=args.num_points, overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()