#!/usr/bin/env python3
import argparse
import re
import shutil
import sys
from pathlib import Path

def find_cam_ids(scene_path: Path):
    """
    Detect camera IDs present in a scene directory by scanning entries with '_cam<id>'.
    Returns a sorted list of string IDs.
    """
    ids = set()
    pattern = re.compile(r'_cam(\d+)')
    for p in scene_path.iterdir():
        m = pattern.search(p.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids, key=lambda x: int(x))


def copy_dir(src: Path, dst: Path):
    """
    Copy a directory tree from src to dst. Allows existing destinations.
    """
    shutil.copytree(src, dst, dirs_exist_ok=True)


def restructure_scene(scene_path: Path, dst_root: Path, cam_id: str, dry_run: bool = False):
    """
    For a given scene and camera ID, copy directories/files that belong to that camera
    into dst_root/train_pbr_cam<cam_id>/<scene_id>/ with the '_cam<id>' suffix removed.
    """
    scene_id = scene_path.name
    target_scene_dir = dst_root / f"val_cam{cam_id}" / scene_id
    if dry_run:
        print(f"[DRY] Create {target_scene_dir}")
    else:
        target_scene_dir.mkdir(parents=True, exist_ok=True)

    # Strict matching to avoid accidental matches (e.g., cam1 vs cam11)
    dir_re = re.compile(rf'^(.+)_cam{cam_id}$')
    file_re = re.compile(rf'^(.+)_cam{cam_id}(\.[^.]+)$')

    for item in scene_path.iterdir():
        name = item.name

        if item.is_dir():
            m = dir_re.match(name)
            if not m:
                continue
            base_name = m.group(1)  # e.g., 'rgb', 'mask', 'mask_visib', 'depth', 'aolp', 'dolp'
            dest = target_scene_dir / base_name
            if dry_run:
                print(f"[DRY] Copy dir {item} -> {dest}")
            else:
                copy_dir(item, dest)

        elif item.is_file():
            m = file_re.match(name)
            if not m:
                continue
            base_name, ext = m.group(1), m.group(2)
            dest = target_scene_dir / f"{base_name}{ext}"  # e.g., scene_camera.json, scene_gt.json, scene_gt_info.json
            if dry_run:
                print(f"[DRY] Copy file {item} -> {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)


def restructure_dataset(src_root: Path, dst_root: Path, dry_run: bool = False):
    """
    Iterate scenes under train_pbr and restructure for each camera.
    """
    if not src_root.exists():
        print(f"Source root does not exist: {src_root}", file=sys.stderr)
        sys.exit(1)

    scenes = [d for d in src_root.iterdir() if d.is_dir()]
    if not scenes:
        print(f"No scene directories found under: {src_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(scenes)} scene(s) under {src_root}")
    for scene in sorted(scenes, key=lambda p: p.name):
        cam_ids = find_cam_ids(scene)
        if not cam_ids:
            print(f"Warning: No camera-specific entries found in scene {scene.name}. Skipping.")
            continue

        print(f"Scene {scene.name}: cameras {', '.join(cam_ids)}")
        for cam_id in cam_ids:
            restructure_scene(scene, dst_root, cam_id, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Restructure IPD BOP dataset from per-scene '_camX' layout to train_pbr_<cam_id>/<scene_id>/..."
    )
    parser.add_argument("--src", default='data/bop/ipd/val', type=Path,
                        help="Path to source train_pbr directory (e.g., /mnt/.../bop/ipd/train_pbr)")
    parser.add_argument("--dst", default='data/bop/ipd',  type=Path,
                        help="Path to destination root under bop/ipd (e.g., /mnt/.../bop/ipd/ipd_restructured)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned operations without copying.")
    args = parser.parse_args()

    restructure_dataset(args.src, args.dst, dry_run=args.dry_run)
    print("Done." if not args.dry_run else "Dry-run complete.")

if __name__ == "__main__":
    main()