#!/usr/bin/env python3
import argparse
import re
import shutil
import sys
from pathlib import Path

DIR_SUFFIX_RE = re.compile(r'^(.+)_([A-Za-z0-9]+)$')               # e.g., depth_xyz -> (depth, xyz)
FILE_SUFFIX_RE = re.compile(r'^(.+)_([A-Za-z0-9]+)(\.[^.]+)$')     # e.g., scene_gt_xyz.json -> (scene_gt, xyz, .json)

def find_cam_ids(scene_path: Path):
    ids = set()
    for p in scene_path.iterdir():
        name = p.name
        if p.is_dir():
            m = DIR_SUFFIX_RE.match(name)
            if m:
                ids.add(m.group(2))
        elif p.is_file():
            m = FILE_SUFFIX_RE.match(name)
            if m:
                ids.add(m.group(2))
    return sorted(ids)

def copy_dir(src: Path, dst: Path):
    shutil.copytree(src, dst, dirs_exist_ok=True)

def restructure_scene(scene_path: Path, dst_root: Path, prefix: str, cam_id: str, dry_run: bool = False):
    scene_id = scene_path.name
    target_scene_dir = dst_root / f"{prefix}_{cam_id}" / scene_id
    if dry_run:
        print(f"[DRY] Create {target_scene_dir}")
    else:
        target_scene_dir.mkdir(parents=True, exist_ok=True)

    for item in scene_path.iterdir():
        name = item.name

        if item.is_dir():
            m = DIR_SUFFIX_RE.match(name)
            if not m:
                continue
            base, suffix = m.group(1), m.group(2)
            if suffix != cam_id:
                continue
            dest = target_scene_dir / base
            if dry_run:
                print(f"[DRY] Copy dir {item} -> {dest}")
            else:
                copy_dir(item, dest)

        elif item.is_file():
            m = FILE_SUFFIX_RE.match(name)
            if not m:
                continue
            base, suffix, ext = m.group(1), m.group(2), m.group(3)
            if suffix != cam_id:
                continue
            dest = target_scene_dir / f"{base}{ext}"
            if dry_run:
                print(f"[DRY] Copy file {item} -> {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)

def restructure_dataset(src_root: Path, dst_root: Path, prefix: str, dry_run: bool = False):
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
            print(f"Warning: No camera-suffixed entries found in scene {scene.name}. Skipping.")
            continue

        print(f"Scene {scene.name}: cameras {', '.join(cam_ids)}")
        for cam_id in cam_ids:
            restructure_scene(scene, dst_root, prefix, cam_id, dry_run=dry_run)

def main():
    parser = argparse.ArgumentParser(
        description="Restructure BOP dataset by removing trailing '_<cam_id>' and creating <prefix>_<cam_id>/<scene_id>/..."
    )
    parser.add_argument("--src", default='data/bop/itoddmv/val', type=Path,
                        help="Path to source split directory (e.g., /mnt/.../bop/xyzidb/val)")
    parser.add_argument("--dst", default='data/bop/itoddmv', type=Path,
                        help="Destination root (e.g., /mnt/.../bop/xyzidb/xyzidb_restructured)")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Prefix for output split (e.g., val, train_pbr). Defaults to src directory name.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned operations without copying.")
    args = parser.parse_args()

    prefix = args.prefix if args.prefix else args.src.name
    restructure_dataset(args.src, args.dst, prefix, dry_run=args.dry_run)
    print("Done." if not args.dry_run else "Dry-run complete.")

if __name__ == "__main__":
    main()