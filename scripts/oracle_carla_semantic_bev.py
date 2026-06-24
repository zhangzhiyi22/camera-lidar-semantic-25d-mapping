"""Render the CARLA ground-truth upper bound for the aerial semantic BEV task.

This script intentionally reads CARLA semantic/depth sensors and actor labels.
It is for visualization, evaluation, and training supervision only; it is not
a deployable real-world perception pipeline.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from fuse_camera_lidar_semantic_25d import (
    add_gt_boxes_to_grid,
    camera_depth_points,
    height_aware_classes,
    load_calibration,
    load_carla_depth,
    load_carla_semantics,
    load_gt_boxes,
    load_points,
    load_pose,
    project_points,
    rasterize,
    render_pair,
    world_points,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CARLA ground-truth oracle semantic BEV map.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/carla_oracle_semantic_bev")
    parser.add_argument("--length-m", type=float, default=50.0)
    parser.add_argument("--width-m", type=float, default=40.0)
    parser.add_argument("--back-m", type=float, default=10.0)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--window-frames", type=int, default=10)
    parser.add_argument("--depth-stride", type=int, default=5)
    parser.add_argument("--max-depth-m", type=float, default=100.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--gif-duration-ms", type=int, default=180)
    parser.add_argument("--ground-clearance-m", type=float, default=0.45)
    args = parser.parse_args()
    if not 0 <= args.back_m < args.length_m:
        raise ValueError("--back-m must be non-negative and smaller than --length-m.")

    root, output = Path(args.dataset), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    intrinsics, lidar_to_camera = load_calibration(root / "calib.json")
    images = sorted((root / "camera").glob("*.png"))
    if args.max_frames:
        images = images[:args.max_frames]
    recent, previews, report, final_grid = deque(maxlen=args.window_frames), [], [], None

    for image_path in images:
        frame_id = image_path.stem
        lidar_path = root / "lidar" / f"{frame_id}.ply"
        pose_path = root / "poses" / f"{frame_id}.txt"
        semantic_path = root / "semantic" / f"{frame_id}.png"
        depth_path = root / "depth" / f"{frame_id}.png"
        if not all(path.exists() for path in (lidar_path, pose_path, semantic_path, depth_path)):
            continue

        camera = np.asarray(Image.open(image_path).convert("RGB"))
        semantic = load_carla_semantics(semantic_path)
        points = load_points(lidar_path)
        pose = load_pose(pose_path)
        pixels, visible = project_points(points, intrinsics, lidar_to_camera, camera.shape[:2])
        point_classes = np.zeros(len(points), dtype=np.uint8)
        point_classes[visible] = semantic[pixels[visible, 1], pixels[visible, 0]]
        lidar_world = world_points(points, pose)
        point_classes, _ground_z = height_aware_classes(point_classes, lidar_world, args.ground_clearance_m)
        depth_points, depth_classes = camera_depth_points(
            load_carla_depth(depth_path),
            semantic,
            intrinsics,
            lidar_to_camera,
            args.depth_stride,
            args.max_depth_m,
        )
        recent.append((lidar_world, point_classes, world_points(depth_points, pose), depth_classes))
        final_grid = rasterize(recent, pose, args.length_m, args.width_m, args.back_m, args.resolution)
        gt_boxes = add_gt_boxes_to_grid(
            final_grid,
            load_gt_boxes(root / "labels" / f"{frame_id}.json"),
            args.length_m,
            args.width_m,
            args.back_m,
            args.resolution,
        )
        previews.append(
            render_pair(
                camera,
                final_grid,
                frame_id,
                [],
                gt_boxes,
                args.length_m,
                args.width_m,
                args.back_m,
                args.window_frames,
                map_title=f"CARLA ground truth | {args.window_frames}-frame BEV",
                dynamic_label="CARLA GT vehicle / person boxes",
                dynamic_count=len(gt_boxes),
                show_gt_count=False,
            )
        )
        report.append({"frame": frame_id, "ground_truth_boxes": len(gt_boxes)})
        print(f"processed {frame_id}: gt_boxes={len(gt_boxes)}")

    if final_grid is None:
        raise RuntimeError("No complete CARLA RGB/LiDAR/pose/semantic/depth frame sets were found.")
    previews[-1].save(output / "latest_pair.png")
    gif_frames = [frame.convert("P", palette=Image.Palette.ADAPTIVE) for frame in previews]
    gif_frames[0].save(output / "realtime_update.gif", save_all=True, append_images=gif_frames[1:], duration=args.gif_duration_ms, loop=0)
    np.savez_compressed(output / "final_grid.npz", **final_grid)
    summary = {
        "mode": "CARLA ground-truth oracle; not deployable outside CARLA",
        "inputs": ["CARLA RGB", "CARLA semantic camera", "CARLA depth camera", "CARLA LiDAR", "pose/calibration", "CARLA actor labels"],
        "processed_frames": len(report),
        "window_frames": args.window_frames,
        "map": {"length_m": args.length_m, "width_m": args.width_m, "back_m": args.back_m, "resolution_m": args.resolution},
        "frames": report,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
