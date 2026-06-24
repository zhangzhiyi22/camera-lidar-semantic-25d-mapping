"""Build a compact CARLA BEV detection manifest and ground-truth preview.

This is an offline label-processing tool. Runtime models must use RGB, LiDAR,
calibration and pose only; labels are never an online input.
"""

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


CATEGORY_COLORS = {
    "car": (255, 159, 28),
    "truck": (233, 83, 64),
    "bus": (242, 112, 38),
    "motorcycle": (189, 93, 209),
    "bicycle": (64, 136, 220),
    "pedestrian": (226, 58, 172),
}
SEMANTIC_COLORS = {
    1: (183, 191, 196),
    2: (73, 166, 94),
    3: (106, 118, 128),
}
CARLA_SEMANTIC_TAGS = {
    1: (1, 6, 7, 8, 14, 16, 20, 24),
    2: (9, 22),
    3: (2, 3, 5, 11, 12, 15, 17, 18, 19),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create BEV 3D-box training metadata and visual QA from CARLA ground truth.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/carla_gt_boxes_bev")
    parser.add_argument("--extent-m", type=float, default=140.0)
    parser.add_argument("--image-size", type=int, default=900)
    parser.add_argument("--trajectory-frames", type=int, default=20)
    parser.add_argument("--semantic-depth-stride", type=int, default=8)
    parser.add_argument("--semantic-resolution", type=float, default=0.5)
    parser.add_argument("--semantic-history-frames", type=int, default=5)
    parser.add_argument("--gif-duration-ms", type=int, default=140)
    return parser.parse_args()


def read_ascii_ply(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break
        points = np.loadtxt(handle, dtype=np.float32)
    return points.reshape(1, -1) if points.ndim == 1 else points


def load_pose(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["camera_intrinsics"], dtype=np.float64), np.asarray(payload["lidar_to_camera"], dtype=np.float64)


def semantic_depth_world(semantic_path: Path, depth_path: Path, intrinsics: np.ndarray, lidar_to_camera: np.ndarray, pose: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    tags = np.asarray(Image.open(semantic_path).convert("RGB"))[..., 0]
    classes = np.zeros(tags.shape, dtype=np.uint8)
    for class_id, values in CARLA_SEMANTIC_TAGS.items():
        classes[np.isin(tags, values)] = class_id
    depth_rgb = np.asarray(Image.open(depth_path).convert("RGB"), dtype=np.float32)
    depth = (depth_rgb[..., 0] + depth_rgb[..., 1] * 256.0 + depth_rgb[..., 2] * 65536.0) / 16777215.0 * 1000.0
    rows, cols = np.arange(0, tags.shape[0], stride), np.arange(0, tags.shape[1], stride)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    sampled_classes, sampled_depth = classes[vv, uu], depth[vv, uu]
    keep = (sampled_classes > 0) & np.isfinite(sampled_depth) & (sampled_depth > 0.2) & (sampled_depth < 120.0)
    uu, vv, sampled_depth, sampled_classes = uu[keep], vv[keep], sampled_depth[keep], sampled_classes[keep]
    camera_points = np.c_[
        (uu - intrinsics[0, 2]) * sampled_depth / intrinsics[0, 0],
        (vv - intrinsics[1, 2]) * sampled_depth / intrinsics[1, 1],
        sampled_depth,
        np.ones(len(sampled_depth)),
    ].T
    lidar_points = (np.linalg.inv(lidar_to_camera) @ camera_points)[:3].T
    world_points = (pose @ np.c_[lidar_points, np.ones(len(lidar_points))].T)[:3].T
    return world_points, sampled_classes


def horizontal_axes(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    forward = pose[:2, 0].astype(np.float64)
    forward /= np.linalg.norm(forward)
    lateral = pose[:2, 1].astype(np.float64)
    lateral /= np.linalg.norm(lateral)
    return forward, lateral, math.atan2(forward[1], forward[0])


def world_to_bev(points_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    forward, lateral, _ = horizontal_axes(pose)
    delta = points_world[:, :2] - pose[:2, 3]
    return np.c_[delta @ forward, delta @ lateral, points_world[:, 2] - pose[2, 3]]


def box_to_bev(annotation: dict, pose: np.ndarray) -> dict:
    box = annotation["bbox_world"]
    center = np.asarray(box[:3], dtype=np.float64).reshape(1, 3)
    local_center = world_to_bev(center, pose)[0]
    _, _, heading = horizontal_axes(pose)
    yaw = math.atan2(math.sin(box[6] - heading), math.cos(box[6] - heading))
    return {
        "actor_id": annotation["actor_id"],
        "category": annotation["category"],
        "bbox_bev": [float(local_center[0]), float(local_center[1]), float(local_center[2]), *[float(value) for value in box[3:6]], float(yaw)],
        "velocity_world_mps": annotation["velocity_world_mps"],
    }


def pixel_from_bev(x: float, y: float, extent_m: float, image_size: int) -> tuple[float, float]:
    half = extent_m / 2.0
    return (y + half) / extent_m * image_size, (half - x) / extent_m * image_size


def rasterize_semantics(points_world: np.ndarray, classes: np.ndarray, pose: np.ndarray, extent_m: float, resolution_m: float) -> np.ndarray:
    cells = int(round(extent_m / resolution_m))
    half = extent_m / 2.0
    local = world_to_bev(points_world, pose)
    keep = (np.abs(local[:, 0]) < half) & (np.abs(local[:, 1]) < half)
    local, classes = local[keep], classes[keep]
    votes = np.zeros((4, cells * cells), dtype=np.int32)
    rows = np.clip(((half - local[:, 0]) / resolution_m).astype(int), 0, cells - 1)
    cols = np.clip(((local[:, 1] + half) / resolution_m).astype(int), 0, cells - 1)
    flat = rows * cells + cols
    for class_id in (1, 2, 3):
        np.add.at(votes[class_id], flat[classes == class_id], 1)
    grid = votes.argmax(axis=0).astype(np.uint8).reshape(cells, cells)
    observed = votes.sum(axis=0).reshape(cells, cells) > 0
    # Close tiny sampling gaps but never overwrite an observed semantic class.
    for class_id in (1, 2, 3):
        mask = (grid == class_id).astype(np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        grid[(~observed) & (closed > 0)] = class_id
    return grid


def box_corners(box: list[float]) -> np.ndarray:
    x, y, _, length, width, _, yaw = box
    corners = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return corners @ rotation.T + np.array([x, y])


def draw_bev(points_world: np.ndarray, semantic_world: np.ndarray, semantic_classes: np.ndarray, pose: np.ndarray, boxes: list[dict], tracks: dict[int, list[np.ndarray]], frame_id: str, extent_m: float, image_size: int, semantic_resolution_m: float) -> Image.Image:
    semantic_grid = rasterize_semantics(semantic_world, semantic_classes, pose, extent_m, semantic_resolution_m)
    semantic_rgb = np.full((*semantic_grid.shape, 3), (249, 251, 252), dtype=np.uint8)
    for class_id, color in SEMANTIC_COLORS.items():
        semantic_rgb[semantic_grid == class_id] = color
    image = Image.fromarray(semantic_rgb, "RGB").resize((image_size, image_size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    half = extent_m / 2
    local = world_to_bev(points_world, pose)
    keep = (np.abs(local[:, 0]) < extent_m / 2) & (np.abs(local[:, 1]) < extent_m / 2)
    local = local[keep]
    ground = np.percentile(local[:, 2], 8) if len(local) else 0.0
    for point in local:
        px, py = pixel_from_bev(point[0], point[1], extent_m, image_size)
        if point[2] >= ground + 1.0:
            draw.point((px, py), fill=(41, 133, 149))

    for offset in range(-int(half), int(half) + 1, 20):
        p0 = pixel_from_bev(-half, offset, extent_m, image_size)
        p1 = pixel_from_bev(half, offset, extent_m, image_size)
        draw.line((p0, p1), fill=(225, 230, 233), width=1)
        p0 = pixel_from_bev(offset, -half, extent_m, image_size)
        p1 = pixel_from_bev(offset, half, extent_m, image_size)
        draw.line((p0, p1), fill=(225, 230, 233), width=1)

    for actor_id, history in tracks.items():
        if len(history) < 2:
            continue
        history_world = np.asarray(history)
        history_local = world_to_bev(history_world, pose)
        history_local = history_local[(np.abs(history_local[:, 0]) < half) & (np.abs(history_local[:, 1]) < half)]
        if len(history_local) > 1:
            draw.line([pixel_from_bev(point[0], point[1], extent_m, image_size) for point in history_local], fill=(71, 78, 150), width=2)

    boxes = sorted(boxes, key=lambda item: item["bbox_bev"][0] ** 2 + item["bbox_bev"][1] ** 2)
    for index, annotation in enumerate(boxes):
        box = annotation["bbox_bev"]
        if abs(box[0]) >= half or abs(box[1]) >= half:
            continue
        color = CATEGORY_COLORS[annotation["category"]]
        corners = [pixel_from_bev(x, y, extent_m, image_size) for x, y in box_corners(box)]
        draw.line(corners + [corners[0]], fill=color, width=3)
        front = pixel_from_bev(box[0] + math.cos(box[6]) * box[3] / 2, box[1] + math.sin(box[6]) * box[3] / 2, extent_m, image_size)
        draw.ellipse((front[0] - 3, front[1] - 3, front[0] + 3, front[1] + 3), fill=color)
        if index < 18:
            draw.text((corners[0][0] + 3, corners[0][1] + 3), annotation["category"], fill=color, font=font)

    ego = pixel_from_bev(0.0, 0.0, extent_m, image_size)
    draw.ellipse((ego[0] - 7, ego[1] - 7, ego[0] + 7, ego[1] + 7), fill=(61, 38, 116), outline=(255, 255, 255), width=2)
    draw.text((14, 14), f"CARLA GT BEV | frame {frame_id}", fill=(23, 31, 38), font=font)
    draw.text((14, image_size - 20), "gray: road   green: vegetation   dark gray: structure   boxes: CARLA 3D ground truth", fill=(54, 65, 72), font=font)
    return image


def compose_pair(camera_path: Path, bev: Image.Image, frame_id: str) -> Image.Image:
    """Show the source RGB frame beside its aligned semantic/detection BEV."""
    panel_width, panel_height = 800, 450
    camera = Image.open(camera_path).convert("RGB").resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    crop_height = round(bev.width * panel_height / panel_width)
    top = (bev.height - crop_height) // 2
    bev_panel = bev.crop((0, top, bev.width, top + crop_height)).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1648, 520), (247, 250, 251))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    canvas.paste(camera, (20, 50))
    canvas.paste(bev_panel, (828, 50))
    draw.text((28, 28), f"Frame {frame_id} | CARLA RGB camera", fill=(25, 35, 42), font=font)
    draw.text((828, 28), "Aligned semantic + LiDAR + 3D detection BEV", fill=(25, 35, 42), font=font)
    draw.text((20, 510), "RGB input", fill=(77, 88, 95), font=font)
    draw.text((828, 510), "semantic ground truth, LiDAR geometry, and dynamic-object boxes", fill=(77, 88, 95), font=font)
    return canvas


def main() -> None:
    args = parse_args()
    root, output = Path(args.dataset), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label_files = sorted((root / "labels").glob("*.json"))
    if not label_files:
        raise FileNotFoundError("No labels/*.json files found.")
    intrinsics, lidar_to_camera = load_calibration(root / "calib.json")

    manifest_frames, previews = [], []
    tracks = defaultdict(list)
    semantic_history = deque(maxlen=args.semantic_history_frames)
    category_counts = defaultdict(int)
    for label_path in label_files:
        frame_id = label_path.stem
        pose_path, lidar_path = root / "poses" / f"{frame_id}.txt", root / "lidar" / f"{frame_id}.ply"
        if not pose_path.exists() or not lidar_path.exists():
            continue
        source = json.loads(label_path.read_text(encoding="utf-8"))
        pose = load_pose(pose_path)
        boxes = [box_to_bev(annotation, pose) for annotation in source["annotations"]]
        for annotation, box in zip(source["annotations"], boxes):
            tracks[box["actor_id"]].append(np.asarray(annotation["bbox_world"][:3], dtype=np.float64))
            tracks[box["actor_id"]] = tracks[box["actor_id"]][-args.trajectory_frames:]
            category_counts[box["category"]] += 1
        camera_path = root / "camera" / f"{frame_id}.png"
        semantic_path, depth_path = root / "semantic" / f"{frame_id}.png", root / "depth" / f"{frame_id}.png"
        if not semantic_path.exists() or not depth_path.exists():
            continue
        points = read_ascii_ply(lidar_path)[:, :3]
        points_world = (pose @ np.c_[points, np.ones(len(points))].T)[:3].T
        semantic_world, semantic_classes = semantic_depth_world(semantic_path, depth_path, intrinsics, lidar_to_camera, pose, args.semantic_depth_stride)
        semantic_history.append((semantic_world, semantic_classes))
        fused_semantic_world = np.concatenate([item[0] for item in semantic_history], axis=0)
        fused_semantic_classes = np.concatenate([item[1] for item in semantic_history], axis=0)
        bev = draw_bev(points_world, fused_semantic_world, fused_semantic_classes, pose, boxes, tracks, frame_id, args.extent_m, args.image_size, args.semantic_resolution)
        previews.append(compose_pair(camera_path, bev, frame_id))
        manifest_frames.append(
            {
                "frame_id": frame_id,
                "camera_path": f"camera/{frame_id}.png",
                "lidar_path": f"lidar/{frame_id}.ply",
                "pose_path": f"poses/{frame_id}.txt",
                "label_path": f"labels/{frame_id}.json",
                "boxes": boxes,
            }
        )
        print(f"processed {frame_id}: boxes={len(boxes)}")

    manifest = {
        "coordinate_system": "Local horizontal BEV frame: x forward, y lateral, z up; derived from world pose and sensor heading.",
        "classes": list(CATEGORY_COLORS),
        "frames": manifest_frames,
    }
    (output / "bev_training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    previews[-1].save(output / "latest_pair.png")
    gif_frames = [frame.convert("P", palette=Image.Palette.ADAPTIVE) for frame in previews]
    gif_frames[0].save(output / "realtime_update.gif", save_all=True, append_images=gif_frames[1:], duration=args.gif_duration_ms, loop=0)
    summary = {"frames": len(manifest_frames), "boxes_by_category": dict(category_counts), "artifacts": ["bev_training_manifest.json", "latest_pair.png", "realtime_update.gif"]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
