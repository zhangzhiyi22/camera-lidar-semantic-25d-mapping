"""Fuse synchronized CARLA camera/LiDAR data into a readable local 2.5D map.

The collector output is expected to contain camera/, lidar/, poses/, and
calib.json. This script keeps the artifacts intentionally small: one final
grid, one latest-frame preview, one GIF, and a compact run report.
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "huggingface"))

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO


CLASS = {"unknown": 0, "road": 1, "vegetation": 2, "building": 3, "vehicle": 4, "person": 5}
PALETTE = np.array(
    [
        [239, 243, 245],
        [193, 198, 202],
        [58, 157, 84],
        [112, 124, 134],
        [45, 115, 207],
        [239, 145, 47],
    ],
    dtype=np.uint8,
)
YOLO_DYNAMIC = {
    "person": CLASS["person"],
    "pedestrian": CLASS["person"],
    "people": CLASS["person"],
    "car": CLASS["vehicle"],
    "van": CLASS["vehicle"],
    "bus": CLASS["vehicle"],
    "truck": CLASS["vehicle"],
    "tricycle": CLASS["vehicle"],
    "awning-tricycle": CLASS["vehicle"],
    "motor": CLASS["vehicle"],
    "motorcycle": CLASS["vehicle"],
    "bicycle": CLASS["vehicle"],
}
CARLA_LABELS = {
    # This Town asset exports asphalt under tag 1 rather than the usual tag 7.
    "road": (1, 6, 7, 8, 14, 16, 20, 24),
    "vegetation": (9, 22),
    "building": (2, 3, 5, 11, 12, 15, 17, 18, 19),
    "vehicle": (10,),
    "person": (4, 25),
}


def read_ascii_ply(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = []
        for line in handle:
            header.append(line)
            if line.strip() == "end_header":
                break
        if not any("format ascii" in line for line in header):
            raise ValueError(f"{path} must be an ASCII PLY file.")
        points = np.loadtxt(handle, dtype=np.float32)
    return points.reshape(1, -1) if points.ndim == 1 else points


def load_points(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        points = np.load(path).astype(np.float32)
    elif path.suffix == ".bin":
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    else:
        points = read_ascii_ply(path)
    return points[:, :3]


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = np.asarray(payload["camera_intrinsics"], dtype=np.float64)
    lidar_to_camera = np.asarray(payload["lidar_to_camera"], dtype=np.float64)
    if intrinsics.shape != (3, 3) or lidar_to_camera.shape != (4, 4):
        raise ValueError("calib.json must provide 3x3 intrinsics and a 4x4 lidar_to_camera matrix.")
    return intrinsics, lidar_to_camera


def local_model_path(model_name: str) -> str:
    """Resolve a Hugging Face snapshot locally so processing never triggers a download."""
    candidate = Path(model_name)
    if candidate.is_dir():
        return str(candidate)
    model_cache = Path(os.environ["HF_HOME"]) / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    snapshots = sorted(path for path in model_cache.glob("*") if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(
            f"SegFormer model is not cached locally: {model_name}. Run once with a trusted network, then rerun offline."
        )
    return str(snapshots[-1])


def load_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64)
    if pose.size != 16:
        raise ValueError(f"{path} does not contain a 4x4 pose.")
    return pose.reshape(4, 4)


def load_carla_semantics(path: Path) -> np.ndarray:
    """Map CARLA's raw semantic tag (PNG red channel) to project classes."""
    raw_tags = np.asarray(Image.open(path).convert("RGB"))[..., 0]
    mapped = np.zeros(raw_tags.shape, dtype=np.uint8)
    for class_name, tags in CARLA_LABELS.items():
        mapped[np.isin(raw_tags, tags)] = CLASS[class_name]
    return mapped


def load_carla_depth(path: Path) -> np.ndarray:
    """Decode CARLA's 24-bit RGB depth encoding into meters."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    normalized = (rgb[..., 0] + rgb[..., 1] * 256.0 + rgb[..., 2] * 65536.0) / 16777215.0
    return normalized * 1000.0


def camera_depth_points(depth_m: np.ndarray, semantic: np.ndarray, intrinsics: np.ndarray, lidar_to_camera: np.ndarray, stride: int, max_depth_m: float) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(0, depth_m.shape[0], stride)
    cols = np.arange(0, depth_m.shape[1], stride)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    depth = depth_m[vv, uu]
    classes = semantic[vv, uu]
    keep = np.isfinite(depth) & (depth > 0.2) & (depth < max_depth_m) & (classes != CLASS["unknown"])
    uu, vv, depth, classes = uu[keep], vv[keep], depth[keep], classes[keep]
    x = (uu - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (vv - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    camera_points = np.c_[x, y, depth, np.ones(len(depth))].T
    lidar_points = (np.linalg.inv(lidar_to_camera) @ camera_points)[:3].T.astype(np.float32)
    return lidar_points, classes.astype(np.uint8)


def project_points(points: np.ndarray, intrinsics: np.ndarray, lidar_to_camera: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    camera_points = lidar_to_camera @ np.c_[points, np.ones(len(points))].T
    depth = camera_points[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = intrinsics @ camera_points[:3]
        u = pixels[0] / pixels[2]
        v = pixels[1] / pixels[2]
    height, width = image_shape
    finite = np.isfinite(u) & np.isfinite(v)
    ui = np.zeros(len(points), dtype=np.int32)
    vi = np.zeros(len(points), dtype=np.int32)
    ui[finite] = np.rint(u[finite]).astype(np.int32)
    vi[finite] = np.rint(v[finite]).astype(np.int32)
    visible = finite & (depth > 0.1) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    return np.c_[ui, vi], visible


def height_aware_classes(point_classes: np.ndarray, points_world: np.ndarray, ground_clearance_m: float) -> tuple[np.ndarray, float]:
    """Use LiDAR geometry to recover a reliable ground class from RGB semantics."""
    ground_z = float(np.percentile(points_world[:, 2], 5))
    classes = point_classes.copy()
    dynamic = np.isin(classes, (CLASS["vehicle"], CLASS["person"]))
    classes[(points_world[:, 2] <= ground_z + ground_clearance_m) & ~dynamic] = CLASS["road"]
    return classes, ground_z


def lift_ground_pixels_to_lidar(
    pixels: np.ndarray,
    visible: np.ndarray,
    points_world: np.ndarray,
    semantic: np.ndarray,
    intrinsics: np.ndarray,
    lidar_to_camera: np.ndarray,
    pose: np.ndarray,
    ground_z: float,
    ground_clearance_m: float,
    stride: int,
    radius_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift RGB road pixels onto the LiDAR-estimated horizontal ground plane.

    This is a lightweight, geometry-constrained version of camera-to-BEV lifting:
    only image regions supported by nearby LiDAR ground returns are lifted.
    """
    seeds = np.zeros(semantic.shape, dtype=np.uint8)
    lidar_ground = visible & (points_world[:, 2] <= ground_z + ground_clearance_m)
    if not lidar_ground.any():
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint8)
    seed_pixels = pixels[lidar_ground]
    seeds[seed_pixels[:, 1], seed_pixels[:, 0]] = 1
    kernel_size = radius_px * 2 + 1
    support = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    road_mask = (support > 0) & (semantic != CLASS["vegetation"])
    rows = np.arange(0, semantic.shape[0], stride)
    cols = np.arange(0, semantic.shape[1], stride)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    keep = road_mask[vv, uu]
    uu, vv = uu[keep], vv[keep]
    if len(uu) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint8)

    rays_camera = np.c_[
        (uu - intrinsics[0, 2]) / intrinsics[0, 0],
        (vv - intrinsics[1, 2]) / intrinsics[1, 1],
        np.ones(len(uu)),
    ]
    camera_to_lidar = np.linalg.inv(lidar_to_camera)
    ray_origin_lidar = camera_to_lidar[:3, 3]
    ray_dirs_lidar = (camera_to_lidar[:3, :3] @ rays_camera.T).T
    ray_origin_world = pose[:3, :3] @ ray_origin_lidar + pose[:3, 3]
    ray_dirs_world = (pose[:3, :3] @ ray_dirs_lidar.T).T
    valid_ray = np.abs(ray_dirs_world[:, 2]) > 1e-5
    distance = np.zeros(len(uu), dtype=np.float64)
    distance[valid_ray] = (ground_z - ray_origin_world[2]) / ray_dirs_world[valid_ray, 2]
    valid_ray &= distance > 0
    if not valid_ray.any():
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint8)
    lifted_world = ray_origin_world + ray_dirs_world[valid_ray] * distance[valid_ray, None]
    lifted_lidar = (np.linalg.inv(pose) @ np.c_[lifted_world, np.ones(len(lifted_world))].T)[:3].T.astype(np.float32)
    return lifted_lidar, np.full(len(lifted_lidar), CLASS["road"], dtype=np.uint8)


def map_ade_labels(label_map: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    output = np.zeros(label_map.shape, dtype=np.uint8)
    for label_id, raw_name in id2label.items():
        name = raw_name.lower()
        if name in CLASS:
            output[label_map == int(label_id)] = CLASS[name]
        elif any(word in name for word in ("road", "street", "sidewalk", "path", "floor", "pavement")):
            output[label_map == int(label_id)] = CLASS["road"]
        elif any(word in name for word in ("tree", "grass", "plant", "field", "flower", "palm", "bush")):
            output[label_map == int(label_id)] = CLASS["vegetation"]
        elif any(word in name for word in ("building", "house", "wall", "fence", "skyscraper", "roof")):
            output[label_map == int(label_id)] = CLASS["building"]
        elif any(word in name for word in ("car", "bus", "truck", "van", "automobile", "bicycle", "motorcycle")):
            output[label_map == int(label_id)] = CLASS["vehicle"]
        elif "person" in name or "pedestrian" in name:
            output[label_map == int(label_id)] = CLASS["person"]
    return output


def predict_semantics(image: np.ndarray, processor, model, device: torch.device) -> np.ndarray:
    inputs = processor(images=Image.fromarray(image), return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    labels = torch.nn.functional.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False).argmax(dim=1)[0]
    return map_ade_labels(labels.cpu().numpy(), model.config.id2label)


def yolo_overrides(semantic: np.ndarray, model, image: np.ndarray, confidence: float, image_size: int) -> tuple[np.ndarray, list[dict]]:
    result = model.predict(image, conf=confidence, imgsz=image_size, verbose=False)[0]
    detections = []
    if result.boxes is None:
        return semantic, detections
    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        if name not in YOLO_DYNAMIC:
            continue
        x0, y0, x1, y1 = np.rint(box.xyxy[0].cpu().numpy()).astype(int)
        x0, x1 = np.clip((x0, x1), 0, semantic.shape[1])
        y0, y1 = np.clip((y0, y1), 0, semantic.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        semantic[y0:y1, x0:x1] = YOLO_DYNAMIC[name]
        detections.append(
            {
                "class": name,
                "confidence": round(float(box.conf[0]), 3),
                "xyxy": [int(x0), int(y0), int(x1), int(y1)],
            }
        )
    return semantic, detections


def world_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (pose @ np.c_[points, np.ones(len(points))].T)[:3].T.astype(np.float32)


def local_axes(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = pose[:2, 0].copy()
    lateral = pose[:2, 1].copy()
    forward /= np.linalg.norm(forward)
    lateral /= np.linalg.norm(lateral)
    return forward, lateral


def smooth_class_map(class_id: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Apply a conservative majority filter only within observed grid cells."""
    observed = count > 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighborhood = np.zeros((len(CLASS), *class_id.shape), dtype=np.uint8)
    for category in range(len(CLASS)):
        mask = ((class_id == category) & observed).astype(np.uint8)
        neighborhood[category] = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    winner = neighborhood.argmax(axis=0).astype(np.uint8)
    support = neighborhood.max(axis=0)
    result = class_id.copy()
    replace = observed & (support >= 3)
    result[replace] = winner[replace]
    return result


def load_gt_boxes(path: Path) -> list[dict]:
    """Load CARLA dynamic-object ground-truth boxes for optional visualization."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    boxes = []
    for annotation in payload.get("annotations", []):
        category = annotation.get("category", "")
        if category in {"car", "truck", "bus", "motorcycle", "bicycle"}:
            class_id = CLASS["vehicle"]
        elif category == "pedestrian":
            class_id = CLASS["person"]
        else:
            continue
        values = annotation.get("bbox_world", [])
        if len(values) != 7:
            continue
        boxes.append({"class_id": class_id, "category": category, "bbox_world": values})
    return boxes


def add_gt_boxes_to_grid(
    grid: dict,
    boxes: list[dict],
    length_m: float,
    width_m: float,
    back_m: float,
    resolution: float,
) -> list[dict]:
    """Rasterize current-frame CARLA boxes into the same local BEV used by fusion."""
    rows, cols = grid["class_id"].shape
    front_m = length_m - back_m
    origin, forward, lateral = grid["origin"], grid["forward"], grid["lateral"]
    rendered = []
    for box in boxes:
        x, y, _z, length, width, _height, yaw = box["bbox_world"]
        local_center = np.array([x, y]) - origin[:2]
        along_center, across_center = local_center @ forward, local_center @ lateral
        if not (-back_m <= along_center < front_m and -width_m / 2 <= across_center < width_m / 2):
            continue
        world_heading = np.array([np.cos(yaw), np.sin(yaw)])
        local_yaw = np.arctan2(world_heading @ lateral, world_heading @ forward)
        heading = np.array([np.cos(local_yaw), np.sin(local_yaw)])
        side = np.array([-heading[1], heading[0]])
        corners_local = np.array([
            [along_center, across_center] + heading * length / 2 + side * width / 2,
            [along_center, across_center] + heading * length / 2 - side * width / 2,
            [along_center, across_center] - heading * length / 2 - side * width / 2,
            [along_center, across_center] - heading * length / 2 + side * width / 2,
        ])
        pixels = np.c_[
            (front_m - corners_local[:, 0]) / resolution,
            (corners_local[:, 1] + width_m / 2) / resolution,
        ].astype(np.int32)
        cv2.fillPoly(grid["class_id"], [pixels[:, ::-1]], int(box["class_id"]))
        cv2.polylines(grid["class_id"], [pixels[:, ::-1]], True, int(box["class_id"]), 1)
        rendered.append({**box, "pixels": pixels})
    return rendered


def add_current_dynamic_points_to_grid(
    grid: dict,
    points_world: np.ndarray,
    point_classes: np.ndarray,
    length_m: float,
    width_m: float,
    back_m: float,
    resolution: float,
) -> int:
    """Give current YOLO-supported LiDAR hits priority over accumulated ground votes."""
    dynamic = np.isin(point_classes, (CLASS["vehicle"], CLASS["person"]))
    if not dynamic.any():
        return 0
    front_m = length_m - back_m
    delta_xy = points_world[dynamic, :2] - grid["origin"][:2]
    along = delta_xy @ grid["forward"]
    across = delta_xy @ grid["lateral"]
    keep = (along >= -back_m) & (along < front_m) & (across >= -width_m / 2) & (across < width_m / 2)
    if not keep.any():
        return 0
    rows = np.clip(((front_m - along[keep]) / resolution).astype(int), 0, grid["class_id"].shape[0] - 1)
    cols = np.clip(((across[keep] + width_m / 2) / resolution).astype(int), 0, grid["class_id"].shape[1] - 1)
    classes = point_classes[dynamic][keep]
    for row, col, class_id in zip(rows, cols, classes):
        row_start, row_end = max(0, row - 1), min(grid["class_id"].shape[0], row + 2)
        col_start, col_end = max(0, col - 1), min(grid["class_id"].shape[1], col + 2)
        grid["class_id"][row_start:row_end, col_start:col_end] = class_id
    return len(rows)


def rasterize(frames: deque, current_pose: np.ndarray, length_m: float, width_m: float, back_m: float, resolution: float) -> dict:
    rows, cols = int(round(length_m / resolution)), int(round(width_m / resolution))
    front_m = length_m - back_m
    origin = current_pose[:3, 3]
    forward, lateral = local_axes(current_pose)
    count = np.zeros(rows * cols, dtype=np.int32)
    z_min = np.full(rows * cols, np.inf, dtype=np.float32)
    z_max = np.full(rows * cols, -np.inf, dtype=np.float32)
    votes = np.zeros((len(CLASS), rows * cols), dtype=np.int32)

    def accumulate(points: np.ndarray, point_classes: np.ndarray) -> None:
        delta_xy = points[:, :2] - origin[:2]
        along = delta_xy @ forward
        across = delta_xy @ lateral
        keep = (along >= -back_m) & (along < front_m) & (across >= -width_m / 2) & (across < width_m / 2)
        if not keep.any():
            return
        along, across = along[keep], across[keep]
        z, classes = points[keep, 2], point_classes[keep]
        row = np.clip(((front_m - along) / resolution).astype(int), 0, rows - 1)
        col = np.clip(((across + width_m / 2) / resolution).astype(int), 0, cols - 1)
        flat = row * cols + col
        np.add.at(count, flat, 1)
        np.minimum.at(z_min, flat, z)
        np.maximum.at(z_max, flat, z)
        for class_id in range(len(CLASS)):
            np.add.at(votes[class_id], flat[classes == class_id], 1)

    for lidar_points, lidar_classes, camera_points, camera_classes in frames:
        accumulate(lidar_points, lidar_classes)
        accumulate(camera_points, camera_classes)

    occupied = count > 0
    class_id = votes.argmax(axis=0).astype(np.uint8)
    class_id[~occupied] = CLASS["unknown"]
    ground = float(np.percentile(z_min[occupied], 10)) if occupied.any() else 0.0
    class_id = class_id.reshape(rows, cols)
    count = count.reshape(rows, cols)
    return {
        "class_id": smooth_class_map(class_id, count),
        "count": count,
        "z_min": z_min.reshape(rows, cols),
        "z_max": z_max.reshape(rows, cols),
        "ground": ground,
        "origin": origin,
        "forward": forward,
        "lateral": lateral,
    }


def render_pair(
    camera: np.ndarray,
    grid: dict,
    frame_id: str,
    detections: list[dict],
    gt_boxes: list[dict],
    length_m: float,
    width_m: float,
    back_m: float,
    window_frames: int,
    map_title: str | None = None,
    dynamic_label: str = "YOLO detections",
    dynamic_count: int | None = None,
    show_gt_count: bool = True,
) -> Image.Image:
    font = ImageFont.load_default()
    panel_width, panel_height = 800, 450
    camera_image = Image.fromarray(camera, "RGB").resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    camera_draw = ImageDraw.Draw(camera_image)
    for detection in detections:
        x0, y0, x1, y1 = detection["xyxy"]
        x_scale, y_scale = panel_width / camera.shape[1], panel_height / camera.shape[0]
        coordinates = (int(x0 * x_scale), int(y0 * y_scale), int(x1 * x_scale), int(y1 * y_scale))
        class_id = YOLO_DYNAMIC[detection["class"]]
        color = tuple(PALETTE[class_id])
        camera_draw.rectangle(coordinates, outline=color, width=2)
        camera_draw.text((coordinates[0] + 2, max(0, coordinates[1] - 11)), f"{detection['class']} {detection['confidence']:.2f}", fill=color, font=font)
    map_rgb = PALETTE[grid["class_id"]]
    occupied = grid["count"] > 0
    if occupied.any():
        height = np.clip(grid["z_max"] - grid["ground"], 0, 12)
        shade = 0.82 + 0.22 * height / 12.0
        map_rgb[occupied] = np.clip(map_rgb[occupied].astype(np.float32) * shade[occupied, None], 0, 255).astype(np.uint8)
    map_width = round(panel_height * map_rgb.shape[1] / map_rgb.shape[0])
    map_image = Image.fromarray(map_rgb, "RGB").resize((map_width, panel_height), Image.Resampling.NEAREST)

    canvas = Image.new("RGB", (1648, 520), (246, 249, 250))
    canvas.paste(camera_image, (20, 50))
    canvas.paste(map_image, (828, 50))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 28), f"Frame {frame_id} | RGB input", fill=(25, 35, 42), font=font)
    draw.text((828, 28), map_title or f"SegFormer + LiDAR | {window_frames}-frame BEV", fill=(25, 35, 42), font=font)

    for box in gt_boxes:
        grid_corners = box["pixels"].astype(float)
        corners = np.c_[
            828 + grid_corners[:, 1] * map_width / grid["class_id"].shape[1],
            50 + grid_corners[:, 0] * panel_height / grid["class_id"].shape[0],
        ]
        draw.line([tuple(point) for point in np.r_[corners, corners[:1]]], fill=(15, 20, 24), width=2)

    # Mark the current platform at its actual map position, not in a decorative frame.
    platform_y = 50 + int((length_m - back_m) / length_m * panel_height)
    platform_x = 828 + map_width // 2
    draw.polygon([(platform_x, platform_y - 10), (platform_x - 7, platform_y + 8), (platform_x + 7, platform_y + 8)], fill=(20, 29, 35))
    draw.text((platform_x + 12, platform_y - 7), "UAV", fill=(20, 29, 35), font=font)
    draw.text((828 + map_width + 14, 82), "Class legend", fill=(25, 35, 42), font=font)
    legend = [("road / ground", 1), ("vegetation", 2), ("building / structure", 3), ("vehicle", 4), ("person", 5)]
    x, y = 828 + map_width + 14, 112
    for label, class_id in legend:
        draw.rectangle((x, y, x + 12, y + 12), fill=tuple(PALETTE[class_id]))
        draw.text((x + 18, y), label, fill=(52, 63, 70), font=font)
        y += 30
    draw.text((828 + map_width + 14, 300), f"LiDAR occupied cells: {int(occupied.sum())}", fill=(77, 88, 95), font=font)
    count = len(detections) if dynamic_count is None else dynamic_count
    draw.text((828 + map_width + 14, 326), f"{dynamic_label}: {count}", fill=(77, 88, 95), font=font)
    if show_gt_count:
        draw.text((828 + map_width + 14, 348), f"GT boxes (optional): {len(gt_boxes)}", fill=(77, 88, 95), font=font)
    draw.text((20, 510), "RGB camera", fill=(77, 88, 95), font=font)
    draw.text((828, 510), f"{length_m:.0f} m x {width_m:.0f} m local semantic map", fill=(77, 88, 95), font=font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact visual camera/LiDAR semantic 2.5D result.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/camera_lidar_semantic_25d")
    parser.add_argument("--length-m", type=float, default=50.0)
    parser.add_argument("--width-m", type=float, default=40.0)
    parser.add_argument("--back-m", type=float, default=10.0, help="Map coverage behind the platform.")
    parser.add_argument("--resolution", type=float, default=0.50)
    parser.add_argument("--window-frames", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--gif-duration-ms", type=int, default=180)
    parser.add_argument("--semantic-source", choices=("auto", "carla", "model"), default="auto")
    parser.add_argument("--depth-stride", type=int, default=5, help="Use every Nth depth pixel for dense semantic projection.")
    parser.add_argument("--max-depth-m", type=float, default=100.0)
    parser.add_argument("--use-depth", action="store_true", help="Fuse depth-camera points in addition to LiDAR.")
    parser.add_argument("--camera-ground-lift", action="store_true", help="Lift RGB regions constrained by LiDAR ground returns into BEV.")
    parser.add_argument("--ground-clearance-m", type=float, default=0.45)
    parser.add_argument("--ground-lift-stride", type=int, default=8)
    parser.add_argument("--ground-lift-radius-px", type=int, default=17)
    parser.add_argument("--use-yolo", action="store_true", help="Also overlay YOLO dynamic-object detections.")
    parser.add_argument("--use-gt-boxes", action="store_true", help="Overlay CARLA labels/*.json vehicle/person boxes on the BEV map.")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=1280, help="Inference size for small aerial objects.")
    parser.add_argument("--semantic-model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--yolo-model", default="models/visdrone_yolov8n.pt")
    args = parser.parse_args()
    if not 0 <= args.back_m < args.length_m:
        raise ValueError("--back-m must be non-negative and smaller than --length-m.")

    root, output = Path(args.dataset), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    intrinsics, lidar_to_camera = load_calibration(root / "calib.json")
    images = sorted((root / "camera").glob("*.png"))
    if args.max_frames:
        images = images[:args.max_frames]
    has_carla_labels = (root / "semantic").is_dir() and any((root / "semantic").glob("*.png"))
    use_carla_labels = args.semantic_source == "carla" or (args.semantic_source == "auto" and has_carla_labels)
    if args.semantic_source == "carla" and not has_carla_labels:
        raise FileNotFoundError("--semantic-source carla requires dataset/semantic/*.png.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = semantic_model = yolo_model = None
    if not use_carla_labels:
        semantic_model_path = local_model_path(args.semantic_model)
        processor = SegformerImageProcessor.from_pretrained(semantic_model_path, local_files_only=True)
        semantic_model = SegformerForSemanticSegmentation.from_pretrained(semantic_model_path, local_files_only=True).to(device).eval()
    if args.use_yolo:
        yolo_model = YOLO(args.yolo_model)
    recent = deque(maxlen=args.window_frames)
    report, previews, final_grid = [], [], None

    for image_path in images:
        frame_id = image_path.stem
        lidar_path = root / "lidar" / f"{frame_id}.ply"
        pose_path = root / "poses" / f"{frame_id}.txt"
        if not lidar_path.exists() or not pose_path.exists():
            continue
        started = time.perf_counter()
        camera = np.asarray(Image.open(image_path).convert("RGB"))
        semantic_path = root / "semantic" / f"{frame_id}.png"
        depth_path = root / "depth" / f"{frame_id}.png"
        if use_carla_labels and semantic_path.exists():
            semantic = load_carla_semantics(semantic_path)
        else:
            semantic = predict_semantics(camera, processor, semantic_model, device)
        detections = []
        if yolo_model is not None:
            semantic, detections = yolo_overrides(semantic, yolo_model, camera, args.yolo_confidence, args.yolo_imgsz)
        points = load_points(lidar_path)
        pixels, visible = project_points(points, intrinsics, lidar_to_camera, camera.shape[:2])
        point_classes = np.zeros(len(points), dtype=np.uint8)
        point_classes[visible] = semantic[pixels[visible, 1], pixels[visible, 0]]
        pose = load_pose(pose_path)
        lidar_world = world_points(points, pose)
        point_classes, ground_z = height_aware_classes(point_classes, lidar_world, args.ground_clearance_m)
        if args.use_depth and depth_path.exists():
            depth_points, depth_classes = camera_depth_points(load_carla_depth(depth_path), semantic, intrinsics, lidar_to_camera, args.depth_stride, args.max_depth_m)
        elif args.camera_ground_lift:
            depth_points, depth_classes = lift_ground_pixels_to_lidar(
                pixels,
                visible,
                lidar_world,
                semantic,
                intrinsics,
                lidar_to_camera,
                pose,
                ground_z,
                args.ground_clearance_m,
                args.ground_lift_stride,
                args.ground_lift_radius_px,
            )
        else:
            depth_points, depth_classes = np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint8)
        recent.append((lidar_world, point_classes, world_points(depth_points, pose), depth_classes))
        final_grid = rasterize(recent, pose, args.length_m, args.width_m, args.back_m, args.resolution)
        dynamic_lidar_hits = add_current_dynamic_points_to_grid(
            final_grid,
            lidar_world,
            point_classes,
            args.length_m,
            args.width_m,
            args.back_m,
            args.resolution,
        )
        gt_boxes = []
        if args.use_gt_boxes:
            gt_boxes = add_gt_boxes_to_grid(
                final_grid,
                load_gt_boxes(root / "labels" / f"{frame_id}.json"),
                args.length_m,
                args.width_m,
                args.back_m,
                args.resolution,
            )
        previews.append(render_pair(camera, final_grid, frame_id, detections, gt_boxes, args.length_m, args.width_m, args.back_m, args.window_frames))
        report.append({"frame": frame_id, "detections": detections, "dynamic_lidar_hits": dynamic_lidar_hits, "gt_boxes": len(gt_boxes), "latency_ms": round((time.perf_counter() - started) * 1000, 1)})
        print(f"processed {frame_id}: cells={int((final_grid['count'] > 0).sum())} gt_boxes={len(gt_boxes)}")

    if final_grid is None:
        raise RuntimeError("No complete camera/LiDAR/pose frame triplets were found.")
    previews[-1].save(output / "latest_pair.png")
    gif_frames = [frame.convert("P", palette=Image.Palette.ADAPTIVE) for frame in previews]
    gif_frames[0].save(output / "realtime_update.gif", save_all=True, append_images=gif_frames[1:], duration=args.gif_duration_ms, loop=0)
    np.savez_compressed(output / "final_grid.npz", **final_grid)
    summary = {
        "processed_frames": len(report),
        "semantic_source": "CARLA ground truth" if use_carla_labels else "SegFormer prediction",
        "depth_fusion": args.use_depth and any((root / "depth").glob("*.png")),
        "camera_ground_lift": args.camera_ground_lift,
        "gt_box_overlay": args.use_gt_boxes,
        "fusion_inputs": ["RGB camera", "LiDAR", "pose/calibration"],
        "window_frames": args.window_frames,
        "map": {"length_m": args.length_m, "width_m": args.width_m, "back_m": args.back_m, "resolution_m": args.resolution},
        "mean_latency_ms": round(float(np.mean([item["latency_ms"] for item in report])), 1),
        "final_occupied_cells": int((final_grid["count"] > 0).sum()),
        "frames": report,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
