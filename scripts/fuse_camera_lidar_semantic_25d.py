"""Camera + LiDAR semantic 2.5D local-map prototype.

Expected input layout:
  dataset/
    camera/000000.png
    lidar/000000.ply              # ASCII PLY: x y z [intensity]
    calib.json                    # camera intrinsics and LiDAR-to-camera pose
    poses/000000.txt              # optional 4x4 world_from_lidar matrices

See CAMERA_LIDAR_DATA_REQUIREMENTS.md for the exact calib.json schema.
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / "Ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "matplotlib_cache"))
os.environ.setdefault("HF_HOME", str(Path.cwd() / "model_cache"))

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO


CLASS = {"unknown": 0, "road": 1, "vegetation": 2, "building": 3, "vehicle": 4, "person": 5}
PALETTE = np.array(
    [
        [238, 242, 244],
        [190, 198, 201],
        [61, 157, 83],
        [133, 143, 153],
        [48, 119, 204],
        [244, 154, 51],
    ],
    dtype=np.uint8,
)
YOLO_DYNAMIC = {"person": CLASS["person"], "car": CLASS["vehicle"], "bus": CLASS["vehicle"], "truck": CLASS["vehicle"], "motorcycle": CLASS["vehicle"], "bicycle": CLASS["vehicle"]}


def read_ascii_ply(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = []
        for line in f:
            header.append(line)
            if line.strip() == "end_header":
                break
        if not any("format ascii" in line for line in header):
            raise ValueError(f"{path} is not ASCII PLY. Convert binary PLY to ASCII or .npy before using this prototype.")
        points = np.loadtxt(f, dtype=np.float32)
    return points.reshape(1, -1) if points.ndim == 1 else points


def load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        points = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".bin":
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    elif path.suffix.lower() == ".ply":
        points = read_ascii_ply(path)
    else:
        raise ValueError(f"Unsupported LiDAR format: {path.suffix}. Use .ply (ASCII), .bin, or .npy.")
    return points[:, :3]


def load_calib(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    k = np.asarray(payload["camera_intrinsics"], dtype=np.float64)
    t_camera_lidar = np.asarray(payload["lidar_to_camera"], dtype=np.float64)
    if k.shape != (3, 3) or t_camera_lidar.shape != (4, 4):
        raise ValueError("calib.json must contain camera_intrinsics [3,3] and lidar_to_camera [4,4].")
    return k, t_camera_lidar


def project_points(points: np.ndarray, k: np.ndarray, t_camera_lidar: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    hom = np.c_[points, np.ones(len(points))].T
    cam = t_camera_lidar @ hom
    z = cam[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = k @ cam[:3]
        u = uvw[0] / uvw[2]
        v = uvw[1] / uvw[2]
    h, w = image_shape
    valid = np.isfinite(u) & np.isfinite(v) & (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uv = np.c_[np.rint(u).astype(np.int32, copy=False), np.rint(v).astype(np.int32, copy=False)]
    return uv, valid


def map_ade_labels(label_map: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    out = np.zeros(label_map.shape, dtype=np.uint8)
    for idx, raw_name in id2label.items():
        name = raw_name.lower()
        if any(token in name for token in ("road", "street", "sidewalk", "path", "floor", "pavement")):
            out[label_map == int(idx)] = CLASS["road"]
        elif any(token in name for token in ("tree", "grass", "plant", "field", "flower", "palm", "bush")):
            out[label_map == int(idx)] = CLASS["vegetation"]
        elif any(token in name for token in ("building", "house", "wall", "fence", "skyscraper", "roof")):
            out[label_map == int(idx)] = CLASS["building"]
        elif any(token in name for token in ("car", "bus", "truck", "van", "automobile", "bicycle", "motorcycle")):
            out[label_map == int(idx)] = CLASS["vehicle"]
        elif "person" in name or "pedestrian" in name:
            out[label_map == int(idx)] = CLASS["person"]
    return out


def semantic_prediction(image: np.ndarray, processor, model, device: torch.device) -> np.ndarray:
    inputs = processor(images=Image.fromarray(image), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    labels = torch.nn.functional.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False).argmax(dim=1)[0]
    return map_ade_labels(labels.cpu().numpy(), model.config.id2label)


def apply_yolo_overrides(semantic: np.ndarray, yolo, image: np.ndarray, conf: float) -> tuple[np.ndarray, list[dict]]:
    result = yolo.predict(image, conf=conf, verbose=False)[0]
    detections = []
    for box in result.boxes:
        cls_name = result.names[int(box.cls[0])]
        if cls_name not in YOLO_DYNAMIC:
            continue
        x0, y0, x1, y1 = np.rint(box.xyxy[0].cpu().numpy()).astype(int)
        x0, x1 = np.clip([x0, x1], 0, semantic.shape[1])
        y0, y1 = np.clip([y0, y1], 0, semantic.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        semantic[y0:y1, x0:x1] = YOLO_DYNAMIC[cls_name]
        detections.append({"class": cls_name, "confidence": float(box.conf[0]), "bbox": [int(x0), int(y0), int(x1), int(y1)]})
    return semantic, detections


def rasterize(points: np.ndarray, point_classes: np.ndarray, current_pose: np.ndarray, length_m: float, width_m: float, resolution: float) -> dict:
    local = points if current_pose is None else (np.linalg.inv(current_pose) @ np.c_[points, np.ones(len(points))].T)[:3].T
    rows, cols = int(round(length_m / resolution)), int(round(width_m / resolution))
    half_w = width_m / 2.0
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    keep = (x >= 0) & (x < length_m) & (y >= -half_w) & (y < half_w)
    x, y, z, point_classes = x[keep], y[keep], z[keep], point_classes[keep]
    iy = np.clip(((length_m - x) / resolution).astype(int), 0, rows - 1)
    ix = np.clip(((y + half_w) / resolution).astype(int), 0, cols - 1)
    flat = iy * cols + ix
    count = np.zeros(rows * cols, dtype=np.int32)
    z_min = np.full(rows * cols, np.inf, dtype=np.float32)
    z_max = np.full(rows * cols, -np.inf, dtype=np.float32)
    votes = np.zeros((len(CLASS), rows * cols), dtype=np.int32)
    np.add.at(count, flat, 1)
    np.minimum.at(z_min, flat, z)
    np.maximum.at(z_max, flat, z)
    for cid in range(len(CLASS)):
        np.add.at(votes[cid], flat[point_classes == cid], 1)
    cls = votes.argmax(axis=0).astype(np.uint8)
    cls[count == 0] = CLASS["unknown"]
    valid = count > 0
    ground = float(np.percentile(z_min[valid], 10)) if valid.any() else 0.0
    return {"class": cls.reshape(rows, cols), "z_min": z_min.reshape(rows, cols), "z_max": z_max.reshape(rows, cols), "count": count.reshape(rows, cols), "ground": ground}


def render_grid(grid: dict, output: Path) -> None:
    cls = grid["class"]
    image = PALETTE[cls]
    valid = grid["count"] > 0
    if valid.any():
        height = np.clip(grid["z_max"] - grid["ground"], 0, 12)
        shade = 0.82 + 0.25 * (height / 12.0)
        image[valid] = np.clip(image[valid].astype(np.float32) * shade[valid, None], 0, 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, "RGB").resize((image.shape[1] * 5, image.shape[0] * 5), Image.Resampling.NEAREST).save(output)


def load_pose(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if values.size != 16:
        raise ValueError(f"Pose {path} must contain 16 values for a 4x4 world_from_lidar matrix.")
    return values.reshape(4, 4)


def world_transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (pose @ np.c_[points, np.ones(len(points))].T)[:3].T.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse camera semantics and LiDAR into a local 2.5D grid.")
    parser.add_argument("--dataset", required=True, help="Dataset with camera/, lidar/, and calib.json.")
    parser.add_argument("--output-dir", default="outputs/camera_lidar_semantic_25d")
    parser.add_argument("--length-m", type=float, default=30.0)
    parser.add_argument("--width-m", type=float, default=20.0)
    parser.add_argument("--resolution", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=5, help="Sliding-window size when poses/ is available.")
    parser.add_argument("--yolo-confidence", type=float, default=0.30)
    parser.add_argument("--semantic-model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--yolo-model", default="yolo11n.pt")
    args = parser.parse_args()

    root, out = Path(args.dataset), Path(args.output_dir)
    k, t_camera_lidar = load_calib(root / "calib.json")
    image_files = sorted((root / "camera").glob("*"))
    if args.max_frames:
        image_files = image_files[:args.max_frames]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained(args.semantic_model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.semantic_model).to(device).eval()
    yolo = YOLO(args.yolo_model)
    report = []
    recent = deque(maxlen=args.window_frames)

    for image_file in image_files:
        lidar_file = next((root / "lidar" / f"{image_file.stem}{suffix}" for suffix in (".ply", ".bin", ".npy") if (root / "lidar" / f"{image_file.stem}{suffix}").exists()), None)
        if lidar_file is None:
            continue
        start = time.perf_counter()
        image = np.asarray(Image.open(image_file).convert("RGB"))
        semantic = semantic_prediction(image, processor, model, device)
        semantic, detections = apply_yolo_overrides(semantic, yolo, image, args.yolo_confidence)
        points = load_points(lidar_file)
        uv, visible = project_points(points, k, t_camera_lidar, image.shape[:2])
        point_classes = np.zeros(len(points), dtype=np.uint8)
        point_classes[visible] = semantic[uv[visible, 1], uv[visible, 0]]
        pose_file = root / "poses" / f"{image_file.stem}.txt"
        if pose_file.exists():
            current_pose = load_pose(pose_file)
            recent.append((world_transform(points, current_pose), point_classes))
            fused_points = np.concatenate([item[0] for item in recent], axis=0)
            fused_classes = np.concatenate([item[1] for item in recent], axis=0)
            grid = rasterize(fused_points, fused_classes, current_pose, args.length_m, args.width_m, args.resolution)
            fusion_mode = f"sliding window ({len(recent)} frames)"
        else:
            grid = rasterize(points, point_classes, None, args.length_m, args.width_m, args.resolution)
            fusion_mode = "single frame (no pose file)"
        stem = image_file.stem
        render_grid(grid, out / "grids" / f"{stem}_semantic_25d.png")
        np.savez_compressed(out / "grids" / f"{stem}_semantic_25d.npz", **grid)
        report.append({"frame": stem, "lidar": lidar_file.name, "fusion_mode": fusion_mode, "detections": detections, "latency_ms": round((time.perf_counter() - start) * 1000, 1)})
        print(f"processed {stem}: detections={len(detections)} latency={report[-1]['latency_ms']}ms")

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
