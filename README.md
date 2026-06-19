# Camera-LiDAR Semantic 2.5D Mapping

面向无人机或车载平台的局部 2.5D 语义栅格建图原型。

输入是同步的 RGB 相机图像和 LiDAR 点云；输出是带高度、点密度、语义类别的局部俯视栅格。语义分割由 SegFormer 完成，YOLO 专门增强车、人和其他动态交通参与者识别。

## Pipeline

```text
RGB image
  -> SegFormer: road / building / vegetation semantic classes
  -> YOLO: person / car / bus / truck / motorcycle / bicycle
  -> YOLO dynamic detections override scene semantic classes

LiDAR point cloud + camera-LiDAR calibration
  -> project LiDAR points into RGB semantic image
  -> assign semantic class to each visible LiDAR point
  -> use height and point density to build a local 2.5D semantic grid
  -> save PNG visualization, compressed grid layers, and detection report
```

When `poses/` is available, the program maintains a sliding window and fuses recent point clouds in the current local frame. Without poses it still produces a single-frame obstacle grid.

## Required Data

```text
your_dataset/
|-- camera/
|   |-- 000000.png
|   `-- 000001.png
|-- lidar/
|   |-- 000000.ply       # ASCII PLY, or .bin / .npy
|   `-- 000001.ply
|-- poses/               # optional but required for multi-frame fusion
|   |-- 000000.txt       # 4x4 world_from_lidar matrix, 16 numbers
|   `-- 000001.txt
`-- calib.json
```

Image and point-cloud filenames must match so that each pair is time synchronized. `calib.json` must include the camera intrinsics and the rigid transform from LiDAR to camera:

```json
{
  "camera_intrinsics": [
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
  ],
  "lidar_to_camera": [
    [r11, r12, r13, tx],
    [r21, r22, r23, ty],
    [r31, r32, r33, tz],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

## Labels

Manual labels are **not required** for the first run. The pipeline uses pretrained models.

For deployment-quality aerial performance, annotate your own data later:

```text
YOLO fine-tuning: 100-300 images with person / vehicle bounding boxes.
Semantic fine-tuning: 100-300 images with pixel-level masks.
Evaluation: at least 50 held-out images.
```

## CARLA Collection Recommendation

CARLA is recommended for the first full validation set because it can synchronously export RGB, LiDAR, depth, semantic ground truth, pose, and calibration without manual annotation.

Collect these synchronized streams at the same simulation tick:

```text
sensor.camera.rgb
sensor.lidar.ray_cast
sensor.camera.semantic_segmentation    # ground truth for validation/fine-tuning
sensor.camera.depth                    # optional geometry validation
sensor transform / platform transform  # pose and extrinsics
```

Suggested UAV-like configuration:

```text
camera/LiDAR height: 20-40 m
camera pitch: -60 to -90 degrees
LiDAR range: 50-80 m
capture rate: 10 Hz, synchronous simulation mode
local 2.5D map: 30 m x 30 m, 0.10-0.20 m/cell, 5-10 frame sliding window
```

## Install

```bash
python -m pip install ultralytics opencv-python transformers torch torchvision
```

The first run downloads the SegFormer and YOLO pretrained weights automatically.

## Run

```bash
python scripts\fuse_camera_lidar_semantic_25d.py \
  --dataset path\to\your_dataset \
  --window-frames 5 \
  --length-m 30 \
  --width-m 20 \
  --resolution 0.20 \
  --output-dir outputs\semantic_25d
```

## Outputs

```text
outputs/semantic_25d/
|-- grids/
|   |-- 000000_semantic_25d.png
|   `-- 000000_semantic_25d.npz
`-- report.json
```

Each `.npz` stores class IDs, per-cell minimum/maximum height, point count, and local ground reference. `report.json` records YOLO detections, fusion mode, and per-frame latency.

## Repository Files

```text
scripts/fuse_camera_lidar_semantic_25d.py
  Camera/LiDAR fusion, SegFormer semantics, YOLO dynamic objects, and 2.5D rasterization.

CAMERA_LIDAR_DATA_REQUIREMENTS.md
  Detailed data format and calibration requirements.
```
