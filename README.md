# CARLA Aerial Camera-LiDAR Semantic BEV Mapping

This project builds a local semantic bird's-eye-view (BEV) map from synchronized CARLA aerial RGB, LiDAR, calibration, and pose data.

The current pipeline is designed for a downward-looking UAV-like platform. It does not use CARLA semantic camera, depth camera, or object labels at runtime. Those CARLA ground-truth streams are retained only to train and evaluate the models offline.

## Current System

```text
RGB camera
  -> Fine-tuned SegFormer
  -> road / vegetation / building pixel semantics

RGB camera
  -> VisDrone YOLOv8n
  -> vehicle / pedestrian detections

LiDAR + camera calibration
  -> project 3D points to RGB pixels
  -> assign SegFormer or YOLO class to each visible point

Pose + 10-frame window
  -> align semantic points in the current local frame
  -> rasterize a 50 m x 40 m BEV semantic grid
```

SegFormer supplies dense scene semantics. The VisDrone YOLO model detects small aerial traffic objects. LiDAR provides physical position and height. Current-frame LiDAR points inside a YOLO vehicle/person detection take priority over accumulated ground votes, so dynamic objects remain visible in the BEV map.

## Runtime Models

| Model | Local path | Purpose |
| --- | --- | --- |
| Fine-tuned SegFormer B0 | `outputs/models/segformer_carla_aerial_160/` | Aerial road, vegetation, and building semantics |
| VisDrone YOLOv8n | `models/visdrone_yolov8n.pt` | Aerial vehicle and pedestrian detection |

The SegFormer model began from `nvidia/segformer-b0-finetuned-ade-512-512` and was fine-tuned on the CARLA RGB/semantic-camera pairs. The YOLO model is `dronefreak/visdrone-yolov8n`, trained on the public VisDrone aerial detection benchmark. Its model card is available at <https://huggingface.co/dronefreak/visdrone-yolov8n>.

## Semantic Classes

| Color | Class | Source at runtime |
| --- | --- | --- |
| Light gray | road / ground | SegFormer plus LiDAR ground geometry |
| Green | vegetation | SegFormer plus LiDAR projection |
| Dark gray | building / structure | SegFormer plus LiDAR projection |
| Blue | vehicle | VisDrone YOLO box supported by LiDAR points |
| Orange | person | VisDrone YOLO box supported by LiDAR points |

The black triangle labeled `UAV` in the BEV preview is the current platform position and heading.

## Repository Layout

```text
201/
|-- README.md
|-- scripts/
|   |-- collect_carla_25d.py
|   |-- finetune_segformer_carla.py
|   |-- fuse_camera_lidar_semantic_25d.py
|   `-- build_bev_box_dataset.py
|-- carla_bev_training_001/       # Current 100-frame CARLA collection
|-- carla_uav_collection/         # Earlier CARLA collection kept for reference
|-- models/
|   `-- visdrone_yolov8n.pt
`-- outputs/
    |-- models/
    |   `-- segformer_carla_aerial_160/
    `-- rgb_lidar_pose_segformer_visdrone_yolo/
```

Generated data, model weights, caches, and output artifacts are ignored by Git. Only source code and documentation should be committed.

## Dataset Format

The runtime mapping script requires the following synchronized streams:

```text
dataset/
|-- camera/
|   `-- 000000.png
|-- lidar/
|   `-- 000000.ply
|-- poses/
|   `-- 000000.txt
`-- calib.json
```

`camera`, `lidar`, and `poses` use the same frame identifier. Each pose file is a 4 x 4 `world_from_lidar` matrix with 16 numeric values. `calib.json` contains:

```json
{
  "camera_intrinsics": [["fx", 0, "cx"], [0, "fy", "cy"], [0, 0, 1]],
  "lidar_to_camera": [["r11", "r12", "r13", "tx"], ["r21", "r22", "r23", "ty"], ["r31", "r32", "r33", "tz"], [0, 0, 0, 1]]
}
```

The current CARLA training dataset also includes these offline-only streams:

```text
semantic/  Raw CARLA semantic camera tags; used to fine-tune SegFormer.
depth/     Raw CARLA depth images; optional geometry validation data.
labels/    CARLA actor ground-truth 3D boxes; used for evaluation/training only.
```

Do not pass `--semantic-source carla`, `--use-depth`, or `--use-gt-boxes` for the deployed inference workflow. The current final result intentionally does not use those sources.

## Environment

The current workspace uses `D:\anaconda3\python.exe` for mapping and training. Required packages are:

```powershell
D:\anaconda3\python.exe -m pip install torch transformers ultralytics opencv-python pillow numpy
```

CARLA collection runs in the separate CARLA Python environment because it needs the CARLA API package.

## Train SegFormer

Fine-tuning uses RGB as input and CARLA semantic camera images only as offline pixel-level targets. The saved model predicts from RGB alone.

```powershell
D:\anaconda3\python.exe scripts\finetune_segformer_carla.py `
  --dataset carla_bev_training_001 `
  --output-dir outputs\models\segformer_carla_aerial_160 `
  --epochs 10 `
  --batch-size 4 `
  --height 160 `
  --width 288 `
  --learning-rate 0.00002 `
  --person-weight 8
```

The script uses 80 percent of frames for training and every fifth frame for validation. It writes one best model after training, along with `training_report.json`. The current 100-frame collection is useful as a proof of pipeline, but additional towns, camera heights, weather, and routes are required for a meaningful generalization evaluation.

## Run the Current Final Pipeline

This command uses only RGB, LiDAR, pose/calibration, the fine-tuned SegFormer, and the VisDrone YOLO model. It does not use CARLA semantic, depth, or actor labels.

```powershell
D:\anaconda3\python.exe scripts\fuse_camera_lidar_semantic_25d.py `
  --dataset carla_bev_training_001 `
  --output-dir outputs\rgb_lidar_pose_segformer_visdrone_yolo `
  --semantic-source model `
  --semantic-model outputs\models\segformer_carla_aerial_160 `
  --camera-ground-lift `
  --use-yolo `
  --yolo-model models\visdrone_yolov8n.pt `
  --yolo-confidence 0.25 `
  --yolo-imgsz 1280 `
  --window-frames 10 `
  --length-m 50 `
  --width-m 40 `
  --back-m 10 `
  --resolution 0.5 `
  --max-frames 100
```

`--camera-ground-lift` uses RGB segmentation regions constrained by LiDAR ground returns to create a denser road surface. The 1280-pixel YOLO inference size is intentional: aerial cars and pedestrians are small and are easily lost at the usual 640-pixel detector size.

## Outputs

The final command creates only four artifacts:

```text
outputs/rgb_lidar_pose_segformer_visdrone_yolo/
|-- latest_pair.png       # Left: RGB plus YOLO boxes. Right: semantic BEV map.
|-- realtime_update.gif   # BEV evolution over all processed frames.
|-- final_grid.npz        # Numeric map layers for later research code.
`-- run_summary.json      # Per-frame detections, latency, and run settings.
```

`final_grid.npz` contains `class_id`, `count`, `z_min`, `z_max`, ground reference, origin, and local axes. `run_summary.json` records YOLO detections and the number of dynamic LiDAR hits used in each frame.

## CARLA Collection

The collection script is `scripts/collect_carla_25d.py`. It collects synchronized RGB camera, semantic camera, depth camera, LiDAR, pose, calibration, and dynamic actor ground-truth boxes. The additional CARLA streams make training and evaluation possible without manual labeling.

For a stronger dataset, collect multiple independent sequences with:

- Different CARLA towns and road layouts.
- Day, dusk, rain, and shadow conditions.
- Several UAV heights and camera pitches.
- Dense and sparse traffic, including pedestrians.
- Separate routes for training, validation, and final test.

## Deployment Boundary

CARLA semantic images and JSON actor boxes are simulator ground truth. They are valid training and evaluation targets, but they do not exist for a real drone flight.

The intended deployed inputs are only:

```text
RGB camera + LiDAR + camera-LiDAR calibration + platform pose
```

SegFormer supplies scene classes, VisDrone YOLO supplies dynamic-object proposals, and LiDAR establishes their BEV positions. For a production detector, fine-tune the VisDrone YOLO model with the CARLA `labels/` boxes or real aerial data, then validate on a flight sequence never used during training.
